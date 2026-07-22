"""Durable SafeTrace batch soak harness for controlled local validation."""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.api.jobs as jobs_module
from src.api.batches import BatchStore
from src.api.jobs import AnalysisSettings, JobStore, TERMINAL_STATES, execute_analysis_job

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def discover_inputs(input_path: Path) -> list[tuple[str, bytes]]:
    if input_path.is_dir():
        return [
            (path.relative_to(input_path).as_posix(), path.read_bytes())
            for path in sorted(input_path.rglob("*"))
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ]
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            return [
                (entry.filename, archive.read(entry))
                for entry in archive.infolist()
                if not entry.is_dir() and Path(entry.filename).suffix.lower() in VIDEO_SUFFIXES
            ]
    if input_path.suffix.lower() in {".json", ".csv"}:
        from scripts.benchmark_safetrace_profiles import load_manifest

        _metadata, samples = load_manifest(input_path)
        discovered = []
        for sample in samples:
            media = Path(str(sample["mediaPath"]))
            if not media.is_absolute():
                media = ROOT / media
            if media.is_file():
                discovered.append((str(sample.get("sourceRelativePath") or media.name), media.read_bytes()))
        return discovered
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_SUFFIXES:
        return [(input_path.name, input_path.read_bytes())]
    raise ValueError(f"No supported soak input found at {input_path}")


def snapshot_state(
    *,
    output_dir: Path,
    batch_store: BatchStore,
    job_store: JobStore,
    batch_ids: list[str],
    sequence: int,
) -> dict[str, Any]:
    batches = []
    for batch_id in batch_ids:
        record = batch_store.get(batch_id, job_store)
        if record is not None:
            batches.append(record.payload())
    payload = {
        "sequence": sequence,
        "capturedAtEpoch": time.time(),
        "jobStatusCounts": job_store.status_counts(include_disk=True),
        "batches": batches,
    }
    snapshots = output_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    (snapshots / f"snapshot_{sequence:05d}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "soak_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def manifest_integrity(root: Path) -> dict[str, Any]:
    manifests = list(root.rglob("manifest.json"))
    invalid: list[str] = []
    ids: list[str] = []
    for path in manifests:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid.append(str(path))
            continue
        identifier = payload.get("jobId") or payload.get("batchId")
        if identifier:
            ids.append(str(identifier))
    temp_files = [str(path) for path in root.rglob("manifest.json.*.tmp")]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    return {
        "manifestCount": len(manifests),
        "invalidManifests": invalid,
        "temporaryManifestFiles": temp_files,
        "duplicateManifestIds": duplicates,
        "passed": not invalid and not temp_files and not duplicates,
    }


def run_jobs(job_store: JobStore, job_ids: list[str], concurrency: int) -> None:
    with ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="safetrace-soak") as executor:
        futures = [executor.submit(execute_analysis_job, job_store, job_id) for job_id in job_ids]
        for future in as_completed(futures):
            future.result()


def drain_jobs(job_store: JobStore, job_ids: list[str], concurrency: int, timeout_seconds: float = 1800.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        pending = []
        next_retry_delay = 0.0
        for job_id in job_ids:
            record = job_store.require(job_id)
            if record.status in TERMINAL_STATES or record.status == "paused":
                continue
            if record.status == "retry_wait":
                if record.next_retry_at is not None:
                    from datetime import datetime, timezone

                    delay = max((record.next_retry_at - datetime.now(timezone.utc)).total_seconds(), 0.0)
                    next_retry_delay = max(next_retry_delay, delay)
                    if delay > 0:
                        continue
                if not job_store.activate_retry(job_id):
                    continue
            pending.append(job_id)
        if not pending:
            active = [job_store.require(job_id) for job_id in job_ids if job_store.require(job_id).status not in TERMINAL_STATES]
            if not active:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Soak jobs did not reach terminal state before the harness timeout.")
            time.sleep(min(max(next_retry_delay, 0.05), 1.0))
            continue
        run_jobs(job_store, pending, concurrency)
        if time.monotonic() >= deadline:
            raise TimeoutError("Soak jobs did not reach terminal state before the harness timeout.")


def install_transient_failure_once() -> Callable[[], None]:
    original = jobs_module.run_pipeline
    lock = threading.Lock()
    failed = False

    def flaky(**kwargs):
        nonlocal failed
        with lock:
            if not failed:
                failed = True
                raise OSError("Injected transient soak failure")
        return original(**kwargs)

    jobs_module.run_pipeline = flaky

    def restore() -> None:
        jobs_module.run_pipeline = original

    return restore


def _settings(mode: str, device: str) -> AnalysisSettings:
    return AnalysisSettings(
        fps=1.0,
        top_k=5,
        enable_vlm=False,
        device=device,
        safe_mode=True,
        vlm_profile="rule_based",
        vlm_enabled=False,
        review_mode=mode,
        use_case_profile={"profileId": "general_safety"},
    )


def run_soak(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = output_dir / "runtime"
    jobs_root = runtime_root / "jobs"
    batches_root = runtime_root / "batches"
    job_store = JobStore(jobs_root)
    batch_store = BatchStore(batches_root)
    files = discover_inputs(args.input)
    if args.include_corrupt:
        files.append(("invalid/corrupt.mp4", b"not-a-video"))
    if not files:
        raise ValueError("Soak input did not contain any video files.")
    if args.simulate_low_disk:
        report = {
            "status": "capacity_rejected",
            "reason": "simulated_low_disk_threshold",
            "jobsCreated": 0,
            "integrity": manifest_integrity(runtime_root),
        }
        (output_dir / "soak_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    existing_batches = sorted(path.name for path in batches_root.glob("batch_*") if path.is_dir()) if args.resume else []
    batch_ids: list[str] = list(existing_batches)
    restore_failure: Callable[[], None] | None = None
    if args.inject_transient_failure:
        restore_failure = install_transient_failure_once()

    started = time.perf_counter()
    target_jobs = args.job_count if args.job_count > 0 else (1_000_000 if args.duration_seconds > 0 else len(files))
    cycle = 0
    try:
        while sum(len(batch_store.require(batch_id, job_store).job_ids) for batch_id in batch_ids) < target_jobs:
            if args.duration_seconds > 0 and time.perf_counter() - started >= args.duration_seconds:
                break
            cycle += 1
            remaining = target_jobs - sum(len(batch_store.require(batch_id, job_store).job_ids) for batch_id in batch_ids)
            cycle_files = files[: max(1, min(len(files), remaining))]
            batch = batch_store.create_from_files(
                files=cycle_files,
                source_filename=f"soak-cycle-{cycle}",
                query="general safety violations",
                settings=_settings(args.mode, args.device),
                job_store=job_store,
                import_key=f"{args.run_id}-cycle-{cycle}",
            )
            batch_ids.append(batch.batch_id)
            pending = [job_id for job_id in batch.job_ids if job_store.require(job_id).status not in TERMINAL_STATES]
            if args.restart_after > 0 and cycle == 1 and pending:
                first = pending.pop(0)
                execute_analysis_job(job_store, first)
                if pending:
                    recovering = job_store.require(pending[0])
                    job_store.update_status(recovering.job_id, status="running", progress=0.4, current_step="Injected restart")
                    recovering.updated_at = recovering.updated_at.replace(year=max(2000, recovering.updated_at.year - 1))
                    job_store.persist_job(recovering)
                completed_before = {job_id: job_store.require(job_id).result for job_id in batch.job_ids if job_store.require(job_id).status == "completed"}
                job_store = JobStore(jobs_root)
                batch_store = BatchStore(batches_root)
                pending = [job_id for job_id in batch.job_ids if job_store.require(job_id).status not in TERMINAL_STATES]
                for job_id, result in completed_before.items():
                    if job_store.require(job_id).result != result:
                        raise RuntimeError(f"Completed result changed across restart: {job_id}")
            if not args.dry_run:
                drain_jobs(job_store, pending, args.concurrency)
            snapshot_state(
                output_dir=output_dir,
                batch_store=batch_store,
                job_store=job_store,
                batch_ids=batch_ids,
                sequence=cycle,
            )
            if args.dry_run:
                break
    finally:
        if restore_failure is not None:
            restore_failure()

    final_batches = [batch_store.require(batch_id, job_store).payload() for batch_id in batch_ids]
    all_jobs = [job_store.require(job_id) for batch in final_batches for job_id in batch["jobIds"]]
    integrity = manifest_integrity(runtime_root)
    report = {
        "schemaVersion": 1,
        "runId": args.run_id,
        "input": str(args.input),
        "mode": args.mode,
        "device": args.device,
        "concurrency": args.concurrency,
        "requestedDurationSeconds": args.duration_seconds,
        "actualDurationSeconds": round(time.perf_counter() - started, 3),
        "restartInjected": bool(args.restart_after),
        "transientFailureInjected": bool(args.inject_transient_failure),
        "corruptFileIncluded": bool(args.include_corrupt),
        "batchIds": batch_ids,
        "jobStatusCounts": job_store.status_counts(include_disk=True),
        "retryCount": sum(job.retry_count for job in all_jobs),
        "batches": final_batches,
        "integrity": integrity,
        "passed": integrity["passed"] and all(job.status in TERMINAL_STATES for job in all_jobs),
    }
    (output_dir / "soak_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (output_dir / "soak_report.md").write_text(
        "# SafeTrace Soak Report\n\n"
        f"- Run: `{args.run_id}`\n"
        f"- Duration: `{report['actualDurationSeconds']}` seconds\n"
        f"- Status counts: `{json.dumps(report['jobStatusCounts'], sort_keys=True)}`\n"
        f"- Retry count: `{report['retryCount']}`\n"
        f"- Integrity passed: `{integrity['passed']}`\n"
        f"- Overall passed: `{report['passed']}`\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=f"soak-{int(time.time())}")
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--job-count", type=int, default=0)
    parser.add_argument("--concurrency", choices=(1, 2), type=int, default=1)
    parser.add_argument("--mode", choices=("fast_local", "comprehensive"), default="fast_local")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--restart-after", type=int, default=0)
    parser.add_argument("--inject-transient-failure", action="store_true")
    parser.add_argument("--include-corrupt", action="store_true")
    parser.add_argument("--simulate-low-disk", action="store_true")
    parser.add_argument("--snapshot-seconds", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = run_soak(args)
    print(json.dumps({"runId": args.run_id, "output": str(args.output_dir), "passed": report.get("passed"), "status": report.get("status")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
