"""Real-hardware Phase R scheduler smoke with isolated job stores.

This deliberately runs the production pipeline against copied local video input
instead of a fake pipeline.  It never uses the repository's normal job store or
modifies an existing upload.  The JSON report is suitable for the Phase R
acceptance gate, including the controlled restart/recovery exercise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TERMINAL = {"completed", "failed", "cancelled"}
BATCH_TERMINAL = {"completed", "completed_with_failures", "failed", "partial", "cancelled"}
ACTIVE = {
    "running", "running_preprocess", "running_detector", "running_refinement", "running_report",
    "waiting_for_gpu", "waiting_for_mobilesam", "waiting_for_vlm",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_local_full() -> None:
    os.environ["SAFETRACE_RUNTIME_PROFILE"] = "local_full"
    os.environ["SAFETRACE_DEVICE"] = "cuda"
    os.environ["SAFETRACE_ENABLE_GPU_AUTO"] = "true"
    os.environ["SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM"] = "true"
    os.environ["SAFETRACE_MOBILESAM_ENABLED"] = "auto"
    os.environ["SAFETRACE_MOBILESAM_WORKER_ENABLED"] = "true"
    os.environ["SAFETRACE_CHAT_ENABLED"] = "auto"
    os.environ["SAFETRACE_CHAT_AUTOLOAD"] = "true"
    os.environ["SAFETRACE_JOB_CONCURRENCY"] = "2"
    os.environ["SAFETRACE_ANALYSIS_CONCURRENCY"] = "2"
    os.environ["SAFETRACE_PREPROCESS_CONCURRENCY"] = "2"
    os.environ["SAFETRACE_GPU_INFERENCE_CONCURRENCY"] = "2"
    os.environ["SAFETRACE_MOBILESAM_CONCURRENCY"] = "1"
    os.environ["SAFETRACE_VLM_CONCURRENCY"] = "1"
    os.environ["SAFETRACE_PER_BATCH_CONCURRENCY"] = "2"
    os.environ["SAFETRACE_SCHEDULER_POLICY"] = "oldest_batch_first"
    # The smoke validates the Fast Local base engine and MobileSAM. It does not
    # spend GPU time on VLM generation while testing worker-slot fairness.
    os.environ["SAFETRACE_VLM_ENABLED"] = "disabled"
    os.environ["SAFETRACE_ENABLE_VLM"] = "false"


def _rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return None


def _gpu_memory() -> dict[str, float | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocatedMb": None, "reservedMb": None}
        return {
            "allocatedMb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
            "reservedMb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
        }
    except Exception:
        return {"allocatedMb": None, "reservedMb": None}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_videos() -> list[Path]:
    upload_root = REPO_ROOT / "data" / "api_jobs"
    candidates: list[Path] = []
    for name in ("sample-5s-720p.mp4", "sample-10s-720p.mp4", "sample-15s-720p.mp4"):
        candidates.extend(upload_root.glob(f"*/uploads/{name}"))
    unique: list[Path] = []
    digests: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: (item.name, item.stat().st_size, str(item))):
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            continue
        digest = _hash(candidate)
        if digest in digests:
            continue
        unique.append(candidate)
        digests.add(digest)
        if len(unique) == 4:
            break
    if len(unique) < 3:
        raise RuntimeError("Could not locate three distinct local sample videos under data/api_jobs.")
    return unique


def _copy_inputs(work_dir: Path) -> list[Path]:
    destination = work_dir / "inputs"
    destination.mkdir(parents=True, exist_ok=True)
    copies: list[Path] = []
    for index, source in enumerate(_source_videos(), start=1):
        target = destination / f"phase-r-{index:02d}-{source.name}"
        shutil.copy2(source, target)
        copies.append(target)
    return copies


def _poll_job(client, job_id: str, *, timeout_seconds: float = 600.0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout_seconds
    snapshots: list[dict[str, Any]] = []
    last_key: tuple[Any, ...] | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        response.raise_for_status()
        payload = response.json()
        key = (payload.get("status"), payload.get("stage"), payload.get("updatedAt"))
        if key != last_key:
            snapshots.append({"observedAt": _utc(), **payload})
            last_key = key
        if payload.get("status") in BATCH_TERMINAL:
            return payload, snapshots
        time.sleep(0.35)
    raise TimeoutError(f"Timed out waiting for job {job_id}")


def _post_single(client, path: Path, *, query: str) -> str:
    with path.open("rb") as handle:
        response = client.post(
            "/api/analyze",
            files={"file": (path.name, handle, "video/mp4")},
            data={
                "query": query,
                "fps": "1",
                "topK": "2",
                "enableVlm": "false",
                "vlmEnabled": "false",
                "vlmProfile": "rule_based",
                "device": "cuda",
                "reviewMode": "fast_local",
            },
        )
    response.raise_for_status()
    return str(response.json()["jobId"])


def _post_batch(client, paths: list[Path], *, query: str) -> dict[str, Any]:
    with ExitStack() as stack:
        files = [
            ("files", (path.name, stack.enter_context(path.open("rb")), "video/mp4"))
            for path in paths
        ]
        response = client.post(
            "/api/batches/analyze",
            files=files,
            data={
                "query": query,
                "fps": "1",
                "topK": "2",
                "enableVlm": "false",
                "vlmEnabled": "false",
                "vlmProfile": "rule_based",
                "device": "cuda",
                "reviewMode": "fast_local",
            },
        )
    response.raise_for_status()
    return response.json()


def _wait_batch_with_later_job(client, batch_id: str, later_job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 900.0
    observations: list[dict[str, Any]] = []
    two_active_seen = False
    parent_active_after_child_complete = False
    final: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/batches/{batch_id}")
        response.raise_for_status()
        payload = response.json()
        children = list(payload.get("acceptedFiles") or [])
        active_count = sum(item.get("status") in ACTIVE for item in children)
        completed_count = sum(item.get("status") == "completed" for item in children)
        two_active_seen = two_active_seen or active_count >= 2
        parent_active_after_child_complete = parent_active_after_child_complete or (
            completed_count > 0 and payload.get("status") not in TERMINAL
        )
        observations.append(
            {
                "observedAt": _utc(),
                "batchStatus": payload.get("status"),
                "children": [
                    {key: item.get(key) for key in ("jobId", "status", "queuePosition", "workerSlot", "capacityReason")}
                    for item in children
                ],
            }
        )
        if payload.get("status") in TERMINAL:
            final = payload
            break
        time.sleep(0.35)
    if final is None:
        raise TimeoutError(f"Timed out waiting for batch {batch_id}")
    later, later_history = _poll_job(client, later_job_id)
    return {
        "final": final,
        "observations": observations,
        "twoActiveSeen": two_active_seen,
        "parentActiveAfterChildComplete": parent_active_after_child_complete,
        "laterJob": later,
        "laterHistory": later_history,
    }


def _result_summary(client, job_id: str) -> dict[str, Any]:
    status = client.get(f"/api/jobs/{job_id}")
    status.raise_for_status()
    result = client.get(f"/api/jobs/{job_id}/result")
    technical = client.get(f"/api/reports/{job_id}/technical-json")
    return {
        "statusCode": status.status_code,
        "resultStatusCode": result.status_code,
        "technicalStatusCode": technical.status_code,
        "status": status.json(),
        "hasEngineMetrics": bool((result.json() if result.status_code == 200 else {}).get("engineMetrics")),
    }


def _no_manifest_temps(root: Path) -> list[str]:
    return [str(path) for path in root.rglob("manifest.json.*.tmp")]


def _recovery_worker(job_root: Path, job_ids: list[str]) -> int:
    _configure_local_full()
    from concurrent.futures import ThreadPoolExecutor
    from src.api.jobs import JobStore, execute_analysis_job

    store = JobStore(job_root)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda job_id: execute_analysis_job(store, job_id), job_ids))
    return 0


def _manifest_states(job_root: Path, job_ids: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for job_id in job_ids:
        path = job_root / job_id / "manifest.json"
        try:
            result[job_id] = str(json.loads(path.read_text(encoding="utf-8")).get("status"))
        except Exception:
            result[job_id] = "missing"
    return result


def _exercise_recovery(work_dir: Path, inputs: list[Path]) -> dict[str, Any]:
    """Kill two genuine active workers, then recover them from their manifests."""
    _configure_local_full()
    from src.api.jobs import AnalysisSettings, JobStore, schedule_analysis_jobs

    job_root = work_dir / "recovery_jobs"
    store = JobStore(job_root)
    settings = AnalysisSettings(
        fps=1.0,
        top_k=2,
        enable_vlm=False,
        vlm_enabled=False,
        vlm_profile="rule_based",
        device="cuda",
        safe_mode=True,
    )
    records = []
    for index, source in enumerate(inputs[:2], start=1):
        records.append(
            store.create_job(
                filename=f"recovery-{index}-{source.name}",
                content=source.read_bytes(),
                query="driver seatbelt review recovery smoke",
                settings=settings,
                source_metadata={"batchId": "batch-recovery-phase-r", "recoverOnRestart": True, "childSequence": index},
            )
        )
    stdout_path = work_dir / "recovery_worker_stdout.log"
    stderr_path = work_dir / "recovery_worker_stderr.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--recovery-worker",
        "--job-root",
        str(job_root),
        *[argument for record in records for argument in ("--job-id", record.job_id)],
    ]
    process: subprocess.Popen[str] | None = None
    interrupted = False
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, text=True)
            deadline = time.monotonic() + 180.0
            while time.monotonic() < deadline:
                states = _manifest_states(job_root, [record.job_id for record in records])
                if sum(state in ACTIVE for state in states.values()) >= 2:
                    interrupted = True
                    process.terminate()
                    process.wait(timeout=30)
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.35)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.kill()
        return {"passed": False, "interrupted": interrupted, "error": f"{type(exc).__name__}: {exc}"}

    # Treat the abrupt process stop exactly as the new backend would. This is
    # deliberately isolated from existing job folders.
    import src.api.jobs as jobs_module

    previous_stale_minutes = jobs_module.SETTINGS.stale_running_minutes
    jobs_module.SETTINGS.stale_running_minutes = 0.0
    try:
        restarted = JobStore(job_root)
        # The zero-minute threshold is only needed for this one restart scan.
        # Do not let a freshly persisted ``recovering`` state be classified as
        # stale again while we inspect it.
        recovered_states = {
            record.job_id: restarted._jobs[record.job_id].status  # noqa: SLF001 - controlled recovery probe
            for record in records
        }
        jobs_module.SETTINGS.stale_running_minutes = previous_stale_minutes
        rescheduled = []
        for record in records:
            current = restarted.require(record.job_id)
            if current.status == "recovering" and restarted.activate_retry(record.job_id):
                rescheduled.append(record.job_id)
        schedule_analysis_jobs(restarted, rescheduled)
        deadline = time.monotonic() + 900.0
        while time.monotonic() < deadline:
            if all(restarted.require(record.job_id).status in TERMINAL for record in records):
                break
            time.sleep(0.5)
        final_states = {record.job_id: restarted.require(record.job_id).status for record in records}
        duplicate_results = [
            record.job_id
            for record in records
            if len(list((job_root / record.job_id).glob("result*.json"))) > 1
        ]
        return {
            "interrupted": interrupted,
            "workerExitCode": process.returncode if process else None,
            "recoveredStates": recovered_states,
            "rescheduled": rescheduled,
            "finalStates": final_states,
            "duplicateResults": duplicate_results,
            "manifestTemps": _no_manifest_temps(job_root),
            "passed": (
                interrupted
                and len(rescheduled) == 2
                and all(state == "completed" for state in final_states.values())
                and not duplicate_results
                and not _no_manifest_temps(job_root)
            ),
        }
    finally:
        jobs_module.SETTINGS.stale_running_minutes = previous_stale_minutes


def run_smoke(output: Path, *, include_recovery: bool) -> dict[str, Any]:
    _configure_local_full()
    from fastapi.testclient import TestClient
    from src.api.batches import BatchStore
    from src.api.jobs import JobStore
    from src.api.server import create_app

    work_dir = output.parent / f"runtime_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    work_dir.mkdir(parents=True, exist_ok=False)
    inputs = _copy_inputs(work_dir)
    before = {"rssMb": _rss_mb(), "gpu": _gpu_memory()}
    jobs = JobStore(work_dir / "jobs")
    batches = BatchStore(work_dir / "batches")
    app = create_app(jobs, batches)
    report: dict[str, Any] = {
        "generatedAt": _utc(),
        "runtimeProfile": "local_full",
        "workDir": str(work_dir),
        "inputs": [{"name": item.name, "sizeBytes": item.stat().st_size, "sha256": _hash(item)} for item in inputs],
        "memoryBefore": before,
    }
    with TestClient(app) as client:
        health = client.get("/api/health")
        status = client.get("/api/system/status")
        report["health"] = {"statusCode": health.status_code, "body": health.json() if health.status_code == 200 else None}
        report["systemStatus"] = status.json() if status.status_code == 200 else {"statusCode": status.status_code}

        single_id = _post_single(client, inputs[0], query="general safety review")
        single_status, single_history = _poll_job(client, single_id)
        report["singleRealVideo"] = {
            "jobId": single_id,
            "status": single_status,
            "history": single_history,
            "result": _result_summary(client, single_id),
        }

        batch = _post_batch(client, inputs[:3], query="driver seatbelt compliance review")
        # Give the old batch a deterministic head start before a newer standalone
        # job enters the scheduler.
        time.sleep(0.25)
        later_id = _post_single(client, inputs[-1], query="later standalone queue fairness review")
        batch_result = _wait_batch_with_later_job(client, str(batch["batchId"]), later_id)
        child_summaries = {job_id: _result_summary(client, job_id) for job_id in batch["jobIds"]}
        start_times = {
            job_id: summary["status"].get("startedAt")
            for job_id, summary in child_summaries.items()
        }
        third_id = str(batch["jobIds"][2])
        later_start = batch_result["laterJob"].get("startedAt")
        report["batchAndFairness"] = {
            "batch": batch,
            "batchResult": batch_result,
            "children": child_summaries,
            "childStartTimes": start_times,
            "thirdChildBeforeLaterStandalone": bool(
                start_times.get(third_id) and later_start and start_times[third_id] <= later_start
            ),
            "scheduler": client.get("/api/system/status").json().get("queue", {}).get("scheduler", {}),
            "batchIdAsJobStatus": client.get(f"/api/jobs/{batch['batchId']}").status_code,
        }
        report["manifestTemps"] = _no_manifest_temps(work_dir)
        report["memoryAfterBatch"] = {"rssMb": _rss_mb(), "gpu": _gpu_memory()}

    report["recovery"] = _exercise_recovery(work_dir, inputs) if include_recovery else {
        "passed": False,
        "skipped": True,
        "reason": "Controlled real-worker interruption was not requested.",
    }
    batch_final = report["batchAndFairness"]["batchResult"]["final"]
    children_completed = all(
        details["status"].get("status") == "completed"
        for details in report["batchAndFairness"]["children"].values()
    )
    report["acceptance"] = {
        "preflightMustBeCheckedSeparately": True,
        "singleCompleted": report["singleRealVideo"]["status"].get("status") == "completed",
        "batchCompleted": batch_final.get("status") in {"completed", "completed_with_failures"},
        "allBatchChildrenCompleted": children_completed,
        "twoActiveJobsObserved": report["batchAndFairness"]["batchResult"]["twoActiveSeen"],
        "parentStayedActiveAfterChildCompleted": report["batchAndFairness"]["batchResult"]["parentActiveAfterChildComplete"],
        "olderBatchKeptPriority": report["batchAndFairness"]["thirdChildBeforeLaterStandalone"],
        "batchIdRejectedByJobsEndpoint": report["batchAndFairness"]["batchIdAsJobStatus"] == 404,
        "noManifestTemps": not report["manifestTemps"],
        "recoveryPassed": bool(report["recovery"].get("passed")),
    }
    report["acceptance"]["passed"] = all(report["acceptance"].values())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real Phase R CUDA scheduler smoke.")
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--skip-recovery", action="store_true")
    parser.add_argument("--recovery-worker", action="store_true")
    parser.add_argument("--job-root", type=Path)
    parser.add_argument("--job-id", action="append", default=[])
    args = parser.parse_args()
    if args.recovery_worker:
        if not args.job_root or not args.job_id:
            parser.error("--recovery-worker requires --job-root and --job-id")
        return _recovery_worker(args.job_root, list(args.job_id))
    output = args.output or REPO_ROOT / ".ai-pipeline" / "005_tasks" / "phase_r_runtime_parallel_scheduler" / "real_parallel_smoke.json"
    report = run_smoke(output, include_recovery=not args.skip_recovery)
    print(json.dumps(report["acceptance"], indent=2))
    return 0 if report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
