"""Automated SafeTrace lifecycle smoke and accelerated synthetic stress run."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

import src.api.jobs as jobs_module
from src.api.batches import BatchStore
from src.api.jobs import AnalysisSettings, JobStore, execute_analysis_job
from src.api.recovery_checkpoints import latest_valid_checkpoint, write_checkpoint
from src.api.result_cache import cache_key
from src.api.retention_scheduler import RetentionScheduler
from src.api.server import create_app
from src.config import SETTINGS


def settings() -> AnalysisSettings:
    return AnalysisSettings(
        fps=1.0, top_k=5, enable_vlm=False, device="cpu", vlm_profile="rule_based",
        vlm_enabled=False, safe_mode=True, use_case_profile={"profileId": "general_safety"},
        review_mode="fast_local",
    )


def integrity(root: Path) -> dict[str, Any]:
    manifests = list(root.rglob("manifest.json"))
    invalid = []
    ids = []
    for path in manifests:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid.append(str(path)); continue
        identifier = payload.get("jobId") or payload.get("batchId")
        if identifier: ids.append(str(identifier))
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    temps = [str(path) for path in root.rglob("*.tmp")]
    partial_exports = [str(path) for path in root.rglob(".export_*.tmp")]
    return {"manifestCount": len(manifests), "invalid": invalid, "duplicateIds": duplicates, "temporaryFiles": temps, "partialExports": partial_exports, "passed": not invalid and not duplicates and not temps and not partial_exports}


def tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def rss_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def run(output_dir: Path, stress_jobs: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    original_pipeline = jobs_module.run_pipeline
    original_data = SETTINGS.data_dir
    original_scheduler = SETTINGS.retention_scheduler_enabled
    pipeline_calls = 0

    def fake_pipeline(**_kwargs):
        nonlocal pipeline_calls
        pipeline_calls += 1
        return []

    jobs_module.run_pipeline = fake_pipeline
    SETTINGS.retention_scheduler_enabled = False
    started = time.perf_counter()
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, details: Any = None) -> None:
        checks.append({"name": name, "passed": bool(condition), "details": details})
        if not condition:
            raise AssertionError(f"{name}: {details}")

    try:
        temporary_root = output_dir / f".workflow_runtime_{uuid.uuid4().hex}"
        temporary_root.mkdir(parents=True)
        try:
            runtime = temporary_root / "data"
            SETTINGS.data_dir = runtime
            storage_before = tree_bytes(runtime)
            memory_before = rss_bytes()
            jobs = JobStore(runtime / "api_jobs")
            batches = BatchStore(runtime / "api_batches")
            app = create_app(jobs, batches)
            with TestClient(app) as client:
                check("health", client.get("/api/health").status_code == 200)
                check("system", client.get("/api/system/status").status_code == 200)
                response = client.post(
                    "/api/batches/analyze",
                    files=[
                        ("files", ("camera-a.mp4", b"video-a", "video/mp4")),
                        ("files", ("camera-a-copy.mp4", b"video-a", "video/mp4")),
                        ("files", ("notes.txt", b"invalid", "text/plain")),
                        ("query", (None, "general safety")),
                        ("relativePaths", (None, "Fleet A/Day/camera-a.mp4")),
                        ("relativePaths", (None, "Fleet A/Day/camera-a-copy.mp4")),
                        ("relativePaths", (None, "Fleet A/notes.txt")),
                        ("importKey", (None, "phase-q-full-workflow")),
                    ],
                )
                check("nested_batch_created", response.status_code == 200, response.text)
                batch = response.json()
                check("partial_acceptance", len(batch["acceptedFiles"]) == 2 and len(batch["rejectedFiles"]) == 1, batch)
                check("hierarchy", bool(batch.get("hierarchy")))
                batch_status = client.get(f"/api/batches/{batch['batchId']}").json()
                check("batch_completed", batch_status["status"] in {"completed", "completed_with_failures"}, batch_status["status"])
                repeated = client.post(
                    "/api/batches/analyze",
                    files=[
                        ("files", ("repeat.mp4", b"repeat-video", "video/mp4")),
                        ("files", ("repeat.txt", b"invalid", "text/plain")),
                        ("query", (None, "general safety")),
                        ("relativePaths", (None, "Fleet A/Night/repeat.mp4")),
                        ("relativePaths", (None, "Fleet A/Night/repeat.txt")),
                        ("importKey", (None, "phase-q-repeat")),
                    ],
                )
                check("repeated_batch", repeated.status_code == 200 and len(repeated.json()["acceptedFiles"]) == 1, repeated.text)

                interrupted = jobs.create_job(
                    filename="interrupted.mp4", content=b"interrupt", query="general safety", settings=settings(),
                    source_metadata={"recoverOnRestart": True, "sourceRelativePath": "Fleet B/interrupted.mp4", "sourceGroupPath": "Fleet B"},
                )
                jobs.update_status(interrupted.job_id, status="running", progress=0.5, current_step="Detector frame loop")
                interrupted.updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
                jobs.persist_job(interrupted)

            restarted_jobs = JobStore(runtime / "api_jobs")
            restarted_batches = BatchStore(runtime / "api_batches")
            with TestClient(create_app(restarted_jobs, restarted_batches)) as client:
                recovery = client.get("/api/recovery").json()
                check("recovery_discovered", interrupted.job_id in {item["jobId"] for item in recovery["jobs"]}, recovery)
                check("later", client.post("/api/recovery/later", json={"jobIds": [interrupted.job_id]}).status_code == 200)
                check("later_not_reprompted", client.get("/api/recovery").json()["candidateCount"] == 0)
                check("resume_deferred", client.post("/api/recovery/resume", json={"jobIds": [interrupted.job_id]}).status_code == 200)
                check("resumed_completed", client.get(f"/api/jobs/{interrupted.job_id}").json()["status"] == "completed")

                discard = restarted_jobs.create_job(filename="discard.mp4", content=b"discard", query="general safety", settings=settings(), source_metadata={"recoverOnRestart": True})
                restarted_jobs.update_status(discard.job_id, status="running", progress=0.45, current_step="Detector frame loop")
                discard = restarted_jobs.require(discard.job_id)
                discard.updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
                restarted_jobs.persist_job(discard)
                client.get("/api/recovery")
                discarded = client.post("/api/recovery/discard", json={"jobIds": [discard.job_id]})
                check("discard_injection", discarded.status_code == 200 and not discard.job_dir.exists(), discarded.text)

                retry = restarted_jobs.create_job(filename="retry.mp4", content=b"retry", query="general safety", settings=settings())
                restarted_jobs.update_status(retry.job_id, status="failed", progress=1.0, current_step="Injected transient failure", error="injected", error_type="OSError")
                check("retry_activated", restarted_jobs.reset_for_retry(retry.job_id))
                execute_analysis_job(restarted_jobs, retry.job_id)
                check("retry_completed", restarted_jobs.require(retry.job_id).status == "completed")

                corrupt = restarted_jobs.create_job(filename="corrupt.mp4", content=b"corrupt", query="general safety", settings=settings())
                restarted_jobs.persist_recovery_stage(corrupt.job_id, "frame_sampling_complete", completed_frame_index=3)
                corrupt_record = restarted_jobs.require(corrupt.job_id)
                newest = write_checkpoint(
                    corrupt_record.job_dir, job_id=corrupt.job_id, stage="detector_work_complete",
                    identity=dict(corrupt_record.metrics.get("executionIdentity") or {}), progress=0.6,
                    status="running", current_step="detector", completed_frame_index=4,
                )
                newest.write_text("{corrupt", encoding="utf-8")
                selected = latest_valid_checkpoint(
                    corrupt_record.job_dir,
                    dict(corrupt_record.metrics.get("executionIdentity") or {}),
                    expected_job_id=corrupt_record.job_id,
                )
                check("corrupt_checkpoint_fallback", selected["checkpoint"]["stage"] == "frame_sampling_complete", selected)
                restarted_jobs.delete(corrupt.job_id)

                source = restarted_jobs.require(interrupted.job_id)
                exact = restarted_jobs.create_job(filename="exact.mp4", content=b"interrupt", query="general safety", settings=settings())
                exact.source_metadata["checksumSha256"] = source.source_metadata["checksumSha256"]
                restarted_jobs.persist_job(exact)
                calls_before = pipeline_calls
                execute_analysis_job(restarted_jobs, exact.job_id)
                check("exact_cache_hit", pipeline_calls == calls_before and restarted_jobs.require(exact.job_id).metrics.get("resultCacheHit") is True)
                changed = restarted_jobs.create_job(filename="changed.mp4", content=b"interrupt", query="changed query", settings=settings())
                changed.source_metadata["checksumSha256"] = source.source_metadata["checksumSha256"]
                restarted_jobs.persist_job(changed)
                execute_analysis_job(restarted_jobs, changed.job_id)
                check("query_cache_miss", pipeline_calls == calls_before + 1 and cache_key(changed) != cache_key(source))

                completed_id = batch["jobIds"][0]
                check("pin", client.post(f"/api/jobs/{completed_id}/pin").status_code == 200)
                exported = client.post(f"/api/jobs/{completed_id}/export", json={"selectedEvidenceIds": []}).json()
                check("export_verified", exported.get("verified") is True, exported)
                preview = client.post("/api/storage/cleanup/preview", json={}).json()
                check("pinned_protected", completed_id not in {item.get("id") for item in preview["candidates"]})
                check("cleanup_apply", client.post("/api/storage/cleanup/apply", json={"confirm": True}).status_code == 200)
                check("unpin", client.post(f"/api/jobs/{completed_id}/unpin").status_code == 200)

                delete_target = batch["jobIds"][1]
                exported_deleted = client.post(f"/api/jobs/{delete_target}/export-and-delete", json={"confirmDelete": True}).json()
                check("export_and_delete", exported_deleted.get("status") == "exported_and_deleted", exported_deleted)
                check("owned_data_removed", client.get(f"/api/jobs/{delete_target}").status_code == 404)
                check("dashboard_updated", client.get("/api/dashboard/summary").status_code == 200)
                check("storage_updated", client.get("/api/storage/summary").status_code == 200)

                stress_started = time.perf_counter()
                stress_records = []
                for index in range(stress_jobs):
                    query = f"stress-{index % 5}"
                    media = f"payload-{index % 10}".encode()
                    record = restarted_jobs.create_job(filename=f"stress-{index}.mp4", content=media, query=query, settings=settings())
                    stress_records.append(record)
                sequential = stress_records[: stress_jobs // 2]
                parallel = stress_records[stress_jobs // 2 :]
                for record in sequential:
                    execute_analysis_job(restarted_jobs, record.job_id)
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="phase-q-stress") as executor:
                    list(executor.map(lambda item: execute_analysis_job(restarted_jobs, item.job_id), parallel))
                stress_ids = [record.job_id for record in stress_records]
                restart_probe = JobStore(runtime / "api_jobs")
                check("stress_restart_continuity", all(restart_probe.require(job_id).status == "completed" for job_id in stress_ids))
                stress_statuses = [restarted_jobs.require(job_id).status for job_id in stress_ids]
                check("stress_completed", all(status == "completed" for status in stress_statuses), stress_statuses)
                dashboard_latencies = []
                for _ in range(10):
                    dashboard_started = time.perf_counter()
                    status = client.get("/api/dashboard/summary").status_code
                    dashboard_latencies.append(time.perf_counter() - dashboard_started)
                    check("dashboard_poll", status == 200, status)
                low_disk_before = len(restarted_jobs.records())
                original_min = SETTINGS.min_free_disk_gb
                SETTINGS.min_free_disk_gb = 10_000.0
                try:
                    low = client.post("/api/analyze", files={"file": ("low.mp4", b"x", "video/mp4")}, data={"query": "test"})
                finally:
                    SETTINGS.min_free_disk_gb = original_min
                check("low_disk_rejected", low.status_code == 507 and len(restarted_jobs.records()) == low_disk_before)
                retention = RetentionScheduler(restarted_jobs, restarted_batches).run_once(apply=False)
                check("retention_cycle", retention["status"] == "previewed", retention)
                stress_seconds = time.perf_counter() - stress_started

            final_integrity = integrity(runtime)
            check("final_integrity", final_integrity["passed"], final_integrity)
            report = {
                "schemaVersion": 1,
                "startedAt": datetime.now(timezone.utc).isoformat(),
                "durationSeconds": round(time.perf_counter() - started, 3),
                "pipelineCalls": pipeline_calls,
                "checks": checks,
                "stress": {
                    "requestedJobs": stress_jobs,
                    "completedJobs": stress_jobs,
                    "durationSeconds": round(stress_seconds, 3),
                    "concurrencyRuns": {"1": len(sequential), "2": len(parallel)},
                    "restartInjections": 2,
                    "dashboardPolls": 10,
                    "dashboardLatencyMeanMs": round(sum(dashboard_latencies) / len(dashboard_latencies) * 1000, 3),
                    "dashboardLatencyMaxMs": round(max(dashboard_latencies) * 1000, 3),
                    "retriedJobs": 1,
                    "discardedJobs": 1,
                    "corruptCheckpointFallbacks": 1,
                    "recoveryRuns": 1,
                    "cacheChecks": 2,
                    "schedulerOverlapSkips": 0,
                    "unhandledExceptions": 0,
                },
                "storage": {"beforeBytes": storage_before, "afterBytes": tree_bytes(runtime)},
                "memory": {"beforeRssBytes": memory_before, "afterRssBytes": rss_bytes()},
                "integrity": final_integrity,
                "passed": all(item["passed"] for item in checks),
            }
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
    finally:
        jobs_module.run_pipeline = original_pipeline
        SETTINGS.data_dir = original_data
        SETTINGS.retention_scheduler_enabled = original_scheduler

    json_path = output_dir / "full_workflow_smoke.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Full User Workflow Smoke", "", f"- Passed: `{report['passed']}`", f"- Duration: `{report['durationSeconds']} s`", f"- Stress jobs: `{stress_jobs}`", "", "## Checks", ""]
    lines.extend(f"- {'PASS' if item['passed'] else 'FAIL'}: {item['name']}" for item in checks)
    (output_dir / "full_workflow_smoke.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(checks), "stressJobs": stress_jobs, "output": str(json_path)}))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stress-jobs", type=int, default=30)
    args = parser.parse_args(argv)
    report = run(args.output_dir, max(0, args.stress_jobs))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
