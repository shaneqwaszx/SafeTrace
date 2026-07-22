"""Exercise completed-child access while a real eight-video batch is active."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TERMINAL = {"completed", "failed", "cancelled"}
DEFAULT_SOURCE = ROOT / "data/api_jobs/job_20260701_131858_dba6eda0/uploads/sample-15s-720p.mp4"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resources() -> dict[str, Any]:
    result: dict[str, Any] = {"observedAt": _utc()}
    try:
        import psutil

        process = psutil.Process()
        result["rssMb"] = round(process.memory_info().rss / 1048576, 2)
        result["availableMemoryMb"] = round(psutil.virtual_memory().available / 1048576, 2)
    except (ImportError, OSError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            result["cudaAllocatedMb"] = round(torch.cuda.memory_allocated() / 1048576, 2)
            result["cudaReservedMb"] = round(torch.cuda.memory_reserved() / 1048576, 2)
    except ImportError:
        pass
    return result


def _child_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "jobId", "filename", "status", "queuePosition", "workerSlot", "capacityReason",
            "queueWaitSeconds", "analysisRuntimeSeconds", "elapsedSeconds",
        )
    }


def _fetch_completed_child(client: Any, job_id: str) -> dict[str, Any]:
    status = client.get(f"/api/jobs/{job_id}")
    result = client.get(f"/api/jobs/{job_id}/result")
    technical = client.get(f"/api/reports/{job_id}/technical-json")
    body = result.json() if result.status_code == 200 else {}
    return {
        "jobId": job_id,
        "statusCode": status.status_code,
        "resultStatusCode": result.status_code,
        "technicalStatusCode": technical.status_code,
        "status": status.json().get("status") if status.status_code == 200 else None,
        "resultJobId": body.get("jobId"),
        "findingCount": len(body.get("violations") or []),
        "evidenceCount": len(body.get("frames") or []),
        "engineMetricsPresent": bool(body.get("engineMetrics")),
    }


def run(source: Path, output: Path, *, timeout_seconds: float, work_dir: Path | None = None) -> dict[str, Any]:
    os.environ.setdefault("SAFETRACE_RUNTIME_PROFILE", "local_full")
    os.environ.setdefault("SAFETRACE_ANALYSIS_CONCURRENCY", "2")
    os.environ.setdefault("SAFETRACE_WORKER_CONCURRENCY", "2")
    os.environ.setdefault("SAFETRACE_VLM_ENABLED", "0")
    os.environ.setdefault("SAFETRACE_ENABLE_VLM", "0")
    os.environ.setdefault("SAFETRACE_LIGHTWEIGHT_VLM_ENABLED", "0")

    from fastapi.testclient import TestClient

    from src.api.batches import BatchStore
    from src.api.jobs import JobStore
    from src.api.server import create_app

    managed_work = work_dir is None
    if work_dir is None:
        from scripts.manage_dev_workspace import create_run_directory
        work_root = create_run_directory("smoke", "phase_w_partial_batch")
    else:
        work_root = work_dir.resolve()
        work_root.mkdir(parents=True, exist_ok=True)
    jobs = JobStore(work_root / "jobs")
    batches = BatchStore(work_root / "batches")
    app = create_app(jobs, batches)
    content = source.read_bytes()
    files = [
        ("files", (f"phase-w-{index + 1:02d}-{source.name}", content, "video/mp4"))
        for index in range(8)
    ]
    observations: list[dict[str, Any]] = []
    opened: list[dict[str, Any]] = []
    first_selected: str | None = None
    parent_active_when_opened = False
    second_open_preserved_first_selection = False
    started = time.monotonic()

    with TestClient(app) as client:
        created = client.post(
            "/api/batches/analyze",
            files=files,
            data={
                "query": "driver or occupant without seatbelt",
                "fps": "1",
                "topK": "2",
                "enableVlm": "false",
                "vlmEnabled": "false",
                "vlmProfile": "rule_based",
                "device": "cuda",
                "reviewMode": "fast_local",
                "useCaseProfile": json.dumps({
                    "profileId": "general_safety",
                    "label": "General safety review",
                    "effectiveQuery": "driver or occupant without seatbelt",
                }),
            },
        )
        created.raise_for_status()
        batch = created.json()
        batch_id = str(batch["batchId"])
        deadline = time.monotonic() + timeout_seconds
        final: dict[str, Any] | None = None

        while time.monotonic() < deadline:
            response = client.get(f"/api/batches/{batch_id}")
            response.raise_for_status()
            payload = response.json()
            children = list(payload.get("acceptedFiles") or [])
            completed = [item for item in children if item.get("status") == "completed"]
            observations.append({
                "observedAt": _utc(),
                "batchStatus": payload.get("status"),
                "completed": len(completed),
                "total": len(children),
                "children": [_child_snapshot(item) for item in children],
                "resources": _resources(),
            })
            if completed and not opened:
                first_selected = str(completed[0]["jobId"])
                opened.append(_fetch_completed_child(client, first_selected))
                parent_active_when_opened = payload.get("status") not in TERMINAL
            elif len(completed) >= 2 and len(opened) == 1:
                second_id = next(
                    str(item["jobId"]) for item in completed if str(item["jobId"]) != first_selected
                )
                opened.append(_fetch_completed_child(client, second_id))
                second_open_preserved_first_selection = first_selected == opened[0]["jobId"]
            if payload.get("status") in TERMINAL:
                final = payload
                break
            time.sleep(0.35)

        if final is None:
            raise TimeoutError(f"Batch {batch_id} did not finish within {timeout_seconds:.0f}s")
        wrong_job_status = client.get(f"/api/jobs/{batch_id}").status_code
        child_summaries = [_fetch_completed_child(client, str(job_id)) for job_id in batch["jobIds"]]

    manifest_temps = [str(path) for path in work_root.rglob("manifest.json.*.tmp")]
    payload = {
        "generatedAt": _utc(),
        "source": str(source),
        "batchId": batch_id,
        "jobIds": list(batch["jobIds"]),
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "parentActiveWhenFirstChildOpened": parent_active_when_opened,
        "openedCompletedChildren": opened,
        "secondOpenPreservedFirstSelection": second_open_preserved_first_selection,
        "wrongBatchAsJobStatusCode": wrong_job_status,
        "finalBatch": final,
        "childSummaries": child_summaries,
        "observations": observations,
        "manifestTempFiles": manifest_temps,
        "resourcesBefore": observations[0]["resources"] if observations else {},
        "resourcesAfter": _resources(),
        "acceptance": {
            "allChildrenCompleted": all(item["status"] == "completed" for item in child_summaries),
            "completedChildAccessibleWhileParentActive": parent_active_when_opened and bool(opened),
            "twoCompletedChildrenOpened": len(opened) >= 2,
            "selectionStayedStable": second_open_preserved_first_selection,
            "batchJobIdRejected": wrong_job_status == 404,
            "noManifestTemps": not manifest_temps,
        },
    }
    payload["passed"] = all(payload["acceptance"].values())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if managed_work and payload["passed"]:
        from scripts.manage_dev_workspace import finalize_run_directory
        finalize_run_directory(work_root, succeeded=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".ai-pipeline/005_tasks/phase_w_general_profile_performance_cleanup/partial_batch_access.json",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = run(
        source,
        args.output.resolve(),
        timeout_seconds=args.timeout_seconds,
        work_dir=args.work_dir,
    )
    print(json.dumps({"passed": payload["passed"], "batchId": payload["batchId"], **payload["acceptance"]}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
