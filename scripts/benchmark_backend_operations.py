"""Measure SafeTrace metadata and storage operations without changing inference."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.api.batches import BatchStore
from src.api.jobs import JobStore
from src.api.operations import dashboard_summary, storage_summary
from src.api.server import create_app


def measure(action: Callable[[], Any], iterations: int) -> dict[str, Any]:
    values = []
    for _ in range(iterations):
        started = time.perf_counter()
        action()
        values.append((time.perf_counter() - started) * 1000)
    ordered = sorted(values)
    return {
        "iterations": iterations,
        "meanMs": statistics.mean(values),
        "p50Ms": statistics.median(values),
        "p95Ms": ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.95)))],
        "maxMs": max(values),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args(argv)
    jobs = JobStore(args.runtime_root / "api_jobs")
    batches = BatchStore(args.runtime_root / "api_batches")
    app = create_app(jobs, batches)
    with TestClient(app) as client:
        report = {
            "runtimeRoot": str(args.runtime_root),
            "jobCount": len(jobs.records()),
            "batchCount": len(batches.records()),
            "dashboardHelper": measure(lambda: dashboard_summary(jobs, batches), args.iterations),
            "dashboardApi": measure(lambda: client.get("/api/dashboard/summary").raise_for_status(), args.iterations),
            "batchListApi": measure(lambda: client.get("/api/batches?page=1&pageSize=25").raise_for_status(), args.iterations),
            "jobListApi": measure(lambda: client.get("/api/jobs?page=1&pageSize=25").raise_for_status(), args.iterations),
            "storageScan": measure(lambda: storage_summary(jobs, batches), max(1, min(args.iterations, 3))),
        }
        first_completed = next((item for item in jobs.records() if item.status == "completed"), None)
        if first_completed:
            report["jobStatusApi"] = measure(lambda: client.get(f"/api/jobs/{first_completed.job_id}").raise_for_status(), args.iterations)
            report["resultLoadApi"] = measure(lambda: client.get(f"/api/jobs/{first_completed.job_id}/result").raise_for_status(), args.iterations)
            report["sampleJob"] = {
                "jobId": first_completed.job_id,
                "fileCount": sum(1 for item in first_completed.job_dir.rglob("*") if item.is_file()),
                "bytes": sum(item.stat().st_size for item in first_completed.job_dir.rglob("*") if item.is_file()),
                "detectorLoadSeconds": ((first_completed.result or {}).get("engineMetrics") or {}).get("stageDurations", {}).get("detectorLoad"),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
