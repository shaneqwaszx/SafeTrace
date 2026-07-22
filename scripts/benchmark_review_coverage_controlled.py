"""Controlled Fast versus Comprehensive review benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE = ROOT / "data/api_jobs/job_20260701_131858_dba6eda0/uploads/sample-15s-720p.mp4"
QUERY = "driver or occupant without seatbelt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resources() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        import psutil
        payload["rssMb"] = round(psutil.Process().memory_info().rss / 1048576, 2)
    except (ImportError, OSError):
        pass
    try:
        import torch
        if torch.cuda.is_available():
            payload.update({
                "cudaAllocatedMb": round(torch.cuda.memory_allocated() / 1048576, 2),
                "cudaReservedMb": round(torch.cuda.memory_reserved() / 1048576, 2),
            })
    except ImportError:
        pass
    return payload


def _execute(source: Path, *, mode: str, profile: str, store: Any, bypass_cache: bool, label: str) -> dict[str, Any]:
    from src.api.jobs import AnalysisSettings, execute_analysis_job

    before = _resources()
    started = time.perf_counter()
    record = store.create_job(
        filename=source.name,
        content=source.read_bytes(),
        query=QUERY,
        settings=AnalysisSettings(
            fps=3.0,
            top_k=5,
            enable_vlm=False,
            device="cuda",
            vlm_profile="rule_based",
            vlm_enabled=False,
            safe_mode=True,
            use_case_profile={"profileId": profile, "effectiveQuery": QUERY},
            review_mode=mode,
        ),
        source_metadata={"sourceRelativePath": f"phase-w-benchmark/{label}/{source.name}"},
    )
    record.metrics["bypassResultCacheOnce"] = bool(bypass_cache)
    execute_analysis_job(store, record.job_id)
    completed = store.require(record.job_id)
    wall = time.perf_counter() - started
    result = completed.result or {}
    technical = dict(result.get("technicalDetails") or {})
    diagnostics = dict(technical.get("componentDiagnostics") or {})
    engine = dict(result.get("engineMetrics") or {})
    processing = dict(technical.get("processingMetadata") or {})
    timing = completed.timing_payload()
    return {
        "jobId": record.job_id,
        "status": completed.status,
        "wallSeconds": round(wall, 4),
        "queueWaitSeconds": timing.get("queueWaitSeconds"),
        "processingRuntimeSeconds": timing.get("analysisRuntimeSeconds"),
        "totalElapsedSeconds": timing.get("elapsedSeconds"),
        "cacheHit": bool(completed.metrics.get("resultCacheHit")),
        "cacheKey": completed.metrics.get("resultCacheKey"),
        "sampledFrameCount": processing.get("sampledFrameCount"),
        "sourceVideoFrameCount": processing.get("sourceVideoFrameCount"),
        "windowCount": processing.get("windowCount"),
        "selectedEvidenceFrameCount": len(result.get("frames") or []),
        "findingNames": [str(item.get("id")) for item in result.get("violations") or []],
        "findingCount": len(result.get("violations") or []),
        "evidenceCount": len(result.get("evidence") or []),
        "stageDurations": engine.get("stageDurations") or {},
        "engineCounts": engine.get("counts") or {},
        "optionalWorkers": engine.get("optionalWorkers") or {},
        "componentTimings": diagnostics.get("timings") or {},
        "workerCount": diagnostics.get("analysisConcurrency"),
        "thirdWorkerAdmission": diagnostics.get("thirdWorkerAdmission"),
        "resourcesBefore": before,
        "resourcesAfter": _resources(),
    }


def _child(args: argparse.Namespace) -> int:
    os.environ.update({
        "SAFETRACE_RUNTIME_PROFILE": "local_full",
        "SAFETRACE_DEVICE": "cuda",
        "SAFETRACE_ANALYSIS_SAFE_MODE": "true",
        "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM": "true",
        "SAFETRACE_MOBILESAM_ENABLED": "auto",
        "SAFETRACE_MOBILESAM_WORKER_ENABLED": "true",
        "SAFETRACE_ENABLE_VLM": "false",
        "SAFETRACE_MID_VLM_ENABLED": "false",
        "SAFETRACE_ANALYSIS_CONCURRENCY": "1",
        "SAFETRACE_ANALYSIS_MIN_WORKERS": "1",
        "SAFETRACE_ANALYSIS_MAX_WORKERS": "1",
    })
    from src.api.jobs import JobStore

    store = JobStore(args.child_work / "jobs")
    warmup = None
    if args.state in {"warm", "cache_enabled"}:
        warmup = _execute(
            args.source,
            mode=args.mode,
            profile=args.profile,
            store=store,
            bypass_cache=args.state == "warm",
            label="warmup",
        )
    measured = _execute(
        args.source,
        mode=args.mode,
        profile=args.profile,
        store=store,
        bypass_cache=args.state != "cache_enabled",
        label="measured",
    )
    payload = {"warmup": warmup, "measured": measured}
    args.child_output.parent.mkdir(parents=True, exist_ok=True)
    args.child_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if measured["status"] == "completed" else 2


def _median(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(statistics.median(values), 4) if values else None


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Controlled Fast vs Comprehensive performance",
        "",
        "No general speed claim is made outside these controlled sources and settings.",
        "",
        "| Profile | Coverage | State | Repeats | Median processing | Cache hits | Findings | Evidence |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['profile']} | {row['mode']} | {row['state']} | {row['repeats']} | "
            f"{row['medianProcessingSeconds']} | {row['cacheHits']} | {', '.join(row['findingNames']) or 'none'} | {row['evidenceCountRange']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        report["interpretation"],
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / ".ai-pipeline/005_tasks/phase_w_general_profile_performance_cleanup/controlled_performance_benchmark.json")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=("fast_local", "comprehensive"))
    parser.add_argument("--profile", choices=("seatbelt_compliance", "general_safety"))
    parser.add_argument("--state", choices=("cold", "warm", "cache_enabled"))
    parser.add_argument("--child-work", type=Path)
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    args.source = args.source.resolve()
    if args.child:
        return _child(args)
    if not args.source.is_file():
        raise SystemExit(f"Benchmark source not found: {args.source}")
    managed_work = args.work_dir is None
    if args.work_dir:
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=True)
    else:
        from scripts.manage_dev_workspace import create_run_directory
        work = create_run_directory("smoke", "phase_w_performance")
    combinations = [
        (profile, mode, state, repeat)
        for repeat in range(1, max(args.repeats, 1) + 1)
        for state in ("cold", "warm", "cache_enabled")
        for profile, mode in (
            ("seatbelt_compliance", "fast_local"),
            ("seatbelt_compliance", "comprehensive"),
            ("general_safety", "comprehensive"),
            ("general_safety", "fast_local"),
        )
    ]
    rows: list[dict[str, Any]] = []
    for index, (profile, mode, state, repeat) in enumerate(combinations, start=1):
        child_root = work / f"run_{index:03d}_{profile}_{mode}_{state}_{repeat}"
        output = child_root / "result.json"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--child",
            "--source", str(args.source), "--mode", mode, "--profile", profile,
            "--state", state, "--child-work", str(child_root), "--child-output", str(output),
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1800, check=False)
        if completed.returncode != 0 or not output.is_file():
            rows.append({
                "profile": profile, "mode": mode, "state": state, "repeat": repeat,
                "status": "failed", "returnCode": completed.returncode,
                "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-4000:],
            })
            continue
        payload = json.loads(output.read_text(encoding="utf-8"))
        rows.append({
            "profile": profile, "mode": mode, "state": state, "repeat": repeat,
            "status": "completed", "warmup": payload.get("warmup"), **payload["measured"],
        })
    summary: list[dict[str, Any]] = []
    for profile in ("seatbelt_compliance", "general_safety"):
        for mode in ("fast_local", "comprehensive"):
            for state in ("cold", "warm", "cache_enabled"):
                group = [row for row in rows if row.get("profile") == profile and row.get("mode") == mode and row.get("state") == state and row.get("status") == "completed"]
                findings = sorted({name for row in group for name in row.get("findingNames") or []})
                evidence = [int(row.get("evidenceCount") or 0) for row in group]
                summary.append({
                    "profile": profile,
                    "mode": mode,
                    "state": state,
                    "repeats": len(group),
                    "medianProcessingSeconds": _median(group, "processingRuntimeSeconds"),
                    "medianWallSeconds": _median(group, "wallSeconds"),
                    "cacheHits": sum(bool(row.get("cacheHit")) for row in group),
                    "findingNames": findings,
                    "evidenceCountRange": f"{min(evidence)}-{max(evidence)}" if evidence else "none",
                })
    passed = len(rows) == len(combinations) and all(row.get("status") == "completed" for row in rows)
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "source": str(args.source),
        "sourceChecksum": _sha256(args.source),
        "controls": {
            "query": QUERY, "device": "cuda", "workerMin": 1, "workerMax": 1,
            "vlm": False, "mobileSam": "auto-worker", "evidenceSettings": "unchanged",
            "runOrder": "counterbalanced_with_fresh_process_per_case",
        },
        "repeatTarget": args.repeats,
        "rows": rows,
        "summary": summary,
        "interpretation": (
            "The table reports measured differences for one controlled source only. Cache-hit materialization is "
            "separated from model processing, and cold/warm state uses process isolation; these results do not "
            "establish a universal Fast-versus-Comprehensive speed ordering."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    if managed_work and passed:
        from scripts.manage_dev_workspace import finalize_run_directory
        finalize_run_directory(work, succeeded=True)
    print(json.dumps({"passed": passed, "output": str(args.output), "summary": summary}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
