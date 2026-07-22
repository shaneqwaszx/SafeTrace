"""Compare Seatbelt Compliance with General Safety on identical controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUERY = "driver or occupant without seatbelt"
DEFAULT_EXTERIOR = ROOT / ".ai-pipeline/005_tasks/phase_o_accuracy_soak_detector_readiness/baseline_fast_cuda_c1/runtime/runtime-sample-5s/jobs/job_20260715_152803_0a39d82f/uploads/sample-5s-720p.mp4"
DEFAULT_REPEAT = ROOT / "data/api_jobs/job_20260701_131858_dba6eda0/uploads/sample-15s-720p.mp4"


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(source: Path, *, control_id: str, profile: str, work_dir: Path) -> dict[str, Any]:
    from src.api.jobs import AnalysisSettings, JobStore, execute_analysis_job

    run_root = work_dir / f"{control_id}_{profile}"
    store = JobStore(run_root / "jobs")
    record = store.create_job(
        filename=source.name,
        content=source.read_bytes(),
        query=QUERY,
        settings=AnalysisSettings(
            fps=1.0,
            top_k=5,
            enable_vlm=False,
            device="cuda",
            vlm_profile="rule_based",
            vlm_enabled=False,
            safe_mode=True,
            use_case_profile={
                "profileId": profile,
                "label": "Seatbelt Compliance" if profile == "seatbelt_compliance" else "General safety review",
                "effectiveQuery": QUERY,
            },
            review_mode="fast_local",
        ),
        source_metadata={"sourceRelativePath": f"phase-w/{control_id}/{source.name}"},
    )
    record.metrics["bypassResultCacheOnce"] = True
    execute_analysis_job(store, record.job_id)
    completed = store.require(record.job_id)
    result = completed.result or {}
    technical = dict(result.get("technicalDetails") or {})
    diagnostics = dict(technical.get("componentDiagnostics") or {})
    processing = dict(technical.get("processingMetadata") or {})
    scene_records = list(diagnostics.get("sceneApplicabilityRecords") or [])
    findings = list(result.get("violations") or [])
    review_levels = {
        str(item.get("id")): str(item.get("reviewLevel") or item.get("evidenceStrength") or "")
        for item in findings
    }
    origins = sorted({
        str(item.get("originProfileId"))
        for item in findings
        if item.get("originProfileId")
    })
    sampling_runs = list(processing.get("samplingRuns") or [])
    sampled_ids = [
        str(frame.get("sourceFrameIndex"))
        for run in sampling_runs
        for frame in list(run.get("sampledFrames") or [])
    ]
    selected_ids = [str(frame.get("sourceFrameIndex")) for frame in result.get("frames") or []]
    suppressed = [
        item
        for scene_record in scene_records
        for item in list(scene_record.get("suppressedFindings") or [])
    ]
    return {
        "controlId": control_id,
        "profile": profile,
        "query": QUERY,
        "source": str(source),
        "sourceChecksum": _checksum(source),
        "jobId": record.job_id,
        "status": completed.status,
        "sampledFrameIds": sampled_ids,
        "selectedFrameIds": selected_ids,
        "sampledFrameCount": processing.get("sampledFrameCount"),
        "profileApplicable": any(bool(item.get("scene", {}).get("profileApplicable")) for item in scene_records),
        "personVisibleFrames": sum(bool(item.get("scene", {}).get("personVisible")) for item in scene_records),
        "preGateCandidates": sum(int(item.get("preGateCandidateCount") or 0) for item in scene_records),
        "postGateCandidates": sum(int(item.get("postGateCandidateCount") or 0) for item in scene_records),
        "acceptedFindingNames": [str(item.get("id")) for item in findings],
        "originProfiles": origins,
        "reviewLevels": review_levels,
        "eventCount": len(result.get("events") or []),
        "evidenceCount": len(result.get("evidence") or []),
        "suppressionReasons": sorted({str(item.get("reason")) for item in suppressed}),
        "composition": diagnostics.get("generalProfileComposition"),
        "manifestTempFiles": [str(path) for path in run_root.rglob("manifest.json.*.tmp")],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# General profile parity",
        "",
        f"Generated: {report['generatedAt']}",
        "",
        f"Verdict: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "| Control | Profile | Sampled | Candidates | Findings | Origins | Evidence |",
        "|---|---|---:|---:|---|---|---:|",
    ]
    for row in report["controls"]:
        lines.append(
            f"| {row['controlId']} | {row['profile']} | {row.get('sampledFrameCount')} | "
            f"{row['postGateCandidates']} | {', '.join(row['acceptedFindingNames']) or 'none'} | "
            f"{', '.join(row['originProfiles']) or 'none'} | {row['evidenceCount']} |"
        )
    lines.extend(["", "## Checks", "", *[f"- {key}: {value}" for key, value in report["checks"].items()]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", type=Path, default=ROOT / ".ai-pipeline/video/video_2026-06-18_11-38-42.mp4")
    parser.add_argument("--repeat", type=Path, default=DEFAULT_REPEAT)
    parser.add_argument("--exterior", type=Path, default=DEFAULT_EXTERIOR)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / ".ai-pipeline/005_tasks/phase_w_general_profile_performance_cleanup/general_profile_parity_report.json")
    args = parser.parse_args()

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
    })
    sources = [
        ("interior_positive", args.positive.resolve()),
        ("repeat_control", args.repeat.resolve()),
        ("exterior_negative", args.exterior.resolve()),
    ]
    missing = [str(path) for _, path in sources if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing parity source(s): {missing}")
    managed_work = args.work_dir is None
    if args.work_dir:
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=True)
    else:
        from scripts.manage_dev_workspace import create_run_directory
        work = create_run_directory("smoke", "phase_w_general_parity")
    controls = [
        _run(path, control_id=control, profile=profile, work_dir=work)
        for control, path in sources
        for profile in ("seatbelt_compliance", "general_safety")
    ]
    by_key = {(item["controlId"], item["profile"]): item for item in controls}
    positive_seatbelt = by_key[("interior_positive", "seatbelt_compliance")]
    positive_general = by_key[("interior_positive", "general_safety")]
    checks = {
        "identicalPositiveSampling": positive_seatbelt["sampledFrameIds"] == positive_general["sampledFrameIds"],
        "seatbeltPositivePresent": "seatbelt_missing" in positive_seatbelt["acceptedFindingNames"],
        "generalSeatbeltParity": "seatbelt_missing" in positive_general["acceptedFindingNames"],
        "generalRetainsSeatbeltOrigin": "seatbelt_compliance" in positive_general["originProfiles"],
        "generalRetainsReviewLevel": positive_general["reviewLevels"].get("seatbelt_missing") == "review_candidate",
        "exteriorSeatbeltSuppressed": not by_key[("exterior_negative", "seatbelt_compliance")]["acceptedFindingNames"],
        "exteriorGeneralSuppressed": not by_key[("exterior_negative", "general_safety")]["acceptedFindingNames"],
        "allCompleted": all(item["status"] == "completed" for item in controls),
        "noManifestTemps": all(not item["manifestTempFiles"] for item in controls),
    }
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "configuration": {"query": QUERY, "fps": 1.0, "topK": 5, "device": "cuda", "workerCount": 1, "cache": "disabled", "vlm": False},
        "checks": checks,
        "passed": all(checks.values()),
        "controls": controls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    if managed_work and report["passed"]:
        from scripts.manage_dev_workspace import finalize_run_directory
        finalize_run_directory(work, succeeded=True)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
