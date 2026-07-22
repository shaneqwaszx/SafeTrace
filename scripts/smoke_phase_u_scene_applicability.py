"""Run real detector/MobileSAM scene-gate checks on exterior and interior videos."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _run(source: Path, *, label: str, work_dir: Path) -> dict[str, Any]:
    from src.api.jobs import AnalysisSettings, JobStore, execute_analysis_job

    store = JobStore(work_dir / label / "jobs")
    record = store.create_job(
        filename=source.name,
        content=source.read_bytes(),
        query="driver seatbelt compliance",
        settings=AnalysisSettings(
            fps=1.0,
            top_k=5,
            enable_vlm=False,
            device="cuda",
            vlm_profile="rule_based",
            vlm_enabled=False,
            safe_mode=True,
            use_case_profile={"profileId": "seatbelt_compliance", "label": "Seatbelt Compliance"},
            review_mode="fast_local",
        ),
        source_metadata={
            "sourceRelativePath": f"phase-u/{label}/{source.name}",
            "sourceGroupPath": f"phase-u/{label}",
        },
    )
    execute_analysis_job(store, record.job_id)
    completed = store.require(record.job_id)
    result = completed.result or {}
    technical = dict(result.get("technicalDetails") or {})
    diagnostics = dict(technical.get("componentDiagnostics") or {})
    scene = dict(diagnostics.get("sceneApplicability") or {})
    return {
        "label": label,
        "source": str(source),
        "jobId": completed.job_id,
        "status": completed.status,
        "summary": result.get("summary"),
        "acceptedViolationNames": [item.get("name") for item in result.get("violations") or []],
        "evidenceCount": len(result.get("evidence") or []),
        "frameCount": len(result.get("frames") or []),
        "diagnosticFrameCount": len(result.get("diagnosticFrames") or []),
        "sceneApplicability": scene,
        "suppressedFindings": diagnostics.get("sceneSuppressedFindings") or [],
        "mobileSam": {
            "attempted": diagnostics.get("mobileSamWorkerAttempted") or diagnostics.get("mobileSamAttempted"),
            "succeeded": diagnostics.get("mobileSamWorkerSucceeded") or diagnostics.get("mobileSamLoaded"),
            "fallbackReason": diagnostics.get("mobileSamFallbackReason"),
            "device": diagnostics.get("actualMobileSamDevice"),
        },
        "engineMetricsPresent": bool(result.get("engineMetrics")),
        "manifestTempFiles": [str(path) for path in store.root_dir.rglob("manifest.json.*.tmp")],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exterior", type=Path, required=True)
    parser.add_argument("--interior", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["SAFETRACE_RUNTIME_PROFILE"] = "local_full"
    os.environ["SAFETRACE_DEVICE"] = "cuda"
    os.environ["SAFETRACE_ANALYSIS_SAFE_MODE"] = "true"
    os.environ["SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM"] = "true"
    os.environ["SAFETRACE_MOBILESAM_ENABLED"] = "auto"
    os.environ["SAFETRACE_MOBILESAM_WORKER_ENABLED"] = "true"
    os.environ["SAFETRACE_ENABLE_VLM"] = "false"
    os.environ["SAFETRACE_MID_VLM_ENABLED"] = "false"

    args.work_dir.mkdir(parents=True, exist_ok=True)
    exterior = _run(args.exterior.resolve(), label="exterior", work_dir=args.work_dir)
    interior = _run(args.interior.resolve(), label="interior_control", work_dir=args.work_dir)
    exterior_pass = (
        exterior["status"] == "completed"
        and not exterior["acceptedViolationNames"]
        and exterior["evidenceCount"] == 0
        and not exterior["manifestTempFiles"]
    )
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "profile": "local_full",
        "exteriorAcceptancePassed": exterior_pass,
        "interiorControlCompleted": interior["status"] == "completed",
        "results": [exterior, interior],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if exterior_pass and report["interiorControlCompleted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
