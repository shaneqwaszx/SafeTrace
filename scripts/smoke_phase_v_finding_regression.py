"""Run real Phase V applicability and finding/evidence consistency checks."""
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
    from fastapi.testclient import TestClient

    from src.api.batches import BatchStore
    from src.api.jobs import AnalysisSettings, JobStore, execute_analysis_job
    from src.api.server import create_app

    store = JobStore(work_dir / label / "jobs")
    batch_store = BatchStore(work_dir / label / "batches")
    record = store.create_job(
        filename=source.name,
        content=source.read_bytes(),
        query="driver or occupant without seatbelt",
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
        source_metadata={"sourceRelativePath": f"phase-v/{label}/{source.name}"},
    )
    execute_analysis_job(store, record.job_id)
    completed = store.require(record.job_id)
    result = completed.result or {}
    technical = dict(result.get("technicalDetails") or {})
    diagnostics = dict(technical.get("componentDiagnostics") or {})
    records = list(diagnostics.get("sceneApplicabilityRecords") or [])
    suppressed = [
        item
        for scene_record in records
        for item in list(scene_record.get("suppressedFindings") or [])
    ]
    with TestClient(create_app(store, batch_store)) as client:
        api_response = client.get(f"/api/jobs/{record.job_id}/result")
        api_result = api_response.json() if api_response.status_code == 200 else {}
        technical_response = client.get(f"/api/reports/{record.job_id}/technical-json")
    violation_count = len(result.get("violations") or [])
    api_violation_count = len(api_result.get("violations") or [])
    evidence_count = len(result.get("evidence") or [])
    semantics_consistent = (
        violation_count == api_violation_count
        and bool(result.get("summary", {}).get("violationsDetected")) == bool(violation_count)
        and (violation_count > 0 or evidence_count == 0)
    )
    return {
        "label": label,
        "source": str(source),
        "jobId": record.job_id,
        "status": completed.status,
        "profile": "seatbelt_compliance",
        "query": record.query,
        "sampledFrames": (technical.get("processingMetadata") or {}).get("sampledFrameCount"),
        "selectedFrameTimestamps": [frame.get("timestampSeconds") for frame in result.get("frames") or []],
        "personVisibleFrames": sum(bool(item.get("scene", {}).get("personVisible")) for item in records),
        "interiorVisibleFrames": sum(bool(item.get("scene", {}).get("vehicleInteriorVisible")) for item in records),
        "applicableFrames": sum(bool(item.get("scene", {}).get("profileApplicable")) for item in records),
        "globallyInapplicable": not any(bool(item.get("scene", {}).get("profileApplicable")) for item in records),
        "preGateCandidates": sum(int(item.get("preGateCandidateCount") or 0) for item in records),
        "postGateCandidates": sum(int(item.get("postGateCandidateCount") or 0) for item in records),
        "acceptedFindings": violation_count,
        "acceptedFindingNames": [item.get("id") for item in result.get("violations") or []],
        "groupedEvents": len(result.get("events") or []),
        "evidenceCount": evidence_count,
        "evidenceStatus": result.get("evidenceStatus"),
        "suppressionReasons": sorted({str(item.get("reason")) for item in suppressed}),
        "normalisedPersistedApiCounts": {
            "persisted": violation_count,
            "api": api_violation_count,
        },
        "semanticsConsistent": semantics_consistent,
        "resultEndpointStatus": api_response.status_code,
        "technicalJsonStatus": technical_response.status_code,
        "mobileSam": {
            "attempted": diagnostics.get("mobileSamWorkerAttempted") or diagnostics.get("mobileSamAttempted"),
            "succeeded": diagnostics.get("mobileSamWorkerSucceeded") or diagnostics.get("mobileSamLoaded"),
            "fallbackReason": diagnostics.get("mobileSamFallbackReason"),
        },
        "manifestTempFiles": [str(path) for path in store.root_dir.rglob("manifest.json.*.tmp")],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", type=Path, default=REPO_ROOT / ".ai-pipeline/video/video_2026-06-18_11-38-42.mp4")
    parser.add_argument("--repeat", type=Path, action="append", default=[])
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

    sources = [("positive_control", args.positive.resolve())]
    sources.extend((f"repeat_{index}", path.resolve()) for index, path in enumerate(args.repeat, start=1))
    missing = [str(path) for _, path in sources if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing smoke input(s): {missing}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    results = [_run(path, label=label, work_dir=args.work_dir) for label, path in sources]
    positive = results[0]
    positive_passed = (
        positive["status"] == "completed"
        and int(positive["sampledFrames"] or 0) > 0
        and positive["personVisibleFrames"] > 0
        and positive["interiorVisibleFrames"] > 0
        and positive["applicableFrames"] > 0
        and not positive["globallyInapplicable"]
        and positive["semanticsConsistent"]
    )
    repeats_passed = all(
        item["status"] == "completed" and item["semanticsConsistent"] and not item["manifestTempFiles"]
        for item in results[1:]
    )
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "runtimeProfile": "local_full",
        "positiveControlPassed": positive_passed,
        "repeatVideosPassed": repeats_passed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if positive_passed and repeats_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
