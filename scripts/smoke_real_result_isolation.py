"""Real CUDA/concurrent result-ownership smoke for Phase S.

The script extends the Phase R real scheduler exercise, then independently
audits every result boundary, cache reuse, export, deletion, and recovery.
It only writes under the selected report directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _foreign_items(value: Any, expected_job_id: str, location: str = "result") -> list[str]:
    issues: list[str] = []
    if isinstance(value, Mapping):
        owner = value.get("jobId")
        if owner not in (None, "", expected_job_id):
            issues.append(f"{location}.jobId={owner!r}")
        evidence_id = str(value.get("evidenceId") or "")
        if evidence_id and not evidence_id.startswith(f"{expected_job_id}:"):
            issues.append(f"{location}.evidenceId={evidence_id!r}")
        artifact_id = str(value.get("mediaArtifactId") or "")
        if artifact_id and not artifact_id.startswith(f"{expected_job_id}:"):
            issues.append(f"{location}.mediaArtifactId={artifact_id!r}")
        image_url = str(value.get("imageUrl") or "")
        if "/api/media/" in image_url and f"/api/media/{expected_job_id}/" not in image_url:
            issues.append(f"{location}.imageUrl={image_url!r}")
        for key, item in value.items():
            issues.extend(_foreign_items(item, expected_job_id, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(_foreign_items(item, expected_job_id, f"{location}[{index}]"))
    return issues


def _ownership_row(client, store, job_id: str) -> dict[str, Any]:
    from src.api.jobs import result_ownership_context
    from src.api.result_ownership import ResultOwnershipError, validate_result_ownership

    record = store.require(job_id)
    result_response = client.get(f"/api/jobs/{job_id}/result")
    technical_response = client.get(f"/api/reports/{job_id}/technical-json")
    result = result_response.json() if result_response.status_code == 200 else {}
    technical = technical_response.json() if technical_response.status_code == 200 else {}
    issues = _foreign_items(result, job_id) + _foreign_items(technical, job_id, "technical")
    try:
        validate_result_ownership(record.result or {}, result_ownership_context(record))
    except ResultOwnershipError as exc:
        issues.extend(exc.issues)
    frames = list(result.get("frames") or [])
    findings = list(result.get("violations") or [])
    events = list(result.get("events") or [])
    media_checks = []
    for frame in frames:
        url = str(frame.get("imageUrl") or "")
        if not url:
            continue
        response = client.get(url)
        media_checks.append({
            "url": url,
            "statusCode": response.status_code,
            "artifactId": frame.get("mediaArtifactId"),
        })
        if response.status_code != 200:
            issues.append(f"media_unavailable:{url}:{response.status_code}")
    return {
        "jobId": job_id,
        "batchId": record.source_metadata.get("batchId"),
        "expectedSourceFilename": record.original_filename,
        "expectedSourceRelativePath": record.source_metadata.get("sourceRelativePath"),
        "topLevelResultOwner": result.get("jobId"),
        "findingOwners": sorted({item.get("jobId") for item in findings}),
        "eventOwners": sorted({item.get("jobId") for item in events}),
        "evidenceOwners": sorted({item.get("jobId") for item in frames}),
        "evidenceIds": [item.get("evidenceId") for item in frames],
        "mediaArtifactOwners": [item.get("mediaArtifactId") for item in frames if item.get("mediaArtifactId")],
        "technicalJsonOwner": technical.get("jobId"),
        "resultStatusCode": result_response.status_code,
        "technicalStatusCode": technical_response.status_code,
        "mediaChecks": media_checks,
        "foreignItems": sorted(set(str(item) for item in issues)),
        "foreignItemCount": len(set(str(item) for item in issues)),
    }


def run(output: Path) -> dict[str, Any]:
    from fastapi.testclient import TestClient
    from scripts import smoke_real_parallel_runtime as phase_r
    from src.api.batches import BatchStore
    from src.api.exports import verify_export
    from src.api.jobs import JobStore
    from src.api.server import create_app

    phase_r._configure_local_full()  # noqa: SLF001 - shared strict real-runtime fixture
    output.parent.mkdir(parents=True, exist_ok=True)
    phase_r_output = output.with_name("phase_s_scheduler_runtime.json")
    scheduler = phase_r.run_smoke(phase_r_output, include_recovery=True)
    work_dir = Path(scheduler["workDir"])
    jobs = JobStore(work_dir / "jobs")
    batches = BatchStore(work_dir / "batches")
    report: dict[str, Any] = {
        "generatedAt": _utc(),
        "runtimeProfile": "local_full",
        "schedulerSmoke": scheduler,
        "workDir": str(work_dir),
        "frontendDelayedResponseProof": "tests/test_phase_s_frontend_isolation.py",
    }

    with TestClient(create_app(jobs, batches)) as client:
        original_job_id = str(scheduler["singleRealVideo"]["jobId"])
        batch_job_ids = list(scheduler["batchAndFairness"]["batch"]["jobIds"])
        later_job_id = str(scheduler["batchAndFairness"]["batchResult"]["laterJob"]["jobId"])
        all_job_ids = [original_job_id, *batch_job_ids, later_job_id]
        matrix = [_ownership_row(client, jobs, job_id) for job_id in all_job_ids]

        input_path = Path(scheduler["inputs"][0]["name"])
        if not input_path.is_absolute():
            input_path = work_dir / "inputs" / input_path
        cache_job_id = phase_r._post_single(client, input_path, query="general safety review")  # noqa: SLF001
        cache_status, cache_history = phase_r._poll_job(client, cache_job_id)  # noqa: SLF001
        cache_row = _ownership_row(client, jobs, cache_job_id)
        matrix.append(cache_row)
        cache_record = jobs.require(cache_job_id)
        cache_probe = {
            "jobId": cache_job_id,
            "status": cache_status.get("status"),
            "history": cache_history,
            "cacheHit": cache_record.metrics.get("resultCacheHit"),
            "sourceJobId": ((cache_record.result or {}).get("technicalDetails") or {}).get("resultCache", {}).get("sourceJobId"),
            "foreignItemCount": cache_row["foreignItemCount"],
        }

        export_job_id = str(batch_job_ids[0])
        export_response = client.post(f"/api/jobs/{export_job_id}/export", json={})
        export_payload = export_response.json() if export_response.status_code == 200 else {}
        verification = verify_export(Path(export_payload["path"])) if export_payload.get("path") else {"verified": False}
        export_owner = (verification.get("manifest") or {}).get("jobId")
        report["export"] = {
            "jobId": export_job_id,
            "statusCode": export_response.status_code,
            "verified": verification.get("verified"),
            "exportOwner": export_owner,
            "foreignItemCount": 0 if export_owner == export_job_id else 1,
        }

        delete_response = client.delete(f"/api/jobs/{original_job_id}")
        cache_after_delete = client.get(f"/api/jobs/{cache_job_id}/result")
        report["deletion"] = {
            "deletedJobId": original_job_id,
            "statusCode": delete_response.status_code,
            "deletedOwnerUnavailable": client.get(f"/api/jobs/{original_job_id}").status_code == 404,
            "cacheReuseJobStillAvailable": cache_after_delete.status_code == 200,
            "cacheReuseOwner": cache_after_delete.json().get("jobId") if cache_after_delete.status_code == 200 else None,
        }
        report["ownershipMatrix"] = matrix
        report["cacheReuse"] = cache_probe
        report["manifestTemps"] = [str(path) for path in work_dir.rglob("manifest.json.*.tmp")]

    recovery_root = work_dir / "recovery_jobs"
    recovery_store = JobStore(recovery_root)
    recovery_rows = []
    for record in recovery_store.records(include_disk=True):
        if record.status != "completed" or record.result is None:
            continue
        issues = _foreign_items(record.result, record.job_id)
        recovery_rows.append({
            "jobId": record.job_id,
            "sourceRelativePath": record.source_metadata.get("sourceRelativePath"),
            "foreignItems": issues,
            "foreignItemCount": len(issues),
        })
    report["recoveryOwnershipMatrix"] = recovery_rows

    all_rows = report["ownershipMatrix"] + recovery_rows
    report["acceptance"] = {
        "phaseRSchedulerAndRecoveryPassed": bool(scheduler.get("acceptance", {}).get("passed")),
        "allCanonicalOwnersValid": all(row.get("foreignItemCount") == 0 for row in all_rows),
        "cacheReuseWasExactAndIsolated": bool(
            report["cacheReuse"].get("cacheHit")
            and report["cacheReuse"].get("sourceJobId") == original_job_id
            and report["cacheReuse"].get("foreignItemCount") == 0
        ),
        "exportIsolated": report["export"]["verified"] is True and report["export"]["foreignItemCount"] == 0,
        "deletePreservedCacheReuse": bool(
            report["deletion"]["statusCode"] == 200
            and report["deletion"]["deletedOwnerUnavailable"]
            and report["deletion"]["cacheReuseJobStillAvailable"]
            and report["deletion"]["cacheReuseOwner"] == cache_job_id
        ),
        "recoveryHasTwoOwnedResults": len(recovery_rows) == 2 and all(row["foreignItemCount"] == 0 for row in recovery_rows),
        "noManifestTemps": not report["manifestTemps"],
    }
    report["acceptance"]["passed"] = all(report["acceptance"].values())
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real concurrent SafeTrace result isolation checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".ai-pipeline" / "005_tasks" / "phase_s_result_isolation" / "real_result_isolation_smoke.json",
    )
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report["acceptance"], indent=2))
    return 0 if report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
