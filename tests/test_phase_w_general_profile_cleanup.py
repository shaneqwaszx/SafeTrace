from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from scripts import manage_dev_workspace
from src.api.batches import BatchStore
from src.api.jobs import AnalysisSettings, JobStore
from src.api.server import create_app
from src.processing_contracts import compare_processing_contract, require_baseline_update_reason
from src.profile_composition import (
    GENERAL_PROFILE_COMPONENTS,
    GENERAL_PROFILE_EXCLUSIONS,
    general_profile_registry_payload,
)
from src.rule_engine import evaluate
from src.scene_applicability import apply_scene_applicability_gate
from src.schemas import Detection, Violation


def detection(label: str) -> Detection:
    return Detection(
        label=label,
        raw_label=label,
        confidence=0.9,
        bbox=[0, 0, 20, 20],
        coarse_mask=np.ones((20, 20), dtype=bool),
    )


def test_general_registry_includes_supported_profiles_and_excludes_metadata_only():
    included = {item["profileId"] for item in GENERAL_PROFILE_COMPONENTS}
    excluded = {item["profileId"]: item["reason"] for item in GENERAL_PROFILE_EXCLUSIONS}
    assert {"seatbelt_compliance", "phone_use", "helmet_ppe", "hands_on_wheel"} <= included
    assert excluded["uniform_compliance"].startswith("metadata_only")
    assert excluded["custom_policy"].startswith("metadata_only")
    payload = general_profile_registry_payload()
    assert payload["compositionPolicy"].startswith("union_of_supported")


def test_general_profile_keeps_eligible_seatbelt_cue_with_origin_and_review_level():
    detections = [detection("person"), detection("suitcase"), detection("laptop")]
    accepted, suppressed, _scene = apply_scene_applicability_gate(
        detections, evaluate(detections), profile_id="general_safety"
    )
    assert [item.name for item in accepted] == ["seatbelt_missing"]
    assert {item["name"] for item in suppressed} == {"helmet_missing"}
    evidence = accepted[0].evidence
    assert evidence["originProfileId"] == "seatbelt_compliance"
    assert evidence["originProfileLabel"] == "Seatbelt compliance"
    assert evidence["originRule"] == "rule_seatbelt_missing"
    assert evidence["reviewLevel"] == "review_candidate"
    assert evidence["deduplicationKey"] == "seatbelt_compliance:seatbelt_missing"
    assert accepted[0].confidence <= 0.49


def test_general_profile_deduplicates_overlap_but_keeps_distinct_findings():
    candidates = [
        Violation("seatbelt_missing", "one", confidence=0.42, evidence={"reviewRequired": True}),
        Violation("seatbelt_missing", "duplicate", confidence=0.40, evidence={"reviewRequired": True}),
        Violation("helmet_missing", "distinct", confidence=0.45, evidence={"directSupportLabels": ["head"]}),
    ]
    detections = [detection("person"), detection("head")]
    accepted, suppressed, _scene = apply_scene_applicability_gate(
        detections, candidates, profile_id="general_safety"
    )
    assert [item.name for item in accepted].count("seatbelt_missing") == 1
    assert "helmet_missing" in [item.name for item in accepted]
    assert "duplicate_general_profile_candidate" in {item["reason"] for item in suppressed}


def test_general_profile_preserves_exterior_negative_control():
    detections = [detection("road"), detection("car"), detection("tree")]
    accepted, _suppressed, scene = apply_scene_applicability_gate(
        detections, evaluate(detections), profile_id="general_safety"
    )
    assert accepted == []
    assert scene.exterior_only is True


def _batch(tmp_path: Path):
    jobs = JobStore(tmp_path / "jobs")
    batches = BatchStore(tmp_path / "batches")
    record = batches.create_from_files(
        files=(("first.mp4", b"first"), ("second.mp4", b"second")),
        source_filename="batch",
        query="driver or occupant without seatbelt",
        settings=AnalysisSettings(fps=1.0, top_k=1, enable_vlm=False, device="cpu"),
        job_store=jobs,
    )
    return jobs, batches, record


def test_batch_throughput_separates_processing_cache_queue_and_total(tmp_path):
    jobs, batches, batch = _batch(tmp_path)
    first, second = [jobs.require(job_id) for job_id in batch.job_ids]
    for record in (first, second):
        record.status = "completed"
    first.started_at = first.created_at + timedelta(seconds=10)
    first.finished_at = first.created_at + timedelta(seconds=30)
    first.metrics["resultCacheHit"] = False
    second.started_at = second.created_at + timedelta(seconds=5)
    second.finished_at = second.created_at + timedelta(seconds=6)
    second.metrics["resultCacheHit"] = True
    batch.status = "completed"

    metrics = batches._throughput(batch, jobs)

    assert metrics["meanChildRuntimeSeconds"] == 20.0
    assert metrics["longestChildRuntimeSeconds"] == 20.0
    assert metrics["meanCacheHitMaterializationSeconds"] == 1.0
    assert metrics["meanChildQueueWaitSeconds"] == 7.5
    assert metrics["meanChildTotalElapsedSeconds"] == 18.0
    assert metrics["cacheHitCount"] == 1
    assert metrics["cacheMissCount"] == 1
    assert "processing time for completed cache misses" in metrics["metricDefinitions"]["meanChildRuntimeSeconds"]


def test_completed_child_result_is_available_while_parent_batch_is_active(tmp_path):
    jobs, batches, batch = _batch(tmp_path)
    completed_id, running_id = batch.job_ids
    jobs.complete_job(completed_id, {
        "status": "completed",
        "media": {"id": "first", "name": "first.mp4", "type": "video", "sizeBytes": 5},
        "query": "driver or occupant without seatbelt",
        "summary": {
            "framesAnalyzed": 0,
            "framesWithViolations": 0,
            "uniqueViolationTypes": 0,
            "summaryText": "No violations found. No evidence frames were generated.",
        },
        "violations": [],
        "events": [],
        "frames": [],
        "evidence": [],
    })
    jobs.update_status(running_id, status="running", progress=0.3, current_step="Detector running")
    with TestClient(create_app(jobs, batches)) as client:
        parent = client.get(f"/api/batches/{batch.batch_id}")
        child_status = client.get(f"/api/jobs/{completed_id}")
        child_result = client.get(f"/api/jobs/{completed_id}/result")
        batch_results = client.get(f"/api/batches/{batch.batch_id}/results")
    assert parent.status_code == 200
    assert parent.json()["status"] not in {"completed", "failed", "cancelled"}
    assert child_status.status_code == 200
    assert child_result.status_code == 200
    assert completed_id in batch_results.json()["resultsByJobId"]


def test_frontend_preserves_selected_child_while_parent_polling():
    app = (Path(__file__).resolve().parents[1] / "frontend-react/src/App.tsx").read_text(encoding="utf-8")
    assert "analysisResult && (!isLoading || Boolean(selectedBatchJobId))" in app
    assert "Batch: {completed} of {total} completed - processing continues." in app
    assert "Back to active batch" in app
    assert "selectedBatchJobIdRef.current !== jobId" in app
    assert "pollBackendBatch" in app


def test_processing_contract_detects_unapproved_change_and_requires_update_reason():
    contract = {
        "contractVersion": "test",
        "controls": [{
            "controlId": "positive",
            "profile": "general_safety",
            "invariants": {"sampledFrameCount": 5},
            "behavior": {"requiredFindings": ["seatbelt_missing"], "profileApplicable": True},
        }],
    }
    passing = [{"controlId": "positive", "profile": "general_safety", "sampledFrameCount": 5, "acceptedFindingNames": ["seatbelt_missing"], "profileApplicable": True}]
    assert compare_processing_contract(contract, passing)["passed"] is True
    changed = [{**passing[0], "sampledFrameCount": 4}]
    assert compare_processing_contract(contract, changed)["passed"] is False
    try:
        require_baseline_update_reason("")
    except ValueError:
        pass
    else:
        raise AssertionError("empty baseline reason was accepted")


def test_dev_work_root_is_marked_and_successful_run_is_removed(monkeypatch):
    relative = ".ai-pipeline/006_work/pytest_phase_w_unit"
    monkeypatch.setenv("SAFETRACE_DEV_WORK_ROOT", relative)
    root = manage_dev_workspace.initialize_work_root()
    run = manage_dev_workspace.create_run_directory("pytest", "unit")
    assert (root / manage_dev_workspace.MARKER).is_file()
    assert (run / manage_dev_workspace.MARKER).is_file()
    manage_dev_workspace.finalize_run_directory(run, succeeded=True)
    assert not run.exists()


def test_dev_work_root_rejects_path_traversal():
    try:
        manage_dev_workspace.initialize_work_root(manage_dev_workspace.ROOT.parent / "outside")
    except ValueError:
        pass
    else:
        raise AssertionError("outside work root was accepted")


def test_cleanup_apply_requires_hashed_approval_manifest():
    source = (Path(__file__).resolve().parents[1] / "scripts/manage_dev_workspace.py").read_text(encoding="utf-8")
    assert "cleanup --apply requires --approval-sha256" in source
    assert "Approval manifest SHA-256 mismatch" in source
    assert "for approved in candidates" in source
