from __future__ import annotations

from pathlib import Path

import numpy as np

from src.api.jobs import AnalysisSettings, JobStore, build_analysis_setup_payload
from src.api.normalization import normalize_pipeline_results
from src.rule_engine import evaluate
from src.scene_applicability import apply_scene_applicability_gate
from src.schemas import Detection


def detection(label: str) -> Detection:
    return Detection(
        label=label,
        raw_label=label,
        confidence=0.9,
        bbox=[0, 0, 20, 20],
        coarse_mask=np.ones((20, 20), dtype=bool),
    )


def normalize(tmp_path: Path, rows):
    return normalize_pipeline_results(
        job_id="job_phase_v",
        media_name="positive.mp4",
        media_type="video",
        media_size_bytes=1,
        query="driver or occupant without seatbelt",
        raw_frames=rows,
        media_dir=tmp_path / "media",
        register_media=lambda _filename, _path: None,
    )


def test_person_without_custom_cabin_classes_is_applicable_manual_review():
    candidates = evaluate([detection("person"), detection("suitcase"), detection("laptop")])
    accepted, suppressed, scene = apply_scene_applicability_gate(
        [detection("person"), detection("suitcase"), detection("laptop")],
        candidates,
        profile_id="seatbelt_compliance",
    )
    assert scene.person_visible is True
    assert scene.vehicle_interior_visible is True
    assert scene.exterior_only is False
    assert scene.profile_applicable is True
    assert scene.manual_review_required is True
    assert [item.name for item in accepted] == ["seatbelt_missing"]
    assert {item["name"] for item in suppressed} == {"helmet_missing"}
    assert accepted[0].confidence <= 0.49
    assert accepted[0].evidence["candidateViolation"] is True
    assert accepted[0].evidence["violationAccepted"] is True
    assert accepted[0].evidence["manualReviewRequired"] is True


def test_exterior_no_human_remains_inapplicable():
    candidates = evaluate([detection("tree"), detection("road"), detection("car")])
    accepted, _suppressed, scene = apply_scene_applicability_gate(
        [detection("tree"), detection("road"), detection("car")],
        candidates,
    )
    assert accepted == []
    assert scene.profile_applicable is False
    assert scene.exterior_only is True


def test_general_safety_reuses_seatbelt_applicability_without_promoting_review_cue():
    candidates = evaluate([detection("person")])
    accepted, suppressed, _scene = apply_scene_applicability_gate(
        [detection("person")], candidates, profile_id="general_safety"
    )
    assert [item.name for item in accepted] == ["seatbelt_missing"]
    assert {item["name"] for item in suppressed} == {"helmet_missing"}
    seatbelt = accepted[0]
    assert seatbelt.confidence <= 0.49
    assert seatbelt.evidence["evidenceStrength"] == "review_candidate"
    assert seatbelt.evidence["originProfileId"] == "seatbelt_compliance"
    assert seatbelt.evidence["profileApplicability"]["evaluatedAsProfileId"] == "seatbelt_compliance"


def test_findings_survive_when_annotation_evidence_is_unavailable(tmp_path):
    result = normalize(
        tmp_path,
        [{
            "frame_id": "frame_1",
            "timestamp_seconds": 1.0,
            "violations": [{
                "name": "seatbelt_missing",
                "confidence": 0.42,
                "evidence": {"evidenceStrength": "review_candidate", "reviewRequired": True},
            }],
            "annotated_path": str(tmp_path / "missing.jpg"),
        }],
    )
    assert len(result["violations"]) == 1
    assert len(result["events"]) == 1
    assert len(result["frames"]) == 1
    assert result["evidence"] == []
    assert result["evidenceStatus"] == "unavailable"
    assert result["summary"]["violationsDetected"] is True
    assert result["summary"]["acceptedFindingCount"] == 1
    assert "No violations found" not in result["summary"]["summaryText"]


def test_zero_findings_yield_empty_events_evidence_and_exact_empty_summary(tmp_path):
    result = normalize(
        tmp_path,
        [{"frame_id": "frame_1", "timestamp_seconds": 1.0, "violations": []}],
    )
    assert result["violations"] == []
    assert result["events"] == []
    assert result["frames"] == []
    assert result["evidence"] == []
    assert result["evidenceStatus"] == "not_generated"
    assert result["summary"]["violationsDetected"] is False
    assert result["summary"]["acceptedFindingCount"] == 0
    assert result["summary"]["summaryText"] == "No violations found. No evidence frames were generated."


def test_valid_annotation_produces_available_evidence(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    result = normalize(
        tmp_path,
        [{
            "frame_id": "frame_1",
            "violations": [{"name": "seatbelt_missing", "confidence": 0.42}],
            "annotated_path": str(image),
        }],
    )
    assert len(result["violations"]) == 1
    assert len(result["evidence"]) == 1
    assert result["evidenceStatus"] == "available"


def test_analysis_setup_is_canonical_and_survives_job_status(tmp_path):
    store = JobStore(tmp_path / "jobs")
    record = store.create_job(
        filename="clip.mp4",
        content=b"clip",
        query="driver or occupant without seatbelt",
        settings=AnalysisSettings(
            fps=2.0,
            top_k=5,
            enable_vlm=False,
            device="cpu",
            use_case_profile={"profileId": "seatbelt_compliance", "label": "Seatbelt Compliance"},
            review_mode="fast_local",
        ),
    )
    setup = build_analysis_setup_payload(record)
    assert setup["requestedCoverage"]["label"] == "Fast Local Analysis"
    assert setup["profile"]["label"] == "Seatbelt Compliance"
    assert setup["query"] == "driver or occupant without seatbelt"
    assert setup["frameSampling"]["requestedFps"] == 2.0
    assert setup["frameSampling"]["requestedEvidenceFrames"] == 5
    assert record.status_payload()["analysisSetup"] == setup


def test_frontend_distinguishes_no_findings_from_unavailable_evidence():
    root = Path(__file__).resolve().parents[1] / "frontend-react" / "src"
    evidence_source = (root / "components" / "EvidenceFrames.tsx").read_text(encoding="utf-8")
    setup_source = (root / "components" / "AnalysisSetupSummary.tsx").read_text(encoding="utf-8")
    app_source = (root / "App.tsx").read_text(encoding="utf-8")
    assert "acceptedFindingCount > 0 && evidenceStatus === 'unavailable'" in evidence_source
    assert "Safety findings detected" in evidence_source
    assert "acceptedFindingCount === 0" in evidence_source
    assert "Analysis setup" in setup_source
    for label in ("Review coverage", "Profile", "Query", "Frame sampling"):
        assert label in setup_source
    assert "analysisResult?.analysisSetup ?? batchStatus?.analysisSetup ?? jobStatus?.analysisSetup" in app_source
