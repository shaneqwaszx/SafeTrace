from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from src.api.adaptive_workers import AdaptiveWorkerManager, ResourceSnapshot
from src.api.normalization import normalize_pipeline_results
from src.config import SETTINGS
from src.rule_engine import evaluate
from src.scene_applicability import (
    apply_mid_vlm_verification,
    apply_scene_applicability_gate,
    assess_scene,
    mid_vlm_scene_prompt,
    parse_mid_vlm_scene_response,
)
from src.schemas import Detection, Violation


def resource_snapshot(*, gpu_free=9000.0, ram=12000.0, disk=100000.0):
    return ResourceSnapshot(
        captured_at=1.0,
        gpu_available=True,
        gpu_free_mb=gpu_free,
        gpu_total_mb=12282.0,
        gpu_allocated_mb=500.0,
        gpu_reserved_mb=800.0,
        system_available_mb=ram,
        process_rss_mb=1500.0,
        free_disk_mb=disk,
    )


def configure_adaptive(monkeypatch):
    monkeypatch.setattr(SETTINGS, "adaptive_workers_enabled", True)
    monkeypatch.setattr(SETTINGS, "adaptive_worker_min", 2)
    monkeypatch.setattr(SETTINGS, "adaptive_worker_max", 3)
    monkeypatch.setattr(SETTINGS, "adaptive_third_worker_validated", True)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_up_queue_depth", 3)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_up_sustain_seconds", 15.0)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_down_idle_seconds", 60.0)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_cooldown_seconds", 60.0)
    monkeypatch.setattr(SETTINGS, "adaptive_gpu_free_memory_margin_mb", 4096.0)
    monkeypatch.setattr(SETTINGS, "adaptive_system_free_memory_margin_mb", 4096.0)
    monkeypatch.setattr(SETTINGS, "adaptive_max_recent_error_rate", 0.2)


def test_adaptive_workers_start_at_two_and_admit_third_after_sustained_safe_demand(monkeypatch):
    configure_adaptive(monkeypatch)
    now = [100.0]
    manager = AdaptiveWorkerManager(resource_probe=resource_snapshot, clock=lambda: now[0])
    assert manager.payload()["currentCapacity"] == 2
    assert manager.evaluate(active_count=2, queued_count=3, compatible_queued_count=1, active_third_slot=False) == 2
    now[0] += 16
    assert manager.evaluate(active_count=2, queued_count=3, compatible_queued_count=1, active_third_slot=False) == 3
    payload = manager.payload()
    assert payload["admissionReason"] == "third_worker_admitted"
    assert payload["stageCaps"]["mobileSam"] == 1
    assert payload["stageCaps"]["vlm"] == 1


@pytest.mark.parametrize(
    ("snapshot", "reason"),
    [
        (resource_snapshot(gpu_free=1000), "blocked_by_gpu_memory"),
        (resource_snapshot(ram=1000), "blocked_by_system_memory"),
        (resource_snapshot(disk=100), "blocked_by_disk"),
    ],
)
def test_third_worker_refused_under_resource_pressure(monkeypatch, snapshot, reason):
    configure_adaptive(monkeypatch)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_up_sustain_seconds", 0.0)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_cooldown_seconds", 0.0)
    manager = AdaptiveWorkerManager(resource_probe=lambda: snapshot, clock=lambda: 100.0)
    assert manager.evaluate(active_count=2, queued_count=3, compatible_queued_count=1, active_third_slot=False) == 2
    assert manager.payload()["admissionReason"] == reason


def test_third_worker_refused_for_incompatible_configuration(monkeypatch):
    configure_adaptive(monkeypatch)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_up_sustain_seconds", 0.0)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_cooldown_seconds", 0.0)
    manager = AdaptiveWorkerManager(resource_probe=resource_snapshot, clock=lambda: 100.0)
    assert manager.evaluate(active_count=2, queued_count=3, compatible_queued_count=0, active_third_slot=False) == 2
    assert manager.payload()["admissionReason"] == "no_compatible_queued_job"


def test_scale_down_drains_active_third_without_termination(monkeypatch):
    configure_adaptive(monkeypatch)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_up_sustain_seconds", 0.0)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_cooldown_seconds", 0.0)
    now = [100.0]
    manager = AdaptiveWorkerManager(resource_probe=resource_snapshot, clock=lambda: now[0])
    assert manager.evaluate(active_count=2, queued_count=3, compatible_queued_count=1, active_third_slot=False) == 3
    now[0] += 61
    assert manager.evaluate(active_count=3, queued_count=0, compatible_queued_count=0, active_third_slot=True) == 2
    assert manager.payload()["draining"] is True
    now[0] += 1
    assert manager.evaluate(active_count=2, queued_count=0, compatible_queued_count=0, active_third_slot=False) == 2
    assert manager.payload()["currentCapacity"] == 2


def test_third_worker_remains_locked_until_real_device_validation(monkeypatch):
    configure_adaptive(monkeypatch)
    monkeypatch.setattr(SETTINGS, "adaptive_third_worker_validated", False)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_up_sustain_seconds", 0.0)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_cooldown_seconds", 0.0)
    manager = AdaptiveWorkerManager(resource_probe=resource_snapshot, clock=lambda: 100.0)
    assert manager.evaluate(active_count=2, queued_count=4, compatible_queued_count=2, active_third_slot=False) == 2
    assert manager.payload()["admissionReason"] == "third_worker_not_device_validated"


def test_recent_oom_blocks_third_worker(monkeypatch):
    configure_adaptive(monkeypatch)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_up_sustain_seconds", 0.0)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_cooldown_seconds", 0.0)
    manager = AdaptiveWorkerManager(resource_probe=resource_snapshot, clock=lambda: 100.0)
    manager.record_outcome(status="failed", error_type="CudaOutOfMemoryError")
    assert manager.evaluate(active_count=2, queued_count=4, compatible_queued_count=2, active_third_slot=False) == 2
    assert manager.payload()["admissionReason"] == "blocked_by_recent_oom"


def test_scale_down_cooldown_prevents_immediate_readmission(monkeypatch):
    configure_adaptive(monkeypatch)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_up_sustain_seconds", 0.0)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_down_idle_seconds", 1.0)
    monkeypatch.setattr(SETTINGS, "adaptive_scale_cooldown_seconds", 10.0)
    now = [100.0]
    manager = AdaptiveWorkerManager(resource_probe=resource_snapshot, clock=lambda: now[0])
    assert manager.evaluate(active_count=2, queued_count=4, compatible_queued_count=2, active_third_slot=False) == 3
    now[0] += 2
    manager.evaluate(active_count=2, queued_count=0, compatible_queued_count=0, active_third_slot=False)
    assert manager.payload()["currentCapacity"] == 2
    now[0] += 1
    assert manager.evaluate(active_count=2, queued_count=4, compatible_queued_count=2, active_third_slot=False) == 2
    assert manager.payload()["admissionReason"] == "scale_cooldown_active"


def detection(label: str, confidence: float = 0.9) -> Detection:
    mask = np.ones((20, 20), dtype=bool)
    return Detection(label=label, raw_label=label, confidence=confidence, bbox=[0, 0, 20, 20], coarse_mask=mask)


def violation(name: str) -> Violation:
    return Violation(name=name, description=name, confidence=0.7)


def test_exterior_no_human_suppresses_all_in_cabin_candidates():
    detections = [detection("tree"), detection("road"), detection("car")]
    candidates = [violation("seatbelt_missing"), violation("hands_off_steering_wheel"), violation("phone_use")]
    accepted, suppressed, scene = apply_scene_applicability_gate(detections, candidates)
    assert accepted == []
    assert len(suppressed) == 3
    assert scene.person_visible is False
    assert scene.vehicle_interior_visible is False
    assert scene.external_road_scene is True
    assert all(item["reason"] in {"no_applicable_subject_or_scene", "no_driver_or_steering_context", "no_person_and_phone_context"} for item in suppressed)


def test_person_only_seatbelt_candidate_survives_only_as_manual_review():
    candidates = evaluate([detection("person")])
    assert {item.name for item in candidates} == {"helmet_missing", "seatbelt_missing"}
    accepted, suppressed, _scene = apply_scene_applicability_gate(
        [detection("person")], candidates, profile_id="seatbelt_compliance"
    )
    assert [item.name for item in accepted] == ["seatbelt_missing"]
    assert {item["name"] for item in suppressed} == {"helmet_missing"}
    seatbelt = accepted[0]
    assert seatbelt.confidence <= 0.49
    assert seatbelt.evidence["evidenceStrength"] == "review_candidate"
    assert seatbelt.evidence["manualReviewRequired"] is True
    assert seatbelt.evidence["violationAccepted"] is True


def test_interior_person_torso_remains_eligible():
    detections = [detection("person"), detection("torso"), detection("steering_wheel")]
    accepted, suppressed, scene = apply_scene_applicability_gate(detections, [violation("seatbelt_missing")])
    assert [item.name for item in accepted] == ["seatbelt_missing"]
    assert suppressed == []
    assert scene.vehicle_interior_visible is True
    assert scene.driver_region_visible is True


def test_mid_vlm_contract_is_strict_and_non_authoritative():
    prompt = mid_vlm_scene_prompt(finding="seatbelt_missing", profile="seatbelt_compliance", query="driver seatbelt")
    assert "exactly these keys" in prompt
    response = {
        "sceneType": "vehicle_interior",
        "personVisible": True,
        "vehicleInteriorVisible": True,
        "driverRegionVisible": True,
        "profileApplicable": True,
        "violationSupported": False,
        "violationType": "seatbelt_missing",
        "confidence": 0.65,
        "reason": "belt path is unclear",
    }
    assert parse_mid_vlm_scene_response(response)["confidence"] == 0.65
    target = violation("seatbelt_missing")
    keep, reason = apply_mid_vlm_verification(
        target,
        assess_scene([detection("person"), detection("torso"), detection("steering_wheel")]),
        response,
    )
    assert keep is True
    assert reason == "mid_vlm_disagrees_manual_review"
    assert target.evidence["midVlmNonAuthoritative"] is True
    assert target.evidence["reviewRequired"] is True
    with pytest.raises(ValueError, match="mid_vlm_schema_keys"):
        parse_mid_vlm_scene_response({"sceneType": "external"})


def test_normalization_canonical_order_and_numeric_presentation(tmp_path):
    registered = {}
    rows = []
    for frame_id, timestamp, source_index in [("frame_10", 10.0, 100), ("frame_2", 2.0, 20), ("frame_9", 9.0, 90)]:
        image = tmp_path / f"{frame_id}.jpg"
        image.write_bytes(frame_id.encode())
        rows.append({
            "frame_id": frame_id,
            "timestamp_seconds": timestamp,
            "source_frame_index": source_index,
            "violations": [{"name": "seatbelt_missing", "confidence": 0.7}],
            "annotated_path": str(image),
        })
    result = normalize_pipeline_results(
        job_id="job_order",
        media_name="clip.mp4",
        media_type="video",
        media_size_bytes=1,
        query="seatbelt",
        raw_frames=rows,
        media_dir=tmp_path / "media",
        register_media=lambda filename, path: registered.setdefault(filename, Path(path)),
        source_relative_path="Fleet/clip.mp4",
    )
    assert [item["sourceFrameIndex"] for item in result["frames"]] == [20, 90, 100]
    assert [item["frameNumber"] for item in result["frames"]] == [1, 2, 3]
    assert len({item["id"] for item in result["frames"]}) == 3


def test_mid_vlm_defaults_reuse_local_512m_without_enabling_it():
    assert str(SETTINGS.mid_vlm_model_path).replace("\\", "/").endswith("models/vlm/lightweight-512m")
    assert SETTINGS.mid_vlm_concurrency == 1
    assert SETTINGS.mid_vlm_enabled is False


def test_mid_vlm_worker_client_accepts_only_strict_scene_json(monkeypatch, tmp_path):
    import src.mid_vlm_verifier as verifier_module

    model_dir = tmp_path / "lightweight-512m"
    model_dir.mkdir()
    monkeypatch.setattr(verifier_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(verifier_module.SETTINGS, "data_dir", tmp_path / "data")
    monkeypatch.setattr(verifier_module.SETTINGS, "mid_vlm_model_path", model_dir)
    monkeypatch.setattr(verifier_module.SETTINGS, "mid_vlm_enabled", True)
    monkeypatch.setattr(verifier_module.SETTINGS, "mid_vlm_device", "cpu")

    def fake_run(command, **kwargs):  # noqa: ARG001
        output_path = Path(command[command.index("--output-json") + 1])
        output_path.write_text(json.dumps({
            "ok": True,
            "sceneVerification": {
                "sceneType": "vehicle_interior",
                "personVisible": True,
                "vehicleInteriorVisible": True,
                "driverRegionVisible": True,
                "profileApplicable": True,
                "violationSupported": False,
                "violationType": "seatbelt_missing",
                "confidence": 0.61,
                "reason": "belt path is occluded",
            },
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(verifier_module.subprocess, "run", fake_run)
    verifier = verifier_module.MidVlmSceneVerifier()
    response = verifier.verify(
        np.zeros((32, 32, 3), dtype=np.uint8),
        finding="seatbelt_missing",
        profile="seatbelt_compliance",
        query="driver seatbelt",
    )
    assert response and response["reason"] == "belt path is occluded"
    assert verifier.last_diagnostics["accepted"] is True
    assert not list((tmp_path / "data" / "mid_vlm_worker").glob("*"))


def test_mid_vlm_failure_preserves_applicable_baseline(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    class FakeDetector:
        checkpoint = "fake.pt"

        def detect(self, image):  # noqa: ARG002
            return [detection("person"), detection("torso"), detection("steering_wheel")]

    class FailingMidVerifier:
        enabled = True
        model_dir = Path("models/vlm/lightweight-512m")
        device = "cuda"
        last_diagnostics = {"attempted": True, "accepted": False, "reason": "worker_timeout"}

        def verify(self, *args, **kwargs):  # noqa: ARG002
            return None

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "mid_vlm_enabled", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "mid_vlm_max_frames", 2)
    monkeypatch.setattr(pipeline_module.SETTINGS, "data_dir", tmp_path / "data")
    monkeypatch.setattr(pipeline_module, "MidVlmSceneVerifier", FailingMidVerifier)
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda _path: np.zeros((32, 32, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "imwrite_rgb", lambda *_args: None)
    monkeypatch.setattr(pipeline_module, "draw_overlays", lambda image, _detections: image)
    pipeline = pipeline_module.SafeTracePipeline(
        detector=FakeDetector(),
        use_case_profile={"profileId": "seatbelt_compliance"},
        workspace=tmp_path / "workspace",
    )
    result = pipeline.analyze_frame(frame)
    assert "seatbelt_missing" in {item.name for item in result.violations}
    assert pipeline.component_diagnostics["midVlmAttempts"] == 1
    assert pipeline.component_diagnostics["midVlmRecords"][0]["decision"] == "baseline_preserved_after_verifier_failure"
