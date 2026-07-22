import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

import src.api.batches as batches_module
import src.api.jobs as jobs_module
from src.api.batches import BatchStore
import src.api.server as server_module
from src.api.jobs import JobStore
from src.api.normalization import normalize_pipeline_results
from src.api.server import create_app


def make_client(tmp_path):
    app = create_app(JobStore(tmp_path / "jobs"), BatchStore(tmp_path / "batches"))
    return TestClient(app)


def completed_status(client, job_id):
    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    return response.json()


def test_normalizer_preserves_evidence_strength_and_verifier_metadata(tmp_path):
    result = normalize_pipeline_results(
        job_id="job_test",
        media_name="sample.jpg",
        media_type="image",
        media_size_bytes=123,
        query="driver without seatbelt",
        raw_frames=[
            {
                "frame_id": "sample_000001",
                "frame_path": "data/frames/sample.jpg",
                "score": 0.8,
                "detections": [{"label": "person", "confidence": 0.9, "bbox": [1, 2, 3, 4]}],
                "violations": [
                    {
                        "name": "seatbelt_missing",
                        "severity": "medium",
                        "confidence": 0.39,
                        "description": "Possible missing seatbelt based on generic person evidence.",
                        "evidence": {
                            "evidenceStrength": "review_candidate",
                            "confidenceReason": "Base evidence is weak and the visual review disagreed.",
                            "ruleSupport": "person_proxy_only",
                            "reviewRequired": True,
                            "verifierAgreement": "disagrees",
                            "verifierDisagreementReason": "VLM saw a belt-like path.",
                            "verifierConfidenceHint": "medium",
                            "finalReviewerNote": "Visual review may disagree with the rule finding; review original footage.",
                        },
                    }
                ],
                "explanation": "Base finding with visual review disagreement.",
                "explanation_source": "rule_template_plus_lightweight_vlm",
            }
        ],
        media_dir=tmp_path / "media",
        register_media=lambda filename, path: None,  # noqa: ARG005
    )

    violation = result["frames"][0]["violations"][0]
    assert violation["evidenceStrength"] == "review_candidate"
    assert violation["verifierAgreement"] == "disagrees"
    assert violation["verifierDisagreementReason"] == "VLM saw a belt-like path."
    assert violation["reviewRequired"] is True
    assert result["violations"][0]["evidenceStrength"] == "review_candidate"
    assert result["violations"][0]["verifierAgreement"] == "disagrees"
    assert result["events"][0]["supportingFrames"][0]["evidenceStrength"] == "review_candidate"


def test_analyze_completes_with_monkeypatched_pipeline(monkeypatch, tmp_path):
    annotated = tmp_path / "frame_000046_annotated.jpg"
    annotated.write_bytes(b"fake-jpeg-bytes")

    def fake_run_pipeline(**kwargs):
        assert kwargs["query"] == "worker without helmet"
        assert kwargs["fps"] == 1.0
        assert kwargs["top_k"] == 5
        assert kwargs["device"] == "cpu"
        assert kwargs["enable_vlm"] is False
        assert kwargs["upload_path"].exists()
        return [
            {
                "frame_id": "video_20260618_000046",
                "frame_path": "data/frames/video_20260618_000046.jpg",
                "score": 0.059,
                "detections": [{"label": "person", "confidence": 0.97, "bbox": [1, 2, 3, 4]}],
                "violations": [
                    {
                        "name": "helmet_missing",
                        "severity": "high",
                        "confidence": 0.98,
                        "description": "Worker head detected without overlapping helmet.",
                    }
                ],
                "explanation": "A worker is visible without helmet overlap.",
                "annotated_path": str(annotated),
            },
            {
                "frame_id": "video_20260618_000047",
                "frame_path": "data/frames/video_20260618_000047.jpg",
                "score": 0.041,
                "detections": [],
                "violations": [],
                "explanation": None,
                "annotated_path": None,
            },
        ]

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
    client = make_client(tmp_path)

    response = client.post(
        "/api/analyze",
        files={"file": ("sample.jpg", b"tiny image", "image/jpeg")},
        data={
            "query": "worker without helmet",
            "fps": "1.0",
            "topK": "5",
            "enableVlm": "false",
            "device": "cpu",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"

    job_id = body["jobId"]
    status = completed_status(client, job_id)
    assert status["status"] == "completed"
    assert status["progress"] == 1.0
    assert status["progressPercent"] == 100
    assert status["stage"] == "completed"
    assert status["message"] == "Analysis completed"
    assert status["updatedAt"]
    assert status["createdAt"]
    assert status["queuedAt"]
    assert status["startedAt"]
    assert status["finishedAt"]
    assert status["completedAt"]
    assert status["failedAt"] is None
    assert status["cancelledAt"] is None
    assert status["elapsedSeconds"] >= 0
    assert status["queueWaitSeconds"] >= 0
    assert status["analysisRuntimeSeconds"] >= 0
    assert status["heartbeatAt"] is None
    assert status["requestedModeLabel"] == "Fast Local Analysis"
    assert status["explanationOutcomeLabel"] == "Fast Local Analysis"
    assert status["vlmAttempted"] is False
    assert status["actualDeviceLabel"] in {"CPU", "CUDA", "Auto selected CUDA", "auto"}
    assert status["engineRuntimeSummary"]["requestedMode"] == "Fast Local Analysis"
    assert "runtime" in status["engineRuntimeSummary"]
    diagnostics = status["componentDiagnostics"]
    assert diagnostics["safeMode"] is False
    assert diagnostics["embeddingRequested"] is True
    assert diagnostics["vlmEffectiveEnabled"] is False
    assert diagnostics["currentPipelineStage"] == "completed"

    result_response = client.get(f"/api/jobs/{job_id}/result")
    assert result_response.status_code == 200
    result = result_response.json()
    assert result["jobId"] == job_id
    assert result["status"] == "completed"
    assert result["elapsedSeconds"] >= 0
    assert result["queueWaitSeconds"] >= 0
    assert result["analysisRuntimeSeconds"] >= 0
    assert result["completedAt"]
    assert result["media"]["name"] == "sample.jpg"
    assert result["summary"]["framesAnalyzed"] == 2
    assert result["summary"]["framesWithViolations"] == 1
    assert result["violations"][0]["name"] == "Missing Helmet"
    assert result["frames"][0]["imageUrl"].startswith(f"/api/media/{job_id}/")
    assert len(result["frames"]) == 1
    assert result["evidence"] == result["frames"]
    assert result["diagnosticFrames"] == []
    assert result["engineMetrics"]["schemaVersion"] == 1
    assert result["engineMetrics"]["counts"]["analyzedFrames"] == 2
    assert result["engineMetrics"]["counts"]["detections"] == 1
    assert result["technicalDetails"]["engineMetrics"]["correctness"]["status"] == "completed"
    assert result["technicalDetails"]["jobTiming"]["elapsedSeconds"] == result["elapsedSeconds"]

    media_response = client.get(result["frames"][0]["imageUrl"])
    assert media_response.status_code == 200
    assert media_response.content == b"fake-jpeg-bytes"

    report_response = client.get(f"/api/reports/{job_id}/technical-json")
    assert report_response.status_code == 200
    assert report_response.json()["technicalDetails"]["job"]["status"] == "completed"
    assert report_response.json()["technicalDetails"]["jobTiming"]["completedAt"] == result["completedAt"]


def test_job_status_payload_reports_elapsed_fields_for_lifecycle_states(tmp_path):
    settings = jobs_module.AnalysisSettings(fps=1.0, top_k=1, enable_vlm=False, device="cpu")
    created = jobs_module._parse_datetime("2026-07-08T01:00:00+00:00")
    started = jobs_module._parse_datetime("2026-07-08T01:00:12+00:00")
    finished = jobs_module._parse_datetime("2026-07-08T01:01:42+00:00")
    now = jobs_module._parse_datetime("2026-07-08T01:00:30+00:00")

    record = jobs_module.JobRecord(
        job_id="job_timing",
        status="queued",
        progress=0.0,
        current_step="Queued",
        error=None,
        query="test",
        settings=settings,
        original_filename="sample.jpg",
        media_type="image",
        size_bytes=10,
        job_dir=tmp_path,
        upload_path=tmp_path / "upload.jpg",
        output_dir=tmp_path / "media",
        created_at=created,
        updated_at=created,
    )

    queued = record.timing_payload(now=now)
    assert queued["elapsedSeconds"] == 30
    assert queued["queueWaitSeconds"] == 30
    assert queued["analysisRuntimeSeconds"] is None

    record.status = "running"
    record.started_at = started
    running = record.timing_payload(now=now)
    assert running["elapsedSeconds"] == 30
    assert running["queueWaitSeconds"] == 12
    assert running["analysisRuntimeSeconds"] == 18

    record.status = "completed"
    record.finished_at = finished
    completed = record.status_payload()
    assert completed["completedAt"] == finished.isoformat()
    assert completed["elapsedSeconds"] == 102
    assert completed["queueWaitSeconds"] == 12
    assert completed["analysisRuntimeSeconds"] == 90

    record.status = "failed"
    failed = record.status_payload()
    assert failed["failedAt"] == finished.isoformat()
    assert failed["completedAt"] is None

    record.status = "cancelled"
    cancelled = record.status_payload()
    assert cancelled["cancelledAt"] == finished.isoformat()
    assert cancelled["completedAt"] is None


def test_analyze_persists_use_case_profile_metadata(monkeypatch, tmp_path):
    def fake_run_pipeline(**kwargs):  # noqa: ARG001
        return []

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
    client = make_client(tmp_path)
    profile = {
        "profileId": "seatbelt_compliance",
        "label": "Seatbelt compliance",
        "category": "Driving safety",
        "description": "Review visible seatbelt evidence.",
        "defaultQuery": "driver or occupant without seatbelt",
        "backendSupportLevel": "partial",
        "supportedChecks": ["Missing Seatbelt", "driver visible"],
        "unsupportedChecks": ["legal determination"],
        "limitations": "Seatbelt support depends on visible in-cabin evidence.",
        "checks": ["Missing seatbelt", "Driver visible"],
        "customText": "",
        "requestedQuery": "driver without seatbelt",
        "effectiveQuery": "driver or occupant without seatbelt. Reviewer refinement: driver without seatbelt",
    }

    response = client.post(
        "/api/analyze",
        files={"file": ("sample.jpg", b"tiny image", "image/jpeg")},
        data={
            "query": "driver without seatbelt",
            "fps": "1.0",
            "topK": "5",
            "enableVlm": "false",
            "device": "cpu",
            "useCaseProfile": json.dumps(profile),
        },
    )

    assert response.status_code == 200
    job_id = response.json()["jobId"]
    status = completed_status(client, job_id)
    stored_profile = status["componentDiagnostics"]["useCaseProfile"]
    assert stored_profile["profileId"] == "seatbelt_compliance"
    assert stored_profile["label"] == "Seatbelt compliance"
    assert stored_profile["checks"] == ["Missing seatbelt", "Driver visible"]
    assert stored_profile["defaultQuery"] == "driver or occupant without seatbelt"
    assert stored_profile["backendSupportLevel"] == "partial"
    assert stored_profile["supportedChecks"] == ["Missing Seatbelt", "driver visible"]
    assert stored_profile["effectiveQuery"].startswith("driver or occupant without seatbelt")

    result = client.get(f"/api/jobs/{job_id}/result").json()
    result_profile = result["technicalDetails"]["jobMetrics"]["componentDiagnostics"]["useCaseProfile"]
    assert result_profile["profileId"] == "seatbelt_compliance"
    assert result_profile["backendSupportLevel"] == "partial"


def test_delete_completed_job_is_scoped_to_job_id(monkeypatch, tmp_path):
    def fake_run_pipeline(**kwargs):  # noqa: ARG001
        return []

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
    client = make_client(tmp_path)

    first = client.post(
        "/api/analyze",
        files={"file": ("first.jpg", b"tiny image", "image/jpeg")},
        data={"query": "driver without seatbelt"},
    ).json()["jobId"]
    second = client.post(
        "/api/analyze",
        files={"file": ("second.jpg", b"tiny image", "image/jpeg")},
        data={"query": "worker without helmet"},
    ).json()["jobId"]
    completed_status(client, first)
    completed_status(client, second)

    delete_response = client.delete(f"/api/jobs/{first}")
    assert delete_response.status_code == 200
    assert client.get(f"/api/jobs/{first}").status_code == 404
    assert client.get(f"/api/jobs/{second}").status_code == 200


def test_failed_analysis_returns_structured_error(monkeypatch, tmp_path):
    def failing_pipeline(**kwargs):  # noqa: ARG001
        raise RuntimeError("secret checkpoint traceback")

    monkeypatch.setattr(jobs_module, "run_pipeline", failing_pipeline)
    client = make_client(tmp_path)

    response = client.post(
        "/api/analyze",
        files={"file": ("sample.jpg", b"tiny image", "image/jpeg")},
        data={"query": "worker without helmet"},
    )

    assert response.status_code == 200
    job_id = response.json()["jobId"]
    status = completed_status(client, job_id)
    assert status["status"] == "failed"
    assert status["progressPercent"] == 100
    assert status["stage"] == "failed"
    assert status["message"] == "Analysis failed"
    assert status["finishedAt"]
    assert status["heartbeatAt"] is None
    assert "secret checkpoint traceback" not in status["error"]
    assert status["metrics"]["errorType"] == "RuntimeError"
    assert status["metrics"]["errorMessage"] == "Analysis could not be completed. Please try again."
    assert status["componentDiagnostics"]["errorType"] == "RuntimeError"

    result_response = client.get(f"/api/jobs/{job_id}/result")
    assert result_response.status_code == 409
    assert result_response.json()["detail"]["message"] == status["error"]


def test_running_job_heartbeat_updates_message_and_timestamp(tmp_path):
    store = JobStore(tmp_path / "jobs")
    record = store.create_job(
        filename="sample.jpg",
        content=b"tiny image",
        query="worker without helmet",
        settings=jobs_module.AnalysisSettings(fps=1.0, top_k=5, enable_vlm=False, device="cpu"),
    )
    store.update_status(
        record.job_id,
        status="running",
        progress=0.35,
        current_step="Running SafeTrace analysis. This stage may take a few minutes.",
    )
    before = store.require(record.job_id).status_payload()

    assert store.heartbeat(
        record.job_id,
        progress=0.35,
        current_step="Running SafeTrace analysis. Still working locally after 8s; sampling evidence.",
    )

    after = store.require(record.job_id).status_payload()
    assert after["status"] == "running"
    assert after["stage"] == "analyzing"
    assert after["progressPercent"] == 35
    assert "Still working locally" in after["message"]
    assert after["heartbeatAt"]
    assert after["updatedAt"] >= before["updatedAt"]

    store.complete_job(record.job_id, {"technicalDetails": {}})
    assert store.heartbeat(record.job_id, current_step="Should not update") is False


def test_analyze_uses_selected_vlm_profile_from_backend_settings(monkeypatch, tmp_path):
    lightweight = tmp_path / "models" / "vlm" / "lightweight-256m"
    root_vlm = tmp_path / "models" / "vlm"
    lightweight.mkdir(parents=True)
    (lightweight / "model.safetensors").write_bytes(b"placeholder")
    captured = []

    def fake_run_pipeline(**kwargs):
        captured.append(kwargs)
        return []

    monkeypatch.setattr(server_module.SETTINGS, "vlm_profile", "rule_based")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_model_dir", root_vlm)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_model_path", lightweight)
    monkeypatch.setattr(server_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(jobs_module.SETTINGS, "vlm_lightweight_model_path", lightweight)
    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
    client = make_client(tmp_path)

    settings_response = client.post(
        "/api/system/vlm/settings",
        json={"selectedProfile": "lightweight_256m", "enabled": True},
    )
    assert settings_response.status_code == 200

    response = client.post(
        "/api/analyze",
        files={"file": ("sample.jpg", b"tiny image", "image/jpeg")},
        data={
            "query": "worker without helmet",
            "enableVlm": "true",
            "device": "cpu",
        },
    )

    assert response.status_code == 200
    assert captured
    assert captured[0]["enable_vlm"] is True
    assert captured[0]["vlm_profile"] == "lightweight_256m"
    assert captured[0]["vlm_model_dir"] == lightweight
    assert captured[0]["vlm_model_dir"] != root_vlm


def test_analyze_rule_based_does_not_request_vlm_profile_load(monkeypatch, tmp_path):
    captured = []

    def fake_run_pipeline(**kwargs):
        captured.append(kwargs)
        return []

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
    client = make_client(tmp_path)

    response = client.post(
        "/api/analyze",
        files={"file": ("sample.jpg", b"tiny image", "image/jpeg")},
        data={
            "query": "worker without helmet",
            "enableVlm": "true",
            "vlmProfile": "rule_based",
            "vlmEnabled": "true",
            "device": "cpu",
        },
    )

    assert response.status_code == 200
    assert captured
    assert captured[0]["enable_vlm"] is False
    assert captured[0]["vlm_profile"] == "rule_based"
    assert captured[0]["vlm_model_dir"] is None


def test_analyze_visual_explanations_off_does_not_request_vlm(monkeypatch, tmp_path):
    lightweight = tmp_path / "models" / "vlm" / "lightweight-256m"
    lightweight.mkdir(parents=True)
    (lightweight / "model.safetensors").write_bytes(b"placeholder")
    captured = []

    def fake_run_pipeline(**kwargs):
        captured.append(kwargs)
        return []

    monkeypatch.setattr(jobs_module.SETTINGS, "vlm_lightweight_model_path", lightweight)
    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
    client = make_client(tmp_path)

    response = client.post(
        "/api/analyze",
        files={"file": ("sample.jpg", b"tiny image", "image/jpeg")},
        data={
            "query": "worker without helmet",
            "enableVlm": "false",
            "vlmProfile": "lightweight_256m",
            "vlmEnabled": "true",
            "device": "cpu",
        },
    )

    assert response.status_code == 200
    assert captured
    assert captured[0]["enable_vlm"] is False
    assert captured[0]["vlm_profile"] == "lightweight_256m"
    assert captured[0]["vlm_model_dir"] is None


def test_analyze_hard_disabled_vlm_cannot_be_activated(monkeypatch, tmp_path):
    lightweight = tmp_path / "models" / "vlm" / "lightweight-256m"
    lightweight.mkdir(parents=True)
    (lightweight / "model.safetensors").write_bytes(b"placeholder")
    captured = []

    def fake_run_pipeline(**kwargs):
        captured.append(kwargs)
        return []

    monkeypatch.setattr(server_module.SETTINGS, "vlm_profile", "rule_based")
    monkeypatch.setattr(server_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_enabled", "false")
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_model_path", lightweight)
    monkeypatch.setattr(jobs_module.SETTINGS, "vlm_enabled", "false")
    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
    client = make_client(tmp_path)

    settings_response = client.post(
        "/api/system/vlm/settings",
        json={"selectedProfile": "lightweight_256m", "enabled": True},
    )
    assert settings_response.status_code == 200
    assert settings_response.json()["active"] is False

    response = client.post(
        "/api/analyze",
        files={"file": ("sample.jpg", b"tiny image", "image/jpeg")},
        data={
            "query": "worker without helmet",
            "enableVlm": "true",
            "vlmProfile": "lightweight_256m",
            "vlmEnabled": "true",
            "device": "cpu",
        },
    )

    assert response.status_code == 200
    assert captured
    assert captured[0]["enable_vlm"] is False
    assert captured[0]["vlm_profile"] == "lightweight_256m"
    assert captured[0]["vlm_model_dir"] is None


def test_safe_mode_suppresses_direct_vlm_but_respects_requested_device(monkeypatch, tmp_path):
    lightweight = tmp_path / "models" / "vlm" / "lightweight-256m"
    lightweight.mkdir(parents=True)
    (lightweight / "model.safetensors").write_bytes(b"placeholder")
    captured = []

    def fake_run_pipeline(**kwargs):
        captured.append(kwargs)
        diagnostics = kwargs["component_diagnostics"]
        diagnostics.update(
            {
                "safeMode": True,
                "requestedDevice": kwargs["device"],
                "device": kwargs["device"],
                "actualDetectorDevice": kwargs["device"],
                "actualLightweightVlmDevice": kwargs["device"],
                "cudaAvailableAtJobStart": kwargs["device"] == "cuda",
                "embeddingRequested": False,
                "embeddingLoaded": False,
                "vlmAttempted": False,
                "vlmLoaded": False,
                "mobileSamRequested": False,
                "mobileSamLoaded": False,
                "currentPipelineStage": "safe_direct_frame_scan",
            }
        )
        return []

    monkeypatch.setattr(server_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(server_module.SETTINGS, "vlm_lightweight_model_path", lightweight)
    monkeypatch.setattr(jobs_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
    client = make_client(tmp_path)

    settings_response = client.post(
        "/api/system/vlm/settings",
        json={"selectedProfile": "lightweight_256m", "enabled": True},
    )
    assert settings_response.status_code == 200
    settings_body = settings_response.json()
    assert settings_body["enabled"] is False
    assert settings_body["active"] is False
    assert settings_body["actualExplanationMode"] == "rule_based"
    assert settings_body["vlmSuppressedReason"] == "safe_mode"

    response = client.post(
        "/api/analyze",
        files={"file": ("sample.jpg", b"tiny image", "image/jpeg")},
        data={
            "query": "worker without helmet",
            "enableVlm": "true",
            "vlmProfile": "lightweight_256m",
            "vlmEnabled": "true",
            "device": "cuda",
        },
    )

    assert response.status_code == 200
    assert captured
    assert captured[0]["safe_mode"] is True
    assert captured[0]["device"] == "cuda"
    assert captured[0]["enable_vlm"] is False
    assert captured[0]["vlm_model_dir"] is None
    status = completed_status(client, response.json()["jobId"])
    diagnostics = status["componentDiagnostics"]
    assert diagnostics["safeMode"] is True
    assert diagnostics["device"] == "cuda"
    assert diagnostics["actualDetectorDevice"] == "cuda"
    assert status["actualDeviceLabel"] == "CUDA"
    assert diagnostics["embeddingRequested"] is False
    assert diagnostics["vlmLoaded"] is False
    assert diagnostics["mobileSamLoaded"] is False


def test_pipeline_timeout_marks_job_failed_with_component_diagnostics(monkeypatch, tmp_path):
    def slow_pipeline(**kwargs):
        kwargs["component_diagnostics"]["currentPipelineStage"] = "embedding_model_load"
        time.sleep(0.2)
        return []

    monkeypatch.setattr(jobs_module.SETTINGS, "analysis_job_timeout_seconds", 0.01)
    monkeypatch.setattr(jobs_module, "run_pipeline", slow_pipeline)
    client = make_client(tmp_path)

    response = client.post(
        "/api/analyze",
        files={"file": ("sample.jpg", b"tiny image", "image/jpeg")},
        data={"query": "worker without helmet", "device": "cpu"},
    )

    assert response.status_code == 200
    status = completed_status(client, response.json()["jobId"])
    assert status["status"] == "failed"
    assert status["message"] == "Analysis failed"
    assert status["metrics"]["errorType"] == "PipelineTimeoutError"
    assert status["componentDiagnostics"]["errorType"] == "PipelineTimeoutError"
    assert status["componentDiagnostics"]["currentPipelineStage"] == "embedding_model_load"


def test_analyze_rejects_invalid_requests(tmp_path):
    client = make_client(tmp_path)

    empty_query = client.post(
        "/api/analyze",
        files={"file": ("sample.jpg", b"tiny image", "image/jpeg")},
        data={"query": "   "},
    )
    assert empty_query.status_code == 400
    assert empty_query.json()["detail"]["message"] == "Query is required"

    empty_file = client.post(
        "/api/analyze",
        files={"file": ("sample.jpg", b"", "image/jpeg")},
        data={"query": "worker without helmet"},
    )
    assert empty_file.status_code == 400
    assert empty_file.json()["detail"]["message"] == "Uploaded file is empty"


def test_batch_atomic_write_uses_unique_temp_paths_and_retries_permission_error(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    real_replace = batches_module.os.replace
    temp_names: list[str] = []
    attempts = {"count": 0}

    def flaky_replace(source, target):
        attempts["count"] += 1
        temp_names.append(Path(source).name)
        if attempts["count"] == 1:
            raise PermissionError("temporary Windows lock")
        return real_replace(source, target)

    monkeypatch.setattr(batches_module.os, "replace", flaky_replace)

    batches_module._atomic_write_json(manifest, {"status": "first"}, retries=2, backoff_seconds=0)
    batches_module._atomic_write_json(manifest, {"status": "second"}, retries=2, backoff_seconds=0)

    assert json.loads(manifest.read_text(encoding="utf-8")) == {"status": "second"}
    assert len(temp_names) == 3
    assert temp_names[0] == temp_names[1]
    assert temp_names[1] != temp_names[2]
    assert not list(tmp_path.glob("manifest.json.*.tmp"))


def test_job_atomic_write_uses_unique_temp_paths_and_retries_permission_error(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    replace_calls = []

    def flaky_replace(source, destination):
        replace_calls.append((Path(source).name, Path(destination).name))
        if len(replace_calls) == 1:
            raise PermissionError("temporary Windows lock")
        return original_replace(source, destination)

    original_replace = jobs_module.os.replace
    monkeypatch.setattr(jobs_module.os, "replace", flaky_replace)

    jobs_module._atomic_write_json(manifest, {"status": "first"}, retries=2, backoff_seconds=0)
    jobs_module._atomic_write_json(manifest, {"status": "second"}, retries=2, backoff_seconds=0)

    assert json.loads(manifest.read_text(encoding="utf-8")) == {"status": "second"}
    first_write_temp = replace_calls[0][0]
    second_write_temp = replace_calls[2][0]
    assert first_write_temp != "manifest.json.tmp"
    assert first_write_temp != second_write_temp
    assert not list(tmp_path.glob("manifest.json.*.tmp"))


def test_job_atomic_write_preserves_existing_manifest_after_final_replace_failure(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    jobs_module._atomic_write_json(manifest, {"status": "existing"})

    def deny_replace(source, destination):  # noqa: ARG001
        raise PermissionError("locked manifest")

    monkeypatch.setattr(jobs_module.os, "replace", deny_replace)

    try:
        jobs_module._atomic_write_json(manifest, {"status": "new"}, retries=2, backoff_seconds=0)
    except jobs_module.JobManifestPersistenceError:
        pass
    else:  # pragma: no cover - sanity guard
        raise AssertionError("expected manifest persistence error")

    assert json.loads(manifest.read_text(encoding="utf-8")) == {"status": "existing"}
    assert not list(tmp_path.glob("manifest.json.*.tmp"))


def test_job_atomic_write_concurrent_writes_leave_valid_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"

    def write(index):
        jobs_module._atomic_write_json(manifest, {"index": index})

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write, range(12)))

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert isinstance(payload["index"], int)
    assert not list(tmp_path.glob("manifest.json.*.tmp"))


def test_manifest_warning_during_media_registration_does_not_fail_completed_job(monkeypatch, tmp_path):
    annotated = tmp_path / "frame_000046_annotated.jpg"
    annotated.write_bytes(b"fake-jpeg-bytes")
    store = JobStore(tmp_path / "jobs")
    record = store.create_job(
        filename="sample.jpg",
        content=b"tiny image",
        query="driver without seatbelt",
        settings=jobs_module.AnalysisSettings(
            fps=1.0,
            top_k=2,
            enable_vlm=True,
            vlm_profile="lightweight_512m",
            vlm_enabled=True,
            device="cpu",
            safe_mode=True,
        ),
    )

    def fake_run_pipeline(**kwargs):  # noqa: ARG001
        return [
            {
                "frame_id": "video_20260618_000046",
                "score": 0.9,
                "detections": [{"label": "person", "confidence": 0.9, "bbox": [1, 2, 3, 4]}],
                "violations": [
                    {
                        "name": "seatbelt_missing",
                        "severity": "high",
                        "confidence": 0.8,
                        "description": "No clear belt crossing the torso.",
                    }
                ],
                "explanation": "Rule-based explanation: VLM timed out.",
                "explanation_source": "rule_based",
                "annotated_path": str(annotated),
                "search_metadata": {
                    "requestedVisualExplanationMode": "lightweight_512m",
                    "actualExplanationMode": "rule_based",
                    "lightweightVlmWorkerAttempted": True,
                    "lightweightVlmWorkerTimedOut": True,
                    "lightweightVlmFallbackReason": "worker_timeout",
                },
            }
        ]

    original_replace = jobs_module.os.replace
    deny_manifests = {"enabled": True}

    def flaky_manifest_replace(source, destination):
        if deny_manifests["enabled"] and Path(destination).name == "manifest.json":
            raise PermissionError("simulated manifest lock")
        return original_replace(source, destination)

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(jobs_module.os, "replace", flaky_manifest_replace)

    jobs_module.execute_analysis_job(store, record.job_id)

    completed = store.require(record.job_id)
    assert completed.status == "completed"
    assert completed.result is not None
    assert completed.persistence_warning
    assert completed.result["technicalDetails"]["manifestPersistenceWarning"]
    assert completed.result["frames"][0]["imageUrl"].startswith(f"/api/media/{record.job_id}/")
    assert completed.result["frames"][0]["technicalEvidence"]["searchMetadata"]["lightweightVlmFallbackReason"] == "worker_timeout"
    assert not list(completed.job_dir.glob("manifest.json.*.tmp"))


def test_repeated_batch_status_get_does_not_rewrite_unchanged_manifest(monkeypatch, tmp_path):
    def fake_run_pipeline(**kwargs):  # noqa: ARG001
        return []

    persist_calls: list[str] = []
    original_persist = batches_module.BatchStore.persist_batch

    def counting_persist(self, record):
        persist_calls.append(record.batch_id)
        return original_persist(self, record)

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(batches_module.BatchStore, "persist_batch", counting_persist)
    client = make_client(tmp_path)

    response = client.post(
        "/api/batches/analyze",
        files=[
            ("files", ("first.mp4", b"video-one", "video/mp4")),
            ("files", ("second.mp4", b"video-two", "video/mp4")),
        ],
        data={"query": "driver without seatbelt", "enableVlm": "false", "device": "cpu"},
    )

    assert response.status_code == 200
    batch_id = response.json()["batchId"]
    first_status = client.get(f"/api/batches/{batch_id}")
    assert first_status.status_code == 200
    persist_calls.clear()

    for _ in range(3):
        repeated = client.get(f"/api/batches/{batch_id}")
        assert repeated.status_code == 200

    assert persist_calls == []


def test_batch_status_returns_warning_when_manifest_persist_fails_but_existing_manifest_is_valid(monkeypatch, tmp_path):
    def fake_run_pipeline(**kwargs):  # noqa: ARG001
        return []

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
    client = make_client(tmp_path)
    response = client.post(
        "/api/batches/analyze",
        files=[
            ("files", ("first.mp4", b"video-one", "video/mp4")),
            ("files", ("second.mp4", b"video-two", "video/mp4")),
        ],
        data={"query": "driver without seatbelt", "enableVlm": "false", "device": "cpu"},
    )

    assert response.status_code == 200
    batch_id = response.json()["batchId"]
    batch_store = client.app.state.batch_store
    job_store = client.app.state.job_store
    record = batch_store.require(batch_id, job_store)
    assert record.manifest_path.is_file()
    job_store.update_status(
        record.job_ids[0],
        status="failed",
        progress=1.0,
        current_step="Analysis failed",
        error="Synthetic failure",
    )

    def fail_write(path, payload, **kwargs):  # noqa: ARG001
        raise batches_module.BatchManifestPersistenceError("simulated manifest replace denial")

    monkeypatch.setattr(batches_module, "_atomic_write_json", fail_write)

    status = client.get(f"/api/batches/{batch_id}")

    assert status.status_code == 200
    body = status.json()
    assert body["batchId"] == batch_id
    assert body["persistenceWarning"]
    assert "simulated manifest" in body["persistenceWarning"]


def test_repeated_batch_status_polling_stays_200_and_leaves_valid_manifest(monkeypatch, tmp_path):
    def fake_run_pipeline(**kwargs):  # noqa: ARG001
        return []

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)
    client = make_client(tmp_path)
    response = client.post(
        "/api/batches/analyze",
        files=[
            ("files", ("first.mp4", b"video-one", "video/mp4")),
            ("files", ("second.mp4", b"video-two", "video/mp4")),
        ],
        data={"query": "driver without seatbelt", "enableVlm": "false", "device": "cpu"},
    )

    assert response.status_code == 200
    batch_id = response.json()["batchId"]

    def poll_once():
        return client.get(f"/api/batches/{batch_id}").status_code

    with ThreadPoolExecutor(max_workers=4) as executor:
        statuses = list(executor.map(lambda _: poll_once(), range(12)))

    assert statuses == [200] * 12
    record = client.app.state.batch_store.require(batch_id, client.app.state.job_store)
    manifest = json.loads(record.manifest_path.read_text(encoding="utf-8"))
    assert manifest["batchId"] == batch_id
    assert not list(record.batch_dir.glob("manifest.json.*.tmp"))


def test_batch_id_is_not_accepted_as_job_id_for_status(tmp_path):
    client = make_client(tmp_path)

    response = client.get("/api/jobs/batch_20260706_071643_7a721401")

    assert response.status_code == 404
    assert response.json()["detail"]["message"] == "Job not found"


def _create_fast_job(store: JobStore, *, enable_vlm: bool = False, safe_mode: bool = False):
    return store.create_job(
        filename="sample.mp4",
        content=b"tiny video bytes",
        query="driver without seatbelt",
        settings=jobs_module.AnalysisSettings(
            fps=1.0,
            top_k=1,
            enable_vlm=enable_vlm,
            vlm_enabled=enable_vlm,
            vlm_profile="lightweight_512m" if enable_vlm else "rule_based",
            device="cpu",
            safe_mode=safe_mode,
        ),
    )


def test_analysis_concurrency_two_allows_two_fake_jobs_to_run_together(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "jobs")
    first = _create_fast_job(store)
    second = _create_fast_job(store)
    barrier = threading.Barrier(2)
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def fake_run_pipeline(**kwargs):  # noqa: ARG001
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            barrier.wait(timeout=2)
            time.sleep(0.05)
            return []
        finally:
            with active_lock:
                active -= 1

    monkeypatch.setattr(jobs_module.SETTINGS, "analysis_concurrency", 2)
    monkeypatch.setattr(jobs_module.SETTINGS, "worker_concurrency", 1)
    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda job_id: jobs_module.execute_analysis_job(store, job_id), [first.job_id, second.job_id]))

    assert max_active == 2
    assert store.require(first.job_id).status == "completed"
    assert store.require(second.job_id).status == "completed"
    assert store.require(first.job_id).metrics["componentDiagnostics"]["analysisConcurrency"] == 2
    assert store.require(second.job_id).metrics["componentDiagnostics"]["analysisConcurrency"] == 2


def test_analysis_concurrency_one_preserves_sequential_fake_jobs(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "jobs")
    first = _create_fast_job(store)
    second = _create_fast_job(store)
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def fake_run_pipeline(**kwargs):  # noqa: ARG001
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.1)
            return []
        finally:
            with active_lock:
                active -= 1

    monkeypatch.setattr(jobs_module.SETTINGS, "analysis_concurrency", 1)
    monkeypatch.setattr(jobs_module.SETTINGS, "worker_concurrency", 1)
    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda job_id: jobs_module.execute_analysis_job(store, job_id), [first.job_id, second.job_id]))

    assert max_active == 1
    assert store.require(first.job_id).status == "completed"
    assert store.require(second.job_id).status == "completed"


def test_vlm_concurrency_remains_capped_when_analysis_concurrency_is_higher(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "jobs")
    first = _create_fast_job(store, enable_vlm=True, safe_mode=True)
    second = _create_fast_job(store, enable_vlm=True, safe_mode=True)
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def fake_run_pipeline(**kwargs):
        nonlocal active, max_active
        assert kwargs["enable_vlm"] is True
        assert kwargs["vlm_profile"] == "lightweight_512m"
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.1)
            return []
        finally:
            with active_lock:
                active -= 1

    monkeypatch.setattr(jobs_module.SETTINGS, "analysis_concurrency", 2)
    monkeypatch.setattr(jobs_module.SETTINGS, "worker_concurrency", 1)
    monkeypatch.setattr(jobs_module.SETTINGS, "vlm_concurrency", 1)
    monkeypatch.setattr(jobs_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(jobs_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(jobs_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(jobs_module, "run_pipeline", fake_run_pipeline)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda job_id: jobs_module.execute_analysis_job(store, job_id), [first.job_id, second.job_id]))

    assert max_active == 1
    assert store.require(first.job_id).status == "completed"
    assert store.require(second.job_id).status == "completed"
    first_diag = store.require(first.job_id).metrics["componentDiagnostics"]
    assert first_diag["analysisConcurrency"] == 2
    assert first_diag["vlmConcurrency"] == 1
    assert first_diag["vlmConcurrencyLimited"] is True
