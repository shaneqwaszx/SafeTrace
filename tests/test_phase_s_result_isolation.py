from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

import src.api.jobs as jobs_module
from src.api.batches import BatchStore
from src.api.exports import create_export, verify_export
from src.api.jobs import AnalysisSettings, JobStore, execute_analysis_job, result_ownership_context
from src.api.recovery_checkpoints import latest_valid_checkpoint, write_checkpoint
from src.api.result_cache import cache_root_for
from src.api.result_ownership import validate_result_ownership
from src.api.server import create_app


def _settings() -> AnalysisSettings:
    return AnalysisSettings(
        fps=1.0,
        top_k=2,
        enable_vlm=False,
        device="cpu",
        vlm_profile="rule_based",
        vlm_enabled=False,
        safe_mode=True,
        use_case_profile={"profileId": "general_safety"},
        review_mode="fast_local",
    )


def _create(
    store: JobStore,
    *,
    filename: str,
    content: bytes,
    relative: str,
    batch_id: str | None = None,
):
    group = str(Path(relative).parent).replace("\\", "/")
    return store.create_job(
        filename=filename,
        content=content,
        query="general safety",
        settings=_settings(),
        source_metadata={
            "originalFilename": filename,
            "sourceRelativePath": relative,
            "sourceGroupPath": group,
            "batchId": batch_id,
            "recoverOnRestart": True,
        },
    )


def _fake_pipeline_factory(*, barrier: threading.Barrier | None = None, calls: list[str] | None = None):
    def fake_pipeline(**kwargs):
        upload_path = Path(kwargs["upload_path"])
        workspace = Path(kwargs["workspace"])
        marker = upload_path.read_bytes().decode("utf-8")
        if calls is not None:
            calls.append(marker)
        annotated = workspace / "annotated" / "frame_1.jpg"
        source_frame = workspace / "frames" / "frame_1.jpg"
        annotated.parent.mkdir(parents=True, exist_ok=True)
        source_frame.parent.mkdir(parents=True, exist_ok=True)
        annotated.write_bytes(f"annotated:{marker}".encode())
        source_frame.write_bytes(f"source:{marker}".encode())
        if barrier is not None:
            barrier.wait(timeout=5)
        return [{
            "frame_id": "frame_1",
            "source_frame_index": 1,
            "timestamp_seconds": 1.0,
            "score": 0.91,
            "frame_path": str(source_frame),
            "annotated_path": str(annotated),
            "detections": [{"label": marker, "confidence": 0.91}],
            "violations": [{
                "name": f"finding_{marker}",
                "severity": "medium",
                "confidence": 0.7,
                "description": f"finding owned by {marker}",
                "evidence": {"evidenceStrength": "likely_violation"},
            }],
        }]
    return fake_pipeline


def _assert_owned(record) -> None:
    validate_result_ownership(record.result, result_ownership_context(record))
    result = record.result
    assert result["jobId"] == record.job_id
    assert result["sourceRelativePath"] == record.source_metadata["sourceRelativePath"]
    assert all(frame["jobId"] == record.job_id for frame in result["frames"])
    assert all(frame["evidenceId"].startswith(f"{record.job_id}:") for frame in result["frames"])
    assert all(event["jobId"] == record.job_id for event in result["events"])
    assert all(item["jobId"] == record.job_id for item in result["violations"])


def test_two_concurrent_jobs_with_same_names_keep_results_workspaces_and_media_isolated(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "data" / "api_jobs")
    first = _create(store, filename="camera.mp4", content=b"JOB_A", relative="red/camera.mp4")
    second = _create(store, filename="camera.mp4", content=b"JOB_B", relative="green/camera.mp4")
    barrier = threading.Barrier(2)
    monkeypatch.setattr(jobs_module.SETTINGS, "analysis_concurrency", 2)
    monkeypatch.setattr(jobs_module.SETTINGS, "worker_concurrency", 1)
    monkeypatch.setattr(jobs_module, "run_pipeline", _fake_pipeline_factory(barrier=barrier))

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda job_id: execute_analysis_job(store, job_id), [first.job_id, second.job_id]))

    first = store.require(first.job_id)
    second = store.require(second.job_id)
    _assert_owned(first)
    _assert_owned(second)
    assert first.result["frames"][0]["imageUrl"] == f"/api/media/{first.job_id}/frame_1_annotated.jpg"
    assert second.result["frames"][0]["imageUrl"] == f"/api/media/{second.job_id}/frame_1_annotated.jpg"
    assert first.media_files["frame_1_annotated.jpg"].read_bytes() == b"annotated:JOB_A"
    assert second.media_files["frame_1_annotated.jpg"].read_bytes() == b"annotated:JOB_B"
    assert first.job_dir / "workspace" != second.job_dir / "workspace"
    assert not list(store.root_dir.rglob("manifest.json.*.tmp"))


def test_api_technical_and_media_routes_fail_closed_on_nested_foreign_owner(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "jobs")
    first = _create(store, filename="a.mp4", content=b"JOB_A", relative="a/a.mp4")
    second = _create(store, filename="b.mp4", content=b"JOB_B", relative="b/b.mp4")
    monkeypatch.setattr(jobs_module, "run_pipeline", _fake_pipeline_factory())
    execute_analysis_job(store, first.job_id)
    execute_analysis_job(store, second.job_id)
    with TestClient(create_app(store, BatchStore(tmp_path / "batches"))) as client:
        foreign = store.require(first.job_id)
        foreign.result["frames"][0]["jobId"] = second.job_id
        foreign.result["frames"][0]["imageUrl"] = f"/api/media/{second.job_id}/frame_1_annotated.jpg"
        for route in (
            f"/api/jobs/{first.job_id}/result",
            f"/api/reports/{first.job_id}/technical-json",
            f"/api/media/{first.job_id}/frame_1_annotated.jpg",
        ):
            response = client.get(route)
            assert response.status_code == 409, (route, response.text)
            assert response.json()["detail"]["code"] == "cross_job_result_contamination"


def test_exact_cache_reuse_rewrites_all_owners_returns_deep_copy_and_survives_origin_delete(monkeypatch, tmp_path):
    calls: list[str] = []
    store = JobStore(tmp_path / "data" / "api_jobs")
    monkeypatch.setattr(jobs_module, "run_pipeline", _fake_pipeline_factory(calls=calls))
    first = _create(store, filename="same.mp4", content=b"SAME", relative="first/same.mp4")
    execute_analysis_job(store, first.job_id)
    second = _create(store, filename="same.mp4", content=b"SAME", relative="second/same.mp4")
    execute_analysis_job(store, second.job_id)

    first = store.require(first.job_id)
    second = store.require(second.job_id)
    assert calls == ["SAME"]
    assert second.metrics["resultCacheHit"] is True
    _assert_owned(first)
    _assert_owned(second)
    assert second.result["technicalDetails"]["resultCache"]["sourceJobId"] == first.job_id
    second.result["frames"][0]["violations"].append({"id": "local-mutation"})
    assert all(item.get("id") != "local-mutation" for item in first.result["frames"][0]["violations"])
    media_path = Path(second.media_files["frame_1_annotated.jpg"])
    assert store.delete(first.job_id)
    assert media_path.is_file() and media_path.read_bytes() == b"annotated:SAME"
    metadata = next(cache_root_for(second).glob("*/metadata.json"))
    cache_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    assert cache_metadata["jobReferences"] == [second.job_id]
    assert cache_metadata["referenceCount"] == 1


def test_batch_results_are_keyed_and_never_flatten_child_evidence(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "jobs")
    batches = BatchStore(tmp_path / "batches")
    monkeypatch.setattr(jobs_module.SETTINGS, "analysis_concurrency", 2)
    monkeypatch.setattr(jobs_module, "run_pipeline", _fake_pipeline_factory())
    with TestClient(create_app(store, batches)) as client:
        response = client.post(
            "/api/batches/analyze",
            files=[
                ("files", ("vehicle-a.mp4", b"JOB_A", "video/mp4")),
                ("files", ("vehicle-b.mp4", b"JOB_B", "video/mp4")),
                ("files", ("vehicle-c.mp4", b"JOB_C", "video/mp4")),
            ],
            data={"query": "general safety", "device": "cpu"},
        )
        assert response.status_code == 200
        created = response.json()
        payload = client.get(f"/api/batches/{created['batchId']}/results").json()

    assert payload["rawEvidenceFlattened"] is False
    assert set(payload["resultsByJobId"]) == set(created["jobIds"])
    for job_id, child in payload["resultsByJobId"].items():
        assert child["jobId"] == job_id
        assert child["result"]["jobId"] == job_id
        assert {frame["jobId"] for frame in child["result"]["frames"]} == {job_id}


def test_export_and_recovery_ownership_are_fail_closed(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "data" / "api_jobs")
    monkeypatch.setattr(jobs_module, "run_pipeline", _fake_pipeline_factory())
    record = _create(store, filename="export.mp4", content=b"EXPORT", relative="fleet/export.mp4")
    execute_analysis_job(store, record.job_id)
    record = store.require(record.job_id)
    exported = create_export(store, record.job_id)
    export_root = Path(exported["path"])
    verified = verify_export(export_root)
    assert verified["verified"] is True
    assert verified["manifest"]["jobId"] == record.job_id
    technical = json.loads((export_root / "technical_result.json").read_text(encoding="utf-8"))
    assert technical["jobId"] == record.job_id
    assert {frame["jobId"] for frame in technical["frames"]} == {record.job_id}

    foreign_root = tmp_path / "job_owner_a"
    write_checkpoint(
        foreign_root,
        job_id="job_owner_b",
        stage="upload_complete",
        identity={"sourceChecksum": "foreign"},
        progress=0.1,
        status="running",
        current_step="upload",
    )
    recovered = latest_valid_checkpoint(
        foreign_root,
        {"sourceChecksum": "foreign"},
        expected_job_id="job_owner_a",
    )
    assert recovered["checkpoint"] is None
    assert recovered["invalidNewerCheckpoints"][0]["reason"] == "checkpoint_job_owner_mismatch"
