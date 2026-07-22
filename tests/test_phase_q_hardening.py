from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.api.exports as exports_module
import src.api.jobs as jobs_module
from src.api.batches import BatchStore
from src.api.exports import create_export, list_exports, verify_export
from src.api.jobs import AnalysisSettings, JobStore, execute_analysis_job
from src.api.operations import cleanup_preview
from src.api.recovery_checkpoints import STAGES, latest_valid_checkpoint, write_checkpoint
from src.api.result_ownership import stamp_result_ownership
from src.api.retention_scheduler import RetentionScheduler
from src.api.server import create_app
from src.config import SETTINGS


def settings() -> AnalysisSettings:
    return AnalysisSettings(
        fps=1.0, top_k=5, enable_vlm=False, device="cpu", vlm_profile="rule_based",
        vlm_enabled=False, safe_mode=True, use_case_profile={"profileId": "general_safety"},
        review_mode="fast_local",
    )


def create_job(store: JobStore, name: str = "sample.mp4", query: str = "general safety"):
    return store.create_job(
        filename=name, content=b"phase-q-media", query=query, settings=settings(),
        source_metadata={"recoverOnRestart": True, "sourceRelativePath": f"Fleet/{name}", "sourceGroupPath": "Fleet"},
    )


def complete(monkeypatch, store: JobStore, name: str = "sample.mp4"):
    monkeypatch.setattr(jobs_module, "run_pipeline", lambda **_kwargs: [])
    record = create_job(store, name)
    execute_analysis_job(store, record.job_id)
    return store.require(record.job_id)


def test_checkpoint_corrupt_newest_falls_back_and_identity_change_invalidates_downstream(tmp_path):
    root = tmp_path / "job"
    identity = {"sourceChecksum": "a", "query": "seatbelt", "detector": "one"}
    first = write_checkpoint(root, job_id="job_test", stage="upload_complete", identity=identity, progress=0.1, status="running", current_step="upload")
    newest = write_checkpoint(root, job_id="job_test", stage="detector_work_complete", identity=identity, progress=0.7, status="running", current_step="detector")
    newest.write_text("{broken", encoding="utf-8")
    selected = latest_valid_checkpoint(root, identity)
    assert selected["checkpointPath"] == str(first)
    assert selected["invalidNewerCheckpoints"][0]["reason"] == "corrupt_or_incomplete_checkpoint"

    valid_downstream = write_checkpoint(root, job_id="job_test", stage="aggregation_complete", identity=identity, progress=0.9, status="running", current_step="aggregation")
    changed = latest_valid_checkpoint(root, {**identity, "query": "helmet"})
    assert changed["checkpointPath"] == str(first)
    assert changed["invalidationReason"] == "configuration_identity_changed_downstream_invalidated"
    assert any(item["path"] == str(valid_downstream) for item in changed["invalidNewerCheckpoints"])


def test_concurrent_checkpoint_writes_are_unique_valid_and_leave_no_temps(tmp_path):
    root = tmp_path / "job"
    identity = {"sourceChecksum": "same", "query": "general"}
    written = []

    def persist(index: int):
        written.append(write_checkpoint(
            root, job_id="job_concurrent", stage="detector_work_complete", identity=identity,
            progress=0.5, status="running", current_step="detector", completed_frame_index=index,
        ))

    threads = [threading.Thread(target=persist, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len({path.name for path in written}) == 8
    assert len(list((root / "checkpoints").glob("checkpoint_*.json"))) == 8
    assert not list((root / "checkpoints").glob("*.tmp"))
    assert latest_valid_checkpoint(root, identity)["checkpoint"] is not None


def test_completed_job_records_all_applicable_safe_boundaries(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "data" / "api_jobs")
    record = complete(monkeypatch, store)
    stages = {
        json.loads(path.read_text(encoding="utf-8"))["payload"]["stage"]
        for path in (record.job_dir / "checkpoints").glob("checkpoint_*.json")
    }
    assert {
        "upload_complete", "media_probe_complete", "frame_sampling_complete", "ranking_complete",
        "detector_work_complete", "aggregation_complete", "evidence_report_complete", "completed",
    }.issubset(stages)
    assert stages.issubset(set(STAGES))


def test_resume_reuses_checksummed_post_pipeline_artifact(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "data" / "api_jobs")
    record = create_job(store, "resume-partial.mp4")
    partial = record.job_dir / jobs_module.PARTIAL_PIPELINE_FILENAME
    partial.write_text("[]", encoding="utf-8")
    import hashlib
    checksum = hashlib.sha256(partial.read_bytes()).hexdigest()
    store.persist_recovery_stage(
        record.job_id,
        "detector_work_complete",
        partial_output_reference=partial.name,
        partial_output_checksum_sha256=checksum,
    )
    store.update_status(
        record.job_id, status="failed", progress=1.0, current_step="Interrupted",
        error="interrupted", error_type="InterruptedJob",
    )
    assert store.reset_for_retry(record.job_id, recovery_action="resumed_by_user")

    def should_not_run(**_kwargs):
        raise AssertionError("pipeline should be reused from the valid partial artifact")

    monkeypatch.setattr(jobs_module, "run_pipeline", should_not_run)
    execute_analysis_job(store, record.job_id)
    completed_record = store.require(record.job_id)
    assert completed_record.status == "completed"
    assert completed_record.metrics["recoveryPartialResultReused"] is True
    assert completed_record.metrics["recoveryResumeCheckpoint"] == "detector_work_complete"


def test_pin_unpin_persists_restart_and_blocks_retention(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "data" / "api_jobs")
    record = complete(monkeypatch, store)
    assert store.set_pin(record.job_id, True)["pinned"] is True
    recovered = JobStore(store.root_dir).require(record.job_id)
    assert recovered.source_metadata["pinned"] is True
    recovered.finished_at = datetime.now(timezone.utc) - timedelta(days=100)
    recovered.updated_at = recovered.finished_at
    store.persist_job(recovered)
    monkeypatch.setattr(SETTINGS, "completed_retention_days", 0.0)
    assert record.job_id not in {item.get("id") for item in cleanup_preview(store)["candidates"]}
    assert store.set_pin(record.job_id, False)["pinned"] is False
    assert record.job_id in {item.get("id") for item in cleanup_preview(store)["candidates"]}


def test_export_manifest_checksums_selected_evidence_and_restart_history(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "data" / "api_jobs")
    record = complete(monkeypatch, store)
    image = record.output_dir / "evidence.jpg"
    image.write_bytes(b"image")
    store.register_media_file(record.job_id, image.name, image)
    record.result["frames"] = [
        {"id": "frame-a", "frameNumber": 1, "timestamp": "00:00:01", "timestampSeconds": 1.0, "imageUrl": f"/api/media/{record.job_id}/evidence.jpg", "violations": []},
        {"id": "frame-b", "frameNumber": 2, "timestamp": "00:00:02", "timestampSeconds": 2.0, "imageUrl": None, "violations": []},
    ]
    record.result["evidence"] = [record.result["frames"][0]]
    record.result["evidenceStatus"] = "available"
    record.result = stamp_result_ownership(record.result, jobs_module.result_ownership_context(record))
    store.persist_job(record)
    exported = create_export(store, record.job_id, selected_evidence_ids=["frame-a"])
    assert exported["verified"] is True
    verification = verify_export(Path(exported["path"]))
    assert verification["verified"] is True
    assert verification["manifest"]["originalVideoIncluded"] is False
    evidence = json.loads((Path(exported["path"]) / "evidence_metadata.json").read_text(encoding="utf-8"))
    assert [item["frameId"] for item in evidence] == ["frame-a"]
    assert list_exports(JobStore(store.root_dir))[0]["exportId"] == exported["exportId"]


def test_failed_export_cleans_partial_output(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "data" / "api_jobs")
    record = complete(monkeypatch, store)
    image = record.output_dir / "evidence.jpg"
    image.write_bytes(b"image")
    store.register_media_file(record.job_id, image.name, image)
    original = exports_module.shutil.copy2

    def fail_copy(source, target):
        if Path(source).name == "evidence.jpg":
            raise OSError("injected export failure")
        return original(source, target)

    monkeypatch.setattr(exports_module.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="injected export failure"):
        create_export(store, record.job_id)
    export_root = store.root_dir.parent / "exports"
    assert not list(export_root.glob(".export_*.tmp"))
    assert not list(export_root.glob("export_*"))


def test_export_and_delete_api_requires_verification_and_preserves_export(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "data" / "api_jobs")
    record = complete(monkeypatch, store)
    with TestClient(create_app(store, BatchStore(tmp_path / "data" / "api_batches"))) as client:
        assert client.post(f"/api/jobs/{record.job_id}/export-and-delete", json={}).status_code == 400
        response = client.post(f"/api/jobs/{record.job_id}/export-and-delete", json={"confirmDelete": True})
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "exported_and_deleted"
        assert payload["export"]["verified"] is True
        assert client.get(f"/api/jobs/{record.job_id}").status_code == 404
        assert Path(payload["export"]["path"]).is_dir()


def test_retention_scheduler_is_single_non_overlapping_and_audited(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "data" / "api_jobs")
    batches = BatchStore(tmp_path / "data" / "api_batches")
    scheduler = RetentionScheduler(store, batches)
    monkeypatch.setattr(SETTINGS, "retention_scheduler_enabled", True)
    monkeypatch.setattr(SETTINGS, "retention_startup_delay_seconds", 100.0)
    assert scheduler.start() is True
    assert scheduler.start() is False
    scheduler.stop()
    assert scheduler.running is False

    scheduler._run_lock.acquire()
    try:
        assert scheduler.run_once()["status"] == "skipped_overlap"
    finally:
        scheduler._run_lock.release()
    result = scheduler.run_once(apply=True)
    assert result["status"] == "completed"
    assert list((tmp_path / "data" / "cleanup_audit").glob("cleanup_*.json"))


def test_retention_scheduler_expires_generated_state_and_protects_owned_records(monkeypatch, tmp_path):
    data = tmp_path / "data"
    monkeypatch.setattr(SETTINGS, "data_dir", data)
    monkeypatch.setattr(SETTINGS, "completed_retention_days", 0.0)
    monkeypatch.setattr(SETTINGS, "failed_retention_days", 0.0)
    monkeypatch.setattr(SETTINGS, "cache_retention_days", 0.0)
    monkeypatch.setattr(SETTINGS, "log_retention_days", 0.0)
    store = JobStore(data / "api_jobs")
    batches = BatchStore(data / "api_batches")
    expired = complete(monkeypatch, store, "expired.mp4")
    failed = create_job(store, "failed.mp4")
    store.update_status(failed.job_id, status="failed", progress=1.0, current_step="Failed", error="test")
    pinned = complete(monkeypatch, store, "pinned.mp4")
    store.set_pin(pinned.job_id, True)
    exported = complete(monkeypatch, store, "exported.mp4")
    exported.source_metadata["exported"] = True
    store.persist_job(exported)
    active = create_job(store, "active.mp4")
    partial = data / "exports" / ".export_failed.tmp"
    partial.mkdir(parents=True)
    (partial / "partial.json").write_text("{}", encoding="utf-8")
    log = data / "logs" / "old.log"
    log.parent.mkdir(parents=True)
    log.write_text("old", encoding="utf-8")
    old = time.time() - 3600
    log.touch()
    import os
    os.utime(log, (old, old))

    result = RetentionScheduler(store, batches).run_once(apply=True)
    assert result["status"] == "completed"
    assert store.get(expired.job_id) is None
    assert store.get(failed.job_id) is None
    assert store.get(pinned.job_id) is not None
    assert store.get(exported.job_id) is not None
    assert store.get(active.job_id) is not None
    assert not partial.exists()
    assert not log.exists()
    assert result["apply"]["actualReclaimedBytes"] > 0


def test_lifespan_starts_one_scheduler_and_has_no_on_event_hook(monkeypatch, tmp_path):
    monkeypatch.setattr(SETTINGS, "retention_scheduler_enabled", True)
    monkeypatch.setattr(SETTINGS, "retention_startup_delay_seconds", 100.0)
    app = create_app(JobStore(tmp_path / "jobs"), BatchStore(tmp_path / "batches"))
    assert app.router.on_startup == []
    with TestClient(app):
        scheduler = app.state.retention_scheduler
        assert scheduler.running is True
        assert scheduler.run_count == 0
    assert scheduler.running is False


def test_active_write_delete_conflicts_and_repeated_export_delete_is_idempotent(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "data" / "api_jobs")
    active = create_job(store, "active.mp4")
    store.update_status(active.job_id, status="running", progress=0.5, current_step="Detector")
    with TestClient(create_app(store, BatchStore(tmp_path / "data" / "api_batches"))) as client:
        assert client.delete(f"/api/jobs/{active.job_id}").status_code == 409

    done = complete(monkeypatch, store, "done.mp4")
    with TestClient(create_app(store, BatchStore(tmp_path / "data" / "api_batches"))) as client:
        exported = client.post(f"/api/jobs/{done.job_id}/export", json={}).json()
        assert client.delete(f"/api/exports/{exported['exportId']}").status_code == 200
        assert client.delete(f"/api/exports/{exported['exportId']}").status_code == 404
    restarted = JobStore(store.root_dir).require(done.job_id)
    assert restarted.source_metadata.get("exported") is False


def test_recovery_decision_later_is_durable_and_not_reprompted(monkeypatch, tmp_path):
    monkeypatch.setattr(SETTINGS, "recovery_policy", "prompt")
    monkeypatch.setattr(SETTINGS, "stale_running_minutes", 0.0)
    store = JobStore(tmp_path / "jobs")
    record = create_job(store)
    store.update_status(record.job_id, status="running", progress=0.5, current_step="Detector")
    record = store.require(record.job_id)
    record.updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
    store.persist_job(record)
    with TestClient(create_app(JobStore(store.root_dir), BatchStore(tmp_path / "batches"))) as client:
        assert client.get("/api/recovery").json()["candidateCount"] == 1
        assert client.post("/api/recovery/later", json={"jobIds": [record.job_id]}).status_code == 200
        assert client.get("/api/recovery").json()["candidateCount"] == 0
        deferred = client.get("/api/recovery?includeDeferred=true").json()
        assert deferred["candidateCount"] == 1
        assert deferred["jobs"][0]["recoveryAction"] == "deferred_by_user"
