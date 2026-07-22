from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.api.jobs as jobs_module
import src.api.server as server_module
from src.api.batches import BatchStore
from src.api.jobs import (
    AnalysisSettings,
    JobRestartConflictError,
    JobStore,
    execute_analysis_job,
)
from src.api.result_cache import cache_identity, cache_key, cache_root_for
from src.api.server import create_app
from src.config import SETTINGS


def _settings() -> AnalysisSettings:
    return AnalysisSettings(
        fps=1.0,
        top_k=5,
        enable_vlm=False,
        device="cpu",
        vlm_profile="rule_based",
        vlm_enabled=False,
        safe_mode=True,
        use_case_profile={"profileId": "general_safety"},
        review_mode="fast_local",
    )


def _job(store: JobStore, name: str = "sample.mp4", content: bytes = b"phase-t-source"):
    return store.create_job(
        filename=name,
        content=content,
        query="general safety",
        settings=_settings(),
        source_metadata={
            "recoverOnRestart": True,
            "sourceRelativePath": f"Fleet/{name}",
            "sourceGroupPath": "Fleet",
        },
    )


def _interrupt_with_artifacts(store: JobStore, record, stage: str) -> dict[str, Path]:
    store.persist_recovery_stage(record.job_id, stage, completed_candidate_ids=["old-frame"])
    artifacts = {
        "partial": record.job_dir / jobs_module.PARTIAL_PIPELINE_FILENAME,
        "workspace": record.job_dir / "workspace" / "old-frame.jpg",
        "frame": record.job_dir / "frames" / "sampled.jpg",
        "annotation": record.job_dir / "annotated" / "annotated.jpg",
        "evidence": record.job_dir / "evidence" / "evidence.json",
        "report": record.job_dir / "reports" / "technical.json",
        "media": record.output_dir / "partial.jpg",
    }
    for path in artifacts.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old-partial-artifact")
    record.result_path = record.job_dir / jobs_module.RESULT_FILENAME
    record.result_path.write_text('{"old": true}', encoding="utf-8")
    record.media_files = {"partial.jpg": artifacts["media"]}
    record.metrics.update({
        "lastHeartbeatAt": "stale",
        "recoveryPartialResultReused": True,
        "recoveryResumeCheckpoint": stage,
        "completedEvidenceIds": ["old-evidence"],
    })
    store.persist_job(record)
    store.update_status(
        record.job_id,
        status="failed",
        progress=0.7,
        current_step=f"Interrupted during {stage}",
        error="interrupted",
        technical_error="old technical error",
        error_type="InterruptedJob",
    )
    return artifacts


@pytest.mark.parametrize("stage", ["frame_sampling_complete", "detector_work_complete"])
def test_restart_fresh_removes_stage_artifacts_and_preserves_exact_source(tmp_path, stage):
    store = JobStore(tmp_path / "data" / "api_jobs")
    source = b"exact-original-upload"
    record = _job(store, f"{stage}.mp4", source)
    upload_path = record.upload_path
    checksum = hashlib.sha256(source).hexdigest()
    artifacts = _interrupt_with_artifacts(store, record, stage)

    unrelated = _job(store, "unrelated.mp4", b"unrelated-source")
    unrelated_marker = unrelated.job_dir / "workspace" / "keep.txt"
    unrelated_marker.parent.mkdir(parents=True)
    unrelated_marker.write_text("keep", encoding="utf-8")

    restarted = store.restart_fresh(record.job_id)
    fresh = store.require(record.job_id)

    assert restarted["sourcePreserved"] is True
    assert restarted["sourceChecksum"] == checksum
    assert fresh.upload_path == upload_path
    assert fresh.upload_path.read_bytes() == source
    assert fresh.status == "queued"
    assert fresh.progress == 0.0
    assert fresh.current_step == "Queued for fresh analysis"
    assert fresh.error is None and fresh.technical_error is None and fresh.error_type is None
    assert fresh.retry_count == 0 and fresh.next_retry_at is None
    assert fresh.started_at is None and fresh.finished_at is None
    assert fresh.result is None and fresh.result_path is None and fresh.media_files == {}
    assert fresh.recovery_action == "restart_fresh_by_user"
    assert fresh.metrics["bypassResultCacheOnce"] is True
    assert fresh.metrics["restartGeneration"] == 1
    assert "lastHeartbeatAt" not in fresh.metrics
    assert "recoveryPartialResultReused" not in fresh.metrics
    assert all(not path.exists() for path in artifacts.values())
    assert not (fresh.job_dir / jobs_module.RESULT_FILENAME).exists()
    checkpoints = list((fresh.job_dir / "checkpoints").glob("checkpoint_*.json"))
    assert len(checkpoints) == 1
    payload = json.loads(checkpoints[0].read_text(encoding="utf-8"))["payload"]
    assert payload["stage"] == "upload_complete"
    assert payload["jobId"] == fresh.job_id
    assert not list(fresh.job_dir.glob(".restart_fresh_*.tmp"))
    assert unrelated_marker.read_text(encoding="utf-8") == "keep"


def test_restart_fresh_bypasses_old_partial_and_exact_cache_once(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "data" / "api_jobs")
    record = _job(store, "rerun.mp4")
    _interrupt_with_artifacts(store, record, "detector_work_complete")
    calls = []

    def fresh_pipeline(**_kwargs):
        calls.append("ran")
        return []

    monkeypatch.setattr(jobs_module, "run_pipeline", fresh_pipeline)
    store.restart_fresh(record.job_id)
    execute_analysis_job(store, record.job_id)
    execute_analysis_job(store, record.job_id)

    completed = store.require(record.job_id)
    assert calls == ["ran"]
    assert completed.status == "completed"
    assert completed.metrics.get("recoveryPartialResultReused") is not True
    assert completed.metrics["bypassResultCacheOnce"] is False
    assert completed.result is not None
    assert completed.result["jobId"] == record.job_id
    assert len(list(record.job_dir.glob(jobs_module.RESULT_FILENAME))) == 1


def _write_cache_entry(record, references: list[str]) -> Path:
    entry = cache_root_for(record) / cache_key(record)
    entry.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cacheEntryId": entry.name,
        "cacheKey": cache_key(record),
        "identity": cache_identity(record),
        "complete": True,
        "jobReferences": references,
        "referenceCount": len(references),
    }
    (entry / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (entry / "result.json").write_text("{}", encoding="utf-8")
    return entry


def test_restart_fresh_releases_local_cache_reference_without_purging_unrelated(tmp_path):
    store = JobStore(tmp_path / "data" / "api_jobs")
    selected = _job(store, "selected.mp4", b"selected")
    unrelated = _job(store, "unrelated.mp4", b"unrelated")
    _interrupt_with_artifacts(store, selected, "frame_sampling_complete")
    selected_entry = _write_cache_entry(selected, [selected.job_id])
    unrelated_entry = _write_cache_entry(unrelated, [unrelated.job_id])

    store.restart_fresh(selected.job_id)

    assert selected_entry.is_dir()
    selected_metadata = json.loads((selected_entry / "metadata.json").read_text(encoding="utf-8"))
    assert selected_metadata["jobReferences"] == []
    assert unrelated_entry.is_dir()
    unrelated_metadata = json.loads((unrelated_entry / "metadata.json").read_text(encoding="utf-8"))
    assert unrelated_metadata["jobReferences"] == [unrelated.job_id]


def test_optional_cache_purge_is_source_compatible_scoped_and_shared_safe(tmp_path):
    store = JobStore(tmp_path / "data" / "api_jobs")
    purge_job = _job(store, "purge.mp4", b"purge-source")
    shared_job = _job(store, "shared.mp4", b"shared-source")
    unrelated = _job(store, "other.mp4", b"other-source")
    _interrupt_with_artifacts(store, purge_job, "detector_work_complete")
    _interrupt_with_artifacts(store, shared_job, "detector_work_complete")
    purge_entry = _write_cache_entry(purge_job, [purge_job.job_id])
    shared_entry = _write_cache_entry(shared_job, [shared_job.job_id, "outside-job"])
    unrelated_entry = _write_cache_entry(unrelated, [unrelated.job_id])

    purged = store.restart_fresh(
        purge_job.job_id,
        purge_compatible_cache=True,
        cache_purge_selected_job_ids={purge_job.job_id},
    )
    retained = store.restart_fresh(
        shared_job.job_id,
        purge_compatible_cache=True,
        cache_purge_selected_job_ids={shared_job.job_id},
    )

    assert not purge_entry.exists()
    assert purged["cachePurge"]["purged"] == [purge_entry.name]
    assert shared_entry.is_dir()
    assert retained["cachePurge"]["retained"][0]["reason"] == "shared_with_unselected_jobs"
    assert unrelated_entry.is_dir()


def test_restart_fresh_conflicts_with_active_writer_without_mutation(tmp_path):
    store = JobStore(tmp_path / "data" / "api_jobs")
    record = _job(store, "active.mp4")
    marker = record.job_dir / "workspace" / "active.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("active", encoding="utf-8")
    store.update_status(record.job_id, status="running_detector", progress=0.4, current_step="Detector active")

    with pytest.raises(JobRestartConflictError, match="actively writing"):
        store.restart_fresh(record.job_id)

    current = store.require(record.job_id)
    assert current.status == "running_detector"
    assert marker.read_text(encoding="utf-8") == "active"
    assert current.upload_path.read_bytes() == b"phase-t-source"


def test_restart_fresh_api_requires_confirmations_and_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "_execute_job_group", lambda *_args, **_kwargs: None)
    store = JobStore(tmp_path / "data" / "api_jobs")
    record = _job(store, "idempotent.mp4")
    _interrupt_with_artifacts(store, record, "frame_sampling_complete")
    app = create_app(store, BatchStore(tmp_path / "data" / "api_batches"))

    with TestClient(app) as client:
        assert client.post("/api/recovery/restart-fresh", json={"jobIds": [record.job_id]}).status_code == 400
        assert client.post("/api/recovery/restart-fresh", json={
            "jobIds": [record.job_id],
            "confirmRestartFresh": True,
            "purgeCompatibleCache": True,
        }).status_code == 400
        first = client.post("/api/recovery/restart-fresh", json={
            "jobIds": [record.job_id],
            "confirmRestartFresh": True,
        })
        assert first.status_code == 200
        assert first.json()["restartedJobIds"] == [record.job_id]
        assert client.get("/api/recovery").json()["candidateCount"] == 0
        generation = store.require(record.job_id).metrics["restartGeneration"]
        second = client.post("/api/recovery/restart-fresh", json={
            "jobIds": [record.job_id],
            "confirmRestartFresh": True,
        })
        assert second.status_code == 200
        assert second.json()["restartedJobIds"] == []
        assert second.json()["alreadyRestartedJobIds"] == [record.job_id]
        assert store.require(record.job_id).metrics["restartGeneration"] == generation


def _interrupted_batch(tmp_path):
    jobs = JobStore(tmp_path / "data" / "api_jobs")
    batches = BatchStore(tmp_path / "data" / "api_batches")
    batch = batches.create_from_files(
        files=[("one.mp4", b"one"), ("two.mp4", b"two"), ("three.mp4", b"three")],
        source_filename="batch",
        query="general safety",
        settings=_settings(),
        job_store=jobs,
    )
    completed = jobs.require(batch.job_ids[0])
    jobs.update_status(completed.job_id, status="completed", progress=1.0, current_step="Completed")
    completed_marker = completed.job_dir / "reports" / "completed.txt"
    completed_marker.parent.mkdir(parents=True)
    completed_marker.write_text("preserve", encoding="utf-8")
    for job_id in batch.job_ids[1:]:
        _interrupt_with_artifacts(jobs, jobs.require(job_id), "detector_work_complete")
    return jobs, batches, batch, completed_marker


def test_batch_restart_all_unfinished_preserves_completed_sibling(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "_execute_job_group", lambda *_args, **_kwargs: None)
    jobs, batches, batch, completed_marker = _interrupted_batch(tmp_path)
    with TestClient(create_app(jobs, batches)) as client:
        response = client.post("/api/recovery/restart-fresh", json={
            "batchIds": [batch.batch_id],
            "confirmRestartFresh": True,
        })
    assert response.status_code == 200
    assert set(response.json()["restartedJobIds"]) == set(batch.job_ids[1:])
    assert jobs.require(batch.job_ids[0]).status == "completed"
    assert completed_marker.read_text(encoding="utf-8") == "preserve"
    assert all(jobs.require(job_id).status == "queued" for job_id in batch.job_ids[1:])


def test_batch_restart_selected_child_preserves_other_unfinished_sibling(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "_execute_job_group", lambda *_args, **_kwargs: None)
    jobs, batches, batch, _completed_marker = _interrupted_batch(tmp_path)
    selected, untouched = batch.job_ids[1:]
    with TestClient(create_app(jobs, batches)) as client:
        response = client.post("/api/recovery/restart-fresh", json={
            "jobIds": [selected],
            "confirmRestartFresh": True,
        })
    assert response.status_code == 200
    assert response.json()["restartedJobIds"] == [selected]
    assert jobs.require(selected).status == "queued"
    assert jobs.require(untouched).status == "failed"
    assert (jobs.require(untouched).job_dir / jobs_module.PARTIAL_PIPELINE_FILENAME).is_file()


def test_backend_restart_discard_policy_does_not_delete_without_user_decision(monkeypatch, tmp_path):
    monkeypatch.setattr(SETTINGS, "recovery_policy", "discard")
    root = tmp_path / "data" / "api_jobs"
    first_store = JobStore(root)
    record = _job(first_store, "restart-policy.mp4")
    _interrupt_with_artifacts(first_store, record, "frame_sampling_complete")

    recovered_store = JobStore(root)
    with TestClient(create_app(recovered_store, BatchStore(tmp_path / "data" / "api_batches"))) as client:
        assert client.get(f"/api/jobs/{record.job_id}").status_code == 200
        recovered = recovered_store.require(record.job_id)
        assert recovered.upload_path.read_bytes() == b"phase-t-source"
        assert recovered.status == "failed"
        recovery = client.get("/api/recovery").json()
        assert recovery["candidateCount"] == 1
        assert recovery["jobs"][0]["jobId"] == record.job_id
