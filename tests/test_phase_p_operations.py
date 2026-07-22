from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

import src.api.jobs as jobs_module
from src.api.batches import BatchStore
from src.api.jobs import AnalysisSettings, JobStore, execute_analysis_job
from src.api.operations import cleanup_preview, dashboard_summary, storage_summary
from src.api.result_cache import cache_key
from src.api.server import create_app
import src.api.server as server_module
from src.config import SETTINGS


def settings(query_profile: str = "general_safety") -> AnalysisSettings:
    return AnalysisSettings(
        fps=1.0,
        top_k=5,
        enable_vlm=False,
        device="cpu",
        vlm_profile="rule_based",
        vlm_enabled=False,
        safe_mode=True,
        use_case_profile={"profileId": query_profile},
        review_mode="fast_local",
    )


def create_job(store: JobStore, *, name: str = "sample.mp4", query: str = "general safety"):
    return store.create_job(
        filename=name,
        content=b"phase-p-media",
        query=query,
        settings=settings(),
        source_metadata={"recoverOnRestart": True, "sourceRelativePath": f"Vehicle A/{name}", "sourceGroupPath": "Vehicle A"},
    )


def interrupt(store: JobStore, job_id: str) -> None:
    record = store.require(job_id)
    store.update_status(job_id, status="running", progress=0.5, current_step="Detector inference")
    record = store.require(job_id)
    record.updated_at = datetime.now(timezone.utc) - timedelta(hours=2)
    store.persist_job(record)


def test_prompt_policy_discovers_interrupted_job_without_auto_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(SETTINGS, "recovery_policy", "prompt")
    monkeypatch.setattr(SETTINGS, "stale_running_minutes", 1.0)
    jobs = JobStore(tmp_path / "jobs")
    batches = BatchStore(tmp_path / "batches")
    record = create_job(jobs)
    interrupt(jobs, record.job_id)

    with TestClient(create_app(JobStore(tmp_path / "jobs"), batches)) as client:
        payload = client.get("/api/recovery").json()
        assert payload["policy"] == "prompt"
        assert payload["candidateCount"] == 1
        assert payload["jobs"][0]["lastStage"] in {"Awaiting recovery decision", "Analysis interrupted"}
        assert client.get(f"/api/jobs/{record.job_id}").json()["status"] in {"paused", "failed"}


def test_recovery_resume_preserves_completed_and_discard_removes_only_interrupted(monkeypatch, tmp_path):
    monkeypatch.setattr(SETTINGS, "recovery_policy", "prompt")
    monkeypatch.setattr(SETTINGS, "stale_running_minutes", 1.0)
    monkeypatch.setattr(jobs_module, "run_pipeline", lambda **_kwargs: [])
    jobs = JobStore(tmp_path / "jobs")
    batches = BatchStore(tmp_path / "batches")
    completed = create_job(jobs, name="completed.mp4")
    execute_analysis_job(jobs, completed.job_id)
    completed_result = jobs.require(completed.job_id).result

    resumable = create_job(jobs, name="resume.mp4")
    interrupt(jobs, resumable.job_id)
    with TestClient(create_app(JobStore(tmp_path / "jobs"), batches)) as client:
        response = client.post("/api/recovery/resume", json={"jobIds": [resumable.job_id]})
        assert response.status_code == 200
        assert resumable.job_id in response.json()["resumedJobIds"]
        assert client.get(f"/api/jobs/{resumable.job_id}").json()["status"] == "completed"
        assert jobs.require(completed.job_id).result == completed_result

    discarded = create_job(jobs, name="discard.mp4")
    interrupt(jobs, discarded.job_id)
    discarded_dir = discarded.job_dir
    with TestClient(create_app(JobStore(tmp_path / "jobs"), batches)) as client:
        response = client.post("/api/recovery/discard", json={"jobIds": [discarded.job_id]})
        assert response.status_code == 200
        assert response.json()["actualReclaimedBytes"] > 0
    assert not discarded_dir.exists()
    assert jobs.require(completed.job_id).result == completed_result


def test_exact_result_cache_hit_and_query_miss(monkeypatch, tmp_path):
    calls = 0

    def pipeline(**_kwargs):
        nonlocal calls
        calls += 1
        return []

    monkeypatch.setattr(jobs_module, "run_pipeline", pipeline)
    jobs = JobStore(tmp_path / "data" / "api_jobs")
    first = create_job(jobs, query="general safety")
    execute_analysis_job(jobs, first.job_id)
    assert calls == 1

    second = create_job(jobs, query="general safety")
    assert cache_key(first) == cache_key(second)
    execute_analysis_job(jobs, second.job_id)
    assert calls == 1
    assert jobs.require(second.job_id).result["technicalDetails"]["resultCache"]["hit"] is True

    changed = create_job(jobs, query="phone use")
    assert cache_key(first) != cache_key(changed)
    execute_analysis_job(jobs, changed.job_id)
    assert calls == 2


def test_incomplete_result_is_never_a_cache_hit(monkeypatch, tmp_path):
    calls = 0

    def pipeline(**_kwargs):
        nonlocal calls
        calls += 1
        return []

    monkeypatch.setattr(jobs_module, "run_pipeline", pipeline)
    jobs = JobStore(tmp_path / "data" / "api_jobs")
    queued = create_job(jobs)
    duplicate = create_job(jobs)
    execute_analysis_job(jobs, duplicate.job_id)
    assert calls == 1
    assert jobs.require(queued.job_id).status == "queued"


def test_storage_accounting_cleanup_preview_and_protection(monkeypatch, tmp_path):
    monkeypatch.setattr(SETTINGS, "completed_retention_days", 1.0)
    monkeypatch.setattr(SETTINGS, "failed_retention_days", 1.0)
    jobs = JobStore(tmp_path / "data" / "api_jobs")
    batches = BatchStore(tmp_path / "data" / "api_batches")
    expired = create_job(jobs, name="expired.mp4")
    jobs.update_status(expired.job_id, status="failed", progress=1.0, current_step="Failed", error="test")
    expired = jobs.require(expired.job_id)
    expired.finished_at = datetime.now(timezone.utc) - timedelta(days=5)
    expired.updated_at = expired.finished_at
    jobs.persist_job(expired)

    active = create_job(jobs, name="active.mp4")
    summary = storage_summary(jobs, batches)
    assert summary["categories"]["uploads"] > 0
    assert summary["protected"]["activeJobs"] >= 1
    preview = cleanup_preview(jobs)
    assert expired.job_id in {item.get("id") for item in preview["candidates"]}
    assert active.job_id not in {item.get("id") for item in preview["candidates"]}


def test_dashboard_and_paginated_api_are_backend_canonical(tmp_path):
    jobs = JobStore(tmp_path / "data" / "api_jobs")
    batches = BatchStore(tmp_path / "data" / "api_batches")
    create_job(jobs)
    dashboard = dashboard_summary(jobs, batches)
    assert dashboard["jobsByStatus"]["queued"] == 1
    assert dashboard["accuracyMetricsAvailable"] is False
    with TestClient(create_app(jobs, batches)) as client:
        assert client.get("/api/dashboard/summary").status_code == 200
        jobs_page = client.get("/api/jobs?page=1&pageSize=1").json()
        assert jobs_page["total"] == 1
        assert len(jobs_page["items"]) == 1
        assert client.get("/api/storage/summary").status_code == 200
        preview = client.post("/api/storage/cleanup/preview", json={}).json()
        assert preview["dryRun"] is True
        assert client.post("/api/storage/cleanup/apply", json={}).status_code == 400


def test_low_disk_guard_rejects_new_upload_without_creating_job(monkeypatch, tmp_path):
    jobs = JobStore(tmp_path / "data" / "api_jobs")
    batches = BatchStore(tmp_path / "data" / "api_batches")
    monkeypatch.setattr(SETTINGS, "min_free_disk_gb", 2.0)
    monkeypatch.setattr(SETTINGS, "min_free_disk_mb", 0.0)
    monkeypatch.setattr(server_module.shutil, "disk_usage", lambda _path: type("Usage", (), {"total": 10 * 1024**3, "used": 9 * 1024**3, "free": 1024**3})())
    with TestClient(create_app(jobs, batches)) as client:
        response = client.post(
            "/api/analyze",
            files={"file": ("sample.mp4", b"video", "video/mp4")},
            data={"query": "general safety"},
        )
    assert response.status_code == 507
    assert jobs.records() == []
