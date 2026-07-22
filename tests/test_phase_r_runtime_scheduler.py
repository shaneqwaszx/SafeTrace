"""Phase R profile and scheduler regressions using only fast test jobs."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import src.api.jobs as jobs_module
from src.api.jobs import AnalysisSettings, JobStore, LocalAnalysisScheduler
from src.config import Settings


ROOT = Path(__file__).resolve().parents[1]


def _job(store: JobStore, name: str, *, batch_id: str | None = None, child: int = 0):
    metadata = {}
    if batch_id:
        metadata = {
            "batchId": batch_id,
            "batchEnqueueSequence": 1_725_000_000_000,
            "childSequence": child,
        }
    return store.create_job(
        filename=name,
        content=b"real scheduler test payload",
        query="general safety",
        settings=AnalysisSettings(fps=1.0, top_k=1, enable_vlm=False, device="cpu"),
        source_metadata=metadata,
    )


def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def test_runtime_profiles_keep_test_portable_and_full_local_contracts(monkeypatch):
    # These assertions exercise profile defaults, not a developer's local
    # explicit override loaded from a machine-specific env file.
    for name in (
        "SAFETRACE_DEVICE",
        "SAFETRACE_ENABLE_VLM",
        "SAFETRACE_VLM_ENABLED",
        "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED",
        "SAFETRACE_MOBILESAM_WORKER_ENABLED",
        "SAFETRACE_JOB_CONCURRENCY",
        "SAFETRACE_PER_BATCH_CONCURRENCY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SAFETRACE_RUNTIME_PROFILE", "test")
    test = Settings()
    assert test.runtime_profile == "test"
    assert test.device == "cpu"
    assert test.job_concurrency == 1

    monkeypatch.setenv("SAFETRACE_RUNTIME_PROFILE", "portable_fast_local")
    portable = Settings()
    assert portable.runtime_profile == "portable_fast_local"
    assert portable.job_concurrency == 1
    assert portable.vlm_enabled == "auto"
    assert portable.enable_vlm is False
    assert portable.lightweight_vlm_worker_enabled is False

    monkeypatch.setenv("SAFETRACE_RUNTIME_PROFILE", "local_full")
    full = Settings()
    assert full.runtime_profile == "local_full"
    assert full.device == "cuda"
    assert full.job_concurrency == 2
    assert full.per_batch_concurrency == 2
    assert full.mobile_sam_worker_enabled is True
    assert full.lightweight_vlm_worker_enabled is True
    assert full.require_gpu is True
    assert full.require_chat is True
    assert full.require_mobilesam is True
    assert full.allow_cpu_fallback is False


def test_oldest_batch_children_keep_priority_over_later_standalone(monkeypatch, tmp_path):
    store = JobStore(tmp_path / "jobs")
    first = _job(store, "batch-one.mp4", batch_id="batch-old", child=1)
    second = _job(store, "batch-two.mp4", batch_id="batch-old", child=2)
    third = _job(store, "batch-three.mp4", batch_id="batch-old", child=3)
    later = _job(store, "later-standalone.mp4")
    scheduler = LocalAnalysisScheduler()
    release = {record.job_id: threading.Event() for record in (first, second, third, later)}
    started: list[str] = []
    started_lock = threading.Lock()

    def fake_execute(job_store, job_id):
        job_store.update_status(
            job_id,
            status="running",
            progress=0.2,
            current_step="Fake real worker started",
        )
        with started_lock:
            started.append(job_id)
        assert release[job_id].wait(3)
        job_store.update_status(
            job_id,
            status="completed",
            progress=1.0,
            current_step="Fake real worker completed",
        )

    monkeypatch.setattr(jobs_module, "execute_analysis_job", fake_execute)
    monkeypatch.setattr(jobs_module.SETTINGS, "analysis_concurrency", 2)
    monkeypatch.setattr(jobs_module.SETTINGS, "job_concurrency", 2)
    monkeypatch.setattr(jobs_module.SETTINGS, "per_batch_concurrency", 2)

    scheduler.submit(store, [first.job_id, second.job_id, third.job_id])
    _wait_until(lambda: len(started) == 2)
    assert started == [first.job_id, second.job_id]
    assert scheduler.snapshot()["activeJobIds"] == [first.job_id, second.job_id]

    scheduler.submit(store, [later.job_id])
    assert later.job_id in scheduler.snapshot()["queuedJobIds"]
    assert store.require(third.job_id).status_payload()["scheduler"]["queuePosition"] == 1

    release[first.job_id].set()
    _wait_until(lambda: third.job_id in started)
    assert started[:3] == [first.job_id, second.job_id, third.job_id]
    assert later.job_id not in started

    for event in release.values():
        event.set()
    _wait_until(lambda: all(store.require(record.job_id).status == "completed" for record in (first, second, third, later)))


def test_runtime_setting_context_restores_global_values(monkeypatch):
    original = {
        name: getattr(jobs_module.SETTINGS, name)
        for name in ("device", "enable_vlm", "vlm_profile", "analysis_safe_mode")
    }
    monkeypatch.setattr(jobs_module.SETTINGS, "safe_mode_allow_mobilesam", True)

    with jobs_module._compatible_runtime_settings(
        device="cpu",
        enable_vlm=False,
        vlm_profile="rule_based",
        vlm_model_dir=None,
        safe_mode=False,
        review_mode="fast_local",
    ):
        assert jobs_module.SETTINGS.device == "cpu"
        assert jobs_module.SETTINGS.enable_vlm is False

    assert {name: getattr(jobs_module.SETTINGS, name) for name in original} == original


def test_batch_frontend_keeps_parent_context_and_renders_scheduler_truthfully():
    app_source = (ROOT / "frontend-react" / "src" / "App.tsx").read_text(encoding="utf-8")
    sidebar_source = (ROOT / "frontend-react" / "src" / "components" / "Sidebar.tsx").read_text(encoding="utf-8")

    assert "setBatchStatus(status)" in app_source
    assert "Queue ${file.queuePosition}" in app_source
    assert "Worker ${file.workerSlot}" in app_source
    assert "Requested: ${file.requestedModeLabel}" in app_source
    assert "setSelectedBatchJobId(representativeJobId)" not in app_source
    assert "Visual explanations: provider ready" in sidebar_source
    assert "Visual explanations: rule-based fallback" in sidebar_source
