from __future__ import annotations

import io
import json
import time
import threading
import zipfile
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

import src.api.jobs as jobs_module
from scripts.benchmark_safetrace_profiles import no_regression_gate
from src.api.batches import BatchStore
from src.api.jobs import AnalysisSettings, JobStore
from src.api.normalization import normalize_pipeline_results
from src.api.server import create_app


def settings(review_mode: str = "fast_local") -> AnalysisSettings:
    return AnalysisSettings(
        fps=1.0,
        top_k=5,
        enable_vlm=False,
        device="cpu",
        review_mode=review_mode,
    )


def zipped(files: list[tuple[str, bytes]]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files:
            archive.writestr(name, content)
    return payload.getvalue()


def test_nested_zip_preserves_hierarchy_checksum_and_partial_acceptance(tmp_path):
    jobs = JobStore(tmp_path / "jobs")
    batches = BatchStore(tmp_path / "batches")
    batch = batches.create_from_zip(
        filename="overnight.zip",
        content=zipped(
            [
                ("Vehicle A/route-001.mp4", b"video-a"),
                ("Vehicle B/Shift 1/route-010.mp4", b"video-b"),
                ("Vehicle A/notes.txt", b"bad"),
            ]
        ),
        query="general safety",
        settings=settings(),
        job_store=jobs,
    )

    payload = batch.payload()
    assert [item["sourceRelativePath"] for item in payload["acceptedFiles"]] == [
        "Vehicle A/route-001.mp4",
        "Vehicle B/Shift 1/route-010.mp4",
    ]
    assert all(len(item["checksumSha256"]) == 64 for item in payload["acceptedFiles"])
    assert payload["rejectedFiles"][0]["sourceRelativePath"] == "Vehicle A/notes.txt"
    assert {item["sourceGroupPath"] for item in payload["groupSummaries"]} == {
        "Vehicle A",
        "Vehicle B/Shift 1",
    }
    assert payload["hierarchy"]["children"][0]["name"] == "Vehicle A"


def test_directory_upload_contract_preserves_same_leaf_under_different_parents(tmp_path):
    client = TestClient(create_app(JobStore(tmp_path / "jobs"), BatchStore(tmp_path / "batches")))
    response = client.post(
        "/api/batches/analyze",
        files=[
            ("files", ("a.mp4", b"a", "video/mp4")),
            ("files", ("b.mp4", b"b", "video/mp4")),
        ],
        data={
            "query": "general safety",
            "relativePaths": ["Depot A/Shift 1/a.mp4", "Depot B/Shift 1/b.mp4"],
            "reviewMode": "fast_local",
        },
    )
    assert response.status_code == 200
    groups = {item["sourceGroupPath"] for item in response.json()["groupSummaries"]}
    assert groups == {"Depot A/Shift 1", "Depot B/Shift 1"}


def test_streamed_job_creation_reads_bounded_chunks(tmp_path):
    class RecordingStream(io.BytesIO):
        def __init__(self, content: bytes):
            super().__init__(content)
            self.requested = []

        def read(self, size=-1):
            self.requested.append(size)
            return super().read(size)

    stream = RecordingStream(b"x" * (2 * 1024 * 1024 + 7))
    record = JobStore(tmp_path / "jobs").create_job_from_stream(
        filename="large.mp4",
        stream=stream,
        query="general safety",
        settings=settings(),
    )
    assert record.size_bytes == 2 * 1024 * 1024 + 7
    assert stream.requested and set(stream.requested) == {1024 * 1024}


def test_explicit_subsecond_timestamp_survives_normalization(monkeypatch, tmp_path):
    from src.api.normalization import SETTINGS as normalization_settings

    monkeypatch.setattr(normalization_settings, "diagnostic_frames_enabled", True)
    result = normalize_pipeline_results(
        job_id="job_trace",
        media_name="shift.mp4",
        media_type="video",
        media_size_bytes=10,
        query="general safety",
        raw_frames=[{
            "frame_id": "shift_000001",
            "frame_path": "frames/shift_000001.jpg",
            "timestamp_seconds": 0.5,
            "source_frame_index": 15,
            "score": 0.8,
            "violations": [],
        }],
        media_dir=tmp_path / "media",
        register_media=lambda *_args: None,
    )
    assert result["frames"] == []
    assert result["diagnosticFrames"][0]["timestamp"] == "00:00:00.500"
    assert result["diagnosticFrames"][0]["timestampSeconds"] == 0.5
    assert result["diagnosticFrames"][0]["sourceFrameIndex"] == 15


def test_import_key_is_idempotent_for_repeated_batch_request(tmp_path):
    jobs = JobStore(tmp_path / "jobs")
    batches = BatchStore(tmp_path / "batches")
    first = batches.create_from_files(
        files=[("Depot A/camera.mp4", b"video")], source_filename="overnight", query="safety",
        settings=settings(), job_store=jobs, import_key="import-123",
    )
    second = batches.create_from_files(
        files=[("Depot A/camera.mp4", b"video")], source_filename="overnight", query="safety",
        settings=settings(), job_store=jobs, import_key="import-123",
    )
    assert second.batch_id == first.batch_id
    assert second.job_ids == first.job_ids


def test_secondary_review_is_nonauthoritative_and_auditable(tmp_path):
    jobs = JobStore(tmp_path / "jobs")
    record = jobs.create_job(filename="camera.mp4", content=b"video", query="safety", settings=settings())
    jobs.complete_job(record.job_id, {"jobId": record.job_id, "status": "completed", "technicalDetails": {}})
    client = TestClient(create_app(jobs, BatchStore(tmp_path / "batches")))
    response = client.post(
        f"/api/jobs/{record.job_id}/secondary-reviews",
        json={
            "provider": "external-review-provider",
            "model": "review-model-v1",
            "evidenceFrameIds": ["frame_1"],
            "attempted": True,
            "succeeded": True,
            "accepted": True,
            "explanation": "Head visibility is unclear.",
            "agreement": "inconclusive",
            "latencySeconds": 1.2,
        },
    )
    assert response.status_code == 200
    assert response.json()["authoritative"] is False
    technical = client.get(f"/api/reports/{record.job_id}/technical-json").json()
    reviews = technical["technicalDetails"]["secondaryReviews"]
    assert reviews[0]["provider"] == "external-review-provider"
    assert reviews[0]["authoritative"] is False


def test_batch_restart_recovers_unfinished_and_preserves_completed(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_module.SETTINGS, "stale_running_minutes", 1.0)
    root = tmp_path / "jobs"
    jobs = JobStore(root)
    batch = BatchStore(tmp_path / "batches").create_from_files(
        files=[("Vehicle A/done.mp4", b"done"), ("Vehicle A/recover.mp4", b"recover")],
        source_filename="overnight",
        query="general safety",
        settings=settings(),
        job_store=jobs,
    )
    done = jobs.require(batch.job_ids[0])
    jobs.complete_job(done.job_id, {"jobId": done.job_id, "status": "completed", "technicalDetails": {}})
    recovering = jobs.require(batch.job_ids[1])
    jobs.update_status(recovering.job_id, status="running", progress=0.4, current_step="Analyzing")
    recovering.updated_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    jobs.persist_job(recovering)

    restarted = JobStore(root)
    assert restarted.require(done.job_id).status == "completed"
    assert restarted.require(done.job_id).result is not None
    assert restarted.require(recovering.job_id).status == "recovering"
    assert restarted.require(recovering.job_id).recovery_action == "requeue_after_restart"
    assert not (restarted.require(recovering.job_id).job_dir / "execution.lock").exists()


def test_batch_controls_retry_failed_and_pause_resume(tmp_path):
    jobs = JobStore(tmp_path / "jobs")
    batches = BatchStore(tmp_path / "batches")
    batch = batches.create_from_files(
        files=[("Vehicle A/a.mp4", b"a")],
        source_filename="batch",
        query="general safety",
        settings=settings(),
        job_store=jobs,
    )
    job_id = batch.job_ids[0]
    assert batches.pause(batch.batch_id, jobs)["pausedJobIds"] == [job_id]
    assert jobs.require(job_id).status == "paused"
    assert batches.resume(batch.batch_id, jobs)["resumedJobIds"] == [job_id]
    jobs.update_status(job_id, status="failed", progress=1.0, current_step="Failed", error="bad")
    assert batches.retry_failed(batch.batch_id, jobs)["retriedJobIds"] == [job_id]
    assert jobs.require(job_id).status == "queued"


def test_batch_dispatch_uses_bounded_parallel_workers(monkeypatch, tmp_path):
    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_pipeline(**_kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.5)
        with lock:
            active -= 1
        return []

    monkeypatch.setattr(jobs_module, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(jobs_module.SETTINGS, "analysis_concurrency", 2)
    monkeypatch.setattr(jobs_module.SETTINGS, "batch_max_active_jobs", 2)
    client = TestClient(create_app(JobStore(tmp_path / "jobs"), BatchStore(tmp_path / "batches")))
    response = client.post(
        "/api/batches/analyze",
        files=[
            ("files", ("a.mp4", b"a", "video/mp4")),
            ("files", ("b.mp4", b"b", "video/mp4")),
        ],
        # This is an analysis-worker bound fake pipeline test.  Make the
        # request explicitly CPU-bound so a physical CUDA GPU does not turn it
        # into a GPU-lane contention test.
        data={"query": "general safety", "relativePaths": ["A/a.mp4", "B/b.mp4"], "device": "cpu"},
    )
    assert response.status_code == 200
    assert peak == 2
    assert client.get(f"/api/batches/{response.json()['batchId']}").json()["status"] == "completed"


def test_comprehensive_review_expands_sampling_without_changing_fast_defaults(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    snapshots = []

    class FakePipeline:
        component_diagnostics = {}

        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def run(self, paths, *, query, fps, k):  # noqa: ARG002
            snapshots.append((fps, k, jobs_module.SETTINGS.max_frames, jobs_module.SETTINGS.vlm_max_evidence_frames))
            return []

    monkeypatch.setattr(pipeline_module, "SafeTracePipeline", FakePipeline)
    fast_defaults = (jobs_module.SETTINGS.max_frames, jobs_module.SETTINGS.vlm_max_evidence_frames)
    jobs_module.run_pipeline(
        upload_path=tmp_path / "a.mp4", query="safety", fps=1.0, top_k=5, device="cpu",
        enable_vlm=False, review_mode="fast_local",
    )
    jobs_module.run_pipeline(
        upload_path=tmp_path / "a.mp4", query="safety", fps=1.0, top_k=5, device="cpu",
        enable_vlm=False, review_mode="comprehensive",
    )
    assert snapshots[1][0] > snapshots[0][0]
    assert snapshots[1][1] > snapshots[0][1]
    assert snapshots[1][2] > snapshots[0][2]
    assert (jobs_module.SETTINGS.max_frames, jobs_module.SETTINGS.vlm_max_evidence_frames) == fast_defaults


def test_no_regression_gate_requires_real_seatbelt_and_helmet_metrics():
    unavailable = no_regression_gate({}, {})
    assert unavailable["passed"] is False
    baseline = {"seatbelt_compliance": {"f1": 0.8}, "helmet_ppe": {"f1": 0.7}}
    candidate = {"seatbelt_compliance": {"f1": 0.79}, "helmet_ppe": {"f1": 0.71}}
    assert no_regression_gate(baseline, candidate, tolerance=0.02)["passed"] is True
