from __future__ import annotations

import io
import zipfile

from fastapi.testclient import TestClient

import src.api.jobs as jobs_module
from src.api.batches import BatchStore
from src.api.jobs import JobStore
from src.api.server import create_app


def make_client(tmp_path):
    job_store = JobStore(tmp_path / "jobs")
    batch_store = BatchStore(tmp_path / "batches")
    app = create_app(job_store, batch_store)
    return TestClient(app), job_store, batch_store


def make_zip(files: dict[str, bytes]) -> bytes:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, content in files.items():
            archive.writestr(filename, content)
    return archive_bytes.getvalue()


def fake_pipeline(**kwargs):  # noqa: ARG001
    return []


def test_zip_batch_creates_manifest_and_one_job_per_video(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_module, "run_pipeline", fake_pipeline)
    client, _job_store, batch_store = make_client(tmp_path)
    archive = make_zip(
        {
            "camera-a.mp4": b"video-a",
            "nested/camera-b.mov": b"video-b",
        }
    )

    response = client.post(
        "/api/batches/analyze",
        files=[("files", ("footage.zip", archive, "application/zip"))],
        data={"query": "worker without helmet", "device": "cpu"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert len(body["acceptedFiles"]) == 2
    assert len(body["jobIds"]) == 2
    assert body["rejectedFiles"] == []

    manifest = batch_store.root_dir / body["batchId"] / "manifest.json"
    assert manifest.exists()

    status_response = client.get(f"/api/batches/{body['batchId']}")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["status"] == "completed"
    assert status["statusCounts"]["completed"] == 2

    for job_id in body["jobIds"]:
        result_response = client.get(f"/api/jobs/{job_id}/result")
        assert result_response.status_code == 200
        assert result_response.json()["query"] == "worker without helmet"


def test_zip_slip_archive_is_rejected_without_creating_jobs(tmp_path):
    client, job_store, _batch_store = make_client(tmp_path)
    archive = make_zip({"../escape.mp4": b"video"})

    response = client.post(
        "/api/batches/analyze",
        files=[("files", ("unsafe.zip", archive, "application/zip"))],
        data={"query": "worker without helmet"},
    )

    assert response.status_code == 400
    assert "unsafe path" in response.json()["detail"]["message"]
    assert job_store.status_counts(include_disk=True) == {}


def test_zip_unsupported_entries_are_reported_per_file(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_module, "run_pipeline", fake_pipeline)
    client, _job_store, _batch_store = make_client(tmp_path)
    archive = make_zip({"camera.mp4": b"video", "notes.txt": b"not media"})

    response = client.post(
        "/api/batches/analyze",
        files=[("files", ("mixed.zip", archive, "application/zip"))],
        data={"query": "worker without helmet"},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["acceptedFiles"]) == 1
    assert body["rejectedFiles"][0]["filename"] == "notes.txt"
    assert body["rejectedFiles"][0]["sourceRelativePath"] == "notes.txt"
    assert body["rejectedFiles"][0]["reason"] == "Unsupported file type for bulk video analysis."


def test_zip_file_count_limit_is_enforced(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_module.SETTINGS, "bulk_max_files", 1)
    client, _job_store, _batch_store = make_client(tmp_path)
    archive = make_zip({"a.mp4": b"a", "b.mp4": b"b"})

    response = client.post(
        "/api/batches/analyze",
        files=[("files", ("too-many.zip", archive, "application/zip"))],
        data={"query": "worker without helmet"},
    )

    assert response.status_code == 413
    assert "too many files" in response.json()["detail"]["message"]


def test_zip_uncompressed_size_limit_is_enforced(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_module.SETTINGS, "bulk_max_files", 25)
    monkeypatch.setattr(jobs_module.SETTINGS, "bulk_max_uncompressed_mb", 0.000001)
    client, _job_store, _batch_store = make_client(tmp_path)
    archive = make_zip({"large.mp4": b"too large"})

    response = client.post(
        "/api/batches/analyze",
        files=[("files", ("large.zip", archive, "application/zip"))],
        data={"query": "worker without helmet"},
    )

    assert response.status_code == 413
    assert "too large after extraction" in response.json()["detail"]["message"]


def test_direct_multi_file_batch_reports_rejected_files(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_module, "run_pipeline", fake_pipeline)
    client, _job_store, _batch_store = make_client(tmp_path)

    response = client.post(
        "/api/batches/analyze",
        files=[
            ("files", ("camera-a.mp4", b"video-a", "video/mp4")),
            ("files", ("notes.txt", b"not media", "text/plain")),
        ],
        data={"query": "worker without helmet"},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["acceptedFiles"]) == 1
    assert body["rejectedFiles"][0]["filename"] == "notes.txt"


def test_delete_batch_removes_batch_and_owned_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_module, "run_pipeline", fake_pipeline)
    client, _job_store, _batch_store = make_client(tmp_path)
    archive = make_zip({"camera.mp4": b"video"})
    created = client.post(
        "/api/batches/analyze",
        files=[("files", ("footage.zip", archive, "application/zip"))],
        data={"query": "worker without helmet"},
    ).json()

    response = client.delete(f"/api/batches/{created['batchId']}")

    assert response.status_code == 200
    assert client.get(f"/api/batches/{created['batchId']}").status_code == 404
    assert client.get(f"/api/jobs/{created['jobIds'][0]}").status_code == 404
