from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import src.api.jobs as jobs_module
from scripts.benchmark_safetrace_profiles import (
    _event_window_score,
    is_accuracy_eligible,
    load_manifest,
    main as benchmark_main,
    no_regression_gate,
    validate_manifest,
)
from scripts.run_overnight_soak import run_soak


def manifest_payload(media_path: str, *, label_status: str = "uncertain", annotation_status: str = "approved"):
    return {
        "schemaVersion": 1,
        "datasetId": "phase-o-test",
        "timestampToleranceSeconds": 2.0,
        "samples": [{
            "id": "sample-1",
            "mediaPath": media_path,
            "sourceRelativePath": "Vehicle A/sample.mp4",
            "vehicleGroup": "Vehicle A",
            "mediaType": "video",
            "profile": "seatbelt_compliance",
            "expectedFinding": "insufficient_visibility" if label_status == "uncertain" else "seatbelt_missing",
            "labelStatus": label_status,
            "startSeconds": 1.0,
            "endSeconds": 2.0,
            "humanLabelConfidence": 0.7,
            "annotatorId": "a",
            "annotationStatus": annotation_status,
            "reviewerId": "b",
            "reviewerStatus": "agrees",
        }],
    }


def test_manifest_accepts_uncertain_but_excludes_it_from_accuracy(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest_payload("missing.mp4")), encoding="utf-8")
    metadata, samples = load_manifest(path)
    assert metadata["datasetId"] == "phase-o-test"
    assert samples[0]["labelStatus"] == "uncertain"
    assert is_accuracy_eligible(samples[0]) is False


def test_manifest_validation_rejects_duplicate_ids_and_invalid_time():
    sample = manifest_payload("x.mp4")["samples"][0]
    duplicate = dict(sample)
    with pytest.raises(ValueError, match="Duplicate manifest id"):
        validate_manifest([sample, duplicate])
    invalid = {**sample, "id": "other", "startSeconds": 4.0, "endSeconds": 2.0}
    with pytest.raises(ValueError, match="endSeconds precedes"):
        validate_manifest([invalid])


def test_timestamp_tolerance_matches_overlapping_event():
    score = _event_window_score(
        [{"finding": "seatbelt_missing", "startSeconds": 10.0, "endSeconds": 12.0}],
        [{"type": "seatbelt_missing", "startTimestamp": "00:00:08.500", "endTimestamp": "00:00:09.000"}],
        tolerance=2.0,
    )
    assert score == {"eventTruePositives": 1, "eventFalseNegatives": 0, "eventFalsePositives": 0}


def test_benchmark_dry_run_and_resume_do_not_fabricate_accuracy(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_payload("missing.mp4", label_status="unlabelled", annotation_status="pending")), encoding="utf-8")
    output = tmp_path / "benchmark"
    args = ["--manifest", str(manifest), "--mode", "fast_local", "--device", "cpu", "--concurrency", "1", "--output-dir", str(output), "--dry-run"]
    assert benchmark_main(args) == 0
    first = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    assert first["accuracyMetricsAvailable"] is False
    assert first["accuracyGatePassed"] is False
    item = output / "items" / "fast_local_sample-1.json"
    before = item.read_text(encoding="utf-8")
    assert benchmark_main(args[:-1] + ["--resume", "--dry-run"]) == 0
    assert item.read_text(encoding="utf-8") == before


def test_no_regression_gate_blocks_missing_metrics_and_more_false_positives():
    assert no_regression_gate({}, {})["passed"] is False
    baseline = {
        "seatbelt_compliance": {"f1": 0.8, "falsePositives": 1},
        "helmet_ppe": {"f1": 0.8, "falsePositives": 1},
    }
    candidate = {
        "seatbelt_compliance": {"f1": 0.81, "falsePositives": 2},
        "helmet_ppe": {"f1": 0.81, "falsePositives": 1},
    }
    assert no_regression_gate(baseline, candidate)["passed"] is False


def soak_args(input_dir: Path, output_dir: Path, **overrides):
    values = {
        "input": input_dir,
        "output_dir": output_dir,
        "run_id": "phase-o-test",
        "duration_seconds": 0.0,
        "job_count": 2,
        "concurrency": 2,
        "mode": "fast_local",
        "device": "cpu",
        "restart_after": 1,
        "inject_transient_failure": False,
        "include_corrupt": False,
        "simulate_low_disk": False,
        "snapshot_seconds": 0.01,
        "resume": False,
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_soak_snapshot_restart_continuation_and_duplicate_prevention(monkeypatch, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "a.mp4").write_bytes(b"a")
    (inputs / "b.mp4").write_bytes(b"b")
    monkeypatch.setattr(jobs_module, "run_pipeline", lambda **_kwargs: [])
    report = run_soak(soak_args(inputs, tmp_path / "output"))
    assert report["passed"] is True
    assert report["jobStatusCounts"] == {"completed": 2}
    assert report["integrity"]["duplicateManifestIds"] == []
    assert list((tmp_path / "output" / "snapshots").glob("*.json"))


def test_soak_transient_retry_and_corrupt_quarantine(monkeypatch, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "valid.mp4").write_bytes(b"valid")
    calls = 0

    def pipeline(**kwargs):
        nonlocal calls
        calls += 1
        content = Path(kwargs["upload_path"]).read_bytes()
        if content == b"not-a-video":
            raise ValueError("corrupt media")
        if calls == 1:
            raise OSError("transient")
        return []

    monkeypatch.setattr(jobs_module, "run_pipeline", pipeline)
    report = run_soak(soak_args(
        inputs,
        tmp_path / "output",
        restart_after=0,
        include_corrupt=True,
        job_count=2,
    ))
    assert report["integrity"]["passed"] is True
    assert report["jobStatusCounts"]["completed"] == 1
    assert report["jobStatusCounts"]["failed"] == 1
    assert report["retryCount"] >= 1


def test_soak_low_disk_simulation_rejects_before_job_creation(monkeypatch, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "a.mp4").write_bytes(b"a")
    monkeypatch.setattr(jobs_module, "run_pipeline", lambda **_kwargs: [])
    report = run_soak(soak_args(inputs, tmp_path / "output", simulate_low_disk=True))
    assert report["status"] == "capacity_rejected"
    assert report["jobsCreated"] == 0
