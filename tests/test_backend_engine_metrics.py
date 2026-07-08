import json
from pathlib import Path

from scripts import benchmark_backend_engine, compare_backend_benchmarks
from src.engine_metrics import build_engine_metrics


def sample_benchmark_report(*, duration: float = 10.0, detector: float = 2.0, evidence: int = 2, names=None):
    names = ["Missing Helmet"] if names is None else names
    return {
        "schemaVersion": 1,
        "runs": [
            {
                "status": "completed",
                "engineMetrics": {
                    "totalDurationSeconds": duration,
                    "stageDurations": {"detectorInference": detector},
                    "counts": {"evidenceFrames": evidence},
                    "correctness": {
                        "status": "completed",
                        "resultSchemaKeys": ["jobId", "status", "summary", "frames", "technicalDetails"],
                        "violationNames": names,
                    },
                },
            }
        ],
    }


def test_engine_metrics_builder_preserves_schema_and_counts():
    result = {
        "jobId": "job_test",
        "status": "completed",
        "summary": {"framesAnalyzed": 1, "uniqueViolationTypes": 1},
        "violations": [{"name": "Missing Helmet", "severity": "high"}],
        "frames": [
            {
                "timestamp": "00:00:01",
                "imageUrl": "/api/media/job/frame.jpg",
                "violations": [{"name": "Missing Helmet"}],
                "technicalEvidence": {"detections": [{"label": "person"}]},
            }
        ],
        "technicalDetails": {
            "pipelineWallClockSeconds": 1.25,
            "processingMetadata": {"sampledFrameCount": 3},
            "componentDiagnostics": {
                "device": "cpu",
                "stageTimings": {
                    "safe_frame_sampling": 0.1,
                    "detector_load": 0.2,
                    "safe_ranking_detector_inference": 0.3,
                    "safe_ranking_rule_evaluation": 0.05,
                },
            },
        },
    }

    metrics = build_engine_metrics(result, {}, resource_snapshot={"cpuProcessTimeSeconds": 0.4})

    assert metrics["totalDurationSeconds"] == 1.25
    assert metrics["stageDurations"]["detectorInference"] == 0.3
    assert metrics["stageDurations"]["ruleEvaluation"] == 0.05
    assert metrics["counts"]["sampledFrames"] == 3
    assert metrics["counts"]["analyzedFrames"] == 1
    assert metrics["counts"]["detections"] == 1
    assert metrics["correctness"]["violationNames"] == ["Missing Helmet"]
    assert "technicalDetails" in metrics["correctness"]["resultSchemaKeys"]


def test_backend_benchmark_no_input_prints_usage(capsys):
    exit_code = benchmark_backend_engine.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Benchmark the SafeTrace backend engine" in captured.out
    assert "--input" in captured.out


def test_pipeline_reuses_process_local_detector_cache(monkeypatch):
    import src.pipeline as pipeline_module

    pipeline_module.clear_yolo_detector_cache()

    class FakeDetector:
        created = 0

        def __init__(self):
            type(self).created += 1
            self.checkpoint = Path("fake-yolo.pt")

        def detect(self, image):  # noqa: ARG002
            return []

    monkeypatch.setattr(pipeline_module, "YoloDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "safe_mode_allow_mobilesam", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_enabled", "false")
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_enabled", "false")
    monkeypatch.setattr(pipeline_module.SETTINGS, "device", "cpu")
    monkeypatch.setattr(pipeline_module.SETTINGS, "yolo_checkpoint", Path("missing-yolov9.pt"))
    monkeypatch.setattr(pipeline_module.SETTINGS, "yolo_fallback_checkpoint", Path("missing-yolov8.pt"))

    first = pipeline_module.SafeTracePipeline()
    second = pipeline_module.SafeTracePipeline()

    assert FakeDetector.created == 1
    assert first.component_diagnostics["detectorCacheHit"] is False
    assert second.component_diagnostics["detectorCacheHit"] is True
    pipeline_module.clear_yolo_detector_cache()


def test_pipeline_evicts_invalid_detector_cache_entry(monkeypatch):
    import src.pipeline as pipeline_module

    pipeline_module.clear_yolo_detector_cache()

    class FakeDetector:
        created = 0

        def __init__(self):
            type(self).created += 1
            self.checkpoint = Path("fake-yolo.pt")

        def detect(self, image):  # noqa: ARG002
            return []

    monkeypatch.setattr(pipeline_module, "YoloDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module.SETTINGS, "device", "cpu")
    monkeypatch.setattr(pipeline_module.SETTINGS, "yolo_checkpoint", Path("missing-yolov9.pt"))
    monkeypatch.setattr(pipeline_module.SETTINGS, "yolo_fallback_checkpoint", Path("missing-yolov8.pt"))

    key = pipeline_module._detector_cache_key()
    pipeline_module._DETECTOR_CACHE[key] = object()

    detector, cache_hit = pipeline_module._cached_yolo_detector()

    assert cache_hit is False
    assert isinstance(detector, FakeDetector)
    assert callable(detector.detect)
    assert FakeDetector.created == 1
    pipeline_module.clear_yolo_detector_cache()


def test_pipeline_detector_cache_respects_monkeypatched_detector_factory(monkeypatch):
    import src.pipeline as pipeline_module

    pipeline_module.clear_yolo_detector_cache()

    class FirstDetector:
        created = 0
        checkpoint = Path("first-yolo.pt")

        def __init__(self):
            type(self).created += 1

        def detect(self, image):  # noqa: ARG002
            return []

    class SecondDetector:
        created = 0
        checkpoint = Path("second-yolo.pt")

        def __init__(self):
            type(self).created += 1

        def detect(self, image):  # noqa: ARG002
            return []

    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "safe_mode_allow_mobilesam", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_enabled", "false")
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_enabled", "false")
    monkeypatch.setattr(pipeline_module.SETTINGS, "device", "cpu")
    monkeypatch.setattr(pipeline_module.SETTINGS, "yolo_checkpoint", Path("missing-yolov9.pt"))
    monkeypatch.setattr(pipeline_module.SETTINGS, "yolo_fallback_checkpoint", Path("missing-yolov8.pt"))

    monkeypatch.setattr(pipeline_module, "YoloDetector", FirstDetector)
    first = pipeline_module.SafeTracePipeline()

    monkeypatch.setattr(pipeline_module, "YoloDetector", SecondDetector)
    second = pipeline_module.SafeTracePipeline()
    third = pipeline_module.SafeTracePipeline()

    assert isinstance(first.detector, FirstDetector)
    assert isinstance(second.detector, SecondDetector)
    assert isinstance(third.detector, SecondDetector)
    assert FirstDetector.created == 1
    assert SecondDetector.created == 1
    assert first.component_diagnostics["detectorCacheHit"] is False
    assert second.component_diagnostics["detectorCacheHit"] is False
    assert third.component_diagnostics["detectorCacheHit"] is True
    pipeline_module.clear_yolo_detector_cache()


def test_backend_benchmark_dry_run_writes_json_csv_and_markdown(tmp_path):
    output = tmp_path / "benchmark"

    exit_code = benchmark_backend_engine.main(["--dry-run", "--mode", "rule_based", "--output", str(output)])

    assert exit_code == 0
    report = json.loads((output / "backend_benchmark_report.json").read_text(encoding="utf-8"))
    assert report["schemaVersion"] == 1
    assert report["dryRun"] is True
    assert report["runs"][0]["engineMetrics"]["counts"]["sampledFrames"] == 1
    assert (output / "backend_benchmark_summary.csv").is_file()
    assert (output / "backend_benchmark_summary.md").is_file()


def test_compare_backend_benchmarks_passes_within_tolerance(tmp_path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(sample_benchmark_report(duration=10.0)), encoding="utf-8")
    after.write_text(json.dumps(sample_benchmark_report(duration=10.5)), encoding="utf-8")

    exit_code = compare_backend_benchmarks.main(
        ["--before", str(before), "--after", str(after), "--output", str(tmp_path / "comparison")]
    )

    assert exit_code == 0
    payload = json.loads((tmp_path / "comparison" / "backend_benchmark_comparison.json").read_text(encoding="utf-8"))
    assert payload["passed"] is True


def test_compare_backend_benchmarks_flags_performance_and_logic_regressions(tmp_path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(sample_benchmark_report(duration=10.0, evidence=4)), encoding="utf-8")
    after.write_text(
        json.dumps(sample_benchmark_report(duration=13.0, detector=4.0, evidence=1, names=["Missing Seatbelt"])),
        encoding="utf-8",
    )

    exit_code = compare_backend_benchmarks.main(
        ["--before", str(before), "--after", str(after), "--output", str(tmp_path / "comparison")]
    )

    assert exit_code == 1
    payload = json.loads((tmp_path / "comparison" / "backend_benchmark_comparison.json").read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert {item["type"] for item in payload["performanceRegressions"]} == {
        "total_duration_regression",
        "detector_inference_regression",
    }
    assert {item["type"] for item in payload["logicRegressions"]} == {
        "evidence_count_drop",
        "violation_names_changed",
    }
