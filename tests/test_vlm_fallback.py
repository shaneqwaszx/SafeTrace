import json
import subprocess

import httpx
import numpy as np
from pathlib import Path

import src.vlm_reasoner as vlm_reasoner
from src.safe_frame_ranking import parse_query_intent, score_frame_for_safe_mode, select_ranked_frames
from src.schemas import Detection, Violation
from src.vlm_reasoner import VLM_PROMPT, VlmReasoner, is_local_vlm_base_url, is_useful_vlm_output, sanitize_vlm_output


class FakeGenerateResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class NoopDetector:
    checkpoint = "checkpoints/yolov8s-seg.pt"

    def detect(self, image):  # noqa: ARG002
        return []


class FakeProcessorBatch(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeChatTemplateProcessor:
    def __init__(self, decoded_text="Visible helmet evidence is uncertain due to glare."):
        self.decoded_text = decoded_text
        self.messages = None
        self.template_kwargs = None
        self.call_kwargs = None
        self.decode_output_ids = None
        self.decode_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "USER: <image>\nDescribe visible SafeTrace evidence.\nASSISTANT:"

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return FakeProcessorBatch({"input_ids": [1, 2, 3]})

    def batch_decode(self, output_ids, **kwargs):
        self.decode_output_ids = output_ids
        self.decode_kwargs = kwargs
        return [self.decoded_text]


class FakeNoTemplateProcessor(FakeChatTemplateProcessor):
    apply_chat_template = None


class FakeLocalModel:
    def __init__(self, fail=False):
        self.fail = fail
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        if self.fail:
            raise RuntimeError("image token mismatch")
        return [[1, 2, 3, 4]]


def make_local_reasoner(processor, model):
    reasoner = VlmReasoner.__new__(VlmReasoner)
    reasoner.device = "cpu"
    reasoner.provider = "local"
    reasoner.enabled = True
    reasoner._loaded = True
    reasoner._processor = processor
    reasoner._model = model
    reasoner.last_explanation_source = "rule_based"
    return reasoner


def make_violation():
    return Violation(
        name="helmet_missing",
        description="Worker head detected without overlapping helmet.",
        severity="high",
        confidence=0.9,
    )


def make_detection(label, bbox=(10, 10, 90, 90), confidence=0.9):
    mask = np.zeros((100, 100), dtype=bool)
    x1, y1, x2, y2 = [int(value) for value in bbox]
    mask[y1:y2, x1:x2] = True
    return Detection(
        label=label,
        raw_label=label,
        confidence=confidence,
        bbox=[float(value) for value in bbox],
        coarse_mask=mask,
    )


def test_rule_engine_calibrates_person_only_helmet_and_seatbelt_as_review_candidate():
    from src.rule_engine import evaluate

    violations = evaluate([make_detection("person")])
    by_name = {violation.name: violation for violation in violations}

    assert by_name["helmet_missing"].confidence < 0.6
    assert by_name["helmet_missing"].severity == "medium"
    assert by_name["helmet_missing"].evidence["evidenceStrength"] == "review_candidate"
    assert by_name["helmet_missing"].evidence["ruleSupport"] == "person_proxy_only"
    assert by_name["seatbelt_missing"].confidence < 0.6
    assert by_name["seatbelt_missing"].severity == "medium"
    assert by_name["seatbelt_missing"].evidence["evidenceStrength"] == "review_candidate"
    assert by_name["seatbelt_missing"].evidence["ruleSupport"] == "person_proxy_only"


def test_rule_engine_generic_coco_objects_do_not_create_confidence_one_ppe_or_seatbelt():
    from src.rule_engine import evaluate

    violations = evaluate(
        [
            make_detection("person", confidence=0.97),
            make_detection("suitcase", bbox=(5, 60, 40, 95), confidence=0.82),
            make_detection("laptop", bbox=(55, 60, 95, 90), confidence=0.76),
            make_detection("skis", bbox=(10, 5, 95, 15), confidence=0.72),
        ]
    )
    by_name = {violation.name: violation for violation in violations}

    assert by_name["helmet_missing"].confidence < 0.6
    assert by_name["helmet_missing"].evidence["evidenceStrength"] == "review_candidate"
    assert by_name["seatbelt_missing"].confidence < 0.6
    assert by_name["seatbelt_missing"].evidence["evidenceStrength"] == "review_candidate"


def test_rule_engine_direct_torso_context_stays_likely_but_not_confidence_one():
    from src.rule_engine import evaluate

    violations = evaluate([make_detection("person"), make_detection("torso", bbox=(20, 25, 80, 90))])
    seatbelt = next(violation for violation in violations if violation.name == "seatbelt_missing")

    assert 0.6 <= seatbelt.confidence < 0.8
    assert seatbelt.evidence["evidenceStrength"] == "likely_violation"
    assert seatbelt.evidence["ruleSupport"] == "torso_or_cabin_context"


def test_local_vlm_frame_limit_defaults_to_five_and_disables_shared_job_budget(monkeypatch):
    from src.config import Settings

    for key in (
        "SAFETRACE_VLM_FRAME_LIMIT",
        "SAFETRACE_VLM_MAX_EVIDENCE_FRAMES",
        "SAFETRACE_VLM_MAX_FRAMES",
        "SAFETRACE_VLM_JOB_TIMEOUT_SECONDS",
        "SAFETRACE_LIGHTWEIGHT_VLM_TOTAL_BUDGET_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings()

    assert settings.vlm_max_frames == 5
    assert settings.vlm_max_evidence_frames == 5
    assert settings.vlm_job_timeout_seconds == 0.0
    assert settings.lightweight_vlm_total_budget_seconds == 0.0


def output_path_from_worker_command(command):
    return Path(command[command.index("--output-json") + 1])


def input_path_from_worker_command(command):
    return Path(command[command.index("--input-json") + 1])


def test_vlm_rejects_non_local_base_url():
    assert is_local_vlm_base_url("http://127.0.0.1:11434") is True
    assert is_local_vlm_base_url("http://localhost:11434") is True
    assert is_local_vlm_base_url("https://example.com") is False


def test_disabled_vlm_returns_rule_based_fallback(monkeypatch):
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_enabled", "disabled")
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "enable_vlm", True)

    reasoner = VlmReasoner(device="cpu", enabled=True)
    text = reasoner.explain_violation(np.zeros((4, 4, 3), dtype=np.uint8), [make_violation()])

    assert reasoner.last_explanation_source == "rule_based"
    assert "Rule-based explanation" in text
    assert "helmet_missing" in text


def test_pipeline_rule_based_mode_does_not_construct_vlm_reasoner(monkeypatch):
    import src.pipeline as pipeline_module

    class FakeIndex:
        def __init__(self, embedder):  # noqa: ARG002
            pass

    def fail_if_vlm_constructed(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("Rule-based analysis should not construct VlmReasoner")

    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_enabled", "auto")
    monkeypatch.setattr(pipeline_module, "ClipEmbedder", lambda: object())
    monkeypatch.setattr(pipeline_module, "FaissIndex", FakeIndex)
    monkeypatch.setattr(pipeline_module, "YoloDetector", NoopDetector)
    monkeypatch.setattr(pipeline_module, "MobileSamSegmenter", lambda: object())
    monkeypatch.setattr(pipeline_module, "VlmReasoner", fail_if_vlm_constructed)

    pipeline = pipeline_module.SafeTracePipeline()

    assert pipeline.vlm.provider == "rule_based"


def test_pipeline_fast_mode_does_not_construct_mobile_sam_by_default(monkeypatch):
    import src.pipeline as pipeline_module

    class FakeIndex:
        def __init__(self, embedder):  # noqa: ARG002
            pass

    def fail_if_mobile_sam_constructed(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("Fast rule-based analysis should not construct MobileSAM")

    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_enabled", "auto")
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_enabled", "disabled")
    monkeypatch.setattr(pipeline_module, "ClipEmbedder", lambda: object())
    monkeypatch.setattr(pipeline_module, "FaissIndex", FakeIndex)
    monkeypatch.setattr(pipeline_module, "YoloDetector", NoopDetector)
    monkeypatch.setattr(pipeline_module, "MobileSamSegmenter", fail_if_mobile_sam_constructed)

    pipeline = pipeline_module.SafeTracePipeline()

    assert pipeline.segmenter.available is False


def test_pipeline_safe_mode_skips_embeddings_vlm_and_mobile_sam(monkeypatch):
    import src.pipeline as pipeline_module

    def fail_if_called(component):
        def inner(*args, **kwargs):  # noqa: ARG001
            raise AssertionError(f"safe mode should not construct {component}")
        return inner

    class FakeDetector:
        checkpoint = "checkpoints/yolov8s-seg.pt"

        def detect(self, image):  # noqa: ARG002
            return []

    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "safe_mode_allow_mobilesam", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "lightweight_256m")
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_enabled", "auto")
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_worker_enabled", False)
    monkeypatch.setattr(pipeline_module, "ClipEmbedder", fail_if_called("ClipEmbedder"))
    monkeypatch.setattr(pipeline_module, "FaissIndex", fail_if_called("FaissIndex"))
    monkeypatch.setattr(pipeline_module, "MobileSamSegmenter", fail_if_called("MobileSamSegmenter"))
    monkeypatch.setattr(pipeline_module, "VlmReasoner", fail_if_called("VlmReasoner"))
    monkeypatch.setattr(pipeline_module, "YoloDetector", FakeDetector)

    pipeline = pipeline_module.SafeTracePipeline()

    assert pipeline.safe_mode is True
    assert pipeline.embedder is None
    assert pipeline.index is None
    assert pipeline.segmenter.available is False
    assert pipeline.vlm.provider == "rule_based"
    assert pipeline.component_diagnostics["embeddingRequested"] is False
    assert pipeline.component_diagnostics["vlmLoaded"] is False
    assert pipeline.component_diagnostics["mobileSamLoaded"] is False


def test_pipeline_safe_mode_allows_mobile_sam_after_frame_selection(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    constructed = {"mobile_sam": 0, "refined": 0}

    class FakeDetector:
        checkpoint = "checkpoints/yolov8s-seg.pt"

        def detect(self, image):  # noqa: ARG002
            return [Detection(label="person", raw_label="person", confidence=0.9, bbox=[0, 0, 10, 10])]

    class FakeMobileSam:
        available = True

        def __init__(self, *args, **kwargs):  # noqa: ARG002
            constructed["mobile_sam"] += 1

        def refine(self, image, detections):  # noqa: ARG002
            constructed["refined"] += 1
            return detections

    frames = [tmp_path / f"frame_{index:06d}.jpg" for index in range(3)]
    for frame in frames:
        frame.write_bytes(b"placeholder")

    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "safe_mode_allow_mobilesam", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_enabled", "true")
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_worker_enabled", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "top_k", 1)
    monkeypatch.setattr(pipeline_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline_module, "ClipEmbedder", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ClipEmbedder should not load")))
    monkeypatch.setattr(pipeline_module, "FaissIndex", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("FaissIndex should not load")))
    monkeypatch.setattr(pipeline_module, "VlmReasoner", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("VLM should not load")))
    monkeypatch.setattr(pipeline_module, "YoloDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "MobileSamSegmenter", FakeMobileSam)
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((32, 32, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "imwrite_rgb", lambda path, image: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline_module, "evaluate_rules", lambda detections: [make_violation()])
    monkeypatch.setattr(
        pipeline_module.SafeTracePipeline,
        "_collect_input_frames",
        lambda self, inputs, fps, max_frames, **kwargs: (frames, [], {"videos": 0, "images": len(frames)}),  # noqa: ARG005
    )

    pipeline = pipeline_module.SafeTracePipeline()

    assert constructed["mobile_sam"] == 0

    result = pipeline.run([frames[0]], query="driver without seatbelt", fps=1.0, k=1)

    assert result
    assert constructed["mobile_sam"] == 1
    assert constructed["refined"] == 1
    assert pipeline.component_diagnostics["safeMode"] is True
    assert pipeline.component_diagnostics["safeModeMobileSamAllowed"] is True
    assert pipeline.component_diagnostics["embeddingRequested"] is False
    assert pipeline.component_diagnostics["vlmLoaded"] is False
    assert pipeline.component_diagnostics["mobileSamAttempted"] is True
    assert pipeline.component_diagnostics["mobileSamLoaded"] is True
    assert pipeline.component_diagnostics["effectiveExplanationMode"] == "rule_based_with_mobilesam"


def test_pipeline_safe_mode_mobile_sam_failure_uses_detector_box_fallback(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    class FakeDetector:
        checkpoint = "checkpoints/yolov8s-seg.pt"

        def detect(self, image):  # noqa: ARG002
            return [Detection(label="person", raw_label="person", confidence=0.9, bbox=[0, 0, 10, 10])]

    class FailingMobileSam:
        available = True

        def __init__(self, *args, **kwargs):  # noqa: ARG002
            pass

        def refine(self, image, detections):  # noqa: ARG002
            raise RuntimeError("refine timeout")

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"placeholder")

    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "safe_mode_allow_mobilesam", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_enabled", "true")
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_worker_enabled", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline_module, "ClipEmbedder", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ClipEmbedder should not load")))
    monkeypatch.setattr(pipeline_module, "FaissIndex", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("FaissIndex should not load")))
    monkeypatch.setattr(pipeline_module, "VlmReasoner", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("VLM should not load")))
    monkeypatch.setattr(pipeline_module, "YoloDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "MobileSamSegmenter", FailingMobileSam)
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((32, 32, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "imwrite_rgb", lambda path, image: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline_module, "evaluate_rules", lambda detections: [make_violation()])
    monkeypatch.setattr(
        pipeline_module.SafeTracePipeline,
        "_collect_input_frames",
        lambda self, inputs, fps, max_frames, **kwargs: ([frame], [], {"videos": 0, "images": 1}),  # noqa: ARG005
    )

    pipeline = pipeline_module.SafeTracePipeline()
    result = pipeline.run([frame], query="driver without seatbelt", fps=1.0, k=1)

    assert result
    assert pipeline.component_diagnostics["mobileSamAttempted"] is True
    assert pipeline.component_diagnostics["mobileSamLoaded"] is False
    assert pipeline.component_diagnostics["mobileSamFallbackReason"] == "RuntimeError"


def test_mobile_sam_worker_disabled_does_not_launch_subprocess(monkeypatch, tmp_path):
    import src.mobile_sam_worker_client as client_module

    checkpoint = tmp_path / "mobile_sam.pt"
    checkpoint.write_bytes(b"checkpoint")

    def fail_if_run(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("worker subprocess should not launch when disabled")

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "mobile_sam_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "mobile_sam_worker_enabled", False)
    monkeypatch.setattr(client_module.subprocess, "run", fail_if_run)

    detection = make_detection("person")
    segmenter = client_module.MobileSamWorkerSegmenter(checkpoint=checkpoint)
    refined = segmenter.refine(tmp_path / "frame.jpg", [detection])

    assert refined[0].refined_mask is not None
    assert segmenter.last_diagnostics["mobileSamWorkerAttempted"] is False
    assert segmenter.last_diagnostics["mobileSamRefinementSource"] == "disabled"


def test_mobile_sam_worker_success_result_is_consumed(monkeypatch, tmp_path):
    import src.mobile_sam_worker_client as client_module
    from src.mask_encoding import encode_bool_mask

    checkpoint = tmp_path / "mobile_sam.pt"
    checkpoint.write_bytes(b"checkpoint")
    mask = np.zeros((100, 100), dtype=bool)
    mask[5:20, 5:20] = True

    def fake_run(command, **kwargs):  # noqa: ARG001
        output_path = output_path_from_worker_command(command)
        output_path.write_text(
            client_module.json.dumps(
                {"ok": True, "detections": [{"index": 0, "hasRefinedMask": True, "refinedMask": encode_bool_mask(mask)}]}
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "mobile_sam_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "mobile_sam_worker_enabled", True)
    monkeypatch.setattr(client_module.SETTINGS, "mobile_sam_worker_timeout_seconds", 7)
    monkeypatch.setattr(client_module.subprocess, "run", fake_run)

    detection = make_detection("person")
    segmenter = client_module.MobileSamWorkerSegmenter(checkpoint=checkpoint)
    refined = segmenter.refine(tmp_path / "frame.jpg", [detection])

    assert bool(refined[0].refined_mask[6, 6]) is True
    assert segmenter.last_diagnostics["mobileSamWorkerAttempted"] is True
    assert segmenter.last_diagnostics["mobileSamWorkerSucceeded"] is True
    assert segmenter.last_diagnostics["mobileSamWorkerTimedOut"] is False
    assert segmenter.last_diagnostics["mobileSamRefinementSource"] == "worker"


def test_mobile_sam_worker_timeout_falls_back_without_failure(monkeypatch, tmp_path):
    import src.mobile_sam_worker_client as client_module

    checkpoint = tmp_path / "mobile_sam.pt"
    checkpoint.write_bytes(b"checkpoint")

    def timeout_run(command, **kwargs):  # noqa: ARG001
        raise subprocess.TimeoutExpired(command, 1.0)

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "mobile_sam_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "mobile_sam_worker_enabled", True)
    monkeypatch.setattr(client_module.subprocess, "run", timeout_run)

    detection = make_detection("person")
    segmenter = client_module.MobileSamWorkerSegmenter(checkpoint=checkpoint)
    refined = segmenter.refine(tmp_path / "frame.jpg", [detection])

    assert refined[0].refined_mask is not None
    assert segmenter.last_diagnostics["mobileSamWorkerTimedOut"] is True
    assert segmenter.last_diagnostics["mobileSamFallbackReason"] == "worker_timeout"
    assert segmenter.last_diagnostics["mobileSamRefinementSource"] == "fallback"


def test_mobile_sam_worker_nonzero_exit_falls_back_without_failure(monkeypatch, tmp_path):
    import src.mobile_sam_worker_client as client_module

    checkpoint = tmp_path / "mobile_sam.pt"
    checkpoint.write_bytes(b"checkpoint")

    def nonzero_run(command, **kwargs):  # noqa: ARG001
        output_path = output_path_from_worker_command(command)
        output_path.write_text('{"ok": false, "errorType": "NativeCrash"}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 9, "", "native crash")

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "mobile_sam_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "mobile_sam_worker_enabled", True)
    monkeypatch.setattr(client_module.subprocess, "run", nonzero_run)

    detection = make_detection("person")
    segmenter = client_module.MobileSamWorkerSegmenter(checkpoint=checkpoint)
    refined = segmenter.refine(tmp_path / "frame.jpg", [detection])

    assert refined[0].refined_mask is not None
    assert segmenter.last_diagnostics["mobileSamWorkerExitCode"] == 9
    assert segmenter.last_diagnostics["mobileSamFallbackReason"] == "NativeCrash"
    assert segmenter.last_diagnostics["mobileSamRefinementSource"] == "fallback"


def test_mobile_sam_worker_invalid_json_falls_back_without_failure(monkeypatch, tmp_path):
    import src.mobile_sam_worker_client as client_module

    checkpoint = tmp_path / "mobile_sam.pt"
    checkpoint.write_bytes(b"checkpoint")

    def invalid_json_run(command, **kwargs):  # noqa: ARG001
        output_path = output_path_from_worker_command(command)
        output_path.write_text("not-json", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "mobile_sam_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "mobile_sam_worker_enabled", True)
    monkeypatch.setattr(client_module.subprocess, "run", invalid_json_run)

    detection = make_detection("person")
    segmenter = client_module.MobileSamWorkerSegmenter(checkpoint=checkpoint)
    refined = segmenter.refine(tmp_path / "frame.jpg", [detection])

    assert refined[0].refined_mask is not None
    assert segmenter.last_diagnostics["mobileSamWorkerExitCode"] == 0
    assert "invalid_worker_json" in segmenter.last_diagnostics["mobileSamFallbackReason"]
    assert segmenter.last_diagnostics["mobileSamRefinementSource"] == "fallback"


def test_lightweight_vlm_worker_disabled_does_not_launch_subprocess(monkeypatch, tmp_path):
    import src.lightweight_vlm_worker_client as client_module

    model_dir = tmp_path / "models" / "vlm" / "lightweight-256m"
    model_dir.mkdir(parents=True)

    def fail_if_run(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("worker subprocess should not launch when disabled")

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_worker_enabled", False)
    monkeypatch.setattr(client_module.subprocess, "run", fail_if_run)

    reasoner = client_module.LightweightVlmWorkerReasoner(model_dir=model_dir)
    text = reasoner.explain_violation(np.zeros((16, 16, 3), dtype=np.uint8), [make_violation()])

    assert "Rule-based explanation" in text
    assert reasoner.last_diagnostics["lightweightVlmWorkerAttempted"] is False
    assert reasoner.last_diagnostics["lightweightVlmExplanationSource"] == "disabled"


def test_lightweight_vlm_worker_success_result_is_consumed(monkeypatch, tmp_path):
    import src.lightweight_vlm_worker_client as client_module

    model_dir = tmp_path / "models" / "vlm" / "lightweight-256m"
    model_dir.mkdir(parents=True)

    def fake_run(command, **kwargs):  # noqa: ARG001
        request = client_module.json.loads(input_path_from_worker_command(command).read_text(encoding="utf-8"))
        assert request["profile"] == "lightweight_256m"
        assert request["maxTokens"] <= 96
        assert request["generationTimeoutSeconds"] >= 10.0
        assert request["imageRegion"]["source"] == "detector_crop"
        assert request["imageRegion"]["matchedLabels"] == ["person"]
        assert request["detections"][0]["label"] == "person"
        assert request["analysisContext"]["profileLabel"] == "Helmet / PPE compliance"
        assert request["analysisContext"]["effectiveQuery"] == "worker without helmet"
        assert request["analysisContext"]["friendlyFindingName"] == "missing helmet"
        output_path = output_path_from_worker_command(command)
        output_path.write_text(
            client_module.json.dumps(
                {
                    "ok": True,
                    "explanation": "Visible safety evidence shows the worker area and helmet finding for review.",
                    "explanationSource": "vlm_lightweight",
                    "modelProfile": "lightweight_256m",
                    "generationTimeoutSeconds": request["generationTimeoutSeconds"],
                    "maxTokens": request["maxTokens"],
                    "imageRegion": request["imageRegion"],
                    "analysisContext": request["analysisContext"],
                    "workerDurationSeconds": 1.25,
                    "modelLoadSeconds": 0.75,
                    "generationSeconds": 0.5,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_worker_timeout_seconds", 7)
    monkeypatch.setattr(client_module.subprocess, "run", fake_run)

    reasoner = client_module.LightweightVlmWorkerReasoner(model_dir=model_dir)
    text = reasoner.explain_violation(
        np.zeros((100, 100, 3), dtype=np.uint8),
        [make_violation()],
        detections=[make_detection("person", bbox=(20, 10, 80, 90))],
        context={
            "profileLabel": "Helmet / PPE compliance",
            "effectiveQuery": "worker without helmet",
            "findingName": "helmet_missing",
            "friendlyFindingName": "missing helmet",
            "reviewLevel": "review_candidate",
        },
    )

    assert "Visible safety evidence" in text
    assert reasoner.last_explanation_source == "vlm_lightweight"
    assert reasoner.last_diagnostics["lightweightVlmWorkerAttempted"] is True
    assert reasoner.last_diagnostics["lightweightVlmWorkerSucceeded"] is True
    assert reasoner.last_diagnostics["lightweightVlmExplanationSource"] == "vlm_lightweight"
    assert reasoner.last_diagnostics["lightweightVlmMaxTokens"] <= 96
    assert reasoner.last_diagnostics["lightweightVlmWorkerDurationSeconds"] is not None
    assert reasoner.last_diagnostics["lightweightVlmWorkerModelLoadSeconds"] == 0.75
    assert reasoner.last_diagnostics["lightweightVlmWorkerGenerationSeconds"] == 0.5
    assert reasoner.last_diagnostics["lightweightVlmImageRegion"]["source"] == "detector_crop"
    assert reasoner.last_diagnostics["lightweightVlmAnalysisContext"]["profileLabel"] == "Helmet / PPE compliance"


def test_lightweight_vlm_auto_cuda_uses_512m_primary(monkeypatch, tmp_path):
    import src.lightweight_vlm_worker_client as client_module

    lightweight_256 = tmp_path / "models" / "vlm" / "lightweight-256m"
    lightweight_512 = tmp_path / "models" / "vlm" / "lightweight-512m"
    lightweight_256.mkdir(parents=True)
    lightweight_512.mkdir(parents=True)
    (lightweight_256 / "model.safetensors").write_bytes(b"model")
    (lightweight_512 / "model.safetensors").write_bytes(b"model")

    monkeypatch.setattr(client_module.SETTINGS, "vlm_lightweight_model_path", lightweight_256)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_lightweight_512m_model_path", lightweight_512)
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_primary", "auto")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_cpu_prefer_256m", True)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(client_module, "selected_component_device", lambda settings, component: "cuda")

    reasoner = client_module.LightweightVlmWorkerReasoner()

    assert reasoner.device == "cuda"
    assert reasoner.profile_id == "lightweight_512m"
    assert reasoner.model_dir == lightweight_512
    assert reasoner.selection_reason == "512m_installed"


def test_lightweight_vlm_forced_cpu_can_prefer_256m(monkeypatch, tmp_path):
    import src.lightweight_vlm_worker_client as client_module

    lightweight_256 = tmp_path / "models" / "vlm" / "lightweight-256m"
    lightweight_512 = tmp_path / "models" / "vlm" / "lightweight-512m"
    lightweight_256.mkdir(parents=True)
    lightweight_512.mkdir(parents=True)
    (lightweight_256 / "model.safetensors").write_bytes(b"model")
    (lightweight_512 / "model.safetensors").write_bytes(b"model")

    monkeypatch.setattr(client_module.SETTINGS, "vlm_lightweight_model_path", lightweight_256)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_lightweight_512m_model_path", lightweight_512)
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_primary", "auto")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_cpu_prefer_256m", True)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(client_module, "selected_component_device", lambda settings, component: "cpu")

    reasoner = client_module.LightweightVlmWorkerReasoner()

    assert reasoner.device == "cpu"
    assert reasoner.profile_id == "lightweight_256m"
    assert reasoner.model_dir == lightweight_256
    assert reasoner.selection_reason == "cpu_policy_prefers_256m"


def test_enhanced_2b_worker_success_marks_enhanced_source(monkeypatch, tmp_path):
    import src.lightweight_vlm_worker_client as client_module

    model_dir = tmp_path / "models" / "vlm" / "enhanced-2b"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"model")

    def fake_run(command, **kwargs):  # noqa: ARG001
        request = client_module.json.loads(input_path_from_worker_command(command).read_text(encoding="utf-8"))
        assert request["profile"] == "enhanced_2b"
        assert request["device"] == "cuda"
        assert kwargs["env"]["SAFETRACE_DEVICE"] == "cuda"
        assert kwargs["env"]["SAFETRACE_VLM_ENHANCED_MODEL_PATH"] == str(model_dir)
        assert request["analysisContext"]["profileLabel"] == "Seatbelt compliance"
        assert request["analysisContext"]["effectiveQuery"] == "driver missing seatbelt"
        assert request["analysisContext"]["ruleSupport"] == "torso_or_cabin_context"
        output_path = output_path_from_worker_command(command)
        output_path.write_text(
            client_module.json.dumps(
                {
                    "ok": True,
                    "explanation": "Visible seatbelt evidence is unclear because the torso is partly occluded.",
                    "explanationSource": "vlm_enhanced",
                    "modelProfile": "enhanced_2b",
                    "generationTimeoutSeconds": request["generationTimeoutSeconds"],
                    "maxTokens": request["maxTokens"],
                    "imageRegion": request["imageRegion"],
                    "analysisContext": request["analysisContext"],
                    "workerDurationSeconds": 2.0,
                    "modelLoadSeconds": 1.0,
                    "generationSeconds": 1.0,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(client_module.subprocess, "run", fake_run)

    reasoner = client_module.LightweightVlmWorkerReasoner(model_dir=model_dir, device="cuda")
    text = reasoner.explain_violation(
        np.zeros((16, 16, 3), dtype=np.uint8),
        [Violation(name="seatbelt_missing", description="Missing seatbelt", severity="high", confidence=0.9)],
        context={
            "profileLabel": "Seatbelt compliance",
            "effectiveQuery": "driver missing seatbelt",
            "findingName": "seatbelt_missing",
            "friendlyFindingName": "missing seatbelt",
            "ruleSupport": "torso_or_cabin_context",
            "reviewLevel": "likely_violation",
        },
    )

    assert "Visible seatbelt evidence" in text
    assert reasoner.profile_id == "enhanced_2b"
    assert reasoner.last_explanation_source == "vlm_enhanced"
    assert reasoner.last_diagnostics["lightweightVlmExplanationSource"] == "vlm_enhanced"
    assert reasoner.last_diagnostics["lightweightVlmModelProfile"] == "enhanced_2b"
    assert reasoner.last_diagnostics["lightweightVlmAnalysisContext"]["profileLabel"] == "Seatbelt compliance"


def test_lightweight_512m_quality_failure_falls_back_to_256m(monkeypatch, tmp_path):
    import src.lightweight_vlm_worker_client as client_module

    lightweight_256 = tmp_path / "models" / "vlm" / "lightweight-256m"
    lightweight_512 = tmp_path / "models" / "vlm" / "lightweight-512m"
    lightweight_256.mkdir(parents=True)
    lightweight_512.mkdir(parents=True)
    (lightweight_256 / "model.safetensors").write_bytes(b"model")
    (lightweight_512 / "model.safetensors").write_bytes(b"model")
    calls = []

    def fake_run(command, **kwargs):  # noqa: ARG001
        request = client_module.json.loads(input_path_from_worker_command(command).read_text(encoding="utf-8"))
        calls.append(request["profile"])
        output_path = output_path_from_worker_command(command)
        if request["profile"] == "lightweight_512m":
            output_path.write_text(
                client_module.json.dumps(
                    {
                        "ok": False,
                        "fallbackReason": "quality:non-informative structured output",
                        "qualityIssue": "non-informative structured output",
                        "rawTextPreview": "visible_evidence: no visual_status: occluded short_reason: no",
                        "cleanTextPreview": "visible_evidence: no visual_status: occluded short_reason: no",
                        "workerDurationSeconds": 2.0,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 3, "", "")
        output_path.write_text(
            client_module.json.dumps(
                {
                    "ok": True,
                    "explanation": "The belt path is not visible across the torso.",
                    "explanationSource": "vlm_lightweight",
                    "modelProfile": "lightweight_256m",
                    "workerDurationSeconds": 1.0,
                    "modelLoadSeconds": 0.5,
                    "generationSeconds": 0.5,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_lightweight_model_path", lightweight_256)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_lightweight_512m_model_path", lightweight_512)
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_primary", "auto")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_fallback", "256m")
    monkeypatch.setattr(client_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(client_module, "selected_component_device", lambda settings, component: "cuda")
    monkeypatch.setattr(client_module.subprocess, "run", fake_run)

    reasoner = client_module.LightweightVlmWorkerReasoner()
    text = reasoner.explain_violation(
        np.zeros((16, 16, 3), dtype=np.uint8),
        [Violation(name="seatbelt_missing", description="Missing seatbelt", severity="high", confidence=0.9)],
    )

    assert text == "The belt path is not visible across the torso."
    assert calls == ["lightweight_512m", "lightweight_256m"]
    assert reasoner.last_diagnostics["lightweightVlmModelProfile"] == "lightweight_256m"
    assert reasoner.last_diagnostics["lightweightVlmPrimaryAttemptedProfile"] == "lightweight_512m"
    assert reasoner.last_diagnostics["lightweightVlmPrimaryFallbackReason"] == "quality:non-informative structured output"
    assert reasoner.last_diagnostics["lightweightVlmWorkerSucceeded"] is True


def test_lightweight_256m_crop_rejection_retries_full_frame(monkeypatch, tmp_path):
    import src.lightweight_vlm_worker_client as client_module

    lightweight_256 = tmp_path / "models" / "vlm" / "lightweight-256m"
    lightweight_512 = tmp_path / "models" / "vlm" / "lightweight-512m"
    lightweight_256.mkdir(parents=True)
    lightweight_512.mkdir(parents=True)
    (lightweight_256 / "model.safetensors").write_bytes(b"model")
    (lightweight_512 / "model.safetensors").write_bytes(b"model")
    calls = []

    def fake_run(command, **kwargs):  # noqa: ARG001
        request = client_module.json.loads(input_path_from_worker_command(command).read_text(encoding="utf-8"))
        calls.append((request["profile"], request["imageRegion"]["source"]))
        output_path = output_path_from_worker_command(command)
        if request["profile"] == "lightweight_512m":
            output_path.write_text(
                client_module.json.dumps(
                    {
                        "ok": False,
                        "fallbackReason": "quality:too short",
                        "qualityIssue": "too short",
                        "rawTextPreview": "yes.",
                        "cleanTextPreview": "yes.",
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 3, "", "")
        if request["imageRegion"]["source"] == "detector_crop":
            output_path.write_text(
                client_module.json.dumps(
                    {
                        "ok": False,
                        "fallbackReason": "quality:missing safety-specific detail",
                        "qualityIssue": "missing safety-specific detail",
                        "rawTextPreview": "There is one person.",
                        "cleanTextPreview": "There is one person.",
                        "imageRegion": request["imageRegion"],
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 3, "", "")
        output_path.write_text(
            client_module.json.dumps(
                {
                    "ok": True,
                    "explanation": "The belt path is not visible across the torso.",
                    "explanationSource": "vlm_lightweight",
                    "modelProfile": "lightweight_256m",
                    "imageRegion": request["imageRegion"],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_lightweight_model_path", lightweight_256)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_lightweight_512m_model_path", lightweight_512)
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_primary", "auto")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_fallback", "256m")
    monkeypatch.setattr(client_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(client_module, "selected_component_device", lambda settings, component: "cuda")
    monkeypatch.setattr(client_module.subprocess, "run", fake_run)

    reasoner = client_module.LightweightVlmWorkerReasoner()
    text = reasoner.explain_violation(
        np.zeros((100, 100, 3), dtype=np.uint8),
        [Violation(name="seatbelt_missing", description="Missing seatbelt", severity="high", confidence=0.9)],
        detections=[make_detection("person", bbox=(20, 10, 80, 90))],
    )

    assert text == "The belt path is not visible across the torso."
    assert calls == [
        ("lightweight_512m", "detector_crop"),
        ("lightweight_256m", "detector_crop"),
        ("lightweight_256m", "full_frame"),
    ]
    assert reasoner.last_diagnostics["lightweightVlmWorkerSucceeded"] is True
    assert reasoner.last_diagnostics["lightweightVlmFullFrameRetry"] is True
    assert reasoner.last_diagnostics["lightweightVlmImageRegion"]["source"] == "full_frame"


def test_lightweight_vlm_worker_timeout_falls_back_without_failure(monkeypatch, tmp_path):
    import src.lightweight_vlm_worker_client as client_module

    model_dir = tmp_path / "models" / "vlm" / "lightweight-256m"
    model_dir.mkdir(parents=True)

    def timeout_run(command, **kwargs):  # noqa: ARG001
        raise subprocess.TimeoutExpired(command, 1.0)

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(client_module.subprocess, "run", timeout_run)

    reasoner = client_module.LightweightVlmWorkerReasoner(model_dir=model_dir)
    text = reasoner.explain_violation(np.zeros((16, 16, 3), dtype=np.uint8), [make_violation()])

    assert "Rule-based explanation" in text
    assert reasoner.last_diagnostics["lightweightVlmWorkerTimedOut"] is True
    assert reasoner.last_diagnostics["lightweightVlmFallbackReason"] == "worker_timeout"
    assert reasoner.last_diagnostics["lightweightVlmExplanationSource"] == "rule_based"
    assert reasoner.last_diagnostics["lightweightVlmWorkerDurationSeconds"] is not None


def test_lightweight_vlm_worker_nonzero_exit_falls_back_without_failure(monkeypatch, tmp_path):
    import src.lightweight_vlm_worker_client as client_module

    model_dir = tmp_path / "models" / "vlm" / "lightweight-256m"
    model_dir.mkdir(parents=True)

    def nonzero_run(command, **kwargs):  # noqa: ARG001
        output_path = output_path_from_worker_command(command)
        output_path.write_text('{"ok": false, "errorType": "NativeCrash"}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 9, "", "native crash")

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(client_module.subprocess, "run", nonzero_run)

    reasoner = client_module.LightweightVlmWorkerReasoner(model_dir=model_dir)
    text = reasoner.explain_violation(np.zeros((16, 16, 3), dtype=np.uint8), [make_violation()])

    assert "Rule-based explanation" in text
    assert reasoner.last_diagnostics["lightweightVlmWorkerExitCode"] == 9
    assert reasoner.last_diagnostics["lightweightVlmFallbackReason"] == "NativeCrash"
    assert reasoner.last_diagnostics["lightweightVlmWorkerStderrPreview"] == "native crash"


def test_lightweight_vlm_worker_invalid_json_falls_back_without_failure(monkeypatch, tmp_path):
    import src.lightweight_vlm_worker_client as client_module

    model_dir = tmp_path / "models" / "vlm" / "lightweight-256m"
    model_dir.mkdir(parents=True)

    def invalid_json_run(command, **kwargs):  # noqa: ARG001
        output_path = output_path_from_worker_command(command)
        output_path.write_text("not-json", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(client_module.subprocess, "run", invalid_json_run)

    reasoner = client_module.LightweightVlmWorkerReasoner(model_dir=model_dir)
    text = reasoner.explain_violation(np.zeros((16, 16, 3), dtype=np.uint8), [make_violation()])

    assert "Rule-based explanation" in text
    assert reasoner.last_diagnostics["lightweightVlmWorkerExitCode"] == 0
    assert "invalid_worker_json" in reasoner.last_diagnostics["lightweightVlmFallbackReason"]


def test_lightweight_vlm_worker_quality_rejection_records_reason(monkeypatch, tmp_path):
    import src.lightweight_vlm_worker_client as client_module

    model_dir = tmp_path / "models" / "vlm" / "lightweight-256m"
    model_dir.mkdir(parents=True)

    def rejected_run(command, **kwargs):  # noqa: ARG001
        output_path = output_path_from_worker_command(command)
        output_path.write_text(
            client_module.json.dumps(
                {
                    "ok": False,
                    "errorType": "VlmFallback",
                    "fallbackReason": "quality:too short",
                    "qualityIssue": "too short",
                    "rawTextPreview": "unclear",
                    "cleanTextPreview": "unclear",
                    "generationTimeoutSeconds": 50.0,
                    "maxTokens": 64,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 3, "", "")

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(client_module.subprocess, "run", rejected_run)

    reasoner = client_module.LightweightVlmWorkerReasoner(model_dir=model_dir)
    text = reasoner.explain_violation(np.zeros((16, 16, 3), dtype=np.uint8), [make_violation()])

    assert "Rule-based explanation" in text
    assert reasoner.last_diagnostics["lightweightVlmWorkerExitCode"] == 3
    assert reasoner.last_diagnostics["lightweightVlmFallbackReason"] == "quality:too short"
    assert reasoner.last_diagnostics["lightweightVlmQualityIssue"] == "too short"
    assert reasoner.last_diagnostics["lightweightVlmCleanTextPreview"] == "unclear"


def test_lightweight_vlm_worker_rejects_generic_person_activity(monkeypatch, tmp_path):
    import src.lightweight_vlm_worker_client as client_module

    model_dir = tmp_path / "models" / "vlm" / "lightweight-256m"
    model_dir.mkdir(parents=True)

    def generic_run(command, **kwargs):  # noqa: ARG001
        output_path = output_path_from_worker_command(command)
        output_path.write_text(
            client_module.json.dumps(
                {
                    "ok": True,
                    "explanation": "A person is holding a paper near the vehicle.",
                    "explanationSource": "vlm_lightweight",
                    "modelProfile": "lightweight_256m",
                    "generationTimeoutSeconds": 20.0,
                    "maxTokens": 64,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(client_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "project_root", tmp_path)
    monkeypatch.setattr(client_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(client_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(client_module.subprocess, "run", generic_run)

    reasoner = client_module.LightweightVlmWorkerReasoner(model_dir=model_dir)
    text = reasoner.explain_violation(np.zeros((16, 16, 3), dtype=np.uint8), [make_violation()])

    assert "Rule-based explanation" in text
    assert reasoner.last_diagnostics["lightweightVlmWorkerAttempted"] is True
    assert reasoner.last_diagnostics["lightweightVlmWorkerSucceeded"] is False
    assert reasoner.last_diagnostics["lightweightVlmFallbackReason"] == "quality:generic person activity"
    assert reasoner.last_diagnostics["lightweightVlmQualityIssue"] == "generic person activity"


def test_lightweight_profile_uses_structured_seatbelt_prompt(monkeypatch):
    from src.vlm_reasoner import VLM_PROMPT, VlmReasoner

    monkeypatch.setattr("src.vlm_reasoner.SETTINGS.vlm_profile", "lightweight_256m")
    reasoner = VlmReasoner(enabled=False)

    prompt = reasoner._prompt_for(
        [Violation(name="seatbelt_missing", description="Missing seatbelt", severity="high", confidence=0.9)]
    )

    assert "Focus: torso area and possible belt path." in prompt
    assert "visible_evidence:" in prompt
    assert "visual_status:\n" in prompt
    assert "short_reason:" in prompt
    assert "confidence_hint:" in prompt
    assert "worn, not visible, not worn, or cannot determine" not in prompt
    assert "low, medium, or high" not in prompt
    assert "Do not list unrelated objects" in prompt
    assert "seatbelt_missing" not in prompt
    assert prompt != VLM_PROMPT


def test_lightweight_profile_uses_structured_helmet_prompt(monkeypatch):
    from src.vlm_reasoner import VlmReasoner

    monkeypatch.setattr("src.vlm_reasoner.SETTINGS.vlm_profile", "lightweight_256m")
    reasoner = VlmReasoner(enabled=False)

    prompt = reasoner._prompt_for([make_violation()])

    assert "Focus: head and upper-body area." in prompt
    assert "visible_evidence:" in prompt
    assert "visual_status:\n" in prompt
    assert "short_reason:" in prompt
    assert "confidence_hint:" in prompt
    assert "PPE present, missing, not visible, or cannot determine" not in prompt


def test_lightweight_profile_uses_structured_phone_prompt(monkeypatch):
    from src.vlm_reasoner import VlmReasoner

    monkeypatch.setattr("src.vlm_reasoner.SETTINGS.vlm_profile", "lightweight_256m")
    reasoner = VlmReasoner(enabled=False)

    prompt = reasoner._prompt_for(
        [Violation(name="phone_use", description="Driver using phone", severity="medium", confidence=0.5)]
    )

    assert "Focus: hand, face, and driver area." in prompt
    assert "visible_evidence:" in prompt
    assert "visual_status:\n" in prompt
    assert "short_reason:" in prompt
    assert "confidence_hint:" in prompt
    assert "phone visible, active use unclear, no phone visible, or cannot determine" not in prompt


def test_vlm_quality_accepts_safety_relevant_uncertainty():
    text = (
        "visible evidence: driver torso is partly occluded. "
        "seatbelt status: cannot determine. "
        "reason: the belt path is not visible from this angle. "
        "confidence: low"
    )

    assert is_useful_vlm_output(text)


def test_vlm_quality_accepts_possible_missing_seatbelt():
    text = (
        "visible evidence: a driver torso is visible. "
        "seatbelt status: not worn. "
        "reason: no belt appears to cross the torso in this frame. "
        "confidence: medium"
    )

    assert is_useful_vlm_output(text)


def test_vlm_quality_rejects_unrelated_paper_description():
    text = "A person is holding a paper near a vehicle and appears to be sitting."

    assert vlm_reasoner.vlm_output_quality_issue(text) == "generic person activity"


def test_vlm_quality_rejects_prompt_placeholders_and_generic_finding_labels():
    placeholder_text = (
        "visible evidence: <what is visible about torso/seatbelt> "
        "seatbelt status: worn / not visible / not worn / cannot determine "
        "reason: cannot determine confidence: low"
    )

    assert vlm_reasoner.vlm_output_quality_issue(placeholder_text) == "prompt placeholder echo"
    assert (
        vlm_reasoner.vlm_output_quality_issue("worn, not visible, not worn, or cannot determine")
        == "prompt option-list echo"
    )
    assert (
        vlm_reasoner.vlm_output_quality_issue(
            "visible evidence: seatbelt compliance seatbelt status: not visible reason: the evidence is unclear"
        )
        == "generic finding label"
    )
    assert vlm_reasoner.vlm_output_quality_issue("seatbelt compliance.") == "generic output"
    assert (
        vlm_reasoner.vlm_output_quality_issue(
            "visible_evidence: no visual_status: occluded short_reason: no"
        )
        == "non-informative structured output"
    )
    assert (
        vlm_reasoner.vlm_output_quality_issue(
            "visible_evidence: belt path visual_status: occluded short_reason: unclear confidence_hint:"
        )
        == "non-informative structured output"
    )


def test_pipeline_safe_mode_uses_mobile_sam_worker_after_frame_selection(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    constructed = {"worker": 0, "refined": 0}

    class FakeDetector:
        checkpoint = "checkpoints/yolov8s-seg.pt"

        def detect(self, image):  # noqa: ARG002
            return [make_detection("person")]

    class FakeWorker:
        available = True

        def __init__(self, *args, **kwargs):  # noqa: ARG002
            constructed["worker"] += 1
            self.last_diagnostics = {
                "mobileSamWorkerEnabled": True,
                "mobileSamWorkerTimeoutSeconds": 60,
                "mobileSamWorkerAttempted": False,
                "mobileSamWorkerSucceeded": False,
                "mobileSamWorkerTimedOut": False,
                "mobileSamWorkerExitCode": None,
                "mobileSamFallbackReason": None,
                "mobileSamRefinementSource": "disabled",
            }

        def refine(self, image, detections):  # noqa: ARG002
            constructed["refined"] += 1
            self.last_diagnostics = {
                "mobileSamWorkerEnabled": True,
                "mobileSamWorkerTimeoutSeconds": 60,
                "mobileSamWorkerAttempted": True,
                "mobileSamWorkerSucceeded": True,
                "mobileSamWorkerTimedOut": False,
                "mobileSamWorkerExitCode": 0,
                "mobileSamFallbackReason": None,
                "mobileSamRefinementSource": "worker",
            }
            detections[0].refined_mask = detections[0].coarse_mask
            return detections

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"placeholder")

    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "safe_mode_allow_mobilesam", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_enabled", "true")
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_worker_enabled", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_worker_timeout_seconds", 60)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "top_k", 1)
    monkeypatch.setattr(pipeline_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline_module, "ClipEmbedder", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ClipEmbedder should not load")))
    monkeypatch.setattr(pipeline_module, "FaissIndex", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("FaissIndex should not load")))
    monkeypatch.setattr(pipeline_module, "VlmReasoner", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("VLM should not load")))
    monkeypatch.setattr(pipeline_module, "MobileSamSegmenter", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct MobileSAM should not load")))
    monkeypatch.setattr(pipeline_module, "MobileSamWorkerSegmenter", FakeWorker)
    monkeypatch.setattr(pipeline_module, "YoloDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((100, 100, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "imwrite_rgb", lambda path, image: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline_module, "evaluate_rules", lambda detections: [make_violation()])
    monkeypatch.setattr(
        pipeline_module.SafeTracePipeline,
        "_collect_input_frames",
        lambda self, inputs, fps, max_frames, **kwargs: ([frame], [], {"videos": 0, "images": 1}),  # noqa: ARG005
    )

    pipeline = pipeline_module.SafeTracePipeline()
    result = pipeline.run([frame], query="driver without seatbelt", fps=1.0, k=1)

    assert result
    assert constructed["worker"] == 1
    assert constructed["refined"] == 1
    assert pipeline.component_diagnostics["mobileSamWorkerEnabled"] is True
    assert pipeline.component_diagnostics["mobileSamWorkerAttempted"] is True
    assert pipeline.component_diagnostics["mobileSamWorkerSucceeded"] is True
    assert pipeline.component_diagnostics["mobileSamWorkerExitCode"] == 0
    assert pipeline.component_diagnostics["mobileSamLoaded"] is False
    assert pipeline.component_diagnostics["mobileSamRefinementSource"] == "worker"
    assert pipeline.component_diagnostics["embeddingRequested"] is False
    assert pipeline.component_diagnostics["vlmLoaded"] is False
    assert result[0]["search_metadata"]["mobileSamRefinement"]["mobileSamRefinementSource"] == "worker"


def test_pipeline_safe_mode_uses_lightweight_vlm_worker_after_frame_selection(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    constructed = {"worker": 0, "explained": 0}

    class FakeDetector:
        checkpoint = "checkpoints/yolov8s-seg.pt"

        def detect(self, image):  # noqa: ARG002
            return [make_detection("person")]

    class FakeVlmWorker:
        provider = "vlm_lightweight_worker"
        enabled = True

        def __init__(self, *args, **kwargs):  # noqa: ARG002
            constructed["worker"] += 1
            self.last_explanation_source = "rule_based"
            self.last_diagnostics = {
                "lightweightVlmWorkerEnabled": True,
                "lightweightVlmWorkerTimeoutSeconds": 60,
                "lightweightVlmWorkerAttempted": False,
                "lightweightVlmWorkerSucceeded": False,
                "lightweightVlmWorkerTimedOut": False,
                "lightweightVlmWorkerExitCode": None,
                "lightweightVlmFallbackReason": None,
                "lightweightVlmExplanationSource": "rule_based",
            }

        def explain_violation(self, image, violations, *, detections=None):  # noqa: ARG002
            constructed["explained"] += 1
            constructed["detection_labels"] = [detection.label for detection in detections or []]
            self.last_explanation_source = "vlm_lightweight"
            self.last_diagnostics = {
                "lightweightVlmWorkerEnabled": True,
                "lightweightVlmWorkerTimeoutSeconds": 60,
                "lightweightVlmWorkerAttempted": True,
                "lightweightVlmWorkerSucceeded": True,
                "lightweightVlmWorkerTimedOut": False,
                "lightweightVlmWorkerExitCode": 0,
                "lightweightVlmFallbackReason": None,
                "lightweightVlmExplanationSource": "vlm_lightweight",
                "lightweightVlmImageRegion": {
                    "source": "detector_crop",
                    "matchedLabels": ["person"],
                },
            }
            return "Visible safety evidence supports the missing helmet finding for reviewer confirmation."

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"placeholder")

    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "safe_mode_allow_mobilesam", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_enabled", "disabled")
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_worker_enabled", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "lightweight_256m")
    monkeypatch.setattr(pipeline_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "lightweight_vlm_worker_timeout_seconds", 60)
    monkeypatch.setattr(pipeline_module.SETTINGS, "top_k", 1)
    monkeypatch.setattr(pipeline_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline_module, "ClipEmbedder", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ClipEmbedder should not load")))
    monkeypatch.setattr(pipeline_module, "FaissIndex", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("FaissIndex should not load")))
    monkeypatch.setattr(pipeline_module, "VlmReasoner", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct VLM should not load")))
    monkeypatch.setattr(pipeline_module, "MobileSamSegmenter", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("MobileSAM should not load")))
    monkeypatch.setattr(pipeline_module, "LightweightVlmWorkerReasoner", FakeVlmWorker)
    monkeypatch.setattr(pipeline_module, "YoloDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((100, 100, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "imwrite_rgb", lambda path, image: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline_module, "evaluate_rules", lambda detections: [make_violation()])
    monkeypatch.setattr(
        pipeline_module.SafeTracePipeline,
        "_collect_input_frames",
        lambda self, inputs, fps, max_frames, **kwargs: ([frame], [], {"videos": 0, "images": 1}),  # noqa: ARG005
    )

    pipeline = pipeline_module.SafeTracePipeline()
    result = pipeline.run([frame], query="worker without helmet", fps=1.0, k=1)

    assert result
    assert constructed["worker"] == 1
    assert constructed["explained"] == 1
    assert pipeline.component_diagnostics["safeMode"] is True
    assert pipeline.component_diagnostics["embeddingRequested"] is False
    assert pipeline.component_diagnostics["vlmLoaded"] is False
    assert pipeline.component_diagnostics["lightweightVlmWorkerEnabled"] is True
    assert pipeline.component_diagnostics["lightweightVlmWorkerAttempted"] is True
    assert pipeline.component_diagnostics["lightweightVlmWorkerSucceeded"] is True
    assert pipeline.component_diagnostics["effectiveExplanationMode"] == "lightweight_256m"
    assert result[0]["explanation_source"] == "rule_template_plus_lightweight_vlm"
    assert "Safety review" in result[0]["explanation"]
    assert "Visual review" in result[0]["explanation"]
    assert "What to check in the original footage" in result[0]["explanation"]
    lightweight = result[0]["search_metadata"]["lightweightVlmExplanation"]
    assert lightweight["lightweightVlmExplanationSource"] == "vlm_lightweight"
    assert lightweight["lightweightVlmWorkerSucceeded"] is True
    assert lightweight["baseExplanationSource"] == "rule_based"
    assert lightweight["finalExplanationSource"] == "rule_template_plus_lightweight_vlm"
    assert lightweight["lightweightVlmContributionAccepted"] is True
    assert constructed["detection_labels"] == ["person"]
    assert (
        result[0]["search_metadata"]["lightweightVlmExplanation"]["lightweightVlmImageRegion"]["source"]
        == "detector_crop"
    )


def test_pipeline_marks_vlm_disagreement_without_overriding_weak_rule(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    class FakeVlmWorker:
        provider = "vlm_lightweight_worker"
        enabled = True

        def __init__(self):
            self.last_explanation_source = "rule_based"
            self.last_diagnostics = {}

        def explain_violation(self, image, violations, *, detections=None):  # noqa: ARG002
            self.last_explanation_source = "vlm_lightweight"
            self.last_diagnostics = {
                "lightweightVlmWorkerEnabled": True,
                "lightweightVlmWorkerAttempted": True,
                "lightweightVlmWorkerSucceeded": True,
                "lightweightVlmWorkerTimedOut": False,
                "lightweightVlmWorkerExitCode": 0,
                "lightweightVlmFallbackReason": None,
                "lightweightVlmExplanationSource": "vlm_lightweight",
                "lightweightVlmCleanTextPreview": "visible_evidence: belt path visible visual_status: wearing a belt short_reason: The person is wearing a belt. confidence_hint: medium",
            }
            return "visible_evidence: belt path visible visual_status: wearing a belt short_reason: The person is wearing a belt. confidence_hint: medium"

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"placeholder")
    weak_rule = Violation(
        name="seatbelt_missing",
        description="Possible missing seatbelt based on generic person evidence; review original footage before acting.",
        severity="medium",
        confidence=0.42,
        evidence={
            "evidenceStrength": "review_candidate",
            "confidenceReason": "Only generic person evidence is available.",
            "ruleSupport": "person_proxy_only",
            "reviewRequired": True,
        },
    )

    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "lightweight_512m")
    monkeypatch.setattr(pipeline_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_frames", 1)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_evidence_frames", 1)
    monkeypatch.setattr(pipeline_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((100, 100, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "imwrite_rgb", lambda path, image: None)  # noqa: ARG005

    pipeline = pipeline_module.SafeTracePipeline(
        embedder=object(),
        index=object(),
        detector=object(),
        segmenter=pipeline_module.CoarseMaskSegmenter(),
        vlm=FakeVlmWorker(),
    )
    analysis = pipeline.analyze_frame(
        frame,
        precomputed={"detections": [make_detection("person")], "violations": [weak_rule]},
    )

    assert analysis.violations[0].name == "seatbelt_missing"
    assert analysis.violations[0].confidence < 0.4
    assert analysis.violations[0].evidence["verifierAgreement"] == "disagrees"
    assert "Visual review may disagree" in analysis.violations[0].evidence["finalReviewerNote"]
    assert analysis.explanation_source == "rule_template_plus_lightweight_vlm"
    assert "visual review disagrees" in analysis.explanation.lower()


def test_pipeline_marks_vlm_uncertainty_as_inconclusive(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    class FakeVlmWorker:
        provider = "vlm_lightweight_worker"
        enabled = True
        last_explanation_source = "vlm_lightweight"
        last_diagnostics = {
            "lightweightVlmWorkerEnabled": True,
            "lightweightVlmWorkerAttempted": True,
            "lightweightVlmWorkerSucceeded": True,
            "lightweightVlmWorkerTimedOut": False,
            "lightweightVlmWorkerExitCode": 0,
            "lightweightVlmFallbackReason": None,
            "lightweightVlmExplanationSource": "vlm_lightweight",
        }

        def explain_violation(self, image, violations, *, detections=None):  # noqa: ARG002
            return "visible_evidence: torso partially blocked visual_status: unclear short_reason: Belt path cannot be determined because the torso is occluded. confidence_hint: low"

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"placeholder")
    weak_rule = Violation(
        name="seatbelt_missing",
        description="Possible missing seatbelt based on generic person evidence.",
        severity="medium",
        confidence=0.42,
        evidence={"evidenceStrength": "review_candidate", "reviewRequired": True},
    )

    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "lightweight_512m")
    monkeypatch.setattr(pipeline_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_frames", 1)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_evidence_frames", 1)
    monkeypatch.setattr(pipeline_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((100, 100, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "imwrite_rgb", lambda path, image: None)  # noqa: ARG005

    pipeline = pipeline_module.SafeTracePipeline(
        embedder=object(),
        index=object(),
        detector=object(),
        segmenter=pipeline_module.CoarseMaskSegmenter(),
        vlm=FakeVlmWorker(),
    )
    analysis = pipeline.analyze_frame(
        frame,
        precomputed={"detections": [make_detection("person")], "violations": [weak_rule]},
    )

    assert analysis.violations[0].evidence["verifierAgreement"] == "inconclusive"
    assert analysis.violations[0].confidence <= 0.5
    assert "inconclusive" in analysis.violations[0].evidence["finalReviewerNote"].lower()


def test_pipeline_lightweight_vlm_budget_records_skip_reason(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    constructed = {"worker": 0, "explained": 0}

    class FakeDetector:
        checkpoint = "checkpoints/yolov8s-seg.pt"

        def detect(self, image):  # noqa: ARG002
            return [make_detection("person")]

    class FakeVlmWorker:
        provider = "vlm_lightweight_worker"
        enabled = True

        def __init__(self, *args, **kwargs):  # noqa: ARG002
            constructed["worker"] += 1
            self.last_explanation_source = "rule_based"
            self.last_diagnostics = {
                "lightweightVlmWorkerEnabled": True,
                "lightweightVlmWorkerTimeoutSeconds": 60,
                "lightweightVlmWorkerAttempted": False,
                "lightweightVlmWorkerSucceeded": False,
                "lightweightVlmWorkerTimedOut": False,
                "lightweightVlmWorkerExitCode": None,
                "lightweightVlmFallbackReason": None,
                "lightweightVlmExplanationSource": "rule_based",
            }

        def explain_violation(self, image, violations):  # noqa: ARG002
            constructed["explained"] += 1
            self.last_explanation_source = "vlm_lightweight"
            self.last_diagnostics = {
                "lightweightVlmWorkerEnabled": True,
                "lightweightVlmWorkerTimeoutSeconds": 60,
                "lightweightVlmWorkerAttempted": True,
                "lightweightVlmWorkerSucceeded": True,
                "lightweightVlmWorkerTimedOut": False,
                "lightweightVlmWorkerExitCode": 0,
                "lightweightVlmFallbackReason": None,
                "lightweightVlmExplanationSource": "vlm_lightweight",
                "lightweightVlmCleanTextPreview": "Driver seatbelt evidence is visible for review.",
            }
            return "Driver seatbelt evidence is visible for review."

    frames = [tmp_path / "frame_a.jpg", tmp_path / "frame_b.jpg"]
    for frame in frames:
        frame.write_bytes(b"placeholder")

    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "safe_mode_allow_mobilesam", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_enabled", "disabled")
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_worker_enabled", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "lightweight_256m")
    monkeypatch.setattr(pipeline_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_evidence_frames", 1)
    monkeypatch.setattr(pipeline_module.SETTINGS, "top_k", 2)
    monkeypatch.setattr(pipeline_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline_module, "ClipEmbedder", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ClipEmbedder should not load")))
    monkeypatch.setattr(pipeline_module, "FaissIndex", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("FaissIndex should not load")))
    monkeypatch.setattr(pipeline_module, "VlmReasoner", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct VLM should not load")))
    monkeypatch.setattr(pipeline_module, "MobileSamSegmenter", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("MobileSAM should not load")))
    monkeypatch.setattr(pipeline_module, "LightweightVlmWorkerReasoner", FakeVlmWorker)
    monkeypatch.setattr(pipeline_module, "YoloDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((100, 100, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "imwrite_rgb", lambda path, image: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline_module, "evaluate_rules", lambda detections: [make_violation()])
    monkeypatch.setattr(
        pipeline_module.SafeTracePipeline,
        "_collect_input_frames",
        lambda self, inputs, fps, max_frames, **kwargs: (frames, [], {"videos": 0, "images": len(frames)}),  # noqa: ARG005
    )

    pipeline = pipeline_module.SafeTracePipeline()
    result = pipeline.run(frames, query="driver without seatbelt", fps=1.0, k=2)

    assert len(result) == 2
    assert constructed["explained"] == 1
    first = result[0]["search_metadata"]["lightweightVlmExplanation"]
    second = result[1]["search_metadata"]["lightweightVlmExplanation"]
    assert first["lightweightVlmWorkerAttempted"] is True
    assert first["lightweightVlmWorkerSucceeded"] is True
    assert first["lightweightVlmEvidenceBudget"] == 1
    assert second["lightweightVlmWorkerAttempted"] is False
    assert second["lightweightVlmWorkerSucceeded"] is False
    assert second["lightweightVlmFallbackReason"] == "visual_review_frame_limit_reached"
    assert pipeline.component_diagnostics["lightweightVlmFramesAttempted"] == 1
    assert pipeline.component_diagnostics["lightweightVlmFramesSkipped"] == 1


def test_pipeline_lightweight_timeout_disables_remaining_job_attempts(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    class FakeVlmWorker:
        provider = "vlm_lightweight_worker"
        enabled = True

        def __init__(self):
            self.count = 0
            self.last_explanation_source = "rule_based"
            self.last_diagnostics = {}

        def explain_violation(self, image, violations):  # noqa: ARG002
            self.count += 1
            self.last_explanation_source = "rule_based"
            self.last_diagnostics = {
                "lightweightVlmWorkerEnabled": True,
                "lightweightVlmWorkerTimeoutSeconds": 60,
                "lightweightVlmWorkerAttempted": True,
                "lightweightVlmWorkerSucceeded": False,
                "lightweightVlmWorkerTimedOut": True,
                "lightweightVlmWorkerExitCode": None,
                "lightweightVlmFallbackReason": "worker_timeout",
                "lightweightVlmExplanationSource": "rule_based",
                "lightweightVlmWorkerDurationSeconds": 60.0,
            }
            return "Rule-based explanation: fallback after timeout."

    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "lightweight_256m")
    monkeypatch.setattr(pipeline_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_frames", 3)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_evidence_frames", 3)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_job_timeout_seconds", 180)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_disable_after_timeout", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((100, 100, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "imwrite_rgb", lambda path, image: None)  # noqa: ARG005

    fake_vlm = FakeVlmWorker()
    pipeline = pipeline_module.SafeTracePipeline(
        embedder=object(),
        index=object(),
        detector=object(),
        segmenter=pipeline_module.CoarseMaskSegmenter(),
        vlm=fake_vlm,
    )
    precomputed = {
        "detections": [make_detection("person")],
        "violations": [
            Violation(
                name="seatbelt_missing",
                description="Possible missing seatbelt.",
                severity="medium",
                confidence=0.55,
                evidence={
                    "evidenceStrength": "review_candidate",
                    "ruleSupport": "torso_or_cabin_context",
                    "confidenceReason": "Belt path could not be confirmed.",
                },
            )
        ],
    }

    first = pipeline.analyze_frame(tmp_path / "first.jpg", precomputed=precomputed)
    second = pipeline.analyze_frame(tmp_path / "second.jpg", precomputed=precomputed)

    assert fake_vlm.count == 1
    assert first.explanation_source == "rule_based"
    assert second.explanation_source == "rule_based"
    assert pipeline.component_diagnostics["totalVlmAttempts"] == 1
    assert pipeline.component_diagnostics["totalVlmTimedOut"] == 1
    assert pipeline.component_diagnostics["lightweightVlmDisabledReason"] == "local_visual_review_timeout"
    assert pipeline.component_diagnostics["vlmSkippedReasons"]["local_visual_review_timeout"] == 1


def test_pipeline_lightweight_quality_failure_disables_remaining_job_attempts(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    class FakeVlmWorker:
        provider = "vlm_lightweight_worker"
        enabled = True

        def __init__(self):
            self.count = 0
            self.last_explanation_source = "rule_based"
            self.last_diagnostics = {}

        def explain_violation(self, image, violations):  # noqa: ARG002
            self.count += 1
            self.last_explanation_source = "rule_based"
            self.last_diagnostics = {
                "lightweightVlmWorkerEnabled": True,
                "lightweightVlmWorkerAttempted": True,
                "lightweightVlmWorkerSucceeded": False,
                "lightweightVlmWorkerTimedOut": False,
                "lightweightVlmWorkerExitCode": 3,
                "lightweightVlmFallbackReason": "quality:generic object inventory",
                "lightweightVlmQualityIssue": "generic object inventory",
                "lightweightVlmExplanationSource": "rule_based",
                "lightweightVlmWorkerDurationSeconds": 2.0,
            }
            return "Rule-based explanation: fallback after generic output."

    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "lightweight_256m")
    monkeypatch.setattr(pipeline_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_frames", 3)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_evidence_frames", 3)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_job_timeout_seconds", 60)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_quality_failures", 1)
    monkeypatch.setattr(pipeline_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((100, 100, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "imwrite_rgb", lambda path, image: None)  # noqa: ARG005

    fake_vlm = FakeVlmWorker()
    pipeline = pipeline_module.SafeTracePipeline(
        embedder=object(),
        index=object(),
        detector=object(),
        segmenter=pipeline_module.CoarseMaskSegmenter(),
        vlm=fake_vlm,
    )
    precomputed = {
        "detections": [make_detection("person")],
        "violations": [
            Violation(
                name="seatbelt_missing",
                description="Possible missing seatbelt.",
                severity="medium",
                confidence=0.55,
                evidence={
                    "evidenceStrength": "review_candidate",
                    "ruleSupport": "torso_or_cabin_context",
                    "confidenceReason": "Belt path could not be confirmed.",
                },
            )
        ],
    }

    pipeline.analyze_frame(tmp_path / "first.jpg", precomputed=precomputed)
    pipeline.analyze_frame(tmp_path / "second.jpg", precomputed=precomputed)

    assert fake_vlm.count == 1
    assert pipeline.component_diagnostics["totalVlmAttempts"] == 1
    assert pipeline.component_diagnostics["totalVlmRejected"] == 1
    assert pipeline.component_diagnostics["lightweightVlmDisabledReason"] == "local_visual_review_quality_guard"
    assert pipeline.component_diagnostics["vlmSkippedReasons"]["local_visual_review_quality_guard"] == 1


def test_pipeline_lightweight_job_time_budget_stops_remaining_attempts(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    class FakeVlmWorker:
        provider = "vlm_lightweight_worker"
        enabled = True

        def __init__(self):
            self.count = 0
            self.last_explanation_source = "rule_based"
            self.last_diagnostics = {}

        def explain_violation(self, image, violations):  # noqa: ARG002
            self.count += 1
            self.last_explanation_source = "vlm_lightweight"
            self.last_diagnostics = {
                "lightweightVlmWorkerEnabled": True,
                "lightweightVlmWorkerAttempted": True,
                "lightweightVlmWorkerSucceeded": True,
                "lightweightVlmWorkerTimedOut": False,
                "lightweightVlmWorkerExitCode": 0,
                "lightweightVlmFallbackReason": None,
                "lightweightVlmExplanationSource": "vlm_lightweight",
                "lightweightVlmWorkerDurationSeconds": 61.0,
                "lightweightVlmCleanTextPreview": "Visible seatbelt evidence is not conclusive from this frame.",
            }
            return "Visible seatbelt evidence is not conclusive from this frame."

    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "lightweight_256m")
    monkeypatch.setattr(pipeline_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_frames", 3)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_evidence_frames", 3)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_job_timeout_seconds", 60)
    monkeypatch.setattr(pipeline_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((100, 100, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "imwrite_rgb", lambda path, image: None)  # noqa: ARG005

    fake_vlm = FakeVlmWorker()
    pipeline = pipeline_module.SafeTracePipeline(
        embedder=object(),
        index=object(),
        detector=object(),
        segmenter=pipeline_module.CoarseMaskSegmenter(),
        vlm=fake_vlm,
    )
    precomputed = {"detections": [make_detection("person")], "violations": [make_violation()]}

    first = pipeline.analyze_frame(tmp_path / "first.jpg", precomputed=precomputed)
    first_diag = dict(pipeline._last_lightweight_vlm_frame_diagnostics)
    second = pipeline.analyze_frame(tmp_path / "second.jpg", precomputed=precomputed)

    assert fake_vlm.count == 1
    assert first.explanation_source == "rule_template_plus_lightweight_vlm"
    assert "Safety review" in first.explanation
    assert "Visual review" in first.explanation
    assert first_diag["lightweightVlmExplanationSource"] == "vlm_lightweight"
    assert first_diag["finalExplanationSource"] == "rule_template_plus_lightweight_vlm"
    assert second.explanation_source == "rule_based"
    assert pipeline.component_diagnostics["totalVlmAttempts"] == 1
    assert pipeline.component_diagnostics["totalVlmSucceeded"] == 1
    assert pipeline.component_diagnostics["totalVlmDurationSeconds"] == 61.0
    assert pipeline.component_diagnostics["lightweightVlmDisabledReason"] == "local_visual_review_runtime_guard_elapsed"
    assert pipeline.component_diagnostics["vlmSkippedReasons"]["local_visual_review_runtime_guard_elapsed"] == 1


def test_pipeline_lightweight_worker_reports_not_attempted_reason_when_no_violations(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    constructed = {"worker": 0, "explained": 0}

    class FakeDetector:
        checkpoint = "checkpoints/yolov8s-seg.pt"

        def detect(self, image):  # noqa: ARG002
            return [make_detection("suitcase")]

    class FakeVlmWorker:
        provider = "vlm_lightweight_worker"
        enabled = True

        def __init__(self, *args, **kwargs):  # noqa: ARG002
            constructed["worker"] += 1
            self.last_explanation_source = "rule_based"
            self.last_diagnostics = {
                "lightweightVlmWorkerEnabled": True,
                "lightweightVlmWorkerTimeoutSeconds": 120,
                "lightweightVlmWorkerAttempted": False,
                "lightweightVlmWorkerSucceeded": False,
                "lightweightVlmWorkerTimedOut": False,
                "lightweightVlmWorkerExitCode": None,
                "lightweightVlmFallbackReason": None,
                "lightweightVlmExplanationSource": "rule_based",
            }

        def explain_violation(self, image, violations):  # noqa: ARG002
            constructed["explained"] += 1
            return "This should not run without violations."

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"placeholder")

    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "safe_mode_allow_mobilesam", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_enabled", "disabled")
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_worker_enabled", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "lightweight_256m")
    monkeypatch.setattr(pipeline_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "lightweight_vlm_worker_timeout_seconds", 120)
    monkeypatch.setattr(pipeline_module.SETTINGS, "top_k", 1)
    monkeypatch.setattr(pipeline_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline_module, "ClipEmbedder", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ClipEmbedder should not load")))
    monkeypatch.setattr(pipeline_module, "FaissIndex", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("FaissIndex should not load")))
    monkeypatch.setattr(pipeline_module, "VlmReasoner", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct VLM should not load")))
    monkeypatch.setattr(pipeline_module, "MobileSamSegmenter", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("MobileSAM should not load")))
    monkeypatch.setattr(pipeline_module, "LightweightVlmWorkerReasoner", FakeVlmWorker)
    monkeypatch.setattr(pipeline_module, "YoloDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((100, 100, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "imwrite_rgb", lambda path, image: None)  # noqa: ARG005
    monkeypatch.setattr(pipeline_module, "evaluate_rules", lambda detections: [])
    monkeypatch.setattr(
        pipeline_module.SafeTracePipeline,
        "_collect_input_frames",
        lambda self, inputs, fps, max_frames, **kwargs: ([frame], [], {"videos": 0, "images": 1}),  # noqa: ARG005
    )

    pipeline = pipeline_module.SafeTracePipeline()
    result = pipeline.run([frame], query="driver without seatbelt", fps=1.0, k=1)

    assert result
    assert constructed["worker"] == 1
    assert constructed["explained"] == 0
    assert pipeline.component_diagnostics["lightweightVlmWorkerEnabled"] is True
    assert pipeline.component_diagnostics["lightweightVlmWorkerAttempted"] is False
    assert pipeline.component_diagnostics["lightweightVlmFallbackReason"] == "no_eligible_violation"
    assert result[0]["search_metadata"]["lightweightVlmExplanation"]["lightweightVlmWorkerAttempted"] is False
    assert result[0]["search_metadata"]["lightweightVlmExplanation"]["lightweightVlmFallbackReason"] == "no_eligible_violation"


def test_pipeline_safe_mode_direct_run_completes_without_embeddings(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    class FakeDetector:
        checkpoint = "checkpoints/yolov8s-seg.pt"

        def detect(self, image):  # noqa: ARG002
            return []

    def fail_if_called(component):
        def inner(*args, **kwargs):  # noqa: ARG001
            raise AssertionError(f"safe mode should not construct {component}")
        return inner

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"placeholder")
    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "lightweight_256m")
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_worker_enabled", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "top_k", 1)
    monkeypatch.setattr(pipeline_module, "ClipEmbedder", fail_if_called("ClipEmbedder"))
    monkeypatch.setattr(pipeline_module, "FaissIndex", fail_if_called("FaissIndex"))
    monkeypatch.setattr(pipeline_module, "MobileSamSegmenter", fail_if_called("MobileSamSegmenter"))
    monkeypatch.setattr(pipeline_module, "VlmReasoner", fail_if_called("VlmReasoner"))
    monkeypatch.setattr(pipeline_module, "YoloDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((4, 4, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "evaluate_rules", lambda detections: [make_violation()])

    pipeline = pipeline_module.SafeTracePipeline()
    monkeypatch.setattr(
        pipeline,
        "_collect_input_frames",
        lambda inputs, fps, max_frames, **kwargs: ([frame], [], {"videos": 0, "images": 1}),  # noqa: ARG005
    )

    result = pipeline.run([frame], query="worker without helmet", fps=1.0, k=1)

    assert len(result) == 1
    assert result[0]["violations"][0]["name"] == "helmet_missing"
    assert result[0]["explanation_source"] == "rule_based"
    assert result[0]["search_metadata"]["embeddingBypassed"] is True
    assert pipeline.component_diagnostics["embeddingRequested"] is False
    assert pipeline.component_diagnostics["vlmAttempted"] is False
    assert result[0]["search_metadata"]["mode"] == "safe_ranked_frame_scan"
    assert result[0]["search_metadata"]["queryIntent"]["intents"]


def test_safe_mode_query_intent_parsing_for_driving_and_ppe_queries():
    seatbelt = parse_query_intent("driver not wearing seatbelt")
    phone = parse_query_intent("using mobile phone while driving")
    helmet = parse_query_intent("worker without hard hat")
    damage = parse_query_intent("damaged equipment near machinery")

    assert "seatbelt" in seatbelt.intents
    assert "person" in seatbelt.intents
    assert "seatbelt" in seatbelt.relevant_labels
    assert "phone" in phone.intents
    assert "hand" in phone.relevant_labels
    assert "helmet" in helmet.intents
    assert "helmet" in helmet.relevant_labels
    assert "damage" in damage.intents
    assert "machinery" in damage.intents


def test_safe_mode_ranking_prefers_violation_and_person_frames_over_road_only():
    intent = parse_query_intent("driver without seatbelt")
    road = score_frame_for_safe_mode(
        frame_path=Path("road.jpg"),
        frame_index=0,
        total_frames=3,
        detections=[make_detection("car", bbox=(45, 35, 60, 48))],
        violations=[],
        query_intent=intent,
        image_shape=(100, 100, 3),
    )
    person = score_frame_for_safe_mode(
        frame_path=Path("driver.jpg"),
        frame_index=1,
        total_frames=3,
        detections=[make_detection("person"), make_detection("torso", bbox=(20, 30, 80, 80))],
        violations=[Violation(name="seatbelt_missing", description="Missing seatbelt.", confidence=0.95)],
        query_intent=intent,
        image_shape=(100, 100, 3),
    )

    assert person.raw_score > road.raw_score
    assert "violation candidate was found" in person.reasons
    assert "road-only or distant-vehicle frame deprioritized" in road.reasons


def test_safe_mode_selection_keeps_late_high_scoring_violation_frame():
    intent = parse_query_intent("driver without seatbelt")
    candidates = []
    for index in range(6):
        detections = [make_detection("car", bbox=(45, 40, 55, 48))]
        violations = []
        if index == 5:
            detections = [make_detection("person"), make_detection("torso", bbox=(20, 30, 80, 80))]
            violations = [Violation(name="seatbelt_missing", description="Missing seatbelt.", confidence=0.9)]
        candidates.append(
            score_frame_for_safe_mode(
                frame_path=Path(f"frame_{index}.jpg"),
                frame_index=index,
                total_frames=6,
                detections=detections,
                violations=violations,
                query_intent=intent,
                image_shape=(100, 100, 3),
            )
        )

    selected = select_ranked_frames(candidates, top_k=3)

    assert any(candidate.frame_index == 5 for candidate in selected)
    assert selected[0].frame_index == 5
    assert selected[0].selected_for == "violation_evidence"


def test_safe_mode_temporal_diversity_fills_low_information_frames():
    intent = parse_query_intent("driver without seatbelt")
    candidates = [
        score_frame_for_safe_mode(
            frame_path=Path(f"empty_{index}.jpg"),
            frame_index=index,
            total_frames=9,
            detections=[],
            violations=[],
            query_intent=intent,
            image_shape=(100, 100, 3),
        )
        for index in range(9)
    ]

    selected = select_ranked_frames(candidates, top_k=3)
    selected_indexes = {candidate.frame_index for candidate in selected}

    assert len(selected) == 3
    assert 0 in selected_indexes
    assert max(selected_indexes) >= 7
    assert any(candidate.selected_for == "temporal_diversity" for candidate in selected)


def test_safe_mode_pipeline_outputs_ranking_diagnostics(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    frames = [tmp_path / f"frame_{index:06d}.jpg" for index in range(4)]
    for frame in frames:
        frame.write_bytes(b"placeholder")

    class FakeDetector:
        checkpoint = "checkpoints/yolov8s-seg.pt"

        def detect(self, image):  # noqa: ARG002
            current = str(pipeline_module._TEST_CURRENT_FRAME)
            if current.endswith("000003.jpg"):
                return [make_detection("person"), make_detection("torso", bbox=(20, 30, 80, 80))]
            return [make_detection("car", bbox=(45, 40, 55, 48))]

    def fake_imread(path):
        pipeline_module._TEST_CURRENT_FRAME = path
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def fake_write(path, image):  # noqa: ARG001
        return None

    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_worker_enabled", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "top_k", 2)
    monkeypatch.setattr(pipeline_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline_module, "ClipEmbedder", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ClipEmbedder should not load")))
    monkeypatch.setattr(pipeline_module, "FaissIndex", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("FaissIndex should not load")))
    monkeypatch.setattr(pipeline_module, "MobileSamSegmenter", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("MobileSAM should not load")))
    monkeypatch.setattr(pipeline_module, "VlmReasoner", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("VLM should not load")))
    monkeypatch.setattr(pipeline_module, "YoloDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "imread_rgb", fake_imread)
    monkeypatch.setattr(pipeline_module, "imwrite_rgb", fake_write)
    monkeypatch.setattr(
        pipeline_module.SafeTracePipeline,
        "_collect_input_frames",
        lambda self, inputs, fps, max_frames, **kwargs: (frames, [], {"videos": 0, "images": 4}),  # noqa: ARG005
    )

    pipeline = pipeline_module.SafeTracePipeline()
    result = pipeline.run([frames[0]], query="driver without seatbelt", fps=1.0, k=2)

    assert result[0]["frame_path"].endswith("000003.jpg")
    metadata = result[0]["search_metadata"]
    assert metadata["mode"] == "safe_ranked_frame_scan"
    assert metadata["embeddingBypassed"] is True
    assert metadata["queryIntent"]["intents"] == ["seatbelt", "person"]
    assert metadata["selectedFor"] == "violation_evidence"
    assert "rankingReason" in metadata
    assert "detectedObjectSummary" in metadata
    assert pipeline.component_diagnostics["safeRankingFramesScanned"] == 4
    assert pipeline.component_diagnostics["safeRankingSelectedFrames"] == 2


def test_pipeline_explicit_mobile_sam_enablement_still_constructs_segmenter(monkeypatch):
    import src.pipeline as pipeline_module

    class FakeIndex:
        def __init__(self, embedder):  # noqa: ARG002
            pass

    constructed = {"mobile_sam": False}

    class FakeMobileSam:
        available = True

        def __init__(self):
            constructed["mobile_sam"] = True

        def refine(self, image, detections):  # noqa: ARG002
            return detections

    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", False)
    monkeypatch.setattr(pipeline_module.SETTINGS, "mobile_sam_enabled", "auto")
    monkeypatch.setattr(pipeline_module, "ClipEmbedder", lambda: object())
    monkeypatch.setattr(pipeline_module, "FaissIndex", FakeIndex)
    monkeypatch.setattr(pipeline_module, "YoloDetector", NoopDetector)
    monkeypatch.setattr(pipeline_module, "MobileSamSegmenter", FakeMobileSam)

    pipeline = pipeline_module.SafeTracePipeline()

    assert constructed["mobile_sam"] is True
    assert pipeline.segmenter.available is True


def test_pipeline_labels_local_vlm_source_by_selected_profile(monkeypatch):
    import src.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "lightweight_256m")
    assert pipeline_module._profiled_explanation_source("vlm_local") == "vlm_lightweight"

    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "lightweight_512m")
    assert pipeline_module._profiled_explanation_source("vlm_local") == "vlm_lightweight"

    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "enhanced_2b")
    assert pipeline_module._profiled_explanation_source("vlm_local") == "vlm_enhanced"

    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "enhanced_3b")
    assert pipeline_module._profiled_explanation_source("vlm_local") == "vlm_enhanced"

    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "rule_based")
    assert pipeline_module._profiled_explanation_source("vlm_local") == "vlm_local"
    assert pipeline_module._profiled_explanation_source("rule_based") == "rule_based"


def test_ollama_vlm_success_uses_prompt_and_marks_source(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):  # noqa: ARG001
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeGenerateResponse({"response": "A worker is visible with uncertain helmet evidence due to glare."})

    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_enabled", "auto")
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_provider", "ollama")
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_ollama_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_model", "llava")
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_timeout_seconds", 2.0)
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_max_tokens", 90)
    monkeypatch.setattr(vlm_reasoner.httpx, "post", fake_post)

    reasoner = VlmReasoner(device="cpu", enabled=True)
    text = reasoner.explain_violation(np.zeros((4, 4, 3), dtype=np.uint8), [make_violation()])

    assert text.startswith("A worker is visible")
    assert reasoner.last_explanation_source == "vlm_ollama"
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert VLM_PROMPT in captured["json"]["prompt"]
    assert captured["json"]["model"] == "llava"
    assert captured["json"]["images"]


def test_auto_vlm_prefers_existing_local_provider_without_ollama(monkeypatch, tmp_path):
    def fail_if_ollama_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("auto should prefer the existing local VLM provider")

    def fake_load(self):
        self.provider = "local"
        self._loaded = True

    def fake_local_explain(self, image, violations, *, context=None):  # noqa: ARG001
        self.last_explanation_source = "vlm_local"
        return "Local provider sees limited helmet evidence; review glare and angle."

    model_dir = tmp_path / "vlm_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_enabled", "auto")
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_provider", "auto")
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_model_dir", model_dir)
    monkeypatch.setattr(vlm_reasoner, "_local_transformer_available", lambda model_dir: True)  # noqa: ARG005
    monkeypatch.setattr(vlm_reasoner.VlmReasoner, "_load_transformer_provider", fake_load)
    monkeypatch.setattr(vlm_reasoner.VlmReasoner, "_explain_with_transformers", fake_local_explain)
    monkeypatch.setattr(vlm_reasoner.httpx, "post", fail_if_ollama_called)

    reasoner = VlmReasoner(device="cpu", enabled=True)
    text = reasoner.explain_violation(np.zeros((4, 4, 3), dtype=np.uint8), [make_violation()])

    assert reasoner.provider == "local"
    assert reasoner.last_explanation_source == "vlm_local"
    assert text.startswith("Local provider")


def test_auto_vlm_uses_rule_based_when_no_provider_is_available(monkeypatch, tmp_path):
    def fake_get(*args, **kwargs):  # noqa: ARG001
        raise httpx.ConnectError("ollama offline")

    def fail_if_generation_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("auto should not call Ollama generation when status is unavailable")

    missing_model_dir = tmp_path / "missing_vlm_model"
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_enabled", "auto")
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_provider", "auto")
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_model_dir", missing_model_dir)
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_ollama_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr(vlm_reasoner.httpx, "get", fake_get)
    monkeypatch.setattr(vlm_reasoner.httpx, "post", fail_if_generation_called)

    reasoner = VlmReasoner(device="cpu", enabled=True)
    text = reasoner.explain_violation(np.zeros((4, 4, 3), dtype=np.uint8), [make_violation()])

    assert reasoner.provider == "rule_based"
    assert reasoner.last_explanation_source == "rule_based"
    assert "Rule-based explanation" in text


def test_ollama_timeout_falls_back_to_rule_based(monkeypatch):
    def fake_post(*args, **kwargs):  # noqa: ARG001
        raise httpx.TimeoutException("slow local model")

    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_enabled", "auto")
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_provider", "ollama")
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_ollama_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr(vlm_reasoner.httpx, "post", fake_post)

    reasoner = VlmReasoner(device="cpu", enabled=True)
    text = reasoner.explain_violation(np.zeros((4, 4, 3), dtype=np.uint8), [make_violation()])

    assert reasoner.last_explanation_source == "rule_based"
    assert "Rule-based explanation" in text


def test_transformer_vlm_uses_chat_template_image_content(monkeypatch):
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_max_tokens", 16)
    processor = FakeChatTemplateProcessor()
    model = FakeLocalModel()
    reasoner = make_local_reasoner(processor, model)

    text = reasoner._explain_with_transformers(
        np.zeros((4, 4, 3), dtype=np.uint8),
        [make_violation()],
    )

    assert text.startswith("Visible helmet evidence")
    assert reasoner.last_explanation_source == "vlm_local"
    assert processor.messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": processor.messages[0]["content"][0]["image"]},
                {"type": "text", "text": reasoner._prompt_for([make_violation()])},
            ],
        }
    ]
    assert processor.template_kwargs["add_generation_prompt"] is True
    assert processor.call_kwargs["images"]
    assert isinstance(processor.call_kwargs["images"], list)
    assert "<image>" in processor.call_kwargs["text"]
    assert processor.decode_output_ids == [[4]]
    assert processor.decode_kwargs["skip_special_tokens"] is True
    assert model.generate_kwargs["max_new_tokens"] == 16
    assert model.generate_kwargs["do_sample"] is False


def test_enhanced_prompt_includes_profile_query_finding_context(monkeypatch):
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_profile", "enhanced_2b")
    reasoner = VlmReasoner(enabled=False)

    prompt = reasoner._prompt_for(
        [Violation(name="seatbelt_missing", description="Missing seatbelt", severity="medium", confidence=0.55)],
        context={
            "imageRegion": {"source": "detector_crop", "matchedLabels": ["person", "torso"]},
            "detections": [{"label": "person", "confidence": 0.91}],
            "analysisContext": {
                "profileLabel": "Seatbelt compliance",
                "effectiveQuery": "driver missing seatbelt",
                "findingName": "seatbelt_missing",
                "friendlyFindingName": "missing seatbelt",
                "ruleSupport": "torso_or_cabin_context",
                "reviewLevel": "review_candidate",
                "confidenceReason": "Belt path was not confirmed from the frame.",
            },
        },
    )

    assert "Selected use-case profile: Seatbelt compliance." in prompt
    assert "User query: driver missing seatbelt." in prompt
    assert "Rule finding under review: missing seatbelt." in prompt
    assert "Detector context: person 0.91." in prompt
    assert "Base finding context: review level review_candidate; rule support torso_or_cabin_context" in prompt
    assert "Focus: torso area and possible belt path." in prompt
    assert "helmet" not in prompt.lower()
    assert "worn / not visible" not in prompt
    assert "seatbelt status" not in prompt


def test_transformer_vlm_fallback_prompt_keeps_image_token(monkeypatch):
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_max_tokens", 16)
    processor = FakeNoTemplateProcessor()
    model = FakeLocalModel()
    reasoner = make_local_reasoner(processor, model)

    text = reasoner._explain_with_transformers(
        np.zeros((4, 4, 3), dtype=np.uint8),
        [make_violation()],
    )

    assert text.startswith("Visible helmet evidence")
    assert reasoner.last_explanation_source == "vlm_local"
    assert processor.call_kwargs["text"].startswith("<image>\n")
    assert isinstance(processor.call_kwargs["images"], list)


def test_transformer_vlm_generation_failure_returns_rule_based(monkeypatch):
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_max_tokens", 16)
    processor = FakeChatTemplateProcessor()
    model = FakeLocalModel(fail=True)
    reasoner = make_local_reasoner(processor, model)

    text = reasoner._explain_with_transformers(
        np.zeros((4, 4, 3), dtype=np.uint8),
        [make_violation()],
    )

    assert reasoner.last_explanation_source == "rule_based"
    assert "Rule-based explanation" in text


def test_transformer_vlm_timeout_returns_rule_based(monkeypatch):
    def fake_timeout(callable_obj, timeout_seconds):  # noqa: ARG001
        raise TimeoutError("slow local VLM")

    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_timeout_seconds", 0.01)
    monkeypatch.setattr(vlm_reasoner, "_run_with_timeout", fake_timeout)
    processor = FakeChatTemplateProcessor()
    model = FakeLocalModel()
    reasoner = make_local_reasoner(processor, model)

    text = reasoner._explain_with_transformers(
        np.zeros((4, 4, 3), dtype=np.uint8),
        [make_violation()],
    )

    assert reasoner.last_explanation_source == "rule_based"
    assert "Rule-based explanation" in text


def test_pipeline_caps_local_vlm_explanation_attempts(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    class FakeDetector:
        def detect(self, image):  # noqa: ARG002
            return []

    class FakeSegmenter:
        def refine(self, image, detections):  # noqa: ARG002
            return detections

    class FakeVlm:
        provider = "local"
        enabled = True

        def __init__(self):
            self.count = 0
            self.last_explanation_source = "rule_based"

        def explain_violation(self, image, violations):  # noqa: ARG002
            self.count += 1
            self.last_explanation_source = "vlm_local"
            return "Local VLM sees visible helmet evidence with uncertainty."

    fake_vlm = FakeVlm()
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_frames", 1)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "lightweight_256m")
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((4, 4, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "evaluate_rules", lambda detections: [make_violation()])

    pipeline = pipeline_module.SafeTracePipeline(
        embedder=object(),
        index=object(),
        detector=FakeDetector(),
        segmenter=FakeSegmenter(),
        vlm=fake_vlm,
    )

    first = pipeline.analyze_frame(tmp_path / "frame1.jpg")
    second = pipeline.analyze_frame(tmp_path / "frame2.jpg")

    assert fake_vlm.count == 1
    assert first.explanation_source == "rule_template_plus_lightweight_vlm"
    assert "Safety review" in first.explanation
    assert "Visual review" in first.explanation
    assert second.explanation_source == "rule_based"
    assert "Rule-based explanation" in second.explanation


def test_pipeline_default_local_vlm_frame_limit_attempts_selected_frames(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    class FakeDetector:
        def detect(self, image):  # noqa: ARG002
            return []

    class FakeSegmenter:
        def refine(self, image, detections):  # noqa: ARG002
            return detections

    class FakeVlm:
        provider = "local"
        enabled = True

        def __init__(self):
            self.count = 0
            self.last_explanation_source = "rule_based"

        def explain_violation(self, image, violations):  # noqa: ARG002
            self.count += 1
            self.last_explanation_source = "vlm_local"
            return "Local VLM sees visible helmet evidence with uncertainty."

    fake_vlm = FakeVlm()
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_frames", 5)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_evidence_frames", 5)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_job_timeout_seconds", 0)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "lightweight_512m")
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((4, 4, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "evaluate_rules", lambda detections: [make_violation()])

    pipeline = pipeline_module.SafeTracePipeline(
        embedder=object(),
        index=object(),
        detector=FakeDetector(),
        segmenter=FakeSegmenter(),
        vlm=fake_vlm,
    )

    first = pipeline.analyze_frame(tmp_path / "frame1.jpg")
    second = pipeline.analyze_frame(tmp_path / "frame2.jpg")

    assert fake_vlm.count == 2
    assert first.explanation_source == "rule_template_plus_lightweight_vlm"
    assert second.explanation_source == "rule_template_plus_lightweight_vlm"
    assert pipeline.component_diagnostics["vlmFrameLimit"] == 5
    assert pipeline.component_diagnostics["lightweightVlmJobTimeoutSeconds"] == 0
    assert "visual_review_frame_limit_reached" not in pipeline.component_diagnostics["vlmSkippedReasons"]


def test_pipeline_enhanced_mode_attempts_frames_without_shared_lightweight_budget(monkeypatch, tmp_path):
    import src.pipeline as pipeline_module

    class FakeEnhancedVlm:
        provider = "vlm_lightweight_worker"
        enabled = True

        def __init__(self):
            self.count = 0
            self.contexts = []
            self.last_explanation_source = "rule_based"
            self.last_diagnostics = {}

        def explain_violation(self, image, violations, *, detections=None, context=None):  # noqa: ARG002
            self.count += 1
            self.contexts.append(context)
            self.last_explanation_source = "vlm_enhanced"
            self.last_diagnostics = {
                "lightweightVlmWorkerEnabled": True,
                "lightweightVlmWorkerAttempted": True,
                "lightweightVlmWorkerSucceeded": True,
                "lightweightVlmWorkerTimedOut": False,
                "lightweightVlmWorkerExitCode": 0,
                "lightweightVlmFallbackReason": None,
                "lightweightVlmExplanationSource": "vlm_enhanced",
                "lightweightVlmModelProfile": "enhanced_2b",
                "lightweightVlmCleanTextPreview": "visible_evidence: torso partly visible visual_status: belt path unclear short_reason: occlusion limits certainty confidence_hint: medium",
                "lightweightVlmWorkerDurationSeconds": 1.5,
            }
            return "visible_evidence: torso partly visible visual_status: belt path unclear short_reason: occlusion limits certainty confidence_hint: medium"

    fake_vlm = FakeEnhancedVlm()
    monkeypatch.setattr(pipeline_module.SETTINGS, "analysis_safe_mode", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "enable_vlm", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_enabled", "true")
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_profile", "enhanced_2b")
    monkeypatch.setattr(pipeline_module.SETTINGS, "lightweight_vlm_worker_enabled", True)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_frames", 2)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_max_evidence_frames", 2)
    monkeypatch.setattr(pipeline_module.SETTINGS, "vlm_job_timeout_seconds", 0)
    monkeypatch.setattr(pipeline_module.SETTINGS, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline_module, "imread_rgb", lambda path: np.zeros((100, 100, 3), dtype=np.uint8))
    monkeypatch.setattr(pipeline_module, "imwrite_rgb", lambda path, image: None)  # noqa: ARG005

    pipeline = pipeline_module.SafeTracePipeline(
        embedder=object(),
        index=object(),
        detector=object(),
        segmenter=pipeline_module.CoarseMaskSegmenter(),
        vlm=fake_vlm,
        use_case_profile={
            "profileId": "seatbelt",
            "label": "Seatbelt compliance",
            "effectiveQuery": "driver missing seatbelt",
        },
        query_context="driver missing seatbelt",
    )
    precomputed = {
        "detections": [make_detection("person")],
        "violations": [
            Violation(
                name="seatbelt_missing",
                description="Possible missing seatbelt.",
                severity="medium",
                confidence=0.55,
                evidence={
                    "evidenceStrength": "review_candidate",
                    "ruleSupport": "torso_or_cabin_context",
                    "confidenceReason": "Belt path could not be confirmed.",
                },
            )
        ],
    }

    first = pipeline.analyze_frame(tmp_path / "frame1.jpg", precomputed=precomputed)
    second = pipeline.analyze_frame(tmp_path / "frame2.jpg", precomputed=precomputed)

    assert fake_vlm.count == 2
    assert first.explanation_source == "rule_template_plus_lightweight_plus_enhanced"
    assert second.explanation_source == "rule_template_plus_lightweight_plus_enhanced"
    assert pipeline.component_diagnostics["enhancedVlmLayerStatus"] == "succeeded"
    assert pipeline.component_diagnostics["lightweightVlmJobTimeoutSeconds"] == 0
    assert "visual_review_frame_limit_reached" not in pipeline.component_diagnostics["vlmSkippedReasons"]
    assert fake_vlm.contexts[0]["profileLabel"] == "Seatbelt compliance"
    assert fake_vlm.contexts[0]["effectiveQuery"] == "driver missing seatbelt"


def test_sanitize_vlm_output_removes_prompt_echo_and_role_labels():
    prompt = "Describe only visible safety evidence in this frame.\nFindings to inspect: helmet_missing."
    raw = "User: <image>\nDescribe only visible safety evidence in this frame.\nAssistant: A worker is visible without clear helmet evidence due to glare."

    clean = sanitize_vlm_output(raw, prompt)

    assert clean == "A worker is visible without clear helmet evidence due to glare."
    assert is_useful_vlm_output(clean)


def test_prompt_echo_unclear_output_falls_back_to_rule_based(monkeypatch):
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_max_tokens", 16)
    processor = FakeChatTemplateProcessor(
        "User: <image>\nDescribe only visible safety evidence in this frame.\nAssistant: Unclear."
    )
    model = FakeLocalModel()
    reasoner = make_local_reasoner(processor, model)

    text = reasoner._explain_with_transformers(
        np.zeros((4, 4, 3), dtype=np.uint8),
        [make_violation()],
    )

    assert reasoner.last_explanation_source == "rule_based"
    assert "Rule-based explanation" in text


def test_token_leaking_output_is_sanitized_when_still_useful(monkeypatch):
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_max_tokens", 16)
    processor = FakeChatTemplateProcessor(
        "<global-img> <row_1_col_1> A worker is visible without helmet evidence, but glare creates uncertainty."
    )
    model = FakeLocalModel()
    reasoner = make_local_reasoner(processor, model)

    text = reasoner._explain_with_transformers(
        np.zeros((4, 4, 3), dtype=np.uint8),
        [make_violation()],
    )

    assert reasoner.last_explanation_source == "vlm_local"
    assert text == "A worker is visible without helmet evidence, but glare creates uncertainty."
    assert "<" not in text


def test_generic_vlm_output_falls_back_to_rule_based(monkeypatch):
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_max_tokens", 16)
    processor = FakeChatTemplateProcessor("Safety evidence missing.")
    model = FakeLocalModel()
    reasoner = make_local_reasoner(processor, model)

    text = reasoner._explain_with_transformers(
        np.zeros((4, 4, 3), dtype=np.uint8),
        [make_violation()],
    )

    assert reasoner.last_explanation_source == "rule_based"
    assert "Rule-based explanation" in text


def test_generic_object_inventory_output_falls_back_to_rule_based(monkeypatch):
    monkeypatch.setattr(vlm_reasoner.SETTINGS, "vlm_max_tokens", 64)
    processor = FakeChatTemplateProcessor(
        "In this image we can see a vehicle, some objects on the table, "
        "some wires, some objects on the floor, some text on the image and some other objects."
    )
    model = FakeLocalModel()
    reasoner = make_local_reasoner(processor, model)

    text = reasoner._explain_with_transformers(
        np.zeros((4, 4, 3), dtype=np.uint8),
        [make_violation()],
    )

    assert reasoner.last_explanation_source == "rule_based"
    assert reasoner.last_quality_issue == "generic object inventory"
    assert reasoner.last_fallback_reason == "quality:generic object inventory"
    assert "Rule-based explanation" in text


def test_lightweight_vlm_evaluation_harness_covers_safety_scenarios():
    script = Path("scripts/evaluate_lightweight_vlm.py")
    content = script.read_text(encoding="utf-8")

    assert "tmp_vlm_eval" in content
    assert "seatbelt" in content
    assert "helmet" in content
    assert "phone" in content
    assert "generic_safety" in content
    assert "modelLoadSeconds" in content
    assert "generationSeconds" in content
    assert "accepted" in content


def test_profile_vlm_evaluation_harness_reports_missing_model_path(tmp_path):
    from scripts import evaluate_vlm_profile

    output_dir = tmp_path / "eval"
    missing_model = tmp_path / "missing-lightweight-512m"

    exit_code = evaluate_vlm_profile.main(
        [
            "--model-profile",
            "lightweight_512m",
            "--model-path",
            str(missing_model),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 2
    report = json.loads((output_dir / "vlm_profile_eval_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "model_path_missing"
    assert report["modelProfile"] == "lightweight_512m"
    assert report["summary"] == {"total": 0, "accepted": 0, "fallback": 0}
    assert (output_dir / "vlm_profile_eval_summary.csv").is_file()


def test_profile_vlm_evaluation_harness_dry_run_writes_json_and_csv_schema(tmp_path):
    from scripts import evaluate_vlm_profile

    output_dir = tmp_path / "eval"

    exit_code = evaluate_vlm_profile.main(
        [
            "--model-profile",
            "enhanced_3b",
            "--profile",
            "seatbelt",
            "--dry-run",
            "--synthetic-if-empty",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    report = json.loads((output_dir / "vlm_profile_eval_report.json").read_text(encoding="utf-8"))
    assert report["modelProfile"] == "enhanced_3b"
    assert report["dryRun"] is True
    assert report["summary"]["total"] == 1
    assert report["summary"]["accepted"] == 0
    row = report["runs"][0]
    assert row["reviewProfile"] == "seatbelt"
    assert "belt path is visible" in row["prompt"]
    assert "visible_evidence:" in row["prompt"]
    assert "seatbelt status" not in row["prompt"]
    assert row["accepted"] is False
    assert row["qualityReason"] == "dry_run"
    csv_text = (output_dir / "vlm_profile_eval_summary.csv").read_text(encoding="utf-8")
    assert "modelProfile,modelPath,imagePath,reviewProfile" in csv_text
    assert "enhanced_3b" in csv_text
