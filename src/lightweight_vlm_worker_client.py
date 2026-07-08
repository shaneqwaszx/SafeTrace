"""Parent-process client for crash-isolated lightweight VLM explanations."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from PIL import Image

from .config import SETTINGS
from .device_gateway import device_gateway_payload, selected_component_device
from .schemas import Detection, Violation
from .utils import imwrite_rgb, resolve_device
from .vlm_reasoner import RULE_BASED_PROVIDER, RuleBasedReasoner, vlm_output_quality_issue

logger = logging.getLogger("safetrace.vlm.worker")

_SEATBELT_CROP_LABELS = {"person", "driver", "occupant", "torso", "body", "hand", "hands"}
_HELMET_CROP_LABELS = {"person", "worker", "head", "helmet", "hardhat", "hard hat"}
_PHONE_CROP_LABELS = {"person", "driver", "occupant", "hand", "hands", "phone", "cell phone", "mobile"}


def _preview(text: str | None, *, limit: int = 240) -> str | None:
    if not text:
        return None
    return " ".join(str(text).split())[:limit]


def _disabled_mode(value: str | None) -> bool:
    return (value or "").strip().lower() in {"0", "false", "no", "off", "disabled", "none"}


def _worker_dir() -> Path:
    path = SETTINGS.data_dir / "lightweight_vlm_worker"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _lightweight_profile_id() -> str:
    profile = str(getattr(SETTINGS, "vlm_profile", "lightweight_512m") or "lightweight_512m").strip().lower()
    return profile if profile in {"lightweight_256m", "lightweight_512m"} else "lightweight_512m"


def _lightweight_profile_id_for_model_dir(model_dir: Path | None) -> str:
    leaf = (model_dir.name if model_dir else "").strip().lower()
    if leaf == "lightweight-256m":
        return "lightweight_256m"
    if leaf == "lightweight-512m":
        return "lightweight_512m"
    if leaf == "enhanced-2b":
        return "enhanced_2b"
    if leaf == "enhanced-3b":
        return "enhanced_3b"
    return _lightweight_profile_id()


def _lightweight_model_path() -> Path:
    profile = _lightweight_profile_id()
    configured = (
        getattr(SETTINGS, "vlm_lightweight_512m_model_path", SETTINGS.vlm_model_dir)
        if profile == "lightweight_512m"
        else getattr(SETTINGS, "vlm_lightweight_model_path", SETTINGS.vlm_model_dir)
    )
    path = Path(configured)
    if path.is_absolute():
        return path
    return SETTINGS.project_root / path


def _profile_model_path(profile: str) -> Path:
    configured = (
        getattr(SETTINGS, "vlm_lightweight_model_path", SETTINGS.vlm_model_dir)
        if profile == "lightweight_256m"
        else getattr(SETTINGS, "vlm_lightweight_512m_model_path", SETTINGS.vlm_model_dir)
    )
    path = Path(configured)
    return path if path.is_absolute() else SETTINGS.project_root / path


def _path_has_model_files(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}
    try:
        return any(
            child.is_file() and child.name.lower() in {"model.safetensors", "config.json", "preprocessor_config.json"}
            or child.suffix.lower() == ".safetensors"
            for child in path.iterdir()
        )
    except OSError:
        return False


def _select_lightweight_profile_and_path(device: str) -> tuple[str, Path, str]:
    primary = str(getattr(SETTINGS, "lightweight_vlm_primary", "auto") or "auto").strip().lower()
    cpu_prefer_256 = bool(getattr(SETTINGS, "lightweight_vlm_cpu_prefer_256m", True))
    installed_256 = _path_has_model_files(_profile_model_path("lightweight_256m"))
    installed_512 = _path_has_model_files(_profile_model_path("lightweight_512m"))
    if primary in {"256m", "lightweight_256m"}:
        return "lightweight_256m", _profile_model_path("lightweight_256m"), "configured_primary_256m"
    if primary in {"512m", "lightweight_512m"}:
        return "lightweight_512m", _profile_model_path("lightweight_512m"), "configured_primary_512m"
    if device == "cpu" and cpu_prefer_256 and installed_256:
        return "lightweight_256m", _profile_model_path("lightweight_256m"), "cpu_policy_prefers_256m"
    if installed_512:
        return "lightweight_512m", _profile_model_path("lightweight_512m"), "512m_installed"
    if installed_256:
        return "lightweight_256m", _profile_model_path("lightweight_256m"), "512m_unavailable_256m_fallback"
    return "lightweight_512m", _profile_model_path("lightweight_512m"), "no_lightweight_model_available"


def _violation_names(violations: Sequence[Violation]) -> str:
    return " ".join(f"{violation.name} {violation.description}" for violation in violations).lower()


def _relevant_labels_for(violations: Sequence[Violation]) -> set[str]:
    names = _violation_names(violations)
    labels: set[str] = set()
    if "seatbelt" in names or "seat belt" in names:
        labels.update(_SEATBELT_CROP_LABELS)
    if "helmet" in names or "hardhat" in names or "hard hat" in names or "ppe" in names:
        labels.update(_HELMET_CROP_LABELS)
    if "phone" in names or "distract" in names or "mobile" in names:
        labels.update(_PHONE_CROP_LABELS)
    if not labels:
        labels.update({"person", "worker", "driver", "occupant"})
    return labels


def _detection_payload(detections: Sequence[Detection] | None) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for detection in list(detections or [])[:12]:
        payload.append(
            {
                "label": detection.label,
                "rawLabel": detection.raw_label,
                "confidence": float(detection.confidence),
                "bbox": [float(value) for value in detection.bbox],
            }
        )
    return payload


def _analysis_context_payload(context: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    allowed = {
        "selectedUseCaseProfile",
        "profileLabel",
        "userQuery",
        "requestedQuery",
        "effectiveQuery",
        "findingName",
        "friendlyFindingName",
        "ruleConfidence",
        "reviewLevel",
        "ruleSupport",
        "confidenceReason",
    }
    payload = {
        key: value
        for key, value in context.items()
        if key in allowed and value not in (None, "")
    }
    use_case_profile = context.get("useCaseProfile")
    if isinstance(use_case_profile, dict):
        payload["useCaseProfile"] = {
            key: value
            for key, value in use_case_profile.items()
            if key
            in {
                "profileId",
                "label",
                "category",
                "description",
                "backendSupportLevel",
                "supportedChecks",
                "limitations",
                "requestedQuery",
                "effectiveQuery",
            }
            and value not in (None, "")
        }
    return payload


def _bbox_area(bbox: Sequence[float]) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _clamped_padded_bbox(bboxes: Sequence[Sequence[float]], shape: tuple[int, ...]) -> list[int] | None:
    if not bboxes or len(shape) < 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    x1 = min(float(box[0]) for box in bboxes)
    y1 = min(float(box[1]) for box in bboxes)
    x2 = max(float(box[2]) for box in bboxes)
    y2 = max(float(box[3]) for box in bboxes)
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * 0.25
    pad_y = box_h * 0.25
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    if right - left < 16 or bottom - top < 16:
        return None
    full_area = max(1, width * height)
    crop_area = (right - left) * (bottom - top)
    if crop_area / full_area > 0.92:
        return None
    return [left, top, right, bottom]


def _resize_for_vlm(image: np.ndarray, *, max_edge: int = 384) -> tuple[np.ndarray, Dict[str, Any]]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image, {"resized": False, "shape": [height, width]}
    scale = max_edge / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    resized = Image.fromarray(image).resize(new_size, Image.Resampling.BICUBIC)
    return np.asarray(resized), {
        "resized": True,
        "originalShape": [height, width],
        "resizedShape": [new_size[1], new_size[0]],
        "maxEdge": max_edge,
    }


class LightweightVlmWorkerReasoner:
    """Lightweight VLM through a subprocess, never in the API process."""

    provider = "vlm_lightweight_worker"

    def __init__(self, model_dir: str | Path | None = None, device: str | None = None) -> None:
        selected_device = device or selected_component_device(SETTINGS, "lightweightVlm")
        self.device = resolve_device(selected_device)
        if model_dir is None:
            self.profile_id, self.model_dir, self.selection_reason = _select_lightweight_profile_and_path(self.device)
        else:
            self.model_dir = Path(model_dir)
            self.profile_id = _lightweight_profile_id_for_model_dir(self.model_dir)
            self.selection_reason = "explicit_model_dir"
        self.timeout_seconds = max(
            1.0,
            float(getattr(SETTINGS, "lightweight_vlm_worker_timeout_seconds", 60.0) or 60.0),
        )
        self.enabled = bool(getattr(SETTINGS, "lightweight_vlm_worker_enabled", False)) and not _disabled_mode(
            getattr(SETTINGS, "vlm_enabled", "auto")
        )
        self._available = bool(self.enabled and self.model_dir.exists())
        self._fallback_reasoner = RuleBasedReasoner()
        self.last_explanation_source = RULE_BASED_PROVIDER
        self.last_diagnostics: Dict[str, Any] = self._diagnostics(
            attempted=False,
            succeeded=False,
            timed_out=False,
            exit_code=None,
            source="disabled" if not self.enabled else RULE_BASED_PROVIDER,
            reason=None if self._available else "model_missing" if self.enabled else "worker_disabled",
        )

    @property
    def available(self) -> bool:
        return self._available

    def _diagnostics(
        self,
        *,
        attempted: bool,
        succeeded: bool,
        timed_out: bool,
        exit_code: int | None,
        source: str,
        reason: str | None,
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        diagnostics = {
            "lightweightVlmWorkerEnabled": bool(self.enabled),
            "lightweightVlmWorkerTimeoutSeconds": self.timeout_seconds,
            "lightweightVlmWorkerAttempted": bool(attempted),
            "lightweightVlmWorkerSucceeded": bool(succeeded),
            "lightweightVlmWorkerTimedOut": bool(timed_out),
            "lightweightVlmWorkerExitCode": exit_code,
            "lightweightVlmFallbackReason": reason,
            "lightweightVlmExplanationSource": source,
            "lightweightVlmSelectedProfile": self.profile_id,
            "lightweightVlmSelectedModelPath": str(self.model_dir),
            "lightweightVlmSelectionReason": self.selection_reason,
            "lightweightVlmDevice": self.device,
            "lightweightVlmPrimaryPolicy": str(getattr(SETTINGS, "lightweight_vlm_primary", "auto") or "auto"),
            "lightweightVlmFallbackPolicy": str(getattr(SETTINGS, "lightweight_vlm_fallback", "256m") or "256m"),
            "deviceGatewayDecision": device_gateway_payload(SETTINGS).get("components", {}).get("lightweightVlm"),
        }
        if extra:
            diagnostics.update(extra)
        return diagnostics

    def _fallback(
        self,
        image: np.ndarray,
        violations: Sequence[Violation],
        *,
        reason: str,
        attempted: bool = True,
        timed_out: bool = False,
        exit_code: int | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> str:
        self.last_explanation_source = RULE_BASED_PROVIDER
        self.last_diagnostics = self._diagnostics(
            attempted=attempted,
            succeeded=False,
            timed_out=timed_out,
            exit_code=exit_code,
            source=RULE_BASED_PROVIDER if attempted else "disabled",
            reason=reason,
            extra=extra,
        )
        return self._fallback_reasoner.explain_violation(image, violations)

    def _should_try_256m_fallback(self, reason: str | None) -> bool:
        if self.profile_id != "lightweight_512m":
            return False
        fallback_policy = str(getattr(SETTINGS, "lightweight_vlm_fallback", "256m") or "256m").strip().lower()
        if fallback_policy not in {"256m", "lightweight_256m"}:
            return False
        if not _path_has_model_files(_profile_model_path("lightweight_256m")):
            return False
        reason_text = str(reason or "")
        return bool(
            reason_text == "worker_timeout"
            or reason_text == "generation_timeout"
            or reason_text == "VlmFallback"
            or reason_text.startswith("quality:")
        )

    def _try_256m_fallback(
        self,
        image: np.ndarray,
        violations: Sequence[Violation],
        *,
        detections: Sequence[Detection] | None,
        first_reason: str | None,
        context: Dict[str, Any] | None = None,
        first_extra: Dict[str, Any] | None = None,
    ) -> str | None:
        if not self._should_try_256m_fallback(first_reason):
            return None
        fallback_reasoner = LightweightVlmWorkerReasoner(
            model_dir=_profile_model_path("lightweight_256m"),
            device=self.device,
        )
        fallback_reasoner.timeout_seconds = self.timeout_seconds
        text = fallback_reasoner.explain_violation(image, violations, detections=detections, context=context)
        diagnostics = dict(fallback_reasoner.last_diagnostics)
        chain = [
            {
                "modelProfile": self.profile_id,
                "modelPath": str(self.model_dir),
                "reason": first_reason,
                **(first_extra or {}),
            },
            {
                "modelProfile": fallback_reasoner.profile_id,
                "modelPath": str(fallback_reasoner.model_dir),
                "reason": diagnostics.get("lightweightVlmFallbackReason"),
                "succeeded": diagnostics.get("lightweightVlmWorkerSucceeded"),
            },
        ]
        crop_region = diagnostics.get("lightweightVlmImageRegion") or {}
        if (
            not diagnostics.get("lightweightVlmWorkerSucceeded")
            and isinstance(crop_region, dict)
            and crop_region.get("source") == "detector_crop"
        ):
            full_frame_reasoner = LightweightVlmWorkerReasoner(
                model_dir=_profile_model_path("lightweight_256m"),
                device=self.device,
            )
            full_frame_reasoner.timeout_seconds = self.timeout_seconds
            text = full_frame_reasoner.explain_violation(image, violations, detections=None, context=context)
            diagnostics = dict(full_frame_reasoner.last_diagnostics)
            chain.append(
                {
                    "modelProfile": full_frame_reasoner.profile_id,
                    "modelPath": str(full_frame_reasoner.model_dir),
                    "reason": diagnostics.get("lightweightVlmFallbackReason"),
                    "succeeded": diagnostics.get("lightweightVlmWorkerSucceeded"),
                    "imageRegion": diagnostics.get("lightweightVlmImageRegion"),
                    "retry": "full_frame_after_crop_rejection",
                }
            )
            diagnostics["lightweightVlmFullFrameRetry"] = True
        diagnostics["lightweightVlmFallbackChain"] = chain
        diagnostics["lightweightVlmPrimaryAttemptedProfile"] = self.profile_id
        diagnostics["lightweightVlmPrimaryFallbackReason"] = first_reason
        self.last_explanation_source = fallback_reasoner.last_explanation_source
        if diagnostics.get("lightweightVlmWorkerSucceeded"):
            self.last_explanation_source = diagnostics.get("lightweightVlmExplanationSource") or "vlm_lightweight"
        self.last_diagnostics = diagnostics
        return text

    def _materialize_image(self, image) -> Path:
        if isinstance(image, (str, Path)):
            return Path(image)
        if not isinstance(image, np.ndarray):
            raise TypeError("Lightweight VLM worker requires an image path or numpy image array.")
        image_path = _worker_dir() / f"frame_{uuid.uuid4().hex}.jpg"
        imwrite_rgb(image_path, image)
        return image_path

    def _select_worker_image(
        self,
        image: np.ndarray,
        violations: Sequence[Violation],
        detections: Sequence[Detection] | None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        detection_list = list(detections or [])
        labels = _relevant_labels_for(violations)
        relevant = [
            detection
            for detection in detection_list
            if str(detection.label or "").strip().lower() in labels
        ]
        if not relevant:
            relevant = [
                detection
                for detection in detection_list
                if str(detection.label or "").strip().lower() in {"person", "worker", "driver", "occupant"}
            ]
        if not relevant:
            return image, {
                "source": "full_frame",
                "reason": "no_relevant_detection_box",
                "originalShape": list(image.shape[:2]),
            }

        relevant = sorted(relevant, key=lambda detection: _bbox_area(detection.bbox), reverse=True)[:3]
        bbox = _clamped_padded_bbox([detection.bbox for detection in relevant], image.shape)
        if bbox is None:
            return image, {
                "source": "full_frame",
                "reason": "crop_not_useful",
                "matchedLabels": sorted({detection.label for detection in relevant}),
                "originalShape": list(image.shape[:2]),
            }
        left, top, right, bottom = bbox
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            return image, {
                "source": "full_frame",
                "reason": "empty_crop",
                "matchedLabels": sorted({detection.label for detection in relevant}),
                "originalShape": list(image.shape[:2]),
            }
        return crop, {
            "source": "detector_crop",
            "bbox": bbox,
            "matchedLabels": sorted({detection.label for detection in relevant}),
            "originalShape": list(image.shape[:2]),
            "cropShape": list(crop.shape[:2]),
        }

    def _command(self, input_path: Path, output_path: Path) -> list[str]:
        app_root = Path(os.environ.get("SAFETRACE_APP_ROOT") or SETTINGS.project_root).resolve()
        if getattr(sys, "frozen", False):
            return [
                sys.executable,
                "--lightweight-vlm-worker",
                "--input-json",
                str(input_path),
                "--output-json",
                str(output_path),
                "--app-root",
                str(app_root),
            ]
        return [
            sys.executable,
            "-m",
            "src.lightweight_vlm_worker",
            "--input-json",
            str(input_path),
            "--output-json",
            str(output_path),
            "--app-root",
            str(app_root),
        ]

    def _request_payload(
        self,
        image_path: Path,
        violations: Sequence[Violation],
        *,
        detections: Sequence[Detection] | None = None,
        image_region: Dict[str, Any] | None = None,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return {
            "imagePath": str(image_path),
            "modelDir": str(self.model_dir),
            "device": self.device,
            "profile": self.profile_id,
            "maxTokens": max(24, min(int(getattr(SETTINGS, "vlm_max_tokens", 40) or 40), 64)),
            "timeoutSeconds": self.timeout_seconds,
            "generationTimeoutSeconds": max(10.0, min(self.timeout_seconds - 2.0, max(20.0, self.timeout_seconds - 10.0))),
            "imageRegion": image_region or {"source": "full_frame"},
            "detections": _detection_payload(detections),
            "analysisContext": _analysis_context_payload(context),
            "violations": [
                {
                    "name": violation.name,
                    "description": violation.description,
                    "severity": violation.severity,
                    "confidence": float(violation.confidence),
                }
                for violation in violations
            ],
        }

    def _worker_timing_extra(
        self,
        payload: Dict[str, Any] | None = None,
        *,
        duration_seconds: float | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> Dict[str, Any]:
        payload = payload or {}
        return {
            "lightweightVlmWorkerDurationSeconds": (
                round(float(duration_seconds), 3) if duration_seconds is not None else payload.get("workerDurationSeconds")
            ),
            "lightweightVlmWorkerModelLoadSeconds": payload.get("modelLoadSeconds"),
            "lightweightVlmWorkerGenerationSeconds": payload.get("generationSeconds"),
            "lightweightVlmWorkerStdoutPreview": _preview(stdout),
            "lightweightVlmWorkerStderrPreview": _preview(stderr),
        }

    def _consume_result(
        self,
        output_path: Path,
        image: np.ndarray,
        violations: Sequence[Violation],
        *,
        duration_seconds: float | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> str:
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return self._fallback(
                image,
                violations,
                reason=f"invalid_worker_json:{type(exc).__name__}",
                exit_code=0,
                extra=self._worker_timing_extra(duration_seconds=duration_seconds, stdout=stdout, stderr=stderr),
            )

        if not bool(payload.get("ok")):
            reason = str(payload.get("fallbackReason") or payload.get("errorType") or "worker_error")
            return self._fallback(
                image,
                violations,
                reason=reason,
                extra={
                    "lightweightVlmQualityIssue": payload.get("qualityIssue"),
                    "lightweightVlmRawTextPreview": payload.get("rawTextPreview"),
                    "lightweightVlmCleanTextPreview": payload.get("cleanTextPreview"),
                    "lightweightVlmGenerationTimeoutSeconds": payload.get("generationTimeoutSeconds"),
                    "lightweightVlmMaxTokens": payload.get("maxTokens"),
                    "lightweightVlmImageRegion": payload.get("imageRegion"),
                    "lightweightVlmAnalysisContext": payload.get("analysisContext"),
                    **self._worker_timing_extra(
                        payload,
                        duration_seconds=duration_seconds,
                        stdout=stdout,
                        stderr=stderr,
                    ),
                },
            )

        explanation = str(payload.get("explanation") or "").strip()
        quality_issue = vlm_output_quality_issue(explanation)
        if not explanation or quality_issue:
            return self._fallback(
                image,
                violations,
                reason=f"quality:{quality_issue or 'empty_output'}",
                extra={
                    "lightweightVlmQualityIssue": quality_issue or "empty_output",
                    "lightweightVlmCleanTextPreview": payload.get("cleanTextPreview") or explanation[:240],
                    "lightweightVlmRawTextPreview": payload.get("rawTextPreview"),
                    "lightweightVlmGenerationTimeoutSeconds": payload.get("generationTimeoutSeconds"),
                    "lightweightVlmMaxTokens": payload.get("maxTokens"),
                    "lightweightVlmImageRegion": payload.get("imageRegion"),
                    "lightweightVlmAnalysisContext": payload.get("analysisContext"),
                    **self._worker_timing_extra(
                        payload,
                        duration_seconds=duration_seconds,
                        stdout=stdout,
                        stderr=stderr,
                    ),
                },
            )

        model_profile = str(payload.get("modelProfile") or self.profile_id)
        accepted_source = "vlm_enhanced" if model_profile in {"enhanced_2b", "enhanced_3b"} else "vlm_lightweight"
        self.last_explanation_source = accepted_source
        self.last_diagnostics = self._diagnostics(
            attempted=True,
            succeeded=True,
            timed_out=False,
            exit_code=0,
            source=accepted_source,
            reason=None,
            extra={
                "lightweightVlmCleanTextPreview": payload.get("cleanTextPreview") or explanation[:240],
                "lightweightVlmGenerationTimeoutSeconds": payload.get("generationTimeoutSeconds"),
                "lightweightVlmMaxTokens": payload.get("maxTokens"),
                "lightweightVlmImageRegion": payload.get("imageRegion"),
                "lightweightVlmAnalysisContext": payload.get("analysisContext"),
                **self._worker_timing_extra(
                    payload,
                    duration_seconds=duration_seconds,
                    stdout=stdout,
                    stderr=stderr,
                ),
            },
        )
        self.last_diagnostics["lightweightVlmModelProfile"] = model_profile
        return explanation

    def explain_violation(
        self,
        image: np.ndarray,
        violations: Sequence[Violation],
        *,
        detections: Sequence[Detection] | None = None,
        context: Dict[str, Any] | None = None,
    ) -> str:
        if not violations:
            return self._fallback(image, violations, reason="no_violations", attempted=False)
        if not self.enabled:
            return self._fallback(image, violations, reason="worker_disabled", attempted=False)
        if not self.model_dir.exists():
            return self._fallback(image, violations, reason="model_missing", attempted=False)

        worker_started_at = time.perf_counter()
        try:
            worker_image = image
            image_region = {"source": "full_frame"}
            if isinstance(image, np.ndarray):
                worker_image, image_region = self._select_worker_image(image, violations, detections)
                worker_image, resize_metadata = _resize_for_vlm(worker_image)
                image_region = {**image_region, **resize_metadata}
            image_path = self._materialize_image(worker_image)
            work_dir = _worker_dir()
            request_path = work_dir / f"request_{uuid.uuid4().hex}.json"
            output_path = work_dir / f"result_{uuid.uuid4().hex}.json"
            request_payload = self._request_payload(
                image_path,
                violations,
                detections=detections,
                image_region=image_region,
                context=context,
            )
            request_path.write_text(
                json.dumps(request_payload, indent=2, default=str),
                encoding="utf-8",
            )
            env = os.environ.copy()
            app_root = Path(env.get("SAFETRACE_APP_ROOT") or SETTINGS.project_root).resolve()
            env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("SAFETRACE_DEVICE", "cpu")
            env["SAFETRACE_DEVICE"] = self.device
            env.setdefault("SAFETRACE_VLM_PROVIDER", "auto")
            env.setdefault("SAFETRACE_VLM_PROFILE", self.profile_id)
            env["SAFETRACE_VLM_MODEL_PATH"] = str(self.model_dir)
            if self.profile_id == "lightweight_512m":
                env["SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH"] = str(self.model_dir)
            elif self.profile_id == "enhanced_2b":
                env["SAFETRACE_VLM_ENHANCED_MODEL_PATH"] = str(self.model_dir)
            elif self.profile_id == "enhanced_3b":
                env["SAFETRACE_VLM_ENHANCED_3B_MODEL_PATH"] = str(self.model_dir)
            else:
                env["SAFETRACE_VLM_LIGHTWEIGHT_MODEL_PATH"] = str(self.model_dir)
            env.setdefault("SAFETRACE_VLM_MAX_TOKENS", str(request_payload["maxTokens"]))
            result = subprocess.run(
                self._command(request_path, output_path),
                cwd=str(app_root),
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            duration_seconds = time.perf_counter() - worker_started_at
            logger.warning("Lightweight VLM worker timed out after %.1fs; using rule-based fallback.", self.timeout_seconds)
            fallback_text = self._try_256m_fallback(
                image,
                violations,
                detections=detections,
                context=context,
                first_reason="worker_timeout",
                first_extra=self._worker_timing_extra(duration_seconds=duration_seconds),
            )
            if fallback_text is not None:
                return fallback_text
            return self._fallback(
                image,
                violations,
                reason="worker_timeout",
                timed_out=True,
                extra=self._worker_timing_extra(duration_seconds=duration_seconds),
            )
        except Exception as exc:  # pragma: no cover - defensive
            duration_seconds = time.perf_counter() - worker_started_at
            logger.warning("Lightweight VLM worker launch failed: %s", exc)
            return self._fallback(
                image,
                violations,
                reason=f"worker_launch_failed:{type(exc).__name__}",
                extra=self._worker_timing_extra(duration_seconds=duration_seconds),
            )

        duration_seconds = time.perf_counter() - worker_started_at

        if result.returncode != 0:
            reason = f"worker_exit_{result.returncode}"
            extra: Dict[str, Any] = self._worker_timing_extra(
                duration_seconds=duration_seconds,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            if output_path.exists():
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                    reason = str(payload.get("fallbackReason") or payload.get("errorType") or reason)
                    extra.update(
                        {
                            "lightweightVlmQualityIssue": payload.get("qualityIssue"),
                            "lightweightVlmRawTextPreview": payload.get("rawTextPreview"),
                            "lightweightVlmCleanTextPreview": payload.get("cleanTextPreview"),
                            "lightweightVlmGenerationTimeoutSeconds": payload.get("generationTimeoutSeconds"),
                            "lightweightVlmMaxTokens": payload.get("maxTokens"),
                            "lightweightVlmImageRegion": payload.get("imageRegion"),
                            **self._worker_timing_extra(
                                payload,
                                duration_seconds=duration_seconds,
                                stdout=result.stdout,
                                stderr=result.stderr,
                            ),
                        }
                    )
                except (OSError, json.JSONDecodeError):
                    pass
            fallback_text = self._try_256m_fallback(
                image,
                violations,
                detections=detections,
                context=context,
                first_reason=reason,
                first_extra=extra,
            )
            if fallback_text is not None:
                return fallback_text
            logger.warning("Lightweight VLM worker exited with %s; using rule-based fallback.", result.returncode)
            return self._fallback(image, violations, reason=reason, exit_code=result.returncode, extra=extra)

        return self._consume_result(
            output_path,
            image,
            violations,
            duration_seconds=duration_seconds,
            stdout=result.stdout,
            stderr=result.stderr,
        )
