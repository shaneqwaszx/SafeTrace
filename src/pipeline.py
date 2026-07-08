"""End-to-end SafeTrace pipeline.

Public entry points:
- :class:`SafeTracePipeline.ingest` — extract frames from videos / pass-through
  images, embed them, and (re)build the FAISS index.
- :func:`SafeTracePipeline.analyze_query` — semantic search → YOLO detection →
  MobileSAM refinement → rule evaluation → optional VLM explanation, returning
  a structured JSON-friendly list.
"""
from __future__ import annotations

import logging
import os
import inspect
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .clip_embedder import ClipEmbedder
from .config import SETTINGS
from .device_gateway import device_gateway_payload, selected_component_device
from .faiss_index import FaissIndex
from .lightweight_vlm_worker_client import LightweightVlmWorkerReasoner
from .mobile_sam_segmenter import MobileSamSegmenter
from .mobile_sam_worker_client import MobileSamWorkerSegmenter
from .rule_engine import evaluate as evaluate_rules
from .safe_frame_ranking import RankedFrameCandidate, parse_query_intent, score_frame_for_safe_mode, select_ranked_frames
from .schemas import FrameAnalysis
from .schemas import Violation
from .preprocessing import build_processing_metadata
from .utils import (
    collect_inputs,
    draw_overlays,
    extract_frames_with_metadata,
    imread_rgb,
    imwrite_rgb,
    is_video,
)
from .vlm_reasoner import RuleBasedReasoner, VlmReasoner
from .yolo_detector import YoloDetector

logger = logging.getLogger("safetrace.pipeline")

_DETECTOR_CACHE_LOCK = threading.Lock()
_DETECTOR_CACHE: Dict[tuple[str, str, float, float, str], YoloDetector] = {}


def _disabled_mode(value: str | None) -> bool:
    return (value or "").strip().lower() in {"0", "false", "no", "off", "disabled", "none"}


def _analysis_safe_mode() -> bool:
    return bool(getattr(SETTINGS, "analysis_safe_mode", False))


def _safe_mode_allows_mobile_sam() -> bool:
    return _analysis_safe_mode() and bool(getattr(SETTINGS, "safe_mode_allow_mobilesam", False))


def _mobile_sam_worker_enabled() -> bool:
    return bool(getattr(SETTINGS, "mobile_sam_worker_enabled", False))


def _lightweight_vlm_worker_enabled() -> bool:
    return bool(getattr(SETTINGS, "lightweight_vlm_worker_enabled", False))


def _configured_vlm_profile() -> str:
    return str(getattr(SETTINGS, "vlm_profile", "rule_based") or "rule_based").strip().lower()


def _is_lightweight_vlm_profile(profile: str | None = None) -> bool:
    return (profile or _configured_vlm_profile()) in {"lightweight_256m", "lightweight_512m"}


def _is_enhanced_vlm_profile(profile: str | None = None) -> bool:
    return (profile or _configured_vlm_profile()) in {"enhanced_2b", "enhanced_3b"}


class CoarseMaskSegmenter:
    """Fast fallback segmenter that preserves detector-provided coarse masks."""

    available = False

    def refine(self, image, detections):  # noqa: ARG002
        for detection in detections:
            if detection.refined_mask is None and detection.coarse_mask is not None:
                detection.refined_mask = detection.coarse_mask
        return detections


def _mobile_sam_runtime_requested() -> bool:
    if _analysis_safe_mode():
        return _safe_mode_allows_mobile_sam() and not _disabled_mode(
            getattr(SETTINGS, "mobile_sam_enabled", "disabled")
        )
    return not _disabled_mode(getattr(SETTINGS, "mobile_sam_enabled", "disabled"))


def _vlm_runtime_requested() -> bool:
    if _analysis_safe_mode():
        return False
    mode = str(getattr(SETTINGS, "vlm_enabled", "auto") or "").strip().lower()
    return bool(SETTINGS.enable_vlm) and not _disabled_mode(mode)


def _lightweight_vlm_worker_runtime_requested() -> bool:
    if not _analysis_safe_mode():
        return False
    mode = str(getattr(SETTINGS, "vlm_enabled", "auto") or "").strip().lower()
    profile = _configured_vlm_profile()
    return bool(
        SETTINGS.enable_vlm
        and _lightweight_vlm_worker_enabled()
        and not _disabled_mode(mode)
        and _is_lightweight_vlm_profile(profile)
    )


def _profiled_explanation_source(source: Optional[str]) -> Optional[str]:
    if source != "vlm_local":
        return source
    profile = _configured_vlm_profile()
    if _is_lightweight_vlm_profile(profile):
        return "vlm_lightweight"
    if _is_enhanced_vlm_profile(profile):
        return "vlm_enhanced"
    return source


def _render_layered_vlm_explanation(
    *,
    rule_explanation: str,
    vlm_explanation: str,
    violations: Iterable[Any],
    layer_label: str = "Visual review",
) -> str:
    finding_names = ", ".join(str(getattr(violation, "name", "finding")) for violation in violations) or "SafeTrace finding"
    friendly_findings = finding_names.replace("_", " ")
    verifier_notes = []
    for violation in violations:
        evidence = dict(getattr(violation, "evidence", {}) or {})
        agreement = evidence.get("verifierAgreement")
        note = evidence.get("finalReviewerNote")
        if agreement:
            verifier_notes.append(
                f"- {str(getattr(violation, 'name', 'finding')).replace('_', ' ')}: visual review {agreement}. {note or ''}".strip()
            )
    return "\n".join(
        [
            "Safety review",
            f"Possible {friendly_findings}. Treat this as a review cue, not a final decision.",
            "",
            "Why this was flagged",
            (rule_explanation or "Base evidence is available for reviewer confirmation.").strip(),
            "",
            f"{layer_label}:",
            (vlm_explanation or "No accepted visual verifier output was produced.").strip(),
            "",
            "Review level",
            "\n".join(verifier_notes) if verifier_notes else "No visual review agreement metadata was produced.",
            "",
            "What to check in the original footage",
            "Confirm against the original footage when blur, camera angle, or occlusion affects visibility.",
        ]
    )


def _extract_vlm_field(text: str, field: str) -> str | None:
    pattern = (
        r"\b"
        + re.escape(field)
        + r"\s*:\s*(.*?)(?=\b(?:visible_evidence|visual_status|short_reason|confidence_hint)\s*:|$)"
    )
    match = re.search(pattern, text or "", flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    value = re.sub(r"\s+", " ", match.group(1)).strip(" .:-")
    return value or None


def _infer_verifier_agreement(violation: Violation, vlm_text: str) -> tuple[str, str]:
    text = re.sub(r"\s+", " ", (vlm_text or "").lower())
    name = str(getattr(violation, "name", "") or "").lower()

    uncertainty_terms = (
        "unclear",
        "cannot determine",
        "can't determine",
        "not clear",
        "occluded",
        "blur",
        "blocked",
        "review footage",
        "review original",
    )
    has_uncertainty = any(term in text for term in uncertainty_terms)

    if "seatbelt" in name:
        contradiction = any(
            phrase in text
            for phrase in (
                "wearing a belt",
                "belt is visible",
                "seatbelt is visible",
                "seat belt is visible",
                "belt path is visible",
                "belt crosses",
                "belt across the torso",
            )
        ) and not any(phrase in text for phrase in ("not visible", "no belt", "not wearing", "missing"))
        support = any(
            phrase in text
            for phrase in (
                "no belt",
                "no seatbelt",
                "not wearing",
                "missing seatbelt",
                "belt path is not visible",
                "belt is not visible",
                "seatbelt is not visible",
                "no clear belt",
            )
        )
        if contradiction:
            return "disagrees", "VLM saw a belt-like path or stated the person appears to be wearing a belt."
        if has_uncertainty:
            return "inconclusive", "VLM reported uncertainty, occlusion, blur, or insufficient visibility."
        if support:
            return "supports", "VLM did not see a clear belt path across the torso."

    if "helmet" in name or "ppe" in name:
        contradiction = any(
            phrase in text
            for phrase in ("helmet visible", "wearing a helmet", "hard hat visible", "ppe visible")
        ) and not any(phrase in text for phrase in ("not visible", "no helmet", "missing"))
        support = any(
            phrase in text
            for phrase in ("no helmet", "helmet not visible", "helmet is not visible", "missing helmet", "no hard hat")
        )
        if contradiction:
            return "disagrees", "VLM saw helmet/PPE-like evidence that conflicts with the missing-helmet rule."
        if has_uncertainty or "head not visible" in text:
            return "inconclusive", "VLM could not verify the head/PPE region clearly."
        if support:
            return "supports", "VLM did not see helmet/PPE evidence in the visible head area."

    if "phone" in name:
        contradiction = any(
            phrase in text
            for phrase in ("no phone", "phone not visible", "phone is not visible", "not using a phone")
        )
        support = any(
            phrase in text
            for phrase in ("phone visible", "phone near", "phone in hand", "phone by hand", "phone by face")
        )
        if contradiction:
            return "disagrees", "VLM did not see phone-use evidence."
        if has_uncertainty or "active use unclear" in text:
            return "inconclusive", "VLM could not determine active phone use from the frame."
        if support:
            return "supports", "VLM saw phone-like evidence near the hand, face, or driver area."

    if has_uncertainty:
        return "inconclusive", "VLM reported uncertainty or insufficient visibility."
    return "inconclusive", "VLM output was accepted as safety-relevant but did not clearly support or contradict this rule."


def _apply_vlm_verifier_agreement(violations: Iterable[Violation], vlm_text: str) -> dict[str, Any]:
    summary: dict[str, int] = {"supports": 0, "inconclusive": 0, "disagrees": 0}
    confidence_hint = _extract_vlm_field(vlm_text, "confidence_hint")
    changed: list[dict[str, Any]] = []
    for violation in violations:
        evidence = dict(getattr(violation, "evidence", {}) or {})
        agreement, reason = _infer_verifier_agreement(violation, vlm_text)
        summary[agreement] = summary.get(agreement, 0) + 1
        evidence["verifierAgreement"] = agreement
        evidence["verifierDisagreementReason"] = reason if agreement == "disagrees" else None
        evidence["verifierConfidenceHint"] = confidence_hint
        if agreement == "supports":
            evidence["finalReviewerNote"] = (
                evidence.get("finalReviewerNote")
                or "VLM verifier appears to support the rule finding; still confirm against the original footage."
            )
        elif agreement == "inconclusive":
            evidence["reviewRequired"] = True
            evidence["finalReviewerNote"] = (
                "VLM verifier was inconclusive; review the original footage before treating this as confirmed."
            )
            if evidence.get("evidenceStrength") in {"review_candidate", "insufficient_evidence"}:
                violation.confidence = min(float(violation.confidence), 0.5)
        elif agreement == "disagrees":
            evidence["reviewRequired"] = True
            evidence["finalReviewerNote"] = (
                "Visual review may disagree with the rule finding; do not treat this as automatic violation or automatic clearance."
            )
            if evidence.get("evidenceStrength") in {"review_candidate", "insufficient_evidence"} or float(violation.confidence) < 0.6:
                evidence["evidenceStrength"] = "review_candidate"
                evidence["confidenceReason"] = (
                    "Base evidence is weak and the visual verifier disagreed, so this finding is limited to reviewer triage."
                )
                violation.confidence = min(float(violation.confidence), 0.39)
                violation.severity = "medium" if violation.severity == "high" else violation.severity
        violation.evidence = evidence
        changed.append(
            {
                "name": violation.name,
                "verifierAgreement": agreement,
                "confidence": round(float(violation.confidence), 4),
                "evidenceStrength": evidence.get("evidenceStrength"),
            }
        )
    return {"counts": summary, "findings": changed}


def _configured_vlm_evidence_budget() -> int:
    if os.environ.get("SAFETRACE_VLM_FRAME_LIMIT") is not None:
        return max(0, int(getattr(SETTINGS, "vlm_max_evidence_frames", 5) or 0))
    if os.environ.get("SAFETRACE_VLM_MAX_EVIDENCE_FRAMES") is not None:
        return max(0, int(getattr(SETTINGS, "vlm_max_evidence_frames", 0) or 0))
    modern_limit = max(0, int(getattr(SETTINGS, "vlm_max_evidence_frames", 5) or 0))
    legacy_limit = max(0, int(getattr(SETTINGS, "vlm_max_frames", modern_limit) or 0))
    if legacy_limit != modern_limit:
        return min(legacy_limit, modern_limit)
    return modern_limit


def _normalize_visual_review_reason(reason: str | None) -> str | None:
    if not reason:
        return None
    mapping = {
        "vlm_budget_exhausted": "visual_review_frame_limit_reached",
        "vlm_frame_limit_reached": "visual_review_frame_limit_reached",
        "vlm_job_time_budget_exhausted": "local_visual_review_runtime_guard_elapsed",
        "vlm_runtime_guard_elapsed": "local_visual_review_runtime_guard_elapsed",
        "vlm_disabled_after_timeout": "local_visual_review_timeout",
        "worker_timeout": "local_visual_review_timeout",
        "generation_timeout": "local_visual_review_timeout",
        "timeout": "local_visual_review_timeout",
        "vlm_disabled_after_quality_failures": "local_visual_review_quality_guard",
    }
    text = str(reason)
    return mapping.get(text, text)


def _detector_factory_identity() -> str:
    factory = YoloDetector
    module = getattr(factory, "__module__", type(factory).__module__)
    qualname = getattr(factory, "__qualname__", getattr(factory, "__name__", type(factory).__qualname__))
    return f"{module}.{qualname}:{id(factory)}"


def _valid_yolo_detector(detector: Any) -> bool:
    return callable(getattr(detector, "detect", None))


def clear_yolo_detector_cache() -> None:
    with _DETECTOR_CACHE_LOCK:
        _DETECTOR_CACHE.clear()


def yolo_detector_cache_info() -> dict[str, Any]:
    with _DETECTOR_CACHE_LOCK:
        return {
            "size": len(_DETECTOR_CACHE),
            "keys": [str(key) for key in _DETECTOR_CACHE.keys()],
        }


def _detector_cache_key() -> tuple[str, str, float, float, str]:
    ckpt = Path(SETTINGS.yolo_checkpoint)
    fallback = Path(SETTINGS.yolo_fallback_checkpoint)
    chosen = ckpt if ckpt.exists() else fallback if fallback.exists() else ckpt
    return (
        str(chosen.resolve()),
        str(SETTINGS.device),
        float(SETTINGS.yolo_conf_threshold),
        float(SETTINGS.yolo_iou_threshold),
        _detector_factory_identity(),
    )


def _cached_yolo_detector() -> tuple[YoloDetector, bool]:
    key = _detector_cache_key()
    with _DETECTOR_CACHE_LOCK:
        detector = _DETECTOR_CACHE.get(key)
        if detector is not None:
            if not _valid_yolo_detector(detector):
                logger.warning("Evicting invalid cached YOLO detector for key %s", key)
                _DETECTOR_CACHE.pop(key, None)
            else:
                return detector, True
        detector = YoloDetector()
        if not _valid_yolo_detector(detector):
            raise TypeError("YoloDetector factory produced an object without a callable detect() method")
        _DETECTOR_CACHE[key] = detector
        return detector, False


class SafeTracePipeline:
    def __init__(
        self,
        embedder: Optional[ClipEmbedder] = None,
        index: Optional[FaissIndex] = None,
        detector: Optional[YoloDetector] = None,
        segmenter: Optional[MobileSamSegmenter] = None,
        vlm: Optional[VlmReasoner] = None,
        use_case_profile: Optional[Dict[str, Any]] = None,
        query_context: str = "",
    ) -> None:
        self.use_case_profile: Dict[str, Any] = dict(use_case_profile or {})
        self.query_context = query_context
        self.safe_mode = _analysis_safe_mode()
        safe_mode_mobile_sam_allowed = _safe_mode_allows_mobile_sam()
        mobile_sam_requested = _mobile_sam_runtime_requested()
        mobile_sam_worker_enabled = bool(mobile_sam_requested and _mobile_sam_worker_enabled())
        lightweight_vlm_worker_requested = _lightweight_vlm_worker_runtime_requested()
        vlm_enabled_mode = str(getattr(SETTINGS, "vlm_enabled", "auto") or "").strip().lower()
        enhanced_vlm_worker_requested = bool(
            self.safe_mode
            and SETTINGS.enable_vlm
            and not _disabled_mode(vlm_enabled_mode)
            and _is_enhanced_vlm_profile(_configured_vlm_profile())
        )
        self._vlm_evidence_budget = _configured_vlm_evidence_budget()
        self._vlm_job_time_budget_seconds = max(
            0.0,
            float(getattr(SETTINGS, "vlm_job_timeout_seconds", 60.0) or 0.0),
        )
        self._vlm_total_duration_seconds = 0.0
        self._vlm_quality_failures = 0
        self._vlm_disabled_reason: str | None = None
        self._vlm_max_quality_failures = max(
            0,
            int(getattr(SETTINGS, "vlm_max_quality_failures", 1) or 0),
        )
        self._last_lightweight_vlm_frame_diagnostics: Dict[str, Any] = {}
        self.component_diagnostics: Dict = {
            "safeMode": self.safe_mode,
            "device": selected_component_device(SETTINGS, "detector"),
            "requestedVisualExplanationMode": getattr(SETTINGS, "vlm_profile", "rule_based"),
            "effectiveExplanationMode": (
                _configured_vlm_profile()
                if lightweight_vlm_worker_requested or enhanced_vlm_worker_requested
                else
                "rule_based_with_mobilesam"
                if self.safe_mode and mobile_sam_requested
                else "rule_based"
                if self.safe_mode or not _vlm_runtime_requested()
                else getattr(SETTINGS, "vlm_profile", "rule_based")
            ),
            "vlmRequested": bool(getattr(SETTINGS, "enable_vlm", False)) and getattr(SETTINGS, "vlm_profile", "rule_based") != "rule_based",
            "vlmEffectiveEnabled": bool(
                _vlm_runtime_requested() or lightweight_vlm_worker_requested or enhanced_vlm_worker_requested
            ),
            "vlmAttempted": False,
            "vlmLoaded": False,
            "lightweightVlmWorkerEnabled": bool(lightweight_vlm_worker_requested or enhanced_vlm_worker_requested),
            "lightweightVlmWorkerTimeoutSeconds": float(getattr(SETTINGS, "lightweight_vlm_worker_timeout_seconds", 60.0) or 60.0),
            "lightweightVlmWorkerAttempted": False,
            "lightweightVlmWorkerSucceeded": False,
            "lightweightVlmWorkerTimedOut": False,
            "lightweightVlmWorkerExitCode": None,
            "lightweightVlmFallbackReason": None,
            "lightweightVlmExplanationSource": "disabled" if not lightweight_vlm_worker_requested else "rule_based",
            "vlmLayeringMode": "rule_based_base_with_optional_vlm_verifier",
            "baseExplanationSource": "rule_based",
            "finalExplanationSource": "rule_based",
            "deviceGatewayDecision": device_gateway_payload(SETTINGS),
            "lightweightVlmVerifierRole": "visual_verifier_layer",
            "lightweightVlmContributionAccepted": False,
            "enhancedVlmLayerStatus": "unavailable",
            "lightweightVlmEvidenceBudget": self._vlm_evidence_budget,
            "lightweightVlmFrameLimit": self._vlm_evidence_budget,
            "lightweightVlmJobTimeoutSeconds": self._vlm_job_time_budget_seconds,
            "lightweightVlmJobDurationSeconds": 0.0,
            "lightweightVlmJobRemainingSeconds": self._vlm_job_time_budget_seconds,
            "lightweightVlmDisabledReason": None,
            "lightweightVlmRuntimeGuardReason": None,
            "lightweightVlmFramesAttempted": 0,
            "lightweightVlmFramesSucceeded": 0,
            "lightweightVlmFramesSkipped": 0,
            "totalVlmAttempts": 0,
            "totalVlmSucceeded": 0,
            "totalVlmTimedOut": 0,
            "totalVlmRejected": 0,
            "totalVlmDurationSeconds": 0.0,
            "vlmEvidenceBudget": self._vlm_evidence_budget,
            "vlmFrameLimit": self._vlm_evidence_budget,
            "vlmSkippedReasons": {},
            "safeModeMobileSamAllowed": safe_mode_mobile_sam_allowed,
            "mobileSamRequested": mobile_sam_requested,
            "mobileSamAttempted": False,
            "mobileSamLoaded": False,
            "mobileSamFallbackReason": None,
            "mobileSamWorkerEnabled": mobile_sam_worker_enabled,
            "mobileSamWorkerTimeoutSeconds": float(getattr(SETTINGS, "mobile_sam_worker_timeout_seconds", 60.0) or 60.0),
            "mobileSamWorkerAttempted": False,
            "mobileSamWorkerSucceeded": False,
            "mobileSamWorkerTimedOut": False,
            "mobileSamWorkerExitCode": None,
            "mobileSamRefinementSource": "disabled",
            "embeddingRequested": not self.safe_mode,
            "embeddingLoaded": False,
            "detectorRequested": True,
            "detectorLoaded": False,
            "detectorCheckpointUsed": None,
            "currentPipelineStage": "initializing",
            "stageTimings": {},
            "safeFrameRankingEnabled": self.safe_mode,
            "safeFrameRankingStrategy": "object_rule_temporal" if self.safe_mode else None,
        }
        self._stage_started_at: Optional[float] = None
        self._active_stage: Optional[str] = None
        self._mark_stage("initializing")

        self.embedder = embedder
        self.index = index
        if not self.safe_mode:
            self._mark_stage("embedding_model_load")
            self.embedder = self.embedder or ClipEmbedder()
            self.component_diagnostics["embeddingLoaded"] = True
            self.index = self.index or FaissIndex(embedder=self.embedder)

        self._mark_stage("detector_load")
        detector_cache_hit = False
        if detector is not None:
            self.detector = detector
        else:
            self.detector, detector_cache_hit = _cached_yolo_detector()
        self.component_diagnostics["detectorLoaded"] = True
        self.component_diagnostics["detectorCacheHit"] = detector_cache_hit
        checkpoint = getattr(self.detector, "checkpoint", None)
        self.component_diagnostics["detectorCheckpointUsed"] = str(checkpoint) if checkpoint else None

        self._mark_stage("segmentation_setup")
        self._safe_mode_mobile_sam_segmenter = None
        if self.safe_mode:
            # Safe Mode ranking must remain lightweight; MobileSAM is lazy-loaded
            # only for already-selected evidence frames when explicitly allowed.
            self.segmenter = CoarseMaskSegmenter()
            self._safe_mode_mobile_sam_segmenter = segmenter
        else:
            self.segmenter = segmenter or (MobileSamSegmenter() if mobile_sam_requested else CoarseMaskSegmenter())
            self.component_diagnostics["mobileSamLoaded"] = bool(getattr(self.segmenter, "available", False))

        self._mark_stage("explanation_setup")
        if vlm is not None:
            self.vlm = vlm
        elif enhanced_vlm_worker_requested:
            self.vlm = LightweightVlmWorkerReasoner(
                model_dir=SETTINGS.vlm_enhanced_model_path,
                device=selected_component_device(SETTINGS, "enhancedVlm"),
            )
            self.component_diagnostics["lightweightVlmWorkerEnabled"] = bool(getattr(self.vlm, "enabled", False))
            self.component_diagnostics["lightweightVlmExplanationSource"] = (
                "rule_based" if getattr(self.vlm, "enabled", False) else "disabled"
            )
            self.component_diagnostics["enhancedVlmLayerStatus"] = "available"
        elif lightweight_vlm_worker_requested:
            self.vlm = LightweightVlmWorkerReasoner(device=selected_component_device(SETTINGS, "lightweightVlm"))
        elif _vlm_runtime_requested():
            self.vlm = VlmReasoner()
        else:
            self.vlm = RuleBasedReasoner()
        self.component_diagnostics["vlmLoaded"] = (
            bool(getattr(self.vlm, "enabled", False))
            and getattr(self.vlm, "provider", "rule_based") not in {"rule_based", "vlm_lightweight_worker"}
        )
        self.component_diagnostics["effectiveExplanationMode"] = (
            _configured_vlm_profile()
            if lightweight_vlm_worker_requested and bool(getattr(self.vlm, "enabled", False))
            else
            _profiled_explanation_source("vlm_local")
            if self.component_diagnostics["vlmLoaded"]
            else "rule_based"
        )
        self._vlm_explanations_remaining = self._vlm_evidence_budget
        self.last_processing_metadata: Dict = {}
        self._finish_active_stage()

    def _selected_frame_mobile_sam_segmenter(self):
        if self._safe_mode_mobile_sam_segmenter is None:
            if _mobile_sam_worker_enabled():
                self._safe_mode_mobile_sam_segmenter = MobileSamWorkerSegmenter(
                    device=selected_component_device(SETTINGS, "mobileSam")
                )
            else:
                self._safe_mode_mobile_sam_segmenter = MobileSamSegmenter(
                    device=selected_component_device(SETTINGS, "mobileSam")
                )
        return self._safe_mode_mobile_sam_segmenter

    def _merge_mobile_sam_diagnostics(self, segmenter) -> None:
        diagnostics = dict(getattr(segmenter, "last_diagnostics", {}) or {})
        if diagnostics:
            self.component_diagnostics.update(diagnostics)
        if diagnostics.get("mobileSamWorkerEnabled"):
            self.component_diagnostics["mobileSamLoaded"] = False
            self.component_diagnostics["mobileSamFallbackReason"] = diagnostics.get("mobileSamFallbackReason")
            if diagnostics.get("mobileSamWorkerSucceeded"):
                self.component_diagnostics["effectiveExplanationMode"] = "rule_based_with_mobilesam"
            else:
                self.component_diagnostics["effectiveExplanationMode"] = "rule_based"
            return
        if bool(getattr(segmenter, "available", False)):
            self.component_diagnostics["mobileSamLoaded"] = True
            self.component_diagnostics["mobileSamRefinementSource"] = "worker" if diagnostics.get("mobileSamWorkerSucceeded") else "fallback"

    def _merge_lightweight_vlm_diagnostics(self) -> None:
        diagnostics = dict(getattr(self.vlm, "last_diagnostics", {}) or {})
        if not diagnostics:
            return
        self.component_diagnostics.update(diagnostics)
        if diagnostics.get("lightweightVlmWorkerEnabled"):
            self.component_diagnostics["vlmLoaded"] = False
            self.component_diagnostics["vlmAttempted"] = bool(diagnostics.get("lightweightVlmWorkerAttempted"))
            if diagnostics.get("lightweightVlmWorkerSucceeded"):
                self.component_diagnostics["effectiveExplanationMode"] = str(
                    diagnostics.get("lightweightVlmModelProfile") or _configured_vlm_profile()
                )
            elif self.component_diagnostics.get("mobileSamWorkerSucceeded") or self.component_diagnostics.get("mobileSamLoaded"):
                self.component_diagnostics["effectiveExplanationMode"] = "rule_based_with_mobilesam"
            else:
                self.component_diagnostics["effectiveExplanationMode"] = "rule_based"

    def _remaining_vlm_job_time(self) -> float:
        if self._vlm_job_time_budget_seconds <= 0:
            return 0.0
        return max(0.0, self._vlm_job_time_budget_seconds - self._vlm_total_duration_seconds)

    def _record_vlm_skip_reason(self, reason: str | None) -> None:
        normalized = _normalize_visual_review_reason(reason)
        if not normalized:
            return
        skipped = dict(self.component_diagnostics.get("vlmSkippedReasons") or {})
        skipped[str(normalized)] = int(skipped.get(str(normalized), 0)) + 1
        self.component_diagnostics["vlmSkippedReasons"] = skipped

    def _vlm_pre_attempt_skip_reason(self) -> str | None:
        if self._vlm_disabled_reason:
            return self._vlm_disabled_reason
        if self._vlm_explanations_remaining <= 0:
            return "visual_review_frame_limit_reached"
        if self._vlm_job_time_budget_seconds > 0 and self._remaining_vlm_job_time() <= 0:
            self._vlm_disabled_reason = "local_visual_review_runtime_guard_elapsed"
            self.component_diagnostics["lightweightVlmDisabledReason"] = self._vlm_disabled_reason
            self.component_diagnostics["lightweightVlmRuntimeGuardReason"] = self._vlm_disabled_reason
            return self._vlm_disabled_reason
        return None

    def _update_vlm_reliability_state(
        self,
        *,
        attempted: bool,
        succeeded: bool,
        timed_out: bool,
        reason: str | None,
        duration_seconds: float | None,
    ) -> None:
        if duration_seconds is not None:
            self._vlm_total_duration_seconds += max(0.0, float(duration_seconds or 0.0))
            self.component_diagnostics["totalVlmDurationSeconds"] = round(self._vlm_total_duration_seconds, 3)
            self.component_diagnostics["lightweightVlmJobDurationSeconds"] = round(
                self._vlm_total_duration_seconds,
                3,
            )
            self.component_diagnostics["lightweightVlmJobRemainingSeconds"] = round(
                self._remaining_vlm_job_time(),
                3,
            )

        if attempted:
            self.component_diagnostics["totalVlmAttempts"] = (
                int(self.component_diagnostics.get("totalVlmAttempts") or 0) + 1
            )
        else:
            self._record_vlm_skip_reason(reason)
            return

        if succeeded:
            self.component_diagnostics["totalVlmSucceeded"] = (
                int(self.component_diagnostics.get("totalVlmSucceeded") or 0) + 1
            )
            if self._vlm_job_time_budget_seconds > 0 and self._remaining_vlm_job_time() <= 0:
                self._vlm_disabled_reason = "local_visual_review_runtime_guard_elapsed"
                self.component_diagnostics["lightweightVlmDisabledReason"] = self._vlm_disabled_reason
                self.component_diagnostics["lightweightVlmRuntimeGuardReason"] = self._vlm_disabled_reason
            return

        reason_text = str(_normalize_visual_review_reason(reason) or "")
        if timed_out or reason_text == "local_visual_review_timeout":
            self.component_diagnostics["totalVlmTimedOut"] = (
                int(self.component_diagnostics.get("totalVlmTimedOut") or 0) + 1
            )
            if bool(getattr(SETTINGS, "vlm_disable_after_timeout", True)):
                self._vlm_disabled_reason = "local_visual_review_timeout"
        if reason_text.startswith("quality:"):
            self.component_diagnostics["totalVlmRejected"] = (
                int(self.component_diagnostics.get("totalVlmRejected") or 0) + 1
            )
            self._vlm_quality_failures += 1
            if self._vlm_max_quality_failures and self._vlm_quality_failures >= self._vlm_max_quality_failures:
                self._vlm_disabled_reason = "local_visual_review_quality_guard"

        if self._vlm_job_time_budget_seconds > 0 and self._remaining_vlm_job_time() <= 0:
            self._vlm_disabled_reason = self._vlm_disabled_reason or "local_visual_review_runtime_guard_elapsed"

        self.component_diagnostics["lightweightVlmDisabledReason"] = self._vlm_disabled_reason
        self.component_diagnostics["lightweightVlmRuntimeGuardReason"] = self._vlm_disabled_reason

    def _lightweight_vlm_frame_diagnostics(
        self,
        *,
        attempted: bool,
        succeeded: bool,
        reason: str | None,
        source: str | None = None,
        timed_out: bool = False,
        exit_code: int | None = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_reason = _normalize_visual_review_reason(reason)
        accepted_source = source or ("vlm_lightweight" if succeeded else "rule_based")
        is_enhanced_source = accepted_source == "vlm_enhanced"
        final_source = "rule_based"
        if succeeded:
            final_source = (
                "rule_template_plus_lightweight_plus_enhanced"
                if is_enhanced_source
                else "rule_template_plus_lightweight_vlm"
            )
        diagnostics: Dict[str, Any] = {
            "lightweightVlmWorkerEnabled": bool(self.component_diagnostics.get("lightweightVlmWorkerEnabled")),
            "lightweightVlmWorkerTimeoutSeconds": self.component_diagnostics.get(
                "lightweightVlmWorkerTimeoutSeconds"
            ),
            "lightweightVlmWorkerAttempted": bool(attempted),
            "lightweightVlmWorkerSucceeded": bool(succeeded),
            "lightweightVlmWorkerTimedOut": bool(timed_out),
            "lightweightVlmWorkerExitCode": exit_code,
            "lightweightVlmFallbackReason": normalized_reason,
            "lightweightVlmRawFallbackReason": reason,
            "lightweightVlmExplanationSource": accepted_source,
            "vlmLayeringMode": "rule_based_base_with_optional_vlm_verifier",
            "baseExplanationSource": "rule_based",
            "lightweightVlmVerifierRole": "visual_verifier_layer",
            "lightweightVlmContributionAccepted": bool(succeeded and not is_enhanced_source),
            "enhancedVlmLayerStatus": (
                "succeeded"
                if bool(succeeded and is_enhanced_source)
                else self.component_diagnostics.get("enhancedVlmLayerStatus", "unavailable")
            ),
            "finalExplanationSource": final_source,
            "deviceGatewayDecision": self.component_diagnostics.get("deviceGatewayDecision"),
            "lightweightVlmEvidenceBudget": self.component_diagnostics.get("lightweightVlmEvidenceBudget"),
            "lightweightVlmFrameLimit": self.component_diagnostics.get("lightweightVlmFrameLimit"),
            "lightweightVlmRemainingBudget": max(0, int(self._vlm_explanations_remaining or 0)),
            "lightweightVlmRemainingFrameLimit": max(0, int(self._vlm_explanations_remaining or 0)),
            "lightweightVlmJobTimeoutSeconds": self.component_diagnostics.get("lightweightVlmJobTimeoutSeconds"),
            "lightweightVlmJobDurationSeconds": self.component_diagnostics.get("lightweightVlmJobDurationSeconds"),
            "lightweightVlmJobRemainingSeconds": self.component_diagnostics.get("lightweightVlmJobRemainingSeconds"),
            "lightweightVlmDisabledReason": self.component_diagnostics.get("lightweightVlmDisabledReason"),
            "lightweightVlmRuntimeGuardReason": self.component_diagnostics.get("lightweightVlmRuntimeGuardReason"),
        }
        if extra:
            diagnostics.update({key: value for key, value in extra.items() if value is not None})
        return diagnostics

    def _set_lightweight_vlm_frame_diagnostics(
        self,
        *,
        attempted: bool,
        succeeded: bool,
        reason: str | None,
        source: str | None = None,
        timed_out: bool = False,
        exit_code: int | None = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        diagnostics = self._lightweight_vlm_frame_diagnostics(
            attempted=attempted,
            succeeded=succeeded,
            reason=reason,
            source=source,
            timed_out=timed_out,
            exit_code=exit_code,
            extra=extra,
        )
        self._last_lightweight_vlm_frame_diagnostics = diagnostics
        had_attempt = bool(self.component_diagnostics.get("lightweightVlmWorkerAttempted"))
        had_success = bool(self.component_diagnostics.get("lightweightVlmWorkerSucceeded"))
        had_timeout = bool(self.component_diagnostics.get("lightweightVlmWorkerTimedOut"))
        self.component_diagnostics.update(diagnostics)
        self.component_diagnostics["lightweightVlmWorkerAttempted"] = bool(had_attempt or attempted)
        self.component_diagnostics["lightweightVlmWorkerSucceeded"] = bool(had_success or succeeded)
        self.component_diagnostics["lightweightVlmWorkerTimedOut"] = bool(had_timeout or timed_out)
        if had_success and not succeeded:
            self.component_diagnostics["lightweightVlmExplanationSource"] = "mixed"
        if attempted:
            self.component_diagnostics["lightweightVlmFramesAttempted"] = (
                int(self.component_diagnostics.get("lightweightVlmFramesAttempted") or 0) + 1
            )
        else:
            self.component_diagnostics["lightweightVlmFramesSkipped"] = (
                int(self.component_diagnostics.get("lightweightVlmFramesSkipped") or 0) + 1
            )
        if succeeded:
            self.component_diagnostics["lightweightVlmFramesSucceeded"] = (
                int(self.component_diagnostics.get("lightweightVlmFramesSucceeded") or 0) + 1
            )
        duration_value = None
        if extra and extra.get("lightweightVlmWorkerDurationSeconds") is not None:
            duration_value = float(extra.get("lightweightVlmWorkerDurationSeconds") or 0.0)
        self._update_vlm_reliability_state(
            attempted=attempted,
            succeeded=succeeded,
            timed_out=timed_out,
            reason=reason,
            duration_seconds=duration_value,
        )
        self._last_lightweight_vlm_frame_diagnostics["lightweightVlmJobDurationSeconds"] = (
            self.component_diagnostics.get("lightweightVlmJobDurationSeconds")
        )
        self._last_lightweight_vlm_frame_diagnostics["lightweightVlmJobRemainingSeconds"] = (
            self.component_diagnostics.get("lightweightVlmJobRemainingSeconds")
        )
        self._last_lightweight_vlm_frame_diagnostics["lightweightVlmDisabledReason"] = (
            self.component_diagnostics.get("lightweightVlmDisabledReason")
        )

    def _set_lightweight_vlm_frame_diagnostics_from_reasoner(self) -> None:
        diagnostics = dict(getattr(self.vlm, "last_diagnostics", {}) or {})
        reason = diagnostics.get("lightweightVlmFallbackReason")
        normalized_reason = _normalize_visual_review_reason(str(reason) if reason else None)
        source = diagnostics.get("lightweightVlmExplanationSource")
        self._set_lightweight_vlm_frame_diagnostics(
            attempted=bool(diagnostics.get("lightweightVlmWorkerAttempted")),
            succeeded=bool(diagnostics.get("lightweightVlmWorkerSucceeded")),
            reason=normalized_reason,
            source=str(source) if source else None,
            timed_out=bool(diagnostics.get("lightweightVlmWorkerTimedOut")),
            exit_code=diagnostics.get("lightweightVlmWorkerExitCode"),
            extra={
                "lightweightVlmQualityIssue": diagnostics.get("lightweightVlmQualityIssue"),
                "lightweightVlmRawFallbackReason": str(reason) if reason else None,
                "lightweightVlmRawTextPreview": diagnostics.get("lightweightVlmRawTextPreview"),
                "lightweightVlmCleanTextPreview": diagnostics.get("lightweightVlmCleanTextPreview"),
                "lightweightVlmGenerationTimeoutSeconds": diagnostics.get(
                    "lightweightVlmGenerationTimeoutSeconds"
                ),
                "lightweightVlmMaxTokens": diagnostics.get("lightweightVlmMaxTokens"),
                "lightweightVlmModelProfile": diagnostics.get("lightweightVlmModelProfile"),
                "lightweightVlmImageRegion": diagnostics.get("lightweightVlmImageRegion"),
                "lightweightVlmWorkerDurationSeconds": diagnostics.get("lightweightVlmWorkerDurationSeconds"),
                "lightweightVlmWorkerModelLoadSeconds": diagnostics.get("lightweightVlmWorkerModelLoadSeconds"),
                "lightweightVlmWorkerGenerationSeconds": diagnostics.get("lightweightVlmWorkerGenerationSeconds"),
                "lightweightVlmWorkerStdoutPreview": diagnostics.get("lightweightVlmWorkerStdoutPreview"),
                "lightweightVlmWorkerStderrPreview": diagnostics.get("lightweightVlmWorkerStderrPreview"),
            },
        )

    def _lightweight_vlm_frame_diagnostics_snapshot(self) -> Dict[str, Any]:
        if self._last_lightweight_vlm_frame_diagnostics:
            return dict(self._last_lightweight_vlm_frame_diagnostics)
        return self._lightweight_vlm_frame_diagnostics(
            attempted=False,
            succeeded=False,
            reason="mode_not_requested",
            source="rule_based",
        )

    def _vlm_analysis_context(self, violations: Iterable[Violation]) -> Dict[str, Any]:
        violation_list = list(violations or [])
        primary = violation_list[0] if violation_list else None
        evidence = dict(getattr(primary, "evidence", {}) or {}) if primary is not None else {}
        finding_name = str(getattr(primary, "name", "") or "")
        return {
            "useCaseProfile": dict(self.use_case_profile or {}),
            "selectedUseCaseProfile": self.use_case_profile.get("profileId"),
            "profileLabel": self.use_case_profile.get("label"),
            "userQuery": self.query_context,
            "requestedQuery": self.use_case_profile.get("requestedQuery") or self.query_context,
            "effectiveQuery": self.use_case_profile.get("effectiveQuery") or self.query_context,
            "findingName": finding_name,
            "friendlyFindingName": finding_name.replace("_", " "),
            "ruleConfidence": float(getattr(primary, "confidence", 0.0) or 0.0) if primary is not None else None,
            "reviewLevel": evidence.get("evidenceStrength"),
            "ruleSupport": evidence.get("ruleSupport"),
            "confidenceReason": evidence.get("confidenceReason"),
        }

    def _explain_with_configured_reasoner(self, image, violations, detections) -> str:
        try:
            signature = inspect.signature(self.vlm.explain_violation)
            parameters = signature.parameters
            supports_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            parameters = {}
            supports_kwargs = False

        kwargs: Dict[str, Any] = {}
        if supports_kwargs or "detections" in parameters:
            kwargs["detections"] = detections
        if supports_kwargs or "context" in parameters:
            kwargs["context"] = self._vlm_analysis_context(violations)
        if kwargs:
            return self.vlm.explain_violation(image, violations, **kwargs)
        return self.vlm.explain_violation(image, violations)

    def _refine_selected_safe_mode_frame(self, image, detections):
        if not self.safe_mode or not _mobile_sam_runtime_requested() or not detections:
            return detections
        self.component_diagnostics["mobileSamAttempted"] = True
        self._mark_stage("selected_mobilesam_refine")
        try:
            segmenter = self._selected_frame_mobile_sam_segmenter()
            if not bool(getattr(segmenter, "available", False)):
                self.component_diagnostics["mobileSamLoaded"] = False
                self._merge_mobile_sam_diagnostics(segmenter)
                self.component_diagnostics["mobileSamFallbackReason"] = (
                    self.component_diagnostics.get("mobileSamFallbackReason") or "unavailable"
                )
                self.component_diagnostics["mobileSamRefinementSource"] = "fallback"
                return CoarseMaskSegmenter().refine(image, detections)
            refined = segmenter.refine(image, detections)
            self._merge_mobile_sam_diagnostics(segmenter)
            if not self.component_diagnostics.get("mobileSamWorkerEnabled"):
                self.component_diagnostics["mobileSamLoaded"] = True
                self.component_diagnostics["effectiveExplanationMode"] = "rule_based_with_mobilesam"
            elif self.component_diagnostics.get("mobileSamWorkerSucceeded"):
                self.component_diagnostics["effectiveExplanationMode"] = "rule_based_with_mobilesam"
            return refined
        except Exception as exc:  # pragma: no cover - optional runtime safety net
            logger.warning("MobileSAM selected-frame refinement failed: %s", exc)
            self.component_diagnostics["mobileSamLoaded"] = False
            self.component_diagnostics["mobileSamFallbackReason"] = type(exc).__name__
            self.component_diagnostics["mobileSamRefinementSource"] = "fallback"
            return CoarseMaskSegmenter().refine(image, detections)

    def _mark_stage(self, stage: str) -> None:
        now = time.perf_counter()
        if self._active_stage is not None and self._stage_started_at is not None:
            timings = self.component_diagnostics.setdefault("stageTimings", {})
            timings[self._active_stage] = timings.get(self._active_stage, 0.0) + (now - self._stage_started_at)
        self._active_stage = stage
        self._stage_started_at = now
        self.component_diagnostics["currentPipelineStage"] = stage

    def _finish_active_stage(self) -> None:
        if self._active_stage is None:
            return
        self._mark_stage("idle")

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest(
        self,
        inputs: Iterable[str | Path],
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> List[Path]:
        """Convert videos to frames, accept images as-is, then build the FAISS index."""
        fps = fps or SETTINGS.frame_fps
        max_frames = max_frames or SETTINGS.max_frames

        self._mark_stage("frame_sampling")
        all_frames, sampling_runs, input_counts = self._collect_input_frames(inputs, fps=fps, max_frames=max_frames)

        if not all_frames:
            raise ValueError("No usable frames produced from the supplied inputs.")

        if self.index is None:
            raise RuntimeError("Embedding index is unavailable in safe analysis mode.")

        self._mark_stage("embedding_build")
        index_metadata = self.index.build_from_frames(all_frames)
        self.component_diagnostics["embeddingLoaded"] = True
        self.last_processing_metadata = build_processing_metadata(
            sampled_frame_count=len(all_frames),
            sampling_strategy="fixed_fps" if input_counts["videos"] else "image_inputs",
            fps=fps if input_counts["videos"] else None,
            max_frames=max_frames,
            embedding_batch_size=SETTINGS.embedding_batch_size,
            embedding_window_size=SETTINGS.embedding_window_size,
            embedding_window_stride=SETTINGS.embedding_window_stride,
            embedding_pooling_strategy=SETTINGS.embedding_pooling_strategy,
            processing_window_count=len(index_metadata),
            source_video_duration_seconds=(
                max(
                    (
                        float(run["sourceVideoDurationSeconds"])
                        for run in sampling_runs
                        if run.get("sourceVideoDurationSeconds") is not None
                    ),
                    default=None,
                )
            ),
            source_video_frame_count=sum(int(run.get("sourceVideoFrameCount") or 0) for run in sampling_runs) or None,
        )
        self.last_processing_metadata["inputVideoCount"] = input_counts["videos"]
        self.last_processing_metadata["inputImageCount"] = input_counts["images"]
        self.last_processing_metadata["samplingRuns"] = sampling_runs
        self._finish_active_stage()
        return all_frames

    def _collect_input_frames(
        self,
        inputs: Iterable[str | Path],
        *,
        fps: float,
        max_frames: int,
        uniform_over_video: bool = False,
    ) -> tuple[List[Path], List[Dict], Dict[str, int]]:
        videos, images = collect_inputs(inputs)
        all_frames: List[Path] = []
        sampling_runs: List[Dict] = []

        for vid in videos:
            frames, metadata = extract_frames_with_metadata(
                vid,
                SETTINGS.frames_dir,
                fps=fps,
                max_frames=max_frames,
                max_duration_seconds=SETTINGS.max_video_duration_seconds,
                uniform_over_video=uniform_over_video,
            )
            all_frames.extend(frames)
            sampling_runs.append(metadata)

        # Copy/standardize image inputs into the frames folder so the corpus
        # has one canonical location.
        for img in images:
            dst = SETTINGS.frames_dir / img.name
            if str(dst.resolve()) != str(img.resolve()):
                arr = imread_rgb(img)
                imwrite_rgb(dst, arr)
            all_frames.append(dst)

        return all_frames, sampling_runs, {"videos": len(videos), "images": len(images)}

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def analyze_frame(
        self,
        frame_path: str | Path,
        score: float = 1.0,
        *,
        precomputed: Optional[Dict[str, Any]] = None,
    ) -> FrameAnalysis:
        frame_path = Path(frame_path)
        if self.component_diagnostics.get("lightweightVlmWorkerEnabled"):
            self._last_lightweight_vlm_frame_diagnostics = self._lightweight_vlm_frame_diagnostics(
                attempted=False,
                succeeded=False,
                reason="pending",
                source="rule_based",
            )
        else:
            self._last_lightweight_vlm_frame_diagnostics = {}
        self._mark_stage("frame_read")
        image = imread_rgb(frame_path)

        if precomputed:
            detections = list(precomputed.get("detections") or [])
            violations = list(precomputed.get("violations") or [])
            detections = self._refine_selected_safe_mode_frame(image, detections)
        else:
            self._mark_stage("detector_inference")
            detections = self.detector.detect(image)
            self._mark_stage("segmentation_refine")
            detections = self.segmenter.refine(image, detections)
            self._mark_stage("rule_evaluation")
            violations = evaluate_rules(detections)

        explanation: Optional[str] = None
        explanation_source: Optional[str] = None
        if violations:
            self._mark_stage("rule_based_explanation")
            rule_explanation = RuleBasedReasoner().explain_violation(image, violations)
            explanation = rule_explanation
            explanation_source = "rule_based"
            try:
                vlm_is_active = bool(getattr(self.vlm, "enabled", False)) and getattr(self.vlm, "provider", "rule_based") != "rule_based"
                vlm_skip_reason = self._vlm_pre_attempt_skip_reason() if vlm_is_active else None
                if vlm_is_active and vlm_skip_reason:
                    if self.component_diagnostics.get("lightweightVlmWorkerEnabled"):
                        self._set_lightweight_vlm_frame_diagnostics(
                            attempted=False,
                            succeeded=False,
                            reason=vlm_skip_reason,
                            source="rule_based",
                        )
                else:
                    if vlm_is_active:
                        self.component_diagnostics["vlmAttempted"] = True
                        self._vlm_explanations_remaining -= 1
                        self._mark_stage(
                            "lightweight_vlm_worker_explanation"
                            if getattr(self.vlm, "provider", "") == "vlm_lightweight_worker"
                            else "vlm_explanation"
                        )
                        vlm_explanation = self._explain_with_configured_reasoner(image, violations, detections)
                        self._merge_lightweight_vlm_diagnostics()
                        if self.component_diagnostics.get("lightweightVlmWorkerEnabled"):
                            self._set_lightweight_vlm_frame_diagnostics_from_reasoner()
                        reasoner_source = _profiled_explanation_source(
                            getattr(self.vlm, "last_explanation_source", "rule_based")
                        )
                    else:
                        vlm_explanation = rule_explanation
                        reasoner_source = "rule_based"
                    if reasoner_source in {"vlm_lightweight", "vlm_enhanced"}:
                        verifier_agreement = _apply_vlm_verifier_agreement(violations, vlm_explanation)
                        final_explanation_source = (
                            "rule_template_plus_lightweight_vlm"
                            if reasoner_source == "vlm_lightweight"
                            else "rule_template_plus_lightweight_plus_enhanced"
                        )
                        explanation = _render_layered_vlm_explanation(
                            rule_explanation=rule_explanation,
                            vlm_explanation=vlm_explanation,
                            violations=violations,
                            layer_label="Visual review" if reasoner_source == "vlm_lightweight" else "Advanced visual review",
                        )
                        explanation_source = final_explanation_source
                        self.component_diagnostics["effectiveExplanationMode"] = _configured_vlm_profile()
                        self.component_diagnostics["finalExplanationSource"] = final_explanation_source
                        self.component_diagnostics["baseExplanationSource"] = "rule_based"
                        self.component_diagnostics["verifierAgreementSummary"] = verifier_agreement
                        if reasoner_source == "vlm_lightweight":
                            self.component_diagnostics["lightweightVlmContributionAccepted"] = True
                        if reasoner_source == "vlm_enhanced":
                            self.component_diagnostics["lightweightVlmContributionAccepted"] = False
                            self.component_diagnostics["enhancedVlmLayerStatus"] = "succeeded"
                        self._last_lightweight_vlm_frame_diagnostics.update(
                            {
                                "finalExplanationSource": final_explanation_source,
                                "baseExplanationSource": "rule_based",
                                "lightweightVlmContributionAccepted": reasoner_source == "vlm_lightweight",
                                "enhancedVlmLayerStatus": "succeeded" if reasoner_source == "vlm_enhanced" else "unavailable",
                                "verifierAgreementSummary": verifier_agreement,
                                "renderedTemplatePreview": explanation[:500],
                            }
                        )
                    if explanation_source == "rule_based" and not (
                        self.safe_mode and self.component_diagnostics.get("mobileSamLoaded")
                    ):
                        self.component_diagnostics["effectiveExplanationMode"] = "rule_based"
                        self.component_diagnostics["finalExplanationSource"] = "rule_based"
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("VLM explanation failed: %s", exc)
                if self.component_diagnostics.get("lightweightVlmWorkerEnabled"):
                    self._set_lightweight_vlm_frame_diagnostics(
                        attempted=True,
                        succeeded=False,
                        reason=f"worker_error:{type(exc).__name__}",
                        source="rule_based",
                    )
        elif self.component_diagnostics.get("lightweightVlmWorkerEnabled"):
            self._set_lightweight_vlm_frame_diagnostics(
                attempted=False,
                succeeded=False,
                reason="no_eligible_violation",
                source="rule_based",
            )

        annotated_path: Optional[str] = None
        if detections:
            self._mark_stage("annotation_write")
            annotated = draw_overlays(image, detections)
            ann_dir = SETTINGS.data_dir / "annotated"
            ann_dir.mkdir(parents=True, exist_ok=True)
            out_path = ann_dir / f"{frame_path.stem}_annotated.jpg"
            imwrite_rgb(out_path, annotated)
            annotated_path = str(out_path)
        self._finish_active_stage()

        return FrameAnalysis(
            frame_id=frame_path.stem,
            frame_path=str(frame_path),
            score=float(score),
            detections=detections,
            violations=violations,
            explanation=explanation,
            explanation_source=explanation_source,
            annotated_path=annotated_path,
        )

    def analyze_query(self, query: str, k: int | None = None) -> List[Dict]:
        """Run the full pipeline for a natural-language query."""
        self.query_context = query or self.query_context
        if self.safe_mode:
            logger.warning("Safe mode bypasses semantic search; use run() with inputs for direct frame analysis.")
            return []

        k = k or SETTINGS.top_k
        try:
            self._mark_stage("semantic_search")
            hits = self.index.semantic_search(query, k=k)
        except FileNotFoundError:
            logger.error("FAISS index not built. Call ingest() first.")
            return []

        payloads: List[Dict] = []
        for hit in hits:
            fa = self.analyze_frame(hit["frame_path"], score=hit.get("score", 0.0))
            payload = fa.to_dict()
            payload["search_metadata"] = {
                key: value
                for key, value in hit.items()
                if key not in {"frame_path", "frame_id", "score"}
            }
            payload["search_metadata"]["requestedVisualExplanationMode"] = self.component_diagnostics.get(
                "requestedVisualExplanationMode"
            )
            payload["search_metadata"]["actualExplanationMode"] = _profiled_explanation_source(
                payload.get("explanation_source") or "rule_based"
            )
            if self.component_diagnostics.get("lightweightVlmWorkerEnabled"):
                payload["search_metadata"]["lightweightVlmExplanation"] = (
                    self._lightweight_vlm_frame_diagnostics_snapshot()
                )
            if self.last_processing_metadata:
                payload["processing_metadata"] = self.last_processing_metadata
            payloads.append(payload)
        self._finish_active_stage()
        return payloads

    def _rank_safe_mode_frame(
        self,
        frame_path: Path,
        *,
        frame_index: int,
        total_frames: int,
        query_intent,
    ) -> RankedFrameCandidate:
        self._mark_stage("safe_ranking_frame_read")
        image = imread_rgb(frame_path)
        self._mark_stage("safe_ranking_detector_inference")
        detections = self.detector.detect(image)
        self._mark_stage("safe_ranking_rule_evaluation")
        detections = CoarseMaskSegmenter().refine(image, detections)
        violations = evaluate_rules(detections)
        return score_frame_for_safe_mode(
            frame_path=frame_path,
            frame_index=frame_index,
            total_frames=total_frames,
            detections=detections,
            violations=violations,
            query_intent=query_intent,
            image_shape=image.shape,
        )

    def analyze_frames_direct(self, frames: Iterable[Path], *, query: str = "", k: int | None = None) -> List[Dict]:
        """Safe-mode direct frame analysis without embeddings or semantic search."""
        self.query_context = query or self.query_context
        frame_list = list(frames)
        top_k = max(1, int(k or SETTINGS.top_k))
        query_intent = parse_query_intent(query)
        self._mark_stage("safe_frame_ranking")
        ranked_frames = [
            self._rank_safe_mode_frame(
                Path(frame_path),
                frame_index=index,
                total_frames=len(frame_list),
                query_intent=query_intent,
            )
            for index, frame_path in enumerate(frame_list)
        ]
        selected_frames = select_ranked_frames(ranked_frames, top_k=top_k)
        ranking_summary = {
            "strategy": "object_rule_temporal",
            "queryIntent": query_intent.to_dict(),
            "framesScanned": len(ranked_frames),
            "framesSelected": len(selected_frames),
            "topScores": [
                {
                    "frame": candidate.frame_path.name,
                    "score": round(candidate.raw_score, 4),
                    "selectedFor": candidate.selected_for,
                    "reason": candidate.ranking_reason,
                }
                for candidate in selected_frames[:10]
            ],
        }
        self.component_diagnostics["safeFrameRanking"] = ranking_summary
        self.component_diagnostics["safeRankingFramesScanned"] = len(ranked_frames)
        self.component_diagnostics["safeRankingSelectedFrames"] = len(selected_frames)
        self.component_diagnostics["safeRankingQueryIntent"] = query_intent.to_dict()
        if self.last_processing_metadata is not None:
            self.last_processing_metadata["safeFrameRanking"] = ranking_summary
            self.last_processing_metadata["processingWindowCount"] = len(selected_frames)

        payloads: List[Dict] = []
        for rank, candidate in enumerate(selected_frames, start=1):
            fa = self.analyze_frame(
                candidate.frame_path,
                score=candidate.normalized_score,
                precomputed={
                    "detections": candidate.detections,
                    "violations": candidate.violations,
                },
            )
            payload = fa.to_dict()
            payload["search_metadata"] = candidate.search_metadata(rank=rank)
            payload["search_metadata"]["requestedVisualExplanationMode"] = self.component_diagnostics.get(
                "requestedVisualExplanationMode"
            )
            payload["search_metadata"]["actualExplanationMode"] = _profiled_explanation_source(
                payload.get("explanation_source") or "rule_based"
            )
            if self.component_diagnostics.get("mobileSamRequested"):
                payload["search_metadata"]["mobileSamRefinement"] = {
                    "mobileSamWorkerEnabled": self.component_diagnostics.get("mobileSamWorkerEnabled"),
                    "mobileSamWorkerAttempted": self.component_diagnostics.get("mobileSamWorkerAttempted"),
                    "mobileSamWorkerSucceeded": self.component_diagnostics.get("mobileSamWorkerSucceeded"),
                    "mobileSamWorkerTimedOut": self.component_diagnostics.get("mobileSamWorkerTimedOut"),
                    "mobileSamWorkerExitCode": self.component_diagnostics.get("mobileSamWorkerExitCode"),
                    "mobileSamFallbackReason": self.component_diagnostics.get("mobileSamFallbackReason"),
                    "mobileSamRefinementSource": self.component_diagnostics.get("mobileSamRefinementSource"),
                }
            if self.component_diagnostics.get("lightweightVlmWorkerEnabled"):
                payload["search_metadata"]["lightweightVlmExplanation"] = (
                    self._lightweight_vlm_frame_diagnostics_snapshot()
                )
            if self.last_processing_metadata:
                payload["processing_metadata"] = self.last_processing_metadata
            payloads.append(payload)
        self._finish_active_stage()
        return payloads

    def run_safe_mode(
        self,
        inputs: Iterable[str | Path],
        query: str,
        fps: float | None = None,
        k: int | None = None,
    ) -> List[Dict]:
        fps = fps or SETTINGS.frame_fps
        max_frames = SETTINGS.max_frames
        self._mark_stage("safe_frame_sampling")
        all_frames, sampling_runs, input_counts = self._collect_input_frames(
            inputs,
            fps=fps,
            max_frames=max_frames,
            uniform_over_video=True,
        )
        if not all_frames:
            raise ValueError("No usable frames produced from the supplied inputs.")
        self.last_processing_metadata = build_processing_metadata(
            sampled_frame_count=len(all_frames),
            sampling_strategy="safe_object_ranked_frame_scan",
            fps=fps if input_counts["videos"] else None,
            max_frames=max_frames,
            embedding_batch_size=0,
            embedding_window_size=1,
            embedding_window_stride=1,
            embedding_pooling_strategy="mean",
            processing_window_count=min(len(all_frames), max(1, int(k or SETTINGS.top_k))),
            source_video_duration_seconds=(
                max(
                    (
                        float(run["sourceVideoDurationSeconds"])
                        for run in sampling_runs
                        if run.get("sourceVideoDurationSeconds") is not None
                    ),
                    default=None,
                )
            ),
            source_video_frame_count=sum(int(run.get("sourceVideoFrameCount") or 0) for run in sampling_runs) or None,
        )
        self.last_processing_metadata["inputVideoCount"] = input_counts["videos"]
        self.last_processing_metadata["inputImageCount"] = input_counts["images"]
        self.last_processing_metadata["samplingRuns"] = sampling_runs
        self.last_processing_metadata["safeMode"] = True
        self.last_processing_metadata["semanticSearch"] = False
        self.last_processing_metadata["embeddingBypassed"] = True
        return self.analyze_frames_direct(all_frames, query=query, k=k)

    # Convenience: full ingest + analyze in one call.
    def run(
        self,
        inputs: Iterable[str | Path],
        query: str,
        fps: float | None = None,
        k: int | None = None,
    ) -> List[Dict]:
        # Skip re-ingestion if the index already exists and no inputs are given.
        inputs_list = list(inputs)
        if self.safe_mode:
            return self.run_safe_mode(inputs_list, query=query, fps=fps, k=k)
        if inputs_list:
            self.ingest(inputs_list, fps=fps)
        return self.analyze_query(query, k=k)


# Module-level convenience for the spec signature.
_default_pipeline: Optional[SafeTracePipeline] = None


def get_pipeline() -> SafeTracePipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = SafeTracePipeline()
    return _default_pipeline


def analyze_query(query: str, k: int | None = None) -> List[Dict]:
    return get_pipeline().analyze_query(query, k=k)
