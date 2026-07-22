"""Job registry, manifest persistence, and lazy SafeTrace execution."""
from __future__ import annotations

import json
import hashlib
import copy
import logging
import os
import re
import shutil
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from types import MappingProxyType
from typing import Any, BinaryIO, Dict, Literal, Optional

from src.config import SETTINGS
from src.device_gateway import device_gateway_payload
from src.engine_metrics import build_engine_metrics
from src.api.adaptive_workers import AdaptiveWorkerManager

from .normalization import normalize_pipeline_results
from .result_ownership import (
    ResultOwnershipContext,
    ResultOwnershipError,
    stamp_result_ownership,
    validate_result_ownership,
)

JobState = Literal[
    "queued",
    "waiting_for_capacity",
    "waiting_for_job_slot",
    "waiting_for_gpu",
    "waiting_for_mobilesam",
    "waiting_for_vlm",
    "running_preprocess",
    "running_detector",
    "running_refinement",
    "running_report",
    "running",
    "retry_wait",
    "recovering",
    "paused",
    "completed",
    "failed",
    "cancelled",
]
VLM_PROFILE_RULE_BASED = "rule_based"
VLM_PROFILE_LIGHTWEIGHT = "lightweight_256m"
VLM_PROFILE_LIGHTWEIGHT_512M = "lightweight_512m"
VLM_PROFILE_ENHANCED = "enhanced_2b"
VLM_PROFILE_ENHANCED_3B = "enhanced_3b"
VLM_PROFILE_IDS = {
    VLM_PROFILE_RULE_BASED,
    VLM_PROFILE_LIGHTWEIGHT,
    VLM_PROFILE_LIGHTWEIGHT_512M,
    VLM_PROFILE_ENHANCED,
    VLM_PROFILE_ENHANCED_3B,
}
VLM_LIGHTWEIGHT_PROFILES = {VLM_PROFILE_LIGHTWEIGHT, VLM_PROFILE_LIGHTWEIGHT_512M}
VLM_SAFE_MODE_WORKER_PROFILES = {*VLM_LIGHTWEIGHT_PROFILES, VLM_PROFILE_ENHANCED}
MODE_LABELS = {
    VLM_PROFILE_RULE_BASED: "Fast Local Analysis",
    VLM_PROFILE_LIGHTWEIGHT: "Local VLM Assist",
    VLM_PROFILE_LIGHTWEIGHT_512M: "Local VLM Assist",
    VLM_PROFILE_ENHANCED: "Advanced GPU VLM Assist",
    VLM_PROFILE_ENHANCED_3B: "Advanced GPU VLM Assist",
}
MEDIA_EXTENSIONS = {
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".bmp": "image",
    ".webp": "image",
    ".mp4": "video",
    ".mov": "video",
    ".avi": "video",
    ".mkv": "video",
    ".webm": "video",
}
TERMINAL_STATES = {"completed", "failed", "cancelled"}
ACTIVE_STATES = {
    "queued", "waiting_for_capacity", "waiting_for_job_slot", "waiting_for_gpu", "waiting_for_mobilesam",
    "waiting_for_vlm", "running_preprocess", "running_detector", "running_refinement", "running_report",
    "running", "retry_wait", "recovering", "paused",
}
MANIFEST_FILENAME = "manifest.json"
RESULT_FILENAME = "result.json"
LOCK_FILENAME = "execution.lock"
PARTIAL_PIPELINE_FILENAME = "partial_pipeline_result.json"
ANALYSIS_HEARTBEAT_SECONDS = 8.0

# Global settings are legacy configuration consumed by several model wrappers.
# Compatible jobs can share one applied context; incompatible contexts wait until
# the active group drains instead of leaking device/profile state between jobs.
_PIPELINE_SETTINGS_CONDITION = threading.Condition(threading.RLock())
_PIPELINE_SETTINGS_SNAPSHOT: Dict[str, Any] | None = None
_PIPELINE_SETTINGS_KEY: tuple[Any, ...] | None = None
_PIPELINE_SETTINGS_ACTIVE = 0
logger = logging.getLogger("safetrace.api.jobs")
_MANIFEST_LOCKS: Dict[Path, threading.RLock] = {}
_MANIFEST_LOCKS_GUARD = threading.Lock()
_SEMAPHORE_LOCK = threading.Lock()
_ANALYSIS_SEMAPHORE: threading.BoundedSemaphore | None = None
_ANALYSIS_SEMAPHORE_LIMIT: int | None = None
_VLM_SEMAPHORE: threading.BoundedSemaphore | None = None
_VLM_SEMAPHORE_LIMIT: int | None = None
_GPU_SEMAPHORE: threading.BoundedSemaphore | None = None
_GPU_SEMAPHORE_LIMIT: int | None = None
_SCHEDULER_SEQUENCE = 0
_SCHEDULER_SEQUENCE_LOCK = threading.Lock()
_ADAPTIVE_WORKER_MANAGER = AdaptiveWorkerManager()


class UploadValidationError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class PipelineTimeoutError(TimeoutError):
    """Raised when an analysis job exceeds the configured backend timeout."""


class JobManifestPersistenceError(OSError):
    """Raised when a job manifest/result JSON file cannot be atomically replaced."""


class JobRestartConflictError(RuntimeError):
    """Raised when fresh restart cannot safely take ownership of a job."""


@dataclass
class AnalysisSettings:
    fps: float
    top_k: int
    enable_vlm: bool
    device: str
    vlm_profile: str = VLM_PROFILE_RULE_BASED
    vlm_enabled: bool = False
    safe_mode: bool = False
    use_case_profile: Dict[str, Any] = field(default_factory=dict)
    review_mode: str = "fast_local"


@dataclass
class JobRecord:
    job_id: str
    status: JobState
    progress: float
    current_step: str
    error: Optional[str]
    query: str
    settings: AnalysisSettings
    original_filename: str
    media_type: str
    size_bytes: int
    job_dir: Path
    upload_path: Path
    output_dir: Path
    result: Optional[Dict[str, Any]] = None
    result_path: Optional[Path] = None
    technical_error: Optional[str] = None
    error_type: Optional[str] = None
    media_files: Dict[str, Path] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    persistence_warning: Optional[str] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_attempts: int = 0
    next_retry_at: Optional[datetime] = None
    recovery_action: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @property
    def manifest_path(self) -> Path:
        return self.job_dir / MANIFEST_FILENAME

    def timing_payload(self, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        reference = now or _utc_now()
        end_time = self.finished_at if self.status in TERMINAL_STATES else reference
        elapsed_anchor = self.created_at or self.started_at or reference
        elapsed_seconds = max(0.0, (end_time - elapsed_anchor).total_seconds())
        queue_end = self.started_at or (self.finished_at if self.status in TERMINAL_STATES else reference)
        queue_wait_seconds = (
            max(0.0, (queue_end - self.created_at).total_seconds())
            if self.created_at and queue_end
            else None
        )
        analysis_runtime_seconds = (
            max(0.0, (end_time - self.started_at).total_seconds())
            if self.started_at and end_time
            else None
        )
        completed_at = self.finished_at if self.status == "completed" else None
        failed_at = self.finished_at if self.status == "failed" else None
        cancelled_at = self.finished_at if self.status == "cancelled" else None
        stage_started = _parse_datetime(self.metrics.get("stageStartedAt"))
        stage_elapsed_seconds = (
            max(0.0, ((self.finished_at if self.status in TERMINAL_STATES else reference) - stage_started).total_seconds())
            if stage_started is not None
            else None
        )
        return {
            "createdAt": _to_iso(self.created_at),
            "queuedAt": _to_iso(self.created_at),
            "startedAt": _to_iso(self.started_at),
            "finishedAt": _to_iso(self.finished_at),
            "completedAt": _to_iso(completed_at),
            "failedAt": _to_iso(failed_at),
            "cancelledAt": _to_iso(cancelled_at),
            "elapsedSeconds": round(elapsed_seconds, 3),
            "queueWaitSeconds": round(queue_wait_seconds, 3) if queue_wait_seconds is not None else None,
            "analysisRuntimeSeconds": (
                round(analysis_runtime_seconds, 3) if analysis_runtime_seconds is not None else None
            ),
            "stageStartedAt": _to_iso(stage_started),
            "stageElapsedSeconds": round(stage_elapsed_seconds, 3) if stage_elapsed_seconds is not None else None,
        }

    def status_payload(self) -> Dict[str, Any]:
        updated_at = _to_iso(self.updated_at)
        timing = self.timing_payload()
        runtime_summary = _job_runtime_summary_payload(self, timing=timing)
        diagnostics = dict(self.metrics.get("componentDiagnostics") or {})
        return {
            "jobId": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "progressPercent": max(0, min(100, round(self.progress * 100))),
            "stage": _stage_for_status(self.status, self.current_step),
            "message": self.current_step,
            "currentStep": self.current_step,
            "error": self.error,
            "metrics": self.metrics,
            "componentDiagnostics": self.metrics.get("componentDiagnostics"),
            "scheduler": {
                "enqueueSequence": diagnostics.get("enqueueSequence"),
                "batchEnqueueSequence": diagnostics.get("batchEnqueueSequence"),
                "childSequence": diagnostics.get("childSequence"),
                "queuePosition": diagnostics.get("queuePosition"),
                "workerSlot": diagnostics.get("workerSlot"),
                "capacityReason": diagnostics.get("capacityReason") or diagnostics.get("queueWaitReason"),
                "schedulerPolicy": diagnostics.get("schedulerPolicy"),
            },
            "persistenceWarning": self.persistence_warning,
            "manifestPersistenceWarning": self.persistence_warning,
            "sourceMetadata": self.source_metadata,
            "retryCount": self.retry_count,
            "maxAttempts": self.max_attempts,
            "nextRetryAt": _to_iso(self.next_retry_at),
            "recoveryAction": self.recovery_action,
            "analysisSetup": build_analysis_setup_payload(self),
            **timing,
            **runtime_summary,
            "updatedAt": updated_at,
            "heartbeatAt": updated_at if self.status in ACTIVE_STATES else None,
        }


def build_analysis_setup_payload(
    record: JobRecord,
    result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return canonical requested/actual setup without relying on browser state."""
    result_payload = result if result is not None else record.result or {}
    technical = dict(result_payload.get("technicalDetails") or {})
    processing = dict(technical.get("processingMetadata") or {})
    profile = dict(record.settings.use_case_profile or {})
    requested_review = "comprehensive" if record.settings.review_mode == "comprehensive" else "fast_local"
    actual_review = str(result_payload.get("reviewMode") or requested_review)
    sampling_fps = processing.get("samplingFps")
    try:
        interval_seconds = round(1.0 / float(sampling_fps), 3) if float(sampling_fps) > 0 else None
    except (TypeError, ValueError):
        interval_seconds = None
    return {
        "readOnly": record.status in TERMINAL_STATES or result_payload.get("status") == "completed",
        "requestedCoverage": {
            "id": requested_review,
            "label": "Comprehensive Review" if requested_review == "comprehensive" else "Fast Local Analysis",
        },
        "actualCoverage": {
            "id": actual_review,
            "label": "Comprehensive Review" if actual_review == "comprehensive" else "Fast Local Analysis",
        },
        "profile": {
            "id": str(profile.get("profileId") or "general_safety"),
            "label": str(profile.get("label") or "General Safety"),
        },
        "query": record.query,
        "frameSampling": {
            "strategy": processing.get("samplingStrategy") or "requested_frame_sampling",
            "requestedFps": float(record.settings.fps),
            "samplingFps": sampling_fps,
            "samplingIntervalSeconds": interval_seconds,
            "sampledFrameCount": processing.get("sampledFrameCount"),
            "sourceVideoFrameCount": processing.get("sourceVideoFrameCount"),
            "processingWindowCount": processing.get("processingWindowCount"),
            "maximumSampledFrames": processing.get("maxSampledFrames"),
            "requestedEvidenceFrames": int(record.settings.top_k),
        },
    }


@dataclass
class JobExecutionContext:
    """Job-scoped mutable runtime state with an immutable identity/config snapshot."""

    job_id: str
    batch_id: str | None
    immutable_config: Any
    source_identity: Any
    workspace: Path
    temporary_artifact_namespace: Path
    result_accumulator: Dict[str, Any] = field(default_factory=dict)
    evidence_accumulator: list[Dict[str, Any]] = field(default_factory=list)
    diagnostic_accumulator: Dict[str, Any] = field(default_factory=dict)
    execution_generation: int = 0

    @classmethod
    def from_record(cls, record: "JobRecord") -> "JobExecutionContext":
        config = MappingProxyType(copy.deepcopy(_settings_to_manifest(record.settings)))
        source = MappingProxyType(copy.deepcopy(dict(record.source_metadata or {})))
        workspace = record.job_dir / "workspace"
        return cls(
            job_id=record.job_id,
            batch_id=str(record.source_metadata.get("batchId") or "") or None,
            immutable_config=config,
            source_identity=source,
            workspace=workspace,
            temporary_artifact_namespace=workspace / "tmp",
            execution_generation=int(record.metrics.get("restartGeneration") or 0),
        )


def result_ownership_context(record: JobRecord) -> ResultOwnershipContext:
    source = dict(record.source_metadata or {})
    return ResultOwnershipContext(
        job_id=record.job_id,
        batch_id=str(source.get("batchId") or "") or None,
        source_checksum=str(source.get("checksumSha256") or "") or None,
        original_filename=str(source.get("originalFilename") or record.original_filename),
        source_relative_path=str(source.get("sourceRelativePath") or record.original_filename),
        source_group_path=str(source.get("sourceGroupPath") or ""),
        execution_identity=copy.deepcopy(dict(record.metrics.get("executionIdentity") or {})),
    )


def owned_result_snapshot(record: JobRecord) -> Dict[str, Any]:
    if record.result is None:
        raise ResultOwnershipError([f"job {record.job_id!r} has no completed result"])
    validate_result_ownership(record.result, result_ownership_context(record))
    return copy.deepcopy(record.result)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: Optional[datetime]) -> Optional[str]:
    return value.astimezone(timezone.utc).isoformat() if value else None


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _stage_for_status(status: JobState, current_step: str) -> str:
    if status in TERMINAL_STATES:
        return status
    step = (current_step or "").lower()
    if "prepar" in step:
        return "preparing"
    if "normaliz" in step or "report" in step:
        return "normalizing"
    if "analysis" in step or "safetrace" in step:
        return "analyzing"
    return status


def _format_duration_label(seconds: Any) -> Optional[str]:
    if not isinstance(seconds, (int, float)):
        return None
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _requested_mode_label(profile: Any) -> str:
    return MODE_LABELS.get(normalize_vlm_profile(profile), "Fast Local Analysis")


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", {}):
            return value
    return None


def _result_frame_sources(result: Optional[Dict[str, Any]]) -> list[str]:
    if not result:
        return []
    sources: list[str] = []
    for frame in list(result.get("frames") or []):
        if not isinstance(frame, dict):
            continue
        value = frame.get("explanationSource")
        if value:
            sources.append(str(value))
        metadata = ((frame.get("technicalEvidence") or {}).get("searchMetadata") or {})
        if isinstance(metadata, dict):
            worker = metadata.get("lightweightVlmExplanation")
            if isinstance(worker, dict):
                value = worker.get("finalExplanationSource") or worker.get("lightweightVlmExplanationSource")
                if value:
                    sources.append(str(value))
    return sources


def _source_ranked_summary(result: Optional[Dict[str, Any]], diagnostics: Dict[str, Any]) -> str:
    source_text = " ".join(
        [
            *_result_frame_sources(result),
            str(diagnostics.get("finalExplanationSource") or ""),
            str(diagnostics.get("lightweightVlmExplanationSource") or ""),
            str(diagnostics.get("effectiveExplanationMode") or ""),
        ]
    ).lower()
    if "rule_template_plus_lightweight_plus_enhanced" in source_text or "vlm_enhanced" in source_text:
        return "rule_template_plus_lightweight_plus_enhanced"
    if "rule_template_plus_lightweight_vlm" in source_text or "vlm_lightweight" in source_text:
        return "rule_template_plus_lightweight_vlm"
    return "rule_based"


def _fallback_reason_label(reason: Any) -> Optional[str]:
    if not reason:
        return None
    text = str(reason)
    labels = {
        "visual_review_frame_limit_reached": "frame limit reached",
        "local_visual_review_not_selected": "not selected for visual review",
        "local_visual_review_timeout": "visual review timed out",
        "local_visual_review_runtime_guard_elapsed": "runtime guard elapsed",
        "local_visual_review_quality_guard": "quality guard reached",
        "worker_timeout": "visual review timed out",
        "generation_timeout": "visual review timed out",
        "disabled_or_unloaded": "model disabled or not loaded",
        "model_missing": "model missing",
        "worker_disabled": "worker disabled",
        "no_eligible_violation": "no eligible finding",
    }
    if text.startswith("quality:"):
        return f"quality fallback: {text.split(':', 1)[1].replace('_', ' ')}"
    return labels.get(text, text.replace("_", " "))


def _job_runtime_summary_payload(record: JobRecord, *, timing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    timing = timing or record.timing_payload()
    diagnostics = dict(record.metrics.get("componentDiagnostics") or {})
    result = record.result if isinstance(record.result, dict) else None
    requested_profile = normalize_vlm_profile(record.settings.vlm_profile)
    requested_label = _requested_mode_label(requested_profile)
    final_source = _source_ranked_summary(result, diagnostics)
    lightweight_attempted = bool(
        diagnostics.get("lightweightVlmWorkerAttempted")
        or diagnostics.get("totalVlmAttempts")
        or diagnostics.get("vlmAttempted")
    )
    enhanced_attempted = bool(
        lightweight_attempted
        and requested_profile in {VLM_PROFILE_ENHANCED, VLM_PROFILE_ENHANCED_3B}
    )
    lightweight_accepted = final_source == "rule_template_plus_lightweight_vlm"
    enhanced_accepted = final_source == "rule_template_plus_lightweight_plus_enhanced"
    vlm_attempted = bool(lightweight_attempted or enhanced_attempted)
    vlm_accepted = bool(lightweight_accepted or enhanced_accepted)
    fallback_reasons = diagnostics.get("vlmSkippedReasons") if isinstance(diagnostics.get("vlmSkippedReasons"), dict) else {}
    fallback_reason = _first_present(
        diagnostics.get("lightweightVlmFallbackReason"),
        diagnostics.get("lightweightVlmDisabledReason"),
        diagnostics.get("lightweightVlmRuntimeGuardReason"),
        next(iter(fallback_reasons.keys()), None) if fallback_reasons else None,
    )
    fallback_label = _fallback_reason_label(fallback_reason)
    if enhanced_accepted:
        actual_review = "Advanced GPU VLM Assist"
        outcome = "enhanced_accepted"
        outcome_label = "Advanced GPU VLM accepted"
    elif lightweight_accepted:
        actual_review = "Local VLM Assist"
        outcome = "lightweight_accepted"
        outcome_label = "Local VLM accepted"
    elif vlm_attempted and requested_profile in {VLM_PROFILE_ENHANCED, VLM_PROFILE_ENHANCED_3B}:
        actual_review = "Advanced GPU VLM attempted -> Fast fallback"
        outcome = "advanced_attempted_fast_fallback"
        outcome_label = "Advanced GPU VLM attempted; Fast Local Analysis fallback"
    elif vlm_attempted and requested_profile in VLM_LIGHTWEIGHT_PROFILES:
        actual_review = "Local VLM attempted -> Fast fallback"
        outcome = "local_attempted_fast_fallback"
        outcome_label = "Local VLM attempted; Fast Local Analysis fallback"
    else:
        actual_review = "Fast Local Analysis"
        outcome = "fast_local_analysis"
        outcome_label = "Fast Local Analysis"

    selected_device = str(
        _first_present(
            diagnostics.get("actualEnhancedVlmDevice") if enhanced_attempted else None,
            diagnostics.get("actualLightweightVlmDevice") if lightweight_attempted else None,
            diagnostics.get("actualDetectorDevice"),
            diagnostics.get("device"),
            record.settings.device,
        )
        or "auto"
    )
    requested_device = str(record.settings.device or diagnostics.get("requestedDevice") or "auto")
    if requested_device == "auto" and selected_device == "cuda":
        device_label = "Auto selected CUDA"
    elif selected_device == "cuda":
        device_label = "CUDA"
    elif selected_device == "cpu":
        device_label = "CPU"
    else:
        device_label = selected_device

    model_profile = str(diagnostics.get("lightweightVlmModelProfile") or diagnostics.get("lightweightVlmSelectedProfile") or "")
    if not record.settings.enable_vlm or requested_profile == VLM_PROFILE_RULE_BASED:
        vlm_status = "Not requested"
    elif enhanced_accepted:
        vlm_status = "Enhanced attempted, accepted"
    elif enhanced_attempted:
        vlm_status = f"Enhanced attempted, fallback{f': {fallback_label}' if fallback_label else ''}"
    elif lightweight_accepted:
        vlm_status = (
            "512M accepted"
            if "512" in model_profile
            else "256M accepted"
            if "256" in model_profile
            else "Local VLM accepted"
        )
    elif lightweight_attempted:
        vlm_status = f"Local VLM attempted, fallback{f': {fallback_label}' if fallback_label else ''}"
    else:
        vlm_status = f"Not attempted{f': {fallback_label}' if fallback_label else ''}"

    mobile_sam_status = "not used"
    if diagnostics.get("mobileSamWorkerSucceeded") or diagnostics.get("mobileSamLoaded"):
        mobile_sam_status = "attempted / succeeded"
    elif diagnostics.get("mobileSamWorkerAttempted") or diagnostics.get("mobileSamAttempted"):
        mobile_sam_status = f"attempted / fallback{f': {diagnostics.get('mobileSamFallbackReason')}' if diagnostics.get('mobileSamFallbackReason') else ''}"
    elif diagnostics.get("mobileSamRequested"):
        mobile_sam_status = "enabled"

    return {
        "requestedModeLabel": requested_label,
        "requestedVisualExplanationMode": requested_profile,
        "actualExplanationMode": diagnostics.get("effectiveExplanationMode") or final_source,
        "finalExplanationSource": final_source,
        "explanationOutcome": outcome,
        "explanationOutcomeLabel": outcome_label,
        "vlmAttempted": vlm_attempted,
        "lightweightVlmAttempted": lightweight_attempted,
        "enhancedVlmAttempted": enhanced_attempted,
        "vlmAccepted": vlm_accepted,
        "lightweightVlmAccepted": lightweight_accepted,
        "enhancedVlmAccepted": enhanced_accepted,
        "vlmFallbackReason": fallback_reason,
        "vlmFallbackReasonLabel": fallback_label,
        "actualDeviceLabel": device_label,
        "jobRuntimeLabel": _format_duration_label(timing.get("elapsedSeconds")),
        "engineRuntimeSummary": {
            "requestedMode": requested_label,
            "actualReview": actual_review,
            "device": device_label,
            "vlm": vlm_status,
            "mobileSam": mobile_sam_status,
            "runtime": _format_duration_label(timing.get("elapsedSeconds")),
            "requestedDevice": requested_device,
            "actualDetectorDevice": diagnostics.get("actualDetectorDevice") or diagnostics.get("device"),
            "actualMobileSamDevice": diagnostics.get("actualMobileSamDevice"),
            "actualLightweightVlmDevice": diagnostics.get("actualLightweightVlmDevice"),
            "actualEnhancedVlmDevice": diagnostics.get("actualEnhancedVlmDevice"),
            "cudaAvailableAtJobStart": diagnostics.get("cudaAvailableAtJobStart"),
            "gpuName": diagnostics.get("gpuName"),
        },
    }


def _safe_relative_path(path: Path, root: Path) -> Optional[str]:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return None


def _resolve_manifest_path(job_dir: Path, raw_path: Any, default: Path) -> Path:
    if not raw_path:
        return default
    candidate = (job_dir / str(raw_path)).resolve()
    root = job_dir.resolve()
    if candidate == root or root in candidate.parents:
        return candidate
    return default


def _manifest_lock(path: Path) -> threading.RLock:
    key = path.resolve()
    with _MANIFEST_LOCKS_GUARD:
        lock = _MANIFEST_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _MANIFEST_LOCKS[key] = lock
        return lock


def _cleanup_temporary(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _atomic_write_json(path: Path, payload: Dict[str, Any], *, retries: int = 5, backoff_seconds: float = 0.05) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    lock = _manifest_lock(path)
    with lock:
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            last_error: PermissionError | None = None
            for attempt in range(max(1, retries)):
                try:
                    os.replace(temporary, path)
                    return
                except PermissionError as exc:
                    last_error = exc
                    if attempt >= retries - 1:
                        break
                    time.sleep(backoff_seconds * (attempt + 1))
            raise JobManifestPersistenceError(f"Could not replace {path} after {retries} attempts.") from last_error
        finally:
            _cleanup_temporary(temporary)


def safe_filename(filename: str) -> str:
    name = Path(filename or "upload.bin").name
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem).strip("._") or "upload"
    suffix = Path(name).suffix.lower()
    return f"{stem}{suffix}"


def media_type_for(filename: str) -> str:
    return MEDIA_EXTENSIONS.get(Path(filename).suffix.lower(), "unknown")


def allowed_upload_extensions() -> list[str]:
    return sorted(MEDIA_EXTENSIONS)


def max_upload_bytes() -> int:
    return max(int(float(SETTINGS.max_upload_mb) * 1024 * 1024), 1)


def validate_upload_filename(filename: str) -> str:
    clean_name = safe_filename(filename)
    if media_type_for(clean_name) == "unknown":
        allowed = ", ".join(allowed_upload_extensions())
        raise UploadValidationError(
            f"Unsupported upload type. Allowed extensions: {allowed}",
            status_code=415,
        )
    return clean_name


def validate_upload_size(size_bytes: int, *, limit_bytes: Optional[int] = None) -> None:
    limit = limit_bytes if limit_bytes is not None else max_upload_bytes()
    if size_bytes > limit:
        max_mb = limit / (1024 * 1024)
        raise UploadValidationError(
            f"Uploaded file is too large. Maximum size is {max_mb:.1f} MB.",
            status_code=413,
        )


def new_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"job_{stamp}_{uuid.uuid4().hex[:8]}"


def _is_safe_job_id(job_id: str) -> bool:
    return bool(re.fullmatch(r"job_[A-Za-z0-9_.-]+", job_id or ""))


def _settings_to_manifest(settings: AnalysisSettings) -> Dict[str, Any]:
    return {
        "fps": settings.fps,
        "topK": settings.top_k,
        "enableVlm": settings.enable_vlm,
        "device": settings.device,
        "vlmProfile": normalize_vlm_profile(settings.vlm_profile),
        "vlmEnabled": bool(settings.vlm_enabled),
        "safeMode": bool(settings.safe_mode),
        "useCaseProfile": dict(settings.use_case_profile or {}),
        "reviewMode": settings.review_mode,
    }


def _settings_from_manifest(payload: Dict[str, Any]) -> AnalysisSettings:
    return AnalysisSettings(
        fps=float(payload.get("fps") or 1.0),
        top_k=int(payload.get("topK") or payload.get("top_k") or 5),
        enable_vlm=bool(payload.get("enableVlm") or payload.get("enable_vlm") or False),
        device=str(payload.get("device") or "auto"),
        vlm_profile=normalize_vlm_profile(payload.get("vlmProfile") or payload.get("vlm_profile")),
        vlm_enabled=bool(payload.get("vlmEnabled") or payload.get("vlm_enabled") or False),
        safe_mode=bool(payload.get("safeMode") or payload.get("safe_mode") or False),
        use_case_profile=dict(payload.get("useCaseProfile") or payload.get("use_case_profile") or {}),
        review_mode=str(payload.get("reviewMode") or payload.get("review_mode") or "fast_local"),
    )


def normalize_vlm_profile(value: Any) -> str:
    raw = str(value or VLM_PROFILE_RULE_BASED).strip().lower()
    return raw if raw in VLM_PROFILE_IDS else VLM_PROFILE_RULE_BASED


def resolve_configured_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    root_candidate = (SETTINGS.project_root / path).resolve()
    if root_candidate.exists():
        return root_candidate
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return root_candidate


def resolve_vlm_profile_model_dir(profile: Any) -> Optional[Path]:
    selected = normalize_vlm_profile(profile)
    if selected == VLM_PROFILE_LIGHTWEIGHT:
        return resolve_configured_path(Path(SETTINGS.vlm_lightweight_model_path))
    if selected == VLM_PROFILE_LIGHTWEIGHT_512M:
        return resolve_configured_path(Path(SETTINGS.vlm_lightweight_512m_model_path))
    if selected == VLM_PROFILE_ENHANCED:
        return resolve_configured_path(Path(SETTINGS.vlm_enhanced_model_path))
    if selected == VLM_PROFILE_ENHANCED_3B:
        return resolve_configured_path(Path(SETTINGS.vlm_enhanced_3b_model_path))
    return None


def vlm_hard_disabled() -> bool:
    return str(getattr(SETTINGS, "vlm_enabled", "auto") or "").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
        "disabled",
        "none",
    }


def analysis_safe_mode() -> bool:
    return bool(getattr(SETTINGS, "analysis_safe_mode", False))


def lightweight_vlm_worker_requested(settings: AnalysisSettings) -> bool:
    safe_mode = bool(settings.safe_mode or analysis_safe_mode())
    profile = normalize_vlm_profile(settings.vlm_profile)
    return bool(
        safe_mode
        and getattr(SETTINGS, "lightweight_vlm_worker_enabled", False)
        and not vlm_hard_disabled()
        and settings.enable_vlm
        and settings.vlm_enabled
        and profile in VLM_SAFE_MODE_WORKER_PROFILES
    )


def _safe_concurrency(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def analysis_concurrency_limit() -> int:
    # ``analysis_concurrency`` remains the execution alias used by existing
    # callers and tests.  Settings derives it from SAFETRACE_JOB_CONCURRENCY
    # when the newer orchestration control is supplied.
    return _safe_concurrency(
        getattr(SETTINGS, "analysis_concurrency", getattr(SETTINGS, "job_concurrency", getattr(SETTINGS, "worker_concurrency", 1))),
        default=1,
    )


def analysis_execution_ceiling() -> int:
    """Hard semaphore ceiling; adaptive admission remains scheduler-owned."""
    if not _ADAPTIVE_WORKER_MANAGER.enabled:
        return analysis_concurrency_limit()
    return max(analysis_concurrency_limit(), _ADAPTIVE_WORKER_MANAGER.execution_ceiling())


def per_batch_concurrency_limit() -> int:
    """Return the configured batch-child cap with the legacy alias intact.

    ``batch_max_active_jobs`` existed before the runtime-profile work.  A
    source deployment that has not explicitly set the newer environment name
    must continue to honour that setting (and the test/development overrides
    used by existing callers).  When the newer variable is set, it is the
    explicit source of truth.
    """
    if "SAFETRACE_PER_BATCH_CONCURRENCY" in os.environ:
        value = getattr(SETTINGS, "per_batch_concurrency", 1)
    else:
        # Both names are mutable compatibility aliases in the source runtime.
        # Before the new environment name is set, accept an explicit override
        # to either one.  Profile defaults initialise them to the same value.
        value = max(
            _safe_concurrency(getattr(SETTINGS, "batch_max_active_jobs", 1)),
            _safe_concurrency(getattr(SETTINGS, "per_batch_concurrency", 1)),
        )
    return _safe_concurrency(value, default=1)


def vlm_concurrency_limit() -> int:
    return _safe_concurrency(getattr(SETTINGS, "vlm_concurrency", 1), default=1)


def gpu_inference_concurrency_limit() -> int:
    return _safe_concurrency(
        getattr(SETTINGS, "gpu_inference_concurrency", getattr(SETTINGS, "gpu_detector_concurrency", 1)),
        default=1,
    )


def _analysis_semaphore() -> threading.BoundedSemaphore:
    global _ANALYSIS_SEMAPHORE, _ANALYSIS_SEMAPHORE_LIMIT
    limit = analysis_execution_ceiling()
    with _SEMAPHORE_LOCK:
        if _ANALYSIS_SEMAPHORE is None or _ANALYSIS_SEMAPHORE_LIMIT != limit:
            _ANALYSIS_SEMAPHORE = threading.BoundedSemaphore(limit)
            _ANALYSIS_SEMAPHORE_LIMIT = limit
        return _ANALYSIS_SEMAPHORE


@contextmanager
def _analysis_capacity_slot(store: "JobStore", job_id: str):
    semaphore = _analysis_semaphore()
    acquired = semaphore.acquire(blocking=False)
    if not acquired:
        store.update_status(
            job_id,
            status="waiting_for_job_slot",
            progress=0.0,
            current_step="Waiting for an analysis worker slot",
            diagnostic_updates={"queueWaitReason": "job_slot", "capacityReason": "job_slot"},
        )
        semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def _vlm_semaphore() -> threading.BoundedSemaphore:
    global _VLM_SEMAPHORE, _VLM_SEMAPHORE_LIMIT
    limit = vlm_concurrency_limit()
    with _SEMAPHORE_LOCK:
        if _VLM_SEMAPHORE is None or _VLM_SEMAPHORE_LIMIT != limit:
            _VLM_SEMAPHORE = threading.BoundedSemaphore(limit)
            _VLM_SEMAPHORE_LIMIT = limit
        return _VLM_SEMAPHORE


def _gpu_semaphore() -> threading.BoundedSemaphore:
    global _GPU_SEMAPHORE, _GPU_SEMAPHORE_LIMIT
    limit = gpu_inference_concurrency_limit()
    with _SEMAPHORE_LOCK:
        if _GPU_SEMAPHORE is None or _GPU_SEMAPHORE_LIMIT != limit:
            _GPU_SEMAPHORE = threading.BoundedSemaphore(limit)
            _GPU_SEMAPHORE_LIMIT = limit
        return _GPU_SEMAPHORE


@contextmanager
def _resource_capacity_slot(
    store: "JobStore",
    job_id: str,
    *,
    semaphore: threading.BoundedSemaphore,
    wait_status: JobState,
    wait_step: str,
    resource: str,
):
    acquired = semaphore.acquire(blocking=False)
    if not acquired:
        store.update_status(
            job_id,
            status=wait_status,
            progress=0.30,
            current_step=wait_step,
            diagnostic_updates={"queueWaitReason": resource, "capacityReason": resource},
        )
        semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def _runtime_settings_key(
    *,
    device: str,
    enable_vlm: bool,
    vlm_profile: str,
    vlm_model_dir: Optional[Path],
    safe_mode: bool,
    review_mode: str,
) -> tuple[Any, ...]:
    return (
        str(device or "auto").strip().lower(),
        bool(enable_vlm),
        normalize_vlm_profile(vlm_profile),
        str(vlm_model_dir.resolve()) if vlm_model_dir is not None else None,
        bool(safe_mode),
        "comprehensive" if review_mode == "comprehensive" else "fast_local",
    )


@contextmanager
def _compatible_runtime_settings(
    *,
    device: str,
    enable_vlm: bool,
    vlm_profile: str,
    vlm_model_dir: Optional[Path],
    safe_mode: bool,
    review_mode: str,
):
    """Apply legacy global settings once for a compatible group of active jobs.

    The pipeline still has model wrappers which read ``SETTINGS`` at runtime.
    This context preserves compatibility without serializing jobs that requested
    the exact same runtime settings, while making different contexts wait.
    """
    global _PIPELINE_SETTINGS_SNAPSHOT, _PIPELINE_SETTINGS_KEY, _PIPELINE_SETTINGS_ACTIVE

    key = _runtime_settings_key(
        device=device,
        enable_vlm=enable_vlm,
        vlm_profile=vlm_profile,
        vlm_model_dir=vlm_model_dir,
        safe_mode=safe_mode,
        review_mode=review_mode,
    )
    mutable_fields = (
        "device", "lightweight_vlm_device", "enhanced_vlm_device", "mobile_sam_device",
        "enable_vlm", "vlm_profile", "vlm_model_dir", "mobile_sam_enabled",
        "analysis_safe_mode", "safe_mode_allow_mobilesam", "max_frames",
        "vlm_max_evidence_frames", "mobile_sam_frame_limit",
    )
    with _PIPELINE_SETTINGS_CONDITION:
        while _PIPELINE_SETTINGS_ACTIVE and _PIPELINE_SETTINGS_KEY != key:
            _PIPELINE_SETTINGS_CONDITION.wait()
        if not _PIPELINE_SETTINGS_ACTIVE:
            _PIPELINE_SETTINGS_SNAPSHOT = {name: getattr(SETTINGS, name) for name in mutable_fields}
            normalized_review_mode = "comprehensive" if review_mode == "comprehensive" else "fast_local"
            normalized_profile = normalize_vlm_profile(vlm_profile)
            SETTINGS.analysis_safe_mode = bool(safe_mode or analysis_safe_mode())
            SETTINGS.device = str(device or "auto")
            if SETTINGS.device in {"cpu", "cuda"}:
                SETTINGS.lightweight_vlm_device = SETTINGS.device
                SETTINGS.mobile_sam_device = SETTINGS.device
                SETTINGS.enhanced_vlm_device = SETTINGS.device if SETTINGS.device == "cpu" else "cuda"
            SETTINGS.enable_vlm = bool(enable_vlm)
            SETTINGS.vlm_profile = normalized_profile
            if vlm_model_dir is not None:
                SETTINGS.vlm_model_dir = vlm_model_dir
            if SETTINGS.analysis_safe_mode and not bool(getattr(SETTINGS, "safe_mode_allow_mobilesam", False)):
                SETTINGS.mobile_sam_enabled = "disabled"
            if normalized_review_mode == "comprehensive":
                SETTINGS.max_frames = max(int(SETTINGS.max_frames), int(getattr(SETTINGS, "comprehensive_review_max_frames", 1200)))
                SETTINGS.vlm_max_evidence_frames = min(
                    int(SETTINGS.vlm_max_evidence_frames), int(getattr(SETTINGS, "comprehensive_max_vlm_frames", 5))
                )
                SETTINGS.mobile_sam_frame_limit = min(
                    int(getattr(SETTINGS, "top_k", 5)), int(getattr(SETTINGS, "comprehensive_max_segmentation_frames", 8))
                )
            _PIPELINE_SETTINGS_KEY = key
        _PIPELINE_SETTINGS_ACTIVE += 1
    try:
        yield
    finally:
        with _PIPELINE_SETTINGS_CONDITION:
            _PIPELINE_SETTINGS_ACTIVE -= 1
            if _PIPELINE_SETTINGS_ACTIVE == 0:
                for name, value in (_PIPELINE_SETTINGS_SNAPSHOT or {}).items():
                    setattr(SETTINGS, name, value)
                _PIPELINE_SETTINGS_SNAPSHOT = None
                _PIPELINE_SETTINGS_KEY = None
                _PIPELINE_SETTINGS_CONDITION.notify_all()


def should_enable_vlm(settings: AnalysisSettings) -> bool:
    profile = normalize_vlm_profile(settings.vlm_profile)
    if lightweight_vlm_worker_requested(settings):
        return True
    return bool(
        not analysis_safe_mode()
        and not settings.safe_mode
        and not vlm_hard_disabled()
        and settings.enable_vlm
        and settings.vlm_enabled
        and profile != VLM_PROFILE_RULE_BASED
    )


def _detector_checkpoint_candidate() -> Optional[str]:
    if SETTINGS.yolo_checkpoint.exists():
        return str(SETTINGS.yolo_checkpoint)
    if SETTINGS.yolo_fallback_checkpoint.exists():
        return str(SETTINGS.yolo_fallback_checkpoint)
    return None


def _mobile_sam_requested(settings: AnalysisSettings) -> bool:
    safe_mode = bool(settings.safe_mode or analysis_safe_mode())
    safe_mode_allowed = bool(getattr(SETTINGS, "safe_mode_allow_mobilesam", False))
    if safe_mode and not safe_mode_allowed:
        return False
    return str(getattr(SETTINGS, "mobile_sam_enabled", "disabled") or "").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "disabled",
        "none",
    }


def _base_component_diagnostics(settings: AnalysisSettings, *, stage: str = "queued") -> Dict[str, Any]:
    effective_vlm = should_enable_vlm(settings)
    requested_profile = normalize_vlm_profile(settings.vlm_profile)
    mobile_sam_requested = _mobile_sam_requested(settings)
    requested_device = str(settings.device or "auto").strip().lower()
    gateway_settings = SimpleNamespace(
        device=requested_device if requested_device in {"auto", "cpu", "cuda"} else getattr(SETTINGS, "device", "auto"),
        lightweight_vlm_device=(
            requested_device
            if requested_device in {"cpu", "cuda"}
            else getattr(SETTINGS, "lightweight_vlm_device", "auto")
        ),
        enhanced_vlm_device=(
            "cpu"
            if requested_device == "cpu"
            else "cuda"
            if requested_device == "cuda"
            else getattr(SETTINGS, "enhanced_vlm_device", "cuda")
        ),
        mobile_sam_device=(
            requested_device
            if requested_device in {"cpu", "cuda"}
            else getattr(SETTINGS, "mobile_sam_device", "auto")
        ),
    )
    gateway = device_gateway_payload(gateway_settings)
    gateway_components = dict(gateway.get("components") or {})
    detector_device = str(gateway_components.get("detector", {}).get("selected") or settings.device or "auto")
    mobile_sam_device = str(gateway_components.get("mobileSam", {}).get("selected") or "cpu")
    lightweight_device = str(gateway_components.get("lightweightVlm", {}).get("selected") or "cpu")
    enhanced_device = str(gateway_components.get("enhancedVlm", {}).get("selected") or "unavailable")
    mobile_sam_worker_enabled = bool(
        mobile_sam_requested and getattr(SETTINGS, "mobile_sam_worker_enabled", False)
    )
    safe_mode = bool(settings.safe_mode or analysis_safe_mode())
    lightweight_vlm_worker_enabled = lightweight_vlm_worker_requested(settings)
    return {
        "safeMode": safe_mode,
        "requestedDevice": requested_device,
        "device": detector_device,
        "actualDetectorDevice": detector_device,
        "actualMobileSamDevice": mobile_sam_device,
        "actualLightweightVlmDevice": lightweight_device,
        "actualEnhancedVlmDevice": enhanced_device,
        "cudaAvailableAtJobStart": bool(gateway.get("torch", {}).get("cuda_available")),
        "gpuName": gateway.get("torch", {}).get("gpu_name") or gateway.get("hardware", {}).get("gpu_name"),
        "deviceGatewayDecision": gateway,
        "requestedVisualExplanationMode": requested_profile,
        "effectiveExplanationMode": (
            requested_profile
            if lightweight_vlm_worker_enabled
            else
            "rule_based_with_mobilesam"
            if safe_mode and mobile_sam_requested
            else requested_profile
            if effective_vlm
            else VLM_PROFILE_RULE_BASED
        ),
        "vlmRequested": bool(settings.vlm_enabled and requested_profile != VLM_PROFILE_RULE_BASED),
        "vlmEffectiveEnabled": effective_vlm,
        "vlmAttempted": False,
        "vlmLoaded": False,
        "vlmSuppressedReason": (
            None
            if lightweight_vlm_worker_enabled
            else "safe_mode"
            if settings.safe_mode or analysis_safe_mode()
            else "hard_disabled"
            if vlm_hard_disabled()
            else "rule_based"
            if requested_profile == VLM_PROFILE_RULE_BASED
            else "not_requested"
            if not settings.enable_vlm or not settings.vlm_enabled
            else None
        ),
        "lightweightVlmWorkerEnabled": lightweight_vlm_worker_enabled,
        "lightweightVlmWorkerTimeoutSeconds": float(
            getattr(SETTINGS, "lightweight_vlm_worker_timeout_seconds", 60.0) or 60.0
        ),
        "lightweightVlmWorkerAttempted": False,
        "lightweightVlmWorkerSucceeded": False,
        "lightweightVlmWorkerTimedOut": False,
        "lightweightVlmWorkerExitCode": None,
        "lightweightVlmFallbackReason": None,
        "lightweightVlmExplanationSource": "rule_based" if lightweight_vlm_worker_enabled else "disabled",
        "lightweightVlmQualityIssue": None,
        "lightweightVlmRawTextPreview": None,
        "lightweightVlmCleanTextPreview": None,
        "lightweightVlmGenerationTimeoutSeconds": None,
        "lightweightVlmMaxTokens": None,
        "lightweightVlmEvidenceBudget": int(
            getattr(SETTINGS, "vlm_max_evidence_frames", getattr(SETTINGS, "vlm_max_frames", 5)) or 0
        ),
        "lightweightVlmFrameLimit": int(
            getattr(SETTINGS, "vlm_max_evidence_frames", getattr(SETTINGS, "vlm_max_frames", 5)) or 0
        ),
        "vlmFrameLimit": int(
            getattr(SETTINGS, "vlm_max_evidence_frames", getattr(SETTINGS, "vlm_max_frames", 5)) or 0
        ),
        "lightweightVlmJobTimeoutSeconds": float(getattr(SETTINGS, "vlm_job_timeout_seconds", 0.0) or 0.0),
        "lightweightVlmRuntimeGuardReason": None,
        "lightweightVlmFramesAttempted": 0,
        "lightweightVlmFramesSucceeded": 0,
        "lightweightVlmFramesSkipped": 0,
        "analysisConcurrency": analysis_concurrency_limit(),
        "vlmConcurrency": vlm_concurrency_limit(),
        "vlmConcurrencyLimited": False,
        "safeModeMobileSamAllowed": bool(safe_mode and getattr(SETTINGS, "safe_mode_allow_mobilesam", False)),
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
        "embeddingRequested": not safe_mode,
        "embeddingLoaded": False,
        "detectorRequested": True,
        "detectorLoaded": False,
        "detectorCheckpointUsed": _detector_checkpoint_candidate(),
        "useCaseProfile": dict(settings.use_case_profile or {}),
        "currentPipelineStage": stage,
        "stageTimings": {},
        "lastHeartbeat": None,
        "errorType": None,
        "errorMessage": None,
    }


def _merge_component_diagnostics(
    metrics: Dict[str, Any],
    settings: AnalysisSettings,
    *,
    stage: Optional[str] = None,
    updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    diagnostics = _base_component_diagnostics(settings, stage=stage or "queued")
    diagnostics.update(dict(metrics.get("componentDiagnostics") or {}))
    if stage:
        diagnostics["currentPipelineStage"] = stage
    if updates:
        existing_timings = dict(diagnostics.get("stageTimings") or {})
        incoming = dict(updates)
        if "stageTimings" in incoming:
            existing_timings.update(dict(incoming.pop("stageTimings") or {}))
            diagnostics["stageTimings"] = existing_timings
        diagnostics.update(incoming)
    metrics["componentDiagnostics"] = diagnostics
    return diagnostics


class JobStore:
    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = Path(root_dir or SETTINGS.data_dir / "api_jobs")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self.recover_jobs()

    def create_job(
        self,
        *,
        filename: str,
        content: bytes,
        query: str,
        settings: AnalysisSettings,
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> JobRecord:
        from io import BytesIO

        return self.create_job_from_stream(
            filename=filename,
            stream=BytesIO(content),
            query=query,
            settings=settings,
            source_metadata=source_metadata,
        )

    def create_job_from_stream(
        self,
        *,
        filename: str,
        stream: BinaryIO,
        query: str,
        settings: AnalysisSettings,
        source_metadata: Optional[Dict[str, Any]] = None,
        expected_size: Optional[int] = None,
    ) -> JobRecord:
        """Create a job while streaming the upload to disk and hashing it once."""
        clean_name = validate_upload_filename(filename)
        if expected_size is not None:
            validate_upload_size(int(expected_size))

        job_id = new_job_id()
        job_dir = self.root_dir / job_id
        upload_dir = job_dir / "uploads"
        output_dir = job_dir / "media"
        upload_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=True)

        upload_path = upload_dir / clean_name
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with upload_path.open("wb") as target:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    validate_upload_size(size_bytes)
                    digest.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            if size_bytes <= 0:
                raise UploadValidationError("Uploaded file is empty", status_code=400)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

        now = _utc_now()
        metadata = dict(source_metadata or {})
        metadata.setdefault("originalFilename", metadata.get("sourceRelativePath") or clean_name)
        metadata.setdefault("sourceRelativePath", clean_name)
        metadata.setdefault("sourceDirectory", "")
        metadata.setdefault("sourceGroupPath", "")
        metadata.setdefault("sourceGroupLabel", "Ungrouped")
        metadata.setdefault("checksumSha256", digest.hexdigest())
        metadata.setdefault("importedAt", _to_iso(now))
        record = JobRecord(
            job_id=job_id,
            status="queued",
            progress=0.0,
            current_step="Queued for analysis",
            error=None,
            query=query,
            settings=settings,
            original_filename=clean_name,
            media_type=media_type_for(clean_name),
            size_bytes=size_bytes,
            job_dir=job_dir,
            upload_path=upload_path,
            output_dir=output_dir,
            source_metadata=metadata,
            max_attempts=max(0, int(getattr(SETTINGS, "job_retry_max_attempts", 2))),
            created_at=now,
            updated_at=now,
        )
        record.metrics["componentDiagnostics"] = _base_component_diagnostics(settings)
        record.metrics["sourceMetadata"] = metadata
        try:
            from .result_cache import cache_identity, cache_key

            record.metrics["executionIdentity"] = cache_identity(record)
            record.metrics["resultCacheKey"] = cache_key(record)
            record.metrics["resultCacheHit"] = False
        except Exception:
            logger.warning("Could not record execution identity for %s", job_id, exc_info=True)
        self._refresh_metrics(record)
        with self._lock:
            self._jobs[job_id] = record
            self.persist_job(record)
            self._persist_recovery_checkpoint(record, stage="upload_complete")
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                record = self.load_job(job_id)
                if record is not None:
                    self._jobs[job_id] = record
            if record is not None and self._is_stale_active(record):
                self._mark_interrupted(record)
            return record

    def require(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return record

    def update_status(
        self,
        job_id: str,
        *,
        status: JobState,
        progress: float,
        current_step: str,
        error: Optional[str] = None,
        technical_error: Optional[str] = None,
        error_type: Optional[str] = None,
        diagnostic_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            record = self._jobs[job_id]
            now = _utc_now()
            previous_stage = str(record.metrics.get("lastSafeStage") or "")
            next_stage = _stage_for_status(status, current_step)
            if status == "running" and record.started_at is None:
                record.started_at = now
            if status in TERMINAL_STATES and record.finished_at is None:
                record.finished_at = now
            record.status = status
            record.progress = progress
            record.current_step = current_step
            record.metrics["lastSafeStage"] = next_stage
            if previous_stage != next_stage or not record.metrics.get("stageStartedAt"):
                record.metrics["stageStartedAt"] = _to_iso(now)
            record.error = error
            record.technical_error = technical_error
            record.error_type = error_type
            record.updated_at = now
            _merge_component_diagnostics(
                record.metrics,
                record.settings,
                stage=next_stage,
                updates={
                    **(diagnostic_updates or {}),
                    **({"errorType": error_type, "errorMessage": error} if error_type or error else {}),
                },
            )
            self._refresh_metrics(record)
            self.persist_job(record)
            self._persist_recovery_checkpoint(record)

    def heartbeat(
        self,
        job_id: str,
        *,
        current_step: str,
        progress: Optional[float] = None,
        diagnostic_updates: Optional[Dict[str, Any]] = None,
    ) -> bool:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.status not in ACTIVE_STATES:
                return False
            if progress is not None:
                record.progress = max(0.0, min(1.0, progress))
            record.current_step = current_step
            record.metrics["lastSafeStage"] = _stage_for_status(record.status, current_step)
            record.updated_at = _utc_now()
            _merge_component_diagnostics(
                record.metrics,
                record.settings,
                stage=_stage_for_status(record.status, current_step),
                updates={
                    "lastHeartbeat": _to_iso(record.updated_at),
                    **(diagnostic_updates or {}),
                },
            )
            self._refresh_metrics(record)
            self.persist_job(record)
            self._persist_recovery_checkpoint(record)
            return True

    def persist_recovery_stage(
        self,
        job_id: str,
        stage: str,
        *,
        completed_frame_index: int | None = None,
        completed_timestamp: float | None = None,
        completed_candidate_ids: list[str] | None = None,
        completed_evidence_ids: list[str] | None = None,
        partial_output_reference: str | None = None,
        partial_output_checksum_sha256: str | None = None,
    ) -> None:
        """Persist a completed coarse stage without changing the public job state."""
        with self._lock:
            record = self._jobs[job_id]
            diagnostics = dict(record.metrics.get("componentDiagnostics") or {})
            diagnostics.update({
                "recoveryCheckpointStage": stage,
                "lastCompletedFrameIndex": completed_frame_index,
                "lastCompletedTimestamp": completed_timestamp,
                "completedCandidateIds": list(completed_candidate_ids or []),
                "completedEvidenceIds": list(completed_evidence_ids or []),
                "partialOutputReference": partial_output_reference,
                "partialOutputChecksumSha256": partial_output_checksum_sha256,
            })
            record.metrics["componentDiagnostics"] = diagnostics
            record.metrics["lastSafeStage"] = stage
            record.updated_at = _utc_now()
            self.persist_job(record)
            self._persist_recovery_checkpoint(record, stage=stage)

    def complete_job(self, job_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            record = self._jobs[job_id]
            now = _utc_now()
            record.status = "completed"
            record.progress = 1.0
            record.current_step = "Analysis completed"
            record.error = None
            record.error_type = None
            record.finished_at = now
            record.updated_at = now
            _merge_component_diagnostics(
                record.metrics,
                record.settings,
                stage="completed",
                updates={"lastHeartbeat": None, "errorType": None, "errorMessage": None},
            )
            self._refresh_metrics(record)

            normalized = copy.deepcopy(dict(result))
            technical_details = dict(normalized.get("technicalDetails") or {})
            job_timing = record.timing_payload(now=now)
            normalized.update(job_timing)
            record.result = normalized
            normalized.update(_job_runtime_summary_payload(record, timing=job_timing))
            technical_details["jobMetrics"] = dict(record.metrics)
            technical_details["jobTiming"] = job_timing
            normalized["technicalDetails"] = technical_details
            engine_metrics = build_engine_metrics(normalized, record.metrics)
            technical_details["engineMetrics"] = engine_metrics
            normalized["engineMetrics"] = engine_metrics
            normalized["technicalDetails"] = technical_details
            record.result = stamp_result_ownership(normalized, result_ownership_context(record))
            record.result_path = record.job_dir / RESULT_FILENAME
            self.persist_job(record)
            self._persist_recovery_checkpoint(record, stage="completed")

    def register_media_file(self, job_id: str, filename: str, path: Path) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.media_files[filename] = path
            record.updated_at = _utc_now()
            self.persist_job(record)

    def register_media_files(self, job_id: str, files: list[tuple[str, Path]]) -> None:
        if not files:
            return
        with self._lock:
            record = self._jobs[job_id]
            for filename, path in files:
                record.media_files[filename] = path
            record.updated_at = _utc_now()
            self.persist_job(record)

    def attach_secondary_review(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            record = self.require(job_id)
            if record.result is None:
                raise ValueError("Analysis result is not ready for secondary review attachment.")
            review = {
                **payload,
                "reviewId": f"review_{uuid.uuid4().hex[:12]}",
                "jobId": job_id,
                "authoritative": False,
                "attachedAt": _to_iso(_utc_now()),
            }
            technical = dict(record.result.get("technicalDetails") or {})
            reviews = list(technical.get("secondaryReviews") or [])
            reviews.append(review)
            technical["secondaryReviews"] = reviews
            technical["secondaryReviewPolicy"] = {
                "baselineRemainsAuthoritative": True,
                "externalMediaTransmissionEnabled": False,
                "disagreementRequiresHumanReview": True,
            }
            record.result["technicalDetails"] = technical
            record.updated_at = _utc_now()
            self.persist_job(record)
            return review

    def acquire_execution_lock(self, job_id: str) -> bool:
        record = self.require(job_id)
        lock_path = record.job_dir / LOCK_FILENAME
        try:
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_to_iso(_utc_now()) or "")
        return True

    def release_execution_lock(self, job_id: str) -> None:
        record = self.get(job_id)
        if record is None:
            return
        lock_path = record.job_dir / LOCK_FILENAME
        try:
            lock_path.unlink()
        except FileNotFoundError:
            return

    def delete(self, job_id: str) -> bool:
        with self._lock:
            record = self.get(job_id)
            if record is None:
                return False
            if record.status == "running" or (record.job_dir / LOCK_FILENAME).exists():
                raise RuntimeError("Job is actively writing and cannot be deleted.")
            self._jobs.pop(job_id, None)
        try:
            from .result_cache import release_job_reference

            release_job_reference(record)
        except Exception:
            logger.warning("Could not release result-cache reference for %s", job_id, exc_info=True)
        self._delete_record_dir(record)
        return True

    def cleanup_expired_jobs(self, *, retention_hours: Optional[float] = None) -> list[str]:
        retention = timedelta(
            hours=float(SETTINGS.job_retention_hours if retention_hours is None else retention_hours)
        )
        removed: list[str] = []
        now = _utc_now()
        with self._lock:
            self.recover_jobs()
            for record in list(self._jobs.values()):
                if self._is_stale_active(record):
                    self._mark_interrupted(record)
                if record.status not in TERMINAL_STATES:
                    continue
                age_anchor = record.finished_at or record.updated_at or record.created_at
                if now - age_anchor < retention:
                    continue
                if not self._is_safe_job_dir(record.job_dir):
                    continue
                self._jobs.pop(record.job_id, None)
                self._delete_record_dir(record)
                removed.append(record.job_id)
        return removed

    def recover_jobs(self) -> list[str]:
        recovered: list[str] = []
        with self._lock:
            for job_dir in self.root_dir.iterdir():
                if not job_dir.is_dir():
                    continue
                manifest = job_dir / MANIFEST_FILENAME
                if not manifest.exists():
                    continue
                record = self._load_manifest(manifest)
                if record is None:
                    continue
                self._jobs[record.job_id] = record
                recovered.append(record.job_id)
            self.mark_stale_running_jobs()
        return recovered

    def mark_stale_running_jobs(self) -> list[str]:
        interrupted: list[str] = []
        with self._lock:
            for record in list(self._jobs.values()):
                if self._is_stale_active(record):
                    self._mark_interrupted(record)
                    interrupted.append(record.job_id)
        return interrupted

    def status_counts(self, *, include_disk: bool = False) -> Dict[str, int]:
        with self._lock:
            if include_disk:
                self.recover_jobs()
            counts: Dict[str, int] = {}
            for record in self._jobs.values():
                counts[record.status] = counts.get(record.status, 0) + 1
            return counts

    def records(self, *, include_disk: bool = False) -> list[JobRecord]:
        with self._lock:
            if include_disk:
                self.recover_jobs()
            return list(self._jobs.values())

    def pause(self, job_id: str) -> bool:
        with self._lock:
            record = self.require(job_id)
            if record.status not in {"queued", "waiting_for_capacity", "waiting_for_job_slot", "waiting_for_gpu", "waiting_for_mobilesam", "waiting_for_vlm", "retry_wait", "recovering"}:
                return False
            record.status = "paused"
            record.current_step = "Paused before analysis"
            record.updated_at = _utc_now()
            self._refresh_metrics(record)
            self.persist_job(record)
            return True

    def resume(self, job_id: str) -> bool:
        with self._lock:
            record = self.require(job_id)
            if record.status != "paused":
                return False
            record.status = "queued"
            record.current_step = "Queued after resume"
            record.updated_at = _utc_now()
            record.recovery_action = "resumed_by_user"
            self._refresh_metrics(record)
            self.persist_job(record)
            return True

    def set_recovery_action(self, job_id: str, action: str, *, pause: bool = False) -> bool:
        with self._lock:
            record = self.require(job_id)
            if record.status in TERMINAL_STATES and record.error_type != "InterruptedJob":
                return False
            record.recovery_action = action
            if pause and record.status in {"queued", "recovering", "retry_wait"}:
                record.status = "paused"
                record.current_step = "Awaiting recovery decision"
            record.updated_at = _utc_now()
            self._refresh_metrics(record)
            self.persist_job(record)
            return True

    def set_pin(self, job_id: str, pinned: bool) -> Dict[str, Any]:
        with self._lock:
            record = self.require(job_id)
            if record.status != "completed":
                raise ValueError("Only completed results can be pinned.")
            now = _to_iso(_utc_now())
            record.source_metadata["pinned"] = bool(pinned)
            record.source_metadata["pinnedAt"] = now if pinned else None
            record.updated_at = _utc_now()
            self._refresh_metrics(record)
            self.persist_job(record)
            return {
                "jobId": job_id,
                "pinned": bool(pinned),
                "pinnedAt": record.source_metadata.get("pinnedAt"),
                "updatedAt": _to_iso(record.updated_at),
            }

    def reset_for_retry(self, job_id: str, *, recovery_action: str = "retry_failed") -> bool:
        with self._lock:
            record = self.require(job_id)
            if record.status != "failed":
                return False
            record.retry_count += 1
            record.status = "queued"
            record.progress = 0.0
            record.current_step = "Queued for retry"
            record.error = None
            record.technical_error = None
            record.error_type = None
            record.finished_at = None
            record.next_retry_at = None
            record.recovery_action = recovery_action
            record.updated_at = _utc_now()
            self._refresh_metrics(record)
            self.persist_job(record)
            return True

    def mark_retry_wait(self, job_id: str, *, error: Exception) -> float:
        with self._lock:
            record = self.require(job_id)
            record.retry_count += 1
            delay = max(0.0, float(getattr(SETTINGS, "job_retry_backoff_seconds", 5.0))) * max(
                record.retry_count, 1
            )
            record.status = "retry_wait"
            record.progress = max(0.0, min(record.progress, 0.95))
            record.current_step = f"Retry scheduled after transient {type(error).__name__}"
            record.error = "Temporary analysis error; SafeTrace will retry this job."
            record.technical_error = str(error)
            record.error_type = type(error).__name__
            record.next_retry_at = _utc_now() + timedelta(seconds=delay)
            record.recovery_action = "automatic_transient_retry"
            record.updated_at = _utc_now()
            self._refresh_metrics(record)
            self.persist_job(record)
            return delay

    def activate_retry(self, job_id: str) -> bool:
        with self._lock:
            record = self.require(job_id)
            if record.status not in {"retry_wait", "recovering"}:
                return False
            record.status = "queued"
            record.current_step = "Queued after recovery"
            record.error = None
            record.next_retry_at = None
            record.updated_at = _utc_now()
            self._refresh_metrics(record)
            self.persist_job(record)
            return True

    def restart_fresh(
        self,
        job_id: str,
        *,
        purge_compatible_cache: bool = False,
        cache_purge_selected_job_ids: set[str] | None = None,
    ) -> Dict[str, Any]:
        """Transactionally reset one unfinished job while preserving its upload."""
        with self._lock:
            record = self.require(job_id)
            if record.status == "completed" or record.source_metadata.get("pinned"):
                raise JobRestartConflictError("Completed or pinned jobs cannot be restarted fresh.")
            running_states = {
                "running", "running_preprocess", "running_detector", "running_refinement", "running_report",
            }
            if record.status in running_states or (record.job_dir / LOCK_FILENAME).exists():
                raise JobRestartConflictError(
                    "Job is actively writing; wait for interruption recovery before restarting fresh."
                )
            if not record.upload_path.is_file():
                raise JobRestartConflictError("The original uploaded source is missing.")

            digest = hashlib.sha256()
            with record.upload_path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            source_checksum = digest.hexdigest()
            expected_checksum = str(record.source_metadata.get("checksumSha256") or "")
            if expected_checksum and source_checksum != expected_checksum:
                raise JobRestartConflictError("The uploaded source checksum no longer matches the job manifest.")

            snapshot = copy.deepcopy(record)
            quarantine = record.job_dir / f".restart_fresh_{uuid.uuid4().hex}.tmp"
            quarantine.mkdir(parents=False, exist_ok=False)
            cleanup_names = {
                "checkpoints", "workspace", "media", "evidence", "frames", "annotated", "tmp", "reports",
                RESULT_FILENAME, PARTIAL_PIPELINE_FILENAME,
            }
            targets = [record.job_dir / name for name in cleanup_names]
            targets.extend(record.job_dir.glob(f"{PARTIAL_PIPELINE_FILENAME}.*.tmp"))
            targets.extend(record.job_dir.glob(f"{RESULT_FILENAME}.*.tmp"))
            targets.extend(record.job_dir.glob(f"{MANIFEST_FILENAME}*.tmp"))
            moved: list[tuple[Path, Path]] = []
            try:
                for source in dict.fromkeys(targets):
                    if not source.exists() or source == quarantine:
                        continue
                    destination = quarantine / source.name
                    if destination.exists():
                        destination = quarantine / f"{uuid.uuid4().hex}_{source.name}"
                    os.replace(source, destination)
                    moved.append((source, destination))

                now = _utc_now()
                original_created_at = _to_iso(record.created_at)
                preserved_identity = copy.deepcopy(dict(record.metrics.get("executionIdentity") or {}))
                preserved_cache_key = record.metrics.get("resultCacheKey")
                generation = int(record.metrics.get("restartGeneration") or 0) + 1
                record.output_dir.mkdir(parents=True, exist_ok=True)
                record.result = None
                record.result_path = None
                record.media_files = {}
                record.status = "queued"
                record.progress = 0.0
                record.current_step = "Queued for fresh analysis"
                record.error = None
                record.technical_error = None
                record.error_type = None
                record.persistence_warning = None
                record.retry_count = 0
                record.next_retry_at = None
                record.started_at = None
                record.finished_at = None
                record.created_at = now
                record.updated_at = now
                record.recovery_action = "restart_fresh_by_user"
                record.source_metadata.setdefault("originalCreatedAt", original_created_at)
                record.source_metadata["restartFreshAt"] = _to_iso(now)
                record.source_metadata["restartFreshGeneration"] = generation
                record.metrics = {
                    "executionIdentity": preserved_identity,
                    "resultCacheKey": preserved_cache_key,
                    "resultCacheHit": False,
                    "bypassResultCacheOnce": True,
                    "restartGeneration": generation,
                    "componentDiagnostics": {
                        **_base_component_diagnostics(record.settings, stage="queued_restart_fresh"),
                        "restartFresh": True,
                        "restartFreshGeneration": generation,
                        "sourceChecksumVerified": True,
                    },
                }
                self._refresh_metrics(record)
                self.persist_job(record)
                self._persist_recovery_checkpoint(record, stage="upload_complete")
            except Exception:
                shutil.rmtree(record.output_dir, ignore_errors=True)
                for original, staged in reversed(moved):
                    if staged.exists():
                        os.replace(staged, original)
                record.__dict__.clear()
                record.__dict__.update(snapshot.__dict__)
                self.persist_job(record)
                shutil.rmtree(quarantine, ignore_errors=True)
                raise

            cache_warning = None
            purge_result: Dict[str, Any] = {
                "requested": purge_compatible_cache,
                "purged": [],
                "retained": [],
            }
            try:
                from .result_cache import purge_compatible_cache_entries, release_job_reference

                release_job_reference(snapshot)
                if purge_compatible_cache:
                    purge_result = purge_compatible_cache_entries(
                        [snapshot],
                        selected_job_ids=set(cache_purge_selected_job_ids or {job_id}),
                    )
            except Exception as exc:
                cache_warning = f"{type(exc).__name__}: {exc}"
                logger.warning("Fresh restart cache cleanup failed for %s", job_id, exc_info=True)
            shutil.rmtree(quarantine, ignore_errors=True)
            return {
                "jobId": job_id,
                "status": record.status,
                "recoveryAction": record.recovery_action,
                "sourceChecksum": source_checksum,
                "sourcePreserved": record.upload_path.is_file(),
                "restartGeneration": generation,
                "cacheBypassOnce": True,
                "cachePurge": purge_result,
                "cacheCleanupWarning": cache_warning,
            }

    def persist_job(self, record: JobRecord) -> None:
        if record.result is not None:
            if record.persistence_warning:
                record.result["persistenceWarning"] = record.persistence_warning
                technical_details = dict(record.result.get("technicalDetails") or {})
                technical_details["manifestPersistenceWarning"] = record.persistence_warning
                record.result["technicalDetails"] = technical_details
            validate_result_ownership(record.result, result_ownership_context(record))
            result_path = record.result_path or (record.job_dir / RESULT_FILENAME)
            _atomic_write_json(result_path, record.result)
            record.result_path = result_path

        media_files = {}
        for filename, path in record.media_files.items():
            relative = _safe_relative_path(path, record.job_dir)
            if relative is not None:
                media_files[filename] = relative

        result_path = _safe_relative_path(record.result_path, record.job_dir) if record.result_path else None
        manifest = {
            "jobId": record.job_id,
            "status": record.status,
            "progress": record.progress,
            "currentStep": record.current_step,
            "error": record.error,
            "technicalError": record.technical_error,
            "errorType": record.error_type,
            "createdAt": _to_iso(record.created_at),
            "updatedAt": _to_iso(record.updated_at),
            "startedAt": _to_iso(record.started_at),
            "finishedAt": _to_iso(record.finished_at),
            "query": record.query,
            "settings": _settings_to_manifest(record.settings),
            "input": {
                "originalFilename": record.original_filename,
                "uploadPath": _safe_relative_path(record.upload_path, record.job_dir),
                "mediaType": record.media_type,
                "sizeBytes": record.size_bytes,
                "sourceMetadata": record.source_metadata,
            },
            "output": {
                "mediaDir": _safe_relative_path(record.output_dir, record.job_dir),
                "resultPath": result_path,
                "mediaFiles": media_files,
            },
            "metrics": record.metrics,
            "retry": {
                "count": record.retry_count,
                "maxAttempts": record.max_attempts,
                "nextRetryAt": _to_iso(record.next_retry_at),
                "recoveryAction": record.recovery_action,
            },
            "persistenceWarning": record.persistence_warning,
            "manifestPersistenceWarning": record.persistence_warning,
        }
        try:
            _atomic_write_json(record.manifest_path, manifest)
        except JobManifestPersistenceError as exc:
            warning = str(exc)
            self._record_manifest_warning(record, warning)
            if record.manifest_path.exists():
                logger.warning("Job manifest persistence failed for %s: %s", record.job_id, warning)
                return
            raise

    def load_job(self, job_id: str) -> Optional[JobRecord]:
        if not _is_safe_job_id(job_id):
            return None
        return self._load_manifest(self.root_dir / job_id / MANIFEST_FILENAME)

    def _load_manifest(self, manifest_path: Path) -> Optional[JobRecord]:
        root = self.root_dir.resolve()
        job_dir = manifest_path.parent.resolve()
        if job_dir == root or root not in job_dir.parents:
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        job_id = str(manifest.get("jobId") or job_dir.name)
        if not _is_safe_job_id(job_id):
            return None

        input_meta = dict(manifest.get("input") or {})
        output_meta = dict(manifest.get("output") or {})
        clean_name = safe_filename(str(input_meta.get("originalFilename") or "upload.bin"))
        upload_path = _resolve_manifest_path(job_dir, input_meta.get("uploadPath"), job_dir / "uploads" / clean_name)
        output_dir = _resolve_manifest_path(job_dir, output_meta.get("mediaDir"), job_dir / "media")

        result_path = None
        result = None
        raw_result_path = output_meta.get("resultPath")
        if raw_result_path:
            result_path = _resolve_manifest_path(job_dir, raw_result_path, job_dir / RESULT_FILENAME)
            if result_path.exists():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    result = None

        media_files: Dict[str, Path] = {}
        for filename, raw_path in dict(output_meta.get("mediaFiles") or {}).items():
            clean_media_name = Path(str(filename)).name
            media_files[clean_media_name] = _resolve_manifest_path(job_dir, raw_path, output_dir / clean_media_name)

        status = str(manifest.get("status") or "failed")
        if status not in {
            "queued", "waiting_for_capacity", "waiting_for_job_slot", "waiting_for_gpu", "waiting_for_mobilesam", "waiting_for_vlm", "running_preprocess", "running_detector", "running_refinement", "running_report", "running", "retry_wait", "recovering", "paused",
            "completed", "failed", "cancelled",
        }:
            status = "failed"

        retry_meta = dict(manifest.get("retry") or {})

        record = JobRecord(
            job_id=job_id,
            status=status,  # type: ignore[arg-type]
            progress=float(manifest.get("progress") or 0.0),
            current_step=str(manifest.get("currentStep") or "Recovered job"),
            error=manifest.get("error"),
            query=str(manifest.get("query") or ""),
            settings=_settings_from_manifest(dict(manifest.get("settings") or {})),
            original_filename=clean_name,
            media_type=str(input_meta.get("mediaType") or media_type_for(clean_name)),
            size_bytes=int(input_meta.get("sizeBytes") or 0),
            job_dir=job_dir,
            upload_path=upload_path,
            output_dir=output_dir,
            result=result,
            result_path=result_path,
            technical_error=manifest.get("technicalError"),
            error_type=manifest.get("errorType"),
            media_files=media_files,
            metrics=dict(manifest.get("metrics") or {}),
            persistence_warning=manifest.get("persistenceWarning") or manifest.get("manifestPersistenceWarning"),
            source_metadata=dict(input_meta.get("sourceMetadata") or {}),
            retry_count=int(retry_meta.get("count") or 0),
            max_attempts=int(retry_meta.get("maxAttempts") or 0),
            next_retry_at=_parse_datetime(retry_meta.get("nextRetryAt")),
            recovery_action=retry_meta.get("recoveryAction"),
            created_at=_parse_datetime(manifest.get("createdAt")) or _utc_now(),
            updated_at=_parse_datetime(manifest.get("updatedAt")) or _utc_now(),
            started_at=_parse_datetime(manifest.get("startedAt")),
            finished_at=_parse_datetime(manifest.get("finishedAt")),
        )
        self._refresh_metrics(record)
        if record.result is not None:
            try:
                if record.result.get("resultSchemaVersion") is None:
                    record.result = stamp_result_ownership(record.result, result_ownership_context(record))
                else:
                    validate_result_ownership(record.result, result_ownership_context(record))
            except ResultOwnershipError as exc:
                diagnostics = dict(record.metrics.get("componentDiagnostics") or {})
                diagnostics.update(
                    {
                        "resultIntegrityStatus": "blocked",
                        "resultIntegrityErrorCode": exc.code,
                        "resultIntegrityIssues": list(exc.issues),
                    }
                )
                record.metrics["componentDiagnostics"] = diagnostics
        return record

    def _refresh_metrics(self, record: JobRecord) -> None:
        metrics = dict(record.metrics)
        _merge_component_diagnostics(metrics, record.settings)
        metrics.update(
            {
                "queuedAt": _to_iso(record.created_at),
                "startedAt": _to_iso(record.started_at),
                "finishedAt": _to_iso(record.finished_at),
                "inputSizeBytes": record.size_bytes,
                "inputMediaType": record.media_type,
                "inputExtension": Path(record.original_filename).suffix.lower(),
                "statusOutcome": record.status,
                "reviewMode": record.settings.review_mode,
                "sourceMetadata": record.source_metadata,
                "retryCount": record.retry_count,
                "maxAttempts": record.max_attempts,
                "nextRetryAt": _to_iso(record.next_retry_at),
                "recoveryAction": record.recovery_action,
            }
        )
        if record.finished_at:
            started_anchor = record.started_at or record.created_at
            metrics["totalWallClockSeconds"] = max(
                (record.finished_at - started_anchor).total_seconds(),
                0.0,
            )
        elif record.created_at and record.updated_at:
            metrics["elapsedWallClockSeconds"] = max(
                (record.updated_at - record.created_at).total_seconds(),
                0.0,
            )
        if record.error_type:
            metrics["errorType"] = record.error_type
        if record.error:
            metrics["errorMessage"] = record.error
        if record.persistence_warning:
            metrics["manifestPersistenceWarning"] = record.persistence_warning
            diagnostics = dict(metrics.get("componentDiagnostics") or {})
            diagnostics["manifestPersistenceWarning"] = record.persistence_warning
            metrics["componentDiagnostics"] = diagnostics
        record.metrics = metrics

    def _record_manifest_warning(self, record: JobRecord, warning: str) -> None:
        record.persistence_warning = warning
        metrics = dict(record.metrics)
        metrics["manifestPersistenceWarning"] = warning
        diagnostics = dict(metrics.get("componentDiagnostics") or {})
        diagnostics["manifestPersistenceWarning"] = warning
        metrics["componentDiagnostics"] = diagnostics
        record.metrics = metrics
        if record.result is not None:
            record.result["persistenceWarning"] = warning
            technical_details = dict(record.result.get("technicalDetails") or {})
            technical_details["manifestPersistenceWarning"] = warning
            record.result["technicalDetails"] = technical_details

    def _persist_recovery_checkpoint(self, record: JobRecord, *, stage: Optional[str] = None) -> None:
        try:
            from .recovery_checkpoints import stage_from_state, write_checkpoint

            diagnostics = dict(record.metrics.get("componentDiagnostics") or {})
            checkpoint_stage = stage or diagnostics.get("recoveryCheckpointStage") or stage_from_state(record.current_step, record.status, record.progress)
            result_frames = list((record.result or {}).get("frames") or [])
            evidence_ids = list(diagnostics.get("completedEvidenceIds") or []) or [
                str(frame.get("id")) for frame in result_frames if frame.get("id")
            ]
            last_frame = diagnostics.get("lastCompletedFrameIndex")
            if last_frame is None:
                last_frame = max((int(frame.get("frameNumber") or 0) for frame in result_frames), default=None)
            completed_timestamp = diagnostics.get("lastCompletedTimestamp")
            if completed_timestamp is None:
                timestamps = [frame.get("timestampSeconds") for frame in result_frames if isinstance(frame.get("timestampSeconds"), (int, float))]
                completed_timestamp = max(timestamps) if timestamps else None
            candidate_ids = list(diagnostics.get("completedCandidateIds") or [])
            signature = [checkpoint_stage, last_frame, sorted(candidate_ids), sorted(evidence_ids)]
            if record.metrics.get("lastRecoveryCheckpointSignature") == signature and checkpoint_stage != "completed":
                return
            identity = dict(record.metrics.get("executionIdentity") or {})
            path = write_checkpoint(
                record.job_dir,
                job_id=record.job_id,
                stage=checkpoint_stage,
                identity=identity,
                progress=record.progress,
                status=record.status,
                current_step=record.current_step,
                completed_frame_index=last_frame,
                completed_timestamp=completed_timestamp,
                completed_candidate_ids=candidate_ids,
                completed_evidence_ids=evidence_ids,
                partial_output_reference=(
                    diagnostics.get("partialOutputReference")
                    or (str(record.result_path.relative_to(record.job_dir)) if record.result_path and record.result_path.exists() else None)
                ),
                partial_output_checksum_sha256=diagnostics.get("partialOutputChecksumSha256"),
            )
            record.metrics["lastRecoveryCheckpointStage"] = checkpoint_stage
            record.metrics["lastRecoveryCheckpointPath"] = str(path.relative_to(record.job_dir))
            record.metrics["lastRecoveryCheckpointSignature"] = signature
        except Exception:
            logger.warning("Could not persist recovery checkpoint for %s", record.job_id, exc_info=True)

    def _is_safe_job_dir(self, job_dir: Path) -> bool:
        root = self.root_dir.resolve()
        target = job_dir.resolve()
        return target != root and root in target.parents

    def _delete_record_dir(self, record: JobRecord) -> None:
        if not self._is_safe_job_dir(record.job_dir):
            raise RuntimeError("Refusing to delete job directory outside API job root.")
        shutil.rmtree(record.job_dir, ignore_errors=True)

    def _is_stale_active(self, record: JobRecord) -> bool:
        if record.status not in {"running", "recovering"}:
            return False
        cutoff = _utc_now() - timedelta(minutes=float(SETTINGS.stale_running_minutes))
        return record.updated_at < cutoff

    def _mark_interrupted(self, record: JobRecord) -> None:
        lock_path = record.job_dir / LOCK_FILENAME
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        recoverable = bool(
            record.source_metadata.get("batchId")
            or record.source_metadata.get("recoverOnRestart")
            or record.metrics.get("recoverOnRestart")
        )
        attempts_available = record.retry_count < max(record.max_attempts, 0)
        now = _utc_now()
        if recoverable and attempts_available:
            record.retry_count += 1
            record.status = "recovering"
            record.progress = 0.0
            record.current_step = "Recovering after backend restart"
            record.error = None
            record.error_type = "InterruptedJob"
            record.finished_at = None
            record.next_retry_at = now
            record.recovery_action = "requeue_after_restart"
            stage = "recovering"
        else:
            record.status = "failed"
            record.progress = 1.0
            record.current_step = "Analysis interrupted"
            record.error = "Job interrupted by backend restart."
            record.error_type = "InterruptedJob"
            record.finished_at = now
            record.recovery_action = "manual_retry_required"
            stage = "failed"
        record.updated_at = now
        _merge_component_diagnostics(
            record.metrics,
            record.settings,
            stage=stage,
            updates={"errorType": record.error_type, "errorMessage": record.error},
        )
        self._refresh_metrics(record)
        self.persist_job(record)


def run_pipeline(
    *,
    upload_path: Path,
    query: str,
    fps: float,
    top_k: int,
    device: str,
    enable_vlm: bool,
    vlm_profile: str = VLM_PROFILE_RULE_BASED,
    vlm_model_dir: Optional[Path] = None,
    safe_mode: bool = False,
    use_case_profile: Optional[Dict[str, Any]] = None,
    review_mode: str = "fast_local",
    component_diagnostics: Optional[Dict[str, Any]] = None,
    workspace: Optional[Path] = None,
):
    """Run the existing SafeTrace pipeline lazily.

    The heavy pipeline import and construction happen only inside this function.
    """
    from src.config import SETTINGS
    from src.pipeline import SafeTracePipeline

    normalized_profile = normalize_vlm_profile(vlm_profile)
    effective_safe_mode = bool(safe_mode or analysis_safe_mode())
    resolved_model_dir = vlm_model_dir or resolve_vlm_profile_model_dir(normalized_profile)
    lightweight_worker_effective = bool(
        effective_safe_mode
        and getattr(SETTINGS, "lightweight_vlm_worker_enabled", False)
        and not vlm_hard_disabled()
        and enable_vlm
        and normalized_profile in VLM_SAFE_MODE_WORKER_PROFILES
    )
    effective_vlm_enabled = bool(
        lightweight_worker_effective
        or (
            not effective_safe_mode
            and not vlm_hard_disabled()
            and enable_vlm
            and normalized_profile != VLM_PROFILE_RULE_BASED
        )
    )
    normalized_review_mode = "comprehensive" if review_mode == "comprehensive" else "fast_local"
    if normalized_review_mode == "comprehensive":
        fps = max(float(fps), float(getattr(SETTINGS, "comprehensive_review_fps", 2.0)))
        top_k = max(int(top_k), int(getattr(SETTINGS, "comprehensive_review_top_k", 12)))

    with _compatible_runtime_settings(
        device=device,
        enable_vlm=effective_vlm_enabled,
        vlm_profile=normalized_profile,
        vlm_model_dir=resolved_model_dir if effective_vlm_enabled else None,
        safe_mode=effective_safe_mode,
        review_mode=normalized_review_mode,
    ):
        pipeline = SafeTracePipeline(
            use_case_profile=dict(use_case_profile or {}),
            query_context=query,
            workspace=workspace,
        )
        try:
            return pipeline.run([upload_path], query=query, fps=fps, k=top_k)
        finally:
            if component_diagnostics is not None:
                component_diagnostics.update(dict(getattr(pipeline, "component_diagnostics", {}) or {}))


def _run_pipeline_with_timeout(
    *,
    timeout_seconds: float,
    component_diagnostics: Dict[str, Any],
    **kwargs,
):
    timeout = max(0.0, float(timeout_seconds or 0.0))
    if timeout <= 0:
        return run_pipeline(component_diagnostics=component_diagnostics, **kwargs)

    outcome: Dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = run_pipeline(component_diagnostics=component_diagnostics, **kwargs)
        except BaseException as exc:  # pragma: no cover - re-raised in caller thread
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        stage = component_diagnostics.get("currentPipelineStage") or "unknown"
        raise PipelineTimeoutError(
            f"SafeTrace analysis exceeded {timeout:.0f}s while stage '{stage}' was active."
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _persist_partial_pipeline_result(record: JobRecord, raw_result: list[dict[str, Any]]) -> tuple[str, str]:
    destination = record.job_dir / PARTIAL_PIPELINE_FILENAME
    temporary = record.job_dir / f"{PARTIAL_PIPELINE_FILENAME}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(raw_result, handle, sort_keys=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    return destination.name, checksum


def _reusable_partial_pipeline_result(record: JobRecord) -> list[dict[str, Any]] | None:
    try:
        from .recovery_checkpoints import latest_reusable_partial

        reusable = latest_reusable_partial(
            record.job_dir,
            dict(record.metrics.get("executionIdentity") or {}),
            expected_job_id=record.job_id,
        )
        if reusable is None:
            return None
        record.metrics["recoveryPartialResultReused"] = True
        record.metrics["recoveryResumeCheckpoint"] = reusable["checkpoint"].get("stage")
        record.metrics["recoveryPartialArtifact"] = Path(reusable["artifactPath"]).name
        return list(reusable["rawResult"])
    except Exception:
        logger.warning("Could not reuse partial pipeline output for %s", record.job_id, exc_info=True)
        return None


def _analysis_heartbeat_loop(
    store: JobStore,
    job_id: str,
    stop_event: threading.Event,
    started: float,
    *,
    interval_seconds: float = ANALYSIS_HEARTBEAT_SECONDS,
) -> None:
    while not stop_event.wait(max(float(interval_seconds), 0.1)):
        elapsed = _format_elapsed(time.perf_counter() - started)
        record = store.get(job_id)
        diagnostics = dict((record.metrics if record else {}).get("componentDiagnostics") or {})
        current_stage = diagnostics.get("currentPipelineStage") or "pipeline_running"
        if not store.heartbeat(
            job_id,
            progress=0.35,
            current_step=(
                "Running SafeTrace analysis. Still working locally after "
                f"{elapsed}; current stage: {current_stage}."
            ),
            diagnostic_updates=diagnostics,
        ):
            return


def execute_analysis_job(store: JobStore, job_id: str) -> None:
    try:
        record = store.require(job_id)
    except KeyError:
        return
    if record.status in TERMINAL_STATES or record.status == "paused":
        return
    if not store.acquire_execution_lock(job_id):
        return

    try:
        with _analysis_capacity_slot(store, job_id):
            record = store.require(job_id)
            execution_context = JobExecutionContext.from_record(record)
            execution_context.workspace.mkdir(parents=True, exist_ok=True)
            execution_context.temporary_artifact_namespace.mkdir(parents=True, exist_ok=True)
            if record.status in TERMINAL_STATES or record.status == "paused":
                return
            started = time.perf_counter()
            component_diagnostics = _base_component_diagnostics(record.settings, stage="preparing")
            store.update_status(
                job_id,
                status="running",
                progress=0.15,
                current_step="Preparing selected media",
                diagnostic_updates={"currentPipelineStage": "preparing"},
            )
            store.persist_recovery_stage(job_id, "media_probe_complete")
            try:
                from .result_cache import publish_completed_result, restore_cached_result

                bypass_result_cache = bool(record.metrics.get("bypassResultCacheOnce"))
                cached = None if bypass_result_cache else restore_cached_result(record)
                if cached is not None:
                    cached_result, cached_media = cached
                    record.metrics["resultCacheHit"] = True
                    record.metrics["resultCacheReuseReason"] = "exact_compatible_key"
                    store.register_media_files(job_id, cached_media)
                    store.complete_job(job_id, cached_result)
                    return
                vlm_profile = normalize_vlm_profile(record.settings.vlm_profile)
                effective_vlm_enabled = should_enable_vlm(record.settings)
                raw_result = _reusable_partial_pipeline_result(record)
                if raw_result is None:
                    store.update_status(
                        job_id,
                        status="running",
                        progress=0.35,
                        current_step="Running SafeTrace analysis. This stage may take a few minutes.",
                        diagnostic_updates={"currentPipelineStage": "pipeline_starting"},
                    )
                    heartbeat_stop = threading.Event()
                    heartbeat_thread = threading.Thread(
                        target=_analysis_heartbeat_loop,
                        args=(store, job_id, heartbeat_stop, time.perf_counter()),
                        daemon=True,
                    )
                    heartbeat_thread.start()
                    requested_device = str(record.settings.device or "auto").strip().lower()
                    gpu_requested = requested_device != "cpu" and bool(
                        device_gateway_payload(SETTINGS).get("components", {}).get("detector", {}).get("selected") == "cuda"
                    )
                    gpu_slot = (
                        _resource_capacity_slot(
                            store,
                            job_id,
                            semaphore=_gpu_semaphore(),
                            wait_status="waiting_for_gpu",
                            wait_step="Waiting for a bounded GPU inference lane",
                            resource="gpu",
                        )
                        if gpu_requested
                        else nullcontext()
                    )
                    mid_vlm_enabled = bool(getattr(SETTINGS, "mid_vlm_enabled", False))
                    vlm_slot = (
                        _resource_capacity_slot(
                            store,
                            job_id,
                            semaphore=_vlm_semaphore(),
                            wait_status="waiting_for_vlm",
                            wait_step="Waiting for the local visual-review lane",
                            resource="vlm",
                        )
                        if (effective_vlm_enabled and vlm_profile != VLM_PROFILE_RULE_BASED) or mid_vlm_enabled
                        else nullcontext()
                    )
                    component_diagnostics = _merge_component_diagnostics(
                        dict(record.metrics),
                        record.settings,
                        stage="pipeline_starting",
                        updates={
                            "analysisConcurrency": analysis_concurrency_limit(),
                            "gpuInferenceConcurrency": gpu_inference_concurrency_limit(),
                            "vlmConcurrency": vlm_concurrency_limit(),
                            "vlmConcurrencyLimited": bool(
                                (effective_vlm_enabled and vlm_profile != VLM_PROFILE_RULE_BASED) or mid_vlm_enabled
                            ),
                        },
                    )
                    try:
                        with gpu_slot, vlm_slot:
                            store.update_status(
                                job_id,
                                status="running_detector",
                                progress=0.35,
                                current_step="Running detector and local evidence analysis",
                                diagnostic_updates={"currentPipelineStage": "running_detector", "capacityReason": None},
                            )
                            raw_result = _run_pipeline_with_timeout(
                                timeout_seconds=float(getattr(SETTINGS, "analysis_job_timeout_seconds", 600.0) or 0.0),
                                component_diagnostics=component_diagnostics,
                                upload_path=record.upload_path,
                                query=record.query,
                                fps=record.settings.fps,
                                top_k=record.settings.top_k,
                                device=record.settings.device,
                                enable_vlm=effective_vlm_enabled,
                                vlm_profile=vlm_profile,
                                vlm_model_dir=resolve_vlm_profile_model_dir(vlm_profile) if effective_vlm_enabled else None,
                                safe_mode=record.settings.safe_mode,
                                use_case_profile=dict(record.settings.use_case_profile or {}),
                                review_mode=record.settings.review_mode,
                                workspace=execution_context.workspace,
                            )
                    finally:
                        heartbeat_stop.set()
                        heartbeat_thread.join(timeout=0.5)
                    partial_reference, partial_checksum = _persist_partial_pipeline_result(record, list(raw_result or []))
                else:
                    partial_reference = str(record.metrics.get("recoveryPartialArtifact") or PARTIAL_PIPELINE_FILENAME)
                    partial_checksum = hashlib.sha256((record.job_dir / partial_reference).read_bytes()).hexdigest()
                    component_diagnostics = _merge_component_diagnostics(
                        dict(record.metrics), record.settings, stage="pipeline_reused_from_checkpoint",
                        updates={"recoveryPartialResultReused": True},
                    )
                store.update_status(
                    job_id,
                    status="running_report",
                    progress=0.8,
                    current_step="SafeTrace pipeline completed; preparing report",
                    diagnostic_updates=component_diagnostics,
                )
                candidate_ids = [str(index) for index, _item in enumerate(raw_result or [])]
                store.persist_recovery_stage(job_id, "frame_sampling_complete")
                store.persist_recovery_stage(job_id, "ranking_complete", completed_candidate_ids=candidate_ids)
                store.persist_recovery_stage(job_id, "detector_work_complete", completed_candidate_ids=candidate_ids, partial_output_reference=partial_reference, partial_output_checksum_sha256=partial_checksum)
                if component_diagnostics.get("mobileSamAttempted") or component_diagnostics.get("mobileSamRequested"):
                    store.persist_recovery_stage(job_id, "mobile_sam_subset_complete", completed_candidate_ids=candidate_ids, partial_output_reference=partial_reference, partial_output_checksum_sha256=partial_checksum)
                if component_diagnostics.get("vlmAttempted") or effective_vlm_enabled:
                    store.persist_recovery_stage(job_id, "vlm_subset_complete", completed_candidate_ids=candidate_ids, partial_output_reference=partial_reference, partial_output_checksum_sha256=partial_checksum)
                store.update_status(
                    job_id,
                    status="running",
                    progress=0.85,
                    current_step="Normalizing evidence report",
                    diagnostic_updates={"currentPipelineStage": "normalizing", "capacityReason": None},
                )
                normalization_started = time.perf_counter()
                pending_media_files: list[tuple[str, Path]] = []
                result = normalize_pipeline_results(
                    job_id=job_id,
                    media_name=record.original_filename,
                    media_type=record.media_type,
                    media_size_bytes=record.size_bytes,
                    query=record.query,
                    raw_frames=raw_result,
                    media_dir=record.output_dir,
                    register_media=lambda filename, path: pending_media_files.append((filename, path)),
                    source_relative_path=str(
                        (record.source_metadata or {}).get("sourceRelativePath") or record.original_filename
                    ),
                )
                source_metadata = dict(record.source_metadata or {})
                result["sourceMetadata"] = source_metadata
                result["reviewMode"] = record.settings.review_mode
                result["requestedReviewMode"] = record.settings.review_mode
                result["analysisSetup"] = build_analysis_setup_payload(record, result)
                technical_details = result.setdefault("technicalDetails", {})
                technical_details["sourceMetadata"] = source_metadata
                technical_details["reviewMode"] = {
                    "requested": record.settings.review_mode,
                    "actual": record.settings.review_mode,
                }
                technical_details["analysisSetup"] = result["analysisSetup"]
                for frame in result.get("frames") or []:
                    frame.setdefault("sourceMetadata", source_metadata)
                    frame.setdefault("videoFilename", source_metadata.get("originalFilename") or record.original_filename)
                    frame.setdefault("sourceRelativePath", source_metadata.get("sourceRelativePath") or record.original_filename)
                    frame.setdefault("sourceGroup", source_metadata.get("sourceGroupPath") or "")
                    frame.setdefault("batchId", source_metadata.get("batchId"))
                    frame.setdefault("jobId", job_id)
                    violation_ids = [str(item.get("id") or item.get("name")) for item in frame.get("violations") or []]
                    frame.setdefault("findingId", violation_ids[0] if violation_ids else None)
                    frame.setdefault("findingIds", violation_ids)
                    timestamp = frame.get("timestamp")
                    if timestamp is not None:
                        from src.aggregation import timestamp_to_seconds
                        frame.setdefault("timestampSeconds", timestamp_to_seconds(str(timestamp)))
                        frame.setdefault("timestampLabel", str(timestamp))
                frame_by_id = {str(frame.get("id")): frame for frame in result.get("frames") or []}
                for event in result.get("events") or []:
                    event["batchId"] = source_metadata.get("batchId")
                    event["jobId"] = job_id
                    event["sourceRelativePath"] = source_metadata.get("sourceRelativePath")
                    for support in event.get("supportingFrames") or []:
                        source_frame = frame_by_id.get(str(support.get("frameId")), {})
                        support.update(
                            {
                                "eventId": event.get("id"),
                                "findingId": event.get("type"),
                                "batchId": source_metadata.get("batchId"),
                                "jobId": job_id,
                                "sourceRelativePath": source_metadata.get("sourceRelativePath"),
                                "timestampSeconds": source_frame.get("timestampSeconds"),
                                "timestampLabel": source_frame.get("timestampLabel") or support.get("timestamp"),
                            }
                        )
                evidence_ids = [str(frame.get("id")) for frame in result.get("frames") or [] if frame.get("id")]
                store.persist_recovery_stage(job_id, "aggregation_complete", completed_evidence_ids=evidence_ids, partial_output_reference=partial_reference, partial_output_checksum_sha256=partial_checksum)
                store.register_media_files(job_id, pending_media_files)
                normalization_seconds = time.perf_counter() - normalization_started
                technical_details = result.setdefault("technicalDetails", {})
                technical_details["pipelineWallClockSeconds"] = time.perf_counter() - started
                technical_details["normalizationWallClockSeconds"] = normalization_seconds
                technical_details["reportGenerationWallClockSeconds"] = normalization_seconds
                technical_details["componentDiagnostics"] = component_diagnostics
                store.persist_recovery_stage(
                    job_id,
                    "evidence_report_complete",
                    completed_evidence_ids=evidence_ids,
                    partial_output_reference="outputs",
                    partial_output_checksum_sha256=None,
                )
                record.metrics["bypassResultCacheOnce"] = False
                store.complete_job(job_id, result)
                completed_record = store.require(job_id)
                publish_completed_result(completed_record)
            except Exception as exc:  # pragma: no cover - exercised through API tests
                logger.exception("SafeTrace analysis job %s failed with %s", job_id, type(exc).__name__)
                persisted = store.get(job_id)
                failure_diagnostics = dict(
                    component_diagnostics
                    or ((persisted.metrics if persisted else {}).get("componentDiagnostics") or {})
                )
                failure_diagnostics.update(
                    {
                        "errorType": type(exc).__name__,
                        "errorMessage": str(exc),
                    }
                )
                current = store.get(job_id)
                recoverable_batch = bool(current and current.source_metadata.get("batchId"))
                transient = isinstance(exc, (OSError, TimeoutError, PipelineTimeoutError))
                attempts_left = bool(current and current.retry_count < current.max_attempts)
                if recoverable_batch and transient and attempts_left:
                    delay = store.mark_retry_wait(job_id, error=exc)

                    def retry_later() -> None:
                        if store.activate_retry(job_id):
                            schedule_analysis_jobs(store, [job_id])

                    timer = threading.Timer(delay, retry_later)
                    timer.daemon = True
                    timer.start()
                else:
                    store.update_status(
                        job_id,
                        status="failed",
                        progress=1.0,
                        current_step="Analysis failed",
                        error="Analysis could not be completed. Please try again.",
                        technical_error=str(exc),
                        error_type=type(exc).__name__,
                        diagnostic_updates=failure_diagnostics,
                    )
    finally:
        store.release_execution_lock(job_id)


@dataclass
class _ScheduledJob:
    store: JobStore
    job_id: str
    enqueue_sequence: int
    batch_id: str | None
    batch_enqueue_sequence: int
    child_sequence: int
    worker_slot: int | None = None


class LocalAnalysisScheduler:
    """Bounded in-process scheduler with deterministic batch-child fairness.

    Parent batches never occupy a worker. Child jobs are admitted in manifest
    order. The oldest queued batch is offered slots first, up to its configured
    per-batch cap, then the next oldest batch or standalone job is considered.
    """

    def __init__(self, adaptive_manager: AdaptiveWorkerManager | None = None) -> None:
        self._lock = threading.Condition(threading.RLock())
        self._pending: list[_ScheduledJob] = []
        self._active: Dict[str, _ScheduledJob] = {}
        self._threads: Dict[str, threading.Thread] = {}
        # A batch is ordered by the scheduler sequence assigned when its first
        # child enters this process.  Persisted timestamps are useful metadata,
        # but cannot be compared safely with standalone integer sequences.
        self._batch_enqueue_sequences: Dict[str, int] = {}
        self._events: deque[Dict[str, Any]] = deque(maxlen=100)
        self._adaptive = adaptive_manager or _ADAPTIVE_WORKER_MANAGER
        self._shutdown = threading.Event()
        self._controller_thread = threading.Thread(
            target=self._controller_loop,
            name="safetrace-adaptive-worker-manager",
            daemon=True,
        )
        self._controller_thread.start()

    def _controller_loop(self) -> None:
        while not self._shutdown.wait(1.0):
            with self._lock:
                adaptive_payload = self._adaptive.payload()
                if (
                    self._pending
                    or self._active
                    or int(adaptive_payload.get("currentCapacity") or self._adaptive.minimum)
                    > self._adaptive.minimum
                ):
                    self._dispatch_locked()

    def shutdown(self) -> None:
        self._shutdown.set()
        with self._lock:
            self._lock.notify_all()

    @staticmethod
    def _entry_identity(entry: _ScheduledJob) -> tuple[Any, ...]:
        record = entry.store.require(entry.job_id)
        settings = record.settings
        return (
            str(settings.device or "auto").lower(),
            bool(settings.enable_vlm and settings.vlm_enabled),
            normalize_vlm_profile(settings.vlm_profile),
            bool(settings.safe_mode),
            str(settings.review_mode or "fast_local"),
        )

    def _adaptive_capacity_locked(self) -> int:
        if not self._adaptive.enabled:
            return analysis_concurrency_limit()
        active_identities: list[tuple[Any, ...]] = []
        for entry in self._active.values():
            try:
                active_identities.append(self._entry_identity(entry))
            except KeyError:
                continue
        compatible_queued = 0
        if active_identities:
            primary = active_identities[0]
            for entry in self._pending:
                try:
                    compatible_queued += int(self._entry_identity(entry) == primary)
                except KeyError:
                    continue
        active_vlm = 0
        for entry in self._active.values():
            try:
                settings = entry.store.require(entry.job_id).settings
                active_vlm += int(bool(settings.enable_vlm and settings.vlm_enabled))
            except KeyError:
                continue
        adaptive_capacity = self._adaptive.evaluate(
            active_count=len(self._active),
            queued_count=len(self._pending),
            compatible_queued_count=compatible_queued,
            active_third_slot=any(entry.worker_slot == 3 for entry in self._active.values()),
            vlm_lane_in_use=min(active_vlm, vlm_concurrency_limit()),
            mobile_sam_lane_in_use=min(len(self._active), int(getattr(SETTINGS, "mobile_sam_concurrency", 1) or 1)),
        )
        return max(analysis_concurrency_limit(), adaptive_capacity)

    def _record_event(self, event: str, entry: _ScheduledJob, **details: Any) -> None:
        self._events.append(
            {
                "timestamp": _to_iso(_utc_now()),
                "event": event,
                "jobId": entry.job_id,
                "batchId": entry.batch_id,
                "enqueueSequence": entry.enqueue_sequence,
                "batchEnqueueSequence": entry.batch_enqueue_sequence if entry.batch_id else None,
                "childSequence": entry.child_sequence if entry.batch_id else None,
                **details,
            }
        )

    def submit(self, store: JobStore, job_ids: list[str]) -> None:
        global _SCHEDULER_SEQUENCE
        with self._lock:
            known = {entry.job_id for entry in self._pending} | set(self._active)
            for job_id in dict.fromkeys(job_ids):
                if job_id in known:
                    continue
                try:
                    record = store.require(job_id)
                except KeyError:
                    continue
                if record.status in TERMINAL_STATES or record.status == "paused":
                    continue
                with _SCHEDULER_SEQUENCE_LOCK:
                    _SCHEDULER_SEQUENCE += 1
                    sequence = _SCHEDULER_SEQUENCE
                source = dict(record.source_metadata or {})
                batch_id = str(source.get("batchId") or "").strip() or None
                batch_sequence = (
                    self._batch_enqueue_sequences.setdefault(batch_id, sequence)
                    if batch_id
                    else sequence
                )
                child_sequence = int(source.get("childSequence") or sequence)
                entry = _ScheduledJob(
                    store=store,
                    job_id=job_id,
                    enqueue_sequence=sequence,
                    batch_id=batch_id,
                    batch_enqueue_sequence=batch_sequence,
                    child_sequence=child_sequence,
                )
                self._pending.append(entry)
                self._record_event("queued", entry, capacityReason="job_slot")
                store.update_status(
                    job_id,
                    status="waiting_for_job_slot",
                    progress=0.0,
                    current_step="Waiting for an analysis worker slot",
                    diagnostic_updates={
                        "enqueueSequence": sequence,
                        "batchEnqueueSequence": batch_sequence if batch_id else None,
                        "childSequence": child_sequence if batch_id else None,
                        "queuePosition": len(self._pending),
                        "workerSlot": None,
                        "capacityReason": "job_slot",
                        "schedulerPolicy": str(getattr(SETTINGS, "scheduler_policy", "oldest_batch_first")),
                    },
                )
            self._refresh_queue_positions_locked()
            self._dispatch_locked()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            capacity = self._adaptive_capacity_locked()
            self._dispatch_locked(capacity=capacity)
            return {
                "policy": str(getattr(SETTINGS, "scheduler_policy", "oldest_batch_first")),
                "configuredJobConcurrency": analysis_concurrency_limit(),
                "configuredPerBatchConcurrency": per_batch_concurrency_limit(),
                "activeJobIds": list(self._active),
                "queuedJobIds": [entry.job_id for entry in self._pending],
                "recentEvents": list(self._events),
                "activeWorkers": [
                    {"workerId": f"analysis-{entry.worker_slot}", "workerSlot": entry.worker_slot, "jobId": entry.job_id}
                    for entry in sorted(self._active.values(), key=lambda item: item.worker_slot or 99)
                ],
                "adaptiveWorkers": self._adaptive.payload(),
            }

    def wait_for(self, store: JobStore, job_ids: list[str], *, timeout_seconds: float | None = None) -> bool:
        """Wait for submitted jobs without bypassing scheduler ownership.

        HTTP handlers use this only from Starlette background tasks.  The
        response has already been emitted to a live client, while TestClient
        retains its historical deterministic behaviour of awaiting background
        analysis completion before it tears down the temporary store.
        """
        tracked = set(job_ids)
        deadline = time.monotonic() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
        with self._lock:
            while True:
                states: list[str] = []
                for job_id in tracked:
                    try:
                        states.append(store.require(job_id).status)
                    except KeyError:
                        states.append("deleted")
                # A record becomes terminal just before its execution lock and
                # scheduler ownership are released.  TestClient callers that
                # immediately delete/recover a job need the full lifecycle
                # boundary, not merely the first terminal manifest write.
                active_tracked = any(job_id in self._active for job_id in tracked)
                if (
                    all(state in TERMINAL_STATES or state in {"paused", "deleted"} for state in states)
                    and not active_tracked
                ):
                    return True
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._lock.wait(min(remaining, 0.5))
                else:
                    self._lock.wait(0.5)

    def _active_in_batch(self, batch_id: str | None) -> int:
        if not batch_id:
            return 0
        return sum(1 for entry in self._active.values() if entry.batch_id == batch_id)

    def _refresh_queue_positions_locked(self) -> None:
        """Keep visible queue positions aligned with the dispatch order."""
        # Position represents deterministic queue priority, not the next
        # immediately dispatchable entry.  A child can be priority one while
        # temporarily held by its per-batch cap; a spare worker may still run
        # lower-priority work without changing the visible batch order.
        ordered = sorted(
            self._pending,
            key=lambda entry: (
                entry.batch_enqueue_sequence if entry.batch_id else entry.enqueue_sequence,
                entry.child_sequence if entry.batch_id else entry.enqueue_sequence,
                entry.enqueue_sequence,
            ),
        )
        for position, entry in enumerate(ordered, start=1):
            try:
                record = entry.store.require(entry.job_id)
            except KeyError:
                continue
            entry.store.update_status(
                entry.job_id,
                status=record.status,
                progress=record.progress,
                current_step=record.current_step,
                diagnostic_updates={"queuePosition": position, "workerSlot": None},
            )

    def _next_entry_locked(self, *, require_active_compatibility: bool = False) -> _ScheduledJob | None:
        if not self._pending:
            return None
        per_batch = per_batch_concurrency_limit()
        groups: Dict[str, list[_ScheduledJob]] = {}
        for entry in self._pending:
            group_key = entry.batch_id or f"standalone:{entry.enqueue_sequence}"
            groups.setdefault(group_key, []).append(entry)
        ordered_groups = sorted(
            groups.values(),
            key=lambda entries: (
                min(entry.batch_enqueue_sequence if entry.batch_id else entry.enqueue_sequence for entry in entries),
                min(entry.enqueue_sequence for entry in entries),
            ),
        )
        for entries in ordered_groups:
            entries.sort(key=lambda entry: (entry.child_sequence if entry.batch_id else entry.enqueue_sequence, entry.enqueue_sequence))
            candidate = entries[0]
            if candidate.batch_id and self._active_in_batch(candidate.batch_id) >= per_batch:
                continue
            if require_active_compatibility and self._active:
                try:
                    active_identity = self._entry_identity(next(iter(self._active.values())))
                    if self._entry_identity(candidate) != active_identity:
                        continue
                except KeyError:
                    continue
            return candidate
        return None

    def _dispatch_locked(self, *, capacity: int | None = None) -> None:
        capacity = capacity if capacity is not None else self._adaptive_capacity_locked()
        while len(self._active) < capacity:
            free_slots = [slot for slot in range(1, capacity + 1) if all(item.worker_slot != slot for item in self._active.values())]
            if not free_slots:
                break
            slot = free_slots[0]
            entry = self._next_entry_locked(require_active_compatibility=slot == 3)
            if entry is None:
                break
            self._pending.remove(entry)
            entry.worker_slot = slot
            self._active[entry.job_id] = entry
            self._record_event("dispatched", entry, workerSlot=slot, capacityReason=None)
            try:
                entry.store.update_status(
                    entry.job_id,
                    status="queued",
                    progress=0.0,
                    current_step="Scheduled for local analysis",
                    diagnostic_updates={
                        "queuePosition": None,
                        "workerSlot": slot,
                        "capacityReason": None,
                        "schedulerPolicy": str(getattr(SETTINGS, "scheduler_policy", "oldest_batch_first")),
                    },
                )
            except KeyError:
                self._active.pop(entry.job_id, None)
                self._record_event("skipped", entry, reason="record_deleted_before_dispatch")
                continue
            thread = threading.Thread(
                target=self._run_entry,
                args=(entry,),
                name=f"safetrace-job-{entry.job_id[-8:]}",
                daemon=True,
            )
            self._threads[entry.job_id] = thread
            thread.start()
        self._refresh_queue_positions_locked()

    def _run_entry(self, entry: _ScheduledJob) -> None:
        try:
            execute_analysis_job(entry.store, entry.job_id)
        finally:
            with self._lock:
                try:
                    final_record = entry.store.require(entry.job_id)
                    final_status = final_record.status
                    runtime_seconds = final_record.timing_payload().get("analysisRuntimeSeconds")
                    error_type = final_record.error_type
                except KeyError:
                    final_status = "deleted"
                    runtime_seconds = None
                    error_type = None
                self._adaptive.record_outcome(
                    status=final_status,
                    runtime_seconds=runtime_seconds,
                    error_type=error_type,
                )
                self._record_event("finished", entry, finalStatus=final_status)
                self._active.pop(entry.job_id, None)
                self._threads.pop(entry.job_id, None)
                self._dispatch_locked()
                self._lock.notify_all()


_LOCAL_ANALYSIS_SCHEDULER = LocalAnalysisScheduler()


def schedule_analysis_jobs(store: JobStore, job_ids: list[str]) -> None:
    """Submit jobs to the process-local fair scheduler without blocking HTTP work."""
    _LOCAL_ANALYSIS_SCHEDULER.submit(store, job_ids)


def wait_for_scheduled_jobs(store: JobStore, job_ids: list[str], *, timeout_seconds: float | None = None) -> bool:
    """Wait for a scheduler-owned group without creating a second worker pool."""
    return _LOCAL_ANALYSIS_SCHEDULER.wait_for(store, job_ids, timeout_seconds=timeout_seconds)


def scheduler_status_payload() -> Dict[str, Any]:
    return _LOCAL_ANALYSIS_SCHEDULER.snapshot()
