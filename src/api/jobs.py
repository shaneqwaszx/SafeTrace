"""Job registry, manifest persistence, and lazy SafeTrace execution."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from src.config import SETTINGS
from src.engine_metrics import build_engine_metrics

from .normalization import normalize_pipeline_results

JobState = Literal["queued", "running", "completed", "failed", "cancelled"]
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
ACTIVE_STATES = {"queued", "running"}
MANIFEST_FILENAME = "manifest.json"
RESULT_FILENAME = "result.json"
LOCK_FILENAME = "execution.lock"
ANALYSIS_HEARTBEAT_SECONDS = 8.0

_PIPELINE_SETTINGS_LOCK = threading.Lock()
logger = logging.getLogger("safetrace.api.jobs")
_MANIFEST_LOCKS: Dict[Path, threading.RLock] = {}
_MANIFEST_LOCKS_GUARD = threading.Lock()
_SEMAPHORE_LOCK = threading.Lock()
_ANALYSIS_SEMAPHORE: threading.BoundedSemaphore | None = None
_ANALYSIS_SEMAPHORE_LIMIT: int | None = None
_VLM_SEMAPHORE: threading.BoundedSemaphore | None = None
_VLM_SEMAPHORE_LIMIT: int | None = None


class UploadValidationError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class PipelineTimeoutError(TimeoutError):
    """Raised when an analysis job exceeds the configured backend timeout."""


class JobManifestPersistenceError(OSError):
    """Raised when a job manifest/result JSON file cannot be atomically replaced."""


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
        }

    def status_payload(self) -> Dict[str, Any]:
        updated_at = _to_iso(self.updated_at)
        timing = self.timing_payload()
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
            "persistenceWarning": self.persistence_warning,
            "manifestPersistenceWarning": self.persistence_warning,
            **timing,
            "updatedAt": updated_at,
            "heartbeatAt": updated_at if self.status in ACTIVE_STATES else None,
        }


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
    )


def normalize_vlm_profile(value: Any) -> str:
    raw = str(value or VLM_PROFILE_RULE_BASED).strip().lower()
    return raw if raw in VLM_PROFILE_IDS else VLM_PROFILE_RULE_BASED


def resolve_configured_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (SETTINGS.project_root / path).resolve()


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
    return _safe_concurrency(
        getattr(SETTINGS, "analysis_concurrency", getattr(SETTINGS, "worker_concurrency", 1)),
        default=1,
    )


def vlm_concurrency_limit() -> int:
    return _safe_concurrency(getattr(SETTINGS, "vlm_concurrency", 1), default=1)


def _analysis_semaphore() -> threading.BoundedSemaphore:
    global _ANALYSIS_SEMAPHORE, _ANALYSIS_SEMAPHORE_LIMIT
    limit = analysis_concurrency_limit()
    with _SEMAPHORE_LOCK:
        if _ANALYSIS_SEMAPHORE is None or _ANALYSIS_SEMAPHORE_LIMIT != limit:
            _ANALYSIS_SEMAPHORE = threading.BoundedSemaphore(limit)
            _ANALYSIS_SEMAPHORE_LIMIT = limit
        return _ANALYSIS_SEMAPHORE


def _vlm_semaphore() -> threading.BoundedSemaphore:
    global _VLM_SEMAPHORE, _VLM_SEMAPHORE_LIMIT
    limit = vlm_concurrency_limit()
    with _SEMAPHORE_LOCK:
        if _VLM_SEMAPHORE is None or _VLM_SEMAPHORE_LIMIT != limit:
            _VLM_SEMAPHORE = threading.BoundedSemaphore(limit)
            _VLM_SEMAPHORE_LIMIT = limit
        return _VLM_SEMAPHORE


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
    mobile_sam_worker_enabled = bool(
        mobile_sam_requested and getattr(SETTINGS, "mobile_sam_worker_enabled", False)
    )
    safe_mode = bool(settings.safe_mode or analysis_safe_mode())
    lightweight_vlm_worker_enabled = lightweight_vlm_worker_requested(settings)
    return {
        "safeMode": safe_mode,
        "device": "cpu" if safe_mode else settings.device,
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
    ) -> JobRecord:
        clean_name = validate_upload_filename(filename)
        validate_upload_size(len(content))

        job_id = new_job_id()
        job_dir = self.root_dir / job_id
        upload_dir = job_dir / "uploads"
        output_dir = job_dir / "media"
        upload_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=True)

        upload_path = upload_dir / clean_name
        upload_path.write_bytes(content)

        now = _utc_now()
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
            size_bytes=len(content),
            job_dir=job_dir,
            upload_path=upload_path,
            output_dir=output_dir,
            created_at=now,
            updated_at=now,
        )
        record.metrics["componentDiagnostics"] = _base_component_diagnostics(settings)
        self._refresh_metrics(record)
        with self._lock:
            self._jobs[job_id] = record
            self.persist_job(record)
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
            if status == "running" and record.started_at is None:
                record.started_at = now
            if status in TERMINAL_STATES and record.finished_at is None:
                record.finished_at = now
            record.status = status
            record.progress = progress
            record.current_step = current_step
            record.error = error
            record.technical_error = technical_error
            record.error_type = error_type
            record.updated_at = now
            _merge_component_diagnostics(
                record.metrics,
                record.settings,
                stage=_stage_for_status(status, current_step),
                updates={
                    **(diagnostic_updates or {}),
                    **({"errorType": error_type, "errorMessage": error} if error_type or error else {}),
                },
            )
            self._refresh_metrics(record)
            self.persist_job(record)

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
            return True

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

            normalized = dict(result)
            technical_details = dict(normalized.get("technicalDetails") or {})
            job_timing = record.timing_payload(now=now)
            normalized.update(job_timing)
            technical_details["jobMetrics"] = dict(record.metrics)
            technical_details["jobTiming"] = job_timing
            normalized["technicalDetails"] = technical_details
            engine_metrics = build_engine_metrics(normalized, record.metrics)
            technical_details["engineMetrics"] = engine_metrics
            normalized["engineMetrics"] = engine_metrics
            normalized["technicalDetails"] = technical_details
            record.result = normalized
            record.result_path = record.job_dir / RESULT_FILENAME
            self.persist_job(record)

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
            self._jobs.pop(job_id, None)
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

    def persist_job(self, record: JobRecord) -> None:
        if record.result is not None:
            if record.persistence_warning:
                record.result["persistenceWarning"] = record.persistence_warning
                technical_details = dict(record.result.get("technicalDetails") or {})
                technical_details["manifestPersistenceWarning"] = record.persistence_warning
                record.result["technicalDetails"] = technical_details
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
            },
            "output": {
                "mediaDir": _safe_relative_path(record.output_dir, record.job_dir),
                "resultPath": result_path,
                "mediaFiles": media_files,
            },
            "metrics": record.metrics,
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
        if status not in {"queued", "running", "completed", "failed", "cancelled"}:
            status = "failed"

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
            created_at=_parse_datetime(manifest.get("createdAt")) or _utc_now(),
            updated_at=_parse_datetime(manifest.get("updatedAt")) or _utc_now(),
            started_at=_parse_datetime(manifest.get("startedAt")),
            finished_at=_parse_datetime(manifest.get("finishedAt")),
        )
        self._refresh_metrics(record)
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

    def _is_safe_job_dir(self, job_dir: Path) -> bool:
        root = self.root_dir.resolve()
        target = job_dir.resolve()
        return target != root and root in target.parents

    def _delete_record_dir(self, record: JobRecord) -> None:
        if not self._is_safe_job_dir(record.job_dir):
            raise RuntimeError("Refusing to delete job directory outside API job root.")
        shutil.rmtree(record.job_dir, ignore_errors=True)

    def _is_stale_active(self, record: JobRecord) -> bool:
        if record.status not in ACTIVE_STATES:
            return False
        cutoff = _utc_now() - timedelta(minutes=float(SETTINGS.stale_running_minutes))
        return record.updated_at < cutoff

    def _mark_interrupted(self, record: JobRecord) -> None:
        lock_path = record.job_dir / LOCK_FILENAME
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        record.status = "failed"
        record.progress = 1.0
        record.current_step = "Analysis interrupted"
        record.error = "Job interrupted by backend restart."
        record.error_type = "InterruptedJob"
        record.finished_at = _utc_now()
        record.updated_at = record.finished_at
        _merge_component_diagnostics(
            record.metrics,
            record.settings,
            stage="failed",
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
    component_diagnostics: Optional[Dict[str, Any]] = None,
):
    """Run the existing SafeTrace pipeline lazily.

    The heavy pipeline import and construction happen only inside this function.
    """
    from src.config import SETTINGS
    from src.pipeline import SafeTracePipeline

    with _PIPELINE_SETTINGS_LOCK:
        normalized_profile = normalize_vlm_profile(vlm_profile)
        old_device = SETTINGS.device
        old_enable_vlm = SETTINGS.enable_vlm
        old_vlm_profile = getattr(SETTINGS, "vlm_profile", VLM_PROFILE_RULE_BASED)
        old_vlm_model_dir = SETTINGS.vlm_model_dir
        old_mobile_sam_enabled = getattr(SETTINGS, "mobile_sam_enabled", "disabled")
        old_analysis_safe_mode = getattr(SETTINGS, "analysis_safe_mode", False)
        old_safe_mode_allow_mobilesam = getattr(SETTINGS, "safe_mode_allow_mobilesam", False)
        effective_safe_mode = bool(safe_mode or analysis_safe_mode())
        SETTINGS.analysis_safe_mode = effective_safe_mode
        SETTINGS.device = "cpu" if effective_safe_mode else device
        lightweight_worker_effective = bool(
            effective_safe_mode
            and getattr(SETTINGS, "lightweight_vlm_worker_enabled", False)
            and not vlm_hard_disabled()
            and enable_vlm
            and normalized_profile in VLM_SAFE_MODE_WORKER_PROFILES
        )
        SETTINGS.enable_vlm = bool(
            (
                lightweight_worker_effective
                or (
                    not effective_safe_mode
                    and not vlm_hard_disabled()
                    and enable_vlm
                    and normalized_profile != VLM_PROFILE_RULE_BASED
                )
            )
        )
        SETTINGS.vlm_profile = normalized_profile
        if effective_safe_mode and not bool(getattr(SETTINGS, "safe_mode_allow_mobilesam", False)):
            SETTINGS.mobile_sam_enabled = "disabled"
        if SETTINGS.enable_vlm:
            resolved_model_dir = vlm_model_dir or resolve_vlm_profile_model_dir(normalized_profile)
            if resolved_model_dir is not None:
                SETTINGS.vlm_model_dir = resolved_model_dir
        try:
            pipeline = SafeTracePipeline(
                use_case_profile=dict(use_case_profile or {}),
                query_context=query,
            )
            try:
                return pipeline.run([upload_path], query=query, fps=fps, k=top_k)
            finally:
                if component_diagnostics is not None:
                    component_diagnostics.update(dict(getattr(pipeline, "component_diagnostics", {}) or {}))
        finally:
            SETTINGS.device = old_device
            SETTINGS.enable_vlm = old_enable_vlm
            SETTINGS.vlm_profile = old_vlm_profile
            SETTINGS.vlm_model_dir = old_vlm_model_dir
            SETTINGS.mobile_sam_enabled = old_mobile_sam_enabled
            SETTINGS.analysis_safe_mode = old_analysis_safe_mode
            SETTINGS.safe_mode_allow_mobilesam = old_safe_mode_allow_mobilesam


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
    if record.status in TERMINAL_STATES:
        return
    if not store.acquire_execution_lock(job_id):
        return

    try:
        with _analysis_semaphore():
            record = store.require(job_id)
            if record.status in TERMINAL_STATES:
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
            try:
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
                vlm_profile = normalize_vlm_profile(record.settings.vlm_profile)
                effective_vlm_enabled = should_enable_vlm(record.settings)
                vlm_slot = _vlm_semaphore() if effective_vlm_enabled and vlm_profile != VLM_PROFILE_RULE_BASED else nullcontext()
                component_diagnostics = _merge_component_diagnostics(
                    dict(record.metrics),
                    record.settings,
                    stage="pipeline_starting",
                    updates={
                        "analysisConcurrency": analysis_concurrency_limit(),
                        "vlmConcurrency": vlm_concurrency_limit(),
                        "vlmConcurrencyLimited": bool(effective_vlm_enabled and vlm_profile != VLM_PROFILE_RULE_BASED),
                    },
                )
                try:
                    with vlm_slot:
                        raw_result = _run_pipeline_with_timeout(
                            timeout_seconds=float(getattr(SETTINGS, "analysis_job_timeout_seconds", 600.0) or 0.0),
                            component_diagnostics=component_diagnostics,
                            upload_path=record.upload_path,
                            query=record.query,
                            fps=record.settings.fps,
                            top_k=record.settings.top_k,
                            device="cpu" if record.settings.safe_mode else record.settings.device,
                            enable_vlm=effective_vlm_enabled,
                            vlm_profile=vlm_profile,
                            vlm_model_dir=resolve_vlm_profile_model_dir(vlm_profile) if effective_vlm_enabled else None,
                            safe_mode=record.settings.safe_mode,
                            use_case_profile=dict(record.settings.use_case_profile or {}),
                        )
                finally:
                    heartbeat_stop.set()
                    heartbeat_thread.join(timeout=0.5)
                store.update_status(
                    job_id,
                    status="running",
                    progress=0.8,
                    current_step="SafeTrace pipeline completed; preparing report",
                    diagnostic_updates=component_diagnostics,
                )
                store.update_status(
                    job_id,
                    status="running",
                    progress=0.85,
                    current_step="Normalizing evidence report",
                    diagnostic_updates={"currentPipelineStage": "normalizing"},
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
                )
                store.register_media_files(job_id, pending_media_files)
                normalization_seconds = time.perf_counter() - normalization_started
                technical_details = result.setdefault("technicalDetails", {})
                technical_details["pipelineWallClockSeconds"] = time.perf_counter() - started
                technical_details["normalizationWallClockSeconds"] = normalization_seconds
                technical_details["reportGenerationWallClockSeconds"] = normalization_seconds
                technical_details["componentDiagnostics"] = component_diagnostics
                store.complete_job(job_id, result)
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
