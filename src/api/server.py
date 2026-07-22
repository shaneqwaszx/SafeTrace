"""FastAPI app for the local SafeTrace backend."""
from __future__ import annotations

import os
import shutil
import sys
import json
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from fastapi import BackgroundTasks, Body, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src import __version__
from src.chat_service import (
    ChatDisabledError,
    ChatProviderUnavailableError,
    answer_chat,
    chat_status_payload,
    warmup_chat_provider,
)
from src.config import SETTINGS
from src.device_gateway import device_gateway_payload
from src.vlm_reasoner import path_has_vlm_model_files, vlm_status_payload

from .batches import BatchStore, BatchValidationError
from .jobs import (
    AnalysisSettings,
    JobRestartConflictError,
    JobStore,
    owned_result_snapshot,
    UploadValidationError,
    schedule_analysis_jobs,
    scheduler_status_payload,
    wait_for_scheduled_jobs,
    max_upload_bytes,
    validate_upload_filename,
    validate_upload_size,
)
from .media import resolve_job_media_path
from .operations import (
    CleanupInProgressError,
    apply_cleanup,
    cleanup_preview,
    dashboard_summary,
    directory_size,
    normalized_recovery_policy,
    recovery_candidates,
    storage_summary,
)
from .result_ownership import ResultOwnershipError
from .retention_scheduler import RetentionScheduler
from .exports import create_export, delete_export, list_exports, verify_export
from .schemas import (
    AnalysisResultResponse,
    AnalyzeResponse,
    BatchResponse,
    ChatRequest,
    ChatResponse,
    ChatStatusResponse,
    DeviceMode,
    HealthResponse,
    JobStatusResponse,
    ModelStatus,
    SecondaryReviewRequest,
    SecondaryReviewResponse,
    SystemStatusResponse,
    VlmSettingsRequest,
    VlmSettingsResponse,
)


LOCAL_FRONTEND_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)
PNA_REQUEST_HEADER = "access-control-request-private-network"
PNA_RESPONSE_HEADER = "Access-Control-Allow-Private-Network"
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
VLM_PROFILE_LABELS = {
    VLM_PROFILE_RULE_BASED: "Fast Local Analysis",
    VLM_PROFILE_LIGHTWEIGHT: "Local VLM Assist",
    VLM_PROFILE_LIGHTWEIGHT_512M: "Local VLM Assist",
    VLM_PROFILE_ENHANCED: "Advanced GPU VLM Assist",
    VLM_PROFILE_ENHANCED_3B: "Advanced GPU VLM Assist",
}
VLM_PROFILE_METADATA = {
    VLM_PROFILE_RULE_BASED: {
        "resourceLevel": "lowest",
        "statusCopy": "Stable default. Always available and does not require a VLM model package.",
    },
    VLM_PROFILE_LIGHTWEIGHT: {
        "resourceLevel": "low",
        "fallback": True,
        "statusCopy": (
            "Low-resource 256M verifier fallback layer. Base findings run first; "
            "256M only refines wording/uncertainty when 512M is unavailable or unsuitable."
        ),
    },
    VLM_PROFILE_LIGHTWEIGHT_512M: {
        "resourceLevel": "medium",
        "candidate": True,
        "statusCopy": "512M lightweight verifier layer. Base findings run first; VLM only checks selected evidence frames.",
    },
    VLM_PROFILE_ENHANCED: {
        "resourceLevel": "gpu_high",
        "candidate": True,
        "requiresGpu": True,
        "statusCopy": (
            "Enhanced 2B GPU verifier/explanation layer. It runs after rule-based and lightweight "
            "verification and is unavailable without a PyTorch CUDA runtime."
        ),
    },
    VLM_PROFILE_ENHANCED_3B: {
        "resourceLevel": "very_high",
        "candidate": True,
        "statusCopy": "Enhanced 3B VLM candidate for selected/internal builds.",
    },
}


def _vlm_frame_limit() -> int:
    modern_limit = int(getattr(SETTINGS, "vlm_max_evidence_frames", 5) or 0)
    legacy_limit = int(getattr(SETTINGS, "vlm_max_frames", modern_limit) or 0)
    modern_limit = max(0, modern_limit)
    legacy_limit = max(0, legacy_limit)
    if legacy_limit != modern_limit:
        return min(legacy_limit, modern_limit)
    return modern_limit


def _split_origins(raw: str) -> tuple[str, ...]:
    return tuple(part.strip().rstrip("/") for part in raw.split(",") if part.strip())


def _normalize_origin(origin: str | None) -> str:
    return (origin or "").strip().rstrip("/")


def _cors_allowed_origins() -> list[str]:
    local_origins = LOCAL_FRONTEND_ORIGINS if getattr(SETTINGS, "include_local_cors_origins", True) else ()
    origins = [
        *local_origins,
        *getattr(SETTINGS, "allowed_origins", ()),
        *_split_origins(os.environ.get("SAFETRACE_ALLOWED_ORIGINS", "")),
    ]
    return list(dict.fromkeys(_normalize_origin(origin) for origin in origins if _normalize_origin(origin)))


def _is_cors_origin_allowed(origin: str | None) -> bool:
    normalized = _normalize_origin(origin)
    return bool(normalized and normalized in set(_cors_allowed_origins()))


def _is_private_network_preflight(request: Request) -> bool:
    return (
        request.method == "OPTIONS"
        and request.headers.get(PNA_REQUEST_HEADER, "").lower() == "true"
        and _is_cors_origin_allowed(request.headers.get("origin"))
    )


def _configure_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        max_age=600,
    )

    @app.middleware("http")
    async def private_network_access_header(request: Request, call_next):
        response = await call_next(request)
        if _is_private_network_preflight(request):
            response.headers[PNA_RESPONSE_HEADER] = "true"
        return response


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(SETTINGS.project_root))
    except ValueError:
        return str(path)


def _resolve_configured_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    root_candidate = (SETTINGS.project_root / path).resolve()
    if root_candidate.exists():
        return root_candidate
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return root_candidate


def _frontend_dist_path() -> Path:
    return _resolve_configured_path(Path(SETTINGS.frontend_dist))


def _frontend_status_payload() -> dict:
    dist = _frontend_dist_path()
    index = dist / "index.html"
    return {
        "serveFrontend": bool(SETTINGS.serve_frontend),
        "distPath": _display_path(dist),
        "distExists": dist.is_dir(),
        "indexExists": index.is_file(),
        "message": (
            "Frontend static serving is enabled."
            if SETTINGS.serve_frontend and index.is_file()
            else "Frontend static serving is disabled or the dist index is missing."
        ),
    }


def _path_has_contents(path: Path) -> bool:
    if path.is_file():
        return True
    if path.is_dir():
        try:
            return any(path.iterdir())
        except OSError:
            return False
    return False


def _path_has_model_contents(path: Path) -> bool:
    return path_has_vlm_model_files(path)


def _path_status(path: Path, *, optional: bool = False, unavailable_message: Optional[str] = None) -> ModelStatus:
    display = _display_path(path)
    if path.exists() and _path_has_contents(path):
        return ModelStatus(status="ready", path=display)
    if optional:
        return ModelStatus(
            status="unavailable",
            path=display,
            message=unavailable_message or "Optional model unavailable",
        )
    return ModelStatus(
        status="missing",
        path=display,
        message="Required model path is missing",
    )


def _optional_mode(value: str | None) -> str:
    raw = (value or "auto").strip().lower()
    if raw in {"0", "false", "no", "off", "disabled", "none"}:
        return "disabled"
    if raw in {"1", "true", "yes", "on", "enabled"}:
        return "enabled"
    return "auto"


def _vlm_enabled_mode() -> str:
    return _optional_mode(getattr(SETTINGS, "vlm_enabled", "auto"))


def _analysis_safe_mode() -> bool:
    return bool(getattr(SETTINGS, "analysis_safe_mode", False))


def _safe_mode_allow_mobile_sam() -> bool:
    return _analysis_safe_mode() and bool(getattr(SETTINGS, "safe_mode_allow_mobilesam", False))


def _mobile_sam_worker_enabled() -> bool:
    return bool(getattr(SETTINGS, "mobile_sam_worker_enabled", False))


def _lightweight_vlm_worker_enabled() -> bool:
    return bool(getattr(SETTINGS, "lightweight_vlm_worker_enabled", False))


def _safe_mode_lightweight_vlm_worker_allowed(profile: str | None = None, enabled: bool | None = None) -> bool:
    selected = (profile or getattr(SETTINGS, "vlm_profile", VLM_PROFILE_RULE_BASED) or VLM_PROFILE_RULE_BASED).strip().lower()
    active_requested = _vlm_enabled_mode() != "disabled" if enabled is None else bool(enabled)
    return bool(
        _analysis_safe_mode()
        and _lightweight_vlm_worker_enabled()
        and active_requested
        and selected in {*VLM_LIGHTWEIGHT_PROFILES, VLM_PROFILE_ENHANCED}
        and _vlm_enabled_mode() != "disabled"
    )


def _vlm_suppressed_reason(profile: str | None = None, enabled: bool | None = None) -> str | None:
    if _vlm_enabled_mode() == "disabled":
        return "hard_disabled"
    if _analysis_safe_mode() and not _safe_mode_lightweight_vlm_worker_allowed(profile, enabled):
        return "safe_mode"
    return None


def _vlm_hard_disabled(profile: str | None = None, enabled: bool | None = None) -> bool:
    return _vlm_suppressed_reason(profile, enabled) is not None


def _normalized_vlm_profile(value: str | None) -> str:
    raw = (value or VLM_PROFILE_RULE_BASED).strip().lower()
    return raw if raw in VLM_PROFILE_IDS else VLM_PROFILE_RULE_BASED


def _initial_vlm_enabled(profile: str) -> bool:
    if _vlm_hard_disabled(profile, True):
        return False
    if profile == VLM_PROFILE_RULE_BASED:
        return False
    if bool(getattr(SETTINGS, "enable_vlm", False)):
        return True
    return _vlm_enabled_mode() == "enabled"


def _profile_path(profile: str) -> Path | None:
    if profile == VLM_PROFILE_LIGHTWEIGHT:
        return _resolve_configured_path(Path(SETTINGS.vlm_lightweight_model_path))
    if profile == VLM_PROFILE_LIGHTWEIGHT_512M:
        return _resolve_configured_path(Path(SETTINGS.vlm_lightweight_512m_model_path))
    if profile == VLM_PROFILE_ENHANCED:
        return _resolve_configured_path(Path(SETTINGS.vlm_enhanced_model_path))
    if profile == VLM_PROFILE_ENHANCED_3B:
        return _resolve_configured_path(Path(SETTINGS.vlm_enhanced_3b_model_path))
    return None


def _vlm_profile_status(profile: str, *, runtime_available: bool) -> dict:
    metadata = VLM_PROFILE_METADATA.get(profile, {})
    gateway = device_gateway_payload(SETTINGS)
    enhanced_decision = gateway.get("components", {}).get("enhancedVlm", {})
    if profile == VLM_PROFILE_RULE_BASED:
        return {
            "id": VLM_PROFILE_RULE_BASED,
            "label": VLM_PROFILE_LABELS[VLM_PROFILE_RULE_BASED],
            "installed": True,
            "available": True,
            "requiresActivation": False,
            "resourceLevel": metadata.get("resourceLevel", "lowest"),
            "path": None,
            "message": metadata.get("statusCopy", "Fast Local Analysis is always available."),
            "statusCopy": metadata.get("statusCopy"),
        }

    path = _profile_path(profile)
    installed = bool(path and _path_has_model_contents(path))
    requires_gpu = bool(metadata.get("requiresGpu", False))
    gpu_ready = bool(enhanced_decision.get("available")) if requires_gpu else True
    missing_message = metadata.get("statusCopy") or "VLM profile assets are not installed."
    if installed and runtime_available and gpu_ready:
        message = metadata.get("statusCopy") or "VLM profile is installed and available."
    elif installed and runtime_available and requires_gpu and not gpu_ready:
        message = str(enhanced_decision.get("reason") or "GPU runtime is required for this VLM layer.")
    elif installed:
        message = "VLM profile is installed, but the transformers runtime is unavailable."
    else:
        message = missing_message
    return {
        "id": profile,
        "label": VLM_PROFILE_LABELS[profile],
        "installed": installed,
        "available": installed and runtime_available and gpu_ready,
        "requiresActivation": True,
        "resourceLevel": metadata.get("resourceLevel", "low" if profile in VLM_LIGHTWEIGHT_PROFILES else "high"),
        "path": _display_path(path) if path else None,
        "message": message,
        "statusCopy": metadata.get("statusCopy"),
        "deprecated": bool(metadata.get("deprecated", False)),
        "notViable": bool(metadata.get("notViable", False)),
        "candidate": bool(metadata.get("candidate", False)),
        "legacy": bool(metadata.get("legacy", False)),
        "fallback": bool(metadata.get("fallback", False)),
        "requiresGpu": requires_gpu,
        "deviceDecision": enhanced_decision if requires_gpu else gateway.get("components", {}).get("lightweightVlm"),
    }


def _vlm_settings_from_state(app: FastAPI | None = None) -> tuple[str, bool]:
    selected_profile = _normalized_vlm_profile(getattr(SETTINGS, "vlm_profile", VLM_PROFILE_RULE_BASED))
    enabled = _initial_vlm_enabled(selected_profile)
    if app is not None:
        selected_profile = _normalized_vlm_profile(getattr(app.state, "vlm_selected_profile", selected_profile))
        enabled = bool(getattr(app.state, "vlm_enabled", enabled)) and selected_profile != VLM_PROFILE_RULE_BASED
    if _vlm_hard_disabled(selected_profile, enabled):
        return selected_profile, False
    return selected_profile, enabled


def _vlm_profile_available(profile: str) -> bool:
    return bool(_vlm_profile_status(profile, runtime_available=_vlm_runtime_available()).get("available"))


def _vlm_profiles_payload(*, selected_profile: str, enabled: bool) -> dict:
    suppressed_reason = _vlm_suppressed_reason(selected_profile, enabled)
    lightweight_worker_allowed = _safe_mode_lightweight_vlm_worker_allowed(selected_profile, enabled)
    if suppressed_reason in {"safe_mode", "hard_disabled"}:
        mobile_sam_allowed = _safe_mode_allow_mobile_sam()
        runtime_available = False
        profiles = []
        gateway = device_gateway_payload(SETTINGS)
        for profile_id in VLM_PROFILE_LABELS:
            metadata = VLM_PROFILE_METADATA.get(profile_id, {})
            if profile_id == VLM_PROFILE_RULE_BASED:
                profiles.append(
                    {
                        "id": VLM_PROFILE_RULE_BASED,
                        "label": VLM_PROFILE_LABELS[VLM_PROFILE_RULE_BASED],
                        "installed": True,
                        "available": True,
                        "requiresActivation": False,
                        "resourceLevel": metadata.get("resourceLevel", "lowest"),
                        "path": None,
                        "message": "Fast Local Analysis is available in the local runtime guard.",
                        "statusCopy": metadata.get("statusCopy"),
                    }
                )
                continue
            profile_status = _vlm_profile_status(profile_id, runtime_available=runtime_available)
            profile_status["available"] = False
            if profile_status.get("installed") and profile_status.get("statusCopy"):
                profile_status["message"] = profile_status["statusCopy"]
            profiles.append(profile_status)
        return {
            "selectedProfile": selected_profile,
            "enabled": False,
            "active": False,
            "runtimeAvailable": False,
            "profiles": profiles,
            "message": (
                "Local visual review is disabled by configuration. Fast Local Analysis remains available."
                if suppressed_reason == "hard_disabled"
                else "Local runtime guard active. Fast Local Analysis remains available; "
                "MobileSAM may refine selected evidence frames."
                if mobile_sam_allowed
                else "Local runtime guard active. Fast Local Analysis remains available."
            ),
            "requestedVisualExplanationMode": selected_profile,
            "actualExplanationMode": (
                "rule_based_with_mobilesam"
                if suppressed_reason == "safe_mode" and mobile_sam_allowed
                else VLM_PROFILE_RULE_BASED
            ),
            "vlmAvailability": "disabled",
            "vlmSuppressedReason": suppressed_reason,
            "fallbackReason": (
                "VLM is disabled by SAFETRACE_VLM_ENABLED."
                if suppressed_reason == "hard_disabled"
                else "Local runtime guard suppresses VLM."
            ),
            "lightweightModelPathChecked": None,
            "ruleBasedFallbackActive": True,
            "ruleBasedFallbackAvailable": True,
            "safeModeMobileSamAllowed": mobile_sam_allowed,
            "lightweightVlmWorkerEnabled": False,
            "lightweightVlmWorkerTimeoutSeconds": float(
                getattr(SETTINGS, "lightweight_vlm_worker_timeout_seconds", 60.0) or 60.0
            ),
            "lightweightVlmExplanationSource": "disabled",
            "lightweightVlmEvidenceBudget": _vlm_frame_limit(),
            "lightweightVlmFrameLimit": _vlm_frame_limit(),
            "lightweightVlmJobTimeoutSeconds": float(getattr(SETTINGS, "vlm_job_timeout_seconds", 0.0) or 0.0),
            "lightweightVlmMaxQualityFailures": int(getattr(SETTINGS, "vlm_max_quality_failures", 1) or 0),
            "lightweightVlmPrimaryPolicy": str(getattr(SETTINGS, "lightweight_vlm_primary", "auto") or "auto"),
            "lightweightVlmFallbackPolicy": str(getattr(SETTINGS, "lightweight_vlm_fallback", "256m") or "256m"),
            "lightweightVlmCpuPrefer256m": bool(getattr(SETTINGS, "lightweight_vlm_cpu_prefer_256m", True)),
            "enhancedVlmRequiresGpu": True,
            "deviceGateway": gateway,
        }

    runtime_available = _vlm_runtime_available()
    profiles = [
        _vlm_profile_status(profile_id, runtime_available=runtime_available)
        for profile_id in VLM_PROFILE_LABELS
    ]
    profile_by_id = {profile["id"]: profile for profile in profiles}
    selected = profile_by_id.get(selected_profile, profile_by_id[VLM_PROFILE_RULE_BASED])
    hard_disabled = _vlm_hard_disabled(selected_profile, enabled)
    active = bool(not hard_disabled and selected_profile != VLM_PROFILE_RULE_BASED and enabled and selected["available"])
    lightweight_path = _profile_path(selected_profile) if selected_profile in VLM_LIGHTWEIGHT_PROFILES else _profile_path(VLM_PROFILE_LIGHTWEIGHT_512M)
    actual_mode = selected_profile if active else VLM_PROFILE_RULE_BASED
    vlm_availability = (
        "disabled"
        if hard_disabled
        else "available"
        if bool(selected.get("available"))
        else "missing_runtime"
        if bool(selected.get("installed")) and not runtime_available
        else "missing_assets"
        if selected_profile != VLM_PROFILE_RULE_BASED
        else "rule_based"
    )
    fallback_reason = None
    if hard_disabled:
        message = "Local visual review is disabled by configuration. Fast Local Analysis remains available."
        fallback_reason = "VLM is disabled by SAFETRACE_VLM_ENABLED."
    elif selected_profile == VLM_PROFILE_RULE_BASED:
        message = "Fast Local Analysis is active."
        fallback_reason = "Fast Local Analysis is selected."
    elif enabled and not selected["available"]:
        message = f"{selected['label']} is unavailable. Fast Local Analysis remains available."
        fallback_reason = selected.get("message") or f"{selected['label']} is unavailable."
    elif active and lightweight_worker_allowed:
        message = (
            "Local VLM Assist is selected for evidence explanations. "
            "Fast Local Analysis remains available if local visual review fails or times out."
        )
    elif active:
        message = (
            f"{selected['label']} selected for the next analysis. Evidence cards show local visual review "
            "only when it adds reliable evidence."
        )
    else:
        message = f"{selected['label']} available but inactive." if selected["available"] else (
            f"{selected['label']} is unavailable/not installed. Fast Local Analysis remains available."
        )
        fallback_reason = "VLM activation is off." if selected["available"] else selected.get("message")
    return {
        "selectedProfile": selected_profile,
        "enabled": bool(not hard_disabled and enabled and selected_profile != VLM_PROFILE_RULE_BASED),
        "active": active,
        "runtimeAvailable": runtime_available,
        "profiles": profiles,
        "message": message,
        "requestedVisualExplanationMode": selected_profile,
        "actualExplanationMode": actual_mode,
        "vlmAvailability": vlm_availability,
        "vlmSuppressedReason": suppressed_reason,
        "fallbackReason": fallback_reason,
        "lightweightModelPathChecked": _display_path(lightweight_path) if lightweight_path else None,
        "ruleBasedFallbackActive": bool(lightweight_worker_allowed or not active),
        "ruleBasedFallbackAvailable": True,
        "lightweightVlmWorkerEnabled": bool(lightweight_worker_allowed),
        "lightweightVlmWorkerTimeoutSeconds": float(
            getattr(SETTINGS, "lightweight_vlm_worker_timeout_seconds", 60.0) or 60.0
        ),
        "lightweightVlmExplanationSource": "worker" if lightweight_worker_allowed and active else "rule_based",
        "lightweightVlmEvidenceBudget": _vlm_frame_limit(),
        "lightweightVlmFrameLimit": _vlm_frame_limit(),
        "lightweightVlmJobTimeoutSeconds": float(getattr(SETTINGS, "vlm_job_timeout_seconds", 0.0) or 0.0),
        "lightweightVlmMaxQualityFailures": int(getattr(SETTINGS, "vlm_max_quality_failures", 1) or 0),
        "lightweightVlmPrimaryPolicy": str(getattr(SETTINGS, "lightweight_vlm_primary", "auto") or "auto"),
        "lightweightVlmFallbackPolicy": str(getattr(SETTINGS, "lightweight_vlm_fallback", "256m") or "256m"),
        "lightweightVlmCpuPrefer256m": bool(getattr(SETTINGS, "lightweight_vlm_cpu_prefer_256m", True)),
        "enhancedVlmRequiresGpu": True,
        "deviceGateway": device_gateway_payload(SETTINGS),
    }


def _current_vlm_payload(app: FastAPI | None = None) -> dict:
    selected_profile, enabled = _vlm_settings_from_state(app)
    return _vlm_profiles_payload(selected_profile=selected_profile, enabled=enabled)


def _vlm_model_status_for_payload(vlm_payload: dict) -> ModelStatus:
    selected_profile = _normalized_vlm_profile(str(vlm_payload.get("selectedProfile") or VLM_PROFILE_RULE_BASED))
    profiles = {str(profile.get("id")): profile for profile in list(vlm_payload.get("profiles") or [])}
    selected = profiles.get(selected_profile) or {}
    label = str(selected.get("label") or VLM_PROFILE_LABELS[selected_profile])
    path = _profile_path(selected_profile)
    details = {
        "selectedProfile": selected_profile,
        "enabled": bool(vlm_payload.get("enabled")),
        "active": bool(vlm_payload.get("active")),
        "runtimeAvailable": bool(vlm_payload.get("runtimeAvailable")),
        "provider": "local",
        "selectedProvider": "local" if vlm_payload.get("active") else VLM_PROFILE_RULE_BASED,
        "requestedVisualExplanationMode": vlm_payload.get("requestedVisualExplanationMode"),
        "actualExplanationMode": vlm_payload.get("actualExplanationMode"),
        "vlmAvailability": vlm_payload.get("vlmAvailability"),
        "fallbackReason": vlm_payload.get("fallbackReason"),
        "lightweightModelPathChecked": vlm_payload.get("lightweightModelPathChecked"),
        "ruleBasedFallbackActive": vlm_payload.get("ruleBasedFallbackActive"),
        "ruleBasedFallbackAvailable": vlm_payload.get("ruleBasedFallbackAvailable"),
        "lightweightVlmWorkerEnabled": vlm_payload.get("lightweightVlmWorkerEnabled"),
        "lightweightVlmWorkerTimeoutSeconds": vlm_payload.get("lightweightVlmWorkerTimeoutSeconds"),
        "lightweightVlmExplanationSource": vlm_payload.get("lightweightVlmExplanationSource"),
        "lightweightVlmEvidenceBudget": vlm_payload.get("lightweightVlmEvidenceBudget"),
        "lightweightVlmFrameLimit": vlm_payload.get("lightweightVlmFrameLimit"),
        "lightweightVlmJobTimeoutSeconds": vlm_payload.get("lightweightVlmJobTimeoutSeconds"),
        "lightweightVlmMaxQualityFailures": vlm_payload.get("lightweightVlmMaxQualityFailures"),
        "lightweightVlmPrimaryPolicy": vlm_payload.get("lightweightVlmPrimaryPolicy"),
        "lightweightVlmFallbackPolicy": vlm_payload.get("lightweightVlmFallbackPolicy"),
        "lightweightVlmCpuPrefer256m": vlm_payload.get("lightweightVlmCpuPrefer256m"),
        "enhancedVlmRequiresGpu": vlm_payload.get("enhancedVlmRequiresGpu"),
        "deviceGateway": vlm_payload.get("deviceGateway"),
    }
    if _vlm_hard_disabled(selected_profile, bool(vlm_payload.get("enabled"))):
        payload = vlm_status_payload()
        payload["details"] = {**details, **dict(payload.get("details") or {})}
        return ModelStatus(**payload)

    if selected_profile == VLM_PROFILE_RULE_BASED:
        payload = vlm_status_payload()
        payload["details"] = {**details, **dict(payload.get("details") or {})}
        return ModelStatus(**payload)

    if bool(vlm_payload.get("active")):
        return ModelStatus(
            status="available",
            path=_display_path(path) if path else None,
            message=f"{label} is selected and available.",
            details=details,
        )
    if bool(selected.get("installed")) and not bool(selected.get("available")):
        return ModelStatus(
            status="missing_runtime",
            path=_display_path(path) if path else None,
            message=f"{label} assets are installed, but the transformers runtime is unavailable.",
            actionHint="Install the local VLM runtime or use rule-based explanations.",
            details=details,
        )
    return ModelStatus(
        status="unavailable",
        path=_display_path(path) if path else None,
        message=f"{label} is not active. Fast Local Analysis remains available.",
        actionHint="Install the selected local VLM assets and activate VLM only when needed.",
        details=details,
    )


def _analysis_settings_from_request(
    app: FastAPI,
    *,
    fps: float,
    top_k: int,
    enable_vlm: bool,
    device: str,
    vlm_profile: Optional[str],
    vlm_enabled: Optional[bool],
    use_case_profile: Optional[dict[str, Any]] = None,
    review_mode: str = "fast_local",
) -> AnalysisSettings:
    selected_profile, configured_enabled = _vlm_settings_from_state(app)
    requested_profile = _normalized_vlm_profile(vlm_profile or selected_profile)
    safe_mode = _analysis_safe_mode()
    requested_activation = configured_enabled if vlm_enabled is None else bool(vlm_enabled)
    worker_allowed = _safe_mode_lightweight_vlm_worker_allowed(requested_profile, requested_activation)
    hard_disabled = _vlm_hard_disabled(requested_profile, requested_activation)
    requested_available = (
        requested_profile == VLM_PROFILE_RULE_BASED
        or (not hard_disabled and _vlm_profile_available(requested_profile))
    )
    requested_activation = bool(
        requested_available
        and (not safe_mode or worker_allowed)
        and not hard_disabled
        and requested_activation
        and requested_profile != VLM_PROFILE_RULE_BASED
    )
    effective_vlm_enabled = bool(
        (not safe_mode or worker_allowed)
        and not hard_disabled
        and enable_vlm
        and requested_activation
    )
    return AnalysisSettings(
        fps=fps,
        top_k=top_k,
        enable_vlm=effective_vlm_enabled,
        device=device,
        vlm_profile=requested_profile,
        vlm_enabled=requested_activation,
        safe_mode=safe_mode,
        use_case_profile=dict(use_case_profile or {}),
        review_mode="comprehensive" if review_mode == "comprehensive" else "fast_local",
    )


def _limited_string(value: Any, limit: int = 500) -> str:
    return str(value or "").strip()[:limit]


def _limited_string_list(value: Any, *, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_limited_string(item, 160) for item in value[:limit] if _limited_string(item, 160)]


def _parse_use_case_profile(raw: Optional[str]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    profile_id = _limited_string(parsed.get("profileId"), 80)
    label = _limited_string(parsed.get("label"), 120)
    if not profile_id or not label:
        return {}
    return {
        "profileId": profile_id,
        "label": label,
        "category": _limited_string(parsed.get("category"), 120),
        "description": _limited_string(parsed.get("description"), 500),
        "defaultQuery": _limited_string(parsed.get("defaultQuery"), 300),
        "backendSupportLevel": _limited_string(parsed.get("backendSupportLevel"), 80),
        "supportedChecks": _limited_string_list(parsed.get("supportedChecks")),
        "unsupportedChecks": _limited_string_list(parsed.get("unsupportedChecks")),
        "limitations": _limited_string(parsed.get("limitations"), 500),
        "checks": _limited_string_list(parsed.get("checks")),
        "rules": _limited_string_list(parsed.get("rules")),
        "notes": _limited_string(parsed.get("notes"), 500),
        "customText": _limited_string(parsed.get("customText"), 500),
        "requestedQuery": _limited_string(parsed.get("requestedQuery"), 300),
        "effectiveQuery": _limited_string(parsed.get("effectiveQuery"), 300),
    }


def _mobile_sam_runtime_available() -> bool:
    return importlib_util.find_spec("mobile_sam") is not None


def _vlm_runtime_available() -> bool:
    return importlib_util.find_spec("transformers") is not None


def _mobile_sam_status() -> ModelStatus:
    checkpoint = SETTINGS.mobile_sam_checkpoint
    display = _display_path(checkpoint)
    gateway = device_gateway_payload(SETTINGS)
    mobile_sam_device = gateway.get("components", {}).get("mobileSam", {})
    safe_mode = _analysis_safe_mode()
    safe_mode_mobile_sam_allowed = _safe_mode_allow_mobile_sam()
    mode = (
        _optional_mode(getattr(SETTINGS, "mobile_sam_enabled", "auto"))
        if not safe_mode or safe_mode_mobile_sam_allowed
        else "disabled"
    )
    checkpoint_exists = checkpoint.is_file()
    runtime_available = _mobile_sam_runtime_available()
    details = {
        "enabledMode": mode,
        "checkpointExists": checkpoint_exists,
        "runtimeAvailable": runtime_available,
        "packagedExpectedPath": "checkpoints/mobile_sam.pt",
        "safeMode": safe_mode,
        "safeModeMobileSamAllowed": safe_mode_mobile_sam_allowed,
        "mobileSamEnabled": mode != "disabled",
        "mobileSamWorkerEnabled": bool(mode != "disabled" and _mobile_sam_worker_enabled()),
        "mobileSamWorkerTimeoutSeconds": float(getattr(SETTINGS, "mobile_sam_worker_timeout_seconds", 60.0) or 60.0),
        "mobileSamDevice": mobile_sam_device.get("selected"),
        "mobileSamDeviceReason": mobile_sam_device.get("reason"),
        "deviceGatewayDecision": mobile_sam_device,
        "mobileSamRefinementSource": (
            "worker"
            if mode != "disabled" and _mobile_sam_worker_enabled()
            else "disabled"
            if mode == "disabled"
            else "fallback"
        ),
        "ruleBasedFallbackActive": True,
    }
    if mode == "disabled":
        return ModelStatus(
            status="disabled",
            path=display,
            message=(
                "Local runtime guard active. MobileSAM refinement is disabled."
                if _analysis_safe_mode()
                else "MobileSAM refinement is disabled. Detector-box evidence remains available."
            ),
            actionHint="Set SAFETRACE_MOBILESAM_ENABLED=auto to allow optional refinement.",
            details=details,
        )
    if not checkpoint_exists:
        return ModelStatus(
            status="missing_checkpoint",
            path=display,
            message=(
                "MobileSAM checkpoint is missing. SafeTrace will use detector-box evidence without refined masks."
            ),
            actionHint="Place the optional checkpoint at checkpoints/mobile_sam.pt for refined segmentation masks.",
            details=details,
        )
    if not runtime_available:
        return ModelStatus(
            status="missing_runtime",
            path=display,
            message="MobileSAM checkpoint exists, but the mobile-sam Python runtime is unavailable.",
            actionHint="Install the MobileSAM runtime in the local environment, or keep using detector-box fallback.",
            details=details,
        )
    return ModelStatus(
        status="available",
        path=display,
        message=(
            "MobileSAM worker refinement enabled for selected evidence frames. "
            "Detector-box fallback used if the worker fails."
            if safe_mode_mobile_sam_allowed and _mobile_sam_worker_enabled()
            else "MobileSAM refinement is available for selected evidence frames. "
            "Detector-box fallback remains active."
            if safe_mode_mobile_sam_allowed
            else "MobileSAM refinement is available as an optional detector-box mask refinement."
        ),
        details=details,
    )


def _gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _detector_status() -> ModelStatus:
    if SETTINGS.yolo_checkpoint.exists():
        return ModelStatus(status="ready", path=_display_path(SETTINGS.yolo_checkpoint))
    if SETTINGS.yolo_fallback_checkpoint.exists():
        return ModelStatus(
            status="ready",
            path=_display_path(SETTINGS.yolo_fallback_checkpoint),
            message="Using fallback detector checkpoint",
        )
    return ModelStatus(
        status="missing",
        path=_display_path(SETTINGS.yolo_checkpoint),
        message="No YOLO checkpoint found at primary or fallback path",
    )


def _model_status_payload(status: ModelStatus) -> dict:
    payload = {
        "status": status.status,
        "path": status.path,
        "message": status.message,
    }
    if status.actionHint:
        payload["actionHint"] = status.actionHint
    if status.details:
        payload["details"] = status.details
    return payload


def _runtime_check(
    status: str,
    message: str,
    *,
    path: Optional[str] = None,
    action_hint: Optional[str] = None,
    details: Optional[dict] = None,
) -> dict:
    payload = {
        "status": status,
        "message": message,
        "path": path,
        "actionHint": action_hint,
    }
    if details:
        payload["details"] = details
    return payload


def _model_preflight_check(label: str, status: ModelStatus, *, optional: bool = False) -> dict:
    if status.status in {"ready", "available"}:
        return _runtime_check(
            status.status,
            status.message or f"{label} ready",
            path=status.path,
            action_hint=status.actionHint,
            details=status.details,
        )
    if status.status == "disabled":
        return _runtime_check(
            "disabled",
            status.message or f"{label} disabled",
            path=status.path,
            action_hint=status.actionHint,
            details=status.details,
        )
    if status.status in {"missing_checkpoint", "missing_runtime"}:
        return _runtime_check(
            status.status,
            status.message or f"{label} {status.status.replace('_', ' ')}",
            path=status.path,
            action_hint=status.actionHint,
            details=status.details,
        )
    if status.status == "missing":
        return _runtime_check(
            "missing",
            status.message or f"{label} missing",
            path=status.path,
            action_hint=f"Place the required {label} asset at the configured path.",
        )
    action_hint = None if optional else f"Check the configured {label} path."
    if optional and label == "MobileSAM":
        action_hint = "Install checkpoints/mobile_sam.pt only if refined segmentation masks are needed."
    if optional and label == "VLM":
        action_hint = "Use SAFETRACE_VLM_PROVIDER=auto to prefer the local VLM provider, or explicitly choose ollama."
    return _runtime_check(
        "unavailable",
        status.message or f"{label} unavailable",
        path=status.path,
        action_hint=status.actionHint or action_hint,
        details=status.details,
    )


def _visual_explanations_payload(vlm_status: ModelStatus) -> dict:
    details = dict(vlm_status.details or {})
    enhanced_available = vlm_status.status in {"ready", "available"}
    actual_mode = str(details.get("actualExplanationMode") or "").strip() or (
        "vlm" if enhanced_available else "rule_based"
    )
    selected_provider = str(details.get("selectedProvider") or "").strip().lower()
    provider_vlm_available = selected_provider not in {"", "rule_based", VLM_PROFILE_RULE_BASED}
    source = (
        "vlm"
        if enhanced_available
        and (
            actual_mode not in {"", VLM_PROFILE_RULE_BASED, "rule_based"}
            or provider_vlm_available
        )
        else "rule_based"
    )
    fallback_reason = details.get("fallbackReason")
    enabled = source == "vlm"
    return {
        # Rule-based explanations are always a valid local provider.  The
        # source and fallback fields distinguish that from a VLM contribution
        # without making the status endpoint look unhealthy.
        "status": "available",
        "fallback": "rule_based",
        "explanationSource": source,
        "enhancedVlmAvailable": enhanced_available,
        "message": (
            "Visual explanations are enabled. Evidence cards show local visual review only when "
            "a local VLM adds reliable evidence; Fast Local Analysis remains available."
            if enabled
            else "No executable visual-review provider is selected. Fast Local Analysis uses rule-based explanations."
        ),
        "requestedVisualExplanationMode": details.get("requestedVisualExplanationMode"),
        "actualExplanationMode": actual_mode,
        "fallbackReason": fallback_reason,
        "ruleBasedFallbackActive": bool(details.get("ruleBasedFallbackActive", source == "rule_based")),
        "lightweightModelPathChecked": details.get("lightweightModelPathChecked"),
        "lightweightVlmWorkerEnabled": details.get("lightweightVlmWorkerEnabled"),
        "lightweightVlmWorkerTimeoutSeconds": details.get("lightweightVlmWorkerTimeoutSeconds"),
        "lightweightVlmExplanationSource": details.get("lightweightVlmExplanationSource"),
        "lightweightVlmEvidenceBudget": details.get("lightweightVlmEvidenceBudget"),
        "lightweightVlmFrameLimit": details.get("lightweightVlmFrameLimit"),
        "lightweightVlmJobTimeoutSeconds": details.get("lightweightVlmJobTimeoutSeconds"),
        "lightweightVlmMaxQualityFailures": details.get("lightweightVlmMaxQualityFailures"),
    }


def _visual_explanations_preflight_check(vlm_status: ModelStatus) -> dict:
    payload = _visual_explanations_payload(vlm_status)
    return _runtime_check(
        # A rule-based explanation provider is a usable provider, even when
        # an optional VLM is unavailable.  Retain the established preflight
        # status contract while ``details.explanationSource`` records which
        # provider actually supplied the explanation.
        "available",
        payload["message"],
        details={
            "fallback": payload["fallback"],
            "explanationSource": payload["explanationSource"],
            "enhancedVlmAvailable": payload["enhancedVlmAvailable"],
            "enhancedVlmStatus": vlm_status.status,
            "requestedVisualExplanationMode": payload.get("requestedVisualExplanationMode"),
            "actualExplanationMode": payload.get("actualExplanationMode"),
            "fallbackReason": payload.get("fallbackReason"),
            "ruleBasedFallbackActive": payload.get("ruleBasedFallbackActive"),
            "lightweightModelPathChecked": payload.get("lightweightModelPathChecked"),
            "lightweightVlmWorkerEnabled": payload.get("lightweightVlmWorkerEnabled"),
            "lightweightVlmWorkerTimeoutSeconds": payload.get("lightweightVlmWorkerTimeoutSeconds"),
            "lightweightVlmExplanationSource": payload.get("lightweightVlmExplanationSource"),
            "lightweightVlmEvidenceBudget": payload.get("lightweightVlmEvidenceBudget"),
            "lightweightVlmFrameLimit": payload.get("lightweightVlmFrameLimit"),
            "lightweightVlmJobTimeoutSeconds": payload.get("lightweightVlmJobTimeoutSeconds"),
            "lightweightVlmMaxQualityFailures": payload.get("lightweightVlmMaxQualityFailures"),
        },
    )


def _openmp_status() -> dict:
    kmp_value = os.environ.get("KMP_DUPLICATE_LIB_OK")
    omp_value = os.environ.get("OMP_NUM_THREADS")
    kmp_enabled = str(kmp_value or "").strip().upper() == "TRUE"
    status = "ready" if kmp_enabled else "warning"
    return {
        "status": status,
        "kmpDuplicateLibOk": kmp_enabled,
        "rawKmpDuplicateLibOk": kmp_value,
        "ompNumThreads": omp_value,
        "message": (
            "OpenMP duplicate runtime workaround is enabled."
            if kmp_enabled
            else "OpenMP duplicate runtime workaround is not set for this process."
        ),
        "actionHint": None if kmp_enabled else "Set KMP_DUPLICATE_LIB_OK=TRUE before launching on Windows.",
    }


def _full_local_preflight_report() -> dict | None:
    """Return a prior strict launcher probe without pretending it is a live model object."""
    configured = os.environ.get("SAFETRACE_FULL_LOCAL_PREFLIGHT_REPORT")
    path = Path(configured) if configured else SETTINGS.data_dir / "runtime_preflight_local_full.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "path": _display_path(path),
        "profile": payload.get("profile"),
        "strict": bool(payload.get("strict")),
        "passed": bool(payload.get("passed")),
        "generatedAt": payload.get("generatedAt"),
        "pythonExecutable": payload.get("pythonExecutable"),
        "checks": payload.get("checks") if isinstance(payload.get("checks"), list) else [],
    }


def _assistant_preflight_check(chat: dict) -> dict:
    state = str(chat.get("state") or "unavailable")
    message = str(chat.get("message") or chat.get("reason") or "SafeTrace Assistant status unknown.")
    return _runtime_check(
        state,
        message,
        action_hint=chat.get("action_hint"),
        details={
            "provider": chat.get("provider"),
            "runtimeDiagnostics": chat.get("runtime_diagnostics"),
            "pythonExecutable": chat.get("python_executable"),
            "expectedVenvPython": chat.get("expected_venv_python"),
            "runningInExpectedVenv": chat.get("running_in_expected_venv"),
            "llamaCppImportStatus": chat.get("llama_cpp_import_status"),
            "llamaCppImportErrorType": chat.get("llama_cpp_import_error_type"),
            "llamaCppImportErrorMessage": chat.get("llama_cpp_import_error_message"),
            "setupCommand": chat.get("setup_command"),
            "restartRequired": chat.get("restart_required"),
        },
    )


def _assistant_model_check(chat: dict) -> dict:
    provider = str(chat.get("provider") or "unknown")
    model_path = chat.get("model_path")
    model_exists = chat.get("model_exists")
    if provider != "packaged_llamacpp":
        return _runtime_check(
            "unavailable",
            f"Assistant model file check is not applicable for provider {provider}.",
            path=model_path,
        )
    if model_exists is True:
        return _runtime_check("ready", "Assistant model found", path=model_path)
    if model_exists is False:
        return _runtime_check(
            "missing",
            "Assistant model file is missing.",
            path=model_path,
            action_hint=chat.get("action_hint"),
        )
    return _runtime_check("unavailable", "Assistant model file status unknown.", path=model_path)


def _assistant_runtime_check(chat: dict) -> dict:
    provider = str(chat.get("provider") or "unknown")
    runtime_available = chat.get("runtime_available")
    if provider != "packaged_llamacpp":
        return _runtime_check(
            "unavailable",
            f"llama-cpp runtime check is not applicable for provider {provider}.",
        )
    if runtime_available is True:
        return _runtime_check(
            "ready",
            "Assistant runtime installed",
            details={
                "runtimeDiagnostics": chat.get("runtime_diagnostics"),
                "pythonExecutable": chat.get("python_executable"),
                "llamaCppImportStatus": chat.get("llama_cpp_import_status"),
            },
        )
    if runtime_available is False:
        return _runtime_check(
            "missing",
            "Assistant runtime is missing.",
            action_hint=chat.get("action_hint"),
            details={
                "runtimeDiagnostics": chat.get("runtime_diagnostics"),
                "pythonExecutable": chat.get("python_executable"),
                "expectedVenvPython": chat.get("expected_venv_python"),
                "runningInExpectedVenv": chat.get("running_in_expected_venv"),
                "llamaCppImportStatus": chat.get("llama_cpp_import_status"),
                "llamaCppSpecFound": chat.get("llama_cpp_spec_found"),
                "llamaCppImportErrorType": chat.get("llama_cpp_import_error_type"),
                "llamaCppImportErrorMessage": chat.get("llama_cpp_import_error_message"),
                "setupCommand": chat.get("setup_command"),
                "restartRequired": chat.get("restart_required"),
            },
        )
    return _runtime_check("unavailable", "Assistant runtime status unknown.")


def _preflight_payload(*, models: dict[str, ModelStatus], chat: dict, openmp: dict) -> dict:
    checks = {
        "backend": _runtime_check("ready", "SafeTrace backend is responding."),
        "openmp": _runtime_check(
            openmp["status"],
            openmp["message"],
            action_hint=openmp.get("actionHint"),
            details={
                "kmpDuplicateLibOk": openmp.get("kmpDuplicateLibOk"),
                "ompNumThreads": openmp.get("ompNumThreads"),
            },
        ),
        "embeddingModel": _model_preflight_check("embedding model", models["embeddingModel"]),
        "detector": _model_preflight_check("detector", models["detector"]),
        "visualExplanations": _visual_explanations_preflight_check(models["vlm"]),
        "mobileSam": _model_preflight_check("MobileSAM", models["mobileSam"], optional=True),
        "vlm": _model_preflight_check("VLM", models["vlm"], optional=True),
        "assistant": _assistant_preflight_check(chat),
        "assistantModel": _assistant_model_check(chat),
        "assistantRuntime": _assistant_runtime_check(chat),
    }
    ready = sum(1 for check in checks.values() if check["status"] in {"ready", "available"})
    warnings = len(checks) - ready
    return {"checks": checks, "summary": {"ready": ready, "warnings": warnings}}


def _runtime_payload(
    *,
    store: "JobStore",
    models: dict[str, ModelStatus],
    gpu_available: bool,
    device_gateway: dict,
    chat: dict,
    openmp: dict,
) -> dict:
    return {
        "backend": {
            "status": "ready",
            "api": "safetrace-local",
            "version": "dev",
            "offline": SETTINGS.offline,
            "runtimeProfile": getattr(SETTINGS, "runtime_profile", "portable_fast_local"),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "workingDirectory": str(Path.cwd()),
        "device": {
            "configured": SETTINGS.device,
            "gpuAvailable": gpu_available,
            "gateway": device_gateway,
            "detector": device_gateway.get("components", {}).get("detector"),
            "mobileSam": device_gateway.get("components", {}).get("mobileSam"),
            "lightweightVlm": device_gateway.get("components", {}).get("lightweightVlm"),
            "enhancedVlm": device_gateway.get("components", {}).get("enhancedVlm"),
        },
        "analysis": {
            "safeMode": _analysis_safe_mode(),
            "safeModeMobileSamAllowed": _safe_mode_allow_mobile_sam(),
            "mobileSamWorkerEnabled": bool(_safe_mode_allow_mobile_sam() and _mobile_sam_worker_enabled()),
            "mobileSamWorkerTimeoutSeconds": float(getattr(SETTINGS, "mobile_sam_worker_timeout_seconds", 60.0) or 60.0),
            "lightweightVlmWorkerEnabled": bool(_safe_mode_lightweight_vlm_worker_allowed()),
            "lightweightVlmWorkerTimeoutSeconds": float(
                getattr(SETTINGS, "lightweight_vlm_worker_timeout_seconds", 60.0) or 60.0
            ),
            "lightweightVlmEvidenceBudget": _vlm_frame_limit(),
            "lightweightVlmFrameLimit": _vlm_frame_limit(),
            "lightweightVlmJobTimeoutSeconds": float(getattr(SETTINGS, "vlm_job_timeout_seconds", 0.0) or 0.0),
            "lightweightVlmMaxQualityFailures": int(getattr(SETTINGS, "vlm_max_quality_failures", 1) or 0),
            "effectiveDevice": device_gateway.get("components", {}).get("detector", {}).get("selected", SETTINGS.device),
            "mobileSamDevice": device_gateway.get("components", {}).get("mobileSam", {}).get("selected"),
            "lightweightVlmDevice": device_gateway.get("components", {}).get("lightweightVlm", {}).get("selected"),
            "enhancedVlmDevice": device_gateway.get("components", {}).get("enhancedVlm", {}).get("selected"),
            "safeModeMessage": (
                "MobileSAM worker + Local VLM Assist active. Fast Local Analysis remains available."
                if (
                    _analysis_safe_mode()
                    and _safe_mode_allow_mobile_sam()
                    and _mobile_sam_worker_enabled()
                    and _safe_mode_lightweight_vlm_worker_allowed()
                )
                else "Local VLM Assist worker enabled. Fast Local Analysis remains available; MobileSAM disabled."
                if _analysis_safe_mode() and _safe_mode_lightweight_vlm_worker_allowed()
                else
                "Local runtime guard active. Fast Local Analysis remains available; MobileSAM worker refinement may run on selected evidence frames."
                if _safe_mode_allow_mobile_sam() and _mobile_sam_worker_enabled()
                else "Local runtime guard active. Fast Local Analysis remains available; MobileSAM refinement may run on selected evidence frames."
                if _safe_mode_allow_mobile_sam()
                else "Local runtime guard active. Fast Local Analysis remains available."
                if _analysis_safe_mode()
                else "Standard analysis mode active."
            ),
            "analysisJobTimeoutSeconds": SETTINGS.analysis_job_timeout_seconds,
        },
        "models": {key: _model_status_payload(value) for key, value in models.items()},
        "visual_explanations": _visual_explanations_payload(models["vlm"]),
        "chat": chat,
        "openmp": openmp,
        "strictFullLocalPreflight": _full_local_preflight_report(),
        "frontend": _frontend_status_payload(),
        "uploadLimits": {
            "maxUploadMb": SETTINGS.max_upload_mb,
            "maxVideoDurationSeconds": SETTINGS.max_video_duration_seconds,
            "maxSampledFrames": SETTINGS.max_frames,
        },
        "batchLimits": {
            "bulkMaxFiles": SETTINGS.bulk_max_files,
            "bulkMaxUncompressedMb": SETTINGS.bulk_max_uncompressed_mb,
            "workerConcurrency": SETTINGS.worker_concurrency,
        },
        "jobStorePath": _display_path(store.root_dir),
    }


def get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store


def get_batch_store(request: Request) -> BatchStore:
    return request.app.state.batch_store


def _configure_frontend_static(app: FastAPI) -> None:
    if not bool(SETTINGS.serve_frontend):
        return
    dist = _frontend_dist_path()
    index = dist / "index.html"
    if not index.is_file():
        return
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    def frontend_index() -> FileResponse:
        return FileResponse(index)

    @app.get("/{frontend_path:path}", include_in_schema=False)
    def frontend_fallback(frontend_path: str) -> FileResponse:
        if frontend_path.startswith("api/"):
            raise HTTPException(status_code=404, detail={"message": "API route not found"})
        candidate = (dist / frontend_path).resolve()
        try:
            candidate.relative_to(dist.resolve())
        except ValueError:
            return FileResponse(index)
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


def _upload_http_error(exc: UploadValidationError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail={"message": exc.message})


def _batch_http_error(exc: BatchValidationError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail={"message": exc.message})


def _enforce_ingestion_capacity(store: JobStore) -> None:
    probe = store.root_dir
    probe.mkdir(parents=True, exist_ok=True)
    free_mb = shutil.disk_usage(probe).free / (1024 * 1024)
    required_disk_mb = max(
        float(getattr(SETTINGS, "min_free_disk_mb", 0) or 0),
        float(getattr(SETTINGS, "min_free_disk_gb", 0) or 0) * 1024,
    )
    if required_disk_mb > 0 and free_mb < required_disk_mb:
        raise HTTPException(
            status_code=507,
            detail={"message": "SafeTrace is waiting for free disk capacity before accepting more footage."},
        )
    required_memory_mb = float(getattr(SETTINGS, "min_available_memory_mb", 0) or 0)
    if required_memory_mb <= 0:
        return
    try:
        import psutil

        available_mb = psutil.virtual_memory().available / (1024 * 1024)
    except (ImportError, OSError):
        return
    if available_mb < required_memory_mb:
        raise HTTPException(
            status_code=503,
            detail={"message": "SafeTrace is waiting for available memory before accepting more footage."},
        )


async def _read_upload_content(file: UploadFile, *, limit: Optional[int] = None) -> bytes:
    byte_limit = limit if limit is not None else max_upload_bytes()
    known_size = getattr(file, "size", None)
    if known_size is not None:
        try:
            validate_upload_size(int(known_size), limit_bytes=byte_limit)
        except UploadValidationError as exc:
            raise _upload_http_error(exc) from exc

    chunks = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        try:
            validate_upload_size(total, limit_bytes=byte_limit)
        except UploadValidationError as exc:
            raise _upload_http_error(exc) from exc
        chunks.append(chunk)
    return b"".join(chunks)


def _execute_job_group(store: JobStore, job_ids: list[str]) -> None:
    schedule_analysis_jobs(store, job_ids)
    # Starlette sends the live response before background work runs. Pytest's
    # TestClient deliberately awaits background tasks, so retain its historic
    # deterministic contract without making production uploads synchronous.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        wait_for_scheduled_jobs(store, job_ids)


def create_app(job_store: JobStore | None = None, batch_store: BatchStore | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if getattr(application.state, "lifecycle_started", False):
            yield
            return
        application.state.lifecycle_started = True
        resume_recoverable_batch_jobs(application)
        scheduler = RetentionScheduler(application.state.job_store, application.state.batch_store)
        application.state.retention_scheduler = scheduler
        scheduler.start()
        try:
            yield
        finally:
            scheduler.stop()
            application.state.lifecycle_started = False

    app = FastAPI(title="SafeTrace Local API", version=__version__, lifespan=lifespan)
    app.state.job_store = job_store or JobStore()
    app.state.batch_store = batch_store or BatchStore()
    app.state.recovery_decision_lock = threading.Lock()

    def resume_recoverable_batch_jobs(application: FastAPI) -> None:
        store: JobStore = application.state.job_store
        policy = normalized_recovery_policy()
        for record in store.records(include_disk=True):
            recover_on_restart = bool((record.source_metadata or {}).get("recoverOnRestart"))
            if not recover_on_restart or record.status not in {"queued", "recovering", "retry_wait"}:
                continue
            if policy == "prompt":
                if record.status in {"recovering", "retry_wait"}:
                    store.set_recovery_action(record.job_id, "awaiting_recovery_decision", pause=True)
                continue
            if policy == "discard":
                if record.status in {"recovering", "retry_wait"}:
                    store.set_recovery_action(record.job_id, "awaiting_recovery_decision", pause=True)
                continue
            delay = 0.0
            if record.status == "retry_wait" and record.next_retry_at is not None:
                delay = max((record.next_retry_at - datetime.now(timezone.utc)).total_seconds(), 0.0)

            def resume_job(job_id: str = record.job_id) -> None:
                current = store.get(job_id)
                if current is None:
                    return
                if current.status in {"recovering", "retry_wait"} and not store.activate_retry(job_id):
                    return
                schedule_analysis_jobs(store, [job_id])

            if delay > 0:
                timer = threading.Timer(delay, resume_job)
                timer.daemon = True
                timer.start()
            else:
                thread = threading.Thread(target=resume_job, daemon=True)
                thread.start()
    initial_vlm_profile = _normalized_vlm_profile(getattr(SETTINGS, "vlm_profile", VLM_PROFILE_RULE_BASED))
    app.state.vlm_selected_profile = initial_vlm_profile
    app.state.vlm_enabled = _initial_vlm_enabled(initial_vlm_profile)
    _configure_cors(app)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):  # noqa: ARG001
        return JSONResponse(
            status_code=500,
            content={"detail": {"message": "Internal server error"}},
        )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            api="safetrace-local",
            version="dev",
            offline=SETTINGS.offline,
        )

    @app.get("/api/system/status", response_model=SystemStatusResponse)
    def system_status(request: Request, store: JobStore = Depends(get_job_store)) -> SystemStatusResponse:
        device_gateway = device_gateway_payload(SETTINGS)
        gpu_available = bool(device_gateway.get("torch", {}).get("cuda_available"))
        vlm_profiles = _current_vlm_payload(request.app)
        models = {
            "embeddingModel": _path_status(SETTINGS.siglip_model_dir),
            "detector": _detector_status(),
            "mobileSam": _mobile_sam_status(),
            "vlm": _vlm_model_status_for_payload(vlm_profiles),
            "midVlm": _path_status(
                _resolve_configured_path(Path(SETTINGS.mid_vlm_model_path)),
                optional=True,
                unavailable_message="Optional 512M-1B scene verifier is not installed.",
            ),
        }
        if models["midVlm"].details is None:
            models["midVlm"].details = {}
        models["midVlm"].details.update(
            {
                "enabled": bool(SETTINGS.mid_vlm_enabled),
                "role": "non-authoritative scene applicability verifier",
                "device": SETTINGS.mid_vlm_device,
                "concurrency": min(1, int(SETTINGS.mid_vlm_concurrency)),
                "maxFrames": SETTINGS.mid_vlm_max_frames,
                "timeoutSeconds": SETTINGS.mid_vlm_timeout_seconds,
            }
        )
        limits = {
            "maxUploadMb": SETTINGS.max_upload_mb,
            "bulkMaxFiles": SETTINGS.bulk_max_files,
            "bulkMaxUncompressedMb": SETTINGS.bulk_max_uncompressed_mb,
            "maxVideoDurationSeconds": SETTINGS.max_video_duration_seconds,
            "maxVideoDurationUnlimited": SETTINGS.max_video_duration_seconds <= 0,
            "maxVideoDurationMessage": (
                "No explicit video duration cap is enforced; sampled frames remain bounded."
                if SETTINGS.max_video_duration_seconds <= 0
                else "Video duration cap is enforced during frame extraction."
            ),
            "maxSampledFrames": SETTINGS.max_frames,
            "embeddingBatchSize": SETTINGS.embedding_batch_size,
            "embeddingWindowSize": SETTINGS.embedding_window_size,
            "embeddingWindowStride": SETTINGS.embedding_window_stride,
            "embeddingPoolingStrategy": SETTINGS.embedding_pooling_strategy,
            "workerConcurrency": SETTINGS.worker_concurrency,
            "analysisConcurrency": getattr(SETTINGS, "analysis_concurrency", SETTINGS.worker_concurrency),
            "jobConcurrency": getattr(SETTINGS, "job_concurrency", SETTINGS.analysis_concurrency),
            "vlmConcurrency": getattr(SETTINGS, "vlm_concurrency", 1),
            "batchMaxActiveJobs": SETTINGS.batch_max_active_jobs,
            "perBatchConcurrency": getattr(SETTINGS, "per_batch_concurrency", SETTINGS.batch_max_active_jobs),
            "cpuPreprocessingConcurrency": SETTINGS.cpu_preprocessing_concurrency,
            "gpuDetectorConcurrency": SETTINGS.gpu_detector_concurrency,
            "gpuInferenceConcurrency": getattr(SETTINGS, "gpu_inference_concurrency", SETTINGS.gpu_detector_concurrency),
            "mobileSamConcurrency": SETTINGS.mobile_sam_concurrency,
            "maxQueuedJobs": SETTINGS.max_queued_jobs,
            "schedulerPolicy": getattr(SETTINGS, "scheduler_policy", "oldest_batch_first"),
            "adaptiveWorkersEnabled": SETTINGS.adaptive_workers_enabled,
            "adaptiveWorkerMin": SETTINGS.adaptive_worker_min,
            "adaptiveWorkerMax": SETTINGS.adaptive_worker_max,
            "adaptiveThirdWorkerValidated": SETTINGS.adaptive_third_worker_validated,
            "adaptiveScaleUpQueueDepth": SETTINGS.adaptive_scale_up_queue_depth,
            "adaptiveScaleUpSustainSeconds": SETTINGS.adaptive_scale_up_sustain_seconds,
            "adaptiveScaleDownIdleSeconds": SETTINGS.adaptive_scale_down_idle_seconds,
            "adaptiveScaleCooldownSeconds": SETTINGS.adaptive_scale_cooldown_seconds,
            "adaptiveGpuFreeMemoryMarginMb": SETTINGS.adaptive_gpu_free_memory_margin_mb,
            "adaptiveSystemFreeMemoryMarginMb": SETTINGS.adaptive_system_free_memory_margin_mb,
            "runtimeProfile": getattr(SETTINGS, "runtime_profile", "portable_fast_local"),
            "minFreeDiskMb": SETTINGS.min_free_disk_mb,
            "minAvailableMemoryMb": SETTINGS.min_available_memory_mb,
            "jobRetryMaxAttempts": SETTINGS.job_retry_max_attempts,
            "jobRetryBackoffSeconds": SETTINGS.job_retry_backoff_seconds,
            "comprehensiveReview": {
                "fps": SETTINGS.comprehensive_review_fps,
                "maxFrames": SETTINGS.comprehensive_review_max_frames,
                "topK": SETTINGS.comprehensive_review_top_k,
                "candidateWindowSeconds": SETTINGS.comprehensive_candidate_window_seconds,
                "maxSegmentationFrames": SETTINGS.comprehensive_max_segmentation_frames,
                "maxVlmFrames": SETTINGS.comprehensive_max_vlm_frames,
            },
            "jobRetentionHours": SETTINGS.job_retention_hours,
            "staleRunningMinutes": SETTINGS.stale_running_minutes,
            "recoveryPolicy": normalized_recovery_policy(),
            "retention": {
                "completedRetentionDays": SETTINGS.completed_retention_days,
                "failedRetentionDays": SETTINGS.failed_retention_days,
                "recoveryRetentionHours": SETTINGS.recovery_retention_hours,
                "cacheRetentionDays": SETTINGS.cache_retention_days,
                "cacheMaxGb": SETTINGS.cache_max_gb,
                "logRetentionDays": SETTINGS.log_retention_days,
                "minFreeDiskGb": SETTINGS.min_free_disk_gb,
                "schedulerEnabled": SETTINGS.retention_scheduler_enabled,
                "schedulerIntervalMinutes": SETTINGS.retention_interval_minutes,
                "schedulerStartupDelaySeconds": SETTINGS.retention_startup_delay_seconds,
            },
            "analysisSafeMode": _analysis_safe_mode(),
            "analysisJobTimeoutSeconds": SETTINGS.analysis_job_timeout_seconds,
            "deviceGateway": device_gateway,
        }
        chat = chat_status_payload(allow_model_load=False)
        openmp = _openmp_status()
        return SystemStatusResponse(
            app_version=__version__,
            backend_version=__version__,
            build_mode=os.environ.get("SAFETRACE_BUILD_MODE", "development"),
            runtime_layout=os.environ.get("SAFETRACE_RUNTIME_LAYOUT", "source"),
            safeMode=_analysis_safe_mode(),
            device=SETTINGS.device,
            gpuAvailable=gpu_available,
            models=models,
            limits=limits,
            queue={
                "statusCounts": store.status_counts(),
                "activeStates": ["queued", "running", "running_preprocess", "running_detector", "running_refinement", "running_report"],
                "terminalStates": ["completed", "failed", "cancelled"],
                "scheduler": scheduler_status_payload(),
            },
            runtime=_runtime_payload(
                store=store,
                models=models,
                gpu_available=gpu_available,
                device_gateway=device_gateway,
                chat=chat,
                openmp=openmp,
            ),
            preflight=_preflight_payload(models=models, chat=chat, openmp=openmp),
            vlm=vlm_profiles,
        )

    @app.get("/api/recovery")
    def recovery_list(
        includeDeferred: bool = Query(False),
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        return recovery_candidates(store, batches, include_deferred=includeDeferred)

    @app.post("/api/recovery/resume")
    def recovery_resume(
        http_request: Request,
        background_tasks: BackgroundTasks,
        request: dict[str, Any] = Body(default_factory=dict),
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        decision_lock = http_request.app.state.recovery_decision_lock
        with decision_lock:
            available = recovery_candidates(store, batches, include_deferred=True)["jobs"]
            requested_jobs = set(str(item) for item in request.get("jobIds") or [])
            requested_batches = set(str(item) for item in request.get("batchIds") or [])
            select_all = bool(request.get("all")) or (not requested_jobs and not requested_batches)
            selected = [item for item in available if select_all or item["jobId"] in requested_jobs or item.get("batchId") in requested_batches]
            resumed = []
            skipped = []
            for item in selected:
                record = store.get(item["jobId"])
                if record is None or not record.upload_path.is_file():
                    skipped.append({"jobId": item["jobId"], "reason": "missing_upload"})
                    continue
                if record.status == "failed":
                    activated = store.reset_for_retry(record.job_id, recovery_action="resumed_by_user")
                elif record.status == "paused":
                    activated = store.resume(record.job_id)
                else:
                    store.set_recovery_action(record.job_id, "resumed_by_user")
                    activated = store.activate_retry(record.job_id)
                if activated:
                    resumed.append(record.job_id)
                    background_tasks.add_task(_execute_job_group, store, [record.job_id])
                else:
                    skipped.append({"jobId": record.job_id, "reason": f"status_{record.status}"})
        return {"status": "accepted", "resumedJobIds": resumed, "skipped": skipped}

    @app.post("/api/recovery/later")
    def recovery_later(
        http_request: Request,
        request: dict[str, Any] = Body(default_factory=dict),
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        with http_request.app.state.recovery_decision_lock:
            available = recovery_candidates(store, batches, include_deferred=True)["jobs"]
            requested = set(str(item) for item in request.get("jobIds") or [])
            selected = [item for item in available if not requested or item["jobId"] in requested]
            deferred = [item["jobId"] for item in selected if store.set_recovery_action(item["jobId"], "deferred_by_user", pause=True)]
        return {"status": "deferred", "jobIds": deferred}

    @app.post("/api/recovery/restart-fresh")
    def recovery_restart_fresh(
        http_request: Request,
        background_tasks: BackgroundTasks,
        request: dict[str, Any] = Body(default_factory=dict),
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        if request.get("confirmRestartFresh") is not True:
            raise HTTPException(status_code=400, detail={"message": "Fresh restart confirmation is required"})
        purge_cache = bool(request.get("purgeCompatibleCache"))
        if purge_cache and request.get("confirmPurgeCompatibleCache") is not True:
            raise HTTPException(
                status_code=400,
                detail={"message": "Compatible cache purge requires separate confirmation"},
            )
        with http_request.app.state.recovery_decision_lock:
            available = recovery_candidates(store, batches, include_deferred=True)["jobs"]
            available_ids = {str(item["jobId"]) for item in available}
            requested_jobs = {str(item) for item in request.get("jobIds") or []}
            requested_batches = {str(item) for item in request.get("batchIds") or []}
            select_all = bool(request.get("all")) or (not requested_jobs and not requested_batches)
            selected_ids = set(available_ids if select_all else (available_ids & requested_jobs))
            for batch_id in requested_batches:
                batch = batches.get(batch_id, store)
                if batch is None:
                    continue
                selected_ids.update(
                    job_id
                    for job_id in batch.job_ids
                    if (store.get(job_id) is not None and store.require(job_id).status != "completed")
                )

            already_restarted = []
            for job_id in requested_jobs - selected_ids:
                existing = store.get(job_id)
                if existing is not None and existing.recovery_action == "restart_fresh_by_user":
                    already_restarted.append(job_id)

            restarted = []
            skipped = []
            details = []
            for job_id in sorted(selected_ids):
                record = store.get(job_id)
                if record is None:
                    skipped.append({"jobId": job_id, "reason": "missing_job"})
                    continue
                if record.status == "completed":
                    skipped.append({"jobId": job_id, "reason": "completed_preserved"})
                    continue
                try:
                    result = store.restart_fresh(
                        job_id,
                        purge_compatible_cache=purge_cache,
                        cache_purge_selected_job_ids=selected_ids,
                    )
                except JobRestartConflictError as exc:
                    skipped.append({"jobId": job_id, "reason": "active_conflict", "message": str(exc)})
                    continue
                restarted.append(job_id)
                details.append(result)
            if restarted:
                background_tasks.add_task(_execute_job_group, store, restarted)
        return {
            "status": "accepted",
            "restartedJobIds": restarted,
            "alreadyRestartedJobIds": sorted(already_restarted),
            "skipped": skipped,
            "purgeCompatibleCache": purge_cache,
            "jobs": details,
        }

    @app.post("/api/recovery/discard")
    def recovery_discard(
        http_request: Request,
        request: dict[str, Any] = Body(default_factory=dict),
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        with http_request.app.state.recovery_decision_lock:
            available = recovery_candidates(store, batches, include_deferred=True)["jobs"]
            requested_jobs = set(str(item) for item in request.get("jobIds") or [])
            requested_batches = set(str(item) for item in request.get("batchIds") or [])
            select_all = bool(request.get("all")) or (not requested_jobs and not requested_batches)
            selected = [item for item in available if select_all or item["jobId"] in requested_jobs or item.get("batchId") in requested_batches]
            reclaimed = 0
            deleted = []
            for item in selected:
                record = store.get(item["jobId"])
                if record is None or (record.status in {"running", "completed"} and record.error_type != "InterruptedJob"):
                    continue
                reclaimed += directory_size(record.job_dir)
                if store.delete(record.job_id):
                    deleted.append(record.job_id)
            for batch_id in {item.get("batchId") for item in selected if item.get("batchId")}:
                batch = batches.get(str(batch_id), store)
                if batch and all(store.get(job_id) is None for job_id in batch.job_ids):
                    batches.delete(str(batch_id), store)
        return {"status": "discarded", "deletedJobIds": deleted, "actualReclaimedBytes": reclaimed}

    @app.get("/api/recovery/{batch_id}")
    def recovery_batch(
        batch_id: str,
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        payload = recovery_candidates(store, batches, include_deferred=True)
        match = next((item for item in payload["batches"] if item.get("batchId") == batch_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail={"message": "Recovery batch not found"})
        return match

    @app.get("/api/storage/summary")
    def storage_status(store: JobStore = Depends(get_job_store), batches: BatchStore = Depends(get_batch_store)):
        return storage_summary(store, batches)

    @app.post("/api/storage/cleanup/preview")
    def storage_cleanup_preview(request: dict[str, Any] = Body(default_factory=dict), store: JobStore = Depends(get_job_store)):
        return cleanup_preview(store, scopes=request.get("scopes"))

    @app.post("/api/storage/cleanup/apply")
    def storage_cleanup_apply(
        request: dict[str, Any] = Body(default_factory=dict),
        store: JobStore = Depends(get_job_store),
    ):
        if request.get("confirm") is not True:
            raise HTTPException(status_code=400, detail={"message": "Cleanup confirmation is required"})
        preview = cleanup_preview(store, scopes=request.get("scopes"))
        try:
            return {**apply_cleanup(store, preview), "preview": preview}
        except CleanupInProgressError as exc:
            raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc

    @app.get("/api/storage/cleanup/history")
    def storage_cleanup_history(page: int = Query(1, ge=1), pageSize: int = Query(25, ge=1, le=100)):
        audit_dir = Path(SETTINGS.data_dir) / "cleanup_audit"
        items = []
        if audit_dir.exists():
            for path in sorted(audit_dir.glob("cleanup_*.json"), reverse=True):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                items.append({**payload, "auditPath": str(path)})
        start = (page - 1) * pageSize
        return {"page": page, "pageSize": pageSize, "total": len(items), "items": items[start:start + pageSize]}

    @app.get("/api/dashboard/summary")
    def dashboard_status(store: JobStore = Depends(get_job_store), batches: BatchStore = Depends(get_batch_store)):
        return dashboard_summary(store, batches)

    @app.get("/api/jobs")
    def list_jobs(
        page: int = Query(1, ge=1),
        pageSize: int = Query(25, ge=1, le=100),
        status: Optional[str] = Query(None),
        store: JobStore = Depends(get_job_store),
    ):
        records = sorted(store.records(include_disk=False), key=lambda item: item.updated_at, reverse=True)
        if status:
            records = [item for item in records if item.status == status]
        start = (page - 1) * pageSize
        return {"page": page, "pageSize": pageSize, "total": len(records), "items": [item.status_payload() for item in records[start:start + pageSize]]}

    @app.get("/api/exports")
    def exports_list(
        page: int = Query(1, ge=1),
        pageSize: int = Query(25, ge=1, le=100),
        store: JobStore = Depends(get_job_store),
    ):
        exports = list_exports(store)
        start = (page - 1) * pageSize
        return {"page": page, "pageSize": pageSize, "total": len(exports), "items": exports[start:start + pageSize]}

    @app.get("/api/batches")
    def list_batches(
        page: int = Query(1, ge=1),
        pageSize: int = Query(25, ge=1, le=100),
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        records = sorted(batches.records(include_disk=False, job_store=store), key=lambda item: item.updated_at, reverse=True)
        start = (page - 1) * pageSize
        return {"page": page, "pageSize": pageSize, "total": len(records), "items": [item.payload() for item in records[start:start + pageSize]]}

    @app.post("/api/system/vlm/settings", response_model=VlmSettingsResponse)
    def update_vlm_settings(request: Request, settings: VlmSettingsRequest) -> VlmSettingsResponse:
        request.app.state.vlm_selected_profile = settings.selectedProfile
        hard_disabled = _vlm_hard_disabled(settings.selectedProfile, bool(settings.enabled))
        request.app.state.vlm_enabled = (
            bool(settings.enabled)
            and settings.selectedProfile != VLM_PROFILE_RULE_BASED
            and not hard_disabled
            and _vlm_profile_available(settings.selectedProfile)
        )
        return VlmSettingsResponse(**_current_vlm_payload(request.app))

    @app.get("/api/chat/status", response_model=ChatStatusResponse)
    def chat_status() -> ChatStatusResponse:
        return ChatStatusResponse(**chat_status_payload())

    @app.post("/api/chat/warmup", response_model=ChatStatusResponse)
    def chat_warmup() -> ChatStatusResponse:
        try:
            return ChatStatusResponse(**warmup_chat_provider())
        except ChatDisabledError as exc:
            raise HTTPException(status_code=503, detail={"message": str(exc)}) from exc
        except ChatProviderUnavailableError as exc:
            raise HTTPException(status_code=503, detail={"message": str(exc)}) from exc

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(
        request: ChatRequest,
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ) -> ChatResponse:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail={"message": "Message is required"})
        try:
            payload = answer_chat(
                message=request.message,
                job_store=store,
                batch_store=batches,
                job_id=request.job_id,
                batch_id=request.batch_id,
                include_current_result=request.include_current_result,
            )
        except ChatDisabledError as exc:
            raise HTTPException(status_code=503, detail={"message": str(exc)}) from exc
        except ChatProviderUnavailableError as exc:
            raise HTTPException(status_code=503, detail={"message": str(exc)}) from exc
        return ChatResponse(**payload)

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        query: str = Form(...),
        fps: float = Form(1.0),
        topK: int = Form(5),
        enableVlm: bool = Form(False),
        vlmProfile: Optional[str] = Form(None),
        vlmEnabled: Optional[bool] = Form(None),
        useCaseProfile: Optional[str] = Form(None),
        device: DeviceMode = Form("auto"),
        reviewMode: str = Form("fast_local"),
        store: JobStore = Depends(get_job_store),
    ) -> AnalyzeResponse:
        _enforce_ingestion_capacity(store)
        if not query.strip():
            raise HTTPException(status_code=400, detail={"message": "Query is required"})
        if fps <= 0:
            raise HTTPException(status_code=400, detail={"message": "fps must be greater than zero"})
        if topK <= 0:
            raise HTTPException(status_code=400, detail={"message": "topK must be greater than zero"})

        try:
            clean_filename = validate_upload_filename(file.filename or "upload.bin")
        except UploadValidationError as exc:
            raise _upload_http_error(exc) from exc

        file.file.seek(0)
        try:
            record = store.create_job_from_stream(
                filename=clean_filename,
                stream=file.file,
                query=query.strip(),
                settings=_analysis_settings_from_request(
                    request.app,
                    fps=fps,
                    top_k=topK,
                    enable_vlm=enableVlm,
                    device=device,
                    vlm_profile=vlmProfile,
                    vlm_enabled=vlmEnabled,
                    use_case_profile=_parse_use_case_profile(useCaseProfile),
                    review_mode=reviewMode,
                ),
                source_metadata={
                    "originalFilename": clean_filename,
                    "sourceRelativePath": clean_filename,
                    "sourceDirectory": "",
                    "sourceGroupPath": "",
                    "sourceGroupLabel": "Ungrouped",
                },
            )
        except UploadValidationError as exc:
            raise _upload_http_error(exc) from exc
        background_tasks.add_task(_execute_job_group, store, [record.job_id])
        return AnalyzeResponse(jobId=record.job_id, status="queued")

    @app.post("/api/batches/analyze", response_model=BatchResponse)
    async def analyze_batch(
        request: Request,
        background_tasks: BackgroundTasks,
        files: list[UploadFile] = File(...),
        query: str = Form(...),
        fps: float = Form(1.0),
        topK: int = Form(5),
        enableVlm: bool = Form(False),
        vlmProfile: Optional[str] = Form(None),
        vlmEnabled: Optional[bool] = Form(None),
        useCaseProfile: Optional[str] = Form(None),
        device: DeviceMode = Form("auto"),
        reviewMode: str = Form("fast_local"),
        relativePaths: Optional[list[str]] = Form(None),
        importKey: Optional[str] = Form(None),
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ) -> BatchResponse:
        _enforce_ingestion_capacity(store)
        if not query.strip():
            raise HTTPException(status_code=400, detail={"message": "Query is required"})
        if fps <= 0:
            raise HTTPException(status_code=400, detail={"message": "fps must be greater than zero"})
        if topK <= 0:
            raise HTTPException(status_code=400, detail={"message": "topK must be greater than zero"})
        if not files:
            raise HTTPException(status_code=400, detail={"message": "Select at least one file for batch analysis"})
        queue_counts = store.status_counts(include_disk=True)
        queued_count = sum(
            queue_counts.get(key, 0)
            for key in (
                "queued", "waiting_for_capacity", "waiting_for_job_slot", "waiting_for_gpu",
                "waiting_for_mobilesam", "waiting_for_vlm", "retry_wait", "recovering",
            )
        )
        if queued_count + len(files) > int(getattr(SETTINGS, "max_queued_jobs", 250)):
            raise HTTPException(
                status_code=429,
                detail={"message": "SafeTrace queue capacity reached. Retry when active jobs have completed."},
            )

        settings = _analysis_settings_from_request(
            request.app,
            fps=fps,
            top_k=topK,
            enable_vlm=enableVlm,
            device=device,
            vlm_profile=vlmProfile,
            vlm_enabled=vlmEnabled,
            use_case_profile=_parse_use_case_profile(useCaseProfile),
            review_mode=reviewMode,
        )

        try:
            if len(files) == 1 and (files[0].filename or "").lower().endswith(".zip"):
                archive = files[0]
                archive.file.seek(0)
                batch = batches.create_from_zip_stream(
                    filename=archive.filename or "upload.zip",
                    stream=archive.file,
                    query=query.strip(),
                    settings=settings,
                    job_store=store,
                    import_key=importKey,
                )
            else:
                if any((file.filename or "").lower().endswith(".zip") for file in files):
                    raise HTTPException(
                        status_code=400,
                        detail={"message": "Upload one ZIP archive at a time, or upload video files directly."},
                    )
                supplied_paths = list(relativePaths or [])
                if supplied_paths and len(supplied_paths) != len(files):
                    raise HTTPException(
                        status_code=400,
                        detail={"message": "relativePaths must contain one entry for each uploaded file."},
                    )
                streamed = []
                for index, upload in enumerate(files):
                    upload.file.seek(0)
                    relative = supplied_paths[index] if supplied_paths else (upload.filename or "upload.bin")
                    streamed.append((relative, upload.file, None))
                batch = batches.create_from_streams(
                    files=streamed,
                    source_filename=f"{len(streamed)} selected files",
                    query=query.strip(),
                    settings=settings,
                    job_store=store,
                    import_key=importKey,
                )
        except BatchValidationError as exc:
            raise _batch_http_error(exc) from exc

        background_tasks.add_task(_execute_job_group, store, batch.job_ids)
        return BatchResponse(**batch.payload())

    @app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
    def job_status(job_id: str, store: JobStore = Depends(get_job_store)) -> JobStatusResponse:
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"message": "Job not found"})
        return JobStatusResponse(**record.status_payload())

    @app.get("/api/jobs/{job_id}/result", response_model=AnalysisResultResponse)
    def job_result(job_id: str, store: JobStore = Depends(get_job_store)) -> AnalysisResultResponse:
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"message": "Job not found"})
        if record.status == "failed":
            raise HTTPException(status_code=409, detail={"message": record.error or "Analysis failed"})
        if record.status != "completed" or record.result is None:
            raise HTTPException(status_code=409, detail={"message": "Analysis result is not ready"})
        try:
            return AnalysisResultResponse(**owned_result_snapshot(record))
        except ResultOwnershipError as exc:
            raise HTTPException(status_code=409, detail=exc.detail()) from exc

    @app.get("/api/batches/{batch_id}", response_model=BatchResponse)
    def batch_status(
        batch_id: str,
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ) -> BatchResponse:
        record = batches.get(batch_id, store)
        if record is None:
            raise HTTPException(status_code=404, detail={"message": "Batch not found"})
        return BatchResponse(**record.payload())

    @app.get("/api/batches/{batch_id}/tree")
    def batch_tree(
        batch_id: str,
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        record = batches.get(batch_id, store)
        if record is None:
            raise HTTPException(status_code=404, detail={"message": "Batch not found"})
        return {
            "batchId": batch_id,
            "status": record.status,
            "hierarchy": record.hierarchy,
            "groupSummaries": record.group_summaries,
            "statusCounts": record.status_counts,
        }

    @app.get("/api/batches/{batch_id}/results")
    def batch_results(
        batch_id: str,
        page: int = Query(1, ge=1),
        pageSize: int = Query(25, ge=1, le=100),
        status: Optional[str] = Query(None),
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        try:
            payload = batches.results(batch_id, store)
        except ResultOwnershipError as exc:
            raise HTTPException(status_code=409, detail=exc.detail()) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail={"message": "Batch not found"}) from exc
        results = list(payload["results"])
        if status:
            results = [item for item in results if store.get(item["jobId"]) is not None and store.require(item["jobId"]).status == status]
        start = (page - 1) * pageSize
        return {**payload, "page": page, "pageSize": pageSize, "total": len(results), "results": results[start:start + pageSize]}

    @app.post("/api/batches/{batch_id}/retry-failed")
    def batch_retry_failed(
        batch_id: str,
        background_tasks: BackgroundTasks,
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        try:
            payload = batches.retry_failed(batch_id, store)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail={"message": "Batch not found"}) from exc
        background_tasks.add_task(_execute_job_group, store, payload["retriedJobIds"])
        return payload

    @app.post("/api/batches/{batch_id}/pause")
    def batch_pause(
        batch_id: str,
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        try:
            return batches.pause(batch_id, store)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail={"message": "Batch not found"}) from exc

    @app.post("/api/batches/{batch_id}/resume")
    def batch_resume(
        batch_id: str,
        background_tasks: BackgroundTasks,
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        try:
            payload = batches.resume(batch_id, store)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail={"message": "Batch not found"}) from exc
        background_tasks.add_task(_execute_job_group, store, payload["resumedJobIds"])
        return payload

    @app.post("/api/batches/{batch_id}/cancel")
    def batch_cancel(
        batch_id: str,
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        try:
            return batches.cancel(batch_id, store)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail={"message": "Batch not found"}) from exc

    @app.get("/api/media/{job_id}/{filename:path}")
    def job_media(job_id: str, filename: str, store: JobStore = Depends(get_job_store)) -> FileResponse:
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"message": "Job not found"})
        path = resolve_job_media_path(record, filename)
        if path is None:
            raise HTTPException(status_code=404, detail={"message": "Media file not found"})
        if record.result is not None:
            try:
                owned = owned_result_snapshot(record)
            except ResultOwnershipError as exc:
                raise HTTPException(status_code=409, detail=exc.detail()) from exc
            matching = [
                frame
                for frame in owned.get("frames") or []
                if Path(str(frame.get("imageUrl") or "")).name == filename
            ]
            if not matching or any(
                str(frame.get("mediaArtifactId") or "") != f"{job_id}:{filename}"
                for frame in matching
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "cross_job_result_contamination",
                        "message": "Media artifact ownership does not match the requested job.",
                    },
                )
        return FileResponse(path)

    @app.get("/api/reports/{job_id}/technical-json")
    def technical_json(job_id: str, store: JobStore = Depends(get_job_store)):
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"message": "Job not found"})
        if record.status != "completed" or record.result is None:
            raise HTTPException(status_code=409, detail={"message": "Technical report is not ready"})
        try:
            owned = owned_result_snapshot(record)
        except ResultOwnershipError as exc:
            raise HTTPException(status_code=409, detail=exc.detail()) from exc
        return {
            **owned,
            "technicalDetails": {
                **(owned.get("technicalDetails") or {}),
                "job": record.status_payload(),
            },
        }

    @app.post("/api/jobs/{job_id}/secondary-reviews", response_model=SecondaryReviewResponse)
    def attach_secondary_review(
        job_id: str,
        review: SecondaryReviewRequest,
        store: JobStore = Depends(get_job_store),
    ) -> SecondaryReviewResponse:
        if store.get(job_id) is None:
            raise HTTPException(status_code=404, detail={"message": "Job not found"})
        try:
            payload = store.attach_secondary_review(job_id, review.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc
        return SecondaryReviewResponse(**payload)

    @app.post("/api/jobs/{job_id}/pin")
    def pin_job(job_id: str, store: JobStore = Depends(get_job_store)):
        try:
            return store.set_pin(job_id, True)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail={"message": "Job not found"}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc

    @app.post("/api/jobs/{job_id}/unpin")
    def unpin_job(job_id: str, store: JobStore = Depends(get_job_store)):
        try:
            return store.set_pin(job_id, False)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail={"message": "Job not found"}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc

    @app.post("/api/jobs/{job_id}/export")
    def export_job(
        job_id: str,
        request: dict[str, Any] = Body(default_factory=dict),
        store: JobStore = Depends(get_job_store),
    ):
        try:
            return create_export(
                store,
                job_id,
                selected_evidence_ids=request.get("selectedEvidenceIds"),
                include_evidence_images=bool(request.get("includeEvidenceImages", True)),
            )
        except ResultOwnershipError as exc:
            raise HTTPException(status_code=409, detail=exc.detail()) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail={"message": "Job not found"}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc

    @app.post("/api/jobs/{job_id}/export-and-delete")
    def export_and_delete_job(
        job_id: str,
        request: dict[str, Any] = Body(default_factory=dict),
        store: JobStore = Depends(get_job_store),
    ):
        if request.get("confirmDelete") is not True:
            raise HTTPException(status_code=400, detail={"message": "Export-and-delete confirmation is required"})
        try:
            record = store.require(job_id)
            before = directory_size(record.job_dir)
            exported = create_export(
                store,
                job_id,
                selected_evidence_ids=request.get("selectedEvidenceIds"),
                include_evidence_images=bool(request.get("includeEvidenceImages", True)),
            )
            verification = verify_export(Path(exported["path"]))
            if not verification["verified"]:
                raise RuntimeError("Verified export is required before source deletion.")
            if not store.delete(job_id):
                raise RuntimeError("Source job could not be deleted after export verification.")
            return {
                "jobId": job_id,
                "status": "exported_and_deleted",
                "export": exported,
                "exportedBytes": exported["sizeBytes"],
                "deletedBytes": before,
                "retainedReferences": [exported["exportId"]],
                "protectedArtifacts": [exported["path"]],
                "failures": [],
            }
        except ResultOwnershipError as exc:
            raise HTTPException(status_code=409, detail=exc.detail()) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail={"message": "Job not found"}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc

    @app.delete("/api/exports/{export_id}")
    def remove_export(export_id: str, store: JobStore = Depends(get_job_store)):
        try:
            payload = delete_export(store, export_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
        if payload["status"] == "not_found":
            raise HTTPException(status_code=404, detail={"message": "Export not found"})
        return payload

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str, store: JobStore = Depends(get_job_store)):
        try:
            if not store.delete(job_id):
                raise HTTPException(status_code=404, detail={"message": "Job not found"})
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc
        return {"jobId": job_id, "status": "deleted"}

    @app.delete("/api/batches/{batch_id}")
    def delete_batch(
        batch_id: str,
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        try:
            if not batches.delete(batch_id, store):
                raise HTTPException(status_code=404, detail={"message": "Batch not found"})
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc
        return {"batchId": batch_id, "status": "deleted"}

    _configure_frontend_static(app)

    return app


app = create_app()
