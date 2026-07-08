"""FastAPI app for the local SafeTrace backend."""
from __future__ import annotations

import os
import sys
import json
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
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
    JobStore,
    UploadValidationError,
    execute_analysis_job,
    max_upload_bytes,
    validate_upload_filename,
    validate_upload_size,
)
from .media import resolve_job_media_path
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
    origins = [
        *LOCAL_FRONTEND_ORIGINS,
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
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (SETTINGS.project_root / path).resolve()


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
    if suppressed_reason == "safe_mode":
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
                "Local runtime guard active. Fast Local Analysis remains available; "
                "MobileSAM may refine selected evidence frames."
                if mobile_sam_allowed
                else "Local runtime guard active. Fast Local Analysis remains available."
            ),
            "requestedVisualExplanationMode": selected_profile,
            "actualExplanationMode": "rule_based_with_mobilesam" if mobile_sam_allowed else VLM_PROFILE_RULE_BASED,
            "vlmAvailability": "disabled",
            "vlmSuppressedReason": "safe_mode",
            "fallbackReason": "Local runtime guard suppresses VLM.",
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
        device="cpu" if safe_mode else device,
        vlm_profile=requested_profile,
        vlm_enabled=requested_activation,
        safe_mode=safe_mode,
        use_case_profile=dict(use_case_profile or {}),
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
    return {
        "status": "available",
        "fallback": "rule_based",
        "explanationSource": source,
        "enhancedVlmAvailable": enhanced_available,
        "message": (
            "Visual explanations are enabled. Evidence cards show local visual review only when "
            "a local VLM adds reliable evidence; Fast Local Analysis remains available."
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


def create_app(job_store: JobStore | None = None, batch_store: BatchStore | None = None) -> FastAPI:
    app = FastAPI(title="SafeTrace Local API", version=__version__)
    app.state.job_store = job_store or JobStore()
    app.state.batch_store = batch_store or BatchStore()
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
        }
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
            "vlmConcurrency": getattr(SETTINGS, "vlm_concurrency", 1),
            "jobRetentionHours": SETTINGS.job_retention_hours,
            "staleRunningMinutes": SETTINGS.stale_running_minutes,
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
                "activeStates": ["queued", "running"],
                "terminalStates": ["completed", "failed", "cancelled"],
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
        store: JobStore = Depends(get_job_store),
    ) -> AnalyzeResponse:
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

        content = await _read_upload_content(file)
        if not content:
            raise HTTPException(status_code=400, detail={"message": "Uploaded file is empty"})

        record = store.create_job(
            filename=clean_filename,
            content=content,
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
            ),
        )
        background_tasks.add_task(execute_analysis_job, store, record.job_id)
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
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ) -> BatchResponse:
        if not query.strip():
            raise HTTPException(status_code=400, detail={"message": "Query is required"})
        if fps <= 0:
            raise HTTPException(status_code=400, detail={"message": "fps must be greater than zero"})
        if topK <= 0:
            raise HTTPException(status_code=400, detail={"message": "topK must be greater than zero"})
        if not files:
            raise HTTPException(status_code=400, detail={"message": "Select at least one file for batch analysis"})

        settings = _analysis_settings_from_request(
            request.app,
            fps=fps,
            top_k=topK,
            enable_vlm=enableVlm,
            device=device,
            vlm_profile=vlmProfile,
            vlm_enabled=vlmEnabled,
            use_case_profile=_parse_use_case_profile(useCaseProfile),
        )

        try:
            if len(files) == 1 and (files[0].filename or "").lower().endswith(".zip"):
                archive = files[0]
                content = await _read_upload_content(archive)
                if not content:
                    raise HTTPException(status_code=400, detail={"message": "Uploaded archive is empty"})
                batch = batches.create_from_zip(
                    filename=archive.filename or "upload.zip",
                    content=content,
                    query=query.strip(),
                    settings=settings,
                    job_store=store,
                )
            else:
                if any((file.filename or "").lower().endswith(".zip") for file in files):
                    raise HTTPException(
                        status_code=400,
                        detail={"message": "Upload one ZIP archive at a time, or upload video files directly."},
                    )
                materialized: list[tuple[str, bytes]] = []
                for file in files:
                    content = await _read_upload_content(file)
                    if not content:
                        materialized.append((file.filename or "upload.bin", b""))
                    else:
                        materialized.append((file.filename or "upload.bin", content))
                batch = batches.create_from_files(
                    files=materialized,
                    source_filename=f"{len(materialized)} selected files",
                    query=query.strip(),
                    settings=settings,
                    job_store=store,
                )
        except BatchValidationError as exc:
            raise _batch_http_error(exc) from exc

        for job_id in batch.job_ids:
            background_tasks.add_task(execute_analysis_job, store, job_id)
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
        return AnalysisResultResponse(**record.result)

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

    @app.get("/api/media/{job_id}/{filename:path}")
    def job_media(job_id: str, filename: str, store: JobStore = Depends(get_job_store)) -> FileResponse:
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"message": "Job not found"})
        path = resolve_job_media_path(record, filename)
        if path is None:
            raise HTTPException(status_code=404, detail={"message": "Media file not found"})
        return FileResponse(path)

    @app.get("/api/reports/{job_id}/technical-json")
    def technical_json(job_id: str, store: JobStore = Depends(get_job_store)):
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"message": "Job not found"})
        if record.status != "completed" or record.result is None:
            raise HTTPException(status_code=409, detail={"message": "Technical report is not ready"})
        return {
            **record.result,
            "technicalDetails": {
                **(record.result.get("technicalDetails") or {}),
                "job": record.status_payload(),
            },
        }

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str, store: JobStore = Depends(get_job_store)):
        if not store.delete(job_id):
            raise HTTPException(status_code=404, detail={"message": "Job not found"})
        return {"jobId": job_id, "status": "deleted"}

    @app.delete("/api/batches/{batch_id}")
    def delete_batch(
        batch_id: str,
        store: JobStore = Depends(get_job_store),
        batches: BatchStore = Depends(get_batch_store),
    ):
        if not batches.delete(batch_id, store):
            raise HTTPException(status_code=404, detail={"message": "Batch not found"})
        return {"batchId": batch_id, "status": "deleted"}

    _configure_frontend_static(app)

    return app


app = create_app()
