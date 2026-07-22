"""Central configuration for SafeTrace.

All paths, thresholds, and toggles live here. Values can be overridden via
environment variables to keep the system fully offline-configurable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_first(keys: tuple[str, ...], default: str) -> str:
    for key in keys:
        value = os.environ.get(key)
        if value is not None:
            return value
    return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _vlm_evidence_frame_budget() -> int:
    if os.environ.get("SAFETRACE_VLM_FRAME_LIMIT") is not None:
        return _env_int("SAFETRACE_VLM_FRAME_LIMIT", 5)
    if os.environ.get("SAFETRACE_VLM_MAX_EVIDENCE_FRAMES") is not None:
        return _env_int("SAFETRACE_VLM_MAX_EVIDENCE_FRAMES", 5)
    if os.environ.get("SAFETRACE_VLM_MAX_FRAMES") is not None:
        return _env_int("SAFETRACE_VLM_MAX_FRAMES", 5)
    return 5


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_csv(key: str, default: str = "") -> tuple[str, ...]:
    raw = os.environ.get(key, default)
    return tuple(part.strip().rstrip("/") for part in raw.split(",") if part.strip())


# Profiles keep the default desktop experience portable while making the
# resource-intensive local stack an explicit, testable opt-in.
_RUNTIME_PROFILES: Dict[str, Dict[str, str]] = {
    "test": {
        "device": "cpu", "analysis_safe_mode": "true", "mobile_sam_enabled": "disabled",
        "mobile_sam_worker_enabled": "false", "enable_vlm": "false", "vlm_enabled": "disabled",
        "lightweight_vlm_worker_enabled": "false", "chat_enabled": "disabled", "job_concurrency": "1",
        "preprocess_concurrency": "1", "gpu_inference_concurrency": "1", "mobile_sam_concurrency": "1",
        "vlm_concurrency": "1", "per_batch_concurrency": "1", "require_gpu": "false",
        "require_chat": "false", "require_mobilesam": "false", "allow_cpu_fallback": "true",
    },
    "portable_fast_local": {
        # Preserve the established source-runtime behaviour: portable mode
        # disables optional heavyweight layers, but it is not the legacy
        # analysis safe-mode switch.  ``local_full`` opts into the strict
        # guarded runtime explicitly.
        "device": "auto", "analysis_safe_mode": "false", "mobile_sam_enabled": "disabled",
        # ``auto`` leaves visual review inactive until explicitly selected;
        # unlike ``disabled`` it preserves the existing API contract for a
        # developer/tester who explicitly selects an installed local profile.
        "mobile_sam_worker_enabled": "false", "enable_vlm": "false", "vlm_enabled": "auto",
        "lightweight_vlm_worker_enabled": "false", "chat_enabled": "auto", "job_concurrency": "1",
        "preprocess_concurrency": "2", "gpu_inference_concurrency": "1", "mobile_sam_concurrency": "1",
        "vlm_concurrency": "1", "per_batch_concurrency": "1", "require_gpu": "false",
        "require_chat": "false", "require_mobilesam": "false", "allow_cpu_fallback": "true",
    },
    "local_full": {
        "device": "cuda", "analysis_safe_mode": "true", "safe_mode_allow_mobilesam": "true",
        "mobile_sam_enabled": "auto", "mobile_sam_worker_enabled": "true", "enable_vlm": "true",
        "vlm_enabled": "auto", "lightweight_vlm_worker_enabled": "true", "chat_enabled": "auto",
        "job_concurrency": "2", "preprocess_concurrency": "2", "gpu_inference_concurrency": "2",
        "mobile_sam_concurrency": "1", "vlm_concurrency": "1", "per_batch_concurrency": "2",
        "require_gpu": "true", "require_chat": "true", "require_mobilesam": "true", "allow_cpu_fallback": "false",
    },
}


def _runtime_profile_name() -> str:
    profile = _env("SAFETRACE_RUNTIME_PROFILE", "portable_fast_local").strip().lower()
    return profile if profile in _RUNTIME_PROFILES else "portable_fast_local"


def _profiled_env(key: str, profile_key: str, default: str) -> str:
    """An explicit environment setting always wins over the selected profile."""
    if key in os.environ:
        return os.environ[key]
    return _RUNTIME_PROFILES[_runtime_profile_name()].get(profile_key, default)


def _chat_speed_profile() -> str:
    return _env("SAFETRACE_CHAT_SPEED_PROFILE", "balanced").strip().lower() or "balanced"


def _chat_profile_int(key: str, balanced_default: int, fast_default: int) -> int:
    return _env_int(key, fast_default if _chat_speed_profile() == "fast" else balanced_default)


def _chat_profile_float(key: str, balanced_default: float, fast_default: float) -> float:
    return _env_float(key, fast_default if _chat_speed_profile() == "fast" else balanced_default)


PROJECT_ROOT = Path(_env("SAFETRACE_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)))


@dataclass
class Settings:
    # ---- Runtime profile ----
    runtime_profile: str = field(default_factory=_runtime_profile_name)

    # ---- Paths ----
    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: Path(_env("SAFETRACE_DATA_DIR", str(PROJECT_ROOT / "data"))))
    frames_dir: Path = field(
        default_factory=lambda: Path(_env("SAFETRACE_FRAMES_DIR", str(PROJECT_ROOT / "data" / "frames")))
    )
    embeddings_path: Path = field(
        default_factory=lambda: Path(_env("SAFETRACE_EMBEDDINGS_PATH", str(PROJECT_ROOT / "data" / "embeddings.npy")))
    )
    metadata_path: Path = field(
        default_factory=lambda: Path(_env("SAFETRACE_METADATA_PATH", str(PROJECT_ROOT / "data" / "metadata.json")))
    )
    index_path: Path = field(
        default_factory=lambda: Path(_env("SAFETRACE_INDEX_PATH", str(PROJECT_ROOT / "data" / "index.faiss")))
    )
    checkpoints_dir: Path = field(
        default_factory=lambda: Path(_env("SAFETRACE_CHECKPOINTS_DIR", str(PROJECT_ROOT / "checkpoints")))
    )

    # ---- Models (local checkpoints / local model dirs) ----
    siglip_model_dir: Path = field(
        default_factory=lambda: Path(
            _env("SAFETRACE_SIGLIP_DIR", str(PROJECT_ROOT / "checkpoints" / "siglip-base-patch16-224"))
        )
    )
    yolo_checkpoint: Path = field(
        default_factory=lambda: Path(
            _env("SAFETRACE_YOLO_CKPT", str(PROJECT_ROOT / "checkpoints" / "yolov9c-seg.pt"))
        )
    )
    yolo_fallback_checkpoint: Path = field(
        default_factory=lambda: Path(
            _env("SAFETRACE_YOLO_FALLBACK_CKPT", str(PROJECT_ROOT / "checkpoints" / "yolov8s-seg.pt"))
        )
    )
    mobile_sam_checkpoint: Path = field(
        default_factory=lambda: Path(
            _env_first(
                ("SAFETRACE_MOBILESAM_CHECKPOINT", "SAFETRACE_MSAM_CKPT"),
                str(PROJECT_ROOT / "checkpoints" / "mobile_sam.pt"),
            )
        )
    )
    vlm_model_dir: Path = field(
        default_factory=lambda: Path(
            _env_first(
                ("SAFETRACE_VLM_MODEL_PATH", "SAFETRACE_VLM_DIR"),
                str(PROJECT_ROOT / "models" / "vlm"),
            )
        )
    )
    vlm_provider: str = field(default_factory=lambda: _env("SAFETRACE_VLM_PROVIDER", "auto"))
    vlm_ollama_base_url: str = field(
        default_factory=lambda: _env("SAFETRACE_VLM_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )
    vlm_model: str = field(default_factory=lambda: _env("SAFETRACE_VLM_MODEL", "local-vlm"))
    vlm_profile: str = field(default_factory=lambda: _env("SAFETRACE_VLM_PROFILE", "rule_based"))
    vlm_lightweight_model_path: Path = field(
        default_factory=lambda: Path(
            _env("SAFETRACE_VLM_LIGHTWEIGHT_MODEL_PATH", str(PROJECT_ROOT / "models" / "vlm" / "lightweight-256m"))
        )
    )
    vlm_lightweight_512m_model_path: Path = field(
        default_factory=lambda: Path(
            _env(
                "SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH",
                str(PROJECT_ROOT / "models" / "vlm" / "lightweight-512m"),
            )
        )
    )
    vlm_enhanced_model_path: Path = field(
        default_factory=lambda: Path(
            _env("SAFETRACE_VLM_ENHANCED_MODEL_PATH", str(PROJECT_ROOT / "models" / "vlm" / "enhanced-2b"))
        )
    )
    vlm_enhanced_3b_model_path: Path = field(
        default_factory=lambda: Path(
            _env(
                "SAFETRACE_VLM_ENHANCED_3B_MODEL_PATH",
                str(PROJECT_ROOT / "models" / "vlm" / "enhanced-3b"),
            )
        )
    )
    vlm_timeout_seconds: float = field(default_factory=lambda: _env_float("SAFETRACE_VLM_TIMEOUT_SECONDS", 10.0))
    vlm_max_frames: int = field(default_factory=lambda: _env_int("SAFETRACE_VLM_MAX_FRAMES", 5))
    vlm_max_evidence_frames: int = field(default_factory=_vlm_evidence_frame_budget)
    vlm_max_tokens: int = field(default_factory=lambda: _env_int("SAFETRACE_VLM_MAX_TOKENS", 40))
    lightweight_vlm_primary: str = field(default_factory=lambda: _env("SAFETRACE_LIGHTWEIGHT_VLM_PRIMARY", "auto"))
    lightweight_vlm_fallback: str = field(default_factory=lambda: _env("SAFETRACE_LIGHTWEIGHT_VLM_FALLBACK", "256m"))
    lightweight_vlm_cpu_prefer_256m: bool = field(
        default_factory=lambda: _env_bool("SAFETRACE_LIGHTWEIGHT_VLM_CPU_PREFER_256M", True)
    )
    lightweight_vlm_timeout_seconds: float = field(
        default_factory=lambda: _env_float("SAFETRACE_LIGHTWEIGHT_VLM_TIMEOUT_SECONDS", 90.0)
    )
    lightweight_vlm_total_budget_seconds: float = field(
        default_factory=lambda: _env_float("SAFETRACE_LIGHTWEIGHT_VLM_TOTAL_BUDGET_SECONDS", 0.0)
    )
    lightweight_vlm_worker_enabled: bool = field(
        default_factory=lambda: _profiled_env(
            "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED", "lightweight_vlm_worker_enabled", "false"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
    )
    lightweight_vlm_worker_timeout_seconds: float = field(
        default_factory=lambda: _env_float("SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS", 60.0)
    )
    vlm_job_timeout_seconds: float = field(default_factory=lambda: _env_float("SAFETRACE_VLM_JOB_TIMEOUT_SECONDS", 0.0))
    vlm_max_quality_failures: int = field(default_factory=lambda: _env_int("SAFETRACE_VLM_MAX_QUALITY_FAILURES", 1))
    vlm_disable_after_timeout: bool = field(default_factory=lambda: _env_bool("SAFETRACE_VLM_DISABLE_AFTER_TIMEOUT", True))

    # ---- Runtime ----
    device: str = field(default_factory=lambda: _profiled_env("SAFETRACE_DEVICE", "device", "auto"))
    require_gpu: bool = field(default_factory=lambda: _profiled_env("SAFETRACE_REQUIRE_GPU", "require_gpu", "false").strip().lower() in {"1", "true", "yes", "y", "on"})
    require_chat: bool = field(default_factory=lambda: _profiled_env("SAFETRACE_REQUIRE_CHAT", "require_chat", "false").strip().lower() in {"1", "true", "yes", "y", "on"})
    require_mobilesam: bool = field(default_factory=lambda: _profiled_env("SAFETRACE_REQUIRE_MOBILESAM", "require_mobilesam", "false").strip().lower() in {"1", "true", "yes", "y", "on"})
    allow_cpu_fallback: bool = field(default_factory=lambda: _profiled_env("SAFETRACE_ALLOW_CPU_FALLBACK", "allow_cpu_fallback", "true").strip().lower() in {"1", "true", "yes", "y", "on"})
    enable_gpu_auto: bool = field(default_factory=lambda: _env_bool("SAFETRACE_ENABLE_GPU_AUTO", True))
    lightweight_vlm_device: str = field(default_factory=lambda: _env("SAFETRACE_LIGHTWEIGHT_VLM_DEVICE", "auto"))
    enhanced_vlm_device: str = field(default_factory=lambda: _env("SAFETRACE_ENHANCED_VLM_DEVICE", "cuda"))
    mobile_sam_device: str = field(default_factory=lambda: _env("SAFETRACE_MOBILESAM_DEVICE", "auto"))
    offline: bool = field(default_factory=lambda: _env_bool("SAFETRACE_OFFLINE", True))
    analysis_safe_mode: bool = field(
        default_factory=lambda: _profiled_env("SAFETRACE_ANALYSIS_SAFE_MODE", "analysis_safe_mode", "false")
        .strip().lower() in {"1", "true", "yes", "y", "on"}
    )
    safe_mode_allow_mobilesam: bool = field(
        default_factory=lambda: _profiled_env(
            "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM", "safe_mode_allow_mobilesam", "false"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
    )
    analysis_job_timeout_seconds: float = field(
        default_factory=lambda: _env_float("SAFETRACE_ANALYSIS_JOB_TIMEOUT_SECONDS", 600.0)
    )
    enable_vlm: bool = field(
        default_factory=lambda: _profiled_env("SAFETRACE_ENABLE_VLM", "enable_vlm", "false")
        .strip().lower() in {"1", "true", "yes", "y", "on"}
    )
    mobile_sam_enabled: str = field(
        default_factory=lambda: _profiled_env("SAFETRACE_MOBILESAM_ENABLED", "mobile_sam_enabled", "disabled")
    )
    mobile_sam_timeout_seconds: float = field(
        default_factory=lambda: _env_float("SAFETRACE_MOBILESAM_TIMEOUT_SECONDS", 20.0)
    )
    mobile_sam_worker_enabled: bool = field(
        default_factory=lambda: _profiled_env(
            "SAFETRACE_MOBILESAM_WORKER_ENABLED", "mobile_sam_worker_enabled", "false"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
    )
    mobile_sam_worker_timeout_seconds: float = field(
        default_factory=lambda: _env_float("SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS", 60.0)
    )
    vlm_enabled: str = field(default_factory=lambda: _profiled_env("SAFETRACE_VLM_ENABLED", "vlm_enabled", "auto"))
    serve_frontend: bool = field(default_factory=lambda: _env_bool("SAFETRACE_SERVE_FRONTEND", False))
    frontend_dist: Path = field(
        default_factory=lambda: Path(_env("SAFETRACE_FRONTEND_DIST", str(PROJECT_ROOT / "frontend-react" / "dist")))
    )
    allowed_origins: tuple[str, ...] = field(default_factory=lambda: _env_csv("SAFETRACE_ALLOWED_ORIGINS"))
    include_local_cors_origins: bool = field(
        default_factory=lambda: _env_bool("SAFETRACE_INCLUDE_LOCAL_CORS_ORIGINS", True)
    )

    # ---- Sampling / pipeline ----
    frame_fps: float = field(default_factory=lambda: _env_float("SAFETRACE_FPS", 1.0))
    max_frames: int = field(default_factory=lambda: _env_int("SAFETRACE_MAX_FRAMES", 600))
    top_k: int = field(default_factory=lambda: _env_int("SAFETRACE_TOPK", 5))
    embedding_batch_size: int = field(default_factory=lambda: _env_int("SAFETRACE_EMB_BATCH", 16))
    embedding_window_size: int = field(default_factory=lambda: _env_int("SAFETRACE_EMB_WINDOW_SIZE", 1))
    embedding_window_stride: int = field(default_factory=lambda: _env_int("SAFETRACE_EMB_WINDOW_STRIDE", 1))
    embedding_pooling_strategy: str = field(default_factory=lambda: _env("SAFETRACE_EMB_POOLING", "mean"))
    max_video_duration_seconds: float = field(
        default_factory=lambda: _env_float("SAFETRACE_MAX_VIDEO_SECONDS", 0.0)
    )
    worker_concurrency: int = field(
        default_factory=lambda: _env_int(
            "SAFETRACE_WORKER_CONCURRENCY",
            int(_profiled_env("SAFETRACE_JOB_CONCURRENCY", "job_concurrency", "1")),
        )
    )
    job_concurrency: int = field(
        default_factory=lambda: _env_int(
            "SAFETRACE_JOB_CONCURRENCY",
            int(_profiled_env("SAFETRACE_JOB_CONCURRENCY", "job_concurrency", "1")),
        )
    )
    analysis_concurrency: int = field(
        default_factory=lambda: _env_int(
            "SAFETRACE_ANALYSIS_CONCURRENCY",
            _env_int("SAFETRACE_JOB_CONCURRENCY", int(_profiled_env("SAFETRACE_JOB_CONCURRENCY", "job_concurrency", "1"))),
        )
    )
    vlm_concurrency: int = field(
        default_factory=lambda: _env_int(
            "SAFETRACE_VLM_CONCURRENCY", int(_profiled_env("SAFETRACE_VLM_CONCURRENCY", "vlm_concurrency", "1"))
        )
    )
    batch_max_active_jobs: int = field(
        default_factory=lambda: _env_int(
            "SAFETRACE_BATCH_MAX_ACTIVE_JOBS",
            int(_profiled_env("SAFETRACE_PER_BATCH_CONCURRENCY", "per_batch_concurrency", "1")),
        )
    )
    per_batch_concurrency: int = field(
        default_factory=lambda: _env_int(
            "SAFETRACE_PER_BATCH_CONCURRENCY",
            int(_profiled_env("SAFETRACE_PER_BATCH_CONCURRENCY", "per_batch_concurrency", "1")),
        )
    )
    cpu_preprocessing_concurrency: int = field(
        default_factory=lambda: _env_int(
            "SAFETRACE_PREPROCESS_CONCURRENCY",
            _env_int(
                "SAFETRACE_CPU_PREPROCESSING_CONCURRENCY",
                int(_profiled_env("SAFETRACE_PREPROCESS_CONCURRENCY", "preprocess_concurrency", "2")),
            ),
        )
    )
    preprocess_concurrency: int = field(
        default_factory=lambda: _env_int(
            "SAFETRACE_PREPROCESS_CONCURRENCY", int(_profiled_env("SAFETRACE_PREPROCESS_CONCURRENCY", "preprocess_concurrency", "2"))
        )
    )
    gpu_detector_concurrency: int = field(
        default_factory=lambda: _env_int(
            "SAFETRACE_GPU_INFERENCE_CONCURRENCY",
            _env_int(
                "SAFETRACE_GPU_DETECTOR_CONCURRENCY",
                int(_profiled_env("SAFETRACE_GPU_INFERENCE_CONCURRENCY", "gpu_inference_concurrency", "1")),
            ),
        )
    )
    gpu_inference_concurrency: int = field(
        default_factory=lambda: _env_int(
            "SAFETRACE_GPU_INFERENCE_CONCURRENCY", int(_profiled_env("SAFETRACE_GPU_INFERENCE_CONCURRENCY", "gpu_inference_concurrency", "1"))
        )
    )
    mobile_sam_concurrency: int = field(
        default_factory=lambda: _env_int(
            "SAFETRACE_MOBILESAM_CONCURRENCY", int(_profiled_env("SAFETRACE_MOBILESAM_CONCURRENCY", "mobile_sam_concurrency", "1"))
        )
    )
    scheduler_policy: str = field(default_factory=lambda: _env("SAFETRACE_SCHEDULER_POLICY", "oldest_batch_first"))
    adaptive_workers_enabled: bool = field(
        default_factory=lambda: _env_bool("SAFETRACE_ADAPTIVE_WORKERS_ENABLED", _runtime_profile_name() == "local_full")
    )
    adaptive_worker_min: int = field(default_factory=lambda: _env_int("SAFETRACE_WORKER_MIN", 2))
    adaptive_worker_max: int = field(default_factory=lambda: _env_int("SAFETRACE_WORKER_MAX", 3))
    # This guard is intentionally opt-in until the real model-backed Phase U
    # comparison establishes that worker three is stable on this device.
    adaptive_third_worker_validated: bool = field(
        default_factory=lambda: _env_bool("SAFETRACE_THIRD_WORKER_VALIDATED", False)
    )
    adaptive_scale_up_queue_depth: int = field(
        default_factory=lambda: _env_int("SAFETRACE_SCALE_UP_QUEUE_DEPTH", 3)
    )
    adaptive_scale_up_sustain_seconds: float = field(
        default_factory=lambda: _env_float("SAFETRACE_SCALE_UP_SUSTAIN_SECONDS", 15.0)
    )
    adaptive_scale_down_idle_seconds: float = field(
        default_factory=lambda: _env_float("SAFETRACE_SCALE_DOWN_IDLE_SECONDS", 60.0)
    )
    adaptive_scale_cooldown_seconds: float = field(
        default_factory=lambda: _env_float("SAFETRACE_SCALE_COOLDOWN_SECONDS", 60.0)
    )
    adaptive_gpu_free_memory_margin_mb: float = field(
        default_factory=lambda: _env_float("SAFETRACE_GPU_FREE_MEMORY_MARGIN_MB", 4096.0)
    )
    adaptive_system_free_memory_margin_mb: float = field(
        default_factory=lambda: _env_float("SAFETRACE_SYSTEM_FREE_MEMORY_MARGIN_MB", 4096.0)
    )
    adaptive_max_recent_error_rate: float = field(
        default_factory=lambda: _env_float("SAFETRACE_ADAPTIVE_MAX_RECENT_ERROR_RATE", 0.2)
    )
    mobile_sam_frame_limit: int = field(default_factory=lambda: _env_int("SAFETRACE_MOBILESAM_FRAME_LIMIT", 5))
    max_queued_jobs: int = field(default_factory=lambda: _env_int("SAFETRACE_MAX_QUEUED_JOBS", 250))
    min_free_disk_mb: float = field(default_factory=lambda: _env_float("SAFETRACE_MIN_FREE_DISK_MB", 2048.0))
    min_available_memory_mb: float = field(
        default_factory=lambda: _env_float("SAFETRACE_MIN_AVAILABLE_MEMORY_MB", 1024.0)
    )
    diagnostic_frames_enabled: bool = field(
        default_factory=lambda: _env_bool("SAFETRACE_DIAGNOSTIC_FRAMES_ENABLED", False)
    )
    mid_vlm_enabled: bool = field(default_factory=lambda: _env_bool("SAFETRACE_MID_VLM_ENABLED", False))
    mid_vlm_model_path: Path = field(
        default_factory=lambda: Path(
            _env("SAFETRACE_MID_VLM_MODEL_PATH", str(Path("models") / "vlm" / "lightweight-512m"))
        )
    )
    mid_vlm_device: str = field(default_factory=lambda: _env("SAFETRACE_MID_VLM_DEVICE", "cuda"))
    mid_vlm_concurrency: int = field(default_factory=lambda: _env_int("SAFETRACE_MID_VLM_CONCURRENCY", 1))
    mid_vlm_max_frames: int = field(default_factory=lambda: _env_int("SAFETRACE_MID_VLM_MAX_FRAMES", 2))
    mid_vlm_timeout_seconds: float = field(
        default_factory=lambda: _env_float("SAFETRACE_MID_VLM_TIMEOUT_SECONDS", 90.0)
    )

    # Comprehensive Review is additive. Fast Local values above remain unchanged.
    comprehensive_review_fps: float = field(
        default_factory=lambda: _env_float("SAFETRACE_COMPREHENSIVE_REVIEW_FPS", 2.0)
    )
    comprehensive_review_max_frames: int = field(
        default_factory=lambda: _env_int("SAFETRACE_COMPREHENSIVE_REVIEW_MAX_FRAMES", 1200)
    )
    comprehensive_review_top_k: int = field(
        default_factory=lambda: _env_int("SAFETRACE_COMPREHENSIVE_REVIEW_TOP_K", 12)
    )
    comprehensive_candidate_window_seconds: float = field(
        default_factory=lambda: _env_float("SAFETRACE_COMPREHENSIVE_CANDIDATE_WINDOW_SECONDS", 3.0)
    )
    comprehensive_max_segmentation_frames: int = field(
        default_factory=lambda: _env_int("SAFETRACE_COMPREHENSIVE_MAX_SEGMENTATION_FRAMES", 8)
    )
    comprehensive_max_vlm_frames: int = field(
        default_factory=lambda: _env_int("SAFETRACE_COMPREHENSIVE_MAX_VLM_FRAMES", 5)
    )

    # ---- Local API hardening ----
    max_upload_mb: float = field(default_factory=lambda: _env_float("SAFETRACE_MAX_UPLOAD_MB", 512.0))
    bulk_max_files: int = field(default_factory=lambda: _env_int("SAFETRACE_BULK_MAX_FILES", 25))
    bulk_max_uncompressed_mb: float = field(
        default_factory=lambda: _env_float("SAFETRACE_BULK_MAX_UNCOMPRESSED_MB", 2048.0)
    )
    bulk_max_compression_ratio: float = field(
        default_factory=lambda: _env_float("SAFETRACE_BULK_MAX_COMPRESSION_RATIO", 100.0)
    )
    job_retention_hours: float = field(default_factory=lambda: _env_float("SAFETRACE_JOB_RETENTION_HOURS", 24.0))
    stale_running_minutes: float = field(default_factory=lambda: _env_float("SAFETRACE_STALE_RUNNING_MINUTES", 30.0))
    recovery_policy: str = field(default_factory=lambda: _env("SAFETRACE_RECOVERY_POLICY", "prompt"))
    completed_retention_days: float = field(default_factory=lambda: _env_float("SAFETRACE_COMPLETED_RETENTION_DAYS", 30.0))
    failed_retention_days: float = field(default_factory=lambda: _env_float("SAFETRACE_FAILED_RETENTION_DAYS", 7.0))
    recovery_retention_hours: float = field(default_factory=lambda: _env_float("SAFETRACE_RECOVERY_RETENTION_HOURS", 72.0))
    cache_retention_days: float = field(default_factory=lambda: _env_float("SAFETRACE_CACHE_RETENTION_DAYS", 14.0))
    cache_max_gb: float = field(default_factory=lambda: _env_float("SAFETRACE_CACHE_MAX_GB", 5.0))
    log_retention_days: float = field(default_factory=lambda: _env_float("SAFETRACE_LOG_RETENTION_DAYS", 14.0))
    min_free_disk_gb: float = field(default_factory=lambda: _env_float("SAFETRACE_MIN_FREE_DISK_GB", 2.0))
    retention_scheduler_enabled: bool = field(default_factory=lambda: _env_bool("SAFETRACE_RETENTION_SCHEDULER_ENABLED", True))
    retention_interval_minutes: float = field(default_factory=lambda: _env_float("SAFETRACE_RETENTION_INTERVAL_MINUTES", 60.0))
    retention_startup_delay_seconds: float = field(default_factory=lambda: _env_float("SAFETRACE_RETENTION_STARTUP_DELAY_SECONDS", 30.0))
    job_retry_max_attempts: int = field(default_factory=lambda: _env_int("SAFETRACE_JOB_RETRY_MAX_ATTEMPTS", 2))
    job_retry_backoff_seconds: float = field(
        default_factory=lambda: _env_float("SAFETRACE_JOB_RETRY_BACKOFF_SECONDS", 5.0)
    )

    # ---- Optional SafeTrace assistant ----
    chat_enabled: str = field(default_factory=lambda: _profiled_env("SAFETRACE_CHAT_ENABLED", "chat_enabled", "auto"))
    chat_provider: str = field(default_factory=lambda: _env("SAFETRACE_CHAT_PROVIDER", "packaged_llamacpp"))
    chat_speed_profile: str = field(default_factory=_chat_speed_profile)
    chat_model_path: Path = field(
        default_factory=lambda: Path(
            _env(
                "SAFETRACE_CHAT_MODEL_PATH",
                str(Path("models") / "chat" / "safetrace-assistant-qwen2.5-1.5b-instruct-q4.gguf"),
            )
        )
    )
    chat_context_window: int = field(default_factory=lambda: _chat_profile_int("SAFETRACE_CHAT_CONTEXT_WINDOW", 4096, 2048))
    chat_max_tokens: int = field(default_factory=lambda: _chat_profile_int("SAFETRACE_CHAT_MAX_TOKENS", 512, 200))
    chat_temperature: float = field(default_factory=lambda: _chat_profile_float("SAFETRACE_CHAT_TEMPERATURE", 0.2, 0.1))
    chat_top_p: float = field(default_factory=lambda: _chat_profile_float("SAFETRACE_CHAT_TOP_P", 0.9, 0.82))
    chat_repeat_penalty: float = field(default_factory=lambda: _env_float("SAFETRACE_CHAT_REPEAT_PENALTY", 1.15))
    chat_autoload: bool = field(default_factory=lambda: _env_bool("SAFETRACE_CHAT_AUTOLOAD", False))
    chat_warmup_on_open: bool = field(default_factory=lambda: _env_bool("SAFETRACE_CHAT_WARMUP_ON_OPEN", False))
    ollama_base_url: str = field(default_factory=lambda: _env("SAFETRACE_OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    ollama_model: str = field(default_factory=lambda: _env("SAFETRACE_OLLAMA_MODEL", "llama3.2:3b"))
    chat_timeout_seconds: float = field(default_factory=lambda: _env_float("SAFETRACE_CHAT_TIMEOUT_SECONDS", 8.0))

    # ---- Detection ----
    yolo_conf_threshold: float = field(default_factory=lambda: _env_float("SAFETRACE_YOLO_CONF", 0.25))
    yolo_iou_threshold: float = field(default_factory=lambda: _env_float("SAFETRACE_YOLO_IOU", 0.45))

    # ---- Rule thresholds ----
    helmet_iou_threshold: float = 0.20      # IoU(head, helmet) below => missing
    hands_wheel_iou_threshold: float = 0.10  # IoU(hand, wheel) below => off-wheel
    phone_hand_iou_threshold: float = 0.30   # IoU(phone, hand) above => phone use
    seatbelt_iou_threshold: float = 0.20     # IoU(seatbelt, torso) below => missing

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.frames_dir, self.checkpoints_dir):
            p.mkdir(parents=True, exist_ok=True)

    def apply_offline_env(self) -> None:
        """Force HuggingFace / transformers into offline mode."""
        if self.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# Class label vocabulary used by the rule engine. Different YOLO checkpoints
# may use different label spellings, so each rule key maps to many possible
# raw class names that we treat as equivalent.
CLASS_ALIASES: Dict[str, set[str]] = {
    "person": {"person"},
    "head": {"head", "face"},
    "helmet": {"helmet", "hard hat", "hardhat", "safety helmet"},
    "hand": {"hand", "hands"},
    "steering_wheel": {"steering wheel", "steering_wheel", "wheel"},
    "phone": {"phone", "cell phone", "cellphone", "mobile phone", "smartphone"},
    "seatbelt": {"seatbelt", "seat belt", "seat_belt", "belt"},
    "torso": {"torso", "upper body", "chest"},
}


def normalize_label(raw: str) -> str | None:
    """Map a raw model label to a canonical SafeTrace label, or None."""
    if not raw:
        return None
    needle = raw.strip().lower()
    for canonical, aliases in CLASS_ALIASES.items():
        if needle in aliases:
            return canonical
    return None


SETTINGS = Settings()
SETTINGS.ensure_dirs()
SETTINGS.apply_offline_env()
