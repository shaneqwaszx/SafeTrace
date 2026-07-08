"""Create a local SafeTrace desktop package prototype.

This script prepares a replaceable-runtime package layout under dist/SafeTrace.
It does not build a final .exe and intentionally excludes local data, uploads,
generated media, reports, and cache folders. It may copy approved local release
assets into generated package output when those assets already exist locally:
SigLIP, YOLO, MobileSAM, the packaged assistant GGUF, and the lightweight
local/non-Ollama VLM profile.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


PACKAGE_DIRNAME = "SafeTrace"
DEFAULT_BACKEND_EXE = Path("dist") / "backend" / "safetrace-backend.exe"
PACKAGED_BACKEND_EXE = Path("backend") / "safetrace-backend.exe"
EMBEDDING_MODEL_SOURCE = Path("checkpoints") / "siglip-base-patch16-224"
EMBEDDING_MODEL_PACKAGE_PATH = Path("checkpoints") / "siglip-base-patch16-224"
FALLBACK_DETECTOR_SOURCE = Path("checkpoints") / "yolov8s-seg.pt"
FALLBACK_DETECTOR_PACKAGE_PATH = Path("checkpoints") / "yolov8s-seg.pt"
PRIMARY_DETECTOR_SOURCE = Path("checkpoints") / "yolov9c-seg.pt"
PRIMARY_DETECTOR_PACKAGE_PATH = Path("checkpoints") / "yolov9c-seg.pt"
MOBILE_SAM_SOURCE = Path("checkpoints") / "mobile_sam.pt"
MOBILE_SAM_PACKAGE_PATH = Path("checkpoints") / "mobile_sam.pt"
CHAT_MODEL_SOURCE_DIR = Path("models") / "chat"
CHAT_MODEL_PATTERN = "*.gguf"
CHAT_MODEL_PACKAGE_DIR = Path("models") / "chat"
DEFAULT_CHAT_MODEL_NAME = "safetrace-assistant-qwen2.5-1.5b-instruct-q4.gguf"
VLM_SOURCE_DIR = Path("models") / "vlm"
VLM_PACKAGE_DIR = Path("models") / "vlm"
VLM_LIGHTWEIGHT_PROFILE = "lightweight-256m"
VLM_LIGHTWEIGHT_512M_PROFILE = "lightweight-512m"
VLM_ENHANCED_PROFILE = "enhanced-2b"
VLM_ENHANCED_3B_PROFILE = "enhanced-3b"
VLM_LIGHTWEIGHT_SOURCE_DIR = VLM_SOURCE_DIR / VLM_LIGHTWEIGHT_PROFILE
VLM_LIGHTWEIGHT_512M_SOURCE_DIR = VLM_SOURCE_DIR / VLM_LIGHTWEIGHT_512M_PROFILE
VLM_LIGHTWEIGHT_PACKAGE_DIR = VLM_PACKAGE_DIR / VLM_LIGHTWEIGHT_PROFILE
VLM_LIGHTWEIGHT_512M_PACKAGE_DIR = VLM_PACKAGE_DIR / VLM_LIGHTWEIGHT_512M_PROFILE
VLM_ENHANCED_PACKAGE_DIR = VLM_PACKAGE_DIR / VLM_ENHANCED_PROFILE
VLM_ENHANCED_3B_PACKAGE_DIR = VLM_PACKAGE_DIR / VLM_ENHANCED_3B_PROFILE
CONFIG_SOURCE = Path("config") / "safetrace.env"
CONFIG_EXAMPLE_SOURCE = Path("config") / "safetrace.env.example"
OPTIONAL_ASSETS_REPORT = "OPTIONAL_ASSETS_REPORT.txt"

PROTECTED_ASSET_RULES = [
    "*.gguf",
    "*.bin",
    "*.safetensors",
    "*.pt",
    "*.pth",
    "*.onnx",
    "checkpoints/",
    "models/chat/*.gguf",
    "models/vlm/",
    "data/",
    "uploads/",
    "generated/",
    "generated_media/",
    "!dist/SafeTrace/checkpoints/siglip-base-patch16-224/**",
    "!dist/SafeTrace/checkpoints/yolov8s-seg.pt",
    "!dist/SafeTrace/checkpoints/yolov9c-seg.pt",
    "!dist/SafeTrace/checkpoints/mobile_sam.pt",
    "!dist/SafeTrace/models/chat/*.gguf",
    "!dist/SafeTrace/models/vlm/lightweight-256m/**",
    "!dist/SafeTrace/models/vlm/lightweight-512m/**",
    "!dist/SafeTrace/models/vlm/enhanced-3b/**",
]
PACKAGE_ASSET_ALLOWLIST = [
    "dist/SafeTrace/checkpoints/siglip-base-patch16-224/**",
    "dist/SafeTrace/checkpoints/yolov8s-seg.pt",
    "dist/SafeTrace/checkpoints/yolov9c-seg.pt",
    "dist/SafeTrace/checkpoints/mobile_sam.pt",
    "dist/SafeTrace/models/chat/*.gguf",
    "dist/SafeTrace/models/vlm/lightweight-256m/**",
    "dist/SafeTrace/models/vlm/lightweight-512m/**",
    "dist/SafeTrace/models/vlm/enhanced-3b/**",
]
PRESERVE_PATHS = ["config/", "data/", "models/", "logs/", "checkpoints/"]
MAIN_RELEASE_PROFILE_NAME = "SafeTrace_RC_SafeMode_RuleBased"
PACKAGE_ENV_DEFAULTS = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
    "SAFETRACE_APP_ROOT": ".",
    "SAFETRACE_DEVICE": "cpu",
    "SAFETRACE_SIGLIP_DIR": "checkpoints/siglip-base-patch16-224",
    "SAFETRACE_YOLO_CKPT": "checkpoints/yolov9c-seg.pt",
    "SAFETRACE_YOLO_FALLBACK_CKPT": "checkpoints/yolov8s-seg.pt",
    "SAFETRACE_ANALYSIS_SAFE_MODE": "true",
    "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM": "false",
    "SAFETRACE_ANALYSIS_JOB_TIMEOUT_SECONDS": "600",
    "SAFETRACE_MOBILESAM_ENABLED": "false",
    "SAFETRACE_MOBILESAM_CHECKPOINT": "checkpoints/mobile_sam.pt",
    "SAFETRACE_MOBILESAM_TIMEOUT_SECONDS": "20",
    "SAFETRACE_MOBILESAM_WORKER_ENABLED": "false",
    "SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS": "60",
    "SAFETRACE_VLM_ENABLED": "false",
    "SAFETRACE_VLM_PROVIDER": "auto",
    "SAFETRACE_VLM_PROFILE": "rule_based",
    "SAFETRACE_VLM_MODEL_PATH": "models/vlm",
    "SAFETRACE_VLM_DIR": "models/vlm",
    "SAFETRACE_VLM_LIGHTWEIGHT_MODEL_PATH": "models/vlm/lightweight-256m",
    "SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH": "models/vlm/lightweight-512m",
    "SAFETRACE_VLM_ENHANCED_MODEL_PATH": "models/vlm/enhanced-2b",
    "SAFETRACE_VLM_ENHANCED_3B_MODEL_PATH": "models/vlm/enhanced-3b",
    "SAFETRACE_VLM_OLLAMA_BASE_URL": "http://127.0.0.1:11434",
    "SAFETRACE_VLM_MODEL": "local-vlm",
    "SAFETRACE_VLM_TIMEOUT_SECONDS": "10",
    "SAFETRACE_VLM_MAX_FRAMES": "5",
    "SAFETRACE_VLM_FRAME_LIMIT": "5",
    "SAFETRACE_VLM_MAX_EVIDENCE_FRAMES": "5",
    "SAFETRACE_VLM_JOB_TIMEOUT_SECONDS": "0",
    "SAFETRACE_VLM_MAX_QUALITY_FAILURES": "1",
    "SAFETRACE_VLM_DISABLE_AFTER_TIMEOUT": "true",
    "SAFETRACE_VLM_MAX_TOKENS": "40",
    "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED": "false",
    "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS": "60",
    "SAFETRACE_CHAT_ENABLED": "auto",
    "SAFETRACE_CHAT_PROVIDER": "packaged_llamacpp",
    "SAFETRACE_CHAT_SPEED_PROFILE": "fast",
    "SAFETRACE_CHAT_MODEL_PATH": f"models/chat/{DEFAULT_CHAT_MODEL_NAME}",
    "SAFETRACE_SERVE_FRONTEND": "true",
    "SAFETRACE_FRONTEND_DIST": "frontend/dist",
    "SAFETRACE_BUILD_MODE": "release-package",
    "SAFETRACE_RUNTIME_LAYOUT": "packaged",
    "SAFETRACE_ALLOWED_ORIGINS": (
        "https://safetrace-iota.vercel.app,http://127.0.0.1:5173,http://localhost:5173"
    ),
}
PACKAGE_RELEASE_PROFILES = {
    MAIN_RELEASE_PROFILE_NAME: {
        "description": "CPU Safe Mode release candidate with rule-based explanations, improved object/rule frame ranking, optional packaged assets disabled by default, and packaged chatbot enabled.",
        "env": {
            "SAFETRACE_ANALYSIS_SAFE_MODE": "true",
            "SAFETRACE_DEVICE": "cpu",
            "SAFETRACE_MOBILESAM_ENABLED": "false",
            "SAFETRACE_VLM_ENABLED": "false",
            "SAFETRACE_VLM_PROFILE": "rule_based",
            "SAFETRACE_CHAT_ENABLED": "auto",
            "SAFETRACE_CHAT_PROVIDER": "packaged_llamacpp",
        },
        "notes": [
            "This is the main tester release profile.",
            "Rule-based visual explanations and improved object/rule frame ranking are enabled.",
            "MobileSAM is copied as an optional package asset but disabled by default.",
            "Lightweight VLM assets may be copied as optional assets but VLM is disabled by default.",
            "Enhanced VLM assets are intentionally excluded.",
        ],
    },
    "SafeTrace_RC_MobileSAM_RuleBased": {
        "description": "CPU Safe Mode with improved object/rule frame ranking, MobileSAM packaged but optional, VLM disabled, chatbot enabled.",
        "env": {
            "SAFETRACE_ANALYSIS_SAFE_MODE": "true",
            "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM": "true",
            "SAFETRACE_DEVICE": "cpu",
            "SAFETRACE_MOBILESAM_ENABLED": "auto",
            "SAFETRACE_VLM_ENABLED": "false",
            "SAFETRACE_VLM_PROFILE": "rule_based",
            "SAFETRACE_CHAT_ENABLED": "auto",
            "SAFETRACE_CHAT_PROVIDER": "packaged_llamacpp",
        },
        "notes": [
            "Rule-based explanations are the default.",
            "Improved frame ranking uses detector/rule evidence instead of SigLIP/FAISS.",
            "MobileSAM remains optional evidence refinement and is not required for frame discovery.",
        ],
    },
    "SafeTrace_RC_MobileSAM_RuleBased_Experimental": {
        "description": "CPU Safe Mode experimental package with object/rule frame ranking, MobileSAM refinement on selected evidence frames only, VLM disabled, chatbot enabled.",
        "env": {
            "SAFETRACE_ANALYSIS_SAFE_MODE": "true",
            "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM": "true",
            "SAFETRACE_DEVICE": "cpu",
            "SAFETRACE_MOBILESAM_ENABLED": "true",
            "SAFETRACE_MOBILESAM_CHECKPOINT": "checkpoints/mobile_sam.pt",
            "SAFETRACE_MOBILESAM_TIMEOUT_SECONDS": "20",
            "SAFETRACE_VLM_ENABLED": "false",
            "SAFETRACE_VLM_PROFILE": "rule_based",
            "SAFETRACE_CHAT_ENABLED": "auto",
            "SAFETRACE_CHAT_PROVIDER": "packaged_llamacpp",
            "SAFETRACE_BUILD_MODE": "SafeTrace_RC_MobileSAM_RuleBased_Experimental",
            "SAFETRACE_RUNTIME_LAYOUT": "packaged-mobilesam-experimental",
        },
        "notes": [
            "MobileSAM runs only after Safe Mode object/rule frame ranking selects evidence frames.",
            "SigLIP/FAISS remain skipped in Safe Mode.",
            "VLM remains disabled; detector-box rule-based fallback remains active.",
            "Enhanced VLM assets are intentionally excluded.",
        ],
    },
    "SafeTrace_RC_MobileSAM_Worker_RuleBased_Experimental": {
        "description": "CPU Safe Mode experimental package with crash-isolated MobileSAM worker refinement, VLM disabled, detector-box fallback, and chatbot enabled.",
        "env": {
            "SAFETRACE_ANALYSIS_SAFE_MODE": "true",
            "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM": "true",
            "SAFETRACE_DEVICE": "cpu",
            "SAFETRACE_MOBILESAM_ENABLED": "true",
            "SAFETRACE_MOBILESAM_CHECKPOINT": "checkpoints/mobile_sam.pt",
            "SAFETRACE_MOBILESAM_TIMEOUT_SECONDS": "20",
            "SAFETRACE_MOBILESAM_WORKER_ENABLED": "true",
            "SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS": "60",
            "SAFETRACE_VLM_ENABLED": "false",
            "SAFETRACE_VLM_PROFILE": "rule_based",
            "SAFETRACE_CHAT_ENABLED": "auto",
            "SAFETRACE_CHAT_PROVIDER": "packaged_llamacpp",
            "SAFETRACE_BUILD_MODE": "SafeTrace_RC_MobileSAM_Worker_RuleBased_Experimental",
            "SAFETRACE_RUNTIME_LAYOUT": "packaged-mobilesam-worker-experimental",
        },
        "notes": [
            "MobileSAM runs in a separate worker process after Safe Mode frame selection.",
            "If the worker times out, crashes, exits non-zero, or returns invalid JSON, SafeTrace uses detector-box fallback.",
            "SigLIP/FAISS remain skipped in Safe Mode.",
            "VLM remains disabled; Enhanced VLM assets are intentionally excluded.",
        ],
    },
    "SafeTrace_RC_MobileSAM_LightweightVLM_Experimental": {
        "description": "CPU Safe Mode profile with improved frame ranking, MobileSAM packaged, optional lightweight VLM, rule-based fallback, chatbot enabled.",
        "env": {
            "SAFETRACE_ANALYSIS_SAFE_MODE": "true",
            "SAFETRACE_DEVICE": "cpu",
            "SAFETRACE_MOBILESAM_ENABLED": "auto",
            "SAFETRACE_VLM_ENABLED": "auto",
            "SAFETRACE_VLM_PROVIDER": "auto",
            "SAFETRACE_VLM_PROFILE": "lightweight_256m",
            "SAFETRACE_VLM_MAX_FRAMES": "5",
            "SAFETRACE_VLM_FRAME_LIMIT": "5",
            "SAFETRACE_VLM_MAX_EVIDENCE_FRAMES": "5",
            "SAFETRACE_VLM_JOB_TIMEOUT_SECONDS": "0",
            "SAFETRACE_VLM_MAX_QUALITY_FAILURES": "1",
            "SAFETRACE_VLM_MAX_TOKENS": "40",
            "SAFETRACE_CHAT_ENABLED": "auto",
            "SAFETRACE_CHAT_PROVIDER": "packaged_llamacpp",
        },
        "notes": [
            "Experimental VLM activation must stay explicit and subprocess/preflight-safe before becoming a main fallback path.",
            "Rule-based fallback remains active when VLM is missing, slow, or suppressed by Safe Mode.",
            "Enhanced VLM assets are excluded from this profile.",
        ],
    },
    "SafeTrace_RC_MobileSAM_Worker_LightweightVLM_Worker_Experimental": {
        "description": "CPU Safe Mode selected-tester package with crash-isolated MobileSAM worker refinement, crash-isolated lightweight VLM worker explanations, and rule-based fallback.",
        "env": {
            "SAFETRACE_ANALYSIS_SAFE_MODE": "true",
            "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM": "true",
            "SAFETRACE_DEVICE": "cpu",
            "SAFETRACE_MOBILESAM_ENABLED": "true",
            "SAFETRACE_MOBILESAM_CHECKPOINT": "checkpoints/mobile_sam.pt",
            "SAFETRACE_MOBILESAM_WORKER_ENABLED": "true",
            "SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS": "60",
            "SAFETRACE_VLM_ENABLED": "true",
            "SAFETRACE_VLM_PROVIDER": "auto",
            "SAFETRACE_VLM_PROFILE": "lightweight_256m",
            "SAFETRACE_VLM_MODEL_PATH": "models/vlm/lightweight-256m",
            "SAFETRACE_VLM_DIR": "models/vlm",
            "SAFETRACE_VLM_LIGHTWEIGHT_MODEL_PATH": "models/vlm/lightweight-256m",
            "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED": "true",
            "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS": "60",
            "SAFETRACE_VLM_MAX_FRAMES": "5",
            "SAFETRACE_VLM_FRAME_LIMIT": "5",
            "SAFETRACE_VLM_MAX_EVIDENCE_FRAMES": "5",
            "SAFETRACE_VLM_JOB_TIMEOUT_SECONDS": "0",
            "SAFETRACE_VLM_MAX_QUALITY_FAILURES": "1",
            "SAFETRACE_VLM_MAX_TOKENS": "64",
            "SAFETRACE_CHAT_ENABLED": "auto",
            "SAFETRACE_CHAT_PROVIDER": "packaged_llamacpp",
            "SAFETRACE_BUILD_MODE": "SafeTrace_RC_MobileSAM_Worker_LightweightVLM_Worker_Experimental",
            "SAFETRACE_RUNTIME_LAYOUT": "packaged-mobilesam-vlm-workers-experimental",
        },
        "notes": [
            "Selected/internal testing only; do not send to general testers.",
            "MobileSAM and Lightweight VLM both run in separate worker processes after Safe Mode frame selection.",
            "SigLIP/FAISS remain skipped in Safe Mode and neither worker controls frame discovery.",
            "Rule-based fallback remains active if either worker fails, times out, exits non-zero, or returns invalid JSON.",
            "Enhanced VLM assets are intentionally excluded.",
        ],
    },
    "SafeTrace_RC_Lightweight512M_VLM_Experimental": {
        "description": "Selected-tester lightweight VLM candidate package profile. Rule-based fallback remains active and only the 512M candidate VLM tier should be included.",
        "env": {
            "SAFETRACE_ANALYSIS_SAFE_MODE": "true",
            "SAFETRACE_DEVICE": "cpu",
            "SAFETRACE_MOBILESAM_ENABLED": "false",
            "SAFETRACE_VLM_ENABLED": "true",
            "SAFETRACE_VLM_PROVIDER": "auto",
            "SAFETRACE_VLM_PROFILE": "lightweight_512m",
            "SAFETRACE_VLM_MODEL_PATH": "models/vlm/lightweight-512m",
            "SAFETRACE_VLM_DIR": "models/vlm",
            "SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH": "models/vlm/lightweight-512m",
            "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED": "true",
            "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS": "60",
            "SAFETRACE_VLM_FRAME_LIMIT": "5",
            "SAFETRACE_VLM_MAX_EVIDENCE_FRAMES": "5",
            "SAFETRACE_VLM_JOB_TIMEOUT_SECONDS": "0",
            "SAFETRACE_VLM_MAX_QUALITY_FAILURES": "1",
            "SAFETRACE_CHAT_ENABLED": "auto",
            "SAFETRACE_CHAT_PROVIDER": "packaged_llamacpp",
            "SAFETRACE_BUILD_MODE": "SafeTrace_RC_Lightweight512M_VLM_Experimental",
            "SAFETRACE_RUNTIME_LAYOUT": "packaged-lightweight-512m-vlm-experimental",
        },
        "notes": [
            "Selected testers only; candidate must pass labelled VLM validation before release.",
            "Package this profile with rule-based fallback and lightweight-512m only.",
            "Do not include enhanced-3b or the legacy enhanced profile in this package.",
        ],
    },
    "SafeTrace_Internal_Enhanced3B_VLM_Experimental": {
        "description": "Internal enhanced VLM replacement candidate profile. Rule-based fallback remains active and only the 3B enhanced tier should be included.",
        "env": {
            "SAFETRACE_ANALYSIS_SAFE_MODE": "false",
            "SAFETRACE_DEVICE": "cpu",
            "SAFETRACE_MOBILESAM_ENABLED": "false",
            "SAFETRACE_VLM_ENABLED": "true",
            "SAFETRACE_VLM_PROVIDER": "auto",
            "SAFETRACE_VLM_PROFILE": "enhanced_3b",
            "SAFETRACE_VLM_MODEL_PATH": "models/vlm/enhanced-3b",
            "SAFETRACE_VLM_DIR": "models/vlm",
            "SAFETRACE_VLM_ENHANCED_3B_MODEL_PATH": "models/vlm/enhanced-3b",
            "SAFETRACE_VLM_FRAME_LIMIT": "5",
            "SAFETRACE_VLM_MAX_EVIDENCE_FRAMES": "5",
            "SAFETRACE_VLM_JOB_TIMEOUT_SECONDS": "0",
            "SAFETRACE_VLM_MAX_QUALITY_FAILURES": "1",
            "SAFETRACE_CHAT_ENABLED": "auto",
            "SAFETRACE_CHAT_PROVIDER": "packaged_llamacpp",
            "SAFETRACE_BUILD_MODE": "SafeTrace_Internal_Enhanced3B_VLM_Experimental",
            "SAFETRACE_RUNTIME_LAYOUT": "packaged-enhanced-3b-vlm-internal",
        },
        "notes": [
            "Internal evaluation only; do not send to general testers.",
            "Package this profile with rule-based fallback and enhanced-3b only.",
            "Do not include lightweight-512m unless a separate full internal lab build is requested.",
        ],
    },
    "SafeTrace_Internal_Lab_AllModels": {
        "description": "Internal lab profile for comparing rule-based, 512M, 3B, chatbot, and MobileSAM assets. Not for general testers.",
        "env": {
            "SAFETRACE_ANALYSIS_SAFE_MODE": "true",
            "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM": "true",
            "SAFETRACE_DEVICE": "cpu",
            "SAFETRACE_MOBILESAM_ENABLED": "true",
            "SAFETRACE_MOBILESAM_WORKER_ENABLED": "true",
            "SAFETRACE_VLM_ENABLED": "true",
            "SAFETRACE_VLM_PROVIDER": "auto",
            "SAFETRACE_VLM_PROFILE": "lightweight_512m",
            "SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH": "models/vlm/lightweight-512m",
            "SAFETRACE_VLM_ENHANCED_3B_MODEL_PATH": "models/vlm/enhanced-3b",
            "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED": "true",
            "SAFETRACE_CHAT_ENABLED": "auto",
            "SAFETRACE_CHAT_PROVIDER": "packaged_llamacpp",
            "SAFETRACE_BUILD_MODE": "SafeTrace_Internal_Lab_AllModels",
            "SAFETRACE_RUNTIME_LAYOUT": "packaged-internal-lab-all-models",
        },
        "notes": [
            "Internal lab only; not a general tester package.",
            "May include 512M, 3B, chatbot, and MobileSAM for controlled comparison.",
            "Do not use this as the default packaging profile.",
        ],
    },
}


LAUNCHER_TEXT = rf"""@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "FOREGROUND="
if /I "%~1"=="--foreground" set "FOREGROUND=1"

set "APP_ROOT=%~dp0"
for %%I in ("%APP_ROOT%.") do set "APP_ROOT=%%~fI"
cd /d "%APP_ROOT%" || exit /b 1
if not exist "logs" mkdir "logs"

set "BACKEND_EXE=backend\safetrace-backend.exe"
set "BACKEND_URL=http://127.0.0.1:8000/api/health"
set "BACKEND_STDOUT=logs\backend_launcher_stdout.log"
set "BACKEND_STDERR=logs\backend_launcher_stderr.log"
set "SUPERVISOR_SCRIPT=%APP_ROOT%\logs\backend_supervisor.bat"
set "HEALTH_WAIT_SECONDS=90"
set "HEALTH_CHECK_INTERVAL=5"
set /a "HEALTH_WAIT_ITERATIONS=%HEALTH_WAIT_SECONDS% / %HEALTH_CHECK_INTERVAL%"
set "BACKEND_COMMAND=%BACKEND_EXE% --app-root %APP_ROOT% --host 127.0.0.1 --port 8000 --log-level info"

if exist "config\safetrace.env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in ("config\safetrace.env") do (
    if not "%%A"=="" set "%%A=%%B"
  )
) else if exist "config\safetrace.env.example" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in ("config\safetrace.env.example") do (
    if not "%%A"=="" set "%%A=%%B"
  )
)

if not defined SAFETRACE_APP_ROOT set "SAFETRACE_APP_ROOT=%APP_ROOT%"
if not defined SAFETRACE_PROJECT_ROOT set "SAFETRACE_PROJECT_ROOT=%APP_ROOT%"
if not defined SAFETRACE_DATA_DIR set "SAFETRACE_DATA_DIR=%APP_ROOT%\data"
if not defined SAFETRACE_CHECKPOINTS_DIR set "SAFETRACE_CHECKPOINTS_DIR=%APP_ROOT%\checkpoints"
if not defined SAFETRACE_SIGLIP_DIR set "SAFETRACE_SIGLIP_DIR=%APP_ROOT%\checkpoints\siglip-base-patch16-224"
if not defined SAFETRACE_YOLO_CKPT set "SAFETRACE_YOLO_CKPT=%APP_ROOT%\checkpoints\yolov9c-seg.pt"
if not defined SAFETRACE_YOLO_FALLBACK_CKPT set "SAFETRACE_YOLO_FALLBACK_CKPT=%APP_ROOT%\checkpoints\yolov8s-seg.pt"
if not defined KMP_DUPLICATE_LIB_OK set "KMP_DUPLICATE_LIB_OK=TRUE"
if not defined OMP_NUM_THREADS set "OMP_NUM_THREADS=1"
if not defined SAFETRACE_DEVICE set "SAFETRACE_DEVICE=cpu"
if not defined SAFETRACE_ANALYSIS_SAFE_MODE set "SAFETRACE_ANALYSIS_SAFE_MODE=true"
if not defined SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM set "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM=false"
if not defined SAFETRACE_CHAT_ENABLED set "SAFETRACE_CHAT_ENABLED=auto"
if not defined SAFETRACE_CHAT_PROVIDER set "SAFETRACE_CHAT_PROVIDER=packaged_llamacpp"
if not defined SAFETRACE_CHAT_SPEED_PROFILE set "SAFETRACE_CHAT_SPEED_PROFILE=fast"
if not defined SAFETRACE_CHAT_MODEL_PATH set "SAFETRACE_CHAT_MODEL_PATH=%APP_ROOT%\models\chat\{DEFAULT_CHAT_MODEL_NAME}"
if not defined SAFETRACE_SERVE_FRONTEND set "SAFETRACE_SERVE_FRONTEND=true"
if not defined SAFETRACE_FRONTEND_DIST set "SAFETRACE_FRONTEND_DIST=%APP_ROOT%\frontend\dist"
if not defined SAFETRACE_BUILD_MODE set "SAFETRACE_BUILD_MODE=release-package"
if not defined SAFETRACE_RUNTIME_LAYOUT set "SAFETRACE_RUNTIME_LAYOUT=packaged"
if not defined SAFETRACE_MOBILESAM_ENABLED set "SAFETRACE_MOBILESAM_ENABLED=false"
if not defined SAFETRACE_MOBILESAM_CHECKPOINT set "SAFETRACE_MOBILESAM_CHECKPOINT=%APP_ROOT%\checkpoints\mobile_sam.pt"
if not defined SAFETRACE_MOBILESAM_TIMEOUT_SECONDS set "SAFETRACE_MOBILESAM_TIMEOUT_SECONDS=20"
if not defined SAFETRACE_MOBILESAM_WORKER_ENABLED set "SAFETRACE_MOBILESAM_WORKER_ENABLED=false"
if not defined SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS set "SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS=60"
if not defined SAFETRACE_VLM_ENABLED set "SAFETRACE_VLM_ENABLED=false"
if not defined SAFETRACE_VLM_PROVIDER set "SAFETRACE_VLM_PROVIDER=auto"
if not defined SAFETRACE_VLM_PROFILE set "SAFETRACE_VLM_PROFILE=rule_based"
if not defined SAFETRACE_VLM_MODEL_PATH set "SAFETRACE_VLM_MODEL_PATH=%APP_ROOT%\models\vlm"
if not defined SAFETRACE_VLM_DIR set "SAFETRACE_VLM_DIR=%SAFETRACE_VLM_MODEL_PATH%"
if not defined SAFETRACE_VLM_LIGHTWEIGHT_MODEL_PATH set "SAFETRACE_VLM_LIGHTWEIGHT_MODEL_PATH=%APP_ROOT%\models\vlm\lightweight-256m"
if not defined SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH set "SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH=%APP_ROOT%\models\vlm\lightweight-512m"
if not defined SAFETRACE_VLM_ENHANCED_MODEL_PATH set "SAFETRACE_VLM_ENHANCED_MODEL_PATH=%APP_ROOT%\models\vlm\enhanced-2b"
if not defined SAFETRACE_VLM_ENHANCED_3B_MODEL_PATH set "SAFETRACE_VLM_ENHANCED_3B_MODEL_PATH=%APP_ROOT%\models\vlm\enhanced-3b"
if not defined SAFETRACE_VLM_OLLAMA_BASE_URL set "SAFETRACE_VLM_OLLAMA_BASE_URL=http://127.0.0.1:11434"
if not defined SAFETRACE_VLM_MODEL set "SAFETRACE_VLM_MODEL=local-vlm"
if not defined SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED set "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED=false"
if not defined SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS set "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS=60"
if not defined SAFETRACE_ALLOWED_ORIGINS set "SAFETRACE_ALLOWED_ORIGINS=https://safetrace-iota.vercel.app,http://127.0.0.1:5173,http://localhost:5173"

echo [SafeTrace] App root: "%APP_ROOT%"
echo [SafeTrace] Backend health: %BACKEND_URL%
echo [SafeTrace] Live frontend may reconnect to this local runtime.
echo [SafeTrace] Runtime layout: %SAFETRACE_RUNTIME_LAYOUT%
echo [SafeTrace] Safe Mode: %SAFETRACE_ANALYSIS_SAFE_MODE%
echo [SafeTrace] Device: %SAFETRACE_DEVICE%
echo [SafeTrace] Embedding model: "%SAFETRACE_SIGLIP_DIR%"
echo [SafeTrace] Detector fallback: "%SAFETRACE_YOLO_FALLBACK_CKPT%"
echo [SafeTrace] MobileSAM enabled: %SAFETRACE_MOBILESAM_ENABLED%
echo [SafeTrace] MobileSAM asset: "%SAFETRACE_MOBILESAM_CHECKPOINT%"
echo [SafeTrace] MobileSAM worker: %SAFETRACE_MOBILESAM_WORKER_ENABLED%
echo [SafeTrace] MobileSAM worker timeout: %SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS%s
echo [SafeTrace] VLM enabled: %SAFETRACE_VLM_ENABLED%
echo [SafeTrace] VLM profile: %SAFETRACE_VLM_PROFILE%
echo [SafeTrace] VLM assets: "%SAFETRACE_VLM_DIR%"
echo [SafeTrace] Lightweight VLM worker: %SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED%
echo [SafeTrace] Lightweight VLM worker timeout: %SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS%s
echo [SafeTrace] Backend stdout: "%APP_ROOT%\%BACKEND_STDOUT%"
echo [SafeTrace] Backend stderr: "%APP_ROOT%\%BACKEND_STDERR%"
echo [SafeTrace] Backend command: "%BACKEND_COMMAND%"
echo.

if not exist "%BACKEND_EXE%" (
  echo [SafeTrace] Prototype package does not include %BACKEND_EXE%.
  echo [SafeTrace] Build or copy the backend executable before release packaging.
  exit /b 1
)

if defined FOREGROUND (
  echo [SafeTrace] Foreground mode enabled. Backend errors will appear in this window.
  "%BACKEND_EXE%" --app-root "%APP_ROOT%" --host 127.0.0.1 --port 8000 --log-level info
  exit /b %errorlevel%
)

type nul > "%BACKEND_STDOUT%"
type nul > "%BACKEND_STDERR%"

echo [SafeTrace] Checking whether a local SafeTrace backend is already healthy...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try {{ $r = Invoke-WebRequest -UseBasicParsing -Uri '%BACKEND_URL%' -TimeoutSec 2; if ($r.StatusCode -eq 200) {{ exit 0 }} }} catch {{ }}; exit 1" >nul 2>nul
if not errorlevel 1 (
  echo [SafeTrace] Backend is already healthy: %BACKEND_URL%
  echo [SafeTrace] No additional backend supervisor was started.
  exit /b 0
)

echo [SafeTrace] Checking for an existing SafeTrace backend supervisor for this package...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try {{ $root = '%APP_ROOT%'; $script = '%SUPERVISOR_SCRIPT%'; $self = $PID; $p = Get-CimInstance Win32_Process | Where-Object {{ $_.ProcessId -ne $self -and $_.Name -ieq 'cmd.exe' -and ($_.CommandLine -like ('*' + $script + '*') -or ($_.CommandLine -like '*backend_supervisor.bat*' -and $_.CommandLine -like ('*' + $root + '*'))) }} | Select-Object -First 1; if ($p) {{ exit 0 }} }} catch {{ }}; exit 1" >nul 2>nul
if not errorlevel 1 (
  echo [SafeTrace] Existing backend supervisor detected for this package. Waiting for health without starting a duplicate supervisor.
) else (
  echo [SafeTrace] Starting backend supervisor in the background...
> "%SUPERVISOR_SCRIPT%" (
  echo @echo off
  echo setlocal
  echo cd /d "%APP_ROOT%" ^|^| exit /b 1
  echo :backend_loop
  echo echo [%%date%% %%time%%] Starting SafeTrace backend. ^>^> "%BACKEND_STDOUT%"
  echo echo [%%date%% %%time%%] Command: "%BACKEND_COMMAND%" ^>^> "%BACKEND_STDOUT%"
  echo "%BACKEND_EXE%" --app-root "%APP_ROOT%" --host 127.0.0.1 --port 8000 --log-level info 1^>^>"%BACKEND_STDOUT%" 2^>^>"%BACKEND_STDERR%"
  echo set "EXIT_CODE=%%errorlevel%%"
  echo echo [%%date%% %%time%%] Backend exited with %%EXIT_CODE%%. Restarting in 5 seconds. ^>^> "%BACKEND_STDERR%"
  echo powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 5" ^>nul 2^>nul
  echo goto backend_loop
)
  start "SafeTrace Backend Supervisor" /D "%APP_ROOT%" cmd /k "%SUPERVISOR_SCRIPT%"
  echo [SafeTrace] Backend supervisor started. Close that window or press Ctrl+C there to stop restarts.
)

echo [SafeTrace] Waiting for backend health for up to %HEALTH_WAIT_SECONDS% seconds...
set "HEALTH_OK="
for /L %%I in (1,1,%HEALTH_WAIT_ITERATIONS%) do (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try {{ $r = Invoke-WebRequest -UseBasicParsing -Uri '%BACKEND_URL%' -TimeoutSec 2; if ($r.StatusCode -eq 200) {{ exit 0 }} }} catch {{ }}; exit 1" >nul 2>nul
  if not errorlevel 1 (
    set "HEALTH_OK=1"
    goto :health_ok
  )
  set /a "WAIT_SECONDS=%%I * %HEALTH_CHECK_INTERVAL%"
  echo [SafeTrace] Still waiting for backend health... !WAIT_SECONDS!/%HEALTH_WAIT_SECONDS% seconds
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds %HEALTH_CHECK_INTERVAL%" >nul 2>nul
)

:health_ok
if defined HEALTH_OK (
  echo [SafeTrace] Backend is healthy: %BACKEND_URL%
  exit /b 0
)

echo [SafeTrace] ERROR: Backend did not become healthy at %BACKEND_URL%.
echo [SafeTrace] Run SafeTraceLauncher.bat --foreground to see startup errors directly.
echo.
echo [SafeTrace] Startup diagnostics:
echo [SafeTrace] App root: "%APP_ROOT%"
echo [SafeTrace] Runtime layout: %SAFETRACE_RUNTIME_LAYOUT%
echo [SafeTrace] Command used: "%BACKEND_COMMAND%"
echo [SafeTrace] Safe Mode: %SAFETRACE_ANALYSIS_SAFE_MODE%
echo [SafeTrace] Device: %SAFETRACE_DEVICE%
echo [SafeTrace] MobileSAM enabled: %SAFETRACE_MOBILESAM_ENABLED%
echo [SafeTrace] MobileSAM worker: %SAFETRACE_MOBILESAM_WORKER_ENABLED%
echo [SafeTrace] MobileSAM worker timeout: %SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS%s
echo [SafeTrace] VLM enabled: %SAFETRACE_VLM_ENABLED%
echo [SafeTrace] VLM profile: %SAFETRACE_VLM_PROFILE%
echo [SafeTrace] Lightweight VLM worker: %SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED%
echo [SafeTrace] Lightweight VLM worker timeout: %SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS%s
echo.
echo [SafeTrace] Backend process status:
powershell -NoProfile -ExecutionPolicy Bypass -Command "try {{ $items = Get-Process -Name 'safetrace-backend' -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, Path; if ($items) {{ $items | Format-Table -AutoSize | Out-String | Write-Host }} else {{ Write-Host 'No safetrace-backend.exe process is currently running.' }} }} catch {{ Write-Host $_.Exception.Message }}"
echo.
echo [SafeTrace] Port 8000 occupant:
powershell -NoProfile -ExecutionPolicy Bypass -Command "try {{ $conns = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object LocalAddress, LocalPort, OwningProcess; if ($conns) {{ $conns | Format-Table -AutoSize | Out-String | Write-Host }} else {{ Write-Host 'No process is listening on port 8000.' }} }} catch {{ Write-Host $_.Exception.Message }}"
echo.
echo [SafeTrace] Last backend stdout lines:
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path '%BACKEND_STDOUT%') {{ Get-Content -Path '%BACKEND_STDOUT%' -Tail 40 }}"
echo.
echo [SafeTrace] Last backend stderr lines:
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path '%BACKEND_STDERR%') {{ Get-Content -Path '%BACKEND_STDERR%' -Tail 40 }}"
exit /b 1
"""


BACKEND_README = """SafeTrace backend runtime placeholder.

Release packaging should place safetrace-backend.exe and runtime dependencies
in this folder. Keep data, logs, uploads, checkpoints, and model files outside
the backend folder so backend updates do not overwrite local assets.
"""


FRONTEND_README = """SafeTrace frontend dist placeholder.

The no-extra-steps release flow normally uses the live website plus local
runtime. This folder is still supported when a packaged local frontend build is
included.

To include frontend assets in a developer package:

  cd frontend-react
  npm.cmd run build
"""


CHECKPOINT_README = """SafeTrace checkpoint folder.

The no-extra-steps release package should include:

  checkpoints/siglip-base-patch16-224/
  checkpoints/yolov8s-seg.pt
  checkpoints/mobile_sam.pt

Optional, when present:

  checkpoints/yolov9c-seg.pt

This generated README appears because one or more local checkpoints were not
found during package generation. Strict release validation requires SigLIP,
yolov8s-seg.pt, and MobileSAM.
"""


CHAT_README = f"""SafeTrace assistant model folder.

The no-extra-steps release package should include the packaged assistant model:

  models/chat/{DEFAULT_CHAT_MODEL_NAME}

This generated README appears because no local GGUF was found during package
generation. Do not commit GGUF files to Git.
"""


VLM_README = """SafeTrace local VLM asset folder.

The no-extra-steps release package should include the lightweight local/non-Ollama
VLM profile:

  models/vlm/lightweight-256m/

SafeTrace defaults to rule-based explanations. Users can explicitly activate
the lightweight VLM profile when local resources allow it. Enhanced VLM assets
are intentionally excluded from this prototype package. Ollama remains an
optional developer/advanced provider only. Do not commit VLM model files or
checkpoint assets to Git.
"""


CONFIG_README = """SafeTrace config folder.

Strict release packages should include:

  config/safetrace.env

This generated README appears because only config/safetrace.env.example was
available during package generation.
"""


def package_env_values(
    existing: dict[str, str] | None = None,
    *,
    release_profile: str = MAIN_RELEASE_PROFILE_NAME,
) -> dict[str, str]:
    profile_env = PACKAGE_RELEASE_PROFILES[release_profile]["env"]
    return {**PACKAGE_ENV_DEFAULTS, **(existing or {}), **profile_env}


def package_env_text(
    existing: dict[str, str] | None = None,
    *,
    release_profile: str = MAIN_RELEASE_PROFILE_NAME,
) -> str:
    values = package_env_values(existing, release_profile=release_profile)
    ordered_keys = list(PACKAGE_ENV_DEFAULTS)
    lines = [f"{key}={values[key]}" for key in ordered_keys]
    extra_keys = sorted(key for key in values if key not in PACKAGE_ENV_DEFAULTS)
    lines.extend(f"{key}={values[key]}" for key in extra_keys)
    return "\n".join(lines) + "\n"


class AssetValidationError(RuntimeError):
    """Raised when --strict-assets validation fails."""

    def __init__(self, failures: list[str]) -> None:
        super().__init__("Strict asset validation failed:\n" + "\n".join(f"- {failure}" for failure in failures))
        self.failures = failures


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def package_root(repo_root: Path, output_dir: Path | None = None) -> Path:
    base = output_dir or repo_root / "dist"
    return base / PACKAGE_DIRNAME


def manifest_payload(*, release_profile: str = MAIN_RELEASE_PROFILE_NAME) -> dict:
    profile = PACKAGE_RELEASE_PROFILES[release_profile]
    values = package_env_values(release_profile=release_profile)
    vlm_package_dir = vlm_asset_package_dir_for_values(values)
    vlm_assets = (
        str(vlm_package_dir).replace("\\", "/") + "/"
        if release_vlm_expected(values)
        else None
    )
    return {
        "component": "safetrace-desktop-package",
        "version": "0.0.0-dev",
        "build_mode": "release-package-prototype",
        "release_profile": release_profile,
        "schema_version": 2,
        "release_runtime_layout": {
            "launcher": "SafeTrace.exe or SafeTraceLauncher.exe",
            "backend": "backend/safetrace-backend.exe",
            "embeddingModel": "checkpoints/siglip-base-patch16-224/",
            "fallbackDetector": "checkpoints/yolov8s-seg.pt",
            "primaryDetector": "checkpoints/yolov9c-seg.pt",
            "mobileSamCheckpoint": "checkpoints/mobile_sam.pt",
            "chatModel": f"models/chat/{DEFAULT_CHAT_MODEL_NAME}",
            "vlmAssets": vlm_assets,
            "config": "config/safetrace.env",
            "data": "data/",
            "logs": "logs/",
        },
        "frontend": {
            "live_frontend_supported": True,
            "dist_path": "frontend/dist",
            "served_by_backend": True,
        },
        "backend": {
            "layout": "backend/",
            "entrypoint": "safetrace-backend.exe",
            "manifest": "backend/backend_manifest.json",
        },
        "packaged_assets": {
            "embeddingModel": str(EMBEDDING_MODEL_PACKAGE_PATH).replace("\\", "/") + "/",
            "fallbackDetector": str(FALLBACK_DETECTOR_PACKAGE_PATH).replace("\\", "/"),
            "primaryDetector": str(PRIMARY_DETECTOR_PACKAGE_PATH).replace("\\", "/"),
            "mobileSam": str(MOBILE_SAM_PACKAGE_PATH).replace("\\", "/"),
            "chat": str(CHAT_MODEL_PACKAGE_DIR / DEFAULT_CHAT_MODEL_NAME).replace("\\", "/"),
            "vlm": vlm_assets,
            "enhancedVlmPackaged": False,
            "ollamaRequired": False,
        },
        "default_runtime": {
            "safeMode": True,
            "device": "cpu",
            "visualExplanations": "rule-based",
            "mobileSamEnabled": profile["env"].get("SAFETRACE_MOBILESAM_ENABLED", "false") not in {"false", "disabled"},
            "mobileSamWorkerEnabled": profile["env"].get("SAFETRACE_MOBILESAM_WORKER_ENABLED", "false") == "true",
            "vlmEnabled": profile["env"].get("SAFETRACE_VLM_ENABLED", "false") not in {"false", "disabled"},
            "vlmProfile": profile["env"].get("SAFETRACE_VLM_PROFILE", "rule_based"),
            "lightweightVlmWorkerEnabled": (
                profile["env"].get("SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED", "false") == "true"
            ),
            "chatProvider": "packaged_llamacpp",
        },
        "preserve_paths": PRESERVE_PATHS,
        "excluded_asset_rules": PROTECTED_ASSET_RULES,
        "package_asset_allowlist": PACKAGE_ASSET_ALLOWLIST,
        "notes": "Generated dist/SafeTrace output and copied model assets must not be committed.",
    }


def backend_manifest_payload() -> dict:
    return {
        "component": "safetrace-backend",
        "version": "0.0.0-dev",
        "build_mode": "release-package-prototype",
        "requires_frontend_version": ">=0.0.0",
        "schema_version": 1,
        "entrypoint": "safetrace-backend.exe",
        "external_assets": {
            "config": "config/safetrace.env",
            "data": "data/",
            "logs": "logs/",
            "embeddingModel": "checkpoints/siglip-base-patch16-224/",
            "fallbackDetector": "checkpoints/yolov8s-seg.pt",
            "primaryDetector": "checkpoints/yolov9c-seg.pt",
            "mobileSam": "checkpoints/mobile_sam.pt",
            "chat": f"models/chat/{DEFAULT_CHAT_MODEL_NAME}",
            "vlm": "models/vlm/",
        },
        "preserve_paths": PRESERVE_PATHS,
        "notes": "Backend executable updates must not overwrite external model, config, data, or log paths.",
    }


def ensure_dirs(root: Path, paths: Iterable[str]) -> list[Path]:
    created = []
    for relative in paths:
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def path_has_contents(path: Path) -> bool:
    if path.is_file():
        return True
    if not path.is_dir():
        return False
    return any(item.is_file() for item in path.rglob("*"))


def read_env_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def disabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "off", "disabled", "none"}


def release_config_values(
    repo_root: Path,
    *,
    release_profile: str = MAIN_RELEASE_PROFILE_NAME,
) -> dict[str, str]:
    values = read_env_values(repo_root / CONFIG_EXAMPLE_SOURCE)
    values.update(read_env_values(repo_root / CONFIG_SOURCE))
    return package_env_values(values, release_profile=release_profile)


def release_chat_expected(values: dict[str, str]) -> bool:
    if disabled(values.get("SAFETRACE_CHAT_ENABLED", "auto")):
        return False
    return values.get("SAFETRACE_CHAT_PROVIDER", "packaged_llamacpp").strip().lower() == "packaged_llamacpp"


def release_vlm_expected(values: dict[str, str]) -> bool:
    if disabled(values.get("SAFETRACE_VLM_ENABLED", "auto")):
        return False
    provider = values.get("SAFETRACE_VLM_PROVIDER", "auto").strip().lower()
    return provider in {"", "auto", "local", "legacy", "existing", "transformers", "local_transformers", "local_dir"}


def vlm_asset_source_for_values(values: dict[str, str]) -> Path:
    profile = values.get("SAFETRACE_VLM_PROFILE", "rule_based").strip().lower()
    if profile == "lightweight_512m":
        return VLM_LIGHTWEIGHT_512M_SOURCE_DIR
    if profile == "enhanced_3b":
        return VLM_SOURCE_DIR / VLM_ENHANCED_3B_PROFILE
    if profile == "enhanced_2b":
        return VLM_SOURCE_DIR / VLM_ENHANCED_PROFILE
    return VLM_LIGHTWEIGHT_SOURCE_DIR


def vlm_asset_package_dir_for_values(values: dict[str, str]) -> Path:
    source = vlm_asset_source_for_values(values)
    return VLM_PACKAGE_DIR / source.name


def source_backend_exe(repo_root: Path, backend_exe: Path | None = None) -> Path:
    source = backend_exe or repo_root / DEFAULT_BACKEND_EXE
    return source if source.is_absolute() else repo_root / source


def chat_model_sources(repo_root: Path) -> list[Path]:
    source_dir = repo_root / CHAT_MODEL_SOURCE_DIR
    if not source_dir.is_dir():
        return []
    return sorted(path for path in source_dir.glob(CHAT_MODEL_PATTERN) if path.is_file())


def strict_asset_failures(
    repo_root: Path,
    backend_exe: Path | None = None,
    *,
    release_profile: str = MAIN_RELEASE_PROFILE_NAME,
) -> list[str]:
    values = release_config_values(repo_root, release_profile=release_profile)
    failures: list[str] = []
    if not source_backend_exe(repo_root, backend_exe).is_file():
        failures.append(f"Backend executable missing at {DEFAULT_BACKEND_EXE}.")
    if not (repo_root / CONFIG_SOURCE).is_file():
        failures.append("Release config missing at config/safetrace.env.")
    if not path_has_contents(repo_root / EMBEDDING_MODEL_SOURCE):
        failures.append("Embedding model missing at checkpoints/siglip-base-patch16-224.")
    if not (repo_root / FALLBACK_DETECTOR_SOURCE).is_file():
        failures.append("Fallback detector checkpoint missing at checkpoints/yolov8s-seg.pt.")
    if not (repo_root / MOBILE_SAM_SOURCE).is_file():
        failures.append("MobileSAM checkpoint missing at checkpoints/mobile_sam.pt.")
    if release_chat_expected(values) and not chat_model_sources(repo_root):
        failures.append("Packaged chat model missing under models/chat/*.gguf.")
    if release_vlm_expected(values):
        vlm_source = vlm_asset_source_for_values(values)
        if not path_has_contents(repo_root / vlm_source):
            failures.append(f"Selected VLM assets missing under {vlm_source.as_posix()}/.")
    return failures


def copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(
            "*.map",
            "data",
            "uploads",
            "generated",
            "generated_media",
            "checkpoints",
            ".pytest_cache",
            "node_modules",
        ),
    )
    return True


def add_asset_report(
    report: list[dict],
    *,
    name: str,
    source: Path,
    target: Path,
    status: str,
    required_in_strict: bool,
    message: str,
) -> None:
    report.append(
        {
            "name": name,
            "source": str(source),
            "target": str(target),
            "status": status,
            "required_in_strict": required_in_strict,
            "message": message,
        }
    )


def copy_config_files(
    repo_root: Path,
    package: Path,
    report: list[dict],
    *,
    release_profile: str = MAIN_RELEASE_PROFILE_NAME,
) -> tuple[bool, bool]:
    env_source = repo_root / CONFIG_SOURCE
    env_example_source = repo_root / CONFIG_EXAMPLE_SOURCE
    env_target = package / CONFIG_SOURCE
    env_example_target = package / CONFIG_EXAMPLE_SOURCE

    copied_env = False
    copied_example = False
    if env_source.is_file():
        env_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(env_source, env_target)
        copied_env = True
    else:
        write_text(env_target, package_env_text(release_profile=release_profile))
        write_text(package / "config" / "README.txt", CONFIG_README)

    if env_example_source.is_file():
        env_example_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(env_example_source, env_example_target)
        copied_example = True

    env_values = read_env_values(env_target)
    write_text(env_target, package_env_text(env_values, release_profile=release_profile))

    add_asset_report(
        report,
        name="release config",
        source=env_source,
        target=env_target,
        status="included" if copied_env else "generated",
        required_in_strict=True,
        message=(
            "Release config copied and completed with packaged path defaults."
            if copied_env
            else "config/safetrace.env missing; generated packaged defaults and copied example only if present."
        ),
    )
    return copied_env, copied_example


def copy_backend_exe_if_exists(
    repo_root: Path,
    package: Path,
    report: list[dict],
    backend_exe: Path | None = None,
) -> bool:
    source = source_backend_exe(repo_root, backend_exe)
    target = package / PACKAGED_BACKEND_EXE
    if not source.is_file():
        add_asset_report(
            report,
            name="backend executable",
            source=source,
            target=target,
            status="missing",
            required_in_strict=True,
            message="Backend executable missing; package contains backend placeholder.",
        )
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    add_asset_report(
        report,
        name="backend executable",
        source=source,
        target=target,
        status="included",
        required_in_strict=True,
        message="Backend executable copied.",
    )
    return True


def copy_embedding_model(repo_root: Path, package: Path, report: list[dict]) -> bool:
    source = repo_root / EMBEDDING_MODEL_SOURCE
    target = package / EMBEDDING_MODEL_PACKAGE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if not path_has_contents(source):
        target.mkdir(parents=True, exist_ok=True)
        write_text(target / "README.txt", "SafeTrace SigLIP embedding model assets belong in this folder.\n")
        add_asset_report(
            report,
            name="embedding model",
            source=source,
            target=target,
            status="missing",
            required_in_strict=True,
            message="SigLIP embedding model missing; rule-based analysis cannot embed/query evidence.",
        )
        return False
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(".git", ".cache", "__pycache__", ".pytest_cache"),
    )
    add_asset_report(
        report,
        name="embedding model",
        source=source,
        target=target,
        status="included",
        required_in_strict=True,
        message="SigLIP embedding model copied.",
    )
    return True


def copy_detector_checkpoint(
    repo_root: Path,
    package: Path,
    report: list[dict],
    *,
    source_relative: Path,
    target_relative: Path,
    name: str,
    required_in_strict: bool,
) -> bool:
    source = repo_root / source_relative
    target = package / target_relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        add_asset_report(
            report,
            name=name,
            source=source,
            target=target,
            status="missing",
            required_in_strict=required_in_strict,
            message=(
                f"{name} missing; strict release validation fails."
                if required_in_strict
                else f"{name} missing; optional primary detector will be skipped."
            ),
        )
        return False
    shutil.copy2(source, target)
    add_asset_report(
        report,
        name=name,
        source=source,
        target=target,
        status="included",
        required_in_strict=required_in_strict,
        message=f"{name} copied.",
    )
    return True


def copy_mobile_sam_checkpoint(repo_root: Path, package: Path, report: list[dict]) -> bool:
    source = repo_root / MOBILE_SAM_SOURCE
    target = package / MOBILE_SAM_PACKAGE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        write_text(package / "checkpoints" / "README.txt", CHECKPOINT_README)
        add_asset_report(
            report,
            name="MobileSAM checkpoint",
            source=source,
            target=target,
            status="missing",
            required_in_strict=True,
            message="MobileSAM checkpoint missing; detector-box fallback remains available.",
        )
        return False
    shutil.copy2(source, target)
    add_asset_report(
        report,
        name="MobileSAM checkpoint",
        source=source,
        target=target,
        status="included",
        required_in_strict=True,
        message="MobileSAM checkpoint copied.",
    )
    return True


def copy_chat_models(repo_root: Path, package: Path, report: list[dict]) -> list[str]:
    sources = chat_model_sources(repo_root)
    target_dir = package / CHAT_MODEL_PACKAGE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    if not sources:
        write_text(target_dir / "README.txt", CHAT_README)
        add_asset_report(
            report,
            name="packaged chat model",
            source=repo_root / CHAT_MODEL_SOURCE_DIR / CHAT_MODEL_PATTERN,
            target=target_dir,
            status="missing",
            required_in_strict=True,
            message="No GGUF chat model found; assistant reports missing-model state.",
        )
        return []

    copied: list[str] = []
    for source in sources:
        target = target_dir / source.name
        shutil.copy2(source, target)
        copied.append(source.name)
    add_asset_report(
        report,
        name="packaged chat model",
        source=repo_root / CHAT_MODEL_SOURCE_DIR,
        target=target_dir,
        status="included",
        required_in_strict=True,
        message=f"Copied {len(copied)} GGUF chat model file(s): {', '.join(copied)}.",
    )
    return copied


def copy_vlm_assets(
    repo_root: Path,
    package: Path,
    report: list[dict],
    *,
    release_profile: str = MAIN_RELEASE_PROFILE_NAME,
) -> dict[str, object]:
    values = release_config_values(repo_root, release_profile=release_profile)
    selected_profile = values.get("SAFETRACE_VLM_PROFILE", "rule_based").strip().lower()
    summary: dict[str, object] = {
        "vlm_assets_included": False,
        "broad_vlm_assets_included": False,
        "selected_vlm_profile": selected_profile if release_vlm_expected(values) else None,
        "selected_vlm_assets_included": False,
        "included_vlm_profiles": [],
    }
    vlm_root = package / VLM_PACKAGE_DIR
    vlm_root.mkdir(parents=True, exist_ok=True)
    if not release_vlm_expected(values):
        write_text(vlm_root / "README.txt", VLM_README)
        add_asset_report(
            report,
            name="VLM assets",
            source=repo_root / VLM_SOURCE_DIR,
            target=vlm_root,
            status="skipped",
            required_in_strict=False,
            message="Selected release profile uses rule-based explanations and does not copy a VLM model tier.",
        )
        return summary

    selected_source = vlm_asset_source_for_values(values)
    selected_package_dir = vlm_asset_package_dir_for_values(values)
    source = repo_root / selected_source
    target = package / selected_package_dir
    if not path_has_contents(source):
        write_text(vlm_root / "README.txt", VLM_README)
        add_asset_report(
            report,
            name="VLM assets",
            source=source,
            target=target,
            status="missing",
            required_in_strict=False,
            message=f"Selected VLM assets missing for {values.get('SAFETRACE_VLM_PROFILE')}; rule-based fallback remains available.",
        )
        return summary

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".cache",
            "data",
            "uploads",
            "generated",
            "generated_media",
        ),
    )
    add_asset_report(
        report,
        name="VLM assets",
        source=source,
        target=target,
        status="included",
        required_in_strict=False,
        message=(
            f"Copied selected VLM asset tier {values.get('SAFETRACE_VLM_PROFILE')}. "
            "Other VLM tiers are not copied by this release profile."
        ),
    )
    summary["selected_vlm_assets_included"] = True
    summary["included_vlm_profiles"] = [selected_source.name]
    return summary


def asset_report_text(summary: dict) -> str:
    lines = [
        "SafeTrace optional/release package asset report",
        f"Package root: {summary['package_root']}",
        "",
        "Assets:",
    ]
    for item in summary["asset_report"]:
        required = "required in strict mode" if item["required_in_strict"] else "optional"
        lines.extend(
            [
                f"- {item['name']}: {item['status']} ({required})",
                f"  source: {item['source']}",
                f"  target: {item['target']}",
                f"  note: {item['message']}",
            ]
        )
    lines.extend(
        [
            "",
            "Ollama required: false",
            "Default VLM provider: auto (local packaged VLM first, optional Ollama only if explicitly configured).",
            "Generated package output and copied model assets must not be committed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_asset_report(package: Path, summary: dict) -> None:
    write_text(package / OPTIONAL_ASSETS_REPORT, asset_report_text(summary))


def build_prototype(
    repo_root: Path,
    output_dir: Path | None = None,
    *,
    clean: bool = False,
    backend_exe: Path | None = None,
    strict_assets: bool = False,
    release_profile: str = MAIN_RELEASE_PROFILE_NAME,
) -> dict:
    repo_root = repo_root.resolve()
    if release_profile not in PACKAGE_RELEASE_PROFILES:
        raise ValueError(f"Unknown release profile: {release_profile}")
    failures = strict_asset_failures(repo_root, backend_exe, release_profile=release_profile)
    if strict_assets and failures:
        raise AssetValidationError(failures)

    package = package_root(repo_root, output_dir).resolve()

    if clean and package.exists():
        shutil.rmtree(package)

    created_dirs = ensure_dirs(
        package,
        [
            "backend",
            "frontend/dist",
            "config",
            "models/chat",
            "models/vlm",
            "checkpoints",
            "data",
            "logs",
        ],
    )

    asset_report: list[dict] = []
    write_text(package / "SafeTraceLauncher.bat", LAUNCHER_TEXT)
    write_text(package / "backend" / "README.txt", BACKEND_README)
    write_json(package / "backend" / "backend_manifest.json", backend_manifest_payload())
    write_json(package / "packaging_manifest.json", manifest_payload(release_profile=release_profile))

    backend_exe_copied = copy_backend_exe_if_exists(repo_root, package, asset_report, backend_exe)
    copied_config, copied_config_example = copy_config_files(
        repo_root,
        package,
        asset_report,
        release_profile=release_profile,
    )
    frontend_copied = copy_if_exists(repo_root / "frontend-react" / "dist", package / "frontend" / "dist")
    if not frontend_copied:
        write_text(package / "frontend" / "dist" / "README.txt", FRONTEND_README)
    embedding_model_included = copy_embedding_model(repo_root, package, asset_report)
    fallback_detector_included = copy_detector_checkpoint(
        repo_root,
        package,
        asset_report,
        source_relative=FALLBACK_DETECTOR_SOURCE,
        target_relative=FALLBACK_DETECTOR_PACKAGE_PATH,
        name="fallback detector checkpoint",
        required_in_strict=True,
    )
    primary_detector_included = copy_detector_checkpoint(
        repo_root,
        package,
        asset_report,
        source_relative=PRIMARY_DETECTOR_SOURCE,
        target_relative=PRIMARY_DETECTOR_PACKAGE_PATH,
        name="primary detector checkpoint",
        required_in_strict=False,
    )
    mobile_sam_checkpoint_included = copy_mobile_sam_checkpoint(repo_root, package, asset_report)
    chat_models_included = copy_chat_models(repo_root, package, asset_report)
    vlm_asset_summary = copy_vlm_assets(repo_root, package, asset_report, release_profile=release_profile)
    vlm_assets_included = bool(vlm_asset_summary.get("vlm_assets_included", False))
    selected_vlm_assets_included = bool(vlm_asset_summary.get("selected_vlm_assets_included", False))

    warnings = [
        "Excluded local data, uploads, generated reports, generated media, and cache folders.",
        "Model/checkpoint assets are copied only into ignored generated package output.",
        "Ollama is optional and is not required for the no-extra-steps release package.",
    ]
    if not backend_exe_copied:
        warnings.append("Backend executable not found; created a placeholder backend folder only.")
    if not copied_config:
        warnings.append("config/safetrace.env was not found; strict release validation will fail.")
    if not copied_config_example:
        warnings.append("config/safetrace.env.example was not found in the source tree.")
    if not frontend_copied:
        warnings.append("frontend-react/dist was not found; created a frontend placeholder instead.")
    if embedding_model_included:
        warnings.append("Embedding model included from checkpoints/siglip-base-patch16-224/.")
    else:
        warnings.append("Embedding model missing; packaged rule-based analysis will not be ready.")
    if fallback_detector_included:
        warnings.append("Fallback detector checkpoint included from checkpoints/yolov8s-seg.pt.")
    else:
        warnings.append("Fallback detector checkpoint missing; packaged rule-based analysis will not be ready.")
    if primary_detector_included:
        warnings.append("Optional primary detector checkpoint included from checkpoints/yolov9c-seg.pt.")
    else:
        warnings.append("Optional primary detector checkpoint missing; SafeTrace will use yolov8s-seg.pt when present.")
    if mobile_sam_checkpoint_included:
        warnings.append("MobileSAM checkpoint included from local checkpoints/mobile_sam.pt.")
    else:
        warnings.append("MobileSAM checkpoint missing; package will use detector-box fallback.")
    if chat_models_included:
        warnings.append(f"Packaged chat model included: {', '.join(chat_models_included)}.")
    else:
        warnings.append("Packaged chat model missing; assistant remains structured but unavailable.")
    values = release_config_values(repo_root, release_profile=release_profile)
    if selected_vlm_assets_included:
        warnings.append(f"VLM assets included from {vlm_asset_source_for_values(values).as_posix()}/.")
    elif release_vlm_expected(values):
        warnings.append("Selected VLM assets missing; visual explanations use rule-based fallback.")
    else:
        warnings.append("VLM assets skipped for this rule-based release profile.")

    summary = {
        "package_root": str(package),
        "release_profile": release_profile,
        "created_dirs": [str(path) for path in created_dirs],
        "backend_exe_copied": backend_exe_copied,
        "frontend_copied": frontend_copied,
        "config_copied": copied_config,
        "config_example_copied": copied_config_example,
        "embedding_model_included": embedding_model_included,
        "fallback_detector_included": fallback_detector_included,
        "primary_detector_included": primary_detector_included,
        "mobile_sam_checkpoint_included": mobile_sam_checkpoint_included,
        "chat_models_included": chat_models_included,
        "vlm_assets_included": vlm_assets_included,
        "broad_vlm_assets_included": bool(vlm_asset_summary.get("broad_vlm_assets_included", False)),
        "selected_vlm_profile": vlm_asset_summary.get("selected_vlm_profile"),
        "selected_vlm_assets_included": selected_vlm_assets_included,
        "included_vlm_profiles": list(vlm_asset_summary.get("included_vlm_profiles", [])),
        "preserve_paths": PRESERVE_PATHS,
        "excluded_asset_rules": PROTECTED_ASSET_RULES,
        "package_asset_allowlist": PACKAGE_ASSET_ALLOWLIST,
        "asset_report": asset_report,
        "asset_report_path": str(package / OPTIONAL_ASSETS_REPORT),
        "strict_asset_failures": failures,
        "warnings": warnings,
    }
    write_asset_report(package, summary)
    return summary


def print_summary(summary: dict) -> None:
    print(f"SafeTrace desktop prototype: {summary['package_root']}")
    print(f"Release profile: {summary.get('release_profile', MAIN_RELEASE_PROFILE_NAME)}")
    print("Created package folders:")
    for path in summary["created_dirs"]:
        print(f"  - {path}")
    print(f"Backend executable copied: {summary['backend_exe_copied']}")
    print(f"Frontend dist copied: {summary['frontend_copied']}")
    print(f"Release config copied: {summary['config_copied']}")
    print(f"Config example copied: {summary['config_example_copied']}")
    print(f"Embedding model included: {summary['embedding_model_included']}")
    print(f"Fallback detector included: {summary['fallback_detector_included']}")
    print(f"Primary detector included: {summary['primary_detector_included']}")
    print(f"MobileSAM checkpoint included: {summary['mobile_sam_checkpoint_included']}")
    print(f"Chat models included: {len(summary['chat_models_included'])}")
    print(f"Broad VLM assets included: {summary['vlm_assets_included']}")
    print(f"Selected VLM profile: {summary.get('selected_vlm_profile') or 'none'}")
    print(f"Selected VLM assets included: {summary.get('selected_vlm_assets_included', False)}")
    print(f"Asset report: {summary['asset_report_path']}")
    print("Preserved external paths:")
    for path in summary["preserve_paths"]:
        print(f"  - {path}")
    print("Intentional source-control exclusions:")
    for rule in summary["excluded_asset_rules"]:
        print(f"  - {rule}")
    print("Generated package asset allowlist:")
    for rule in summary["package_asset_allowlist"]:
        print(f"  - {rule}")
    print("Asset report:")
    print(asset_report_text(summary), end="")
    print("Warnings:")
    for warning in summary["warnings"]:
        print(f"  - {warning}")


def print_dry_run(
    repo_root: Path,
    package: Path,
    backend_exe: Path | None,
    strict_assets: bool,
    *,
    release_profile: str = MAIN_RELEASE_PROFILE_NAME,
) -> int:
    backend_source = source_backend_exe(repo_root, backend_exe)
    values = release_config_values(repo_root, release_profile=release_profile)
    failures = strict_asset_failures(repo_root, backend_exe, release_profile=release_profile)
    print(f"Would create SafeTrace desktop prototype at: {package}")
    print(f"Would use release profile: {release_profile}")
    print(f"Would copy backend exe if present: {backend_source}")
    print(f"Would copy release config if present: {repo_root / CONFIG_SOURCE}")
    print(f"Would ensure packaged config defaults include: {', '.join(PACKAGE_ENV_DEFAULTS)}")
    print(f"Would copy embedding model if present: {repo_root / EMBEDDING_MODEL_SOURCE}")
    print(f"Would copy fallback detector if present: {repo_root / FALLBACK_DETECTOR_SOURCE}")
    print(f"Would copy optional primary detector if present: {repo_root / PRIMARY_DETECTOR_SOURCE}")
    print(f"Would copy MobileSAM checkpoint if present: {repo_root / MOBILE_SAM_SOURCE}")
    print(f"Would copy chat GGUF models if present: {repo_root / CHAT_MODEL_SOURCE_DIR / CHAT_MODEL_PATTERN}")
    print(f"Would copy selected VLM assets if enabled: {repo_root / vlm_asset_source_for_values(values)}")
    print(f"Chat expected in strict mode: {release_chat_expected(values)}")
    print(f"Local VLM expected in strict mode: {release_vlm_expected(values)}")
    print("Would write package asset report: OPTIONAL_ASSETS_REPORT.txt")
    print("Would exclude from source control:")
    for rule in PROTECTED_ASSET_RULES:
        print(f"  - {rule}")
    print("Would allow inside ignored generated package output:")
    for rule in PACKAGE_ASSET_ALLOWLIST:
        print(f"  - {rule}")
    if failures:
        print("Strict asset validation failures:")
        for failure in failures:
            print(f"  - {failure}")
    elif strict_assets:
        print("Strict asset validation would pass.")
    return 2 if strict_assets and failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="Base output directory; default is ./dist")
    parser.add_argument(
        "--backend-exe",
        type=Path,
        default=None,
        help="Optional path to an already-built safetrace-backend.exe to copy into the package",
    )
    parser.add_argument("--clean", action="store_true", help="Remove existing dist/SafeTrace before creating it")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned package path and exclusions only")
    parser.add_argument(
        "--strict-assets",
        action="store_true",
        help="Fail when release package assets such as backend exe, config, MobileSAM, chat, or VLM are missing",
    )
    parser.add_argument(
        "--release-profile",
        choices=sorted(PACKAGE_RELEASE_PROFILES),
        default=MAIN_RELEASE_PROFILE_NAME,
        help="Runtime profile to apply to generated config; default is the stable Safe Mode RC profile.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_script()
    package = package_root(repo_root, args.output_dir)
    if args.dry_run:
        return print_dry_run(
            repo_root,
            package,
            args.backend_exe,
            args.strict_assets,
            release_profile=args.release_profile,
        )
    try:
        summary = build_prototype(
            repo_root,
            args.output_dir,
            clean=args.clean,
            backend_exe=args.backend_exe,
            strict_assets=args.strict_assets,
            release_profile=args.release_profile,
        )
    except AssetValidationError as exc:
        print(str(exc))
        return 2
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
