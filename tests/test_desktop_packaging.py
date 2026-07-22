import json
import os
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient

import src.api.server as server_module
from scripts.build_backend_exe import (
    BACKEND_EXE_NAME,
    DEFAULT_DIST_DIR,
    DEFAULT_ONEFILE_SPEC,
    DEFAULT_ONEDIR_SPEC,
    EXTERNAL_ASSET_RULES,
    build_command,
)
from scripts.build_desktop_prototype import (
    DEFAULT_CHAT_MODEL_NAME,
    LAUNCHER_TEXT,
    MAIN_RELEASE_PROFILE_NAME,
    PACKAGE_ASSET_ALLOWLIST,
    PACKAGE_ENV_DEFAULTS,
    PACKAGE_RELEASE_PROFILES,
    PROTECTED_ASSET_RULES,
    AssetValidationError,
    build_prototype,
)
from src.api.batches import BatchStore
from src.api.jobs import JobStore
from src.api.server import create_app


def test_config_example_exists_with_packaged_defaults():
    path = Path("config/safetrace.env.example")

    assert path.is_file()
    content = path.read_text(encoding="utf-8")
    assert "KMP_DUPLICATE_LIB_OK=TRUE" in content
    assert "SAFETRACE_CHAT_PROVIDER=packaged_llamacpp" in content
    assert "SAFETRACE_SERVE_FRONTEND=true" in content
    assert "SAFETRACE_FRONTEND_DIST=frontend/dist" in content
    assert "SAFETRACE_SIGLIP_DIR=checkpoints/siglip-base-patch16-224" in content
    assert "SAFETRACE_YOLO_CKPT=checkpoints/yolov9c-seg.pt" in content
    assert "SAFETRACE_YOLO_FALLBACK_CKPT=checkpoints/yolov8s-seg.pt" in content
    assert "SAFETRACE_DEVICE=cpu" in content
    assert "SAFETRACE_ANALYSIS_SAFE_MODE=true" in content
    assert "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM=false" in content
    assert "SAFETRACE_MOBILESAM_ENABLED=false" in content
    assert "SAFETRACE_MOBILESAM_CHECKPOINT=checkpoints/mobile_sam.pt" in content
    assert "SAFETRACE_MOBILESAM_TIMEOUT_SECONDS=20" in content
    assert "SAFETRACE_MOBILESAM_WORKER_ENABLED=false" in content
    assert "SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS=60" in content
    assert "SAFETRACE_VLM_ENABLED=false" in content
    assert "SAFETRACE_VLM_PROVIDER=auto" in content
    assert "SAFETRACE_VLM_PROFILE=rule_based" in content
    assert "SAFETRACE_VLM_MODEL_PATH=models/vlm" in content
    assert "SAFETRACE_VLM_DIR=models/vlm" in content
    assert "SAFETRACE_VLM_LIGHTWEIGHT_MODEL_PATH=models/vlm/lightweight-256m" in content
    assert "SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH=models/vlm/lightweight-512m" in content
    assert "SAFETRACE_VLM_ENHANCED_MODEL_PATH=models/vlm/enhanced-2b" in content
    assert "SAFETRACE_VLM_ENHANCED_3B_MODEL_PATH=models/vlm/enhanced-3b" in content
    assert "SAFETRACE_VLM_MODEL=local-vlm" in content
    assert "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED=false" in content
    assert "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS=60" in content
    assert "SAFETRACE_ALLOWED_ORIGINS=https://safetrace-iota.vercel.app" in content
    assert PACKAGE_ENV_DEFAULTS["SAFETRACE_SIGLIP_DIR"] == "checkpoints/siglip-base-patch16-224"
    assert PACKAGE_ENV_DEFAULTS["SAFETRACE_YOLO_FALLBACK_CKPT"] == "checkpoints/yolov8s-seg.pt"
    assert PACKAGE_ENV_DEFAULTS["SAFETRACE_CHAT_MODEL_PATH"] == f"models/chat/{DEFAULT_CHAT_MODEL_NAME}"
    assert PACKAGE_ENV_DEFAULTS["SAFETRACE_ANALYSIS_SAFE_MODE"] == "true"
    assert PACKAGE_ENV_DEFAULTS["SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM"] == "false"
    assert PACKAGE_ENV_DEFAULTS["SAFETRACE_MOBILESAM_ENABLED"] == "false"
    assert PACKAGE_ENV_DEFAULTS["SAFETRACE_MOBILESAM_TIMEOUT_SECONDS"] == "20"
    assert PACKAGE_ENV_DEFAULTS["SAFETRACE_MOBILESAM_WORKER_ENABLED"] == "false"
    assert PACKAGE_ENV_DEFAULTS["SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS"] == "60"
    assert PACKAGE_ENV_DEFAULTS["SAFETRACE_VLM_ENABLED"] == "false"
    assert PACKAGE_ENV_DEFAULTS["SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED"] == "false"
    assert PACKAGE_ENV_DEFAULTS["SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS"] == "60"


def test_release_profiles_prepare_safe_mode_main_and_optional_profiles():
    main = PACKAGE_RELEASE_PROFILES[MAIN_RELEASE_PROFILE_NAME]
    rule_based = PACKAGE_RELEASE_PROFILES["SafeTrace_RC_MobileSAM_RuleBased"]
    mobilesam_only = PACKAGE_RELEASE_PROFILES["SafeTrace_RC_MobileSAM_RuleBased_Experimental"]
    worker_profile = PACKAGE_RELEASE_PROFILES["SafeTrace_RC_MobileSAM_Worker_RuleBased_Experimental"]
    experimental = PACKAGE_RELEASE_PROFILES["SafeTrace_RC_MobileSAM_LightweightVLM_Experimental"]
    combined_worker_profile = PACKAGE_RELEASE_PROFILES[
        "SafeTrace_RC_MobileSAM_Worker_LightweightVLM_Worker_Experimental"
    ]
    lightweight_512m = PACKAGE_RELEASE_PROFILES["SafeTrace_RC_Lightweight512M_VLM_Experimental"]
    gpu_vlm = PACKAGE_RELEASE_PROFILES["SafeTrace_RC_GPUVLM_Experimental"]
    enhanced_3b = PACKAGE_RELEASE_PROFILES["SafeTrace_Internal_Enhanced3B_VLM_Experimental"]
    full_lab = PACKAGE_RELEASE_PROFILES["SafeTrace_Internal_Lab_AllModels"]

    assert MAIN_RELEASE_PROFILE_NAME == "SafeTrace_RC_SafeMode_RuleBased"
    assert main["env"]["SAFETRACE_ANALYSIS_SAFE_MODE"] == "true"
    assert main["env"]["SAFETRACE_DEVICE"] == "cpu"
    assert main["env"]["SAFETRACE_MOBILESAM_ENABLED"] == "false"
    assert main["env"]["SAFETRACE_VLM_ENABLED"] == "false"
    assert main["env"]["SAFETRACE_VLM_PROFILE"] == "rule_based"
    assert "Enhanced VLM assets are intentionally excluded." in main["notes"]

    assert rule_based["env"]["SAFETRACE_ANALYSIS_SAFE_MODE"] == "true"
    assert rule_based["env"]["SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM"] == "true"
    assert rule_based["env"]["SAFETRACE_DEVICE"] == "cpu"
    assert rule_based["env"]["SAFETRACE_MOBILESAM_ENABLED"] == "auto"
    assert rule_based["env"]["SAFETRACE_VLM_ENABLED"] == "false"
    assert rule_based["env"]["SAFETRACE_VLM_PROFILE"] == "rule_based"
    assert "Improved frame ranking uses detector/rule evidence instead of SigLIP/FAISS." in rule_based["notes"]

    assert mobilesam_only["env"]["SAFETRACE_ANALYSIS_SAFE_MODE"] == "true"
    assert mobilesam_only["env"]["SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM"] == "true"
    assert mobilesam_only["env"]["SAFETRACE_MOBILESAM_ENABLED"] == "true"
    assert mobilesam_only["env"]["SAFETRACE_MOBILESAM_TIMEOUT_SECONDS"] == "20"
    assert mobilesam_only["env"]["SAFETRACE_VLM_ENABLED"] == "false"
    assert mobilesam_only["env"]["SAFETRACE_VLM_PROFILE"] == "rule_based"
    assert mobilesam_only["env"]["SAFETRACE_BUILD_MODE"] == "SafeTrace_RC_MobileSAM_RuleBased_Experimental"
    assert "Enhanced VLM assets are intentionally excluded." in mobilesam_only["notes"]

    assert worker_profile["env"]["SAFETRACE_ANALYSIS_SAFE_MODE"] == "true"
    assert worker_profile["env"]["SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM"] == "true"
    assert worker_profile["env"]["SAFETRACE_MOBILESAM_ENABLED"] == "true"
    assert worker_profile["env"]["SAFETRACE_MOBILESAM_WORKER_ENABLED"] == "true"
    assert worker_profile["env"]["SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS"] == "60"
    assert worker_profile["env"]["SAFETRACE_VLM_ENABLED"] == "false"
    assert worker_profile["env"]["SAFETRACE_VLM_PROFILE"] == "rule_based"
    assert worker_profile["env"]["SAFETRACE_BUILD_MODE"] == "SafeTrace_RC_MobileSAM_Worker_RuleBased_Experimental"
    assert "separate worker process" in worker_profile["notes"][0]

    assert experimental["env"]["SAFETRACE_ANALYSIS_SAFE_MODE"] == "true"
    assert experimental["env"]["SAFETRACE_DEVICE"] == "cpu"
    assert experimental["env"]["SAFETRACE_MOBILESAM_ENABLED"] == "auto"
    assert experimental["env"]["SAFETRACE_VLM_PROVIDER"] == "auto"
    assert experimental["env"]["SAFETRACE_VLM_PROFILE"] == "lightweight_256m"
    assert "Enhanced VLM assets are excluded from this profile." in experimental["notes"]
    assert PACKAGE_ENV_DEFAULTS["SAFETRACE_VLM_PROFILE"] == "rule_based"

    assert combined_worker_profile["env"]["SAFETRACE_ANALYSIS_SAFE_MODE"] == "true"
    assert combined_worker_profile["env"]["SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM"] == "true"
    assert combined_worker_profile["env"]["SAFETRACE_MOBILESAM_ENABLED"] == "true"
    assert combined_worker_profile["env"]["SAFETRACE_MOBILESAM_WORKER_ENABLED"] == "true"
    assert combined_worker_profile["env"]["SAFETRACE_VLM_ENABLED"] == "true"
    assert combined_worker_profile["env"]["SAFETRACE_VLM_PROFILE"] == "lightweight_256m"
    assert combined_worker_profile["env"]["SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED"] == "true"
    assert combined_worker_profile["env"]["SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS"] == "60"
    assert combined_worker_profile["env"]["SAFETRACE_VLM_FRAME_LIMIT"] == "5"
    assert combined_worker_profile["env"]["SAFETRACE_VLM_MAX_EVIDENCE_FRAMES"] == "5"
    assert combined_worker_profile["env"]["SAFETRACE_VLM_JOB_TIMEOUT_SECONDS"] == "0"
    assert combined_worker_profile["env"]["SAFETRACE_VLM_MAX_TOKENS"] == "64"
    assert "Selected/internal testing only" in combined_worker_profile["notes"][0]
    assert "Enhanced VLM assets are intentionally excluded." in combined_worker_profile["notes"]

    assert lightweight_512m["env"]["SAFETRACE_VLM_ENABLED"] == "true"
    assert lightweight_512m["env"]["SAFETRACE_VLM_PROFILE"] == "lightweight_512m"
    assert lightweight_512m["env"]["SAFETRACE_VLM_MODEL_PATH"] == "models/vlm/lightweight-512m"
    assert lightweight_512m["env"]["SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH"] == "models/vlm/lightweight-512m"
    assert lightweight_512m["env"]["SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED"] == "true"
    assert "Do not include enhanced-3b" in lightweight_512m["notes"][2]

    assert gpu_vlm["env"]["SAFETRACE_DEVICE"] == "auto"
    assert gpu_vlm["env"]["SAFETRACE_ENABLE_GPU_AUTO"] == "true"
    assert gpu_vlm["env"]["SAFETRACE_MOBILESAM_ENABLED"] == "true"
    assert gpu_vlm["env"]["SAFETRACE_MOBILESAM_WORKER_ENABLED"] == "true"
    assert gpu_vlm["env"]["SAFETRACE_VLM_ENABLED"] == "true"
    assert gpu_vlm["env"]["SAFETRACE_ENABLE_VLM"] == "true"
    assert gpu_vlm["env"]["SAFETRACE_VLM_PROFILE"] == "enhanced_2b"
    assert gpu_vlm["env"]["SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED"] == "true"
    assert gpu_vlm["env"]["SAFETRACE_PACKAGE_VLM_PROFILES"] == "lightweight-256m,lightweight-512m,enhanced-2b"
    assert "Experimental/internal tester package" in gpu_vlm["notes"][0]

    assert enhanced_3b["env"]["SAFETRACE_VLM_ENABLED"] == "true"
    assert enhanced_3b["env"]["SAFETRACE_VLM_PROFILE"] == "enhanced_3b"
    assert enhanced_3b["env"]["SAFETRACE_VLM_MODEL_PATH"] == "models/vlm/enhanced-3b"
    assert enhanced_3b["env"]["SAFETRACE_VLM_ENHANCED_3B_MODEL_PATH"] == "models/vlm/enhanced-3b"
    assert "Internal evaluation only" in enhanced_3b["notes"][0]

    assert full_lab["env"]["SAFETRACE_VLM_PROFILE"] == "lightweight_512m"
    assert full_lab["env"]["SAFETRACE_MOBILESAM_WORKER_ENABLED"] == "true"
    assert "Internal lab only" in full_lab["notes"][0]


def test_desktop_manifest_example_shape():
    path = Path("packaging/desktop_packaging_manifest.example.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["component"] == "safetrace-desktop-package"
    assert payload["schema_version"] == 2
    assert payload["release_profile"] == "SafeTrace_RC_SafeMode_RuleBased"
    assert payload["default_runtime"]["safeMode"] is True
    assert payload["default_runtime"]["device"] == "cpu"
    assert payload["default_runtime"]["mobileSamEnabled"] is False
    assert payload["default_runtime"]["vlmEnabled"] is False
    assert payload["default_runtime"]["vlmProfile"] == "rule_based"
    assert payload["release_runtime_layout"]["embeddingModel"] == "checkpoints/siglip-base-patch16-224/"
    assert payload["release_runtime_layout"]["fallbackDetector"] == "checkpoints/yolov8s-seg.pt"
    assert payload["release_runtime_layout"]["primaryDetector"] == "checkpoints/yolov9c-seg.pt"
    assert payload["release_runtime_layout"]["mobileSamCheckpoint"] == "checkpoints/mobile_sam.pt"
    assert payload["release_runtime_layout"]["vlmAssets"] is None
    assert payload["frontend"]["dist_path"] == "frontend/dist"
    assert payload["frontend"]["live_frontend_supported"] is True
    assert payload["backend"]["entrypoint"] == "safetrace-backend/safetrace-backend.exe"
    assert payload["backend"]["legacyEntrypoint"] == "safetrace-backend.exe"
    assert payload["packaged_assets"]["ollamaRequired"] is False
    assert payload["packaged_assets"]["embeddingModel"] == "checkpoints/siglip-base-patch16-224/"
    assert payload["packaged_assets"]["fallbackDetector"] == "checkpoints/yolov8s-seg.pt"
    assert payload["packaged_assets"]["primaryDetector"] == "checkpoints/yolov9c-seg.pt"
    assert payload["packaged_assets"]["chat"].startswith("models/chat/")
    assert payload["packaged_assets"]["vlm"] is None
    assert payload["packaged_assets"]["enhancedVlmPackaged"] is False
    assert "config/" in payload["preserve_paths"]
    assert "checkpoints/" in payload["preserve_paths"]
    assert "*.gguf" in payload["excluded_asset_rules"]
    assert "*.onnx" in payload["excluded_asset_rules"]
    assert "dist/SafeTrace/checkpoints/siglip-base-patch16-224/**" in payload["package_asset_allowlist"]
    assert "dist/SafeTrace/checkpoints/yolov8s-seg.pt" in payload["package_asset_allowlist"]
    assert "dist/SafeTrace/models/vlm/lightweight-256m/**" in payload["package_asset_allowlist"]
    assert "dist/SafeTrace/models/vlm/lightweight-512m/**" in payload["package_asset_allowlist"]
    assert "dist/SafeTrace/models/vlm/enhanced-2b/**" in payload["package_asset_allowlist"]
    assert "dist/SafeTrace/models/vlm/enhanced-3b/**" in payload["package_asset_allowlist"]


def test_package_script_creates_layout_and_excludes_generated_data(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "config").mkdir()
    (repo / "config" / "safetrace.env.example").write_text("SAFETRACE_SERVE_FRONTEND=true\n", encoding="utf-8")
    (repo / "frontend-react" / "dist" / "assets").mkdir(parents=True)
    (repo / "frontend-react" / "dist" / "index.html").write_text("<div>SafeTrace</div>", encoding="utf-8")
    (repo / "frontend-react" / "dist" / "assets" / "index.js").write_text("console.log('ok');", encoding="utf-8")
    (repo / "models" / "chat").mkdir(parents=True)
    (repo / "models" / "chat" / "model.gguf").write_bytes(b"model")
    (repo / "models" / "vlm" / "lightweight-256m").mkdir(parents=True)
    (repo / "models" / "vlm" / "lightweight-256m" / "config.json").write_text("{}", encoding="utf-8")
    (repo / "data").mkdir()
    (repo / "data" / "upload.mp4").write_bytes(b"video")

    summary = build_prototype(repo, tmp_path / "out", clean=True)
    package = Path(summary["package_root"])

    assert (package / "SafeTraceLauncher.bat").is_file()
    assert (package / "backend" / "backend_manifest.json").is_file()
    assert not (package / "backend" / "safetrace-backend.exe").exists()
    assert (package / "frontend" / "dist" / "index.html").is_file()
    assert (package / "config" / "safetrace.env.example").is_file()
    assert (package / "config" / "safetrace.env").is_file()
    assert (package / "config" / "README.txt").is_file()
    assert (package / "models" / "chat").is_dir()
    assert (package / "models" / "chat" / "model.gguf").read_bytes() == b"model"
    assert not (package / "models" / "vlm" / "lightweight-256m").exists()
    assert not (package / "models" / "vlm" / "enhanced-2b").exists()
    assert (package / "checkpoints").is_dir()
    assert (package / "checkpoints" / "README.txt").is_file()
    assert (package / "data").is_dir()
    assert (package / "logs").is_dir()
    assert (package / "packaging_manifest.json").is_file()
    assert (package / "OPTIONAL_ASSETS_REPORT.txt").is_file()
    assert not list((package / "backend").rglob("*.gguf"))
    assert not list(package.rglob("*.pt"))
    assert not (package / "data" / "upload.mp4").exists()
    assert summary["backend_exe_copied"] is False
    assert summary["embedding_model_included"] is False
    assert summary["fallback_detector_included"] is False
    assert summary["primary_detector_included"] is False
    assert summary["mobile_sam_checkpoint_included"] is False
    assert summary["chat_models_included"] == ["model.gguf"]
    assert summary["vlm_assets_included"] is False
    assert summary["broad_vlm_assets_included"] is False
    assert summary["selected_vlm_profile"] is None
    assert summary["selected_vlm_assets_included"] is False
    assert summary["included_vlm_profiles"] == []
    assert "*.gguf" in summary["excluded_asset_rules"]
    assert PROTECTED_ASSET_RULES == summary["excluded_asset_rules"]
    assert PACKAGE_ASSET_ALLOWLIST == summary["package_asset_allowlist"]
    env_content = (package / "config" / "safetrace.env").read_text(encoding="utf-8")
    assert "SAFETRACE_SIGLIP_DIR=checkpoints/siglip-base-patch16-224" in env_content
    assert "SAFETRACE_YOLO_FALLBACK_CKPT=checkpoints/yolov8s-seg.pt" in env_content
    assert "SAFETRACE_ANALYSIS_SAFE_MODE=true" in env_content
    assert "SAFETRACE_DEVICE=cpu" in env_content
    assert "SAFETRACE_MOBILESAM_ENABLED=false" in env_content
    assert "SAFETRACE_MOBILESAM_WORKER_ENABLED=false" in env_content
    assert "SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS=60" in env_content
    assert "SAFETRACE_VLM_ENABLED=false" in env_content
    assert "SAFETRACE_VLM_PROFILE=rule_based" in env_content
    assert "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED=false" in env_content
    assert "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS=60" in env_content
    assert "SAFETRACE_ALLOWED_ORIGINS=https://safetrace-iota.vercel.app" in env_content


def test_package_script_can_generate_mobilesam_worker_profile_config(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    summary = build_prototype(
        repo,
        tmp_path / "out",
        clean=True,
        release_profile="SafeTrace_RC_MobileSAM_Worker_RuleBased_Experimental",
    )
    package = Path(summary["package_root"])
    env_content = (package / "config" / "safetrace.env").read_text(encoding="utf-8")
    manifest = json.loads((package / "packaging_manifest.json").read_text(encoding="utf-8"))

    assert summary["release_profile"] == "SafeTrace_RC_MobileSAM_Worker_RuleBased_Experimental"
    assert "SAFETRACE_ANALYSIS_SAFE_MODE=true" in env_content
    assert "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM=true" in env_content
    assert "SAFETRACE_MOBILESAM_ENABLED=true" in env_content
    assert "SAFETRACE_MOBILESAM_WORKER_ENABLED=true" in env_content
    assert "SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS=60" in env_content
    assert "SAFETRACE_VLM_ENABLED=false" in env_content
    assert "SAFETRACE_VLM_PROFILE=rule_based" in env_content
    assert manifest["release_profile"] == "SafeTrace_RC_MobileSAM_Worker_RuleBased_Experimental"
    assert manifest["default_runtime"]["mobileSamWorkerEnabled"] is True
    assert manifest["default_runtime"]["vlmEnabled"] is False
    assert not (package / "models" / "vlm" / "enhanced-2b").exists()


def test_package_script_can_generate_combined_worker_profile_config(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    summary = build_prototype(
        repo,
        tmp_path / "out",
        clean=True,
        release_profile="SafeTrace_RC_MobileSAM_Worker_LightweightVLM_Worker_Experimental",
    )
    package = Path(summary["package_root"])
    env_content = (package / "config" / "safetrace.env").read_text(encoding="utf-8")
    manifest = json.loads((package / "packaging_manifest.json").read_text(encoding="utf-8"))
    launcher = (package / "SafeTraceLauncher.bat").read_text(encoding="utf-8")

    assert summary["release_profile"] == "SafeTrace_RC_MobileSAM_Worker_LightweightVLM_Worker_Experimental"
    assert "SAFETRACE_ANALYSIS_SAFE_MODE=true" in env_content
    assert "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM=true" in env_content
    assert "SAFETRACE_MOBILESAM_ENABLED=true" in env_content
    assert "SAFETRACE_MOBILESAM_WORKER_ENABLED=true" in env_content
    assert "SAFETRACE_VLM_ENABLED=true" in env_content
    assert "SAFETRACE_VLM_PROFILE=lightweight_256m" in env_content
    assert "SAFETRACE_VLM_MODEL_PATH=models/vlm/lightweight-256m" in env_content
    assert "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED=true" in env_content
    assert "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS=60" in env_content
    assert "SAFETRACE_VLM_FRAME_LIMIT=5" in env_content
    assert "SAFETRACE_VLM_MAX_EVIDENCE_FRAMES=5" in env_content
    assert "SAFETRACE_VLM_JOB_TIMEOUT_SECONDS=0" in env_content
    assert "SAFETRACE_VLM_MAX_TOKENS=64" in env_content
    assert manifest["release_profile"] == "SafeTrace_RC_MobileSAM_Worker_LightweightVLM_Worker_Experimental"
    assert manifest["default_runtime"]["mobileSamWorkerEnabled"] is True
    assert manifest["default_runtime"]["vlmEnabled"] is True
    assert manifest["default_runtime"]["vlmProfile"] == "lightweight_256m"
    assert manifest["default_runtime"]["lightweightVlmWorkerEnabled"] is True
    assert "Lightweight VLM worker: %SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED%" in launcher
    assert not (package / "models" / "vlm" / "enhanced-2b").exists()


def test_package_script_copies_existing_backend_exe_only_when_present(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    backend_exe = repo / DEFAULT_DIST_DIR / BACKEND_EXE_NAME
    backend_exe.parent.mkdir(parents=True)
    backend_exe.write_bytes(b"local exe placeholder")

    summary = build_prototype(repo, tmp_path / "out", clean=True)
    package = Path(summary["package_root"])

    assert summary["backend_exe_copied"] is True
    assert (package / "backend" / "safetrace-backend.exe").read_bytes() == b"local exe placeholder"
    assert not list(package.rglob("*.gguf"))
    assert not list(package.rglob("*.pt"))
    assert not list(package.rglob("*.safetensors"))


def test_package_script_copies_optional_mobilesam_checkpoint_when_present(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    checkpoint = repo / "checkpoints" / "mobile_sam.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"mobile sam checkpoint placeholder")
    (repo / "checkpoints" / "other_model.pt").write_bytes(b"do not copy")

    summary = build_prototype(repo, tmp_path / "out", clean=True)
    package = Path(summary["package_root"])

    assert summary["mobile_sam_checkpoint_included"] is True
    assert (package / "checkpoints" / "mobile_sam.pt").read_bytes() == b"mobile sam checkpoint placeholder"
    assert not (package / "checkpoints" / "other_model.pt").exists()
    assert not list(package.rglob("*.gguf"))


def test_package_script_copies_required_embedding_and_detector_assets_when_present(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    siglip_dir = repo / "checkpoints" / "siglip-base-patch16-224"
    siglip_dir.mkdir(parents=True)
    (siglip_dir / "config.json").write_text("{}", encoding="utf-8")
    (siglip_dir / "model.safetensors").write_bytes(b"siglip weights placeholder")
    fallback_detector = repo / "checkpoints" / "yolov8s-seg.pt"
    fallback_detector.write_bytes(b"yolov8 checkpoint placeholder")
    primary_detector = repo / "checkpoints" / "yolov9c-seg.pt"
    primary_detector.write_bytes(b"yolov9 checkpoint placeholder")

    summary = build_prototype(repo, tmp_path / "out", clean=True)
    package = Path(summary["package_root"])

    assert summary["embedding_model_included"] is True
    assert summary["fallback_detector_included"] is True
    assert summary["primary_detector_included"] is True
    assert (package / "checkpoints" / "siglip-base-patch16-224" / "config.json").is_file()
    assert (package / "checkpoints" / "siglip-base-patch16-224" / "model.safetensors").read_bytes() == (
        b"siglip weights placeholder"
    )
    assert (package / "checkpoints" / "yolov8s-seg.pt").read_bytes() == b"yolov8 checkpoint placeholder"
    assert (package / "checkpoints" / "yolov9c-seg.pt").read_bytes() == b"yolov9 checkpoint placeholder"
    report = (package / "OPTIONAL_ASSETS_REPORT.txt").read_text(encoding="utf-8")
    assert "embedding model: included" in report
    assert "fallback detector checkpoint: included" in report
    assert "primary detector checkpoint: included" in report


def test_package_script_copies_optional_chat_model_when_present(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    model = repo / "models" / "chat" / "assistant.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"chat model placeholder")

    summary = build_prototype(repo, tmp_path / "out", clean=True)
    package = Path(summary["package_root"])

    assert summary["chat_models_included"] == ["assistant.gguf"]
    assert (package / "models" / "chat" / "assistant.gguf").read_bytes() == b"chat model placeholder"
    assert "assistant.gguf" in (package / "OPTIONAL_ASSETS_REPORT.txt").read_text(encoding="utf-8")


def test_package_script_copies_selected_lightweight_512m_assets_when_profile_enabled(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    vlm_dir = repo / "models" / "vlm" / "lightweight-512m"
    enhanced_dir = repo / "models" / "vlm" / "enhanced-2b"
    (vlm_dir / "nested").mkdir(parents=True)
    enhanced_dir.mkdir(parents=True)
    (vlm_dir / "config.json").write_text("{}", encoding="utf-8")
    (vlm_dir / "nested" / "weights.safetensors").write_bytes(b"vlm weights placeholder")
    (enhanced_dir / "model.safetensors").write_bytes(b"enhanced placeholder")

    summary = build_prototype(
        repo,
        tmp_path / "out",
        clean=True,
        release_profile="SafeTrace_RC_Lightweight512M_VLM_Experimental",
    )
    package = Path(summary["package_root"])

    assert summary["vlm_assets_included"] is False
    assert summary["broad_vlm_assets_included"] is False
    assert summary["selected_vlm_profile"] == "lightweight_512m"
    assert summary["selected_vlm_assets_included"] is True
    assert summary["included_vlm_profiles"] == ["lightweight-512m"]
    assert (package / "models" / "vlm" / "lightweight-512m" / "config.json").is_file()
    assert (
        package / "models" / "vlm" / "lightweight-512m" / "nested" / "weights.safetensors"
    ).read_bytes() == b"vlm weights placeholder"
    assert not (package / "models" / "vlm" / "lightweight-256m").exists()
    assert not (package / "models" / "vlm" / "enhanced-2b").exists()
    assert "VLM assets: included" in (package / "OPTIONAL_ASSETS_REPORT.txt").read_text(
        encoding="utf-8"
    )


def test_package_script_copies_gpu_vlm_experimental_assets_when_profile_enabled(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for profile in ("lightweight-256m", "lightweight-512m", "enhanced-2b"):
        model_dir = repo / "models" / "vlm" / profile
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(profile.encode("utf-8"))

    summary = build_prototype(
        repo,
        tmp_path / "out",
        clean=True,
        release_profile="SafeTrace_RC_GPUVLM_Experimental",
    )
    package = Path(summary["package_root"])

    assert summary["vlm_assets_included"] is False
    assert summary["broad_vlm_assets_included"] is False
    assert summary["selected_vlm_profile"] == "enhanced_2b"
    assert summary["selected_vlm_assets_included"] is True
    assert summary["included_vlm_profiles"] == ["lightweight-256m", "lightweight-512m", "enhanced-2b"]
    assert (package / "models" / "vlm" / "lightweight-256m" / "config.json").is_file()
    assert (package / "models" / "vlm" / "lightweight-512m" / "config.json").is_file()
    assert (package / "models" / "vlm" / "enhanced-2b" / "config.json").is_file()
    assert not (package / "models" / "vlm" / "enhanced-3b").exists()
    env_content = (package / "config" / "safetrace.env").read_text(encoding="utf-8")
    assert "SAFETRACE_DEVICE=auto" in env_content
    assert "SAFETRACE_VLM_PROFILE=enhanced_2b" in env_content
    assert "SAFETRACE_PACKAGE_VLM_PROFILES=lightweight-256m,lightweight-512m,enhanced-2b" in env_content


def test_package_script_non_strict_allows_missing_release_assets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    summary = build_prototype(repo, tmp_path / "out", clean=True)

    assert summary["backend_exe_copied"] is False
    assert summary["embedding_model_included"] is False
    assert summary["fallback_detector_included"] is False
    assert summary["primary_detector_included"] is False
    assert summary["mobile_sam_checkpoint_included"] is False
    assert summary["chat_models_included"] == []
    assert summary["vlm_assets_included"] is False
    assert summary["broad_vlm_assets_included"] is False
    assert summary["selected_vlm_assets_included"] is False
    assert summary["included_vlm_profiles"] == []
    assert summary["strict_asset_failures"]


def test_package_script_strict_assets_fails_clearly_when_missing(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    try:
        build_prototype(repo, tmp_path / "out", clean=True, strict_assets=True)
    except AssetValidationError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("strict assets should fail without release assets")

    assert "Backend executable missing" in message
    assert "config/safetrace.env" in message
    assert "checkpoints/siglip-base-patch16-224" in message
    assert "checkpoints/yolov8s-seg.pt" in message
    assert "checkpoints/mobile_sam.pt" in message
    assert "models/chat/*.gguf" in message
    assert "models/vlm/lightweight-256m/" not in message


def test_package_script_strict_assets_passes_with_release_assets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    backend_exe = repo / DEFAULT_DIST_DIR / BACKEND_EXE_NAME
    backend_exe.parent.mkdir(parents=True)
    backend_exe.write_bytes(b"exe")
    (repo / "config").mkdir()
    (repo / "config" / "safetrace.env").write_text(
        "\n".join(["SAFETRACE_CHAT_ENABLED=auto", "SAFETRACE_CHAT_PROVIDER=packaged_llamacpp"]),
        encoding="utf-8",
    )
    mobile_sam = repo / "checkpoints" / "mobile_sam.pt"
    mobile_sam.parent.mkdir(parents=True)
    mobile_sam.write_bytes(b"mobile sam")
    siglip_dir = repo / "checkpoints" / "siglip-base-patch16-224"
    siglip_dir.mkdir()
    (siglip_dir / "config.json").write_text("{}", encoding="utf-8")
    (repo / "checkpoints" / "yolov8s-seg.pt").write_bytes(b"yolov8")
    chat_model = repo / "models" / "chat" / "assistant.gguf"
    chat_model.parent.mkdir(parents=True)
    chat_model.write_bytes(b"chat")
    summary = build_prototype(repo, tmp_path / "out", clean=True, strict_assets=True)
    package = Path(summary["package_root"])

    assert summary["strict_asset_failures"] == []
    assert (package / "backend" / "safetrace-backend.exe").is_file()
    assert (package / "config" / "safetrace.env").is_file()
    assert (package / "checkpoints" / "siglip-base-patch16-224" / "config.json").is_file()
    assert (package / "checkpoints" / "yolov8s-seg.pt").is_file()
    assert not (package / "checkpoints" / "yolov9c-seg.pt").exists()
    assert (package / "checkpoints" / "mobile_sam.pt").is_file()
    assert (package / "models" / "chat" / "assistant.gguf").is_file()
    assert not (package / "models" / "vlm" / "lightweight-256m").exists()
    assert not (package / "models" / "vlm" / "enhanced-2b").exists()
    assert (package / "OPTIONAL_ASSETS_REPORT.txt").is_file()


def test_package_script_copies_onedir_backend_runtime(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    backend_dir = repo / DEFAULT_DIST_DIR / "safetrace-backend"
    backend_dir.mkdir(parents=True)
    (backend_dir / BACKEND_EXE_NAME).write_bytes(b"onedir exe")
    (backend_dir / "_internal").mkdir()
    (backend_dir / "_internal" / "runtime.dll").write_bytes(b"dll")

    summary = build_prototype(repo, tmp_path / "out", clean=True)
    package = Path(summary["package_root"])

    assert summary["backend_exe_copied"] is True
    assert (package / "backend" / "safetrace-backend" / BACKEND_EXE_NAME).read_bytes() == b"onedir exe"
    assert (package / "backend" / "safetrace-backend" / "_internal" / "runtime.dll").read_bytes() == b"dll"
    assert "backend\\safetrace-backend\\safetrace-backend.exe" in (
        package / "SafeTraceLauncher.bat"
    ).read_text(encoding="utf-8")


def test_package_script_strict_assets_checks_selected_512m_candidate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    backend_exe = repo / DEFAULT_DIST_DIR / BACKEND_EXE_NAME
    backend_exe.parent.mkdir(parents=True)
    backend_exe.write_bytes(b"exe")
    (repo / "config").mkdir()
    (repo / "config" / "safetrace.env").write_text("", encoding="utf-8")
    mobile_sam = repo / "checkpoints" / "mobile_sam.pt"
    mobile_sam.parent.mkdir(parents=True)
    mobile_sam.write_bytes(b"mobile sam")
    siglip_dir = repo / "checkpoints" / "siglip-base-patch16-224"
    siglip_dir.mkdir()
    (siglip_dir / "config.json").write_text("{}", encoding="utf-8")
    (repo / "checkpoints" / "yolov8s-seg.pt").write_bytes(b"yolov8")
    chat_model = repo / "models" / "chat" / "assistant.gguf"
    chat_model.parent.mkdir(parents=True)
    chat_model.write_bytes(b"chat")

    try:
        build_prototype(
            repo,
            tmp_path / "out",
            clean=True,
            strict_assets=True,
            release_profile="SafeTrace_RC_Lightweight512M_VLM_Experimental",
        )
    except AssetValidationError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("strict 512M profile should fail without 512M assets")

    assert "models/vlm/lightweight-512m/" in message

    vlm_dir = repo / "models" / "vlm" / "lightweight-512m"
    vlm_dir.mkdir(parents=True)
    (vlm_dir / "config.json").write_text("{}", encoding="utf-8")

    summary = build_prototype(
        repo,
        tmp_path / "out2",
        clean=True,
        strict_assets=True,
        release_profile="SafeTrace_RC_Lightweight512M_VLM_Experimental",
    )
    package = Path(summary["package_root"])

    assert summary["strict_asset_failures"] == []
    assert (package / "models" / "vlm" / "lightweight-512m" / "config.json").is_file()
    assert not (package / "models" / "vlm" / "enhanced-3b").exists()


def test_backend_entrypoint_imports_and_loads_env(monkeypatch, tmp_path):
    from src.api import __main__ as backend_entrypoint

    env_file = tmp_path / "safetrace.env"
    env_file.write_text(
        "\n".join(
            [
                "# local settings",
                "SAFETRACE_HOST=127.0.0.2",
                "SAFETRACE_PORT=8010",
                "IGNORED_LINE",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("SAFETRACE_HOST", raising=False)
    monkeypatch.delenv("SAFETRACE_PORT", raising=False)

    loaded = backend_entrypoint.load_env_file(env_file)
    args = backend_entrypoint.parse_args(["--port", "8011"])

    assert loaded == ["SAFETRACE_HOST", "SAFETRACE_PORT"]
    assert args.port == 8011
    assert backend_entrypoint.DEFAULT_HOST == "127.0.0.1"


def test_backend_entrypoint_frozen_default_app_root_uses_package_parent(monkeypatch, tmp_path):
    from src.api import __main__ as backend_entrypoint

    exe = tmp_path / "SafeTrace" / "backend" / "safetrace-backend.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"exe")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))

    assert backend_entrypoint.default_app_root() == tmp_path / "SafeTrace"


def test_backend_entrypoint_frozen_default_app_root_uses_onedir_package_parent(monkeypatch, tmp_path):
    from src.api import __main__ as backend_entrypoint

    exe = tmp_path / "SafeTrace" / "backend" / "safetrace-backend" / "safetrace-backend.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"exe")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))

    assert backend_entrypoint.default_app_root() == tmp_path / "SafeTrace"


def test_backend_entrypoint_packaged_defaults(monkeypatch, tmp_path):
    from src.api import __main__ as backend_entrypoint

    for key in (
        "SAFETRACE_APP_ROOT",
        "SAFETRACE_PROJECT_ROOT",
        "SAFETRACE_DATA_DIR",
        "SAFETRACE_CHECKPOINTS_DIR",
        "SAFETRACE_DEVICE",
        "SAFETRACE_ANALYSIS_SAFE_MODE",
        "SAFETRACE_SIGLIP_DIR",
        "SAFETRACE_YOLO_CKPT",
        "SAFETRACE_YOLO_FALLBACK_CKPT",
        "SAFETRACE_MOBILESAM_CHECKPOINT",
        "SAFETRACE_MOBILESAM_ENABLED",
        "SAFETRACE_MOBILESAM_TIMEOUT_SECONDS",
        "SAFETRACE_MOBILESAM_WORKER_ENABLED",
        "SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS",
        "SAFETRACE_VLM_ENABLED",
        "SAFETRACE_CHAT_MODEL_PATH",
        "SAFETRACE_VLM_PROVIDER",
        "SAFETRACE_VLM_PROFILE",
        "SAFETRACE_VLM_MODEL_PATH",
        "SAFETRACE_VLM_DIR",
        "SAFETRACE_VLM_LIGHTWEIGHT_MODEL_PATH",
        "SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH",
        "SAFETRACE_VLM_ENHANCED_MODEL_PATH",
        "SAFETRACE_VLM_ENHANCED_3B_MODEL_PATH",
        "SAFETRACE_VLM_MODEL",
        "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED",
        "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS",
        "SAFETRACE_ALLOWED_ORIGINS",
        "SAFETRACE_SERVE_FRONTEND",
        "SAFETRACE_FRONTEND_DIST",
        "SAFETRACE_BUILD_MODE",
        "SAFETRACE_RUNTIME_LAYOUT",
    ):
        monkeypatch.delenv(key, raising=False)

    backend_entrypoint.apply_packaged_defaults(tmp_path)

    assert Path(os.environ["SAFETRACE_APP_ROOT"]) == tmp_path
    assert Path(os.environ["SAFETRACE_PROJECT_ROOT"]) == tmp_path
    assert Path(os.environ["SAFETRACE_DATA_DIR"]) == tmp_path / "data"
    assert Path(os.environ["SAFETRACE_CHECKPOINTS_DIR"]) == tmp_path / "checkpoints"
    assert os.environ["SAFETRACE_DEVICE"] == "cpu"
    assert os.environ["SAFETRACE_ANALYSIS_SAFE_MODE"] == "true"
    assert Path(os.environ["SAFETRACE_SIGLIP_DIR"]) == tmp_path / "checkpoints" / "siglip-base-patch16-224"
    assert Path(os.environ["SAFETRACE_YOLO_CKPT"]) == tmp_path / "checkpoints" / "yolov9c-seg.pt"
    assert Path(os.environ["SAFETRACE_YOLO_FALLBACK_CKPT"]) == tmp_path / "checkpoints" / "yolov8s-seg.pt"
    assert Path(os.environ["SAFETRACE_MOBILESAM_CHECKPOINT"]) == tmp_path / "checkpoints" / "mobile_sam.pt"
    assert os.environ["SAFETRACE_MOBILESAM_ENABLED"] == "false"
    assert os.environ["SAFETRACE_MOBILESAM_TIMEOUT_SECONDS"] == "20"
    assert os.environ["SAFETRACE_MOBILESAM_WORKER_ENABLED"] == "false"
    assert os.environ["SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS"] == "60"
    assert os.environ["SAFETRACE_VLM_ENABLED"] == "false"
    assert Path(os.environ["SAFETRACE_CHAT_MODEL_PATH"]) == (
        tmp_path / "models" / "chat" / "safetrace-assistant-qwen2.5-1.5b-instruct-q4.gguf"
    )
    assert os.environ["SAFETRACE_VLM_PROVIDER"] == "auto"
    assert os.environ["SAFETRACE_VLM_PROFILE"] == "rule_based"
    assert Path(os.environ["SAFETRACE_VLM_MODEL_PATH"]) == tmp_path / "models" / "vlm"
    assert Path(os.environ["SAFETRACE_VLM_DIR"]) == tmp_path / "models" / "vlm"
    assert Path(os.environ["SAFETRACE_VLM_LIGHTWEIGHT_MODEL_PATH"]) == (
        tmp_path / "models" / "vlm" / "lightweight-256m"
    )
    assert Path(os.environ["SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH"]) == (
        tmp_path / "models" / "vlm" / "lightweight-512m"
    )
    assert Path(os.environ["SAFETRACE_VLM_ENHANCED_MODEL_PATH"]) == tmp_path / "models" / "vlm" / "enhanced-2b"
    assert Path(os.environ["SAFETRACE_VLM_ENHANCED_3B_MODEL_PATH"]) == tmp_path / "models" / "vlm" / "enhanced-3b"
    assert os.environ["SAFETRACE_VLM_MODEL"] == "local-vlm"
    assert os.environ["SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED"] == "false"
    assert os.environ["SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS"] == "60"
    assert "https://safetrace-iota.vercel.app" in os.environ["SAFETRACE_ALLOWED_ORIGINS"]
    assert os.environ["SAFETRACE_SERVE_FRONTEND"] == "true"
    assert Path(os.environ["SAFETRACE_FRONTEND_DIST"]) == tmp_path / "frontend" / "dist"
    assert os.environ["SAFETRACE_BUILD_MODE"] == "release-package"
    assert os.environ["SAFETRACE_RUNTIME_LAYOUT"] == "packaged"


def test_backend_entrypoint_main_app_root_overrides_copied_relative_env(monkeypatch, tmp_path):
    from src.api import __main__ as backend_entrypoint

    env_file = tmp_path / "config" / "safetrace.env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text(
        "\n".join(
            [
                "SAFETRACE_APP_ROOT=.",
                "SAFETRACE_PROJECT_ROOT=C:/source/repo",
                "SAFETRACE_VLM_MODEL_PATH=models/vlm",
            ]
        ),
        encoding="utf-8",
    )
    for key in ("SAFETRACE_APP_ROOT", "SAFETRACE_PROJECT_ROOT", "SAFETRACE_VLM_MODEL_PATH"):
        monkeypatch.delenv(key, raising=False)

    calls: list[dict] = []

    def fake_run(app_ref, **kwargs):
        calls.append({"app_ref": app_ref, **kwargs})

    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=fake_run))

    assert backend_entrypoint.main(["--app-root", str(tmp_path), "--env-file", str(env_file), "--port", "8911"]) == 0
    assert Path(os.environ["SAFETRACE_APP_ROOT"]) == tmp_path
    assert Path(os.environ["SAFETRACE_PROJECT_ROOT"]) == tmp_path
    assert os.environ["SAFETRACE_VLM_MODEL_PATH"] == "models/vlm"
    assert calls and calls[0]["port"] == 8911


def test_packaged_relative_paths_prefer_project_root_over_launch_cwd(monkeypatch, tmp_path):
    from src.api import jobs as jobs_module

    app_root = tmp_path / "package" / "SafeTrace"
    source_root = tmp_path / "source"
    packaged_vlm = app_root / "models" / "vlm"
    source_vlm = source_root / "models" / "vlm"
    packaged_vlm.mkdir(parents=True)
    source_vlm.mkdir(parents=True)

    monkeypatch.setattr(server_module.SETTINGS, "project_root", app_root)
    monkeypatch.setattr(jobs_module.SETTINGS, "project_root", app_root)
    monkeypatch.chdir(source_root)

    assert server_module._resolve_configured_path(Path("models/vlm")) == packaged_vlm.resolve()
    assert jobs_module.resolve_configured_path(Path("models/vlm")) == packaged_vlm.resolve()


def test_packaged_launcher_has_foreground_mode_logs_and_health_check():
    assert "--foreground" in LAUNCHER_TEXT
    assert "--app-root \"%APP_ROOT%\"" in LAUNCHER_TEXT
    assert "backend_launcher_stdout.log" in LAUNCHER_TEXT
    assert "backend_launcher_stderr.log" in LAUNCHER_TEXT
    assert "Invoke-WebRequest" in LAUNCHER_TEXT
    assert "HEALTH_OK" in LAUNCHER_TEXT
    assert "Run SafeTraceLauncher.bat --foreground" in LAUNCHER_TEXT
    assert "Last backend stderr lines" in LAUNCHER_TEXT
    assert 'set "HEALTH_WAIT_SECONDS=90"' in LAUNCHER_TEXT
    assert 'set "HEALTH_CHECK_INTERVAL=5"' in LAUNCHER_TEXT
    assert "Waiting for backend health for up to %HEALTH_WAIT_SECONDS% seconds" in LAUNCHER_TEXT
    assert "Still waiting for backend health" in LAUNCHER_TEXT
    assert "Startup diagnostics" in LAUNCHER_TEXT
    assert "Backend process status" in LAUNCHER_TEXT
    assert "Port 8000 occupant" in LAUNCHER_TEXT
    assert "Command used" in LAUNCHER_TEXT
    assert "Get-NetTCPConnection -LocalPort 8000" in LAUNCHER_TEXT
    assert "SAFETRACE_ANALYSIS_SAFE_MODE=true" in LAUNCHER_TEXT
    assert "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM=false" in LAUNCHER_TEXT
    assert "SAFETRACE_DEVICE=cpu" in LAUNCHER_TEXT
    assert "SAFETRACE_MOBILESAM_ENABLED=false" in LAUNCHER_TEXT
    assert "SAFETRACE_MOBILESAM_TIMEOUT_SECONDS=20" in LAUNCHER_TEXT
    assert "SAFETRACE_MOBILESAM_WORKER_ENABLED=false" in LAUNCHER_TEXT
    assert "SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS=60" in LAUNCHER_TEXT
    assert "MobileSAM worker:" in LAUNCHER_TEXT
    assert "SAFETRACE_VLM_ENABLED=false" in LAUNCHER_TEXT
    assert "SAFETRACE_VLM_PROFILE=rule_based" in LAUNCHER_TEXT
    assert "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED=false" in LAUNCHER_TEXT
    assert "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS=60" in LAUNCHER_TEXT
    assert "Lightweight VLM worker:" in LAUNCHER_TEXT
    assert "exit /b 1" in LAUNCHER_TEXT
    assert "SafeTrace Backend Supervisor" in LAUNCHER_TEXT
    assert "backend_supervisor.bat" in LAUNCHER_TEXT
    assert 'set "SUPERVISOR_SCRIPT=%APP_ROOT%\\logs\\backend_supervisor.bat"' in LAUNCHER_TEXT
    assert "Existing backend supervisor detected" in LAUNCHER_TEXT
    assert "No additional backend supervisor was started." in LAUNCHER_TEXT
    assert "$_.ProcessId -ne $self" in LAUNCHER_TEXT
    assert "$_.Name -ieq 'cmd.exe'" in LAUNCHER_TEXT
    assert ":backend_loop" in LAUNCHER_TEXT
    assert "Restarting in 5 seconds" in LAUNCHER_TEXT
    assert 'Start-Sleep -Seconds 5' in LAUNCHER_TEXT
    assert "timeout /t" not in LAUNCHER_TEXT


def test_backend_exe_build_command_is_dry_run_friendly(tmp_path):
    command = build_command(tmp_path)

    assert command[:3] == [sys.executable, "-m", "PyInstaller"]
    assert "--distpath" in command
    assert str(tmp_path / "dist" / "backend") in command
    assert str(tmp_path / DEFAULT_ONEDIR_SPEC) in command

    onefile_command = build_command(tmp_path, mode="onefile")
    assert str(tmp_path / DEFAULT_ONEFILE_SPEC) in onefile_command


def test_backend_exe_plan_keeps_runtime_assets_external_and_llamacpp_hidden_import():
    assert "checkpoints/siglip-base-patch16-224/" in EXTERNAL_ASSET_RULES
    assert "checkpoints/yolov8s-seg.pt" in EXTERNAL_ASSET_RULES
    assert "checkpoints/yolov9c-seg.pt" in EXTERNAL_ASSET_RULES
    spec = Path("packaging/backend/safetrace_backend.spec").read_text(encoding="utf-8")
    assert "collect_dynamic_libs" in spec
    assert 'collect_dynamic_libs("llama_cpp")' in spec
    assert "binaries=llama_cpp_binaries" in spec
    assert '"llama_cpp"' in spec
    assert '"llama_cpp.llama_cpp_ext"' in spec
    assert '"src.mobile_sam_worker"' in spec
    assert '"src.mobile_sam_worker_client"' in spec
    assert '"src.lightweight_vlm_worker"' in spec
    assert '"src.lightweight_vlm_worker_client"' in spec
    assert '"src.mask_encoding"' in spec
    assert '"mobile_sam"' in spec
    onedir_spec = Path("packaging/backend/safetrace_backend_onedir.spec").read_text(encoding="utf-8")
    assert "COLLECT(" in onedir_spec
    assert "exclude_binaries=True" in onedir_spec
    assert 'collect_submodules("transformers.models.smolvlm")' in onedir_spec
    assert '"transformers.models.smolvlm.video_processing_smolvlm"' in onedir_spec


def test_backend_entrypoint_uses_freeze_support_for_pyinstaller_children():
    content = Path("src/api/__main__.py").read_text(encoding="utf-8")

    assert "import multiprocessing" in content
    assert "multiprocessing.freeze_support()" in content


def test_static_frontend_serving_preserves_api_routes(monkeypatch, tmp_path):
    frontend_dist = tmp_path / "frontend" / "dist"
    (frontend_dist / "assets").mkdir(parents=True)
    (frontend_dist / "index.html").write_text("<html><body>SafeTrace app</body></html>", encoding="utf-8")
    (frontend_dist / "assets" / "index.js").write_text("console.log('asset');", encoding="utf-8")

    monkeypatch.setattr(server_module.SETTINGS, "serve_frontend", True)
    monkeypatch.setattr(server_module.SETTINGS, "frontend_dist", frontend_dist)
    client = TestClient(create_app(JobStore(tmp_path / "jobs"), BatchStore(tmp_path / "batches")))

    health = client.get("/api/health")
    index = client.get("/")
    nested = client.get("/review/job_123")
    asset = client.get("/assets/index.js")
    missing_api = client.get("/api/not-a-real-route")

    assert health.status_code == 200
    assert health.json()["api"] == "safetrace-local"
    assert index.status_code == 200
    assert "SafeTrace app" in index.text
    assert nested.status_code == 200
    assert "SafeTrace app" in nested.text
    assert asset.status_code == 200
    assert "asset" in asset.text
    assert missing_api.status_code == 404
    assert "SafeTrace app" not in missing_api.text


def test_system_status_reports_packaged_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("SAFETRACE_BUILD_MODE", "prototype")
    monkeypatch.setenv("SAFETRACE_RUNTIME_LAYOUT", "packaged")
    client = TestClient(create_app(JobStore(tmp_path / "jobs"), BatchStore(tmp_path / "batches")))

    response = client.get("/api/system/status")

    assert response.status_code == 200
    body = response.json()
    assert body["build_mode"] == "prototype"
    assert body["runtime_layout"] == "packaged"
    assert body["runtime"]["frontend"]["serveFrontend"] is False


def test_system_status_defaults_to_source_without_models_or_ollama(monkeypatch, tmp_path):
    monkeypatch.delenv("SAFETRACE_BUILD_MODE", raising=False)
    monkeypatch.delenv("SAFETRACE_RUNTIME_LAYOUT", raising=False)
    monkeypatch.setattr(server_module.SETTINGS, "chat_provider", "packaged_llamacpp")
    monkeypatch.setattr(server_module.SETTINGS, "chat_model_path", tmp_path / "missing.gguf")
    client = TestClient(create_app(JobStore(tmp_path / "jobs"), BatchStore(tmp_path / "batches")))

    response = client.get("/api/system/status")

    assert response.status_code == 200
    body = response.json()
    assert body["build_mode"] == "development"
    assert body["runtime_layout"] == "source"
    assert body["runtime"]["chat"]["provider"] == "packaged_llamacpp"
    assert body["preflight"]["checks"]["assistant"]["details"]["provider"] == "packaged_llamacpp"
