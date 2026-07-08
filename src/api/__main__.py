"""Packaging-friendly SafeTrace backend entrypoint."""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from pathlib import Path


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_LOG_LEVEL = "info"


def default_app_root() -> Path:
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()
        if executable.parent.name.lower() == "backend":
            return executable.parent.parent
        return executable.parent
    return Path.cwd()


def load_env_file(path: Path) -> list[str]:
    loaded: list[str] = []
    if not path.is_file():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, value.strip())
        loaded.append(key)
    return loaded


def apply_packaged_defaults(app_root: Path) -> None:
    os.environ.setdefault("SAFETRACE_APP_ROOT", str(app_root))
    os.environ.setdefault("SAFETRACE_PROJECT_ROOT", str(app_root))
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("SAFETRACE_DEVICE", "cpu")
    os.environ.setdefault("SAFETRACE_ANALYSIS_SAFE_MODE", "true")
    os.environ.setdefault("SAFETRACE_CHAT_ENABLED", "auto")
    os.environ.setdefault("SAFETRACE_CHAT_PROVIDER", "packaged_llamacpp")
    os.environ.setdefault("SAFETRACE_CHAT_SPEED_PROFILE", "fast")
    os.environ.setdefault(
        "SAFETRACE_CHAT_MODEL_PATH",
        str(app_root / "models" / "chat" / "safetrace-assistant-qwen2.5-1.5b-instruct-q4.gguf"),
    )
    os.environ.setdefault("SAFETRACE_SERVE_FRONTEND", "true")
    os.environ.setdefault("SAFETRACE_FRONTEND_DIST", str(app_root / "frontend" / "dist"))
    os.environ.setdefault("SAFETRACE_BUILD_MODE", "release-package")
    os.environ.setdefault("SAFETRACE_RUNTIME_LAYOUT", "packaged")
    os.environ.setdefault("SAFETRACE_DATA_DIR", str(app_root / "data"))
    os.environ.setdefault("SAFETRACE_CHECKPOINTS_DIR", str(app_root / "checkpoints"))
    os.environ.setdefault("SAFETRACE_SIGLIP_DIR", str(app_root / "checkpoints" / "siglip-base-patch16-224"))
    os.environ.setdefault("SAFETRACE_YOLO_CKPT", str(app_root / "checkpoints" / "yolov9c-seg.pt"))
    os.environ.setdefault("SAFETRACE_YOLO_FALLBACK_CKPT", str(app_root / "checkpoints" / "yolov8s-seg.pt"))
    os.environ.setdefault("SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM", "false")
    os.environ.setdefault("SAFETRACE_MOBILESAM_ENABLED", "false")
    os.environ.setdefault("SAFETRACE_MOBILESAM_CHECKPOINT", str(app_root / "checkpoints" / "mobile_sam.pt"))
    os.environ.setdefault("SAFETRACE_MOBILESAM_TIMEOUT_SECONDS", "20")
    os.environ.setdefault("SAFETRACE_MOBILESAM_WORKER_ENABLED", "false")
    os.environ.setdefault("SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS", "60")
    os.environ.setdefault("SAFETRACE_VLM_ENABLED", "false")
    os.environ.setdefault("SAFETRACE_VLM_PROVIDER", "auto")
    os.environ.setdefault("SAFETRACE_VLM_PROFILE", "rule_based")
    os.environ.setdefault("SAFETRACE_VLM_MODEL_PATH", str(app_root / "models" / "vlm"))
    os.environ.setdefault("SAFETRACE_VLM_DIR", os.environ["SAFETRACE_VLM_MODEL_PATH"])
    os.environ.setdefault(
        "SAFETRACE_VLM_LIGHTWEIGHT_MODEL_PATH",
        str(app_root / "models" / "vlm" / "lightweight-256m"),
    )
    os.environ.setdefault(
        "SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH",
        str(app_root / "models" / "vlm" / "lightweight-512m"),
    )
    os.environ.setdefault(
        "SAFETRACE_VLM_ENHANCED_MODEL_PATH",
        str(app_root / "models" / "vlm" / "enhanced-2b"),
    )
    os.environ.setdefault(
        "SAFETRACE_VLM_ENHANCED_3B_MODEL_PATH",
        str(app_root / "models" / "vlm" / "enhanced-3b"),
    )
    os.environ.setdefault("SAFETRACE_VLM_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    os.environ.setdefault("SAFETRACE_VLM_MODEL", "local-vlm")
    os.environ.setdefault("SAFETRACE_VLM_TIMEOUT_SECONDS", "10")
    os.environ.setdefault("SAFETRACE_VLM_MAX_FRAMES", "1")
    os.environ.setdefault("SAFETRACE_VLM_MAX_TOKENS", "180")
    os.environ.setdefault("SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED", "false")
    os.environ.setdefault("SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS", "60")
    os.environ.setdefault(
        "SAFETRACE_ALLOWED_ORIGINS",
        "https://safetrace-iota.vercel.app,http://127.0.0.1:5173,http://localhost:5173",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the SafeTrace local backend")
    parser.add_argument("--host", default=os.environ.get("SAFETRACE_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SAFETRACE_PORT", DEFAULT_PORT)))
    parser.add_argument("--log-level", default=os.environ.get("SAFETRACE_LOG_LEVEL", DEFAULT_LOG_LEVEL))
    parser.add_argument("--app-root", type=Path, default=default_app_root())
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--mobile-sam-worker", action="store_true", help="Run a single MobileSAM worker request")
    parser.add_argument("--lightweight-vlm-worker", action="store_true", help="Run a single lightweight VLM worker request")
    parser.add_argument("--input-json", type=Path, default=None, help="Worker request JSON")
    parser.add_argument("--output-json", type=Path, default=None, help="Worker response JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app_root = args.app_root.resolve()
    env_file = args.env_file or app_root / "config" / "safetrace.env"
    load_env_file(env_file)
    apply_packaged_defaults(app_root)

    if args.mobile_sam_worker:
        if args.input_json is None or args.output_json is None:
            print("--mobile-sam-worker requires --input-json and --output-json", file=sys.stderr)
            return 2
        from src.mobile_sam_worker import main as mobile_sam_worker_main

        return mobile_sam_worker_main(
            [
                "--input-json",
                str(args.input_json),
                "--output-json",
                str(args.output_json),
                "--app-root",
                str(app_root),
            ]
        )

    if args.lightweight_vlm_worker:
        if args.input_json is None or args.output_json is None:
            print("--lightweight-vlm-worker requires --input-json and --output-json", file=sys.stderr)
            return 2
        from src.lightweight_vlm_worker import main as lightweight_vlm_worker_main

        return lightweight_vlm_worker_main(
            [
                "--input-json",
                str(args.input_json),
                "--output-json",
                str(args.output_json),
                "--app-root",
                str(app_root),
            ]
        )

    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
