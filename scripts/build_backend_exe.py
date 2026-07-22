"""Prepare or run the SafeTrace backend executable prototype build.

The default mode is a dry run. It prints the PyInstaller command and the
external asset rules without creating an executable or build output.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_ONEFILE_SPEC = Path("packaging") / "backend" / "safetrace_backend.spec"
DEFAULT_ONEDIR_SPEC = Path("packaging") / "backend" / "safetrace_backend_onedir.spec"
DEFAULT_SPEC = DEFAULT_ONEDIR_SPEC
DEFAULT_DIST_DIR = Path("dist") / "backend"
DEFAULT_WORK_DIR = Path("build") / "backend"
BACKEND_EXE_NAME = "safetrace-backend.exe"
BACKEND_ONEDIR_NAME = "safetrace-backend"
EXTERNAL_ASSET_RULES = [
    "config/",
    "models/",
    "models/chat/*.gguf",
    "models/vlm/",
    "data/",
    "logs/",
    "frontend/",
    "*.gguf",
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.bin",
    "*.onnx",
    "checkpoints/",
    "checkpoints/mobile_sam.pt",
    "uploads/",
    "generated/",
    "generated_media/",
    "checkpoints/siglip-base-patch16-224/",
    "checkpoints/yolov8s-seg.pt",
    "checkpoints/yolov9c-seg.pt",
]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def build_command(
    repo_root: Path,
    *,
    spec: Path | None = None,
    dist_dir: Path | None = None,
    work_dir: Path | None = None,
    mode: str = "onedir",
) -> list[str]:
    default_spec = DEFAULT_ONEFILE_SPEC if mode == "onefile" else DEFAULT_ONEDIR_SPEC
    spec_path = resolve_repo_path(repo_root, spec or default_spec)
    dist_path = resolve_repo_path(repo_root, dist_dir or DEFAULT_DIST_DIR)
    work_path = resolve_repo_path(repo_root, work_dir or DEFAULT_WORK_DIR)
    return [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(dist_path),
        "--workpath",
        str(work_path),
        str(spec_path),
    ]


def print_plan(repo_root: Path, command: list[str]) -> None:
    expected_venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    expected_onedir_exe = repo_root / DEFAULT_DIST_DIR / BACKEND_ONEDIR_NAME / BACKEND_EXE_NAME
    expected_onefile_exe = repo_root / DEFAULT_DIST_DIR / BACKEND_EXE_NAME
    print("SafeTrace backend executable prototype")
    print(f"Repository root: {repo_root}")
    print(f"Expected one-dir executable: {expected_onedir_exe}")
    print(f"Legacy one-file executable: {expected_onefile_exe}")
    if expected_venv_python.is_file() and Path(sys.executable).resolve() != expected_venv_python.resolve():
        print(f"WARNING: Build is using {sys.executable}")
        print(f"WARNING: Packaged chat needs llama-cpp-python from {expected_venv_python}")
        print("WARNING: Re-run with .venv\\Scripts\\python.exe scripts\\build_backend_exe.py --run for chat runtime bundling.")
    print("PyInstaller command:")
    print("  " + " ".join(f'"{part}"' if " " in part else part for part in command))
    print("External assets not embedded in the backend executable:")
    for rule in EXTERNAL_ASSET_RULES:
        print(f"  - {rule}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("onedir", "onefile"),
        default="onedir",
        help="PyInstaller backend layout. one-dir is the stable RC default because it avoids one-file temp extraction.",
    )
    parser.add_argument("--spec", type=Path, default=None, help="PyInstaller spec path")
    parser.add_argument("--dist-dir", type=Path, default=DEFAULT_DIST_DIR, help="Generated executable output directory")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="PyInstaller work directory")
    parser.add_argument("--run", action="store_true", help="Actually run PyInstaller; default is a dry run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = repo_root_from_script()
    command = build_command(
        repo_root,
        spec=args.spec,
        dist_dir=args.dist_dir,
        work_dir=args.work_dir,
        mode=args.mode,
    )
    print_plan(repo_root, command)

    if not args.run:
        print("Dry run only. Re-run with --run to create ignored local build output.")
        return 0

    return subprocess.run(command, cwd=repo_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
