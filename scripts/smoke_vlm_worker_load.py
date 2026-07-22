"""Run one SafeTrace VLM worker request without starting the API server.

This smoke helper works with the source worker by default and can exercise a
one-dir packaged backend with ``--backend-exe``.  It reports model loading and
generation separately so a quality-gate rejection is not confused with a
packaging/import failure.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def infer_profile(model_dir: Path) -> str:
    name = model_dir.name.strip().lower()
    if name == "lightweight-256m":
        return "lightweight_256m"
    if name == "enhanced-2b":
        return "enhanced_2b"
    return "lightweight_512m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--app-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional writable directory for transient worker request/result files.",
    )
    parser.add_argument("--backend-exe", type=Path, default=None, help="Optional packaged safetrace-backend.exe")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep the generated request/result directory for inspection.")
    return parser.parse_args()


def effective_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _command(args: argparse.Namespace, input_path: Path, output_path: Path) -> list[str]:
    common = [
        "--input-json",
        str(input_path),
        "--output-json",
        str(output_path),
        "--app-root",
        str(args.app_root.resolve()),
    ]
    if args.backend_exe:
        return [str(args.backend_exe.resolve()), "--lightweight-vlm-worker", *common]
    return [sys.executable, "-m", "src.lightweight_vlm_worker", *common]


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    image = args.image.resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_dir}")
    if not image.is_file():
        raise SystemExit(f"Image does not exist: {image}")

    profile = args.profile or infer_profile(model_dir)
    device = effective_device(args.device)
    request: dict[str, Any] = {
        "imagePath": str(image),
        "modelDir": str(model_dir),
        "device": device,
        "profile": profile,
        "maxTokens": 40,
        "timeoutSeconds": max(15.0, float(args.timeout)),
        "generationTimeoutSeconds": max(10.0, float(args.timeout) - 10.0),
        "imageRegion": {"source": "full_frame"},
        "detections": [{"label": "person", "rawLabel": "person", "confidence": 0.9, "bbox": [0, 0, 1, 1]}],
        "analysisContext": {
            "selectedUseCaseProfile": "seatbelt_compliance",
            "profileLabel": "Seatbelt compliance",
            "userQuery": "driver missing seatbelt",
            "findingName": "seatbelt_missing",
            "friendlyFindingName": "Missing seatbelt",
            "reviewLevel": "review_candidate",
        },
        "violations": [
            {
                "name": "seatbelt_missing",
                "description": "Seatbelt path needs visual review.",
                "severity": "medium",
                "confidence": 0.5,
            }
        ],
    }

    work_parent = args.work_dir.resolve() if args.work_dir else None
    if work_parent is not None:
        work_parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="safetrace_vlm_worker_smoke_", dir=str(work_parent) if work_parent else None))
    try:
        input_path = temp / "request.json"
        output_path = temp / "result.json"
        input_path.write_text(json.dumps(request), encoding="utf-8")
        started = time.perf_counter()
        try:
            process = subprocess.run(
                _command(args, input_path, output_path),
                cwd=str(args.app_root.resolve()),
                text=True,
                capture_output=True,
                timeout=max(20.0, float(args.timeout) + 20.0),
                check=False,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            process = None
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
        elapsed = time.perf_counter() - started
        payload: dict[str, Any] = {}
        if output_path.is_file():
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                payload = {"outputParseError": str(exc)}
    finally:
        if not args.keep_artifacts:
            shutil.rmtree(temp, ignore_errors=True)

    report = {
        "modelDir": str(model_dir),
        "profile": profile,
        "requestedDevice": args.device,
        "actualDevice": payload.get("actualDevice") or device,
        "cudaAvailable": payload.get("cudaAvailable"),
        "processorClass": payload.get("processorClass"),
        "modelClass": payload.get("modelClass"),
        "modelLoaded": payload.get("modelLoaded"),
        "modelLoadSeconds": payload.get("modelLoadSeconds"),
        "generationSeconds": payload.get("generationSeconds"),
        "rawOutput": payload.get("rawTextPreview"),
        "cleanOutput": payload.get("cleanTextPreview"),
        "accepted": bool(payload.get("ok")),
        "rejected": bool(payload) and not bool(payload.get("ok")),
        "qualityIssue": payload.get("qualityIssue"),
        "fallbackReason": payload.get("fallbackReason"),
        "errorType": payload.get("errorType"),
        "errorMessage": payload.get("errorMessage"),
        "workerExitCode": process.returncode if process is not None else None,
        "timedOut": timed_out,
        "elapsedSeconds": round(elapsed, 3),
        "stdout": (process.stdout if process is not None else stdout)[-1000:],
        "stderr": (process.stderr if process is not None else stderr)[-2000:],
    }
    print(json.dumps(report, indent=2, default=str))
    return 0 if process is not None and process.returncode in {0, 3} else 1


if __name__ == "__main__":
    raise SystemExit(main())
