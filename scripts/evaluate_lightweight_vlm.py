"""Evaluate the local Lightweight VLM worker on small evidence-image sets.

This script is for local diagnostics only. It writes generated reports under
tmp_vlm_eval/ by default and does not modify model or checkpoint files.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "vlm" / "lightweight-512m"
DEFAULT_OUTPUT_DIR = ROOT / "tmp_vlm_eval"


SCENARIOS: dict[str, list[dict[str, Any]]] = {
    "seatbelt": [
        {
            "name": "seatbelt_missing",
            "description": "Missing seatbelt review candidate.",
            "severity": "high",
            "confidence": 0.75,
        }
    ],
    "helmet": [
        {
            "name": "helmet_missing",
            "description": "Missing helmet or PPE review candidate.",
            "severity": "high",
            "confidence": 0.75,
        }
    ],
    "phone": [
        {
            "name": "phone_use",
            "description": "Possible phone use or distracted driving review candidate.",
            "severity": "medium",
            "confidence": 0.65,
        }
    ],
    "generic_safety": [
        {
            "name": "safety_review",
            "description": "General safety review candidate.",
            "severity": "medium",
            "confidence": 0.5,
        }
    ],
}


def _candidate_images(limit: int) -> list[Path]:
    images: list[Path] = []
    data_jobs = ROOT / "data" / "api_jobs"
    if data_jobs.exists():
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            images.extend(sorted(data_jobs.rglob(pattern)))
    return images[:limit]


def _synthetic_image(output_dir: Path) -> Path:
    path = output_dir / "synthetic_driver_seatbelt_review.png"
    image = Image.new("RGB", (480, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 480, 320), fill=(235, 238, 240))
    draw.rectangle((110, 70, 360, 300), fill=(210, 215, 220), outline=(90, 90, 90), width=3)
    draw.ellipse((210, 75, 270, 135), fill=(180, 140, 110), outline=(70, 70, 70), width=2)
    draw.rectangle((190, 135, 295, 255), fill=(80, 120, 190), outline=(40, 60, 110), width=2)
    draw.line((185, 145, 300, 255), fill=(40, 40, 40), width=8)
    draw.text((20, 20), "Synthetic structural test image", fill=(20, 20, 20))
    image.save(path)
    return path


def _request_payload(image: Path, scenario: str, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    # Use a broad center crop hint for structural testing; production crops come
    # from detector boxes in the backend path.
    return {
        "imagePath": str(image),
        "modelDir": str(args.model_dir),
        "device": "cpu",
        "profile": "lightweight_512m",
        "maxTokens": args.max_tokens,
        "timeoutSeconds": args.timeout_seconds,
        "generationTimeoutSeconds": args.generation_timeout_seconds,
        "imageRegion": {
            "source": "evaluation_full_frame",
            "bbox": None,
            "note": "Evaluation harness uses existing evidence image without modifying source data.",
        },
        "detections": [],
        "violations": SCENARIOS[scenario],
    }


def _run_worker(request_path: Path, output_path: Path, timeout_seconds: float) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "src.lightweight_vlm_worker",
        "--input-json",
        str(request_path),
        "--output-json",
        str(output_path),
        "--app-root",
        str(ROOT),
    ]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout_seconds + 10,
            check=False,
        )
        elapsed = time.perf_counter() - started
        payload: dict[str, Any] = {}
        if output_path.exists():
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                payload = {"ok": False, "errorType": "InvalidJson", "errorMessage": str(exc)}
        return {
            "exitCode": result.returncode,
            "elapsedSeconds": round(elapsed, 3),
            "stdoutPreview": " ".join(result.stdout.split())[:500],
            "stderrPreview": " ".join(result.stderr.split())[:500],
            "output": payload,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "exitCode": None,
            "elapsedSeconds": round(time.perf_counter() - started, 3),
            "stdoutPreview": " ".join(str(exc.stdout or "").split())[:500],
            "stderrPreview": " ".join(str(exc.stderr or "").split())[:500],
            "output": {
                "ok": False,
                "errorType": "HarnessTimeout",
                "fallbackReason": "harness_timeout",
            },
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the SafeTrace Lightweight VLM worker locally.")
    parser.add_argument("--image", action="append", type=Path, default=[], help="Evidence image to evaluate.")
    parser.add_argument("--limit", type=int, default=1, help="Number of discovered data/api_jobs images to use.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--generation-timeout-seconds", type=float, default=50.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS),
        help="Scenario(s) to run. Defaults to all.",
    )
    parser.add_argument("--synthetic-if-empty", action="store_true", help="Create a generated structural image if no local evidence image is found.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [path for path in args.image if path.exists()]
    if not images:
        images = _candidate_images(max(1, args.limit))
    used_synthetic = False
    if not images and args.synthetic_if_empty:
        images = [_synthetic_image(output_dir)]
        used_synthetic = True
    if not images:
        print("No evidence images found. Pass --image or use --synthetic-if-empty.", file=sys.stderr)
        return 2

    scenarios = args.scenario or sorted(SCENARIOS)
    runs: list[dict[str, Any]] = []
    for image in images[: max(1, args.limit)]:
        for scenario in scenarios:
            safe_stem = f"{image.stem}_{scenario}".replace(" ", "_")
            request_path = output_dir / f"{safe_stem}_request.json"
            output_path = output_dir / f"{safe_stem}_result.json"
            request = _request_payload(image, scenario, args, output_dir)
            request_path.write_text(json.dumps(request, indent=2, default=str), encoding="utf-8")
            run = _run_worker(request_path, output_path, args.timeout_seconds)
            payload = run.get("output") or {}
            runs.append(
                {
                    "image": str(image),
                    "scenario": scenario,
                    "requestPath": str(request_path),
                    "outputPath": str(output_path),
                    "exitCode": run.get("exitCode"),
                    "elapsedSeconds": run.get("elapsedSeconds"),
                    "ok": bool(payload.get("ok")),
                    "fallbackReason": payload.get("fallbackReason"),
                    "qualityIssue": payload.get("qualityIssue"),
                    "rawTextPreview": payload.get("rawTextPreview"),
                    "cleanTextPreview": payload.get("cleanTextPreview") or payload.get("explanation"),
                    "modelLoadSeconds": payload.get("modelLoadSeconds"),
                    "generationSeconds": payload.get("generationSeconds"),
                    "workerDurationSeconds": payload.get("workerDurationSeconds"),
                    "stdoutPreview": run.get("stdoutPreview"),
                    "stderrPreview": run.get("stderrPreview"),
                }
            )

    accepted = sum(1 for run in runs if run["ok"])
    report = {
        "modelDir": str(args.model_dir),
        "timeoutSeconds": args.timeout_seconds,
        "generationTimeoutSeconds": args.generation_timeout_seconds,
        "maxTokens": args.max_tokens,
        "usedSyntheticImage": used_synthetic,
        "runs": runs,
        "summary": {
            "total": len(runs),
            "accepted": accepted,
            "fallback": len(runs) - accepted,
        },
    }
    report_path = output_dir / "lightweight_vlm_eval_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"report: {report_path}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
