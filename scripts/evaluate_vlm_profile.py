"""Evaluate SafeTrace local VLM profile candidates.

This diagnostic harness runs the crash-isolated VLM worker directly against
local evidence images. It writes generated JSON/CSV reports under
tmp_vlm_profile_eval/ by default and never modifies model or checkpoint files.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "tmp_vlm_profile_eval"
MODEL_PATHS = {
    "lightweight_256m": ROOT / "models" / "vlm" / "lightweight-256m",
    "lightweight_512m": ROOT / "models" / "vlm" / "lightweight-512m",
    "enhanced_2b": ROOT / "models" / "vlm" / "enhanced-2b",
    "enhanced_3b": ROOT / "models" / "vlm" / "enhanced-3b",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REVIEW_PROFILES: dict[str, dict[str, Any]] = {
    "seatbelt": {
        "violation": {
            "name": "missing_seatbelt_review",
            "description": "Possible missing seatbelt or seatbelt not visible.",
            "severity": "high",
            "confidence": 0.75,
        },
        "prompt": (
            "Focus on the torso area and possible belt path. State whether a belt path is visible across the torso. "
            "If the torso or belt path is occluded, say unclear."
        ),
    },
    "helmet_ppe": {
        "violation": {
            "name": "missing_helmet_ppe_review",
            "description": "Possible missing helmet or PPE.",
            "severity": "high",
            "confidence": 0.75,
        },
        "prompt": (
            "Focus on the head and upper-body area. State whether the head is visible and whether a helmet or PPE is visible. "
            "If the head is not visible, say unclear."
        ),
    },
    "phone": {
        "violation": {
            "name": "phone_use_review",
            "description": "Possible phone use or distracted driving.",
            "severity": "medium",
            "confidence": 0.65,
        },
        "prompt": (
            "Focus on the hand, face, and driver area. State whether a phone is visibly near a hand, face, or driver area. "
            "If active use is unclear, say unclear."
        ),
    },
    "general": {
        "violation": {
            "name": "general_safety_review",
            "description": "General safety review candidate.",
            "severity": "medium",
            "confidence": 0.5,
        },
        "prompt": (
            "State what visible safety evidence supports or weakens this finding."
        ),
    },
}


def _preview(text: str | None, *, limit: int = 500) -> str | None:
    if not text:
        return None
    return " ".join(str(text).split())[:limit]


def _default_model_path(profile: str) -> Path:
    return MODEL_PATHS[profile]


def _candidate_images(input_dir: Path | None, explicit_images: list[Path], limit: int) -> list[Path]:
    images = [path for path in explicit_images if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if input_dir and input_dir.is_dir():
        for path in sorted(input_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append(path)
    if not images:
        data_jobs = ROOT / "data" / "api_jobs"
        if data_jobs.exists():
            for path in sorted(data_jobs.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    images.append(path)
    return images[: max(1, limit)]


def _synthetic_image(output_dir: Path) -> Path:
    path = output_dir / "synthetic_driver_seatbelt_review.png"
    image = Image.new("RGB", (480, 320), (235, 238, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((115, 65, 370, 300), fill=(210, 215, 220), outline=(80, 80, 80), width=3)
    draw.ellipse((215, 80, 275, 140), fill=(180, 140, 110), outline=(70, 70, 70), width=2)
    draw.rectangle((190, 140, 300, 255), fill=(80, 120, 190), outline=(40, 60, 110), width=2)
    draw.line((185, 145, 305, 255), fill=(40, 40, 40), width=8)
    draw.text((18, 18), "Synthetic structural VLM test image", fill=(20, 20, 20))
    image.save(path)
    return path


def _prompt_for(profile: str) -> str:
    return (
        "Look only at the image crop. Use visible evidence only. If the safety detail is unclear, say unclear. "
        "Do not list unrelated objects. Do not repeat the prompt. Keep every field short. "
        + REVIEW_PROFILES[profile]["prompt"]
        + "\nReply with exactly:\nvisible_evidence:\nvisual_status:\nshort_reason:\nconfidence_hint:"
    )


def _request_payload(
    *,
    image: Path,
    model_profile: str,
    model_path: Path,
    review_profile: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "imagePath": str(image),
        "modelDir": str(model_path),
        "device": "cpu",
        "profile": model_profile,
        "maxTokens": args.max_tokens,
        "timeoutSeconds": args.timeout_seconds,
        "generationTimeoutSeconds": args.generation_timeout_seconds,
        "imageRegion": {
            "source": "evaluation_full_frame",
            "bbox": None,
            "note": "Evaluation harness uses local evidence images without modifying source data.",
        },
        "detections": [],
        "violations": [REVIEW_PROFILES[review_profile]["violation"]],
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
    except subprocess.TimeoutExpired as exc:
        return {
            "exitCode": None,
            "elapsedSeconds": round(time.perf_counter() - started, 3),
            "stdoutPreview": _preview(str(exc.stdout or "")),
            "stderrPreview": _preview(str(exc.stderr or "")),
            "output": {"ok": False, "fallbackReason": "harness_timeout"},
        }

    payload: dict[str, Any] = {}
    if output_path.exists():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            payload = {"ok": False, "errorType": "InvalidJson", "errorMessage": str(exc)}
    return {
        "exitCode": result.returncode,
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "stdoutPreview": _preview(result.stdout),
        "stderrPreview": _preview(result.stderr),
        "output": payload,
    }


def _row_from_run(
    *,
    image: Path,
    model_profile: str,
    model_path: Path,
    review_profile: str,
    prompt: str,
    request_path: Path | None,
    output_path: Path | None,
    run: dict[str, Any],
) -> dict[str, Any]:
    payload = run.get("output") or {}
    accepted = bool(payload.get("ok"))
    return {
        "modelProfile": model_profile,
        "modelPath": str(model_path),
        "imagePath": str(image),
        "reviewProfile": review_profile,
        "prompt": prompt,
        "imageRegion": (payload.get("imageRegion") or {}).get("source", "evaluation_full_frame"),
        "requestPath": str(request_path) if request_path else None,
        "outputPath": str(output_path) if output_path else None,
        "exitCode": run.get("exitCode"),
        "elapsedSeconds": run.get("elapsedSeconds"),
        "accepted": accepted,
        "fallbackReason": payload.get("fallbackReason"),
        "qualityReason": payload.get("qualityIssue") or payload.get("fallbackReason"),
        "rawOutput": payload.get("rawTextPreview"),
        "cleanedOutput": payload.get("cleanTextPreview") or payload.get("explanation"),
        "modelLoadSeconds": payload.get("modelLoadSeconds"),
        "generationSeconds": payload.get("generationSeconds"),
        "totalSeconds": payload.get("workerDurationSeconds") or run.get("elapsedSeconds"),
        "stdoutPreview": run.get("stdoutPreview"),
        "stderrPreview": run.get("stderrPreview"),
    }


def _write_reports(output_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vlm_profile_eval_report.json"
    csv_path = output_dir / "vlm_profile_eval_summary.csv"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    fieldnames = [
        "modelProfile",
        "modelPath",
        "imagePath",
        "reviewProfile",
        "imageRegion",
        "accepted",
        "fallbackReason",
        "qualityReason",
        "modelLoadSeconds",
        "generationSeconds",
        "totalSeconds",
        "rawOutput",
        "cleanedOutput",
        "prompt",
        "exitCode",
        "elapsedSeconds",
        "requestPath",
        "outputPath",
        "stdoutPreview",
        "stderrPreview",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("runs", []):
            writer.writerow({key: row.get(key) for key in fieldnames})
    return json_path, csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SafeTrace local VLM profile candidate.")
    parser.add_argument(
        "--model-profile",
        choices=sorted(MODEL_PATHS),
        default="lightweight_512m",
        help="SafeTrace VLM profile id to evaluate.",
    )
    parser.add_argument("--model-path", type=Path, default=None, help="Override model directory for the profile.")
    parser.add_argument(
        "--profile",
        dest="review_profile",
        action="append",
        choices=sorted(REVIEW_PROFILES),
        help="Safety review profile(s) to test. Defaults to all.",
    )
    parser.add_argument("--image", action="append", type=Path, default=[], help="Evidence image to evaluate.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory of local evidence images.")
    parser.add_argument("--limit", type=int, default=1, help="Maximum number of images to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--generation-timeout-seconds", type=float, default=50.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="Write report schema without launching the worker.")
    parser.add_argument(
        "--synthetic-if-empty",
        action="store_true",
        help="Create a generated structural image if no local evidence image is found.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = (args.model_path or _default_model_path(args.model_profile)).resolve()

    if not model_path.exists() and not args.dry_run:
        report = {
            "modelProfile": args.model_profile,
            "modelPath": str(model_path),
            "status": "model_path_missing",
            "runs": [],
            "summary": {"total": 0, "accepted": 0, "fallback": 0},
        }
        json_path, csv_path = _write_reports(output_dir, report)
        print(f"model path missing: {model_path}", file=sys.stderr)
        print(f"report: {json_path}")
        print(f"csv: {csv_path}")
        return 2

    images = _candidate_images(args.input_dir, args.image, args.limit)
    used_synthetic = False
    if not images and (args.synthetic_if_empty or args.dry_run):
        images = [_synthetic_image(output_dir)]
        used_synthetic = True
    if not images:
        print("No evidence images found. Pass --image/--input-dir or use --synthetic-if-empty.", file=sys.stderr)
        return 2

    review_profiles = args.review_profile or sorted(REVIEW_PROFILES)
    runs: list[dict[str, Any]] = []
    for image in images[: max(1, args.limit)]:
        for review_profile in review_profiles:
            prompt = _prompt_for(review_profile)
            safe_stem = f"{image.stem}_{args.model_profile}_{review_profile}".replace(" ", "_")
            request_path = output_dir / f"{safe_stem}_request.json"
            output_path = output_dir / f"{safe_stem}_result.json"
            if args.dry_run:
                run = {
                    "exitCode": 0,
                    "elapsedSeconds": 0.0,
                    "output": {
                        "ok": False,
                        "fallbackReason": "dry_run",
                        "qualityIssue": "dry_run",
                        "imageRegion": {"source": "evaluation_full_frame"},
                    },
                }
                request_path = None
                output_path = None
            else:
                request = _request_payload(
                    image=image,
                    model_profile=args.model_profile,
                    model_path=model_path,
                    review_profile=review_profile,
                    args=args,
                )
                request_path.write_text(json.dumps(request, indent=2, default=str), encoding="utf-8")
                run = _run_worker(request_path, output_path, args.timeout_seconds)
            runs.append(
                _row_from_run(
                    image=image,
                    model_profile=args.model_profile,
                    model_path=model_path,
                    review_profile=review_profile,
                    prompt=prompt,
                    request_path=request_path,
                    output_path=output_path,
                    run=run,
                )
            )

    accepted = sum(1 for run in runs if run["accepted"])
    report = {
        "modelProfile": args.model_profile,
        "modelPath": str(model_path),
        "dryRun": bool(args.dry_run),
        "usedSyntheticImage": used_synthetic,
        "timeoutSeconds": args.timeout_seconds,
        "generationTimeoutSeconds": args.generation_timeout_seconds,
        "maxTokens": args.max_tokens,
        "runs": runs,
        "summary": {
            "total": len(runs),
            "accepted": accepted,
            "fallback": len(runs) - accepted,
        },
    }
    json_path, csv_path = _write_reports(output_dir, report)
    print(json.dumps(report["summary"], indent=2))
    print(f"report: {json_path}")
    print(f"csv: {csv_path}")
    if args.dry_run:
        return 0
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
