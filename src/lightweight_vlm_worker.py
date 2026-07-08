"""Crash-isolated lightweight VLM worker entrypoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _bounded_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _preview(text: str | None, *, limit: int = 240) -> str | None:
    if not text:
        return None
    compact = " ".join(str(text).split())
    return compact[:limit]


def _configure_worker_env(
    app_root: Path | None,
    model_dir: Path | None,
    *,
    model_profile: str,
    max_tokens: int,
    generation_timeout_seconds: float,
) -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("SAFETRACE_DEVICE", "cpu")
    os.environ["SAFETRACE_ANALYSIS_SAFE_MODE"] = "false"
    os.environ["SAFETRACE_ENABLE_VLM"] = "true"
    os.environ["SAFETRACE_VLM_ENABLED"] = "true"
    os.environ["SAFETRACE_VLM_PROVIDER"] = "auto"
    os.environ["SAFETRACE_VLM_PROFILE"] = model_profile
    os.environ["SAFETRACE_VLM_MAX_TOKENS"] = str(max_tokens)
    os.environ["SAFETRACE_VLM_TIMEOUT_SECONDS"] = f"{generation_timeout_seconds:.1f}"
    if app_root is not None:
        os.environ.setdefault("SAFETRACE_APP_ROOT", str(app_root))
        os.environ.setdefault("SAFETRACE_PROJECT_ROOT", str(app_root))
        os.environ.setdefault("SAFETRACE_DATA_DIR", str(app_root / "data"))
        os.environ.setdefault("SAFETRACE_CHECKPOINTS_DIR", str(app_root / "checkpoints"))
        os.environ.setdefault("SAFETRACE_VLM_DIR", str(app_root / "models" / "vlm"))
        os.environ.setdefault("SAFETRACE_VLM_LIGHTWEIGHT_MODEL_PATH", str(app_root / "models" / "vlm" / "lightweight-256m"))
        os.environ.setdefault("SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH", str(app_root / "models" / "vlm" / "lightweight-512m"))
        os.environ.setdefault("SAFETRACE_VLM_ENHANCED_MODEL_PATH", str(app_root / "models" / "vlm" / "enhanced-2b"))
        os.environ.setdefault("SAFETRACE_VLM_ENHANCED_3B_MODEL_PATH", str(app_root / "models" / "vlm" / "enhanced-3b"))
    if model_dir is not None:
        os.environ["SAFETRACE_VLM_MODEL_PATH"] = str(model_dir)
        if model_profile == "lightweight_512m":
            os.environ["SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH"] = str(model_dir)
        elif model_profile == "enhanced_2b":
            os.environ["SAFETRACE_VLM_ENHANCED_MODEL_PATH"] = str(model_dir)
        elif model_profile == "enhanced_3b":
            os.environ["SAFETRACE_VLM_ENHANCED_3B_MODEL_PATH"] = str(model_dir)
        else:
            os.environ["SAFETRACE_VLM_LIGHTWEIGHT_MODEL_PATH"] = str(model_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one lightweight VLM explanation request outside the API process")
    parser.add_argument("--input-json", "--input", dest="input_json", type=Path, required=True)
    parser.add_argument("--output-json", "--output", dest="output_json", type=Path, required=True)
    parser.add_argument("--app-root", type=Path, default=None)
    return parser.parse_args(argv)


def run_worker(input_json: Path, output_json: Path, *, app_root: Path | None = None) -> int:
    worker_started_at = time.perf_counter()
    model_load_seconds: float | None = None
    generation_seconds: float | None = None

    def timing_payload() -> Dict[str, Any]:
        return {
            "workerDurationSeconds": round(time.perf_counter() - worker_started_at, 3),
            "modelLoadSeconds": round(model_load_seconds, 3) if model_load_seconds is not None else None,
            "generationSeconds": round(generation_seconds, 3) if generation_seconds is not None else None,
        }

    try:
        request = json.loads(input_json.read_text(encoding="utf-8-sig"))
        model_dir = Path(str(request.get("modelDir") or ""))
        if not model_dir.is_absolute() and app_root is not None:
            model_dir = (app_root / model_dir).resolve()
        worker_timeout_seconds = _bounded_float(
            request.get("timeoutSeconds"),
            default=120.0,
            minimum=15.0,
            maximum=300.0,
        )
        max_tokens = _bounded_int(
            request.get("maxTokens"),
            default=40,
            minimum=24,
            maximum=64,
        )
        model_profile = str(request.get("profile") or "lightweight_512m").strip().lower()
        if model_profile not in {"lightweight_256m", "lightweight_512m", "enhanced_3b", "enhanced_2b"}:
            model_profile = "lightweight_512m"
        generation_timeout_seconds = _bounded_float(
            request.get("generationTimeoutSeconds"),
            default=max(20.0, worker_timeout_seconds - 10.0),
            minimum=10.0,
            maximum=max(10.0, worker_timeout_seconds - 2.0),
        )
        _configure_worker_env(
            app_root.resolve() if app_root is not None else None,
            model_dir,
            model_profile=model_profile,
            max_tokens=max_tokens,
            generation_timeout_seconds=generation_timeout_seconds,
        )

        image_path = Path(str(request["imagePath"]))
        device = str(request.get("device") or "cpu")
        image_region = request.get("imageRegion") if isinstance(request.get("imageRegion"), dict) else {"source": "full_frame"}
        detection_context = request.get("detections") if isinstance(request.get("detections"), list) else []
        analysis_context = request.get("analysisContext") if isinstance(request.get("analysisContext"), dict) else {}

        from .schemas import Violation
        from .utils import imread_rgb
        from .vlm_reasoner import VlmReasoner, is_useful_vlm_output

        violations: List[Violation] = []
        for item in list(request.get("violations") or []):
            violations.append(
                Violation(
                    name=str(item.get("name") or "violation"),
                    description=str(item.get("description") or ""),
                    severity=str(item.get("severity") or "medium"),
                    confidence=float(item.get("confidence") or 0.0),
                )
            )

        model_load_started_at = time.perf_counter()
        reasoner = VlmReasoner(model_dir=model_dir, device=device, enabled=True)
        model_load_seconds = time.perf_counter() - model_load_started_at
        image = imread_rgb(image_path)
        generation_started_at = time.perf_counter()
        explanation = reasoner.explain_violation(
            image,
            violations,
            context={
                "imageRegion": image_region,
                "detections": detection_context,
                "analysisContext": analysis_context,
            },
        )
        generation_seconds = time.perf_counter() - generation_started_at
        source = str(getattr(reasoner, "last_explanation_source", "rule_based") or "rule_based")
        if source == "vlm_local":
            source = "vlm_enhanced" if model_profile in {"enhanced_2b", "enhanced_3b"} else "vlm_lightweight"

        if source not in {"vlm_lightweight", "vlm_enhanced"} or not is_useful_vlm_output(explanation):
            fallback_reason = str(
                getattr(reasoner, "last_fallback_reason", None)
                or source
                or "worker_fallback"
            )
            _write_json(
                output_json,
                {
                    "ok": False,
                    "errorType": "VlmFallback",
                    "fallbackReason": fallback_reason,
                    "explanationSource": "rule_based",
                    "modelProfile": model_profile,
                    "qualityIssue": getattr(reasoner, "last_quality_issue", None),
                    "rawTextPreview": _preview(getattr(reasoner, "last_raw_vlm_text", None)),
                    "cleanTextPreview": _preview(getattr(reasoner, "last_clean_vlm_text", None)),
                    "analysisContext": analysis_context,
                    "generationTimeoutSeconds": generation_timeout_seconds,
                    "maxTokens": max_tokens,
                    "imageRegion": image_region,
                    **timing_payload(),
                },
            )
            return 3

        _write_json(
            output_json,
            {
                "ok": True,
                "explanation": explanation,
                "explanationSource": source,
                "modelProfile": model_profile,
                "quality": "accepted",
                "generationTimeoutSeconds": generation_timeout_seconds,
                "maxTokens": max_tokens,
                "rawTextPreview": _preview(getattr(reasoner, "last_raw_vlm_text", None)),
                "cleanTextPreview": _preview(getattr(reasoner, "last_clean_vlm_text", None)),
                "analysisContext": analysis_context,
                "imageRegion": image_region,
                **timing_payload(),
            },
        )
        return 0
    except Exception as exc:  # pragma: no cover - defensive subprocess boundary
        _write_json(
            output_json,
            {
                "ok": False,
                "errorType": type(exc).__name__,
                "errorMessage": str(exc),
                "fallbackReason": type(exc).__name__,
                "traceback": traceback.format_exc(limit=6),
                "explanationSource": "rule_based",
                "modelProfile": locals().get("model_profile", "lightweight_512m"),
                **timing_payload(),
            },
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_worker(args.input_json, args.output_json, app_root=args.app_root)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
