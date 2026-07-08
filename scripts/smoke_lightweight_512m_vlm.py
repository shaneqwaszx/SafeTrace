"""Smoke-test the layered Lightweight 512M VLM verifier path.

This script is local-only and does not download models. It invokes the same
crash-isolated worker client used by the backend and prints a compact JSON
diagnostic payload.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SETTINGS
from src.lightweight_vlm_worker_client import LightweightVlmWorkerReasoner
from src.schemas import Violation
from src.utils import imread_rgb


PROFILE_VIOLATIONS = {
    "seatbelt": Violation(
        name="seatbelt_missing",
        description="Review whether a seatbelt is visibly missing or unclear.",
        severity="high",
        confidence=0.6,
    ),
    "helmet": Violation(
        name="helmet_missing",
        description="Review whether helmet or PPE evidence is visible.",
        severity="high",
        confidence=0.6,
    ),
    "phone": Violation(
        name="phone_use",
        description="Review whether a phone is visible near the driver or hands.",
        severity="medium",
        confidence=0.5,
    ),
    "general": Violation(
        name="safety_review_candidate",
        description="Review visible safety evidence and uncertainty.",
        severity="medium",
        confidence=0.5,
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test SafeTrace Lightweight 512M layered VLM verifier")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_VIOLATIONS), default="seatbelt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    SETTINGS.vlm_profile = "lightweight_512m"
    SETTINGS.enable_vlm = True
    SETTINGS.vlm_enabled = "true"
    SETTINGS.lightweight_vlm_worker_enabled = True
    SETTINGS.lightweight_vlm_worker_timeout_seconds = float(args.timeout_seconds)
    SETTINGS.vlm_max_tokens = 40

    image = imread_rgb(args.image)
    violation = PROFILE_VIOLATIONS[args.profile]
    reasoner = LightweightVlmWorkerReasoner(model_dir=SETTINGS.vlm_lightweight_512m_model_path, device=args.device)
    started = time.perf_counter()
    explanation = reasoner.explain_violation(image, [violation])
    elapsed = time.perf_counter() - started
    diagnostics: dict[str, Any] = dict(reasoner.last_diagnostics)
    accepted = bool(diagnostics.get("lightweightVlmWorkerSucceeded"))
    rendered_template = "\n".join(
        [
            "Rule-based finding:",
            violation.name,
            "",
            "Lightweight visual check:",
            explanation,
            "",
            "Reviewer note:",
            "Confirm against the original footage when blur, camera angle, or occlusion affects visibility.",
        ]
    )
    payload = {
        "attempted": bool(diagnostics.get("lightweightVlmWorkerAttempted")),
        "succeeded": bool(diagnostics.get("lightweightVlmWorkerSucceeded")),
        "timedOut": bool(diagnostics.get("lightweightVlmWorkerTimedOut")),
        "accepted": accepted,
        "rejected": bool(diagnostics.get("lightweightVlmWorkerAttempted") and not accepted),
        "qualityIssue": diagnostics.get("lightweightVlmQualityIssue"),
        "fallbackReason": diagnostics.get("lightweightVlmFallbackReason"),
        "rawText": diagnostics.get("lightweightVlmRawTextPreview"),
        "stdoutPreview": diagnostics.get("lightweightVlmWorkerStdoutPreview"),
        "stderrPreview": diagnostics.get("lightweightVlmWorkerStderrPreview"),
        "parsedFields": {
            "visible_evidence": None,
            "visual_status": None,
            "short_reason": diagnostics.get("lightweightVlmCleanTextPreview"),
            "confidence_hint": None,
        },
        "renderedTemplate": rendered_template,
        "device": reasoner.device,
        "timings": {
            "totalSeconds": round(elapsed, 3),
            "workerSeconds": diagnostics.get("lightweightVlmWorkerDurationSeconds"),
            "modelLoadSeconds": diagnostics.get("lightweightVlmWorkerModelLoadSeconds"),
            "generationSeconds": diagnostics.get("lightweightVlmWorkerGenerationSeconds"),
        },
        "crop": diagnostics.get("lightweightVlmImageRegion"),
        "modelPath": str(SETTINGS.vlm_lightweight_512m_model_path),
    }
    text = json.dumps(payload, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
