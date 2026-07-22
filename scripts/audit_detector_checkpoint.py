"""Report detector identity and SafeTrace-relevant class coverage without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_CONCEPTS = {
    "person": {"person"},
    "phone": {"cell phone", "phone", "mobile phone"},
    "seatbelt": {"seatbelt", "seat belt", "safety belt"},
    "helmet": {"helmet", "hardhat", "hard hat"},
    "head": {"head"},
    "torso": {"torso", "upper body"},
    "hand": {"hand", "hands"},
    "steering_wheel": {"steering wheel"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(path))
    names = model.names
    labels = [str(names[index] if isinstance(names, dict) else names[index]) for index in range(len(names))]
    normalized = {label.strip().lower() for label in labels}
    concepts = {
        concept: bool(normalized.intersection(aliases))
        for concept, aliases in REQUIRED_CONCEPTS.items()
    }
    return {
        "checkpoint": str(path),
        "sizeBytes": path.stat().st_size,
        "sha256": sha256(path),
        "task": getattr(model, "task", None),
        "classCount": len(labels),
        "classes": labels,
        "safeTraceConceptCoverage": concepts,
        "strongSeatbeltSupport": concepts["person"] and concepts["seatbelt"] and concepts["torso"],
        "strongHelmetSupport": concepts["person"] and concepts["helmet"] and concepts["head"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    report = inspect_checkpoint(args.checkpoint.resolve())
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
