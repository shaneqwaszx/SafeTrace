"""Direct local-model smoke for the optional Phase U scene verifier."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/vlm/lightweight-512m"))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from src.config import SETTINGS
    from src.mid_vlm_verifier import MidVlmSceneVerifier
    from src.utils import imread_rgb

    SETTINGS.mid_vlm_enabled = True
    SETTINGS.mid_vlm_model_path = args.model_dir
    SETTINGS.mid_vlm_device = args.device
    SETTINGS.mid_vlm_concurrency = 1
    verifier = MidVlmSceneVerifier()
    response = verifier.verify(
        imread_rgb(args.image),
        finding="seatbelt_missing",
        profile="seatbelt_compliance",
        query="Is a person in a vehicle cabin with a visible seatbelt path?",
    )
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "image": str(args.image.resolve()),
        "modelDir": str(verifier.model_dir.resolve()),
        "device": verifier.device,
        "concurrency": 1,
        "authoritative": False,
        "response": response,
        "diagnostics": verifier.last_diagnostics,
        "passed": response is not None and verifier.last_diagnostics.get("accepted") is True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
