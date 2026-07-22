"""Compare observed Phase W controls with the versioned contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.processing_contracts import compare_processing_contract, require_baseline_update_reason


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=ROOT / "evaluation/processing_contracts/safetrace_processing_contract_v1.json")
    parser.add_argument("--observed", type=Path, default=ROOT / ".ai-pipeline/005_tasks/phase_w_general_profile_performance_cleanup/general_profile_parity_report.json")
    parser.add_argument("--output", type=Path, default=ROOT / ".ai-pipeline/005_tasks/phase_w_general_profile_performance_cleanup/processing_contract_comparison.json")
    parser.add_argument("--baseline-update-reason")
    args = parser.parse_args()
    if args.baseline_update_reason is not None:
        require_baseline_update_reason(args.baseline_update_reason)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    observed_payload = json.loads(args.observed.read_text(encoding="utf-8"))
    controls = list(observed_payload.get("controls") or observed_payload.get("results") or [])
    result = compare_processing_contract(contract, controls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
