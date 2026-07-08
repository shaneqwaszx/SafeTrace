"""Compare SafeTrace backend benchmark reports for regressions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_runs(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [run for run in list(report.get("runs") or []) if run.get("status") == "completed"]


def _avg(runs: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values = []
    for run in runs:
        value: Any = run
        for key in path:
            value = dict(value or {}).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _first_correctness(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    return dict((runs[0].get("engineMetrics") or {}).get("correctness") or {})


def compare_reports(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    duration_tolerance: float = 0.15,
    detector_tolerance: float = 0.15,
    evidence_drop_tolerance: float = 0.20,
) -> dict[str, Any]:
    before_runs = _completed_runs(before)
    after_runs = _completed_runs(after)
    performance: list[dict[str, Any]] = []
    logic: list[dict[str, Any]] = []

    if before_runs and not after_runs:
        logic.append({"type": "final_status_regression", "message": "After benchmark has no completed runs."})

    before_total = _avg(before_runs, ("engineMetrics", "totalDurationSeconds"))
    after_total = _avg(after_runs, ("engineMetrics", "totalDurationSeconds"))
    if before_total is not None and after_total is not None and after_total > before_total * (1 + duration_tolerance):
        performance.append(
            {
                "type": "total_duration_regression",
                "before": round(before_total, 4),
                "after": round(after_total, 4),
                "tolerance": duration_tolerance,
            }
        )

    before_detector = _avg(before_runs, ("engineMetrics", "stageDurations", "detectorInference"))
    after_detector = _avg(after_runs, ("engineMetrics", "stageDurations", "detectorInference"))
    if (
        before_detector is not None
        and after_detector is not None
        and after_detector > before_detector * (1 + detector_tolerance)
    ):
        performance.append(
            {
                "type": "detector_inference_regression",
                "before": round(before_detector, 4),
                "after": round(after_detector, 4),
                "tolerance": detector_tolerance,
            }
        )

    before_evidence = _avg(before_runs, ("engineMetrics", "counts", "evidenceFrames"))
    after_evidence = _avg(after_runs, ("engineMetrics", "counts", "evidenceFrames"))
    if (
        before_evidence is not None
        and after_evidence is not None
        and before_evidence > 0
        and after_evidence < before_evidence * (1 - evidence_drop_tolerance)
    ):
        logic.append(
            {
                "type": "evidence_count_drop",
                "before": round(before_evidence, 4),
                "after": round(after_evidence, 4),
                "tolerance": evidence_drop_tolerance,
            }
        )

    before_correct = _first_correctness(before_runs)
    after_correct = _first_correctness(after_runs)
    before_keys = set(before_correct.get("resultSchemaKeys") or [])
    after_keys = set(after_correct.get("resultSchemaKeys") or [])
    missing_keys = sorted(before_keys - after_keys)
    if missing_keys:
        logic.append({"type": "schema_keys_removed", "missingKeys": missing_keys})

    before_names = sorted(before_correct.get("violationNames") or [])
    after_names = sorted(after_correct.get("violationNames") or [])
    if before_names != after_names:
        logic.append({"type": "violation_names_changed", "before": before_names, "after": after_names})

    passed = not performance and not logic
    return {
        "schemaVersion": 1,
        "passed": passed,
        "performanceRegressions": performance,
        "logicRegressions": logic,
        "summary": {
            "beforeCompletedRuns": len(before_runs),
            "afterCompletedRuns": len(after_runs),
            "beforeAverageTotalDurationSeconds": before_total,
            "afterAverageTotalDurationSeconds": after_total,
            "beforeAverageDetectorInferenceSeconds": before_detector,
            "afterAverageDetectorInferenceSeconds": after_detector,
        },
    }


def _write_outputs(output: Path, payload: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "backend_benchmark_comparison.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    lines = [
        "# SafeTrace Backend Benchmark Comparison",
        "",
        f"- Passed: {payload['passed']}",
        f"- Performance regressions: {len(payload['performanceRegressions'])}",
        f"- Logic regressions: {len(payload['logicRegressions'])}",
        "",
    ]
    (output / "backend_benchmark_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare SafeTrace backend benchmark reports.")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tmp_backend_benchmark") / "comparison")
    parser.add_argument("--duration-tolerance", type=float, default=0.15)
    parser.add_argument("--detector-tolerance", type=float, default=0.15)
    parser.add_argument("--evidence-drop-tolerance", type=float, default=0.20)
    args = parser.parse_args(argv)

    payload = compare_reports(
        _load(args.before),
        _load(args.after),
        duration_tolerance=args.duration_tolerance,
        detector_tolerance=args.detector_tolerance,
        evidence_drop_tolerance=args.evidence_drop_tolerance,
    )
    _write_outputs(args.output, payload)
    print(f"Comparison passed: {payload['passed']}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
