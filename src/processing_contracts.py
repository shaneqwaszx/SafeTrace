"""Versioned processing-contract comparison helpers."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def require_baseline_update_reason(reason: str | None) -> str:
    value = str(reason or "").strip()
    if not value:
        raise ValueError("A processing-contract baseline update requires a non-empty reason.")
    return value


def compare_processing_contract(
    contract: Mapping[str, Any], observed_controls: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    observed_by_key = {
        (str(item.get("controlId")), str(item.get("profile"))): item
        for item in observed_controls
    }
    checks: list[dict[str, Any]] = []
    for expected in contract.get("controls") or []:
        key = (str(expected.get("controlId")), str(expected.get("profile")))
        observed = observed_by_key.get(key)
        if observed is None:
            checks.append({"key": key, "passed": False, "reason": "missing_observed_control"})
            continue
        failures: list[str] = []
        for field, expected_value in (expected.get("invariants") or {}).items():
            if expected_value is not None and observed.get(field) != expected_value:
                failures.append(f"invariant_changed:{field}")
        behavior = dict(expected.get("behavior") or {})
        findings = set(str(item) for item in observed.get("acceptedFindingNames") or [])
        required = set(str(item) for item in behavior.get("requiredFindings") or [])
        absent = set(str(item) for item in behavior.get("absentFindings") or [])
        if not required.issubset(findings):
            failures.append("required_finding_missing")
        if findings & absent:
            failures.append("forbidden_finding_present")
        for field, operator in (("postGateCandidates", "candidate"), ("evidenceCount", "evidence")):
            value = int(observed.get(field) or 0)
            minimum = behavior.get(f"min{operator.title()}Count")
            maximum = behavior.get(f"max{operator.title()}Count")
            if minimum is not None and value < int(minimum):
                failures.append(f"below_minimum:{field}")
            if maximum is not None and value > int(maximum):
                failures.append(f"above_maximum:{field}")
        if behavior.get("profileApplicable") is not None and bool(observed.get("profileApplicable")) != bool(behavior["profileApplicable"]):
            failures.append("profile_applicability_changed")
        if behavior.get("reviewOnlyFinding"):
            review = dict(observed.get("reviewLevels") or {})
            if review.get(str(behavior["reviewOnlyFinding"])) != "review_candidate":
                failures.append("review_finding_promoted_or_missing")
        checks.append({"key": list(key), "passed": not failures, "failures": failures})
    return {
        "contractVersion": contract.get("contractVersion"),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
