"""Backend-owned profile composition and finding provenance."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


GENERAL_PROFILE_ID = "general_safety"

GENERAL_PROFILE_COMPONENTS: tuple[dict[str, Any], ...] = (
    {
        "profileId": "seatbelt_compliance",
        "profileLabel": "Seatbelt compliance",
        "supportLevel": "partial",
        "findingNames": ("seatbelt_missing",),
        "originRules": {"seatbelt_missing": "rule_seatbelt_missing"},
    },
    {
        "profileId": "phone_use",
        "profileLabel": "Phone use / distracted driving",
        "supportLevel": "conditional_rule_support",
        "findingNames": ("phone_use", "driver_distraction"),
        "originRules": {"phone_use": "rule_phone_use", "driver_distraction": "rule_phone_use"},
    },
    {
        "profileId": "helmet_ppe",
        "profileLabel": "Helmet / PPE compliance",
        "supportLevel": "partial",
        "findingNames": ("helmet_missing", "ppe_missing"),
        "originRules": {"helmet_missing": "rule_helmet_missing", "ppe_missing": "rule_helmet_missing"},
    },
    {
        "profileId": "hands_on_wheel",
        "profileLabel": "Hands on steering wheel",
        "supportLevel": "conditional_rule_support",
        "findingNames": ("hands_off_steering_wheel",),
        "originRules": {"hands_off_steering_wheel": "rule_hands_on_wheel"},
    },
)

GENERAL_PROFILE_EXCLUSIONS: tuple[dict[str, str], ...] = (
    {"profileId": "uniform_compliance", "reason": "metadata_only_no_detector_or_rule_support"},
    {"profileId": "custom_policy", "reason": "metadata_only_reviewer_context"},
    {"profileId": "person_near_machinery", "reason": "no_registered_backend_rule"},
)


def general_profile_registry_payload() -> dict[str, Any]:
    return {
        "profileId": GENERAL_PROFILE_ID,
        "profileLabel": "General safety review",
        "compositionPolicy": "union_of_supported_profile_candidates_with_origin_applicability",
        "includedProfiles": deepcopy(list(GENERAL_PROFILE_COMPONENTS)),
        "excludedProfiles": deepcopy(list(GENERAL_PROFILE_EXCLUSIONS)),
    }


def finding_origin(finding_name: str) -> dict[str, str]:
    normalized = str(finding_name or "").strip().lower()
    for profile in GENERAL_PROFILE_COMPONENTS:
        if normalized in profile["findingNames"]:
            return {
                "originProfileId": str(profile["profileId"]),
                "originProfileLabel": str(profile["profileLabel"]),
                "originRule": str(profile["originRules"].get(normalized) or normalized),
            }
    return {
        "originProfileId": GENERAL_PROFILE_ID,
        "originProfileLabel": "General safety review",
        "originRule": normalized or "unknown_rule",
    }


def applicability_profile_id(selected_profile_id: str | None, finding_name: str) -> str:
    selected = str(selected_profile_id or GENERAL_PROFILE_ID).strip().lower()
    return finding_origin(finding_name)["originProfileId"] if selected == GENERAL_PROFILE_ID else selected


def add_finding_provenance(
    evidence: Mapping[str, Any] | None,
    *,
    finding_name: str,
    selected_profile_id: str | None,
    applicable: bool,
    applicability_reason: str,
    confidence: float,
) -> dict[str, Any]:
    payload = dict(evidence or {})
    origin = finding_origin(finding_name)
    review_level = str(
        payload.get("evidenceStrength")
        or ("review_candidate" if bool(payload.get("reviewRequired")) or confidence < 0.60 else "likely_violation")
    )
    payload.update(
        {
            **origin,
            "profileApplicability": {
                "applicable": bool(applicable),
                "reason": applicability_reason,
                "evaluatedAsProfileId": applicability_profile_id(selected_profile_id, finding_name),
            },
            "reviewLevel": review_level,
            "deduplicationKey": f"{origin['originProfileId']}:{str(finding_name).strip().lower()}",
        }
    )
    return payload
