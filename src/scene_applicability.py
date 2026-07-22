"""Fail-closed scene applicability checks for profile-specific findings."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .profile_composition import add_finding_provenance, applicability_profile_id
from .schemas import Detection, Violation

PERSON_LABELS = {"person", "driver", "occupant", "worker", "head", "hand", "torso"}
CABIN_LABELS = {
    "driver", "occupant", "torso", "seatbelt", "seat_belt", "belt",
    "steering_wheel", "dashboard", "vehicle_interior", "car_interior", "windshield",
}
DRIVER_REGION_LABELS = {"driver", "occupant", "torso", "steering_wheel", "dashboard", "hand"}
EXTERIOR_LABELS = {
    "tree", "vegetation", "grass", "road", "street", "sky", "car", "truck", "bus",
    "motorcycle", "bicycle", "traffic_light", "traffic_sign", "building", "landscape",
}
VEGETATION_LABELS = {"tree", "vegetation", "grass", "plant", "bush", "landscape"}

IN_CABIN_FINDINGS = {
    "seatbelt_missing", "hands_off_steering_wheel", "phone_use", "driver_distraction",
}
PPE_FINDINGS = {"helmet_missing", "ppe_missing"}


@dataclass(frozen=True)
class SceneApplicability:
    person_visible: bool
    vehicle_interior_visible: bool
    driver_region_visible: bool
    torso_visible: bool
    head_visible: bool
    phone_candidate: bool
    seatbelt_evidence: bool
    external_road_scene: bool
    exterior_only: bool
    vegetation_dominant: bool
    insufficient_visibility: bool
    profile_applicable: bool
    manual_review_required: bool
    vehicle_interior_evidence: str
    reason: str
    labels: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            "personVisible": payload.pop("person_visible"),
            "vehicleInteriorVisible": payload.pop("vehicle_interior_visible"),
            "driverRegionVisible": payload.pop("driver_region_visible"),
            "torsoVisible": payload.pop("torso_visible"),
            "headVisible": payload.pop("head_visible"),
            "phoneCandidate": payload.pop("phone_candidate"),
            "seatbeltEvidence": payload.pop("seatbelt_evidence"),
            "externalRoadScene": payload.pop("external_road_scene"),
            "exteriorOnly": payload.pop("exterior_only"),
            "vegetationDominant": payload.pop("vegetation_dominant"),
            "insufficientVisibility": payload.pop("insufficient_visibility"),
            "profileApplicable": payload.pop("profile_applicable"),
            "manualReviewRequired": payload.pop("manual_review_required"),
            "vehicleInteriorEvidence": payload.pop("vehicle_interior_evidence"),
            "reason": payload["reason"],
            "detectedLabels": list(payload["labels"]),
        }


def _labels(detections: Iterable[Detection]) -> tuple[str, ...]:
    return tuple(sorted({str(item.label or "").strip().lower() for item in detections if item.label}))


def assess_scene(detections: Sequence[Detection]) -> SceneApplicability:
    labels = _labels(detections)
    label_set = set(labels)
    person = bool(label_set & PERSON_LABELS)
    direct_cabin = bool(label_set & CABIN_LABELS)
    exterior_signal = bool(label_set & EXTERIOR_LABELS) and not direct_cabin
    # The fallback COCO detector cannot emit cabin/torso/seatbelt classes. A
    # visible person with no exterior signal is therefore retained as an
    # explicitly inferred review context, never as confirmed cabin evidence.
    inferred_cabin_review = person and not direct_cabin and not exterior_signal
    cabin = direct_cabin or inferred_cabin_review
    driver = bool(label_set & DRIVER_REGION_LABELS) and person
    torso = bool(label_set & {"torso", "driver", "occupant"}) and person
    head = bool(label_set & {"head", "helmet", "hardhat", "hard_hat"}) and person
    phone = bool(label_set & {"phone", "cell_phone", "mobile_phone"}) and person
    seatbelt = bool(label_set & {"seatbelt", "seat_belt", "belt"}) and person
    exterior = exterior_signal
    exterior_only = exterior_signal and not direct_cabin
    vegetation = bool(label_set & VEGETATION_LABELS) and not person and not cabin
    insufficient = not person and not cabin
    applicable = person and not exterior_only
    manual_review = applicable and inferred_cabin_review
    reason = (
        "applicable_subject_and_direct_context_visible"
        if applicable and direct_cabin
        else "applicable_subject_inferred_interior_review"
        if manual_review
        else "no_applicable_subject_or_scene"
    )
    return SceneApplicability(
        person_visible=person,
        vehicle_interior_visible=cabin,
        driver_region_visible=driver,
        torso_visible=torso,
        head_visible=head,
        phone_candidate=phone,
        seatbelt_evidence=seatbelt,
        external_road_scene=exterior,
        exterior_only=exterior_only,
        vegetation_dominant=vegetation,
        insufficient_visibility=insufficient,
        profile_applicable=applicable,
        manual_review_required=manual_review,
        vehicle_interior_evidence=(
            "direct_detector_context" if direct_cabin else "inferred_person_without_exterior_context"
            if inferred_cabin_review else "none"
        ),
        reason=reason,
        labels=labels,
    )


def finding_applicability(
    violation: Violation,
    scene: SceneApplicability,
    profile_id: str | None = None,
) -> tuple[bool, str]:
    name = str(violation.name or "").strip().lower()
    labels = set(scene.labels)
    selected_profile = str(
        profile_id or (violation.evidence or {}).get("selectedProfileId") or "general_safety"
    ).strip().lower()
    if name == "seatbelt_missing":
        torso_or_cabin = bool(labels & {"torso", "driver", "occupant", "seatbelt", "seat_belt", "belt", "steering_wheel", "dashboard", "vehicle_interior", "car_interior"})
        if not scene.person_visible or scene.exterior_only:
            return False, "no_applicable_subject_or_scene"
        if not torso_or_cabin:
            if selected_profile != "seatbelt_compliance":
                return False, "no_direct_seatbelt_context_for_selected_profile"
            return True, "inferred_interior_person_context_review_required"
    elif name == "hands_off_steering_wheel":
        steering_context = bool(labels & {"steering_wheel", "dashboard", "vehicle_interior", "car_interior"})
        if not scene.person_visible or not steering_context:
            return False, "no_driver_or_steering_context"
    elif name == "phone_use":
        if not scene.person_visible or "phone" not in labels:
            return False, "no_person_and_phone_context"
    elif name in PPE_FINDINGS:
        if not scene.person_visible:
            return False, "no_applicable_subject_or_scene"
        direct_support = set(violation.evidence.get("directSupportLabels") or []) | labels
        if violation.evidence.get("ruleSupport") == "person_proxy_only" or not (
            direct_support & {"head", "helmet", "hardhat", "hard_hat"}
        ):
            return False, "no_head_or_ppe_context"
    elif name in IN_CABIN_FINDINGS and not scene.profile_applicable:
        return False, "no_applicable_subject_or_scene"
    return True, "profile_context_satisfied"


def apply_scene_applicability_gate(
    detections: Sequence[Detection],
    violations: Sequence[Violation],
    *,
    profile_id: str | None = None,
) -> tuple[list[Violation], list[dict[str, Any]], SceneApplicability]:
    scene = assess_scene(detections)
    accepted: list[Violation] = []
    accepted_keys: set[str] = set()
    suppressed: list[dict[str, Any]] = []
    scene_payload = scene.to_payload()
    for violation in violations:
        effective_profile_id = applicability_profile_id(profile_id, violation.name)
        applicable, reason = finding_applicability(violation, scene, effective_profile_id)
        violation.evidence = dict(violation.evidence or {})
        violation.evidence.update(
            {
                "sceneApplicability": scene_payload,
                "sceneApplicable": scene.profile_applicable,
                "candidateViolation": True,
                "violationAccepted": applicable,
                "selectedProfileId": str(profile_id or "general_safety"),
                "applicabilityProfileId": effective_profile_id,
                "manualReviewRequired": bool(scene.manual_review_required),
                "profileApplicable": applicable,
                "sceneApplicabilityReason": reason,
                "personDetected": scene.person_visible,
                "vehicleInteriorVisible": scene.vehicle_interior_visible,
                "driverRegionVisible": scene.driver_region_visible,
                "finalAcceptanceReason": reason if applicable else "suppressed_by_scene_applicability_gate",
            }
        )
        if applicable:
            if reason == "inferred_interior_person_context_review_required":
                violation.confidence = min(float(violation.confidence), 0.49)
                violation.evidence.update(
                    {
                        "reviewRequired": True,
                        "manualReviewRequired": True,
                        "evidenceStrength": "review_candidate",
                        "ruleSupport": "person_inferred_interior_context",
                        "confidenceReason": (
                            "A person is visible without exterior-scene signals, but the fallback detector "
                            "cannot directly confirm the torso or belt path."
                        ),
                        "finalReviewerNote": "Confirm the belt path in the original footage.",
                    }
                )
            violation.evidence = add_finding_provenance(
                violation.evidence,
                finding_name=violation.name,
                selected_profile_id=profile_id,
                applicable=True,
                applicability_reason=reason,
                confidence=float(violation.confidence),
            )
            deduplication_key = str(violation.evidence.get("deduplicationKey") or violation.name)
            if deduplication_key in accepted_keys:
                suppressed.append(
                    {
                        "name": violation.name,
                        "confidence": float(violation.confidence),
                        "reason": "duplicate_general_profile_candidate",
                        "deduplicationKey": deduplication_key,
                        "sceneApplicability": scene_payload,
                    }
                )
                continue
            accepted_keys.add(deduplication_key)
            accepted.append(violation)
        else:
            violation.evidence = add_finding_provenance(
                violation.evidence,
                finding_name=violation.name,
                selected_profile_id=profile_id,
                applicable=False,
                applicability_reason=reason,
                confidence=float(violation.confidence),
            )
            suppressed.append(
                {
                    "name": violation.name,
                    "confidence": float(violation.confidence),
                    "reason": reason,
                    "originProfileId": violation.evidence.get("originProfileId"),
                    "originProfileLabel": violation.evidence.get("originProfileLabel"),
                    "originRule": violation.evidence.get("originRule"),
                    "deduplicationKey": violation.evidence.get("deduplicationKey"),
                    "sceneApplicability": scene_payload,
                }
            )
    return accepted, suppressed, scene


MID_VLM_REQUIRED_FIELDS = {
    "sceneType": str,
    "personVisible": bool,
    "vehicleInteriorVisible": bool,
    "driverRegionVisible": bool,
    "profileApplicable": bool,
    "violationSupported": bool,
    "violationType": str,
    "confidence": (int, float),
    "reason": str,
}


def mid_vlm_scene_prompt(*, finding: str, profile: str, query: str) -> str:
    return (
        "Inspect only the supplied evidence image. Return only one compact JSON object, with no markdown or prose, "
        "using exactly these keys: "
        "sceneType, personVisible, vehicleInteriorVisible, driverRegionVisible, profileApplicable, "
        "violationSupported, violationType, confidence, reason. "
        f"Profile: {profile or 'general_safety'}. Finding candidate: {finding}. Query: {query}. "
        "Use JSON booleans, a confidence from 0 to 1, and keep reason under 12 words. "
        "The verifier may confirm context or express uncertainty, but must not invent a new violation."
    )


def parse_mid_vlm_scene_response(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        payload = json.loads(text)
    else:
        payload = dict(value)
    if set(payload) != set(MID_VLM_REQUIRED_FIELDS):
        raise ValueError("mid_vlm_schema_keys")
    for key, expected in MID_VLM_REQUIRED_FIELDS.items():
        if not isinstance(payload[key], expected) or isinstance(payload[key], bool) and key == "confidence":
            raise ValueError(f"mid_vlm_schema_type:{key}")
    confidence = float(payload["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("mid_vlm_confidence_range")
    payload["confidence"] = confidence
    return payload


def apply_mid_vlm_verification(
    violation: Violation,
    scene: SceneApplicability,
    response: Mapping[str, Any],
) -> tuple[bool, str]:
    """Merge verifier diagnostics while keeping baseline/scene rules authoritative."""
    payload = parse_mid_vlm_scene_response(response)
    violation.evidence = dict(violation.evidence or {})
    violation.evidence["midVlmSceneVerification"] = payload
    violation.evidence["midVlmNonAuthoritative"] = True
    baseline_applicable, baseline_reason = finding_applicability(violation, scene)
    if not baseline_applicable:
        return False, baseline_reason
    if not payload["profileApplicable"]:
        violation.evidence["reviewRequired"] = True
        return False, "mid_vlm_scene_disagreement_conservative_suppression"
    if not payload["violationSupported"]:
        violation.evidence["reviewRequired"] = True
        return True, "mid_vlm_disagrees_manual_review"
    return True, "baseline_applicable_mid_vlm_supports"
