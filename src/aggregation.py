"""Presentation-level aggregation for frame findings.

Aggregation groups nearby frame-level findings into potential video events.
It does not change detector outputs, thresholds, model inference, or rule
decisions; raw frame evidence remains available in the normalized result.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def timestamp_to_seconds(value: str | None) -> int:
    if not value:
        return 0
    parts = [int(part) for part in str(value).split(":") if part.isdigit()]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] if parts else 0


def aggregate_violation_events(
    frames: Iterable[Dict[str, Any]],
    *,
    merge_gap_seconds: int = 5,
) -> List[Dict[str, Any]]:
    events: list[dict[str, Any]] = []
    active_by_type: dict[str, dict[str, Any]] = {}

    for frame in sorted(frames, key=lambda item: timestamp_to_seconds(item.get("timestamp"))):
        frame_second = timestamp_to_seconds(frame.get("timestamp"))
        for violation in frame.get("violations") or []:
            violation_id = str(violation.get("id") or "unknown_violation")
            confidence = float(violation.get("confidence") or 0.0)
            severity = str(violation.get("severity") or "medium").lower()
            support = {
                "frameId": frame.get("id"),
                "frameNumber": frame.get("frameNumber"),
                "timestamp": frame.get("timestamp"),
                "confidence": confidence,
                "imageUrl": frame.get("imageUrl"),
                "evidenceStrength": violation.get("evidenceStrength"),
                "confidenceReason": violation.get("confidenceReason"),
                "verifierAgreement": violation.get("verifierAgreement"),
            }

            existing = active_by_type.get(violation_id)
            if existing is None or frame_second - int(existing["lastSecond"]) > merge_gap_seconds:
                event = {
                    "id": f"{violation_id}_event_{len(events) + 1}",
                    "type": violation_id,
                    "name": violation.get("name") or violation_id,
                    "severity": severity,
                    "description": violation.get("description") or "",
                    "startTimestamp": frame.get("timestamp"),
                    "endTimestamp": frame.get("timestamp"),
                    "startSecond": frame_second,
                    "endSecond": frame_second,
                    "lastSecond": frame_second,
                    "supportingFrames": [support],
                    "confidences": [confidence],
                    "evidenceStrengths": [violation.get("evidenceStrength")],
                    "verifierAgreements": [violation.get("verifierAgreement")],
                }
                events.append(event)
                active_by_type[violation_id] = event
                continue

            existing["endTimestamp"] = frame.get("timestamp")
            existing["endSecond"] = frame_second
            existing["lastSecond"] = frame_second
            existing["supportingFrames"].append(support)
            existing["confidences"].append(confidence)
            existing.setdefault("evidenceStrengths", []).append(violation.get("evidenceStrength"))
            existing.setdefault("verifierAgreements", []).append(violation.get("verifierAgreement"))
            if SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(str(existing["severity"]).lower(), 0):
                existing["severity"] = severity

    normalized_events: list[dict[str, Any]] = []
    for event in events:
        confidences = [float(value) for value in event.pop("confidences")]
        evidence_strengths = [str(value) for value in event.pop("evidenceStrengths", []) if value]
        verifier_agreements = [str(value) for value in event.pop("verifierAgreements", []) if value]
        supporting_frames = list(event["supportingFrames"])
        representative_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        normalized_events.append(
            {
                "id": event["id"],
                "type": event["type"],
                "name": event["name"],
                "severity": event["severity"],
                "description": event["description"],
                "startTimestamp": event["startTimestamp"],
                "endTimestamp": event["endTimestamp"],
                "representativeConfidence": representative_confidence,
                "confidenceMin": min(confidences) if confidences else 0.0,
                "confidenceMax": max(confidences) if confidences else 0.0,
                "evidenceStrengths": sorted(set(evidence_strengths)),
                "verifierAgreements": sorted(set(verifier_agreements)),
                "supportingFrameCount": len(supporting_frames),
                "supportingFrames": supporting_frames,
            }
        )

    normalized_events.sort(
        key=lambda item: (
            SEVERITY_RANK.get(str(item["severity"]).lower(), 0),
            item["representativeConfidence"],
            item["supportingFrameCount"],
        ),
        reverse=True,
    )
    return normalized_events


def summarize_events(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    event_list = list(events)
    event_types = sorted({str(event.get("type") or "") for event in event_list if event.get("type")})
    confidences = [float(event.get("representativeConfidence") or 0.0) for event in event_list]
    highest = None
    for event in event_list:
        severity = str(event.get("severity") or "").lower()
        if highest is None or SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(highest, 0):
            highest = severity
    return {
        "potentialEventCount": len(event_list),
        "eventTypes": event_types,
        "overallConfidence": max(confidences) if confidences else 0.0,
        "highestSeverity": highest,
        "keyEvents": event_list[:3],
    }
