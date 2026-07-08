"""Normalize SafeTrace pipeline output into a stable API response."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.aggregation import aggregate_violation_events, summarize_events
from src.config import SETTINGS
from src.preprocessing import build_processing_metadata

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
EVIDENCE_STRENGTH_RANK = {
    "unsupported_rule": 0,
    "insufficient_evidence": 1,
    "review_candidate": 2,
    "likely_violation": 3,
    "confirmed_violation": 4,
}


def default_evidence_strength(confidence: float) -> str:
    if confidence >= 0.8:
        return "confirmed_violation"
    if confidence >= 0.6:
        return "likely_violation"
    if confidence > 0:
        return "review_candidate"
    return "insufficient_evidence"


def strongest_evidence_strength(values: Iterable[str]) -> str:
    normalized = [str(value or "").strip() for value in values if value]
    if not normalized:
        return "insufficient_evidence"
    return max(normalized, key=lambda value: EVIDENCE_STRENGTH_RANK.get(value, 0))


def humanize_violation_name(value: str) -> str:
    normalized = (value or "").strip().replace("-", "_")
    overrides = {
        "helmet_missing": "Missing Helmet",
        "seatbelt_missing": "Missing Seatbelt",
        "hands_off_steering_wheel": "Hands Off Steering Wheel",
        "phone_use": "Phone Use",
    }
    if normalized in overrides:
        return overrides[normalized]
    return " ".join(part.capitalize() for part in normalized.split("_") if part)


def timestamp_from_frame_id(frame_id: str) -> str:
    match = re.search(r"_(\d{6})$", frame_id or "")
    if not match:
        return "00:00:00"
    seconds = int(match.group(1))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def safe_output_filename(frame_id: str, annotated_path: Path) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", frame_id or annotated_path.stem).strip("._")
    suffix = annotated_path.suffix.lower() if annotated_path.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
    return f"{safe_stem}_annotated{suffix}"


def copy_annotated_image(
    *,
    job_id: str,
    frame_id: str,
    annotated_path: Optional[str],
    media_dir: Path,
    register_media: Callable[[str, Path], None],
) -> Tuple[Optional[str], Optional[str]]:
    if not annotated_path:
        return None, "No annotated evidence image was produced for this frame."

    source = Path(annotated_path)
    if not source.exists() or not source.is_file():
        return None, f"Annotated evidence image is unavailable: {source.name}"
    if source.suffix.lower() not in IMAGE_SUFFIXES:
        return None, f"Unsupported annotated evidence image type: {source.suffix}"

    media_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_output_filename(frame_id, source)
    destination = media_dir / filename
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    register_media(filename, destination)
    return f"/api/media/{job_id}/{filename}", None


def normalize_pipeline_results(
    *,
    job_id: str,
    media_name: str,
    media_type: str,
    media_size_bytes: int,
    query: str,
    raw_frames: Iterable[Dict[str, Any]],
    media_dir: Path,
    register_media: Callable[[str, Path], None],
) -> Dict[str, Any]:
    raw_frame_list = list(raw_frames)
    frames: List[Dict[str, Any]] = []
    grouped: Dict[str, Dict[str, Any]] = {}

    for index, raw in enumerate(raw_frame_list, start=1):
        frame_id = str(raw.get("frame_id") or f"frame_{index:03d}")
        timestamp = timestamp_from_frame_id(frame_id)
        raw_violations = list(raw.get("violations") or [])

        frame_violations: List[Dict[str, Any]] = []
        for raw_violation in raw_violations:
            violation_id = str(raw_violation.get("name") or "unknown_violation")
            severity = str(raw_violation.get("severity") or "medium").lower()
            confidence = float(raw_violation.get("confidence") or 0.0)
            description = str(raw_violation.get("description") or "")
            name = humanize_violation_name(violation_id)
            evidence = dict(raw_violation.get("evidence") or {})
            evidence_strength = str(
                evidence.get("evidenceStrength") or default_evidence_strength(confidence)
            )
            confidence_reason = str(
                evidence.get("confidenceReason")
                or "Confidence is derived from detector/rule overlap and calibrated evidence strength."
            )

            frame_violations.append(
                {
                    "id": violation_id,
                    "name": name,
                    "severity": severity,
                    "confidence": confidence,
                    "description": description,
                    "evidence": evidence,
                    "evidenceStrength": evidence_strength,
                    "confidenceReason": confidence_reason,
                    "reviewRequired": bool(evidence.get("reviewRequired")),
                    "ruleSupport": evidence.get("ruleSupport"),
                    "suppressedFindings": evidence.get("suppressedFindings") or [],
                    "unsupportedRuleReason": evidence.get("unsupportedRuleReason"),
                    "verifierAgreement": evidence.get("verifierAgreement"),
                    "verifierDisagreementReason": evidence.get("verifierDisagreementReason"),
                    "verifierConfidenceHint": evidence.get("verifierConfidenceHint"),
                    "finalReviewerNote": evidence.get("finalReviewerNote"),
                }
            )

            group = grouped.setdefault(
                violation_id,
                {
                    "id": violation_id,
                    "name": name,
                    "severity": severity,
                    "description": description,
                    "affectedFrames": [],
                    "confidences": [],
                    "evidenceStrengths": [],
                    "confidenceReasons": [],
                    "reviewRequired": False,
                    "ruleSupport": evidence.get("ruleSupport"),
                    "verifierAgreement": evidence.get("verifierAgreement"),
                    "verifierDisagreementReason": evidence.get("verifierDisagreementReason"),
                    "finalReviewerNote": evidence.get("finalReviewerNote"),
                },
            )
            if SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(group["severity"], 0):
                group["severity"] = severity
            group["affectedFrames"].append(
                {
                    "frameId": frame_id,
                    "frameNumber": index,
                    "timestamp": timestamp,
                    "confidence": confidence,
                    "evidenceStrength": evidence_strength,
                    "confidenceReason": confidence_reason,
                    "verifierAgreement": evidence.get("verifierAgreement"),
                }
            )
            group["confidences"].append(confidence)
            group["evidenceStrengths"].append(evidence_strength)
            group["confidenceReasons"].append(confidence_reason)
            group["reviewRequired"] = bool(group.get("reviewRequired") or evidence.get("reviewRequired"))
            if evidence.get("verifierAgreement") == "disagrees":
                group["verifierAgreement"] = "disagrees"
                group["verifierDisagreementReason"] = evidence.get("verifierDisagreementReason")
            elif not group.get("verifierAgreement") and evidence.get("verifierAgreement"):
                group["verifierAgreement"] = evidence.get("verifierAgreement")
            if evidence.get("finalReviewerNote"):
                group["finalReviewerNote"] = evidence.get("finalReviewerNote")

        image_url, image_message = copy_annotated_image(
            job_id=job_id,
            frame_id=frame_id,
            annotated_path=raw.get("annotated_path"),
            media_dir=media_dir,
            register_media=register_media,
        )

        frames.append(
            {
                "id": frame_id,
                "frameNumber": index,
                "timestamp": timestamp,
                "queryRelevance": float(raw.get("score") or 0.0),
                "status": "violations_detected" if frame_violations else "no_violations",
                "imageUrl": image_url,
                "imageMessage": image_message,
                "explanationSource": raw.get("explanation_source") or raw.get("explanationSource"),
                "violations": frame_violations,
                "technicalEvidence": {
                    "sourceFramePath": raw.get("frame_path"),
                    "annotatedPath": raw.get("annotated_path"),
                    "detections": raw.get("detections") or [],
                    "explanation": raw.get("explanation"),
                    "explanationSource": raw.get("explanation_source") or raw.get("explanationSource"),
                    "searchMetadata": raw.get("search_metadata") or {},
                    "raw": raw,
                },
            }
        )

    grouped_violations: List[Dict[str, Any]] = []
    for group in grouped.values():
        confidences = group.pop("confidences")
        evidence_strengths = group.pop("evidenceStrengths", [])
        confidence_reasons = group.pop("confidenceReasons", [])
        grouped_violations.append(
            {
                **group,
                "confidenceMin": min(confidences),
                "confidenceMax": max(confidences),
                "evidenceStrength": strongest_evidence_strength(evidence_strengths),
                "confidenceReasons": sorted(set(str(reason) for reason in confidence_reasons if reason)),
            }
        )
    grouped_violations.sort(
        key=lambda item: SEVERITY_RANK.get(str(item["severity"]).lower(), 0),
        reverse=True,
    )

    events = aggregate_violation_events(frames, merge_gap_seconds=5)
    event_summary = summarize_events(events)
    frames_with_violations = sum(1 for frame in frames if frame["violations"])
    highest = event_summary.get("highestSeverity") or (grouped_violations[0]["severity"] if grouped_violations else None)
    summary_text = (
        "SafeTrace grouped repeated frame findings into potential video-level events."
        if events
        else "SafeTrace found safety findings across selected evidence frames."
        if grouped_violations
        else "No matching safety violations were detected in the selected frames."
    )
    processing_metadata = next(
        (
            dict(raw.get("processing_metadata") or {})
            for raw in raw_frame_list
            if raw.get("processing_metadata")
        ),
        None,
    )
    if processing_metadata is None:
        processing_metadata = build_processing_metadata(
            sampled_frame_count=len(frames),
            sampling_strategy="api_normalized_frames",
            fps=None,
            max_frames=SETTINGS.max_frames,
            embedding_batch_size=SETTINGS.embedding_batch_size,
            embedding_window_size=SETTINGS.embedding_window_size,
            embedding_window_stride=SETTINGS.embedding_window_stride,
            embedding_pooling_strategy=SETTINGS.embedding_pooling_strategy,
            processing_window_count=len(frames),
        )

    return {
        "jobId": job_id,
        "status": "completed",
        "media": {
            "id": f"media_{job_id}",
            "name": media_name,
            "type": media_type,
            "sizeBytes": media_size_bytes,
            "durationSeconds": None,
        },
        "query": query,
        "summary": {
            "framesAnalyzed": len(frames),
            "framesWithViolations": frames_with_violations,
            "uniqueViolationTypes": len(grouped_violations),
            "highestSeverity": highest,
            "summaryText": summary_text,
            "potentialEventCount": event_summary["potentialEventCount"],
            "eventTypes": event_summary["eventTypes"],
            "overallConfidence": event_summary["overallConfidence"],
            "keyEvents": event_summary["keyEvents"],
        },
        "violations": grouped_violations,
        "events": events,
        "frames": frames,
        "technicalDetails": {
            "normalizer": "safetrace-api-v1",
            "processingMetadata": processing_metadata,
            "eventAggregation": {
                "mergeGapSeconds": 5,
                "eventCount": len(events),
                "note": "Presentation-level grouping only; detector and rule outputs are unchanged.",
            },
        },
    }
