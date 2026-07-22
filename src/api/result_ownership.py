"""Fail-closed ownership stamping and validation for job result payloads."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

RESULT_SCHEMA_VERSION = 2
INTEGRITY_ERROR_CODE = "cross_job_result_contamination"
_MEDIA_JOB_RE = re.compile(r"/api/media/([^/]+)/")


@dataclass(frozen=True)
class ResultOwnershipContext:
    job_id: str
    batch_id: str | None
    source_checksum: str | None
    original_filename: str
    source_relative_path: str
    source_group_path: str
    execution_identity: Mapping[str, Any]


class ResultOwnershipError(ValueError):
    """Raised when a result contains content owned by another job."""

    code = INTEGRITY_ERROR_CODE

    def __init__(self, issues: Iterable[str]):
        self.issues = tuple(str(issue) for issue in issues)
        super().__init__("; ".join(self.issues) or "Result ownership validation failed")

    def detail(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": "SafeTrace blocked a result containing mismatched job ownership.",
            "ownershipIssues": list(self.issues),
        }


def _owner_fields(context: ResultOwnershipContext) -> dict[str, Any]:
    return {
        "jobId": context.job_id,
        "batchId": context.batch_id,
        "sourceChecksum": context.source_checksum,
        "originalFilename": context.original_filename,
        "sourceRelativePath": context.source_relative_path,
        "sourceGroupPath": context.source_group_path,
    }


def _evidence_id(job_id: str, frame_id: Any) -> str:
    return f"{job_id}:{str(frame_id or 'unknown')}"


def _media_artifact_id(job_id: str, image_url: Any) -> str | None:
    if not image_url:
        return None
    return f"{job_id}:{Path(str(image_url)).name}"


def _check_existing_owner(
    payload: Mapping[str, Any],
    *,
    location: str,
    context: ResultOwnershipContext,
    previous_job_id: str | None,
) -> list[str]:
    issues: list[str] = []
    owner = payload.get("jobId")
    allowed_jobs = {context.job_id}
    if previous_job_id:
        allowed_jobs.add(previous_job_id)
    if owner not in (None, "") and str(owner) not in allowed_jobs:
        issues.append(f"{location}.jobId={owner!r} is not owned by {context.job_id!r}")
    batch = payload.get("batchId")
    if previous_job_id is None and batch not in (None, "", context.batch_id):
        issues.append(f"{location}.batchId={batch!r} does not match {context.batch_id!r}")
    for key, expected in (
        ("sourceChecksum", context.source_checksum),
        ("originalFilename", context.original_filename),
        ("sourceRelativePath", context.source_relative_path),
        ("sourceGroupPath", context.source_group_path),
    ):
        actual = payload.get(key)
        if previous_job_id is None and actual not in (None, "", expected):
            issues.append(f"{location}.{key}={actual!r} does not match {expected!r}")
    image_url = payload.get("imageUrl")
    if image_url:
        match = _MEDIA_JOB_RE.search(str(image_url))
        if match and match.group(1) not in allowed_jobs:
            issues.append(f"{location}.imageUrl routes through foreign job {match.group(1)!r}")
    return issues


def stamp_result_ownership(
    result: Mapping[str, Any],
    context: ResultOwnershipContext,
    *,
    previous_job_id: str | None = None,
) -> dict[str, Any]:
    """Return a deep, job-owned snapshot and reject pre-existing foreign owners."""
    snapshot = copy.deepcopy(dict(result))
    issues: list[str] = []
    owner_fields = _owner_fields(context)

    def stamp(payload: dict[str, Any], location: str) -> None:
        issues.extend(
            _check_existing_owner(
                payload,
                location=location,
                context=context,
                previous_job_id=previous_job_id,
            )
        )
        payload.update(owner_fields)
        image_url = payload.get("imageUrl")
        if image_url and previous_job_id:
            payload["imageUrl"] = re.sub(
                rf"(/api/media/){re.escape(previous_job_id)}(/)",
                rf"\g<1>{context.job_id}\g<2>",
                str(image_url),
            )
        if image_url:
            payload["mediaArtifactId"] = _media_artifact_id(context.job_id, payload.get("imageUrl"))

    stamp(snapshot, "result")
    snapshot["executionIdentity"] = copy.deepcopy(dict(context.execution_identity))
    snapshot["resultSchemaVersion"] = RESULT_SCHEMA_VERSION

    media = snapshot.get("media")
    if isinstance(media, dict):
        stamp(media, "result.media")
    source_metadata = snapshot.get("sourceMetadata")
    if isinstance(source_metadata, dict):
        stamp(source_metadata, "result.sourceMetadata")

    frames_by_id: dict[str, dict[str, Any]] = {}
    stamped_frames: set[int] = set()

    def stamp_frame_collection(collection_name: str) -> None:
        for index, frame in enumerate(snapshot.get(collection_name) or []):
            location = f"result.{collection_name}[{index}]"
            if not isinstance(frame, dict):
                issues.append(f"{location} is not an object")
                continue
            if id(frame) not in stamped_frames:
                stamp(frame, location)
                stamped_frames.add(id(frame))
                frame_id = str(frame.get("id") or f"frame_{index + 1}")
                frame["evidenceId"] = _evidence_id(context.job_id, frame_id)
                frames_by_id.setdefault(frame_id, frame)
                technical = frame.get("technicalEvidence")
                if isinstance(technical, dict):
                    technical["ownership"] = copy.deepcopy(owner_fields)
                frame_source = frame.get("sourceMetadata")
                if isinstance(frame_source, dict):
                    stamp(frame_source, f"{location}.sourceMetadata")
                for violation_index, violation in enumerate(frame.get("violations") or []):
                    if not isinstance(violation, dict):
                        issues.append(f"{location}.violations[{violation_index}] is not an object")
                        continue
                    stamp(violation, f"{location}.violations[{violation_index}]")
                    violation["findingId"] = str(
                        violation.get("findingId") or violation.get("id") or "unknown"
                    )
                    violation["evidenceId"] = frame["evidenceId"]

    for collection_name in ("frames", "evidence", "diagnosticFrames"):
        stamp_frame_collection(collection_name)

    def stamp_event(event: dict[str, Any], location: str) -> None:
        stamp(event, location)
        event["eventId"] = str(event.get("eventId") or event.get("id") or "unknown")
        for support_index, support in enumerate(event.get("supportingFrames") or []):
            if not isinstance(support, dict):
                issues.append(f"{location}.supportingFrames[{support_index}] is not an object")
                continue
            stamp(support, f"{location}.supportingFrames[{support_index}]")
            support["eventId"] = event["eventId"]
            support["findingId"] = str(support.get("findingId") or event.get("type") or "unknown")
            source_frame = frames_by_id.get(str(support.get("frameId") or ""), {})
            support["evidenceId"] = str(
                source_frame.get("evidenceId") or _evidence_id(context.job_id, support.get("frameId"))
            )
            support["mediaArtifactId"] = source_frame.get("mediaArtifactId")

    for index, violation in enumerate(snapshot.get("violations") or []):
        if not isinstance(violation, dict):
            issues.append(f"result.violations[{index}] is not an object")
            continue
        stamp(violation, f"result.violations[{index}]")
        violation["findingId"] = str(violation.get("findingId") or violation.get("id") or "unknown")
        for affected_index, affected in enumerate(violation.get("affectedFrames") or []):
            if not isinstance(affected, dict):
                issues.append(f"result.violations[{index}].affectedFrames[{affected_index}] is not an object")
                continue
            stamp(affected, f"result.violations[{index}].affectedFrames[{affected_index}]")
            affected["findingId"] = violation["findingId"]
            source_frame = frames_by_id.get(str(affected.get("frameId") or ""), {})
            affected["evidenceId"] = str(
                source_frame.get("evidenceId") or _evidence_id(context.job_id, affected.get("frameId"))
            )
            affected["mediaArtifactId"] = source_frame.get("mediaArtifactId")

    for index, event in enumerate(snapshot.get("events") or []):
        if isinstance(event, dict):
            stamp_event(event, f"result.events[{index}]")
        else:
            issues.append(f"result.events[{index}] is not an object")

    summary = snapshot.get("summary")
    if isinstance(summary, dict):
        stamp(summary, "result.summary")
        for index, event in enumerate(summary.get("keyEvents") or []):
            if isinstance(event, dict):
                stamp_event(event, f"result.summary.keyEvents[{index}]")

    technical = snapshot.setdefault("technicalDetails", {})
    if isinstance(technical, dict):
        technical["ownership"] = {
            **copy.deepcopy(owner_fields),
            "executionIdentity": copy.deepcopy(dict(context.execution_identity)),
            "resultSchemaVersion": RESULT_SCHEMA_VERSION,
        }
        technical_source = technical.get("sourceMetadata")
        if isinstance(technical_source, dict):
            stamp(technical_source, "result.technicalDetails.sourceMetadata")
    else:
        issues.append("result.technicalDetails is not an object")

    if issues:
        raise ResultOwnershipError(issues)
    validate_result_ownership(snapshot, context)
    return snapshot


def validate_evidence_ownership(evidence: Mapping[str, Any], expected_job_id: str) -> None:
    issues = []
    if str(evidence.get("jobId") or "") != expected_job_id:
        issues.append(f"evidence.jobId={evidence.get('jobId')!r} does not match {expected_job_id!r}")
    evidence_id = str(evidence.get("evidenceId") or "")
    if not evidence_id.startswith(f"{expected_job_id}:"):
        issues.append(f"evidence.evidenceId={evidence_id!r} is not namespaced to {expected_job_id!r}")
    image_url = evidence.get("imageUrl")
    match = _MEDIA_JOB_RE.search(str(image_url or ""))
    if match and match.group(1) != expected_job_id:
        issues.append(f"evidence.imageUrl routes through foreign job {match.group(1)!r}")
    artifact_id = str(evidence.get("mediaArtifactId") or "")
    if artifact_id and not artifact_id.startswith(f"{expected_job_id}:"):
        issues.append(f"evidence.mediaArtifactId={artifact_id!r} is foreign")
    if issues:
        raise ResultOwnershipError(issues)


def validate_result_ownership(result: Mapping[str, Any], context: ResultOwnershipContext) -> None:
    """Validate all result collections against one immutable owner."""
    issues: list[str] = []
    required_owner = _owner_fields(context)

    def validate_owner(payload: Any, location: str) -> None:
        if not isinstance(payload, Mapping):
            issues.append(f"{location} is not an object")
            return
        for key, expected in required_owner.items():
            actual = payload.get(key)
            if actual != expected:
                issues.append(f"{location}.{key}={actual!r} does not match {expected!r}")

    validate_owner(result, "result")
    if result.get("resultSchemaVersion") != RESULT_SCHEMA_VERSION:
        issues.append("result.resultSchemaVersion is missing or unsupported")
    if dict(result.get("executionIdentity") or {}) != dict(context.execution_identity):
        issues.append("result.executionIdentity does not match the immutable job execution identity")
    if "media" in result:
        validate_owner(result.get("media"), "result.media")
    if "sourceMetadata" in result:
        validate_owner(result.get("sourceMetadata"), "result.sourceMetadata")
    if "summary" in result:
        validate_owner(result.get("summary"), "result.summary")

    for index, violation in enumerate(result.get("violations") or []):
        validate_owner(violation, f"result.violations[{index}]")
        for affected_index, affected in enumerate((violation or {}).get("affectedFrames") or []):
            validate_owner(affected, f"result.violations[{index}].affectedFrames[{affected_index}]")
            try:
                validate_evidence_ownership(affected, context.job_id)
            except ResultOwnershipError as exc:
                issues.extend(f"result.violations[{index}].affectedFrames[{affected_index}]: {item}" for item in exc.issues)

    validated_frames: set[int] = set()
    for collection_name in ("frames", "evidence", "diagnosticFrames"):
        for index, frame in enumerate(result.get(collection_name) or []):
            location = f"result.{collection_name}[{index}]"
            if id(frame) in validated_frames:
                continue
            validated_frames.add(id(frame))
            validate_owner(frame, location)
            if isinstance(frame, Mapping) and "sourceMetadata" in frame:
                validate_owner(frame.get("sourceMetadata"), f"{location}.sourceMetadata")
            try:
                validate_evidence_ownership(frame, context.job_id)
            except ResultOwnershipError as exc:
                issues.extend(f"{location}: {item}" for item in exc.issues)
            for violation_index, violation in enumerate((frame or {}).get("violations") or []):
                validate_owner(violation, f"{location}.violations[{violation_index}]")

    def validate_event(event: Any, location: str) -> None:
        validate_owner(event, location)
        if not isinstance(event, Mapping):
            return
        for support_index, support in enumerate(event.get("supportingFrames") or []):
            validate_owner(support, f"{location}.supportingFrames[{support_index}]")
            try:
                validate_evidence_ownership(support, context.job_id)
            except ResultOwnershipError as exc:
                issues.extend(f"{location}.supportingFrames[{support_index}]: {item}" for item in exc.issues)

    for index, event in enumerate(result.get("events") or []):
        validate_event(event, f"result.events[{index}]")
    summary = result.get("summary") or {}
    for index, event in enumerate(summary.get("keyEvents") or []):
        validate_event(event, f"result.summary.keyEvents[{index}]")

    technical = result.get("technicalDetails") or {}
    ownership = technical.get("ownership") if isinstance(technical, Mapping) else None
    validate_owner(ownership, "result.technicalDetails.ownership")
    if isinstance(ownership, Mapping):
        if dict(ownership.get("executionIdentity") or {}) != dict(context.execution_identity):
            issues.append("result.technicalDetails.ownership.executionIdentity does not match the job")
        if ownership.get("resultSchemaVersion") != RESULT_SCHEMA_VERSION:
            issues.append("result.technicalDetails.ownership.resultSchemaVersion is unsupported")
    if isinstance(technical, Mapping) and "sourceMetadata" in technical:
        validate_owner(technical.get("sourceMetadata"), "result.technicalDetails.sourceMetadata")

    def validate_nested(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            nested_job_id = value.get("jobId")
            if nested_job_id not in (None, "", context.job_id):
                issues.append(f"{location}.jobId={nested_job_id!r} is foreign")
            image_url = value.get("imageUrl")
            match = _MEDIA_JOB_RE.search(str(image_url or ""))
            if match and match.group(1) != context.job_id:
                issues.append(f"{location}.imageUrl routes through foreign job {match.group(1)!r}")
            evidence_id = str(value.get("evidenceId") or "")
            if evidence_id and not evidence_id.startswith(f"{context.job_id}:"):
                issues.append(f"{location}.evidenceId={evidence_id!r} is foreign")
            artifact_id = str(value.get("mediaArtifactId") or "")
            if artifact_id and not artifact_id.startswith(f"{context.job_id}:"):
                issues.append(f"{location}.mediaArtifactId={artifact_id!r} is foreign")
            for key, item in value.items():
                validate_nested(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                validate_nested(item, f"{location}[{index}]")

    validate_nested(result, "result")
    if issues:
        raise ResultOwnershipError(issues)
