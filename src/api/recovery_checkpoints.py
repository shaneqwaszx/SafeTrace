"""Checksummed safe-boundary recovery checkpoints for durable analysis jobs."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEMA_VERSION = 1
STAGES = (
    "upload_complete",
    "media_probe_complete",
    "frame_sampling_complete",
    "ranking_complete",
    "detector_work_complete",
    "mobile_sam_subset_complete",
    "vlm_subset_complete",
    "aggregation_complete",
    "evidence_report_complete",
    "completed",
)
_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock(root: Path) -> threading.RLock:
    resolved = root.resolve()
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(resolved, threading.RLock())


def _digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def stage_from_state(current_step: str, status: str, progress: float) -> str:
    text = (current_step or "").lower()
    if status == "completed":
        return "completed"
    if "report" in text or "normaliz" in text or progress >= 0.85:
        return "aggregation_complete"
    if "pipeline completed" in text or progress >= 0.8:
        return "vlm_subset_complete"
    if "analysis" in text or "detector" in text or progress >= 0.35:
        return "media_probe_complete"
    if "prepar" in text or progress >= 0.15:
        return "upload_complete"
    return "upload_complete"


def write_checkpoint(
    root: Path,
    *,
    job_id: str,
    stage: str,
    identity: dict[str, Any],
    progress: float,
    status: str,
    current_step: str,
    completed_frame_index: int | None = None,
    completed_timestamp: float | None = None,
    completed_candidate_ids: list[str] | None = None,
    completed_evidence_ids: list[str] | None = None,
    partial_output_reference: str | None = None,
    partial_output_checksum_sha256: str | None = None,
) -> Path:
    if stage not in STAGES:
        raise ValueError(f"Unsupported recovery checkpoint stage: {stage}")
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": CHECKPOINT_SCHEMA_VERSION,
        "jobId": job_id,
        "stage": stage,
        "stageIndex": STAGES.index(stage),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "identityChecksum": _digest(identity),
        "progress": progress,
        "status": status,
        "currentStep": current_step,
        "lastCompletedFrameIndex": completed_frame_index,
        "lastCompletedTimestamp": completed_timestamp,
        "completedCandidateIds": sorted(set(completed_candidate_ids or [])),
        "completedEvidenceIds": sorted(set(completed_evidence_ids or [])),
        "partialOutputReference": partial_output_reference,
        "partialOutputChecksumSha256": partial_output_checksum_sha256,
    }
    envelope = {"payload": payload, "checksumSha256": _digest(payload)}
    with _lock(checkpoint_root):
        sequence = len(list(checkpoint_root.glob("checkpoint_*.json"))) + 1
        destination = checkpoint_root / f"checkpoint_{sequence:06d}_{stage}.json"
        temporary = checkpoint_root / f"{destination.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(envelope, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def _load(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = dict(envelope["payload"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, "corrupt_or_incomplete_checkpoint"
    if payload.get("schemaVersion") != CHECKPOINT_SCHEMA_VERSION:
        return None, "checkpoint_schema_mismatch"
    if envelope.get("checksumSha256") != _digest(payload):
        return None, "checkpoint_checksum_mismatch"
    if payload.get("identityChecksum") != _digest(dict(payload.get("identity") or {})):
        return None, "identity_checksum_mismatch"
    return payload, None


def latest_valid_checkpoint(
    root: Path,
    current_identity: dict[str, Any],
    *,
    expected_job_id: str | None = None,
) -> dict[str, Any]:
    checkpoint_root = root / "checkpoints"
    invalid: list[dict[str, str]] = []
    current_checksum = _digest(current_identity)
    for path in sorted(checkpoint_root.glob("checkpoint_*.json"), reverse=True):
        payload, issue = _load(path)
        if payload is None:
            invalid.append({"path": str(path), "reason": issue or "invalid"})
            continue
        if expected_job_id is not None and str(payload.get("jobId") or "") != expected_job_id:
            invalid.append({"path": str(path), "reason": "checkpoint_job_owner_mismatch"})
            continue
        exact = payload.get("identityChecksum") == current_checksum
        stage = str(payload.get("stage") or "upload_complete")
        reusable = exact or stage in {"upload_complete", "media_probe_complete"}
        if not reusable:
            invalid.append({"path": str(path), "reason": "configuration_identity_changed"})
            continue
        return {
            "checkpoint": payload,
            "checkpointPath": str(path),
            "exactIdentityMatch": exact,
            "invalidationReason": None if exact else "configuration_identity_changed_downstream_invalidated",
            "invalidNewerCheckpoints": invalid,
        }
    return {
        "checkpoint": None,
        "checkpointPath": None,
        "exactIdentityMatch": False,
        "invalidationReason": "no_valid_compatible_checkpoint",
        "invalidNewerCheckpoints": invalid,
    }


def latest_reusable_partial(
    root: Path,
    current_identity: dict[str, Any],
    *,
    expected_job_id: str | None = None,
) -> dict[str, Any] | None:
    """Return the newest exact-identity, checksummed JSON pipeline artifact."""
    current_checksum = _digest(current_identity)
    checkpoint_root = root / "checkpoints"
    resolved_root = root.resolve()
    for path in sorted(checkpoint_root.glob("checkpoint_*.json"), reverse=True):
        payload, issue = _load(path)
        if (
            payload is None
            or issue
            or payload.get("identityChecksum") != current_checksum
            or (expected_job_id is not None and str(payload.get("jobId") or "") != expected_job_id)
        ):
            continue
        reference = str(payload.get("partialOutputReference") or "")
        expected = str(payload.get("partialOutputChecksumSha256") or "")
        if not reference or not expected:
            continue
        artifact = (root / reference).resolve()
        if artifact == resolved_root or resolved_root not in artifact.parents or not artifact.is_file():
            continue
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if digest != expected:
            continue
        try:
            result = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(result, list) and _partial_paths_belong_to_job(result, root):
            return {"checkpoint": payload, "checkpointPath": str(path), "artifactPath": str(artifact), "rawResult": result}
    return None


def _partial_paths_belong_to_job(result: list[Any], root: Path) -> bool:
    resolved_root = root.resolve()
    for frame in result:
        if not isinstance(frame, dict):
            return False
        for key in ("frame_path", "annotated_path"):
            raw_path = frame.get(key)
            if not raw_path:
                continue
            path = Path(str(raw_path)).resolve()
            if path != resolved_root and resolved_root not in path.parents:
                return False
    return True
