"""Content-addressed completed-result cache with job-owned restored artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from src import __version__
from src.config import SETTINGS

from .jobs import JobRecord, result_ownership_context
from .result_ownership import (
    ResultOwnershipContext,
    ResultOwnershipError,
    stamp_result_ownership,
    validate_result_ownership,
)

_LOCK = threading.RLock()


@lru_cache(maxsize=16)
def file_sha256(path_value: str, size: int, modified_ns: int) -> str | None:
    path = Path(path_value)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": path.name, "exists": False, "sha256": None}
    stat = path.stat()
    return {
        "path": path.name,
        "exists": True,
        "sha256": file_sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns),
    }


def cache_identity(record: JobRecord) -> dict[str, Any]:
    settings = record.settings
    detector = Path(SETTINGS.yolo_checkpoint)
    if not detector.is_file():
        detector = Path(SETTINGS.yolo_fallback_checkpoint)
    return {
        "schemaVersion": 1,
        "sourceChecksum": record.source_metadata.get("checksumSha256"),
        "query": " ".join(record.query.lower().split()),
        "profile": settings.use_case_profile,
        "reviewMode": settings.review_mode,
        "sampling": {"fps": settings.fps, "topK": settings.top_k, "maxFrames": SETTINGS.max_frames},
        "detector": checkpoint_identity(detector),
        "mobileSam": {
            "enabled": SETTINGS.mobile_sam_enabled,
            "checkpoint": checkpoint_identity(Path(SETTINGS.mobile_sam_checkpoint)),
            "frameLimit": SETTINGS.mobile_sam_frame_limit,
        },
        "vlm": {
            "enabled": bool(settings.enable_vlm and settings.vlm_enabled),
            "profile": settings.vlm_profile,
            "frameLimit": SETTINGS.vlm_max_frames,
        },
        "rules": {
            "confidence": SETTINGS.yolo_conf_threshold,
            "iou": SETTINGS.yolo_iou_threshold,
        },
        "appVersion": __version__,
    }


def cache_key(record: JobRecord) -> str:
    payload = json.dumps(cache_identity(record), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_root_for(record: JobRecord) -> Path:
    return record.job_dir.parent.parent / "result_cache"


def _entry(record: JobRecord) -> Path:
    return cache_root_for(record) / cache_key(record)


def restore_cached_result(record: JobRecord) -> tuple[dict[str, Any], list[tuple[str, Path]]] | None:
    entry = _entry(record)
    metadata_path = entry / "metadata.json"
    result_path = entry / "result.json"
    with _LOCK:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if metadata.get("complete") is not True or metadata.get("cacheKey") != cache_key(record):
            return None
        created = datetime.fromisoformat(str(metadata["createdAt"]).replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
        if age_days > float(SETTINGS.cache_retention_days):
            return None
        old_job_id = str(metadata.get("originJobId") or metadata.get("sourceJobId") or "")
        if not old_job_id or result.get("resultSchemaVersion") is None:
            return None
        origin_context = ResultOwnershipContext(
            job_id=old_job_id,
            batch_id=str(result.get("batchId") or "") or None,
            source_checksum=str(result.get("sourceChecksum") or "") or None,
            original_filename=str(result.get("originalFilename") or ""),
            source_relative_path=str(result.get("sourceRelativePath") or ""),
            source_group_path=str(result.get("sourceGroupPath") or ""),
            execution_identity=dict(result.get("executionIdentity") or metadata.get("identity") or {}),
        )
        try:
            validate_result_ownership(result, origin_context)
            restored = stamp_result_ownership(
                result,
                result_ownership_context(record),
                previous_job_id=old_job_id,
            )
        except ResultOwnershipError:
            return None
        restored.setdefault("technicalDetails", {})["resultCache"] = {
            "hit": True,
            "cacheKey": metadata["cacheKey"],
            "reason": "Exact source, profile, configuration, checkpoint, and app-version match.",
            "sourceJobId": old_job_id,
        }
        media: list[tuple[str, Path]] = []
        media_root = entry / "media"
        if media_root.is_dir():
            record.output_dir.mkdir(parents=True, exist_ok=True)
            for source in media_root.iterdir():
                if not source.is_file():
                    continue
                destination = record.output_dir / source.name
                if not destination.exists():
                    try:
                        os.link(source, destination)
                    except OSError:
                        shutil.copy2(source, destination)
                media.append((source.name, destination))
        metadata["lastAccessAt"] = datetime.now(timezone.utc).isoformat()
        references = set(str(item) for item in metadata.get("jobReferences") or [])
        references.add(record.job_id)
        metadata["jobReferences"] = sorted(references)
        metadata["referenceCount"] = len(references)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return restored, media


def publish_completed_result(record: JobRecord) -> None:
    if record.status != "completed" or record.result is None:
        return
    entry = _entry(record)
    validate_result_ownership(record.result, result_ownership_context(record))
    temporary = entry.with_name(f"{entry.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    metadata = {
        "cacheEntryId": entry.name,
        "cacheKey": cache_key(record),
        "identity": cache_identity(record),
        "executionIdentity": dict(record.metrics.get("executionIdentity") or {}),
        "complete": True,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "lastAccessAt": datetime.now(timezone.utc).isoformat(),
        "sourceJobId": record.job_id,
        "originJobId": record.job_id,
        "artifactOwnershipMode": "immutable_cache_copy_materialized_per_job",
        "jobReferences": [record.job_id],
        "referenceCount": 1,
    }
    with _LOCK:
        if entry.is_dir():
            return
        shutil.rmtree(temporary, ignore_errors=True)
        (temporary / "media").mkdir(parents=True, exist_ok=True)
        (temporary / "result.json").write_text(json.dumps(record.result, indent=2), encoding="utf-8")
        for filename, source in record.media_files.items():
            if source.is_file():
                shutil.copy2(source, temporary / "media" / filename)
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        entry.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(temporary, entry)
        except OSError:
            shutil.rmtree(temporary, ignore_errors=True)


def release_job_reference(record: JobRecord) -> None:
    entry = _entry(record)
    metadata_path = entry / "metadata.json"
    with _LOCK:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        references = {str(item) for item in metadata.get("jobReferences") or []}
        references.discard(record.job_id)
        metadata["jobReferences"] = sorted(references)
        metadata["referenceCount"] = len(references)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def purge_compatible_cache_entries(
    records: list[JobRecord],
    *,
    selected_job_ids: set[str],
) -> dict[str, Any]:
    """Purge selected-source cache entries only when no outside job references them."""
    purged: list[str] = []
    retained: list[dict[str, Any]] = []
    source_checksums = {
        str(record.source_metadata.get("checksumSha256"))
        for record in records
        if record.source_metadata.get("checksumSha256")
    }
    roots = {cache_root_for(record) for record in records}
    with _LOCK:
        for cache_root in roots:
            root = cache_root.resolve()
            for entry in cache_root.iterdir() if cache_root.is_dir() else []:
                target = entry.resolve()
                if target == root or root not in target.parents or not entry.is_dir():
                    continue
                try:
                    metadata = json.loads((entry / "metadata.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    retained.append({"cacheEntryId": entry.name, "reason": "metadata_unreadable"})
                    continue
                identity = dict(metadata.get("identity") or {})
                if str(identity.get("sourceChecksum") or "") not in source_checksums:
                    continue
                outside_references = {
                    str(value) for value in metadata.get("jobReferences") or []
                } - selected_job_ids
                if outside_references:
                    retained.append({
                        "cacheEntryId": entry.name,
                        "reason": "shared_with_unselected_jobs",
                        "jobReferences": sorted(outside_references),
                    })
                    continue
                shutil.rmtree(entry)
                purged.append(entry.name)
    return {"requested": True, "purged": purged, "retained": retained}
