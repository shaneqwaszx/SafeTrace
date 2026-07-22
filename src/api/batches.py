"""Batch upload manifests and ZIP safety validation for the local API."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Literal, Optional

from src.config import SETTINGS

from .jobs import (
    AnalysisSettings,
    JobRecord,
    JobStore,
    MEDIA_EXTENSIONS,
    TERMINAL_STATES,
    build_analysis_setup_payload,
    max_upload_bytes,
    owned_result_snapshot,
    safe_filename,
)

BatchState = Literal[
    "importing", "queued", "waiting_for_capacity", "waiting_for_job_slot", "waiting_for_gpu", "waiting_for_mobilesam", "waiting_for_vlm", "running_preprocess", "running_detector", "running_refinement", "running_report", "running", "retry_wait", "paused", "recovering",
    "completed", "completed_with_failures", "failed", "partial", "cancelled",
]
BATCH_MANIFEST_FILENAME = "manifest.json"
VIDEO_EXTENSIONS = {extension for extension, media_type in MEDIA_EXTENSIONS.items() if media_type == "video"}
logger = logging.getLogger("safetrace.api.batches")
_MANIFEST_LOCKS: dict[Path, threading.RLock] = {}
_MANIFEST_LOCKS_GUARD = threading.Lock()


class BatchValidationError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class BatchManifestPersistenceError(OSError):
    """Raised when a batch manifest cannot be persisted after retries."""


@dataclass
class BatchFile:
    original_filename: str
    filename: str
    size_bytes: int
    media_type: str
    job_id: str
    source_relative_path: str = ""
    source_directory: str = ""
    source_group_path: str = ""
    source_group_label: str = "Ungrouped"
    checksum_sha256: Optional[str] = None
    imported_at: Optional[str] = None
    file_identity: Optional[str] = None
    status: str = "queued"
    error: Optional[str] = None
    violation_count: int = 0
    enqueue_sequence: Optional[int] = None
    batch_enqueue_sequence: Optional[int] = None
    child_sequence: Optional[int] = None
    queue_position: Optional[int] = None
    worker_slot: Optional[int] = None
    capacity_reason: Optional[str] = None
    requested_mode_label: Optional[str] = None
    actual_review: Optional[str] = None
    device_label: Optional[str] = None
    mobile_sam_status: Optional[str] = None

    def payload(self) -> Dict[str, Any]:
        return {
            "originalFilename": self.original_filename,
            "filename": self.filename,
            "sizeBytes": self.size_bytes,
            "mediaType": self.media_type,
            "jobId": self.job_id,
            "sourceRelativePath": self.source_relative_path or self.original_filename,
            "sourceDirectory": self.source_directory,
            "sourceGroupPath": self.source_group_path,
            "sourceGroupLabel": self.source_group_label,
            "checksumSha256": self.checksum_sha256,
            "importedAt": self.imported_at,
            "fileIdentity": self.file_identity,
            "status": self.status,
            "error": self.error,
            "violationCount": self.violation_count,
            "enqueueSequence": self.enqueue_sequence,
            "batchEnqueueSequence": self.batch_enqueue_sequence,
            "childSequence": self.child_sequence,
            "queuePosition": self.queue_position,
            "workerSlot": self.worker_slot,
            "capacityReason": self.capacity_reason,
            "requestedModeLabel": self.requested_mode_label,
            "actualReview": self.actual_review,
            "deviceLabel": self.device_label,
            "mobileSamStatus": self.mobile_sam_status,
        }


@dataclass
class RejectedBatchFile:
    filename: str
    reason: str
    source_relative_path: Optional[str] = None
    category: str = "rejected"

    def payload(self) -> Dict[str, str]:
        return {
            "filename": self.filename,
            "sourceRelativePath": self.source_relative_path or self.filename,
            "reason": self.reason,
            "category": self.category,
        }


@dataclass
class BatchRecord:
    batch_id: str
    status: BatchState
    source_filename: str
    batch_dir: Path
    accepted_files: list[BatchFile] = field(default_factory=list)
    rejected_files: list[RejectedBatchFile] = field(default_factory=list)
    status_counts: Dict[str, int] = field(default_factory=dict)
    persistence_warning: Optional[str] = None
    import_key: Optional[str] = None
    display_label: Optional[str] = None
    paused: bool = False
    group_summaries: list[Dict[str, Any]] = field(default_factory=list)
    hierarchy: Dict[str, Any] = field(default_factory=dict)
    throughput: Dict[str, Any] = field(default_factory=dict)
    analysis_setup: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def manifest_path(self) -> Path:
        return self.batch_dir / BATCH_MANIFEST_FILENAME

    @property
    def job_ids(self) -> list[str]:
        return [item.job_id for item in self.accepted_files]

    def payload(self) -> Dict[str, Any]:
        return {
            "batchId": self.batch_id,
            "status": self.status,
            "sourceFilename": self.source_filename,
            "batchDisplayLabel": self.display_label or self.source_filename,
            "importKey": self.import_key,
            "acceptedFiles": [item.payload() for item in self.accepted_files],
            "rejectedFiles": [item.payload() for item in self.rejected_files],
            "jobIds": self.job_ids,
            "statusCounts": self.status_counts,
            "groupSummaries": self.group_summaries,
            "hierarchy": self.hierarchy,
            "throughput": self.throughput,
            "paused": self.paused,
            "createdAt": _to_iso(self.created_at),
            "updatedAt": _to_iso(self.updated_at),
            "persistenceWarning": self.persistence_warning,
            "analysisSetup": self.analysis_setup,
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: Any) -> datetime:
    if not value:
        return _utc_now()
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return _utc_now()
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _manifest_lock(path: Path) -> threading.RLock:
    key = path.resolve()
    with _MANIFEST_LOCKS_GUARD:
        lock = _MANIFEST_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _MANIFEST_LOCKS[key] = lock
        return lock


def _cleanup_temporary(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _atomic_write_json(path: Path, payload: Dict[str, Any], *, retries: int = 5, backoff_seconds: float = 0.05) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    lock = _manifest_lock(path)
    with lock:
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            last_error: PermissionError | None = None
            for attempt in range(max(1, retries)):
                try:
                    os.replace(temporary, path)
                    return
                except PermissionError as exc:
                    last_error = exc
                    if attempt >= retries - 1:
                        break
                    time.sleep(backoff_seconds * (attempt + 1))
            raise BatchManifestPersistenceError(f"Could not replace {path} after {retries} attempts.") from last_error
        finally:
            _cleanup_temporary(temporary)


def _new_batch_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"batch_{stamp}_{uuid.uuid4().hex[:8]}"


def _is_safe_batch_id(batch_id: str) -> bool:
    return bool(re.fullmatch(r"batch_[A-Za-z0-9_.-]+", batch_id or ""))


def _bulk_uncompressed_limit_bytes() -> int:
    return max(int(float(SETTINGS.bulk_max_uncompressed_mb) * 1024 * 1024), 1)


def _safe_zip_member_name(raw_name: str) -> Optional[str]:
    normalized = raw_name.replace("\\", "/").strip("/")
    if not normalized:
        return None
    if raw_name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", raw_name):
        return None
    parts = PurePosixPath(normalized).parts
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return "/".join(parts)


def _source_path_metadata(relative_path: str) -> Dict[str, str]:
    normalized = _safe_zip_member_name(relative_path)
    if normalized is None:
        raise BatchValidationError(f"Unsafe relative path: {relative_path}", status_code=400)
    parent = str(PurePosixPath(normalized).parent)
    source_directory = "" if parent == "." else parent
    group_path = source_directory
    group_label = PurePosixPath(group_path).name if group_path else "Ungrouped"
    return {
        "sourceRelativePath": normalized,
        "sourceDirectory": source_directory,
        "sourceGroupPath": group_path,
        "sourceGroupLabel": group_label,
    }


def _zip_entry_is_symlink(entry: zipfile.ZipInfo) -> bool:
    unix_mode = (entry.external_attr >> 16) & 0o170000
    return unix_mode == 0o120000


def _file_identity(batch_id: str, relative_path: str, size_bytes: int, checksum: str) -> str:
    import hashlib

    value = f"{batch_id}\n{relative_path}\n{size_bytes}\n{checksum}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _unique_filename(filename: str, used: set[str]) -> str:
    clean_name = safe_filename(filename)
    if clean_name not in used:
        used.add(clean_name)
        return clean_name

    stem = Path(clean_name).stem
    suffix = Path(clean_name).suffix
    index = 2
    while True:
        candidate = f"{stem}_{index}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1


def _is_video_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in VIDEO_EXTENSIONS


def _media_type_for_batch(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    return MEDIA_EXTENSIONS.get(extension, "unknown")


class BatchStore:
    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = Path(root_dir or SETTINGS.data_dir / "api_batches")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._batches: Dict[str, BatchRecord] = {}
        self._lock = threading.RLock()
        self.records(include_disk=True)

    def create_from_zip(
        self,
        *,
        filename: str,
        content: bytes,
        query: str,
        settings: AnalysisSettings,
        job_store: JobStore,
    ) -> BatchRecord:
        return self.create_from_zip_stream(
            filename=filename,
            stream=BytesIO(content),
            query=query,
            settings=settings,
            job_store=job_store,
        )

    def create_from_zip_stream(
        self,
        *,
        filename: str,
        stream,
        query: str,
        settings: AnalysisSettings,
        job_store: JobStore,
        import_key: Optional[str] = None,
    ) -> BatchRecord:
        existing = self._find_by_import_key(import_key, job_store)
        if existing is not None:
            return existing
        source_filename = safe_filename(filename or "upload.zip")
        batch = self._create_empty_batch(source_filename, import_key=import_key)
        batch.status = "importing"
        used_filenames: set[str] = set()
        seen_paths: set[str] = set()

        try:
            stream.seek(0)
            with zipfile.ZipFile(stream) as archive:
                entries = [entry for entry in archive.infolist() if not entry.is_dir()]
                self._validate_archive_shape(entries)

                for entry in entries:
                    safe_member = _safe_zip_member_name(entry.filename)
                    if safe_member is None:
                        raise BatchValidationError(
                            f"Archive contains an unsafe path: {entry.filename}",
                            status_code=400,
                        )
                    if _zip_entry_is_symlink(entry):
                        batch.rejected_files.append(
                            RejectedBatchFile(safe_member, "Symbolic-link ZIP entries are not accepted.", safe_member)
                        )
                        continue
                    if safe_member.casefold() in seen_paths:
                        batch.rejected_files.append(
                            RejectedBatchFile(safe_member, "Duplicate relative path in archive.", safe_member, "duplicate")
                        )
                        continue
                    seen_paths.add(safe_member.casefold())

                    if not _is_video_filename(safe_member):
                        batch.rejected_files.append(
                            RejectedBatchFile(
                                filename=safe_member,
                                reason="Unsupported file type for bulk video analysis.",
                            )
                        )
                        continue

                    if entry.file_size <= 0:
                        batch.rejected_files.append(
                            RejectedBatchFile(
                                filename=safe_member,
                                reason="File is empty.",
                            )
                        )
                        continue

                    if entry.file_size > max_upload_bytes():
                        batch.rejected_files.append(
                            RejectedBatchFile(
                                filename=safe_member,
                                reason="File exceeds the per-video upload limit.",
                            )
                        )
                        continue

                    clean_name = _unique_filename(Path(safe_member).name, used_filenames)
                    try:
                        with archive.open(entry) as member_stream:
                            self._create_job_for_stream(
                                batch=batch,
                                job_store=job_store,
                                original_filename=safe_member,
                                filename=clean_name,
                                stream=member_stream,
                                expected_size=int(entry.file_size),
                                query=query,
                                settings=settings,
                            )
                    except ValueError as exc:
                        batch.rejected_files.append(
                            RejectedBatchFile(safe_member, str(exc), safe_member)
                        )
        except zipfile.BadZipFile as exc:
            self._delete_record_dir(batch)
            self._batches.pop(batch.batch_id, None)
            raise BatchValidationError("Uploaded archive is not a readable ZIP file.", status_code=400) from exc
        except BatchValidationError:
            self._delete_record_dir(batch)
            self._batches.pop(batch.batch_id, None)
            raise

        self._finalize_created_batch(batch, job_store)
        return batch

    def create_from_files(
        self,
        *,
        files: Iterable[tuple[str, bytes]],
        source_filename: str,
        query: str,
        settings: AnalysisSettings,
        job_store: JobStore,
        import_key: Optional[str] = None,
    ) -> BatchRecord:
        return self.create_from_streams(
            files=((name, BytesIO(content), len(content)) for name, content in files),
            source_filename=source_filename,
            query=query,
            settings=settings,
            job_store=job_store,
            import_key=import_key,
        )

    def create_from_streams(
        self,
        *,
        files: Iterable[tuple[str, Any, Optional[int]]],
        source_filename: str,
        query: str,
        settings: AnalysisSettings,
        job_store: JobStore,
        import_key: Optional[str] = None,
    ) -> BatchRecord:
        existing = self._find_by_import_key(import_key, job_store)
        if existing is not None:
            return existing
        batch = self._create_empty_batch(safe_filename(source_filename or "bulk-upload"), import_key=import_key)
        batch.status = "importing"
        used_filenames: set[str] = set()
        materialized = list(files)
        if len(materialized) > int(SETTINGS.bulk_max_files):
            self._delete_record_dir(batch)
            self._batches.pop(batch.batch_id, None)
            raise BatchValidationError(
                f"Bulk upload contains too many files. Maximum is {SETTINGS.bulk_max_files}.",
                status_code=413,
            )

        seen_paths: set[str] = set()
        for original_name, stream, expected_size in materialized:
            safe_relative = _safe_zip_member_name(original_name)
            if safe_relative is None:
                batch.rejected_files.append(
                    RejectedBatchFile(str(original_name), "Unsafe client-supplied relative path.")
                )
                continue
            if safe_relative.casefold() in seen_paths:
                batch.rejected_files.append(
                    RejectedBatchFile(safe_relative, "Duplicate relative path in upload.", safe_relative, "duplicate")
                )
                continue
            seen_paths.add(safe_relative.casefold())
            if not _is_video_filename(safe_relative):
                batch.rejected_files.append(
                    RejectedBatchFile(
                        filename=safe_relative,
                        reason="Unsupported file type for bulk video analysis.",
                        source_relative_path=safe_relative,
                    )
                )
                continue
            if expected_size == 0:
                batch.rejected_files.append(
                    RejectedBatchFile(
                        filename=safe_relative,
                        reason="File is empty.",
                        source_relative_path=safe_relative,
                    )
                )
                continue
            if expected_size is not None and expected_size > max_upload_bytes():
                batch.rejected_files.append(
                    RejectedBatchFile(
                        filename=safe_relative,
                        reason="File exceeds the per-video upload limit.",
                        source_relative_path=safe_relative,
                    )
                )
                continue
            clean_name = _unique_filename(Path(safe_relative).name, used_filenames)
            try:
                self._create_job_for_stream(
                    batch=batch,
                    job_store=job_store,
                    original_filename=safe_relative,
                    filename=clean_name,
                    stream=stream,
                    expected_size=expected_size,
                    query=query,
                    settings=settings,
                )
            except (BatchValidationError, ValueError) as exc:
                batch.rejected_files.append(
                    RejectedBatchFile(safe_relative, str(exc), safe_relative)
                )

        self._finalize_created_batch(batch, job_store)
        return batch

    def get(self, batch_id: str, job_store: JobStore | None = None) -> Optional[BatchRecord]:
        with self._lock:
            record = self._batches.get(batch_id)
            if record is None:
                record = self.load_batch(batch_id)
                if record is not None:
                    self._batches[batch_id] = record
        if record is not None and job_store is not None:
            self.refresh(record, job_store)
        return record

    def require(self, batch_id: str, job_store: JobStore | None = None) -> BatchRecord:
        record = self.get(batch_id, job_store)
        if record is None:
            raise KeyError(batch_id)
        return record

    def records(self, *, include_disk: bool = False, job_store: JobStore | None = None) -> list[BatchRecord]:
        with self._lock:
            if include_disk:
                for batch_dir in self.root_dir.iterdir():
                    manifest = batch_dir / BATCH_MANIFEST_FILENAME
                    if not batch_dir.is_dir() or not manifest.is_file():
                        continue
                    record = self._load_manifest(manifest)
                    if record is not None:
                        self._batches[record.batch_id] = record
            records = list(self._batches.values())
        if job_store is not None:
            for record in records:
                self.refresh(record, job_store)
        return records

    def refresh(self, record: BatchRecord, job_store: JobStore) -> BatchRecord:
        changed = False
        counts: Dict[str, int] = {}
        with self._lock:
            for item in record.accepted_files:
                job = job_store.get(item.job_id)
                if job is None:
                    next_status = "failed"
                    next_error = "Job manifest is missing."
                else:
                    next_status = job.status
                    next_error = job.error
                if item.status != next_status or item.error != next_error:
                    changed = True
                    item.status = next_status
                    item.error = next_error
                if job is not None:
                    diagnostics = dict(job.metrics.get("componentDiagnostics") or {})
                    next_scheduler = (
                        diagnostics.get("enqueueSequence"),
                        diagnostics.get("batchEnqueueSequence"),
                        diagnostics.get("childSequence"),
                        diagnostics.get("queuePosition"),
                        diagnostics.get("workerSlot"),
                        diagnostics.get("capacityReason") or diagnostics.get("queueWaitReason"),
                    )
                    current_scheduler = (
                        item.enqueue_sequence,
                        item.batch_enqueue_sequence,
                        item.child_sequence,
                        item.queue_position,
                        item.worker_slot,
                        item.capacity_reason,
                    )
                    if current_scheduler != next_scheduler:
                        changed = True
                        (
                            item.enqueue_sequence,
                            item.batch_enqueue_sequence,
                            item.child_sequence,
                            item.queue_position,
                            item.worker_slot,
                            item.capacity_reason,
                        ) = next_scheduler
                    runtime = job.status_payload().get("engineRuntimeSummary") or {}
                    next_runtime = (
                        job.status_payload().get("requestedModeLabel"),
                        runtime.get("actualReview"),
                        runtime.get("device"),
                        runtime.get("mobileSam"),
                    )
                    current_runtime = (
                        item.requested_mode_label,
                        item.actual_review,
                        item.device_label,
                        item.mobile_sam_status,
                    )
                    if current_runtime != next_runtime:
                        changed = True
                        (
                            item.requested_mode_label,
                            item.actual_review,
                            item.device_label,
                            item.mobile_sam_status,
                        ) = next_runtime
                counts[item.status] = counts.get(item.status, 0) + 1

            next_status = self._derive_status(record)
            next_groups = self._group_summaries(record, job_store)
            next_hierarchy = self._hierarchy(record)
            next_analysis_setup = self._analysis_setup(record, job_store)
            manifest_missing = not record.manifest_path.is_file()
            if record.status_counts != counts:
                changed = True
                record.status_counts = counts
            if record.status != next_status:
                changed = True
                record.status = next_status
            if record.group_summaries != next_groups:
                changed = True
                record.group_summaries = next_groups
            if record.hierarchy != next_hierarchy:
                changed = True
                record.hierarchy = next_hierarchy
            if record.analysis_setup != next_analysis_setup:
                changed = True
                record.analysis_setup = next_analysis_setup

            # Runtime counters are computed for every response but do not by
            # themselves force a manifest write during status polling.
            record.throughput = self._throughput(record, job_store)
            if changed or manifest_missing:
                record.updated_at = _utc_now()
                self.persist_batch(record)
        return record

    def _analysis_setup(self, record: BatchRecord, job_store: JobStore) -> Dict[str, Any]:
        setups: list[Dict[str, Any]] = []
        for item in record.accepted_files:
            job = job_store.get(item.job_id)
            if job is not None:
                setups.append(build_analysis_setup_payload(job))
        if not setups:
            return dict(record.analysis_setup or {})
        shared = dict(setups[0])
        identity_keys = ("requestedCoverage", "profile", "query")
        shared["childSettingsDiffer"] = any(
            any(setup.get(key) != shared.get(key) for key in identity_keys)
            for setup in setups[1:]
        )
        sampling = dict(shared.get("frameSampling") or {})
        sampled_values = [
            (setup.get("frameSampling") or {}).get("sampledFrameCount")
            for setup in setups
        ]
        source_values = [
            (setup.get("frameSampling") or {}).get("sourceVideoFrameCount")
            for setup in setups
        ]
        sampling["aggregateSampledFrameCount"] = (
            sum(int(value) for value in sampled_values) if sampled_values and all(value is not None for value in sampled_values) else None
        )
        sampling["aggregateSourceVideoFrameCount"] = (
            sum(int(value) for value in source_values) if source_values and all(value is not None for value in source_values) else None
        )
        shared["frameSampling"] = sampling
        shared["batchChildCount"] = len(setups)
        shared["readOnly"] = bool(record.accepted_files) and all(
            item.status in TERMINAL_STATES for item in record.accepted_files
        )
        return shared

    def _hierarchy(self, record: BatchRecord) -> Dict[str, Any]:
        root: Dict[str, Any] = {
            "name": record.display_label or record.source_filename,
            "path": "",
            "type": "batch",
            "children": [],
            "files": [],
        }
        nodes: Dict[str, Dict[str, Any]] = {"": root}
        for item in sorted(record.accepted_files, key=lambda value: value.source_relative_path.casefold()):
            parent = root
            cumulative: list[str] = []
            for part in PurePosixPath(item.source_directory).parts if item.source_directory else ():
                cumulative.append(part)
                path = "/".join(cumulative)
                node = nodes.get(path)
                if node is None:
                    node = {"name": part, "path": path, "type": "group", "children": [], "files": []}
                    parent["children"].append(node)
                    nodes[path] = node
                parent = node
            parent["files"].append(item.payload())
        return root

    def _group_summaries(self, record: BatchRecord, job_store: JobStore) -> list[Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
        severity_rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        for item in record.accepted_files:
            path = item.source_group_path
            summary = groups.setdefault(
                path,
                {
                    "sourceGroupPath": path,
                    "sourceGroupLabel": item.source_group_label,
                    "totalVideos": 0,
                    "statusCounts": {},
                    "videosWithViolations": 0,
                    "violationCountsByType": {},
                    "highestSeverity": None,
                    "earliestTimestampSeconds": None,
                    "latestTimestampSeconds": None,
                    "totalRuntimeSeconds": 0.0,
                },
            )
            summary["totalVideos"] += 1
            counts = summary["statusCounts"]
            counts[item.status] = counts.get(item.status, 0) + 1
            job = job_store.get(item.job_id)
            if job is None:
                continue
            timing = job.timing_payload()
            summary["totalRuntimeSeconds"] += float(timing.get("analysisRuntimeSeconds") or 0.0)
            result = owned_result_snapshot(job) if job.result is not None else {}
            violations = list(result.get("violations") or [])
            item.violation_count = len(violations)
            if violations:
                summary["videosWithViolations"] += 1
            for violation in violations:
                name = str(violation.get("id") or violation.get("name") or "unknown")
                by_type = summary["violationCountsByType"]
                by_type[name] = by_type.get(name, 0) + 1
                severity = str(violation.get("severity") or "").lower()
                current = str(summary.get("highestSeverity") or "").lower()
                if severity_rank.get(severity, 0) > severity_rank.get(current, 0):
                    summary["highestSeverity"] = severity
            for frame in result.get("frames") or []:
                second = frame.get("timestampSeconds")
                if second is None:
                    continue
                second = float(second)
                earliest = summary["earliestTimestampSeconds"]
                latest = summary["latestTimestampSeconds"]
                summary["earliestTimestampSeconds"] = second if earliest is None else min(earliest, second)
                summary["latestTimestampSeconds"] = second if latest is None else max(latest, second)
        for value in groups.values():
            value["totalRuntimeSeconds"] = round(float(value["totalRuntimeSeconds"]), 3)
        return [groups[key] for key in sorted(groups, key=str.casefold)]

    def _throughput(self, record: BatchRecord, job_store: JobStore) -> Dict[str, Any]:
        jobs = [job_store.get(item.job_id) for item in record.accepted_files]
        jobs = [job for job in jobs if job is not None]
        batch_terminal = record.status in {"completed", "completed_with_failures", "failed", "partial", "cancelled"}
        finished_times = [job.finished_at for job in jobs if job.finished_at is not None]
        batch_end = max(finished_times) if batch_terminal and finished_times else _utc_now()
        elapsed = max((batch_end - record.created_at).total_seconds(), 0.001)
        completed = 0
        retries = 0
        peak = 0
        processing_runtimes: list[float] = []
        cache_materialization_runtimes: list[float] = []
        queue_waits: list[float] = []
        total_elapsed: list[float] = []
        cache_hits = 0
        cache_misses = 0
        cache_state_recorded = 0
        for job in jobs:
            completed += int(job.status == "completed")
            retries += int(job.retry_count)
            peak = max(peak, int(job.metrics.get("analysisConcurrency") or 0))
            timing = job.timing_payload()
            if job.status != "completed":
                continue
            runtime = timing.get("analysisRuntimeSeconds")
            queue_wait = timing.get("queueWaitSeconds")
            child_elapsed = timing.get("elapsedSeconds")
            cache_hit = job.metrics.get("resultCacheHit")
            if cache_hit is not None:
                cache_state_recorded += 1
                cache_hits += int(bool(cache_hit))
                cache_misses += int(not bool(cache_hit))
            if runtime is not None:
                target = cache_materialization_runtimes if cache_hit is True else processing_runtimes
                target.append(float(runtime))
            if queue_wait is not None:
                queue_waits.append(float(queue_wait))
            if child_elapsed is not None:
                total_elapsed.append(float(child_elapsed))
        mean_runtime = (
            sum(processing_runtimes) / len(processing_runtimes) if processing_runtimes else None
        )
        remaining = max(0, len(record.accepted_files) - completed)
        eta_confident = bool(len(processing_runtimes) >= 2 and remaining > 0)
        return {
            "batchRuntimeSeconds": round(elapsed, 3),
            "completedVideosPerHour": round((completed * 3600.0) / elapsed, 3),
            "retryCount": retries,
            "configuredPeakConcurrency": max(peak, int(getattr(SETTINGS, "analysis_concurrency", 1))),
            "activeCount": sum(record.status_counts.get(key, 0) for key in ("running", "running_preprocess", "running_detector", "running_refinement", "running_report", "recovering")),
            "queuedCount": sum(
                record.status_counts.get(key, 0)
                for key in ("queued", "waiting_for_capacity", "waiting_for_job_slot", "waiting_for_gpu", "waiting_for_mobilesam", "waiting_for_vlm", "retry_wait", "paused")
            ),
            "meanChildRuntimeSeconds": round(mean_runtime, 3) if mean_runtime is not None else None,
            "longestChildRuntimeSeconds": round(max(processing_runtimes), 3) if processing_runtimes else None,
            "meanChildQueueWaitSeconds": (
                round(sum(queue_waits) / len(queue_waits), 3) if queue_waits else None
            ),
            "meanChildTotalElapsedSeconds": (
                round(sum(total_elapsed) / len(total_elapsed), 3) if total_elapsed else None
            ),
            "longestChildTotalElapsedSeconds": round(max(total_elapsed), 3) if total_elapsed else None,
            "cacheHitCount": cache_hits,
            "cacheMissCount": cache_misses,
            "cacheStateRecordedCount": cache_state_recorded,
            "meanCacheHitMaterializationSeconds": (
                round(sum(cache_materialization_runtimes) / len(cache_materialization_runtimes), 3)
                if cache_materialization_runtimes else None
            ),
            "processingRuntimeSampleCount": len(processing_runtimes),
            "estimatedRemainingSeconds": round(mean_runtime * remaining, 3) if eta_confident and mean_runtime is not None else None,
            "estimatedRemainingConfidence": "sufficient" if eta_confident else "insufficient",
            "metricDefinitions": {
                "batchRuntimeSeconds": "batch creation to current time, or terminal child completion when terminal",
                "meanChildRuntimeSeconds": "mean started-to-finished processing time for completed cache misses",
                "meanCacheHitMaterializationSeconds": "mean started-to-finished cache lookup/materialization time",
                "meanChildQueueWaitSeconds": "mean child creation-to-processing-start time",
                "meanChildTotalElapsedSeconds": "mean child creation-to-terminal time",
            },
        }

    def delete(self, batch_id: str, job_store: JobStore) -> bool:
        record = self.get(batch_id, job_store)
        if record is None:
            return False
        if any((job_store.get(job_id) is not None and job_store.require(job_id).status == "running") for job_id in record.job_ids):
            raise RuntimeError("Batch has active child jobs and cannot be deleted.")
        for job_id in record.job_ids:
            job_store.delete(job_id)
        self._batches.pop(batch_id, None)
        self._delete_record_dir(record)
        return True

    def pause(self, batch_id: str, job_store: JobStore) -> Dict[str, Any]:
        record = self.require(batch_id, job_store)
        paused = [job_id for job_id in record.job_ids if job_store.pause(job_id)]
        record.paused = True
        record.status = "paused"
        record.updated_at = _utc_now()
        self.persist_batch(record)
        return {"batchId": batch_id, "status": record.status, "pausedJobIds": paused}

    def resume(self, batch_id: str, job_store: JobStore) -> Dict[str, Any]:
        record = self.require(batch_id, job_store)
        resumed = [job_id for job_id in record.job_ids if job_store.resume(job_id)]
        record.paused = False
        self.refresh(record, job_store)
        return {"batchId": batch_id, "status": record.status, "resumedJobIds": resumed}

    def retry_failed(self, batch_id: str, job_store: JobStore) -> Dict[str, Any]:
        record = self.require(batch_id, job_store)
        retried = [job_id for job_id in record.job_ids if job_store.reset_for_retry(job_id)]
        self.refresh(record, job_store)
        return {"batchId": batch_id, "status": record.status, "retriedJobIds": retried}

    def cancel(self, batch_id: str, job_store: JobStore) -> Dict[str, Any]:
        record = self.require(batch_id, job_store)
        cancelled: list[str] = []
        active: list[str] = []
        for job_id in record.job_ids:
            job = job_store.get(job_id)
            if job is None or job.status in TERMINAL_STATES:
                continue
            if job.status == "running":
                active.append(job_id)
                continue
            job_store.update_status(
                job_id,
                status="cancelled",
                progress=1.0,
                current_step="Cancelled before analysis",
                error="Cancelled by user.",
            )
            cancelled.append(job_id)
        self.refresh(record, job_store)
        return {
            "batchId": batch_id,
            "status": record.status,
            "cancelledJobIds": cancelled,
            "activeJobsCompletingSafely": active,
        }

    def results(self, batch_id: str, job_store: JobStore) -> Dict[str, Any]:
        record = self.require(batch_id, job_store)
        completed = []
        results_by_job_id: Dict[str, Any] = {}
        for item in record.accepted_files:
            job = job_store.get(item.job_id)
            if job is not None and job.result is not None:
                owned = owned_result_snapshot(job)
                child = {"jobId": item.job_id, "source": item.payload(), "result": owned}
                completed.append(child)
                results_by_job_id[item.job_id] = child
        return {
            "batchId": batch_id,
            "status": record.status,
            "results": completed,
            "resultsByJobId": results_by_job_id,
            "rawEvidenceFlattened": False,
        }

    def persist_batch(self, record: BatchRecord) -> None:
        try:
            record.persistence_warning = None
            _atomic_write_json(record.manifest_path, record.payload())
        except BatchManifestPersistenceError as exc:
            record.persistence_warning = str(exc)
            logger.warning("Batch manifest persistence failed for %s: %s", record.batch_id, exc)
            if record.manifest_path.is_file():
                return
            raise

    def load_batch(self, batch_id: str) -> Optional[BatchRecord]:
        if not _is_safe_batch_id(batch_id):
            return None
        manifest_path = self.root_dir / batch_id / BATCH_MANIFEST_FILENAME
        return self._load_manifest(manifest_path)

    def _create_empty_batch(self, source_filename: str, *, import_key: Optional[str] = None) -> BatchRecord:
        batch_id = _new_batch_id()
        batch_dir = self.root_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        now = _utc_now()
        batch = BatchRecord(
            batch_id=batch_id,
            status="queued",
            source_filename=source_filename,
            batch_dir=batch_dir,
            import_key=import_key,
            display_label=Path(source_filename).stem or source_filename,
            created_at=now,
            updated_at=now,
        )
        self._batches[batch_id] = batch
        self.persist_batch(batch)
        return batch

    def _validate_archive_shape(self, entries: list[zipfile.ZipInfo]) -> None:
        if not entries:
            raise BatchValidationError("Uploaded archive does not contain any files.", status_code=400)
        if len(entries) > int(SETTINGS.bulk_max_files):
            raise BatchValidationError(
                f"Archive contains too many files. Maximum is {SETTINGS.bulk_max_files}.",
                status_code=413,
            )
        total_uncompressed = sum(max(int(entry.file_size), 0) for entry in entries)
        if total_uncompressed > _bulk_uncompressed_limit_bytes():
            limit_mb = _bulk_uncompressed_limit_bytes() / (1024 * 1024)
            raise BatchValidationError(
                f"Archive is too large after extraction. Maximum uncompressed size is {limit_mb:.1f} MB.",
                status_code=413,
            )
        maximum_ratio = max(float(getattr(SETTINGS, "bulk_max_compression_ratio", 100.0)), 1.0)
        for entry in entries:
            if entry.file_size <= 0:
                continue
            ratio = float(entry.file_size) / max(float(entry.compress_size), 1.0)
            if ratio > maximum_ratio:
                raise BatchValidationError(
                    f"Archive entry exceeds the compression-ratio limit: {entry.filename}",
                    status_code=413,
                )

    def _create_job_for_stream(
        self,
        *,
        batch: BatchRecord,
        job_store: JobStore,
        original_filename: str,
        filename: str,
        stream,
        expected_size: Optional[int],
        query: str,
        settings: AnalysisSettings,
    ) -> JobRecord:
        path_meta = _source_path_metadata(original_filename)
        source_metadata = {
            **path_meta,
            "originalFilename": original_filename,
            "batchId": batch.batch_id,
            "batchDisplayLabel": batch.display_label or batch.source_filename,
            "recoverOnRestart": True,
            "batchEnqueueSequence": int(batch.created_at.timestamp() * 1000),
            "childSequence": len(batch.accepted_files) + 1,
        }
        record = job_store.create_job_from_stream(
            filename=filename,
            stream=stream,
            query=query,
            settings=settings,
            source_metadata=source_metadata,
            expected_size=expected_size,
        )
        record.metrics["batchId"] = batch.batch_id
        record.metrics["batchSourceFilename"] = batch.source_filename
        record.metrics["recoverOnRestart"] = True
        job_store.persist_job(record)
        checksum = str(record.source_metadata.get("checksumSha256") or "")
        identity = _file_identity(batch.batch_id, path_meta["sourceRelativePath"], record.size_bytes, checksum)
        record.source_metadata["fileIdentity"] = identity
        job_store.persist_job(record)
        batch.accepted_files.append(
            BatchFile(
                original_filename=original_filename,
                filename=filename,
                size_bytes=record.size_bytes,
                media_type=_media_type_for_batch(filename),
                job_id=record.job_id,
                status=record.status,
                source_relative_path=path_meta["sourceRelativePath"],
                source_directory=path_meta["sourceDirectory"],
                source_group_path=path_meta["sourceGroupPath"],
                source_group_label=path_meta["sourceGroupLabel"],
                checksum_sha256=checksum,
                imported_at=str(record.source_metadata.get("importedAt") or ""),
                file_identity=identity,
                batch_enqueue_sequence=int(record.source_metadata.get("batchEnqueueSequence") or 0) or None,
                child_sequence=int(record.source_metadata.get("childSequence") or 0) or None,
            )
        )
        if not batch.analysis_setup:
            batch.analysis_setup = build_analysis_setup_payload(record)
        return record

    def _find_by_import_key(self, import_key: Optional[str], job_store: JobStore) -> Optional[BatchRecord]:
        key = (import_key or "").strip()
        if not key:
            return None
        with self._lock:
            for batch_dir in self.root_dir.iterdir():
                if not batch_dir.is_dir():
                    continue
                record = self._batches.get(batch_dir.name) or self.load_batch(batch_dir.name)
                if record is not None:
                    self._batches[record.batch_id] = record
                    if record.import_key == key:
                        return self.refresh(record, job_store)
        return None

    def _finalize_created_batch(self, batch: BatchRecord, job_store: JobStore) -> None:
        self.refresh(batch, job_store)

    def _derive_status(self, record: BatchRecord) -> BatchState:
        if not record.accepted_files:
            return "failed"
        statuses = [item.status for item in record.accepted_files]
        if record.paused and not any(status == "running" for status in statuses):
            return "paused"
        if any(status in {"running", "recovering"} for status in statuses):
            return "running"
        if any(status == "retry_wait" for status in statuses):
            return "retry_wait"
        if any(status in {"waiting_for_capacity", "waiting_for_job_slot"} for status in statuses):
            return "waiting_for_capacity"
        if any(status == "waiting_for_gpu" for status in statuses):
            return "waiting_for_gpu"
        if any(status == "waiting_for_mobilesam" for status in statuses):
            return "waiting_for_mobilesam"
        if any(status == "waiting_for_vlm" for status in statuses):
            return "waiting_for_vlm"
        if any(status in {"running_preprocess", "running_detector", "running_refinement", "running_report"} for status in statuses):
            return "running"
        if any(status == "queued" for status in statuses):
            return "queued"
        if all(status == "completed" for status in statuses) and not record.rejected_files:
            return "completed"
        if any(status == "completed" for status in statuses):
            return "completed_with_failures"
        if all(status == "cancelled" for status in statuses):
            return "cancelled"
        return "failed"

    def _load_manifest(self, manifest_path: Path) -> Optional[BatchRecord]:
        root = self.root_dir.resolve()
        batch_dir = manifest_path.parent.resolve()
        if batch_dir == root or root not in batch_dir.parents:
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        batch_id = str(manifest.get("batchId") or batch_dir.name)
        if not _is_safe_batch_id(batch_id):
            return None

        accepted = [
            BatchFile(
                original_filename=str(item.get("originalFilename") or item.get("filename") or "upload.mp4"),
                filename=safe_filename(str(item.get("filename") or "upload.mp4")),
                size_bytes=int(item.get("sizeBytes") or 0),
                media_type=str(item.get("mediaType") or "video"),
                job_id=str(item.get("jobId") or ""),
                source_relative_path=str(item.get("sourceRelativePath") or item.get("originalFilename") or ""),
                source_directory=str(item.get("sourceDirectory") or ""),
                source_group_path=str(item.get("sourceGroupPath") or ""),
                source_group_label=str(item.get("sourceGroupLabel") or "Ungrouped"),
                checksum_sha256=item.get("checksumSha256"),
                imported_at=item.get("importedAt"),
                file_identity=item.get("fileIdentity"),
                status=str(item.get("status") or "queued"),
                error=item.get("error"),
                violation_count=int(item.get("violationCount") or 0),
                enqueue_sequence=item.get("enqueueSequence"),
                batch_enqueue_sequence=item.get("batchEnqueueSequence"),
                child_sequence=item.get("childSequence"),
                queue_position=item.get("queuePosition"),
                worker_slot=item.get("workerSlot"),
                capacity_reason=item.get("capacityReason"),
                requested_mode_label=item.get("requestedModeLabel"),
                actual_review=item.get("actualReview"),
                device_label=item.get("deviceLabel"),
                mobile_sam_status=item.get("mobileSamStatus"),
            )
            for item in list(manifest.get("acceptedFiles") or [])
            if item.get("jobId")
        ]
        rejected = [
            RejectedBatchFile(
                filename=str(item.get("filename") or "unknown"),
                reason=str(item.get("reason") or "Rejected"),
                source_relative_path=item.get("sourceRelativePath"),
                category=str(item.get("category") or "rejected"),
            )
            for item in list(manifest.get("rejectedFiles") or [])
        ]
        status = str(manifest.get("status") or "failed")
        if status not in {
            "importing", "queued", "waiting_for_capacity", "waiting_for_job_slot", "waiting_for_gpu", "waiting_for_mobilesam", "waiting_for_vlm", "running_preprocess", "running_detector", "running_refinement", "running_report", "running", "retry_wait", "paused", "recovering",
            "completed", "completed_with_failures", "failed", "partial", "cancelled",
        }:
            status = "failed"

        return BatchRecord(
            batch_id=batch_id,
            status=status,  # type: ignore[arg-type]
            source_filename=safe_filename(str(manifest.get("sourceFilename") or "bulk-upload")),
            batch_dir=batch_dir,
            accepted_files=accepted,
            rejected_files=rejected,
            status_counts=dict(manifest.get("statusCounts") or {}),
            import_key=manifest.get("importKey"),
            display_label=manifest.get("batchDisplayLabel"),
            paused=bool(manifest.get("paused")),
            group_summaries=list(manifest.get("groupSummaries") or []),
            hierarchy=dict(manifest.get("hierarchy") or {}),
            throughput=dict(manifest.get("throughput") or {}),
            analysis_setup=dict(manifest.get("analysisSetup") or {}),
            persistence_warning=manifest.get("persistenceWarning"),
            created_at=_parse_datetime(manifest.get("createdAt")),
            updated_at=_parse_datetime(manifest.get("updatedAt")),
        )

    def _is_safe_batch_dir(self, batch_dir: Path) -> bool:
        root = self.root_dir.resolve()
        target = batch_dir.resolve()
        return target != root and root in target.parents

    def _delete_record_dir(self, record: BatchRecord) -> None:
        if not self._is_safe_batch_dir(record.batch_dir):
            raise RuntimeError("Refusing to delete batch directory outside API batch root.")
        shutil.rmtree(record.batch_dir, ignore_errors=True)
