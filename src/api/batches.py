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

from .jobs import AnalysisSettings, JobRecord, JobStore, MEDIA_EXTENSIONS, max_upload_bytes, safe_filename

BatchState = Literal["queued", "running", "completed", "failed", "partial", "cancelled"]
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
    status: str = "queued"
    error: Optional[str] = None

    def payload(self) -> Dict[str, Any]:
        return {
            "originalFilename": self.original_filename,
            "filename": self.filename,
            "sizeBytes": self.size_bytes,
            "mediaType": self.media_type,
            "jobId": self.job_id,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class RejectedBatchFile:
    filename: str
    reason: str

    def payload(self) -> Dict[str, str]:
        return {"filename": self.filename, "reason": self.reason}


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
            "acceptedFiles": [item.payload() for item in self.accepted_files],
            "rejectedFiles": [item.payload() for item in self.rejected_files],
            "jobIds": self.job_ids,
            "statusCounts": self.status_counts,
            "createdAt": _to_iso(self.created_at),
            "updatedAt": _to_iso(self.updated_at),
            "persistenceWarning": self.persistence_warning,
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

    def create_from_zip(
        self,
        *,
        filename: str,
        content: bytes,
        query: str,
        settings: AnalysisSettings,
        job_store: JobStore,
    ) -> BatchRecord:
        source_filename = safe_filename(filename or "upload.zip")
        batch = self._create_empty_batch(source_filename)
        used_filenames: set[str] = set()

        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                entries = [entry for entry in archive.infolist() if not entry.is_dir()]
                self._validate_archive_shape(entries)

                for entry in entries:
                    safe_member = _safe_zip_member_name(entry.filename)
                    if safe_member is None:
                        raise BatchValidationError(
                            f"Archive contains an unsafe path: {entry.filename}",
                            status_code=400,
                        )

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

                    with archive.open(entry) as stream:
                        media_bytes = stream.read()
                    clean_name = _unique_filename(Path(safe_member).name, used_filenames)
                    self._create_job_for_file(
                        batch=batch,
                        job_store=job_store,
                        original_filename=safe_member,
                        filename=clean_name,
                        content=media_bytes,
                        query=query,
                        settings=settings,
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
    ) -> BatchRecord:
        batch = self._create_empty_batch(safe_filename(source_filename or "bulk-upload"))
        used_filenames: set[str] = set()
        materialized = list(files)
        if len(materialized) > int(SETTINGS.bulk_max_files):
            self._delete_record_dir(batch)
            self._batches.pop(batch.batch_id, None)
            raise BatchValidationError(
                f"Bulk upload contains too many files. Maximum is {SETTINGS.bulk_max_files}.",
                status_code=413,
            )

        for original_name, content in materialized:
            if not _is_video_filename(original_name):
                batch.rejected_files.append(
                    RejectedBatchFile(
                        filename=original_name,
                        reason="Unsupported file type for bulk video analysis.",
                    )
                )
                continue
            if not content:
                batch.rejected_files.append(
                    RejectedBatchFile(
                        filename=original_name,
                        reason="File is empty.",
                    )
                )
                continue
            if len(content) > max_upload_bytes():
                batch.rejected_files.append(
                    RejectedBatchFile(
                        filename=original_name,
                        reason="File exceeds the per-video upload limit.",
                    )
                )
                continue
            clean_name = _unique_filename(original_name, used_filenames)
            self._create_job_for_file(
                batch=batch,
                job_store=job_store,
                original_filename=original_name,
                filename=clean_name,
                content=content,
                query=query,
                settings=settings,
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
                counts[item.status] = counts.get(item.status, 0) + 1

            next_status = self._derive_status(record)
            manifest_missing = not record.manifest_path.is_file()
            if record.status_counts != counts:
                changed = True
                record.status_counts = counts
            if record.status != next_status:
                changed = True
                record.status = next_status

            if changed or manifest_missing:
                record.updated_at = _utc_now()
                self.persist_batch(record)
        return record

    def delete(self, batch_id: str, job_store: JobStore) -> bool:
        record = self.get(batch_id, job_store)
        if record is None:
            return False
        for job_id in record.job_ids:
            job_store.delete(job_id)
        self._batches.pop(batch_id, None)
        self._delete_record_dir(record)
        return True

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

    def _create_empty_batch(self, source_filename: str) -> BatchRecord:
        batch_id = _new_batch_id()
        batch_dir = self.root_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=False)
        now = _utc_now()
        batch = BatchRecord(
            batch_id=batch_id,
            status="queued",
            source_filename=source_filename,
            batch_dir=batch_dir,
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

    def _create_job_for_file(
        self,
        *,
        batch: BatchRecord,
        job_store: JobStore,
        original_filename: str,
        filename: str,
        content: bytes,
        query: str,
        settings: AnalysisSettings,
    ) -> JobRecord:
        record = job_store.create_job(
            filename=filename,
            content=content,
            query=query,
            settings=settings,
        )
        record.metrics["batchId"] = batch.batch_id
        record.metrics["batchSourceFilename"] = batch.source_filename
        job_store.persist_job(record)
        batch.accepted_files.append(
            BatchFile(
                original_filename=original_filename,
                filename=filename,
                size_bytes=len(content),
                media_type=_media_type_for_batch(filename),
                job_id=record.job_id,
                status=record.status,
            )
        )
        return record

    def _finalize_created_batch(self, batch: BatchRecord, job_store: JobStore) -> None:
        self.refresh(batch, job_store)

    def _derive_status(self, record: BatchRecord) -> BatchState:
        if not record.accepted_files:
            return "failed"
        statuses = [item.status for item in record.accepted_files]
        if any(status == "running" for status in statuses):
            return "running"
        if any(status == "queued" for status in statuses):
            return "queued"
        if all(status == "completed" for status in statuses) and not record.rejected_files:
            return "completed"
        if any(status == "completed" for status in statuses):
            return "partial"
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
                status=str(item.get("status") or "queued"),
                error=item.get("error"),
            )
            for item in list(manifest.get("acceptedFiles") or [])
            if item.get("jobId")
        ]
        rejected = [
            RejectedBatchFile(
                filename=str(item.get("filename") or "unknown"),
                reason=str(item.get("reason") or "Rejected"),
            )
            for item in list(manifest.get("rejectedFiles") or [])
        ]
        status = str(manifest.get("status") or "failed")
        if status not in {"queued", "running", "completed", "failed", "partial", "cancelled"}:
            status = "failed"

        return BatchRecord(
            batch_id=batch_id,
            status=status,  # type: ignore[arg-type]
            source_filename=safe_filename(str(manifest.get("sourceFilename") or "bulk-upload")),
            batch_dir=batch_dir,
            accepted_files=accepted,
            rejected_files=rejected,
            status_counts=dict(manifest.get("statusCounts") or {}),
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
