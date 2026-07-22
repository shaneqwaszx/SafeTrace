"""Atomic verified result exports and export lifecycle management."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .jobs import JobRecord, JobStore, owned_result_snapshot
from .result_cache import cache_identity

_EXPORT_LOCK = threading.RLock()


def _publish_export_directory(temporary: Path, destination: Path, *, retries: int = 5) -> None:
    """Atomically publish a complete export, retrying transient Windows locks.

    Defender, indexing, and temporary-directory scanners can briefly retain a
    handle on a just-written directory.  The destination is unique, so a retry
    cannot overwrite another export; until it succeeds only the hidden
    temporary directory exists.
    """
    last_error: PermissionError | None = None
    for attempt in range(max(1, int(retries))):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt + 1 < max(1, int(retries)):
                time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_root_for(store: JobStore) -> Path:
    return store.root_dir.parent / "exports"


def _safe_export(root: Path, export_id: str) -> Path:
    if not export_id.startswith("export_") or not export_id.replace("_", "").isalnum():
        raise ValueError("Invalid export ID")
    path = (root / export_id).resolve()
    if path == root.resolve() or root.resolve() not in path.parents:
        raise ValueError("Unsafe export path")
    return path


def verify_export(path: Path) -> dict[str, Any]:
    manifest_path = path / "export_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"verified": False, "failures": ["manifest_unreadable"]}
    failures = []
    for item in manifest.get("files") or []:
        file_path = path / str(item.get("path") or "")
        if not file_path.is_file():
            failures.append(f"missing:{item.get('path')}")
        elif sha256(file_path) != item.get("sha256"):
            failures.append(f"checksum:{item.get('path')}")
    return {"verified": not failures, "failures": failures, "manifest": manifest}


def create_export(
    store: JobStore,
    job_id: str,
    *,
    selected_evidence_ids: Iterable[str] | None = None,
    include_evidence_images: bool = True,
) -> dict[str, Any]:
    record = store.require(job_id)
    if record.status != "completed" or record.result is None:
        raise ValueError("Only completed jobs can be exported.")
    owned_result = owned_result_snapshot(record)
    selected = {str(value) for value in selected_evidence_ids or []}
    root = export_root_for(store)
    root.mkdir(parents=True, exist_ok=True)
    export_id = f"export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    destination = _safe_export(root, export_id)
    temporary = root / f".{export_id}.{os.getpid()}.tmp"
    created_at = datetime.now(timezone.utc).isoformat()
    files: list[dict[str, Any]] = []
    with _EXPORT_LOCK:
        record.source_metadata["exportInProgress"] = True
        store.persist_job(record)
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        try:
            summary = {
                "jobId": job_id,
                "status": record.status,
                "query": record.query,
                "summary": owned_result.get("summary"),
                "violations": owned_result.get("violations"),
                "events": owned_result.get("events"),
                "evidenceStatus": owned_result.get("evidenceStatus"),
                "analysisSetup": owned_result.get("analysisSetup"),
                "sourceMetadata": record.source_metadata,
            }
            technical = owned_result
            evidence = []
            evidence_frames = (
                owned_result.get("evidence")
                if owned_result.get("evidenceStatus") is not None
                else owned_result.get("frames")
            ) or []
            for frame in evidence_frames:
                frame_id = str(frame.get("id") or "")
                evidence_id = str(frame.get("evidenceId") or "")
                if selected and frame_id not in selected and evidence_id not in selected:
                    continue
                evidence.append({
                    "evidenceId": evidence_id,
                    "mediaArtifactId": frame.get("mediaArtifactId"),
                    "frameId": frame_id,
                    "frameNumber": frame.get("frameNumber"),
                    "timestamp": frame.get("timestamp"),
                    "timestampSeconds": frame.get("timestampSeconds"),
                    "sourceRelativePath": frame.get("sourceRelativePath"),
                    "sourceGroup": frame.get("sourceGroup"),
                    "batchId": frame.get("batchId"),
                    "jobId": frame.get("jobId"),
                    "sourceChecksum": frame.get("sourceChecksum"),
                    "originalFilename": frame.get("originalFilename"),
                    "findingId": frame.get("findingId"),
                    "violations": frame.get("violations"),
                    "actualAnalysisMode": frame.get("explanationSource"),
                    "imageUrl": frame.get("imageUrl"),
                })
            payloads = {
                "result_summary.json": summary,
                "technical_result.json": technical,
                "evidence_metadata.json": evidence,
            }
            for filename, payload in payloads.items():
                path = temporary / filename
                path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
                files.append({"path": filename, "sizeBytes": path.stat().st_size, "sha256": sha256(path)})
            if include_evidence_images:
                evidence_dir = temporary / "evidence"
                evidence_dir.mkdir()
                wanted_names = {
                    Path(str(frame.get("imageUrl") or "")).name
                    for frame in evidence_frames
                    if (
                        not selected
                        or str(frame.get("id") or "") in selected
                        or str(frame.get("evidenceId") or "") in selected
                    )
                }
                for filename, source in sorted(record.media_files.items()):
                    if wanted_names and filename not in wanted_names:
                        continue
                    if source.is_file():
                        target = evidence_dir / filename
                        shutil.copy2(source, target)
                        files.append({"path": target.relative_to(temporary).as_posix(), "sizeBytes": target.stat().st_size, "sha256": sha256(target)})
            manifest = {
                "schemaVersion": 2,
                "exportId": export_id,
                "jobId": job_id,
                "batchId": record.source_metadata.get("batchId"),
                "createdAt": created_at,
                "status": "completed",
                "originalVideoIncluded": False,
                "selectedEvidenceIds": sorted(selected),
                "source": {
                    "filename": record.original_filename,
                    "sourceRelativePath": record.source_metadata.get("sourceRelativePath"),
                    "sourceChecksum": record.source_metadata.get("checksumSha256"),
                },
                "executionIdentity": cache_identity(record),
                "resultOwnership": (owned_result.get("technicalDetails") or {}).get("ownership"),
                "files": sorted(files, key=lambda item: item["path"]),
            }
            manifest_path = temporary / "export_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
            _publish_export_directory(temporary, destination)
            verification = verify_export(destination)
            if not verification["verified"]:
                raise RuntimeError(f"Export verification failed: {verification['failures']}")
            record.source_metadata["exported"] = True
            record.source_metadata.pop("exportInProgress", None)
            record.source_metadata["lastExportId"] = export_id
            record.source_metadata["lastExportAt"] = created_at
            store.persist_job(record)
            return export_payload(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            if destination.exists() and not verify_export(destination)["verified"]:
                shutil.rmtree(destination, ignore_errors=True)
            current = store.get(job_id)
            if current is not None:
                current.source_metadata.pop("exportInProgress", None)
                store.persist_job(current)
            raise


def export_payload(path: Path) -> dict[str, Any]:
    verification = verify_export(path)
    manifest = verification.get("manifest") or {}
    return {
        "exportId": path.name,
        "jobId": manifest.get("jobId"),
        "batchId": manifest.get("batchId"),
        "status": "verified" if verification["verified"] else "failed",
        "createdAt": manifest.get("createdAt"),
        "sizeBytes": sum(item.stat().st_size for item in path.rglob("*") if item.is_file()),
        "fileCount": sum(1 for item in path.rglob("*") if item.is_file()),
        "verified": verification["verified"],
        "verificationFailures": verification["failures"],
        "path": str(path),
        "manifestPath": str(path / "export_manifest.json"),
    }


def list_exports(store: JobStore) -> list[dict[str, Any]]:
    root = export_root_for(store)
    if not root.exists():
        return []
    return [export_payload(path) for path in sorted(root.glob("export_*"), reverse=True) if path.is_dir()]


def delete_export(store: JobStore, export_id: str) -> dict[str, Any]:
    root = export_root_for(store)
    path = _safe_export(root, export_id)
    with _EXPORT_LOCK:
        if not path.is_dir():
            return {"exportId": export_id, "status": "not_found", "deletedBytes": 0}
        before = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        payload = export_payload(path)
        shutil.rmtree(path)
        job_id = str(payload.get("jobId") or "")
        record = store.get(job_id) if job_id else None
        if record is not None:
            remaining = any(item.get("jobId") == job_id for item in list_exports(store))
            if not remaining:
                record.source_metadata["exported"] = False
                record.source_metadata.pop("lastExportId", None)
                record.source_metadata.pop("lastExportAt", None)
                store.persist_job(record)
        return {"exportId": export_id, "status": "deleted", "deletedBytes": before}
