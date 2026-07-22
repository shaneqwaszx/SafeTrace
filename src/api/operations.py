"""Operational recovery, dashboard, and storage lifecycle helpers."""
from __future__ import annotations

import json
import shutil
import time
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.config import SETTINGS

from .batches import BatchStore
from .jobs import ACTIVE_STATES, JobRecord, JobStore, TERMINAL_STATES

RECOVERY_ACTIONS = {"requeue_after_restart", "manual_retry_required", "awaiting_recovery_decision"}
PROTECTED_STATES = set(ACTIVE_STATES) | {"paused"}
_DASHBOARD_CACHE_LOCK = threading.Lock()
_DASHBOARD_CACHE: dict[tuple[int, int], dict[str, Any]] = {}
_CLEANUP_LOCK = threading.Lock()


class CleanupInProgressError(RuntimeError):
    pass


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def recovery_candidates(
    jobs: JobStore,
    batches: BatchStore,
    *,
    include_deferred: bool = False,
) -> dict[str, Any]:
    jobs.mark_stale_running_jobs()
    candidates: list[dict[str, Any]] = []
    for record in jobs.records(include_disk=True):
        deferred = record.recovery_action == "deferred_by_user"
        interrupted = record.status in {"recovering", "paused"} and record.recovery_action in RECOVERY_ACTIONS
        manual = record.status == "failed" and record.error_type == "InterruptedJob" and not deferred
        if not (interrupted or manual or (include_deferred and deferred)):
            continue
        batch_id = str(record.source_metadata.get("batchId") or "") or None
        try:
            from .recovery_checkpoints import latest_valid_checkpoint

            checkpoint = latest_valid_checkpoint(
                record.job_dir,
                dict(record.metrics.get("executionIdentity") or {}),
                expected_job_id=record.job_id,
            )
        except Exception:
            checkpoint = {"checkpoint": None, "checkpointPath": None, "invalidationReason": "checkpoint_read_failed", "invalidNewerCheckpoints": []}
        candidates.append({
            "jobId": record.job_id,
            "batchId": batch_id,
            "sourceRelativePath": record.source_metadata.get("sourceRelativePath") or record.original_filename,
            "sourceGroupPath": record.source_metadata.get("sourceGroupPath") or "",
            "status": record.status,
            "lastState": record.status,
            "lastStage": record.current_step,
            "progress": record.progress,
            "progressPercent": round(record.progress * 100),
            "resumable": record.upload_path.is_file(),
            "resumableReason": "Upload is durable; the current analysis stage will restart safely." if record.upload_path.is_file() else "Source upload is missing.",
            "proposedAction": "resume_current_job" if record.upload_path.is_file() else "discard",
            "retryCount": record.retry_count,
            "maxAttempts": record.max_attempts,
            "storedBytes": directory_size(record.job_dir),
            "lastUpdate": record.updated_at.isoformat(),
            "recoveryAction": record.recovery_action,
            "lastValidCheckpoint": checkpoint.get("checkpoint"),
            "lastValidCheckpointPath": checkpoint.get("checkpointPath"),
            "checkpointInvalidationReason": checkpoint.get("invalidationReason"),
            "invalidCheckpointCount": len(checkpoint.get("invalidNewerCheckpoints") or []),
        })
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        groups[item.get("batchId") or "standalone"].append(item)
    batch_payloads = []
    for batch_id, children in groups.items():
        batch = batches.get(batch_id, jobs) if batch_id != "standalone" else None
        batch_payloads.append({
            "batchId": None if batch_id == "standalone" else batch_id,
            "sourceFilename": batch.source_filename if batch else "Standalone jobs",
            "sourceHierarchy": batch.hierarchy if batch else None,
            "candidateCount": len(children),
            "storedBytes": sum(int(item["storedBytes"]) for item in children),
            "jobs": children,
        })
    return {
        "policy": normalized_recovery_policy(),
        "candidateCount": len(candidates),
        "storedBytes": sum(int(item["storedBytes"]) for item in candidates),
        "batches": batch_payloads,
        "jobs": candidates,
    }


def normalized_recovery_policy() -> str:
    value = str(getattr(SETTINGS, "recovery_policy", "prompt") or "prompt").strip().lower()
    return value if value in {"prompt", "auto_resume", "discard"} else "prompt"


def storage_summary(jobs: JobStore, batches: BatchStore) -> dict[str, Any]:
    records = jobs.records(include_disk=False)
    categories = Counter()
    for record in records:
        upload_bytes = directory_size(record.job_dir / "uploads")
        evidence_bytes = directory_size(record.output_dir)
        result_bytes = directory_size(record.result_path) if record.result_path else 0
        total_job_bytes = directory_size(record.job_dir)
        state_bytes = max(total_job_bytes - upload_bytes - evidence_bytes - result_bytes, 0)
        categories["uploads"] += upload_bytes
        categories["evidence"] += evidence_bytes
        categories["completedResults" if record.status == "completed" else "durableState"] += result_bytes
        if record.status in {"recovering", "paused"} or record.error_type == "InterruptedJob":
            categories["resumableWork"] += state_bytes
        elif record.status in PROTECTED_STATES:
            categories["activeWork"] += state_bytes
        else:
            categories["durableState"] += state_bytes
    data_root = Path(SETTINGS.data_dir)
    known_roots = {jobs.root_dir.resolve(), batches.root_dir.resolve()}
    unclassified = 0
    if data_root.exists():
        for child in data_root.iterdir():
            if child.resolve() not in known_roots and child.name not in {"frames", "result_cache", "logs", "exports"}:
                unclassified += directory_size(child)
    categories["cache"] = directory_size(data_root / "result_cache")
    categories["logs"] = directory_size(Path("logs")) + directory_size(data_root / "logs")
    categories["exports"] = directory_size(data_root / "exports")
    categories["orphanedOrUnclassified"] = unclassified
    usage = shutil.disk_usage(data_root if data_root.exists() else Path.cwd())
    return {
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "runtimeRoot": str(data_root),
        "categories": dict(categories),
        "totalManagedBytes": sum(categories.values()),
        "freeBytes": usage.free,
        "totalDiskBytes": usage.total,
        "minimumFreeBytes": int(float(getattr(SETTINGS, "min_free_disk_gb", 2.0)) * 1024**3),
        "storagePressure": usage.free < int(float(getattr(SETTINGS, "min_free_disk_gb", 2.0)) * 1024**3),
        "retention": retention_payload(),
        "protected": {
            "activeJobs": sum(record.status in PROTECTED_STATES for record in records),
            "resumableJobs": sum(record.status == "recovering" for record in records),
            "pinnedJobs": sum(bool(record.source_metadata.get("pinned")) for record in records),
            "exportedJobs": sum(bool(record.source_metadata.get("exported")) for record in records),
        },
    }


def retention_payload() -> dict[str, Any]:
    return {
        "completedRetentionDays": SETTINGS.completed_retention_days,
        "failedRetentionDays": SETTINGS.failed_retention_days,
        "recoveryRetentionHours": SETTINGS.recovery_retention_hours,
        "cacheRetentionDays": SETTINGS.cache_retention_days,
        "cacheMaxGb": SETTINGS.cache_max_gb,
        "logRetentionDays": SETTINGS.log_retention_days,
        "recoveryPolicy": normalized_recovery_policy(),
    }


def cleanup_preview(jobs: JobStore, *, scopes: Iterable[str] | None = None) -> dict[str, Any]:
    requested = set(scopes or {"expired_completed", "expired_failed", "expired_recovery", "orphaned_temp", "result_cache", "logs"})
    now = datetime.now(timezone.utc)
    candidates: list[dict[str, Any]] = []
    for record in jobs.records(include_disk=True):
        if record.status in PROTECTED_STATES or record.source_metadata.get("pinned") or record.source_metadata.get("exported") or record.source_metadata.get("exportInProgress"):
            continue
        age = now - (record.finished_at or record.updated_at or record.created_at)
        reason = None
        if record.status == "completed" and "expired_completed" in requested and age.total_seconds() >= SETTINGS.completed_retention_days * 86400:
            reason = "expired_completed"
        elif record.status in {"failed", "cancelled"} and "expired_failed" in requested and age.total_seconds() >= SETTINGS.failed_retention_days * 86400:
            reason = "expired_failed"
        elif record.error_type == "InterruptedJob" and "expired_recovery" in requested and age.total_seconds() >= SETTINGS.recovery_retention_hours * 3600:
            reason = "expired_recovery"
        if reason:
            candidates.append({"kind": "job", "id": record.job_id, "reason": reason, "bytes": directory_size(record.job_dir)})
    orphan_root = jobs.root_dir
    if "orphaned_temp" in requested and orphan_root.exists():
        for path in orphan_root.rglob("manifest.json.*.tmp"):
            candidates.append({"kind": "temporary", "path": str(path), "reason": "orphaned_temp", "bytes": directory_size(path)})
    cache_root = Path(SETTINGS.data_dir) / "result_cache"
    if "result_cache" in requested and cache_root.exists():
        cutoff = time.time() - SETTINGS.cache_retention_days * 86400
        cache_entries = []
        protected_job_ids = {
            record.job_id for record in jobs.records()
            if record.status in PROTECTED_STATES or record.source_metadata.get("pinned") or record.source_metadata.get("exported")
        }
        for entry in cache_root.iterdir():
            metadata = {}
            try:
                metadata = json.loads((entry / "metadata.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            references = {str(value) for value in metadata.get("jobReferences") or []}
            protected = bool(references & protected_job_ids)
            last_access = metadata.get("lastAccessAt") or metadata.get("createdAt")
            try:
                accessed_epoch = datetime.fromisoformat(str(last_access).replace("Z", "+00:00")).timestamp()
            except ValueError:
                accessed_epoch = entry.stat().st_mtime
            cache_entries.append({"path": entry, "bytes": directory_size(entry), "accessed": accessed_epoch, "protected": protected})
        selected_paths: set[str] = set()
        for entry in cache_entries:
            if not entry["protected"] and entry["accessed"] < cutoff:
                selected_paths.add(str(entry["path"]))
                candidates.append({"kind": "cache", "path": str(entry["path"]), "reason": "expired_cache", "bytes": entry["bytes"]})
        remaining_bytes = sum(entry["bytes"] for entry in cache_entries if str(entry["path"]) not in selected_paths)
        maximum_bytes = max(0, int(float(SETTINGS.cache_max_gb) * 1024**3))
        for entry in sorted(cache_entries, key=lambda item: item["accessed"]):
            if maximum_bytes <= 0 or remaining_bytes <= maximum_bytes:
                break
            if entry["protected"] or str(entry["path"]) in selected_paths:
                continue
            selected_paths.add(str(entry["path"]))
            remaining_bytes -= entry["bytes"]
            candidates.append({"kind": "cache", "path": str(entry["path"]), "reason": "cache_size_limit", "bytes": entry["bytes"]})
    if "logs" in requested:
        cutoff = time.time() - SETTINGS.log_retention_days * 86400
        for log_root in (Path("logs"), Path(SETTINGS.data_dir) / "logs"):
            if not log_root.exists():
                continue
            for path in log_root.rglob("*"):
                if path.is_file() and path.stat().st_mtime < cutoff:
                    candidates.append({"kind": "log", "path": str(path), "reason": "expired_log", "bytes": directory_size(path)})
    export_root = jobs.root_dir.parent / "exports"
    if "orphaned_temp" in requested and export_root.exists():
        for path in export_root.glob(".export_*.tmp"):
            candidates.append({"kind": "export_temporary", "path": str(path), "reason": "failed_partial_export", "bytes": directory_size(path)})
    return {
        "dryRun": True,
        "scopes": sorted(requested),
        "candidateCount": len(candidates),
        "estimatedReclaimBytes": sum(int(item["bytes"]) for item in candidates),
        "candidates": candidates,
    }


def apply_cleanup(jobs: JobStore, preview: dict[str, Any]) -> dict[str, Any]:
    if not _CLEANUP_LOCK.acquire(blocking=False):
        raise CleanupInProgressError("A retention cleanup is already running.")
    try:
        return _apply_cleanup_locked(jobs, preview)
    finally:
        _CLEANUP_LOCK.release()


def _apply_cleanup_locked(jobs: JobStore, preview: dict[str, Any]) -> dict[str, Any]:
    reclaimed = 0
    deleted: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    for item in preview["candidates"]:
        before = int(item.get("bytes") or 0)
        if item["kind"] == "job":
            record = jobs.get(item["id"])
            if record is None or record.status in PROTECTED_STATES or record.source_metadata.get("pinned") or record.source_metadata.get("exported") or record.source_metadata.get("exportInProgress"):
                retained.append({**item, "retainedReason": "protected_or_missing"})
                continue
            success = jobs.delete(item["id"])
        else:
            path = Path(item["path"])
            roots = {
                "temporary": jobs.root_dir,
                "cache": Path(SETTINGS.data_dir) / "result_cache",
                "log": Path(SETTINGS.data_dir) / "logs" if (Path(SETTINGS.data_dir) / "logs").resolve() in path.resolve().parents else Path("logs"),
                "export_temporary": jobs.root_dir.parent / "exports",
            }
            root = roots.get(item["kind"], jobs.root_dir).resolve()
            target = path.resolve()
            success = target != root and root in target.parents
            if success:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink(missing_ok=True)
        if success:
            reclaimed += before
            deleted.append(item)
        else:
            retained.append({**item, "retainedReason": "safety_check_failed"})
    audit = {
        "appliedAt": datetime.now(timezone.utc).isoformat(),
        "actualReclaimedBytes": reclaimed,
        "deleted": deleted,
        "retained": retained,
    }
    audit_dir = jobs.root_dir.parent / "cleanup_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / f"cleanup_{int(time.time())}.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def dashboard_summary(jobs: JobStore, batches: BatchStore) -> dict[str, Any]:
    records = jobs.records(include_disk=False)
    batch_records = batches.records(include_disk=False)
    signature = (
        tuple(sorted((record.job_id, record.status, record.updated_at.isoformat()) for record in records)),
        tuple(sorted((record.batch_id, record.status, record.updated_at.isoformat()) for record in batch_records)),
    )
    cache_slot = (id(jobs), id(batches))
    with _DASHBOARD_CACHE_LOCK:
        cached = _DASHBOARD_CACHE.get(cache_slot)
        if cached and cached["signature"] == signature and time.monotonic() - cached["created"] < 5.0:
            return dict(cached["payload"])
    for batch in batch_records:
        batches.refresh(batch, jobs)
    status_counts = Counter(record.status for record in records)
    violations = Counter()
    modes = Counter()
    failure_reasons = Counter()
    groups = Counter()
    runtimes: list[float] = []
    for record in records:
        modes[record.settings.review_mode] += 1
        groups[str(record.source_metadata.get("sourceGroupPath") or "Ungrouped")] += 1
        if record.error_type:
            failure_reasons[record.error_type] += 1
        if record.result:
            for finding in record.result.get("violations") or []:
                violations[str(finding.get("name") or finding.get("type") or "Unknown")] += 1
            runtime = (record.result.get("engineMetrics") or {}).get("totalDurationSeconds")
            if isinstance(runtime, (int, float)):
                runtimes.append(float(runtime))
    storage = storage_summary(jobs, batches)
    completed = status_counts.get("completed", 0)
    payload = {
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "cards": {
            "activeBatches": sum(batch.status not in {"completed", "completed_with_failures", "cancelled"} for batch in batch_records),
            "queuedVideos": sum(status_counts[key] for key in ("queued", "waiting_for_capacity", "retry_wait", "recovering", "paused")),
            "runningVideos": status_counts.get("running", 0),
            "completedVideos": completed,
            "completedWithFailures": sum(batch.status == "completed_with_failures" for batch in batch_records),
            "violationsFound": sum(violations.values()),
            "retryingOrRecovered": sum(record.retry_count > 0 for record in records),
            "storageUsedBytes": storage["totalManagedBytes"],
            "freeBytes": storage["freeBytes"],
            "meanRuntimeSeconds": sum(runtimes) / len(runtimes) if runtimes else None,
        },
        "jobsByStatus": dict(status_counts),
        "violationsByType": dict(violations),
        "videosBySourceGroup": dict(groups),
        "runtimeByMode": dict(modes),
        "failuresByReason": dict(failure_reasons),
        "storageByCategory": storage["categories"],
        "accuracyMetricsAvailable": False,
        "accuracyNotice": "Operational dashboard only. Labelled accuracy is unavailable.",
    }
    refreshed_signature = (
        tuple(sorted((record.job_id, record.status, record.updated_at.isoformat()) for record in records)),
        tuple(sorted((record.batch_id, record.status, record.updated_at.isoformat()) for record in batch_records)),
    )
    with _DASHBOARD_CACHE_LOCK:
        _DASHBOARD_CACHE[cache_slot] = {
            "signature": refreshed_signature,
            "created": time.monotonic(),
            "payload": payload,
        }
    return dict(payload)
