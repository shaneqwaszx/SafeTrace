"""Single-instance, non-overlapping retention scheduler."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from src.config import SETTINGS

from .batches import BatchStore
from .jobs import JobStore
from .operations import CleanupInProgressError, apply_cleanup, cleanup_preview

logger = logging.getLogger("safetrace.api.retention")


class RetentionScheduler:
    def __init__(self, jobs: JobStore, batches: BatchStore) -> None:
        self.jobs = jobs
        self.batches = batches
        self._stop = threading.Event()
        self._run_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.run_count = 0
        self.overlap_skips = 0
        self.last_result: dict[str, Any] | None = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> bool:
        if self.running or not bool(SETTINGS.retention_scheduler_enabled):
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="safetrace-retention", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)

    def run_once(self, *, apply: bool = True) -> dict[str, Any]:
        if not self._run_lock.acquire(blocking=False):
            self.overlap_skips += 1
            return {"status": "skipped_overlap", "overlapSkips": self.overlap_skips}
        try:
            preview = cleanup_preview(self.jobs)
            result: dict[str, Any] = {
                "status": "previewed" if not apply else "completed",
                "startedAt": datetime.now(timezone.utc).isoformat(),
                "preview": preview,
            }
            if apply:
                try:
                    result["apply"] = apply_cleanup(self.jobs, preview)
                except CleanupInProgressError:
                    result["status"] = "skipped_cleanup_lock"
            self.run_count += 1
            result["runCount"] = self.run_count
            result["finishedAt"] = datetime.now(timezone.utc).isoformat()
            self.last_result = result
            return result
        finally:
            self._run_lock.release()

    def _loop(self) -> None:
        delay = max(0.0, float(SETTINGS.retention_startup_delay_seconds))
        if self._stop.wait(delay):
            return
        interval = max(1.0, float(SETTINGS.retention_interval_minutes) * 60.0)
        while not self._stop.is_set():
            try:
                self.run_once(apply=True)
            except Exception:
                logger.exception("SafeTrace retention scheduler run failed")
            if self._stop.wait(interval):
                return
