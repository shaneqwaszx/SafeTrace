"""Bounded resource-aware admission for the optional third local worker."""
from __future__ import annotations

import shutil
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Iterable

from src.config import SETTINGS


@dataclass(frozen=True)
class ResourceSnapshot:
    captured_at: float
    gpu_available: bool = False
    gpu_free_mb: float | None = None
    gpu_total_mb: float | None = None
    gpu_allocated_mb: float | None = None
    gpu_reserved_mb: float | None = None
    system_available_mb: float | None = None
    process_rss_mb: float | None = None
    free_disk_mb: float | None = None
    gpu_utilization_percent: float | None = None
    gpu_temperature_c: float | None = None
    probe_errors: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def probe_resources(runtime_root: Path | None = None) -> ResourceSnapshot:
    """Read admission metrics without making an optional probe a failure."""
    errors: list[str] = []
    gpu_available = False
    gpu_free_mb = gpu_total_mb = gpu_allocated_mb = gpu_reserved_mb = None
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
        if gpu_available:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            gpu_free_mb = free_bytes / (1024 * 1024)
            gpu_total_mb = total_bytes / (1024 * 1024)
            gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        errors.append(f"torch:{type(exc).__name__}")

    available_mb = rss_mb = None
    try:
        import psutil

        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception as exc:  # pragma: no cover - optional dependency
        errors.append(f"psutil:{type(exc).__name__}")

    free_disk_mb = None
    try:
        root = Path(runtime_root or getattr(SETTINGS, "job_store_dir", SETTINGS.data_dir))
        existing = next((candidate for candidate in (root, *root.parents) if candidate.exists()), Path.cwd())
        free_disk_mb = shutil.disk_usage(existing).free / (1024 * 1024)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"disk:{type(exc).__name__}")

    return ResourceSnapshot(
        captured_at=time.time(),
        gpu_available=gpu_available,
        gpu_free_mb=gpu_free_mb,
        gpu_total_mb=gpu_total_mb,
        gpu_allocated_mb=gpu_allocated_mb,
        gpu_reserved_mb=gpu_reserved_mb,
        system_available_mb=available_mb,
        process_rss_mb=rss_mb,
        free_disk_mb=free_disk_mb,
        probe_errors=tuple(errors),
    )


class AdaptiveWorkerManager:
    """Hysteretic controller for a baseline of two and at most three slots.

    It adjusts scheduler admission only. Existing GPU, MobileSAM, and VLM
    semaphores continue to own their stage-level limits.
    """

    def __init__(
        self,
        *,
        resource_probe: Callable[[], ResourceSnapshot] = probe_resources,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lock = threading.RLock()
        self._resource_probe = resource_probe
        self._clock = clock
        self._capacity = self.minimum
        self._busy_since: float | None = None
        self._idle_since: float | None = None
        self._last_transition = 0.0
        self._draining = False
        self._reason = "stable_baseline"
        self._snapshot: ResourceSnapshot | None = None
        self._events: Deque[dict[str, Any]] = deque(maxlen=100)
        self._recent_outcomes: Deque[dict[str, Any]] = deque(maxlen=20)

    @property
    def minimum(self) -> int:
        return max(1, min(2, int(getattr(SETTINGS, "adaptive_worker_min", 2) or 2)))

    @property
    def maximum(self) -> int:
        return max(self.minimum, min(3, int(getattr(SETTINGS, "adaptive_worker_max", 3) or 3)))

    @property
    def enabled(self) -> bool:
        return bool(getattr(SETTINGS, "adaptive_workers_enabled", False))

    def execution_ceiling(self) -> int:
        return max(self.minimum, self.maximum if self.enabled else self.minimum)

    def record_outcome(self, *, status: str, runtime_seconds: float | None = None, error_type: str | None = None) -> None:
        with self._lock:
            self._recent_outcomes.append(
                {
                    "status": status,
                    "runtimeSeconds": runtime_seconds,
                    "errorType": error_type,
                    "recordedAt": self._clock(),
                }
            )

    def _recent_health(self) -> tuple[float, float | None, bool]:
        if not self._recent_outcomes:
            return 0.0, None, False
        failures = [item for item in self._recent_outcomes if item["status"] not in {"completed", "cancelled"}]
        runtimes = [float(item["runtimeSeconds"]) for item in self._recent_outcomes if item.get("runtimeSeconds") is not None]
        oom = any("memory" in str(item.get("errorType") or "").lower() for item in self._recent_outcomes)
        return len(failures) / len(self._recent_outcomes), (sum(runtimes) / len(runtimes) if runtimes else None), oom

    @staticmethod
    def compatible_identity_count(identities: Iterable[tuple[Any, ...]]) -> int:
        values = list(identities)
        if not values:
            return 0
        first = values[0]
        return sum(1 for value in values if value == first)

    def evaluate(
        self,
        *,
        active_count: int,
        queued_count: int,
        compatible_queued_count: int,
        active_third_slot: bool,
        vlm_lane_in_use: int = 0,
        mobile_sam_lane_in_use: int = 0,
        force_probe: ResourceSnapshot | None = None,
    ) -> int:
        now = self._clock()
        snapshot = force_probe or self._resource_probe()
        failure_rate, mean_runtime, recent_oom = self._recent_health()
        reasons: list[str] = []
        queue_threshold = max(1, int(getattr(SETTINGS, "adaptive_scale_up_queue_depth", 3) or 3))
        sustain = max(0.0, float(getattr(SETTINGS, "adaptive_scale_up_sustain_seconds", 15.0) or 0.0))
        cooldown = max(0.0, float(getattr(SETTINGS, "adaptive_scale_cooldown_seconds", 60.0) or 0.0))
        idle_seconds = max(0.0, float(getattr(SETTINGS, "adaptive_scale_down_idle_seconds", 60.0) or 0.0))
        gpu_margin = float(getattr(SETTINGS, "adaptive_gpu_free_memory_margin_mb", 4096.0) or 4096.0)
        ram_margin = float(getattr(SETTINGS, "adaptive_system_free_memory_margin_mb", 4096.0) or 4096.0)
        disk_margin = max(
            float(getattr(SETTINGS, "min_free_disk_mb", 2048.0) or 2048.0),
            float(getattr(SETTINGS, "min_free_disk_gb", 2.0) or 2.0) * 1024,
        )
        max_error_rate = float(getattr(SETTINGS, "adaptive_max_recent_error_rate", 0.2) or 0.2)

        with self._lock:
            self._snapshot = snapshot
            if not self.enabled:
                reasons.append("adaptive_workers_disabled")
            if self.maximum <= self.minimum:
                reasons.append("maximum_locked_to_stable_baseline")
            if not bool(getattr(SETTINGS, "adaptive_third_worker_validated", False)):
                reasons.append("third_worker_not_device_validated")
            if queued_count < queue_threshold:
                reasons.append("queue_below_scale_up_threshold")
            if compatible_queued_count < 1:
                reasons.append("no_compatible_queued_job")
            if active_count < self.minimum:
                reasons.append("baseline_workers_not_busy")
            if not snapshot.gpu_available:
                reasons.append("blocked_by_gpu_unavailable")
            if snapshot.gpu_free_mb is None or snapshot.gpu_free_mb < gpu_margin:
                reasons.append("blocked_by_gpu_memory")
            if snapshot.system_available_mb is None or snapshot.system_available_mb < ram_margin:
                reasons.append("blocked_by_system_memory")
            if snapshot.free_disk_mb is None or snapshot.free_disk_mb < disk_margin:
                reasons.append("blocked_by_disk")
            if recent_oom:
                reasons.append("blocked_by_recent_oom")
            if failure_rate > max_error_rate:
                reasons.append("blocked_by_recent_error_rate")

            demand_ready = active_count >= self.minimum and queued_count >= queue_threshold and compatible_queued_count >= 1
            if demand_ready:
                self._busy_since = self._busy_since or now
                self._idle_since = None
            else:
                self._busy_since = None
                self._idle_since = self._idle_since or now
            if self._busy_since is None or now - self._busy_since < sustain:
                reasons.append("waiting_for_sustained_demand")
            if now - self._last_transition < cooldown:
                reasons.append("scale_cooldown_active")

            if self._capacity == self.minimum:
                if not reasons:
                    self._capacity = self.maximum
                    self._draining = False
                    self._last_transition = now
                    self._reason = "third_worker_admitted"
                    self._events.append({"event": "scale_up", "at": now, "capacity": self._capacity})
                else:
                    self._reason = reasons[0]
            else:
                pressure = any(
                    reason.startswith("blocked_by_") or reason in {"no_compatible_queued_job"}
                    for reason in reasons
                )
                idle_ready = self._idle_since is not None and now - self._idle_since >= idle_seconds
                if pressure or idle_ready:
                    if active_third_slot:
                        self._draining = True
                        self._reason = "third_worker_draining"
                    else:
                        self._capacity = self.minimum
                        self._draining = False
                        self._last_transition = now
                        self._reason = "pressure_scale_down" if pressure else "idle_scale_down"
                        self._events.append({"event": "scale_down", "at": now, "capacity": self._capacity})
                elif self._draining and not active_third_slot:
                    self._capacity = self.minimum
                    self._draining = False
                    self._last_transition = now
                    self._reason = "drain_complete"

            return self.minimum if self._draining else self._capacity

    def payload(self) -> dict[str, Any]:
        with self._lock:
            failure_rate, mean_runtime, recent_oom = self._recent_health()
            workers = []
            for worker_id in range(1, self.maximum + 1):
                state = "ready" if worker_id <= self._capacity else "stopped"
                if worker_id == 3 and self._draining:
                    state = "draining"
                workers.append({"workerId": f"analysis-{worker_id}", "slot": worker_id, "state": state})
            return {
                "enabled": self.enabled,
                "minimumWorkers": self.minimum,
                "maximumWorkers": self.maximum,
                "currentCapacity": self._capacity,
                "effectiveDispatchCapacity": self.minimum if self._draining else self._capacity,
                "thirdWorkerValidated": bool(getattr(SETTINGS, "adaptive_third_worker_validated", False)),
                "draining": self._draining,
                "admissionReason": self._reason,
                "workers": workers,
                "stageCaps": {
                    "gpuInference": int(getattr(SETTINGS, "gpu_inference_concurrency", 1) or 1),
                    "mobileSam": int(getattr(SETTINGS, "mobile_sam_concurrency", 1) or 1),
                    "vlm": int(getattr(SETTINGS, "vlm_concurrency", 1) or 1),
                },
                "resourceSnapshot": self._snapshot.to_payload() if self._snapshot else None,
                "recentErrorRate": round(failure_rate, 4),
                "recentMeanJobRuntimeSeconds": round(mean_runtime, 3) if mean_runtime is not None else None,
                "recentOom": recent_oom,
                "recentEvents": list(self._events),
            }
