"""Real-model Phase U comparison of the stable two-slot and adaptive third slot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TERMINAL = {"completed", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derive_real_clips(sources: list[Path], *, output_dir: Path, count: int) -> list[Path]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    clip_index = 0
    for source in sources:
        capture = cv2.VideoCapture(str(source))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 15.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not capture.isOpened() or frame_count < 2:
            capture.release()
            continue
        segment_frames = max(1, min(frame_count, int(fps * 4)))
        starts = sorted({0, max(0, (frame_count - segment_frames) // 2), max(0, frame_count - segment_frames)})
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
        for start in starts:
            if len(clips) >= count:
                break
            destination = output_dir / f"real-clip-{clip_index:02d}-{source.stem}.mp4"
            clip_index += 1
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            writer = cv2.VideoWriter(
                str(destination),
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(1.0, fps),
                (width, height),
            )
            written = 0
            while written < segment_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
                written += 1
            writer.release()
            if written and destination.is_file() and destination.stat().st_size:
                clips.append(destination)
        capture.release()
        if len(clips) >= count:
            break
    return clips


def discover_videos(limit: int, *, work_dir: Path) -> list[Path]:
    roots = [
        REPO_ROOT / "data" / "api_jobs",
        REPO_ROOT / "evaluation",
        REPO_ROOT / "tmp_backend_benchmark",
        REPO_ROOT / ".ai-pipeline" / "005_tasks" / "phase_r_runtime_parallel_scheduler",
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(path for path in root.rglob("*.mp4") if path.is_file() and path.stat().st_size > 0)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in sorted(candidates, key=lambda item: (item.stat().st_size, str(item))):
        digest = sha256(path)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(path)
        if len(unique) >= limit:
            break
    if len(unique) < 3:
        raise RuntimeError("At least three real local MP4 inputs are required for the adaptive smoke.")
    if len(unique) < limit:
        for clip in _derive_real_clips(unique, output_dir=work_dir / "derived_real_clips", count=limit * 2):
            digest = sha256(clip)
            if digest in seen:
                continue
            seen.add(digest)
            unique.append(clip)
            if len(unique) >= limit:
                break
    if len(unique) < limit:
        raise RuntimeError(f"Could not materialize {limit} distinct real-video inputs; found {len(unique)}.")
    return unique[:limit]


def percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))]


def resource_monitor(stop: threading.Event, samples: list[dict[str, Any]]) -> None:
    while not stop.wait(0.25):
        row: dict[str, Any] = {"capturedAt": utc_now()}
        try:
            import psutil

            row["rssMb"] = psutil.Process().memory_info().rss / (1024 * 1024)
            row["availableRamMb"] = psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                row.update(
                    gpuFreeMb=free / (1024 * 1024),
                    gpuTotalMb=total / (1024 * 1024),
                    gpuAllocatedMb=torch.cuda.memory_allocated() / (1024 * 1024),
                    gpuReservedMb=torch.cuda.memory_reserved() / (1024 * 1024),
                )
        except Exception:
            pass
        samples.append(row)


def run_case(*, videos: list[Path], work_dir: Path, adaptive_third: bool) -> dict[str, Any]:
    from src.api.jobs import AnalysisSettings, JobStore, schedule_analysis_jobs, scheduler_status_payload, wait_for_scheduled_jobs
    from src.config import SETTINGS

    SETTINGS.runtime_profile = "local_full"
    SETTINGS.device = "cuda"
    SETTINGS.analysis_concurrency = 2
    SETTINGS.job_concurrency = 2
    SETTINGS.worker_concurrency = 2
    # The adaptive run explicitly evaluates a third real GPU lane. This is
    # process-local test configuration; production remains locked to two until
    # the resulting stability and throughput evidence is reviewed.
    SETTINGS.gpu_inference_concurrency = 3 if adaptive_third else 2
    SETTINGS.gpu_detector_concurrency = 3 if adaptive_third else 2
    SETTINGS.per_batch_concurrency = 3 if adaptive_third else 2
    SETTINGS.batch_max_active_jobs = 3 if adaptive_third else 2
    SETTINGS.mobile_sam_concurrency = 1
    SETTINGS.vlm_concurrency = 1
    SETTINGS.adaptive_workers_enabled = adaptive_third
    SETTINGS.adaptive_worker_min = 2
    SETTINGS.adaptive_worker_max = 3
    SETTINGS.adaptive_third_worker_validated = adaptive_third
    SETTINGS.adaptive_scale_up_queue_depth = 3
    SETTINGS.adaptive_scale_up_sustain_seconds = 0.5
    SETTINGS.adaptive_scale_down_idle_seconds = 2.0
    SETTINGS.adaptive_scale_cooldown_seconds = 0.0
    SETTINGS.adaptive_gpu_free_memory_margin_mb = 4096.0
    SETTINGS.adaptive_system_free_memory_margin_mb = 4096.0

    case_name = "adaptive3" if adaptive_third else "baseline2"
    store = JobStore(work_dir / case_name / "data" / "api_jobs")
    settings = AnalysisSettings(
        fps=1.0,
        top_k=2,
        enable_vlm=False,
        vlm_enabled=False,
        vlm_profile="rule_based",
        device="cuda",
        safe_mode=True,
        use_case_profile={"profileId": "seatbelt_compliance", "label": "Seatbelt Compliance"},
        review_mode="fast_local",
    )
    records = []
    for index, source in enumerate(videos, start=1):
        records.append(
            store.create_job(
                filename=f"phase-u-{index:02d}-{source.name}",
                content=source.read_bytes(),
                query="driver seatbelt compliance",
                settings=settings,
                source_metadata={
                    "sourceRelativePath": f"phase-u/{index:02d}-{source.name}",
                    "sourceGroupPath": "phase-u",
                    "batchId": f"phase-u-{'adaptive3' if adaptive_third else 'baseline2'}",
                    "batchEnqueueSequence": 1,
                    "childSequence": index,
                },
            )
        )

    samples: list[dict[str, Any]] = []
    stop = threading.Event()
    monitor = threading.Thread(target=resource_monitor, args=(stop, samples), daemon=True)
    monitor.start()
    started = time.perf_counter()
    schedule_analysis_jobs(store, [record.job_id for record in records])
    observations: list[dict[str, Any]] = []
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        snapshot = scheduler_status_payload()
        states = {record.job_id: store.require(record.job_id).status for record in records}
        observations.append(
            {
                "observedAt": utc_now(),
                "states": states,
                "activeWorkers": snapshot.get("activeWorkers"),
                "adaptiveWorkers": snapshot.get("adaptiveWorkers"),
            }
        )
        if all(status in TERMINAL for status in states.values()):
            break
        time.sleep(0.4)
    wait_for_scheduled_jobs(store, [record.job_id for record in records], timeout_seconds=30)
    duration = time.perf_counter() - started

    post_run_adaptive = scheduler_status_payload().get("adaptiveWorkers") or {}
    if adaptive_third:
        scale_down_deadline = time.monotonic() + 8.0
        while time.monotonic() < scale_down_deadline:
            post_run_adaptive = scheduler_status_payload().get("adaptiveWorkers") or {}
            if (
                int(post_run_adaptive.get("currentCapacity") or 0) == 2
                and not bool(post_run_adaptive.get("draining"))
            ):
                break
            time.sleep(0.5)
    stop.set()
    monitor.join(timeout=2)

    jobs = [store.require(record.job_id) for record in records]
    runtimes = [float(job.timing_payload().get("analysisRuntimeSeconds") or 0.0) for job in jobs]
    waits = [float(job.timing_payload().get("queueWaitSeconds") or 0.0) for job in jobs]
    ownership_errors: list[str] = []
    for job in jobs:
        result = job.result or {}
        expected = str(job.source_metadata.get("sourceRelativePath"))
        if str(result.get("sourceRelativePath") or result.get("sourceMetadata", {}).get("sourceRelativePath")) != expected:
            ownership_errors.append(job.job_id)
        for frame in result.get("frames") or []:
            if frame.get("jobId") not in {None, job.job_id}:
                ownership_errors.append(f"{job.job_id}:frame:{frame.get('id')}")

    temp_files = [str(path) for path in store.root_dir.rglob("manifest.json.*.tmp")]
    worker_counts = [len(item.get("activeWorkers") or []) for item in observations]
    return {
        "mode": "adaptive_2_to_3" if adaptive_third else "fixed_2",
        "jobCount": len(jobs),
        "totalRuntimeSeconds": round(duration, 3),
        "meanJobRuntimeSeconds": round(statistics.mean(runtimes), 3) if runtimes else None,
        "p95JobRuntimeSeconds": round(percentile95(runtimes) or 0.0, 3),
        "meanQueueWaitSeconds": round(statistics.mean(waits), 3) if waits else None,
        "peakActiveWorkers": max(worker_counts, default=0),
        "thirdWorkerObserved": any(count >= 3 for count in worker_counts),
        "postRunAdaptiveWorkers": post_run_adaptive,
        "scaledDownToBaseline": (
            not adaptive_third
            or (
                int(post_run_adaptive.get("currentCapacity") or 0) == 2
                and not bool(post_run_adaptive.get("draining"))
            )
        ),
        "gpuInferenceConcurrency": SETTINGS.gpu_inference_concurrency,
        "statuses": {job.job_id: job.status for job in jobs},
        "retryCount": sum(job.retry_count for job in jobs),
        "resultCacheHitCount": sum(bool(job.metrics.get("resultCacheHit")) for job in jobs),
        "ownershipErrors": sorted(set(ownership_errors)),
        "manifestTempFiles": temp_files,
        "gpuAllocatedPeakMb": max((float(item.get("gpuAllocatedMb") or 0.0) for item in samples), default=0.0),
        "gpuReservedPeakMb": max((float(item.get("gpuReservedMb") or 0.0) for item in samples), default=0.0),
        "gpuFreeMinimumMb": min((float(item["gpuFreeMb"]) for item in samples if item.get("gpuFreeMb") is not None), default=None),
        "rssPeakMb": max((float(item.get("rssMb") or 0.0) for item in samples), default=0.0),
        "observations": observations,
        "resourceSamples": samples,
    }


def markdown(report: dict[str, Any]) -> str:
    comparison = report.get("comparison") or {}
    lines = ["# Phase U Adaptive Worker Smoke", "", f"Generated: {report['generatedAt']}", ""]
    for result in report.get("runs") or []:
        lines.extend(
            [
                f"## {result['mode']}",
                f"- Total runtime: {result['totalRuntimeSeconds']}s",
                f"- Mean / p95 job runtime: {result['meanJobRuntimeSeconds']}s / {result['p95JobRuntimeSeconds']}s",
                f"- Peak workers: {result['peakActiveWorkers']}",
                f"- GPU free minimum: {result['gpuFreeMinimumMb']} MB",
                f"- RSS peak: {result['rssPeakMb']} MB",
                f"- Ownership errors: {len(result['ownershipErrors'])}",
                "",
            ]
        )
    lines.extend(["## Decision", f"- Throughput improvement: {comparison.get('throughputImprovementPercent')}%", f"- Third worker accepted: {comparison.get('thirdWorkerAccepted')}", f"- Reason: {comparison.get('reason')}"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--video-count", type=int, default=6)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    videos = discover_videos(max(5, min(6, args.video_count)), work_dir=args.work_dir)
    baseline = run_case(videos=videos, work_dir=args.work_dir, adaptive_third=False)
    adaptive = run_case(videos=videos, work_dir=args.work_dir, adaptive_third=True)
    improvement = ((baseline["totalRuntimeSeconds"] - adaptive["totalRuntimeSeconds"]) / baseline["totalRuntimeSeconds"] * 100) if baseline["totalRuntimeSeconds"] else 0.0
    latency_delta = ((adaptive["meanJobRuntimeSeconds"] - baseline["meanJobRuntimeSeconds"]) / baseline["meanJobRuntimeSeconds"] * 100) if baseline["meanJobRuntimeSeconds"] else 0.0
    stable = (
        adaptive["thirdWorkerObserved"]
        and adaptive["scaledDownToBaseline"]
        and set(adaptive["statuses"].values()) == {"completed"}
        and not adaptive["ownershipErrors"]
        and not adaptive["manifestTempFiles"]
        and adaptive["retryCount"] == 0
        and adaptive["resultCacheHitCount"] == 0
    )
    beneficial = improvement > 0.0 and latency_delta <= 50.0
    report = {
        "generatedAt": utc_now(),
        "inputs": [{"path": str(path), "sha256": sha256(path), "sizeBytes": path.stat().st_size} for path in videos],
        "runs": [baseline, adaptive],
        "comparison": {
            "throughputImprovementPercent": round(improvement, 3),
            "meanLatencyChangePercent": round(latency_delta, 3),
            "stable": stable,
            "beneficial": beneficial,
            "thirdWorkerAccepted": stable and beneficial,
            "reason": "stable_and_beneficial" if stable and beneficial else "third_worker_not_admitted_on_this_device_configuration",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = args.markdown_output or args.output.with_suffix(".md")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["comparison"], indent=2))
    return 0 if stable else 2


if __name__ == "__main__":
    raise SystemExit(main())
