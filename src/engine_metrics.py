"""Backend engine metrics helpers for SafeTrace analysis results.

The helpers in this module are additive: they summarize already-collected
pipeline/job diagnostics without changing detector behavior, thresholds, or
violation logic.
"""
from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any, Dict, Iterable


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number < 0:
        return default
    return number


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(number, default)


def _round_seconds(value: Any) -> float:
    return round(_safe_float(value), 4)


def _sum_stage(stage_timings: Mapping[str, Any], names: Iterable[str]) -> float:
    return _round_seconds(sum(_safe_float(stage_timings.get(name)) for name in names))


def _severity_distribution(violations: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    for violation in violations:
        severity = str(violation.get("severity") or "unknown").lower()
        distribution[severity] = distribution.get(severity, 0) + 1
    return distribution


def _count_frame_detections(frames: Iterable[Mapping[str, Any]]) -> int:
    total = 0
    for frame in frames:
        evidence = dict(frame.get("technicalEvidence") or {})
        detections = evidence.get("detections")
        if isinstance(detections, list):
            total += len(detections)
    return total


def process_resource_snapshot() -> Dict[str, Any]:
    """Return best-effort process resource metrics without requiring psutil."""
    snapshot: Dict[str, Any] = {
        "pid": os.getpid(),
        "cpuProcessTimeSeconds": round(time.process_time(), 4),
        "currentRssMb": None,
        "peakRssMb": None,
    }
    try:  # pragma: no cover - optional dependency varies by environment
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        snapshot["currentRssMb"] = round(memory.rss / (1024 * 1024), 2)
        peak = getattr(memory, "peak_wset", None) or getattr(memory, "vms", None)
        if peak:
            snapshot["peakRssMb"] = round(float(peak) / (1024 * 1024), 2)
    except Exception:
        pass
    return snapshot


def build_engine_metrics(
    result: Mapping[str, Any],
    job_metrics: Mapping[str, Any] | None = None,
    *,
    resource_snapshot: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build an additive engine metrics block from normalized result data."""
    metrics = dict(job_metrics or {})
    technical = dict(result.get("technicalDetails") or {})
    component = dict(technical.get("componentDiagnostics") or metrics.get("componentDiagnostics") or {})
    processing = dict(technical.get("processingMetadata") or {})
    summary = dict(result.get("summary") or {})
    frames = list(result.get("frames") or [])
    violations = list(result.get("violations") or [])
    stage_timings = dict(component.get("stageTimings") or {})
    resources = dict(resource_snapshot or process_resource_snapshot())

    total_duration = (
        technical.get("pipelineWallClockSeconds")
        or metrics.get("totalWallClockSeconds")
        or metrics.get("elapsedWallClockSeconds")
        or 0.0
    )
    analyzed_frames = _safe_int(summary.get("framesAnalyzed"), len(frames))
    detections = _count_frame_detections(frames)

    return {
        "schemaVersion": 1,
        "totalDurationSeconds": _round_seconds(total_duration),
        "analysisDurationSeconds": _round_seconds(technical.get("pipelineWallClockSeconds") or total_duration),
        "stageDurations": {
            "frameExtraction": _sum_stage(stage_timings, ("frame_sampling", "safe_frame_sampling")),
            "preprocessing": _sum_stage(stage_timings, ("embedding_build", "semantic_search", "safe_frame_ranking")),
            "detectorLoad": _sum_stage(stage_timings, ("detector_load",)),
            "detectorInference": _sum_stage(
                stage_timings,
                ("detector_inference", "safe_ranking_detector_inference"),
            ),
            "ruleEvaluation": _sum_stage(stage_timings, ("rule_evaluation", "safe_ranking_rule_evaluation")),
            "aggregation": _round_seconds(technical.get("normalizationWallClockSeconds")),
            "evidenceGeneration": _sum_stage(
                stage_timings,
                ("annotation_write", "segmentation_refine", "selected_mobilesam_refine"),
            ),
            "reportGeneration": _round_seconds(technical.get("reportGenerationWallClockSeconds")),
            "resultWrite": _round_seconds(metrics.get("resultWriteSeconds")),
        },
        "counts": {
            "sampledFrames": _safe_int(processing.get("sampledFrameCount"), analyzed_frames),
            "analyzedFrames": analyzed_frames,
            "detections": detections,
            "violations": _safe_int(summary.get("uniqueViolationTypes"), len(violations)),
            "frameViolations": sum(len(frame.get("violations") or []) for frame in frames),
            "evidenceFrames": len([frame for frame in frames if frame.get("imageUrl") or frame.get("violations")]),
        },
        "resources": {
            **resources,
            "device": component.get("device"),
            "gpuAvailable": component.get("gpuAvailable"),
        },
        "optionalWorkers": {
            "mobileSamAttempted": bool(component.get("mobileSamWorkerAttempted") or component.get("mobileSamAttempted")),
            "mobileSamDurationSeconds": _round_seconds(component.get("mobileSamWorkerDurationSeconds")),
            "vlmAttempted": bool(component.get("lightweightVlmWorkerAttempted") or component.get("vlmAttempted")),
            "vlmDurationSeconds": _round_seconds(
                component.get("totalVlmDurationSeconds") or component.get("lightweightVlmWorkerDurationSeconds")
            ),
            "fallbackReasons": {
                "mobileSam": component.get("mobileSamFallbackReason"),
                "vlm": component.get("lightweightVlmFallbackReason") or component.get("lightweightVlmDisabledReason"),
                "vlmSkippedReasons": component.get("vlmSkippedReasons") or {},
            },
        },
        "correctness": {
            "status": result.get("status"),
            "resultSchemaKeys": sorted(str(key) for key in result.keys()),
            "violationNames": sorted(str(item.get("name") or item.get("id") or "") for item in violations),
            "severityDistribution": _severity_distribution(violations),
            "technicalJsonAvailable": bool(technical),
            "evidenceFrameCount": len(frames),
            "topEvidenceTimestamps": [
                str(frame.get("timestamp"))
                for frame in frames[:5]
                if frame.get("timestamp") is not None
            ],
        },
    }
