"""Reproducible SafeTrace profile benchmark using reviewed human labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.aggregation import timestamp_to_seconds
from src.api.batches import BatchStore
from src.api.jobs import JobStore
from src.api.server import create_app
from src.config import SETTINGS

RANDOM_SEED = 20260715
LABEL_STATUSES = {"positive", "negative", "uncertain", "unlabelled"}
APPROVED_STATUSES = {"approved"}
SUPPORTED_PROFILES = {
    "seatbelt_compliance",
    "helmet_ppe",
    "phone_use",
    "hands_on_wheel",
    "general_safety",
}
PROFILE_SUPPORT = {
    "seatbelt_compliance": "partially_supported",
    "helmet_ppe": "requires_custom_detector",
    "phone_use": "requires_custom_detector",
    "hands_on_wheel": "requires_custom_detector",
    "general_safety": "partially_supported",
}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _normalize_sample(item: dict[str, Any]) -> dict[str, Any]:
    sample = dict(item)
    expected = sample.get("expectedFinding")
    if expected is None:
        values = list(sample.get("expectedFindings") or [])
        expected = values[0] if values else "no_violation"
    sample["expectedFinding"] = str(expected).strip().lower()
    sample["profile"] = str(sample.get("profile") or "general_safety").strip().lower()
    sample["labelStatus"] = str(sample.get("labelStatus") or "unlabelled").strip().lower()
    sample["annotationStatus"] = str(sample.get("annotationStatus") or "pending").strip().lower()
    sample["reviewerStatus"] = str(sample.get("reviewerStatus") or "pending").strip().lower()
    ranges = list(sample.get("timestampRanges") or [])
    if ranges and sample.get("startSeconds") is None:
        sample["startSeconds"] = ranges[0].get("startSeconds")
        sample["endSeconds"] = ranges[0].get("endSeconds")
    sample["startSeconds"] = _parse_optional_float(sample.get("startSeconds"))
    sample["endSeconds"] = _parse_optional_float(sample.get("endSeconds"))
    confidence = sample.get("humanLabelConfidence")
    sample["humanLabelConfidence"] = float(confidence) if confidence not in (None, "") else 0.0
    return sample


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            samples = [_normalize_sample(dict(row)) for row in csv.DictReader(handle)]
        metadata = {"schemaVersion": 1, "datasetId": path.stem, "timestampToleranceSeconds": 2.0}
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_samples = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(raw_samples, list):
            raise ValueError("Labelled manifest must contain a samples list.")
        samples = [_normalize_sample(dict(item)) for item in raw_samples]
        metadata = dict(payload) if isinstance(payload, dict) else {"schemaVersion": 1, "datasetId": path.stem}
        metadata.pop("samples", None)
    validate_manifest(samples)
    return metadata, samples


def validate_manifest(samples: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id") or "").strip()
        if not sample_id:
            raise ValueError(f"Manifest row {index} is missing id.")
        if sample_id in seen:
            raise ValueError(f"Duplicate manifest id: {sample_id}")
        seen.add(sample_id)
        if not str(sample.get("mediaPath") or "").strip():
            raise ValueError(f"Manifest row {sample_id} is missing mediaPath.")
        if sample.get("profile") not in SUPPORTED_PROFILES:
            raise ValueError(f"Unsupported profile in {sample_id}: {sample.get('profile')}")
        if sample.get("labelStatus") not in LABEL_STATUSES:
            raise ValueError(f"Invalid labelStatus in {sample_id}: {sample.get('labelStatus')}")
        if not 0.0 <= float(sample.get("humanLabelConfidence") or 0.0) <= 1.0:
            raise ValueError(f"humanLabelConfidence must be between 0 and 1 in {sample_id}.")
        start, end = sample.get("startSeconds"), sample.get("endSeconds")
        if start is not None and end is not None and float(end) < float(start):
            raise ValueError(f"endSeconds precedes startSeconds in {sample_id}.")


def is_accuracy_eligible(sample: dict[str, Any]) -> bool:
    return (
        sample.get("labelStatus") in {"positive", "negative"}
        and sample.get("annotationStatus") in APPROVED_STATUSES
        and sample.get("reviewerStatus") in {"agrees", "resolved"}
    )


def _event_window_score(
    expected_ranges: list[dict[str, Any]],
    events: list[dict[str, Any]],
    tolerance: float = 2.0,
) -> dict[str, int]:
    matched_events: set[int] = set()
    true_positive = 0
    for expected in expected_ranges:
        finding = str(expected.get("finding") or "").strip().lower()
        start = float(expected.get("startSeconds") or 0.0) - tolerance
        end = float(expected.get("endSeconds") or start) + tolerance
        for index, event in enumerate(events):
            event_type = str(event.get("type") or "").strip().lower()
            event_start = float(event.get("startSecond") or timestamp_to_seconds(event.get("startTimestamp")))
            event_end = float(event.get("endSecond") or timestamp_to_seconds(event.get("endTimestamp")) or event_start)
            if index not in matched_events and event_type == finding and event_end >= start and event_start <= end:
                matched_events.add(index)
                true_positive += 1
                break
    return {
        "eventTruePositives": true_positive,
        "eventFalseNegatives": max(len(expected_ranges) - true_positive, 0),
        "eventFalsePositives": max(len(events) - len(matched_events), 0),
    }


def _predicted_labels(result: dict[str, Any]) -> set[str]:
    return {
        str(item.get("id") or item.get("name") or "").strip().lower()
        for item in result.get("violations") or []
        if item.get("id") or item.get("name")
    }


def _expected_ranges(sample: dict[str, Any]) -> list[dict[str, Any]]:
    if not is_accuracy_eligible(sample) or sample.get("labelStatus") != "positive":
        return []
    finding = str(sample.get("expectedFinding") or "")
    if finding in {"no_violation", "insufficient_visibility"}:
        return []
    return [{
        "finding": finding,
        "startSeconds": sample.get("startSeconds") or 0.0,
        "endSeconds": sample.get("endSeconds") or sample.get("startSeconds") or 0.0,
    }]


def _system_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpuCount": os.cpu_count(),
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        payload.update({"systemMemoryTotalMb": round(memory.total / 1048576, 1), "systemMemoryAvailableMb": round(memory.available / 1048576, 1)})
    except (ImportError, OSError):
        pass
    try:
        import torch

        payload.update({
            "torch": torch.__version__,
            "cudaAvailable": bool(torch.cuda.is_available()),
            "cudaVersion": torch.version.cuda,
            "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except ImportError:
        payload["torch"] = None
    return payload


def _configuration_snapshot(mode: str, device: str, concurrency: int) -> dict[str, Any]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unknown"
    checkpoints = []
    for name, path in (
        ("configuredDetector", Path(SETTINGS.yolo_checkpoint)),
        ("fallbackDetector", Path(SETTINGS.yolo_fallback_checkpoint)),
        ("mobileSam", Path(SETTINGS.mobile_sam_checkpoint)),
    ):
        checkpoints.append({"name": name, "path": str(path), "exists": path.is_file(), "sizeBytes": path.stat().st_size if path.is_file() else None, "sha256": _sha256(path)})
    return {
        "head": head,
        "seed": RANDOM_SEED,
        "mode": mode,
        "device": device,
        "concurrency": concurrency,
        "fastLocal": {"fps": SETTINGS.frame_fps, "maxFrames": SETTINGS.max_frames, "topK": SETTINGS.top_k},
        "comprehensive": {
            "fps": SETTINGS.comprehensive_review_fps,
            "maxFrames": SETTINGS.comprehensive_review_max_frames,
            "topK": SETTINGS.comprehensive_review_top_k,
            "segmentationFrameLimit": SETTINGS.comprehensive_max_segmentation_frames,
            "vlmFrameLimit": SETTINGS.comprehensive_max_vlm_frames,
        },
        "thresholds": {"yoloConfidence": SETTINGS.yolo_conf_threshold, "yoloIou": SETTINGS.yolo_iou_threshold},
        "checkpoints": checkpoints,
        "hardware": _system_snapshot(),
    }


def _runtime_resource_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    try:
        import psutil

        process = psutil.Process()
        snapshot["processRssMb"] = round(process.memory_info().rss / 1048576, 2)
        snapshot["systemAvailableMb"] = round(psutil.virtual_memory().available / 1048576, 2)
    except (ImportError, OSError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            snapshot["cudaAllocatedMb"] = round(torch.cuda.memory_allocated() / 1048576, 2)
            snapshot["cudaReservedMb"] = round(torch.cuda.memory_reserved() / 1048576, 2)
    except ImportError:
        pass
    return snapshot


def _run_sample(sample: dict[str, Any], *, mode: str, device: str, runtime_root: Path, tolerance: float) -> dict[str, Any]:
    sample_id = str(sample["id"])
    media_path = Path(str(sample["mediaPath"]))
    if not media_path.is_absolute():
        media_path = (ROOT / media_path).resolve()
    started = time.perf_counter()
    started_epoch = time.time()
    resources_before = _runtime_resource_snapshot()
    item_root = runtime_root / sample_id
    with TestClient(create_app(JobStore(item_root / "jobs"), BatchStore(item_root / "batches"))) as client:
        with media_path.open("rb") as handle:
            response = client.post(
                "/api/analyze",
                files={"file": (media_path.name, handle, "application/octet-stream")},
                data={
                    "query": sample.get("query") or "general safety violations",
                    "device": device,
                    "reviewMode": mode,
                    "useCaseProfile": json.dumps({"profileId": sample.get("profile") or "general_safety"}),
                    "enableVlm": "false",
                },
            )
        response.raise_for_status()
        job_id = response.json()["jobId"]
        status = client.get(f"/api/jobs/{job_id}").json()
        while status["status"] not in {"completed", "failed", "cancelled"}:
            time.sleep(0.05)
            status = client.get(f"/api/jobs/{job_id}").json()
        result = client.get(f"/api/jobs/{job_id}/result").json() if status["status"] == "completed" else {}
    processing = dict((result.get("technicalDetails") or {}).get("processingMetadata") or {})
    engine = dict(result.get("engineMetrics") or {})
    engine_counts = dict(engine.get("counts") or {})
    engine_stages = dict(engine.get("stageDurations") or {})
    engine_resources = dict(engine.get("resources") or {})
    optional_workers = dict(engine.get("optionalWorkers") or {})
    diagnostics = dict((result.get("technicalDetails") or {}).get("componentDiagnostics") or status.get("componentDiagnostics") or {})
    expected_ranges = _expected_ranges(sample)
    predicted = _predicted_labels(result)
    expected_finding = str(sample.get("expectedFinding") or "")
    eligible = is_accuracy_eligible(sample)
    expected_labels = {expected_finding} if eligible and sample.get("labelStatus") == "positive" and expected_finding not in {"no_violation", "insufficient_visibility"} else set()
    finished_epoch = time.time()
    return {
        "sampleId": sample_id,
        "mediaPath": str(media_path),
        "sourceRelativePath": sample.get("sourceRelativePath"),
        "profile": sample.get("profile"),
        "mode": mode,
        "labelStatus": sample.get("labelStatus"),
        "annotationStatus": sample.get("annotationStatus"),
        "accuracyEligible": eligible,
        "expectedFindings": sorted(expected_labels),
        "predictedFindings": sorted(predicted),
        "status": status["status"],
        "jobId": job_id,
        "startedAtEpoch": started_epoch,
        "finishedAtEpoch": finished_epoch,
        "runtimeSeconds": round(time.perf_counter() - started, 4),
        "queueWaitSeconds": status.get("queueWaitSeconds"),
        "analysisRuntimeSeconds": status.get("analysisRuntimeSeconds"),
        "framesScreened": processing.get("sampledFrameCount") or engine_counts.get("sampledFrames"),
        "framesRanked": processing.get("processingWindowCount") or engine_counts.get("sampledFrames"),
        "framesSelected": engine_counts.get("analyzedFrames"),
        "framesSegmented": diagnostics.get("mobileSamFramesAttempted") or int(bool(optional_workers.get("mobileSamAttempted"))),
        "detectorDevice": diagnostics.get("actualDetectorDevice") or diagnostics.get("device"),
        "mobileSamDevice": diagnostics.get("actualMobileSamDevice"),
        "gpuName": diagnostics.get("gpuName"),
        "detectorLoadSeconds": engine_stages.get("detectorLoad"),
        "detectorInferenceSeconds": engine_stages.get("detectorInference"),
        "peakRssMb": engine_resources.get("peakRssMb"),
        "retryCount": status.get("retryCount") or 0,
        "resourcesBefore": resources_before,
        "resourcesAfter": _runtime_resource_snapshot(),
        "error": status.get("error"),
        "engineMetrics": engine,
        **_event_window_score(expected_ranges, list(result.get("events") or []), tolerance=tolerance),
    }


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for profile in sorted({str(row.get("profile")) for row in rows}):
        eligible = [row for row in rows if row.get("profile") == profile and row.get("accuracyEligible") and row.get("status") == "completed"]
        tp = fp = fn = event_tp = event_fp = event_fn = 0
        for row in eligible:
            expected = set(row.get("expectedFindings") or [])
            predicted = set(row.get("predictedFindings") or [])
            tp += len(expected & predicted)
            fp += len(predicted - expected)
            fn += len(expected - predicted)
            event_tp += int(row.get("eventTruePositives") or 0)
            event_fp += int(row.get("eventFalsePositives") or 0)
            event_fn += int(row.get("eventFalseNegatives") or 0)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        runtimes = sorted(float(row.get("runtimeSeconds") or 0.0) for row in rows if row.get("profile") == profile and row.get("status") == "completed")
        output[profile] = {
            "approvedItems": len(eligible),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "truePositives": tp,
            "falsePositives": fp,
            "falseNegatives": fn,
            "eventRecall": event_tp / (event_tp + event_fn) if event_tp + event_fn else None,
            "eventFalsePositives": event_fp,
            "eventFalseNegatives": event_fn,
            "p50RuntimeSeconds": statistics.median(runtimes) if runtimes else None,
            "p95RuntimeSeconds": runtimes[min(len(runtimes) - 1, max(0, math.ceil(len(runtimes) * 0.95) - 1))] if runtimes else None,
        }
    return output


def no_regression_gate(baseline: dict[str, Any], candidate: dict[str, Any], tolerance: float = 0.02) -> dict[str, Any]:
    checks = []
    for profile in ("seatbelt_compliance", "helmet_ppe"):
        before = baseline.get(profile, {}).get("f1")
        after = candidate.get(profile, {}).get("f1")
        before_fp = baseline.get(profile, {}).get("falsePositives")
        after_fp = candidate.get(profile, {}).get("falsePositives")
        accuracy_ok = before is not None and after is not None and after + tolerance >= before
        fp_ok = True if before_fp is None or after_fp is None else after_fp <= before_fp
        checks.append({
            "profile": profile,
            "baselineF1": before,
            "candidateF1": after,
            "baselineFalsePositives": before_fp,
            "candidateFalsePositives": after_fp,
            "passed": bool(accuracy_ok and fp_ok),
        })
    return {"passed": bool(checks) and all(item["passed"] for item in checks), "checks": checks, "tolerance": tolerance}


def _write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    rows = list(report["rows"])
    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    fields = [
        "sampleId", "profile", "mode", "labelStatus", "annotationStatus", "accuracyEligible", "status",
        "runtimeSeconds", "queueWaitSeconds", "analysisRuntimeSeconds", "framesScreened", "framesRanked",
        "framesSelected", "framesSegmented", "detectorDevice", "mobileSamDevice", "retryCount",
        "startedAtEpoch", "finishedAtEpoch", "detectorLoadSeconds", "detectorInferenceSeconds", "peakRssMb", "error",
    ]
    with (output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    metrics = report["metrics"]
    lines = [
        "# SafeTrace Profile Benchmark",
        "",
        f"- Run ID: `{report['runId']}`",
        f"- Mode: `{report['configuration']['mode']}`",
        f"- Concurrency: `{report['configuration']['concurrency']}`",
        f"- Completed items: `{sum(row.get('status') == 'completed' for row in rows)}` / `{len(rows)}`",
        f"- Approved accuracy items: `{report['approvedAccuracyItemCount']}`",
        f"- Accuracy metrics available: `{report['accuracyMetricsAvailable']}`",
        f"- Accuracy gate passed: `{report['accuracyGatePassed']}`",
        "",
        "Accuracy is unavailable when reviewed labels are absent; runtime output must not be presented as correctness.",
        "",
        "## Per-profile metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2),
        "```",
    ]
    (output_dir / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    random.seed(RANDOM_SEED)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("fast_local", "comprehensive"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--concurrency", choices=(1, 2), type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    metadata, samples = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items_dir = args.output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    tolerance = float(metadata.get("timestampToleranceSeconds") or 2.0)
    configuration = _configuration_snapshot(args.mode, args.device, args.concurrency)
    run_id = f"{args.mode}_{args.device}_c{args.concurrency}_{int(time.time())}"
    rows_by_id: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for sample in samples:
        item_path = items_dir / f"{args.mode}_{sample['id']}.json"
        if args.resume and item_path.is_file():
            try:
                row = json.loads(item_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pending.append(sample)
            else:
                rows_by_id[str(sample["id"])] = row
            continue
        pending.append(sample)

    def evaluate(sample: dict[str, Any]) -> dict[str, Any]:
        media_path = Path(str(sample.get("mediaPath") or ""))
        if not media_path.is_absolute():
            media_path = ROOT / media_path
        if args.dry_run or not media_path.is_file():
            return {
                "sampleId": sample["id"],
                "profile": sample["profile"],
                "mode": args.mode,
                "labelStatus": sample["labelStatus"],
                "annotationStatus": sample["annotationStatus"],
                "accuracyEligible": is_accuracy_eligible(sample),
                "status": "not_evaluated",
                "reason": "dry_run" if args.dry_run else "missing_media",
            }
        return _run_sample(sample, mode=args.mode, device=args.device, runtime_root=args.output_dir / "runtime", tolerance=tolerance)

    with ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix="safetrace-benchmark") as executor:
        futures = {executor.submit(evaluate, sample): sample for sample in pending}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # preserve failed item for resume/audit
                row = {
                    "sampleId": sample["id"], "profile": sample["profile"], "mode": args.mode,
                    "labelStatus": sample["labelStatus"], "annotationStatus": sample["annotationStatus"],
                    "accuracyEligible": is_accuracy_eligible(sample), "status": "failed",
                    "errorType": type(exc).__name__, "error": str(exc),
                }
            rows_by_id[str(sample["id"])] = row
            (items_dir / f"{args.mode}_{sample['id']}.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")

    rows = [rows_by_id[str(sample["id"])] for sample in samples]
    metrics = score_rows(rows)
    approved_count = sum(bool(row.get("accuracyEligible")) for row in rows)
    all_approved_completed = approved_count > 0 and all(
        row.get("status") == "completed" for row in rows if row.get("accuracyEligible")
    )
    required_gate_profiles = ("seatbelt_compliance", "helmet_ppe")
    accuracy_gate_passed = all(
        metrics.get(profile, {}).get("approvedItems", 0) >= 40
        and metrics.get(profile, {}).get("f1") is not None
        for profile in required_gate_profiles
    )
    completed_runtimes = sorted(
        float(row.get("runtimeSeconds") or 0.0)
        for row in rows
        if row.get("status") == "completed"
    )
    report = {
        "schemaVersion": 2,
        "runId": run_id,
        "mode": args.mode,
        "device": args.device,
        "concurrency": args.concurrency,
        "manifest": str(args.manifest),
        "manifestSha256": _sha256(args.manifest),
        "dataset": metadata,
        "configuration": configuration,
        "resumeRequested": args.resume,
        "dryRun": args.dry_run,
        "itemCount": len(rows),
        "approvedAccuracyItemCount": approved_count,
        "accuracyMetricsAvailable": all_approved_completed,
        "accuracyGatePassed": bool(all_approved_completed and accuracy_gate_passed),
        "metrics": metrics,
        "summary": {
            "completedItems": sum(row.get("status") == "completed" for row in rows),
            "failedItems": sum(row.get("status") == "failed" for row in rows),
            "meanRuntimeSeconds": statistics.mean(completed_runtimes) if completed_runtimes else None,
            "p95RuntimeSeconds": completed_runtimes[min(len(completed_runtimes) - 1, max(0, math.ceil(len(completed_runtimes) * 0.95) - 1))] if completed_runtimes else None,
            "framesScreened": sum(int(row.get("framesScreened") or 0) for row in rows),
        },
        "rows": rows,
    }
    _write_outputs(args.output_dir, report)
    print(json.dumps({
        "runId": run_id,
        "output": str(args.output_dir),
        "items": len(rows),
        "completed": sum(row.get("status") == "completed" for row in rows),
        "accuracyMetricsAvailable": report["accuracyMetricsAvailable"],
        "accuracyGatePassed": report["accuracyGatePassed"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
