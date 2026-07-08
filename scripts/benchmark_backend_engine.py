"""Run local SafeTrace backend engine benchmarks.

This script is local-only and writes generated reports under
tmp_backend_benchmark/ by default. It does not download models or mutate
model/checkpoint files.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.jobs import media_type_for, run_pipeline  # noqa: E402
from src.api.normalization import normalize_pipeline_results  # noqa: E402
from src.config import SETTINGS  # noqa: E402
from src.engine_metrics import build_engine_metrics, process_resource_snapshot  # noqa: E402

DEFAULT_OUTPUT = ROOT / "tmp_backend_benchmark"
MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _media_inputs(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES:
        return [path]
    if path.is_dir():
        return [item for item in sorted(path.rglob("*")) if item.is_file() and item.suffix.lower() in MEDIA_SUFFIXES]
    return []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, runs: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "runIndex",
        "inputPath",
        "mode",
        "status",
        "totalDurationSeconds",
        "analysisDurationSeconds",
        "detectorInferenceSeconds",
        "sampledFrames",
        "analyzedFrames",
        "detections",
        "violations",
        "evidenceFrames",
        "violationNames",
        "errorType",
        "errorMessage",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            engine = dict(run.get("engineMetrics") or {})
            stages = dict(engine.get("stageDurations") or {})
            counts = dict(engine.get("counts") or {})
            correctness = dict(engine.get("correctness") or {})
            writer.writerow(
                {
                    "runIndex": run.get("runIndex"),
                    "inputPath": run.get("inputPath"),
                    "mode": run.get("mode"),
                    "status": run.get("status"),
                    "totalDurationSeconds": engine.get("totalDurationSeconds"),
                    "analysisDurationSeconds": engine.get("analysisDurationSeconds"),
                    "detectorInferenceSeconds": stages.get("detectorInference"),
                    "sampledFrames": counts.get("sampledFrames"),
                    "analyzedFrames": counts.get("analyzedFrames"),
                    "detections": counts.get("detections"),
                    "violations": counts.get("violations"),
                    "evidenceFrames": counts.get("evidenceFrames"),
                    "violationNames": ";".join(correctness.get("violationNames") or []),
                    "errorType": run.get("errorType"),
                    "errorMessage": run.get("errorMessage"),
                }
            )


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = dict(report.get("summary") or {})
    lines = [
        "# SafeTrace Backend Benchmark",
        "",
        f"- Mode: `{report.get('mode')}`",
        f"- Runs: {summary.get('runCount', 0)}",
        f"- Completed: {summary.get('completedCount', 0)}",
        f"- Average total duration: {summary.get('averageTotalDurationSeconds')}",
        f"- Average detector inference: {summary.get('averageDetectorInferenceSeconds')}",
        f"- Average analyzed frames: {summary.get('averageAnalyzedFrames')}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [run for run in runs if run.get("status") == "completed"]

    def values(path: tuple[str, ...]) -> list[float]:
        items: list[float] = []
        for run in completed:
            value: Any = run
            for key in path:
                value = dict(value or {}).get(key)
            if isinstance(value, (int, float)):
                items.append(float(value))
        return items

    total = values(("engineMetrics", "totalDurationSeconds"))
    detector = values(("engineMetrics", "stageDurations", "detectorInference"))
    analyzed = values(("engineMetrics", "counts", "analyzedFrames"))
    return {
        "runCount": len(runs),
        "completedCount": len(completed),
        "failedCount": len(runs) - len(completed),
        "averageTotalDurationSeconds": round(statistics.mean(total), 4) if total else None,
        "averageDetectorInferenceSeconds": round(statistics.mean(detector), 4) if detector else None,
        "averageAnalyzedFrames": round(statistics.mean(analyzed), 2) if analyzed else None,
    }


def _dry_run_report(mode: str, output: Path) -> dict[str, Any]:
    result = {
        "jobId": "benchmark_dry_run",
        "status": "completed",
        "media": {"name": "synthetic.jpg", "type": "image", "sizeBytes": 0},
        "query": "benchmark dry run",
        "summary": {
            "framesAnalyzed": 1,
            "framesWithViolations": 0,
            "uniqueViolationTypes": 0,
        },
        "violations": [],
        "frames": [],
        "technicalDetails": {
            "pipelineWallClockSeconds": 0.001,
            "processingMetadata": {"sampledFrameCount": 1},
            "componentDiagnostics": {
                "device": "cpu",
                "stageTimings": {"frame_sampling": 0.001, "detector_inference": 0.0},
            },
        },
    }
    engine = build_engine_metrics(result, {}, resource_snapshot={"cpuProcessTimeSeconds": 0.0})
    result["engineMetrics"] = engine
    result["technicalDetails"]["engineMetrics"] = engine
    run = {
        "runIndex": 1,
        "inputPath": None,
        "mode": mode,
        "status": "completed",
        "engineMetrics": engine,
        "resultSummary": result["summary"],
    }
    return _final_report(mode=mode, inputs=[], runs=[run], output=output, dry_run=True)


def _final_report(*, mode: str, inputs: list[Path], runs: list[dict[str, Any]], output: Path, dry_run: bool) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "dryRun": dry_run,
        "inputCount": len(inputs),
        "inputs": [str(path) for path in inputs],
        "summary": _summarize(runs),
        "runs": runs,
        "outputs": {
            "json": str(output / "backend_benchmark_report.json"),
            "csv": str(output / "backend_benchmark_summary.csv"),
            "markdown": str(output / "backend_benchmark_summary.md"),
        },
    }


def _run_one(input_path: Path, *, mode: str, run_index: int, output: Path, query: str, fps: float, top_k: int) -> dict[str, Any]:
    job_id = f"benchmark_{run_index:03d}"
    media_dir = output / "media" / job_id
    media_files: dict[str, Path] = {}

    def register_media(filename: str, path: Path) -> None:
        media_files[filename] = path

    safe_mode = mode in {"rule_based", "safe_mode"}
    started_wall = time.perf_counter()
    runtime_root = output / "runtime_data" / job_id
    old_paths = {
        "data_dir": SETTINGS.data_dir,
        "frames_dir": SETTINGS.frames_dir,
        "embeddings_path": SETTINGS.embeddings_path,
        "metadata_path": SETTINGS.metadata_path,
        "index_path": SETTINGS.index_path,
    }
    try:
        SETTINGS.data_dir = runtime_root
        SETTINGS.frames_dir = runtime_root / "frames"
        SETTINGS.embeddings_path = runtime_root / "embeddings.npy"
        SETTINGS.metadata_path = runtime_root / "metadata.json"
        SETTINGS.index_path = runtime_root / "index.faiss"
        SETTINGS.data_dir.mkdir(parents=True, exist_ok=True)
        SETTINGS.frames_dir.mkdir(parents=True, exist_ok=True)
        component_diagnostics: dict[str, Any] = {}
        raw = run_pipeline(
            upload_path=input_path,
            query=query,
            fps=fps,
            top_k=top_k,
            device="cpu",
            enable_vlm=False,
            safe_mode=safe_mode,
            component_diagnostics=component_diagnostics,
        )
        pipeline_seconds = time.perf_counter() - started_wall
        normalization_started = time.perf_counter()
        result = normalize_pipeline_results(
            job_id=job_id,
            media_name=input_path.name,
            media_type=media_type_for(input_path.name),
            media_size_bytes=input_path.stat().st_size,
            query=query,
            raw_frames=raw,
            media_dir=media_dir,
            register_media=register_media,
        )
        normalization_seconds = time.perf_counter() - normalization_started
        technical = result.setdefault("technicalDetails", {})
        technical["pipelineWallClockSeconds"] = pipeline_seconds
        technical["normalizationWallClockSeconds"] = normalization_seconds
        technical["reportGenerationWallClockSeconds"] = normalization_seconds
        technical["componentDiagnostics"] = component_diagnostics
        engine = build_engine_metrics(result, {}, resource_snapshot=process_resource_snapshot())
        result["engineMetrics"] = engine
        technical["engineMetrics"] = engine
        return {
            "runIndex": run_index,
            "inputPath": str(input_path),
            "mode": mode,
            "status": "completed",
            "engineMetrics": engine,
            "resultSummary": result.get("summary"),
            "violationNames": engine["correctness"]["violationNames"],
            "mediaFiles": {name: str(path) for name, path in media_files.items()},
        }
    except Exception as exc:  # pragma: no cover - exercised manually for missing runtimes
        return {
            "runIndex": run_index,
            "inputPath": str(input_path),
            "mode": mode,
            "status": "failed",
            "errorType": type(exc).__name__,
            "errorMessage": str(exc),
            "engineMetrics": {
                "totalDurationSeconds": round(time.perf_counter() - started_wall, 4),
                "correctness": {"status": "failed", "violationNames": [], "resultSchemaKeys": []},
            },
        }
    finally:
        SETTINGS.data_dir = old_paths["data_dir"]
        SETTINGS.frames_dir = old_paths["frames_dir"]
        SETTINGS.embeddings_path = old_paths["embeddings_path"]
        SETTINGS.metadata_path = old_paths["metadata_path"]
        SETTINGS.index_path = old_paths["index_path"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the SafeTrace backend engine.")
    parser.add_argument("--input", type=Path, help="Input media file or folder.")
    parser.add_argument("--mode", choices=["rule_based", "safe_mode", "standard"], default="rule_based")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query", default="general safety review")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Write schema-valid synthetic reports without models.")
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if args.dry_run:
        report = _dry_run_report(args.mode, output)
    else:
        if args.input is None:
            parser.print_help()
            return 0
        inputs = _media_inputs(args.input.resolve())
        if not inputs:
            parser.error(f"No supported media files found at {args.input}")
        runs: list[dict[str, Any]] = []
        run_index = 1
        for _ in range(max(int(args.repeat or 1), 1)):
            for input_path in inputs:
                runs.append(
                    _run_one(
                        input_path,
                        mode=args.mode,
                        run_index=run_index,
                        output=output,
                        query=args.query,
                        fps=args.fps,
                        top_k=args.top_k,
                    )
                )
                run_index += 1
        report = _final_report(mode=args.mode, inputs=inputs, runs=runs, output=output, dry_run=False)

    _write_json(output / "backend_benchmark_report.json", report)
    _write_csv(output / "backend_benchmark_summary.csv", report["runs"])
    _write_markdown(output / "backend_benchmark_summary.md", report)
    print(f"Wrote benchmark report: {output / 'backend_benchmark_report.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
