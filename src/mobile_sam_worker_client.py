"""Parent-process client for crash-isolated MobileSAM refinement."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import SETTINGS
from .mask_encoding import decode_bool_mask
from .schemas import Detection
from .utils import imwrite_rgb

logger = logging.getLogger("safetrace.msam.worker")


def _disabled_mode(value: str | None) -> bool:
    return (value or "").strip().lower() in {"0", "false", "no", "off", "disabled", "none"}


def _worker_dir() -> Path:
    path = SETTINGS.data_dir / "mobile_sam_worker"
    path.mkdir(parents=True, exist_ok=True)
    return path


class MobileSamWorkerSegmenter:
    """MobileSAM refinement through a subprocess, never in the API process."""

    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None) -> None:
        self.checkpoint = Path(checkpoint or SETTINGS.mobile_sam_checkpoint)
        requested_device = str(device or SETTINGS.mobile_sam_device or SETTINGS.device or "auto").strip().lower()
        self.device = "cuda" if requested_device == "cuda" else "cpu"
        self.timeout_seconds = max(
            1.0,
            float(getattr(SETTINGS, "mobile_sam_worker_timeout_seconds", 60.0) or 60.0),
        )
        self.enabled = bool(getattr(SETTINGS, "mobile_sam_worker_enabled", False)) and not _disabled_mode(
            getattr(SETTINGS, "mobile_sam_enabled", "disabled")
        )
        self._available = bool(self.enabled and self.checkpoint.is_file())
        self.last_diagnostics: Dict[str, Any] = self._diagnostics(
            attempted=False,
            succeeded=False,
            timed_out=False,
            exit_code=None,
            source="disabled" if not self.enabled else "fallback",
            reason=None if self._available else "checkpoint_missing" if self.enabled else "worker_disabled",
        )

    @property
    def available(self) -> bool:
        return self._available

    def _diagnostics(
        self,
        *,
        attempted: bool,
        succeeded: bool,
        timed_out: bool,
        exit_code: int | None,
        source: str,
        reason: str | None,
    ) -> Dict[str, Any]:
        return {
            "mobileSamWorkerEnabled": bool(self.enabled),
            "actualMobileSamDevice": self.device,
            "mobileSamWorkerTimeoutSeconds": self.timeout_seconds,
            "mobileSamWorkerAttempted": bool(attempted),
            "mobileSamWorkerSucceeded": bool(succeeded),
            "mobileSamWorkerTimedOut": bool(timed_out),
            "mobileSamWorkerExitCode": exit_code,
            "mobileSamFallbackReason": reason,
            "mobileSamRefinementSource": source,
        }

    def _fallback(
        self,
        detections: List[Detection],
        *,
        reason: str,
        attempted: bool = True,
        timed_out: bool = False,
        exit_code: int | None = None,
    ) -> List[Detection]:
        for detection in detections:
            if detection.refined_mask is None and detection.coarse_mask is not None:
                detection.refined_mask = detection.coarse_mask
        self.last_diagnostics = self._diagnostics(
            attempted=attempted,
            succeeded=False,
            timed_out=timed_out,
            exit_code=exit_code,
            source="fallback" if attempted else "disabled",
            reason=reason,
        )
        return detections

    def _materialize_image(self, image) -> Path:
        if isinstance(image, (str, Path)):
            return Path(image)
        if not isinstance(image, np.ndarray):
            raise TypeError("MobileSAM worker requires an image path or numpy image array.")
        image_path = _worker_dir() / f"frame_{uuid.uuid4().hex}.jpg"
        imwrite_rgb(image_path, image)
        return image_path

    def _command(self, input_path: Path, output_path: Path) -> list[str]:
        app_root = Path(os.environ.get("SAFETRACE_APP_ROOT") or SETTINGS.project_root).resolve()
        if getattr(sys, "frozen", False):
            return [
                sys.executable,
                "--mobile-sam-worker",
                "--input-json",
                str(input_path),
                "--output-json",
                str(output_path),
                "--app-root",
                str(app_root),
            ]
        return [
            sys.executable,
            "-m",
            "src.mobile_sam_worker",
            "--input-json",
            str(input_path),
            "--output-json",
            str(output_path),
            "--app-root",
            str(app_root),
        ]

    def _request_payload(self, image_path: Path, detections: List[Detection]) -> Dict[str, Any]:
        return {
            "imagePath": str(image_path),
            "checkpoint": str(self.checkpoint),
            "device": self.device,
            "detections": [
                {
                    "index": index,
                    "label": detection.label,
                    "raw_label": detection.raw_label,
                    "confidence": float(detection.confidence),
                    "bbox": [float(value) for value in detection.bbox],
                }
                for index, detection in enumerate(detections)
            ],
        }

    def _consume_result(self, output_path: Path, detections: List[Detection]) -> List[Detection]:
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.last_diagnostics = self._diagnostics(
                attempted=True,
                succeeded=False,
                timed_out=False,
                exit_code=0,
                source="fallback",
                reason=f"invalid_worker_json:{type(exc).__name__}",
            )
            return self._fallback(detections, reason=self.last_diagnostics["mobileSamFallbackReason"], exit_code=0)

        if not bool(payload.get("ok")):
            reason = str(payload.get("errorType") or payload.get("errorMessage") or "worker_error")
            return self._fallback(detections, reason=reason)

        for item in list(payload.get("detections") or []):
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= len(detections):
                continue
            mask = decode_bool_mask(item.get("refinedMask"))
            if mask is not None:
                detections[index].refined_mask = mask
            elif detections[index].refined_mask is None and detections[index].coarse_mask is not None:
                detections[index].refined_mask = detections[index].coarse_mask

        self.last_diagnostics = self._diagnostics(
            attempted=True,
            succeeded=True,
            timed_out=False,
            exit_code=0,
            source="worker",
            reason=None,
        )
        return detections

    def refine(self, image, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return detections
        if not self.enabled:
            return self._fallback(detections, reason="worker_disabled", attempted=False)
        if not self.checkpoint.is_file():
            return self._fallback(detections, reason="checkpoint_missing", attempted=False)

        try:
            image_path = self._materialize_image(image)
            work_dir = _worker_dir()
            request_path = work_dir / f"request_{uuid.uuid4().hex}.json"
            output_path = work_dir / f"result_{uuid.uuid4().hex}.json"
            request_path.write_text(
                json.dumps(self._request_payload(image_path, detections), indent=2, default=str),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            env.setdefault("OMP_NUM_THREADS", "1")
            env["SAFETRACE_DEVICE"] = self.device
            env["SAFETRACE_MOBILESAM_DEVICE"] = self.device
            result = subprocess.run(
                self._command(request_path, output_path),
                cwd=str(Path(os.environ.get("SAFETRACE_APP_ROOT") or SETTINGS.project_root).resolve()),
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("MobileSAM worker timed out after %.1fs; using detector-box fallback.", self.timeout_seconds)
            self.last_diagnostics = self._diagnostics(
                attempted=True,
                succeeded=False,
                timed_out=True,
                exit_code=None,
                source="fallback",
                reason="worker_timeout",
            )
            return self._fallback(detections, reason="worker_timeout", timed_out=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("MobileSAM worker launch failed: %s", exc)
            return self._fallback(detections, reason=f"worker_launch_failed:{type(exc).__name__}")

        if result.returncode != 0:
            reason = f"worker_exit_{result.returncode}"
            if output_path.exists():
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                    reason = str(payload.get("errorType") or reason)
                except (OSError, json.JSONDecodeError):
                    pass
            logger.warning("MobileSAM worker exited with %s; using detector-box fallback.", result.returncode)
            self.last_diagnostics = self._diagnostics(
                attempted=True,
                succeeded=False,
                timed_out=False,
                exit_code=result.returncode,
                source="fallback",
                reason=reason,
            )
            return self._fallback(detections, reason=reason, exit_code=result.returncode)

        return self._consume_result(output_path, detections)
