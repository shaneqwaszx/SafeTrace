"""Crash-isolated, non-authoritative mid-size scene verifier."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import SETTINGS
from .scene_applicability import mid_vlm_scene_prompt, parse_mid_vlm_scene_response
from .utils import imwrite_rgb, resolve_device

_MID_VLM_LANE = threading.BoundedSemaphore(1)


class MidVlmSceneVerifier:
    """Use one local 512M lane to verify context, never to originate findings."""

    def __init__(self) -> None:
        configured = Path(str(getattr(SETTINGS, "mid_vlm_model_path", "models/vlm/lightweight-512m")))
        self.model_dir = configured if configured.is_absolute() else SETTINGS.project_root / configured
        self.device = resolve_device(str(getattr(SETTINGS, "mid_vlm_device", "cuda") or "cuda"))
        self.timeout_seconds = max(15.0, float(getattr(SETTINGS, "mid_vlm_timeout_seconds", 90.0) or 90.0))
        self.enabled = bool(getattr(SETTINGS, "mid_vlm_enabled", False)) and self.model_dir.is_dir()
        self.last_diagnostics: dict[str, Any] = {
            "attempted": False,
            "accepted": False,
            "reason": None if self.enabled else "disabled_or_model_missing",
            "modelPath": str(self.model_dir),
            "device": self.device,
            "authoritative": False,
        }

    @staticmethod
    def _command(request_path: Path, output_path: Path, app_root: Path) -> list[str]:
        if getattr(sys, "frozen", False):
            return [
                sys.executable,
                "--lightweight-vlm-worker",
                "--input-json",
                str(request_path),
                "--output-json",
                str(output_path),
                "--app-root",
                str(app_root),
            ]
        return [
            sys.executable,
            "-m",
            "src.lightweight_vlm_worker",
            "--input-json",
            str(request_path),
            "--output-json",
            str(output_path),
            "--app-root",
            str(app_root),
        ]

    def verify(
        self,
        image: np.ndarray,
        *,
        finding: str,
        profile: str,
        query: str,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        app_root = Path(os.environ.get("SAFETRACE_APP_ROOT") or SETTINGS.project_root).resolve()
        work_dir = SETTINGS.data_dir / "mid_vlm_worker"
        work_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        image_path = work_dir / f"scene_{token}.jpg"
        request_path = work_dir / f"request_{token}.json"
        output_path = work_dir / f"result_{token}.json"
        resized = image
        height, width = image.shape[:2]
        if max(height, width) > 384:
            scale = 384.0 / max(height, width)
            resized = np.asarray(
                Image.fromarray(image).resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.Resampling.BICUBIC,
                )
            )
        imwrite_rgb(image_path, resized)
        payload = {
            "task": "scene_applicability",
            "imagePath": str(image_path),
            "modelDir": str(self.model_dir),
            "device": self.device,
            "profile": "lightweight_512m",
            "prompt": mid_vlm_scene_prompt(finding=finding, profile=profile, query=query),
            "maxTokens": 96,
            "timeoutSeconds": self.timeout_seconds,
            "generationTimeoutSeconds": max(10.0, self.timeout_seconds - 10.0),
        }
        request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        started = time.perf_counter()
        try:
            with _MID_VLM_LANE:
                result = subprocess.run(
                    self._command(request_path, output_path, app_root),
                    cwd=str(app_root),
                    env={**os.environ, "SAFETRACE_DEVICE": self.device},
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            worker_payload = json.loads(output_path.read_text(encoding="utf-8")) if output_path.is_file() else {}
            if result.returncode != 0 or not worker_payload.get("ok"):
                self.last_diagnostics = {
                    **self.last_diagnostics,
                    "attempted": True,
                    "accepted": False,
                    "reason": worker_payload.get("fallbackReason") or worker_payload.get("errorType") or f"worker_exit_{result.returncode}",
                    "elapsedSeconds": round(time.perf_counter() - started, 3),
                    "stderrPreview": " ".join(result.stderr.split())[:240],
                    "worker": worker_payload,
                }
                return None
            parsed = parse_mid_vlm_scene_response(worker_payload.get("sceneVerification") or {})
            self.last_diagnostics = {
                **self.last_diagnostics,
                "attempted": True,
                "accepted": True,
                "reason": "strict_json_accepted",
                "elapsedSeconds": round(time.perf_counter() - started, 3),
                "worker": worker_payload,
            }
            return parsed
        except subprocess.TimeoutExpired:
            self.last_diagnostics = {
                **self.last_diagnostics,
                "attempted": True,
                "accepted": False,
                "timedOut": True,
                "reason": "worker_timeout",
                "elapsedSeconds": round(time.perf_counter() - started, 3),
            }
            return None
        except Exception as exc:
            self.last_diagnostics = {
                **self.last_diagnostics,
                "attempted": True,
                "accepted": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "elapsedSeconds": round(time.perf_counter() - started, 3),
            }
            return None
        finally:
            for path in (request_path, output_path, image_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
