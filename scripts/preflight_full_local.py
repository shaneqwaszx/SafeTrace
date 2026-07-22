"""Strict readiness probe for the local full SafeTrace runtime.

The script intentionally loads and probes local components before the backend
is started. It never downloads assets or changes model files. A non-zero exit
means the full-local launcher must not claim the stack is ready.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# `python scripts/preflight_full_local.py` places scripts/ on sys.path, not
# the repository root. Add the root explicitly so this probe uses the same
# source package as the backend launcher.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _result(name: str, *, required: bool, started: float, ok: bool, message: str, **details: Any) -> dict[str, Any]:
    return {
        "name": name,
        "required": required,
        "ok": ok,
        "message": message,
        "elapsedSeconds": round(time.perf_counter() - started, 3),
        "details": details,
    }


def _probe_gpu() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return _result(
            "cuda",
            required=True,
            started=started,
            ok=available,
            message="PyTorch CUDA is available." if available else "PyTorch CUDA is unavailable.",
            torchVersion=torch.__version__,
            cudaVersion=torch.version.cuda,
            deviceCount=int(torch.cuda.device_count()) if available else 0,
            gpuName=torch.cuda.get_device_name(0) if available else None,
        )
    except Exception as exc:  # pragma: no cover - machine dependent
        return _result("cuda", required=True, started=started, ok=False, message=f"{type(exc).__name__}: {exc}")


def _probe_detector(settings) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import numpy as np
        from src.yolo_detector import YoloDetector

        detector = YoloDetector(device="cuda")
        detections = detector.detect(np.zeros((96, 160, 3), dtype=np.uint8))
        checkpoint = str(detector.checkpoint)
        del detector
        return _result(
            "detector",
            required=True,
            started=started,
            ok=True,
            message="Detector loaded and completed a CUDA probe.",
            device="cuda",
            checkpoint=checkpoint,
            probeDetectionCount=len(detections),
        )
    except Exception as exc:  # pragma: no cover - machine dependent
        return _result("detector", required=True, started=started, ok=False, message=f"{type(exc).__name__}: {exc}")
    finally:
        gc.collect()


def _probe_embedding() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from src.clip_embedder import ClipEmbedder

        embedder = ClipEmbedder(device="cuda", batch_size=1)
        vector = embedder.embed_text("local safety review")
        model_dir = str(embedder.model_dir)
        del embedder
        return _result(
            "embedding",
            required=True,
            started=started,
            ok=bool(vector.size),
            message="Embedding model loaded and completed a CUDA text probe.",
            device="cuda",
            modelDir=model_dir,
            vectorShape=list(vector.shape),
        )
    except Exception as exc:  # pragma: no cover - machine dependent
        return _result("embedding", required=True, started=started, ok=False, message=f"{type(exc).__name__}: {exc}")
    finally:
        gc.collect()


def _probe_mobilesam() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import numpy as np
        from src.mobile_sam_segmenter import MobileSamSegmenter
        from src.schemas import Detection

        segmenter = MobileSamSegmenter(device="cuda")
        if not segmenter.available:
            return _result(
                "mobileSam",
                required=True,
                started=started,
                ok=False,
                message="MobileSAM did not become available after construction.",
            )
        detections = [Detection(label="person", raw_label="person", confidence=0.9, bbox=[12, 8, 80, 88])]
        refined = segmenter.refine(np.zeros((96, 160, 3), dtype=np.uint8), detections)
        del segmenter
        return _result(
            "mobileSam",
            required=True,
            started=started,
            ok=bool(refined),
            message="MobileSAM loaded and completed a CUDA mask probe.",
            device="cuda",
            checkpoint="checkpoints/mobile_sam.pt",
        )
    except Exception as exc:  # pragma: no cover - machine dependent
        return _result("mobileSam", required=True, started=started, ok=False, message=f"{type(exc).__name__}: {exc}")
    finally:
        gc.collect()


def _probe_chat() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from src.chat_service import _extract_llama_text, _get_packaged_model, warmup_chat_provider

        status = warmup_chat_provider()
        model = _get_packaged_model()
        generated = _extract_llama_text(
            model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are the local SafeTrace assistant. Reply in one short sentence."},
                    {"role": "user", "content": "Confirm that the local assistant generation probe is working."},
                ],
                temperature=0.0,
                max_tokens=20,
            )
        )
        ok = bool(status.get("available")) and bool(generated)
        return _result(
            "assistant",
            required=True,
            started=started,
            ok=ok,
            message=str(status.get("message") or status.get("reason") or "Assistant probe completed."),
            provider=status.get("provider"),
            model=status.get("model"),
            runtimeAvailable=status.get("runtime_available"),
            modelExists=status.get("model_exists"),
            generatedPreview=generated[:240],
        )
    except Exception as exc:  # pragma: no cover - machine dependent
        return _result("assistant", required=True, started=started, ok=False, message=f"{type(exc).__name__}: {exc}")


def _probe_visual_provider(settings) -> dict[str, Any]:
    started = time.perf_counter()
    selected = str(getattr(settings, "vlm_profile", "rule_based") or "rule_based")
    if selected == "rule_based":
        return _result(
            "visualExplanations",
            required=False,
            started=started,
            ok=True,
            message="Rule-based visual explanation fallback is explicitly selected.",
            selectedProfile=selected,
            provider="rule_based",
        )
    from src.api.jobs import resolve_vlm_profile_model_dir

    model_dir = resolve_vlm_profile_model_dir(selected)
    ok = model_dir is not None and model_dir.is_dir()
    return _result(
        "visualExplanations",
        required=False,
        started=started,
        ok=ok,
        message="Selected local VLM assets are present." if ok else "Selected VLM assets are unavailable.",
        selectedProfile=selected,
        modelDir=str(model_dir) if model_dir else None,
    )


def _probe_mid_vlm(settings) -> dict[str, Any]:
    started = time.perf_counter()
    from src.scene_applicability import parse_mid_vlm_scene_response

    configured = Path(str(getattr(settings, "mid_vlm_model_path", "models/vlm/lightweight-512m")))
    model_dir = configured if configured.is_absolute() else REPO_ROOT / configured
    required_files = [model_dir / "config.json", model_dir / "processor_config.json"]
    weights = list(model_dir.glob("*.safetensors")) if model_dir.is_dir() else []
    contract_ok = False
    try:
        parse_mid_vlm_scene_response({
            "sceneType": "unknown", "personVisible": False, "vehicleInteriorVisible": False,
            "driverRegionVisible": False, "profileApplicable": False, "violationSupported": False,
            "violationType": "", "confidence": 0.0, "reason": "preflight",
        })
        contract_ok = True
    except ValueError:
        pass
    assets = model_dir.is_dir() and bool(weights) and any(path.is_file() for path in required_files)
    enabled = bool(getattr(settings, "mid_vlm_enabled", False))
    return _result(
        "midVlm",
        required=enabled,
        started=started,
        ok=bool(assets and contract_ok and int(getattr(settings, "mid_vlm_concurrency", 1)) == 1),
        message="Mid-size scene verifier assets and strict JSON contract are ready." if assets and contract_ok else "Mid-size scene verifier assets or contract are unavailable.",
        enabled=enabled,
        modelDir=str(model_dir),
        assetFilesPresent=assets,
        strictJsonContract=contract_ok,
        concurrency=int(getattr(settings, "mid_vlm_concurrency", 1)),
        authoritative=False,
    )


def run_preflight(profile: str) -> dict[str, Any]:
    os.environ["SAFETRACE_RUNTIME_PROFILE"] = profile
    os.environ.setdefault("SAFETRACE_REQUIRE_GPU", "1")
    os.environ.setdefault("SAFETRACE_REQUIRE_CHAT", "1")
    os.environ.setdefault("SAFETRACE_REQUIRE_MOBILESAM", "1")
    os.environ.setdefault("SAFETRACE_ALLOW_CPU_FALLBACK", "0")
    os.environ.setdefault("SAFETRACE_DEVICE", "cuda")
    os.environ.setdefault("SAFETRACE_ENABLE_GPU_AUTO", "true")
    os.environ.setdefault("SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM", "true")
    os.environ.setdefault("SAFETRACE_MOBILESAM_ENABLED", "auto")
    os.environ.setdefault("SAFETRACE_MOBILESAM_WORKER_ENABLED", "true")
    os.environ.setdefault("SAFETRACE_CHAT_ENABLED", "auto")
    os.environ.setdefault("SAFETRACE_CHAT_AUTOLOAD", "true")

    from src.config import SETTINGS

    checks = [_probe_gpu(), _probe_detector(SETTINGS), _probe_embedding(), _probe_mobilesam(), _probe_chat(), _probe_visual_provider(SETTINGS), _probe_mid_vlm(SETTINGS)]
    strict_ok = all(item["ok"] for item in checks if item["required"])
    by_name = {str(item["name"]): item for item in checks}
    cuda = by_name["cuda"]
    detector = by_name["detector"]
    embedding = by_name["embedding"]
    mobile_sam = by_name["mobileSam"]
    assistant = by_name["assistant"]
    visual = by_name["visualExplanations"]
    mid_vlm = by_name["midVlm"]
    blocking_reasons = [item["message"] for item in checks if item["required"] and not item["ok"]]
    warnings = [item["message"] for item in checks if not item["required"] and not item["ok"]]
    return {
        "profile": profile,
        "runtimeProfile": profile,
        "strict": True,
        "passed": strict_ok,
        "overallReady": strict_ok,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "pythonExecutable": sys.executable,
        "gpu": {
            "required": bool(getattr(SETTINGS, "require_gpu", True)),
            "available": bool(cuda["ok"]),
            "selected": "cuda" if cuda["ok"] else None,
            "deviceName": cuda.get("details", {}).get("gpuName"),
            "torchVersion": cuda.get("details", {}).get("torchVersion"),
            "cudaVersion": cuda.get("details", {}).get("cudaVersion"),
        },
        "detector": {
            "found": Path(str(detector.get("details", {}).get("checkpoint") or "")).is_file(),
            "loaded": bool(detector["ok"]),
            "device": detector.get("details", {}).get("device"),
            "probePassed": bool(detector["ok"]),
        },
        "embedding": {
            "found": Path(str(embedding.get("details", {}).get("modelDir") or "")).is_dir(),
            "loaded": bool(embedding["ok"]),
            "device": embedding.get("details", {}).get("device"),
            "probePassed": bool(embedding["ok"]),
        },
        "mobileSam": {
            "required": bool(getattr(SETTINGS, "require_mobilesam", True)),
            "found": Path(str(mobile_sam.get("details", {}).get("checkpoint") or "")).is_file(),
            "loaded": bool(mobile_sam["ok"]),
            "enabled": str(getattr(SETTINGS, "mobile_sam_enabled", "disabled")) not in {"disabled", "false", "0"},
            "device": mobile_sam.get("details", {}).get("device"),
            "probePassed": bool(mobile_sam["ok"]),
        },
        "assistant": {
            "required": bool(getattr(SETTINGS, "require_chat", True)),
            "modelFound": bool(assistant.get("details", {}).get("modelExists")),
            "runtimeFound": bool(assistant.get("details", {}).get("runtimeAvailable")),
            "loaded": bool(assistant["ok"]),
            "probePassed": bool(assistant["ok"] and assistant.get("details", {}).get("generatedPreview")),
            "generatedPreview": assistant.get("details", {}).get("generatedPreview"),
        },
        "visualExplanations": {
            "requested": str(getattr(SETTINGS, "vlm_profile", "rule_based")),
            "provider": visual.get("details", {}).get("provider"),
            "ready": bool(visual["ok"]),
            "fallback": "rule_based" if visual.get("details", {}).get("provider") == "rule_based" else None,
        },
        "midVlm": {
            "enabled": bool(mid_vlm.get("details", {}).get("enabled")),
            "found": bool(mid_vlm.get("details", {}).get("assetFilesPresent")),
            "strictJsonContract": bool(mid_vlm.get("details", {}).get("strictJsonContract")),
            "concurrency": mid_vlm.get("details", {}).get("concurrency"),
            "authoritative": False,
            "probePassed": bool(mid_vlm["ok"]),
        },
        "blockingReasons": blocking_reasons,
        "warnings": warnings,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the strict SafeTrace local_full runtime.")
    parser.add_argument("--profile", default="local_full", choices=["local_full"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = run_preflight(args.profile)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
