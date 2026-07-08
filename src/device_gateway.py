"""Device-aware routing for SafeTrace analysis layers.

The gateway is intentionally read-only: it inspects available runtime signals and
chooses conservative devices for each component without installing drivers or
changing model assets.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict


def _mode(value: Any, *, default: str = "auto") -> str:
    raw = str(value or default).strip().lower()
    return raw if raw in {"auto", "cpu", "cuda"} else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


@dataclass(frozen=True)
class TorchDeviceStatus:
    installed: bool
    cuda_available: bool
    cuda_device_count: int
    gpu_name: str | None = None
    total_gpu_memory_mb: int | None = None
    reserved_gpu_memory_mb: int | None = None
    allocated_gpu_memory_mb: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class HardwareGpuStatus:
    nvidia_smi_available: bool
    nvidia_gpu_detected: bool
    gpu_name: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ComponentDeviceDecision:
    component: str
    configured: str
    selected: str
    available: bool
    reason: str
    requires_gpu: bool = False


def torch_status() -> TorchDeviceStatus:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        return TorchDeviceStatus(
            installed=False,
            cuda_available=False,
            cuda_device_count=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        cuda_available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if cuda_available else 0
        name = torch.cuda.get_device_name(0) if count else None
        total_mb = None
        reserved_mb = None
        allocated_mb = None
        if count:
            props = torch.cuda.get_device_properties(0)
            total_mb = int(getattr(props, "total_memory", 0) / (1024 * 1024))
            reserved_mb = int(torch.cuda.memory_reserved(0) / (1024 * 1024))
            allocated_mb = int(torch.cuda.memory_allocated(0) / (1024 * 1024))
        return TorchDeviceStatus(
            installed=True,
            cuda_available=cuda_available,
            cuda_device_count=count,
            gpu_name=name,
            total_gpu_memory_mb=total_mb,
            reserved_gpu_memory_mb=reserved_mb,
            allocated_gpu_memory_mb=allocated_mb,
        )
    except Exception as exc:  # pragma: no cover - runtime-specific
        return TorchDeviceStatus(
            installed=True,
            cuda_available=False,
            cuda_device_count=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def hardware_gpu_status() -> HardwareGpuStatus:
    if shutil.which("nvidia-smi") is None:
        return HardwareGpuStatus(
            nvidia_smi_available=False,
            nvidia_gpu_detected=False,
            error="nvidia-smi not found",
        )
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except Exception as exc:  # pragma: no cover - machine-specific
        return HardwareGpuStatus(
            nvidia_smi_available=True,
            nvidia_gpu_detected=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return HardwareGpuStatus(
        nvidia_smi_available=True,
        nvidia_gpu_detected=bool(names),
        gpu_name=names[0] if names else None,
        error=None if completed.returncode == 0 else completed.stderr.strip() or "nvidia-smi failed",
    )


def _choose_device(
    *,
    component: str,
    configured: str,
    torch_info: TorchDeviceStatus,
    hardware_info: HardwareGpuStatus,
    requires_gpu: bool = False,
    cpu_allowed: bool = True,
    gpu_auto_enabled: bool = True,
) -> ComponentDeviceDecision:
    configured_mode = _mode(configured)
    cuda_ready = bool(torch_info.installed and torch_info.cuda_available and torch_info.cuda_device_count > 0)

    if requires_gpu:
        if configured_mode == "cpu":
            return ComponentDeviceDecision(
                component=component,
                configured=configured_mode,
                selected="unavailable",
                available=False,
                reason="GPU is required for this layer; CPU was configured.",
                requires_gpu=True,
            )
        if cuda_ready:
            return ComponentDeviceDecision(
                component=component,
                configured=configured_mode,
                selected="cuda",
                available=True,
                reason="PyTorch CUDA runtime is available.",
                requires_gpu=True,
            )
        reason = "PyTorch CUDA runtime is unavailable."
        if hardware_info.nvidia_gpu_detected:
            reason = "NVIDIA hardware may exist, but PyTorch CUDA runtime is unavailable."
        return ComponentDeviceDecision(
            component=component,
            configured=configured_mode,
            selected="unavailable",
            available=False,
            reason=reason,
            requires_gpu=True,
        )

    if configured_mode == "cuda":
        if cuda_ready:
            return ComponentDeviceDecision(component, configured_mode, "cuda", True, "CUDA explicitly configured.")
        if cpu_allowed:
            return ComponentDeviceDecision(
                component,
                configured_mode,
                "cpu",
                True,
                "CUDA was requested but unavailable; using CPU fallback.",
            )
        return ComponentDeviceDecision(component, configured_mode, "unavailable", False, "CUDA requested but unavailable.")

    if configured_mode == "cpu":
        return ComponentDeviceDecision(component, configured_mode, "cpu", cpu_allowed, "CPU explicitly configured.")

    if gpu_auto_enabled and cuda_ready:
        return ComponentDeviceDecision(component, configured_mode, "cuda", True, "Auto selected CUDA.")
    return ComponentDeviceDecision(component, configured_mode, "cpu", cpu_allowed, "Auto selected CPU fallback.")


def device_gateway_payload(settings: Any) -> Dict[str, Any]:
    torch_info = torch_status()
    hardware_info = hardware_gpu_status()
    gpu_auto_enabled = _env_bool("SAFETRACE_ENABLE_GPU_AUTO", True)
    configured_device = _mode(getattr(settings, "device", "auto"))
    lightweight_device = _mode(getattr(settings, "lightweight_vlm_device", "auto"))
    enhanced_device = _mode(getattr(settings, "enhanced_vlm_device", "cuda"), default="cuda")
    mobile_sam_device = _mode(getattr(settings, "mobile_sam_device", "auto"))
    decisions = {
        "detector": _choose_device(
            component="detector",
            configured=configured_device,
            torch_info=torch_info,
            hardware_info=hardware_info,
            gpu_auto_enabled=gpu_auto_enabled,
        ),
        "mobileSam": _choose_device(
            component="mobileSam",
            configured=mobile_sam_device,
            torch_info=torch_info,
            hardware_info=hardware_info,
            gpu_auto_enabled=gpu_auto_enabled,
        ),
        "lightweightVlm": _choose_device(
            component="lightweightVlm",
            configured=lightweight_device,
            torch_info=torch_info,
            hardware_info=hardware_info,
            gpu_auto_enabled=gpu_auto_enabled,
        ),
        "enhancedVlm": _choose_device(
            component="enhancedVlm",
            configured=enhanced_device,
            torch_info=torch_info,
            hardware_info=hardware_info,
            requires_gpu=True,
            cpu_allowed=False,
            gpu_auto_enabled=gpu_auto_enabled,
        ),
    }
    return {
        "configuredDevice": configured_device,
        "gpuAutoEnabled": gpu_auto_enabled,
        "torch": asdict(torch_info),
        "hardware": asdict(hardware_info),
        "components": {key: asdict(value) for key, value in decisions.items()},
        "gpuUnavailableReason": (
            None
            if torch_info.cuda_available
            else "NVIDIA hardware may exist but PyTorch CUDA runtime is unavailable."
            if hardware_info.nvidia_gpu_detected
            else torch_info.error or hardware_info.error or "No CUDA-capable GPU reported by PyTorch."
        ),
    }


def selected_component_device(settings: Any, component: str) -> str:
    payload = device_gateway_payload(settings)
    decision = payload.get("components", {}).get(component, {})
    selected = str(decision.get("selected") or "cpu")
    return "cpu" if selected == "unavailable" else selected
