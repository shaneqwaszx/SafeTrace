"""Optional local Vision-Language Model reasoning.

SafeTrace preserves the original in-process local transformers VLM path and can
also use a local Ollama vision model when explicitly selected or when auto mode
cannot use the local provider. If no local provider is available, analysis keeps
running with deterministic rule-based explanations.
"""
from __future__ import annotations

import base64
import io
import logging
import re
import threading
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import httpx
import numpy as np
from PIL import Image

from .config import SETTINGS
from .schemas import Violation
from .utils import resolve_device

logger = logging.getLogger("safetrace.vlm")

AUTO_PROVIDER = "auto"
LOCAL_PROVIDER = "local"
OLLAMA_PROVIDER = "ollama"
RULE_BASED_PROVIDER = "rule_based"
LOCAL_VLM_HOSTS = {"127.0.0.1", "localhost", "::1"}
OLLAMA_PROVIDERS = {"ollama", "ollama_vision"}
TRANSFORMER_PROVIDERS = {"local", "legacy", "existing", "transformers", "local_transformers", "local_dir"}
VLM_MODEL_FILENAMES = {
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer.json",
    "preprocessor_config.json",
    "processor_config.json",
}
VLM_MODEL_SUFFIXES = {".safetensors", ".bin"}
VLM_PROMPT = (
    "You are inspecting one vehicle safety evidence frame.\n"
    "Return only the final explanation. Do not repeat the prompt.\n"
    "Do not include User:, Assistant:, XML tags, table tokens, or image tokens.\n"
    "Describe visible evidence related to the listed SafeTrace findings.\n"
    "Mention uncertainty from blur, glare, occlusion, or camera angle.\n"
    "Use 2-4 concise sentences under 90 words."
)
VLM_PROMPT_ECHO_PHRASES = (
    "You are inspecting one vehicle safety evidence frame.",
    "Return only the final explanation.",
    "Do not repeat the prompt.",
    "Do not include User:, Assistant:, XML tags, table tokens, or image tokens.",
    "Describe visible evidence related to the listed SafeTrace findings.",
    "Describe only visible safety evidence in this frame.",
    "Do not make legal conclusions.",
    "Mention uncertainty from camera angle, blur, glare, or occlusion.",
    "Mention uncertainty from blur, glare, occlusion, or camera angle.",
    "Keep answer under 90 words.",
    "Use 2-4 concise sentences under 90 words.",
    "Potential SafeTrace findings to inspect:",
    "Findings to inspect:",
    "You are checking one SafeTrace evidence image for the selected safety finding.",
    "Use only visible evidence.",
    "If the evidence is unclear, say cannot determine.",
    "Do not list unrelated objects.",
    "Return exactly these four short lines:",
    "Finding: seatbelt compliance.",
    "Finding: helmet or PPE compliance.",
    "Finding: phone use or distracted driving.",
    "Look only at the image crop.",
    "Reply with exactly:",
)
VLM_GENERIC_RESPONSES = {
    "unclear",
    "unclear.",
    "safety evidence missing",
    "safety evidence missing.",
    "no safety evidence",
    "no safety evidence.",
    "no visible safety evidence",
    "no visible safety evidence.",
    "not enough information",
    "not enough information.",
    "seatbelt compliance",
    "seatbelt compliance.",
    "helmet or ppe compliance",
    "helmet or ppe compliance.",
    "phone use or distracted driving",
    "phone use or distracted driving.",
}
VLM_USEFUL_KEYWORDS = (
    "visible",
    "evidence",
    "safety",
    "uncertain",
    "uncertainty",
    "cannot determine",
    "not visible",
    "not worn",
    "worn",
    "occluded",
    "blur",
    "glare",
    "occlusion",
    "camera",
    "angle",
    "helmet",
    "seatbelt",
    "seat belt",
    "phone",
    "hand",
    "control",
    "worker",
    "driver",
    "person",
    "vehicle",
    "forklift",
    "restricted",
    "zone",
    "vest",
    "ppe",
    "appears",
    "detected",
)
VLM_SAFETY_EVIDENCE_KEYWORDS = (
    "helmet",
    "hardhat",
    "hard hat",
    "seatbelt",
    "seat belt",
    "belt",
    "phone",
    "mobile",
    "hand",
    "hands",
    "worker",
    "driver",
    "occupant",
    "torso",
    "vehicle safety",
    "ppe",
    "vest",
    "uniform",
    "unsafe",
    "violation",
    "missing",
    "wearing",
    "worn",
    "not worn",
    "cannot determine",
    "not visible",
    "occluded",
)
VLM_GENERIC_INVENTORY_RE = re.compile(
    r"\b(?:some|several|various|many)\s+(?:objects?|items?|things?)\b|"
    r"\bobjects?\s+on\s+(?:the\s+)?(?:table|floor|ground)\b|"
    r"\bother\s+objects?\b",
    re.IGNORECASE,
)
VLM_GENERIC_PERSON_ACTIVITY_RE = re.compile(
    r"\b(?:a\s+)?person\s+(?:is\s+)?(?:holding|standing|sitting|walking|looking|carrying|near|beside)\b",
    re.IGNORECASE,
)
VLM_GENERIC_FINDING_LABEL_RE = re.compile(
    r"\bvisible\s+evidence\s*:\s*(?:seatbelt\s+compliance|helmet\s+or\s+ppe\s+compliance|phone\s+use\s+or\s+distracted\s+driving)\b",
    re.IGNORECASE,
)
VLM_OPTION_LIST_ECHO_RE = re.compile(
    r"\b(?:worn\s*,\s*)?not\s+visible\s*,\s*not\s+worn\s*,?\s+or\s+cannot\s+determine\b|"
    r"\bvisible\s*,\s*not\s+visible\s*,\s*missing\s*,?\s+or\s+cannot\s+determine\b|"
    r"\bvisible\s*,\s*not\s+visible\s*,?\s+or\s+cannot\s+determine\b|"
    r"\blow\s*,\s*medium\s*,?\s+or\s+high\b",
    re.IGNORECASE,
)
VLM_ROLE_LABEL_RE = re.compile(r"\b(?:user|assistant|system)\s*:", re.IGNORECASE)
VLM_ARTIFACT_RE = re.compile(
    r"<\s*/?\s*[^>\s]*(?:image|img|row_|col_|global|table)[^>]*>",
    re.IGNORECASE,
)
VLM_PLACEHOLDER_RE = re.compile(r"<[^>\n]+>")
VLM_ARTIFACT_LEAK_RE = re.compile(
    r"<\s*/?\s*[^>\s]*(?:image|img|row_|col_|global|table)[^>]*>|\b(?:user|assistant)\s*:",
    re.IGNORECASE,
)
VLM_FIELD_RE = re.compile(
    r"\b(visible_evidence|visual_status|short_reason|confidence_hint)\s*:\s*(.*?)(?=\b(?:visible_evidence|visual_status|short_reason|confidence_hint)\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)


def _normalized_mode(value: str | None) -> str:
    raw = (value or "auto").strip().lower()
    if raw in {"0", "false", "no", "off", "disabled", "none"}:
        return "disabled"
    if raw in {"1", "true", "yes", "on", "enabled"}:
        return "enabled"
    return "auto"


def _normalized_provider(value: str | None) -> str:
    raw = (value or AUTO_PROVIDER).strip().lower()
    if raw in {"", AUTO_PROVIDER}:
        return AUTO_PROVIDER
    if raw in OLLAMA_PROVIDERS:
        return OLLAMA_PROVIDER
    if raw in TRANSFORMER_PROVIDERS:
        return LOCAL_PROVIDER
    return raw


def _fallback_explanation(violations: Sequence[Violation]) -> str:
    if not violations:
        return "Rule-based explanation: no safety violations were detected in this frame."

    lines = ["Rule-based explanation: SafeTrace flagged visible evidence for reviewer confirmation."]
    for violation in violations:
        lines.append(f"- {violation.name}: {violation.description}")
    lines.append("Confirm against the original footage, especially if blur, glare, angle, or occlusion affects the view.")
    return "\n".join(lines)


def _normalize_for_quality(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _contains_keyword(text: str, keywords: Sequence[str]) -> bool:
    lowered = text.lower()
    for keyword in keywords:
        value = str(keyword or "").strip().lower()
        if not value:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(value) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            return True
    return False


def _strip_prompt_echoes(text: str, prompt_text: str) -> str:
    cleaned = text.replace(prompt_text, " ")
    cleaned = cleaned.replace(VLM_PROMPT, " ")
    for phrase in VLM_PROMPT_ECHO_PHRASES:
        cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:potential\s+)?safetrace findings to inspect\s*:[^.:\n]*(?:[.\n]|$)", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\bfindings to inspect\s*:[^.:\n]*(?:[.\n]|$)", " ", cleaned, flags=re.I)
    return cleaned


def sanitize_vlm_output(raw_text: str, prompt_text: str) -> str:
    """Remove chat-template echo and vision/table markup before display."""
    if not raw_text:
        return ""

    cleaned = str(raw_text).replace("\r", "\n")
    cleaned = cleaned.replace("\\n", "\n")
    cleaned = _strip_prompt_echoes(cleaned, prompt_text)
    cleaned = VLM_ROLE_LABEL_RE.sub("\n", cleaned)
    cleaned = VLM_ARTIFACT_RE.sub(" ", cleaned)
    cleaned = re.sub(r"</?s>|<pad>|<unk>", " ", cleaned, flags=re.I)

    lines: list[str] = []
    for line in cleaned.splitlines():
        line = line.strip(" \t-")
        if not line:
            continue
        if any(phrase.lower() in line.lower() for phrase in VLM_PROMPT_ECHO_PHRASES):
            continue
        if re.match(r"^(?:findings|potential safetrace findings)\s+to\s+inspect\b", line, flags=re.I):
            continue
        lines.append(line)

    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\n:-")
    if len(cleaned) > 650:
        clipped = cleaned[:650].rsplit(".", 1)[0].strip()
        cleaned = clipped if len(clipped) >= 40 else cleaned[:650].strip()
    return cleaned


def vlm_output_quality_issue(clean_text: str) -> str | None:
    text = clean_text.strip()
    if not text:
        return "empty output"
    if VLM_ARTIFACT_LEAK_RE.search(text):
        return "token or role-label leak"
    if VLM_PLACEHOLDER_RE.search(text):
        return "prompt placeholder echo"
    if VLM_OPTION_LIST_ECHO_RE.search(text):
        return "prompt option-list echo"
    if VLM_GENERIC_FINDING_LABEL_RE.search(text):
        return "generic finding label"
    normalized = _normalize_for_quality(text)
    if normalized in {_normalize_for_quality(value) for value in VLM_GENERIC_RESPONSES}:
        return "generic output"
    if len(text) < 20:
        return "too short"
    if any(phrase.lower() in text.lower() for phrase in VLM_PROMPT_ECHO_PHRASES):
        return "prompt echo"
    fields: dict[str, str] = {}
    for name, value in VLM_FIELD_RE.findall(text):
        cleaned_value = re.sub(
            r"\b(?:visible_evidence|visual_status|short_reason|confidence_hint)\b\s*:?.*$",
            "",
            str(value),
            flags=re.IGNORECASE,
        )
        fields[str(name).lower()] = re.sub(r"\s+", " ", cleaned_value).strip(" .:-").lower()
    if fields:
        emptyish = {"", "no", "none", "n/a", "na", "unknown", "unclear"}
        evidence = fields.get("visible_evidence", "")
        reason = fields.get("short_reason", "")
        status = fields.get("visual_status", "")
        if evidence in emptyish and reason in emptyish:
            return "non-informative structured output"
        if len(reason) < 8 and status in {"occluded", "unclear", "not visible", "not clear"}:
            return "non-informative structured output"
    if not _contains_keyword(text, VLM_USEFUL_KEYWORDS):
        return "missing visible safety detail"
    has_safety_keyword = _contains_keyword(text, VLM_SAFETY_EVIDENCE_KEYWORDS)
    if VLM_GENERIC_INVENTORY_RE.search(text) and not has_safety_keyword:
        return "generic object inventory"
    if VLM_GENERIC_PERSON_ACTIVITY_RE.search(text) and not has_safety_keyword:
        return "generic person activity"
    if not has_safety_keyword:
        return "missing safety-specific detail"
    return None


def is_useful_vlm_output(clean_text: str) -> bool:
    return vlm_output_quality_issue(clean_text) is None


def _run_with_timeout(callable_obj, timeout_seconds: float):
    timeout = max(0.0, float(timeout_seconds or 0.0))
    if timeout <= 0:
        return callable_obj()

    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = callable_obj()
        except BaseException as exc:  # pragma: no cover - re-raised in caller thread
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"Local VLM generation exceeded {timeout:.1f}s.")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _is_vlm_model_file(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name.lower()
    return name in VLM_MODEL_FILENAMES or path.suffix.lower() in VLM_MODEL_SUFFIXES


def path_has_direct_vlm_model_files(path: Path) -> bool:
    if path.is_file():
        return _is_vlm_model_file(path)
    if not path.is_dir():
        return False
    try:
        return any(_is_vlm_model_file(child) for child in path.iterdir())
    except OSError:
        return False


def path_has_vlm_model_files(path: Path) -> bool:
    if path.is_file():
        return _is_vlm_model_file(path)
    if not path.is_dir():
        return False
    try:
        return any(_is_vlm_model_file(child) for child in path.rglob("*"))
    except OSError:
        return False


def _transformers_runtime_available() -> bool:
    return importlib_util.find_spec("transformers") is not None


def _local_transformer_available(model_dir: Path) -> bool:
    return _transformers_runtime_available() and path_has_direct_vlm_model_files(model_dir)


def is_local_vlm_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    return host in LOCAL_VLM_HOSTS


def _ollama_endpoint(path: str) -> str:
    base_url = SETTINGS.vlm_ollama_base_url.rstrip("/")
    return f"{base_url}/{path.lstrip('/')}"


def _ollama_model_names(payload: dict[str, Any]) -> list[str]:
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for model in models:
        if isinstance(model, dict) and model.get("name"):
            names.append(str(model["name"]))
    return names


def _model_name_matches(candidate: str, configured: str) -> bool:
    return candidate == configured or candidate.startswith(f"{configured}:")


def _base_details(*, mode: str, requested_provider: str) -> dict[str, Any]:
    return {
        "enabledMode": mode,
        "provider": requested_provider,
        "model": SETTINGS.vlm_model,
        "maxFrames": SETTINGS.vlm_max_frames,
        "maxEvidenceFrames": getattr(SETTINGS, "vlm_max_evidence_frames", SETTINGS.vlm_max_frames),
        "maxTokens": SETTINGS.vlm_max_tokens,
    }


def _local_provider_status(base_details: dict[str, Any]) -> dict[str, Any]:
    model_dir = Path(SETTINGS.vlm_model_dir)
    runtime_available = _transformers_runtime_available()
    model_dir_available = path_has_direct_vlm_model_files(model_dir)
    known_profile_parent = model_dir.is_dir() and any(
        (model_dir / child).is_dir() for child in ("lightweight-256m", "enhanced-2b")
    )
    details = {
        **base_details,
        "selectedProvider": LOCAL_PROVIDER,
        "providerType": "local_transformers",
        "modelDir": str(model_dir),
        "runtimeAvailable": runtime_available,
        "modelDirAvailable": model_dir_available,
        "profileParentDir": known_profile_parent,
    }
    if not runtime_available:
        return {
            "status": "missing_runtime",
            "path": str(model_dir),
            "message": "Local transformer VLM runtime is not available.",
            "actionHint": "Install the local VLM runtime in the developer environment, or choose another local provider.",
            "details": details,
        }
    if not model_dir_available:
        message = (
            "Local transformer VLM path is a profile parent directory, not a direct loadable model folder."
            if known_profile_parent
            else "Local transformer VLM model directory is missing required model files."
        )
        return {
            "status": "unavailable",
            "path": str(model_dir),
            "message": message,
            "actionHint": (
                "Select a packaged VLM profile such as models/vlm/lightweight-256m or models/vlm/enhanced-2b, "
                "or use rule-based fallback."
            ),
            "details": details,
        }
    return {
        "status": "available",
        "path": str(model_dir),
        "message": "VLM available via local provider.",
        "actionHint": None,
        "details": details,
    }


def _ollama_provider_status(base_details: dict[str, Any]) -> dict[str, Any]:
    base_url = SETTINGS.vlm_ollama_base_url.rstrip("/")
    details = {
        **base_details,
        "selectedProvider": OLLAMA_PROVIDER,
        "providerType": "ollama_vision",
        "baseUrl": base_url,
    }
    if not is_local_vlm_base_url(base_url):
        return {
            "status": "unavailable",
            "message": "VLM provider must point to a local Ollama runtime.",
            "actionHint": "Use SAFETRACE_VLM_OLLAMA_BASE_URL=http://127.0.0.1:11434.",
            "details": details,
        }
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=1.5)
        response.raise_for_status()
    except Exception:
        return {
            "status": "missing_runtime",
            "message": "VLM unavailable because Ollama is not reachable.",
            "actionHint": "Start Ollama locally only when SAFETRACE_VLM_PROVIDER=ollama or when using Ollama in auto mode.",
            "details": {**details, "runtimeAvailable": False},
        }

    try:
        tags_payload = response.json()
    except Exception:
        tags_payload = {}
    names = _ollama_model_names(tags_payload)
    model_available = any(_model_name_matches(name, SETTINGS.vlm_model) for name in names)
    if not model_available:
        return {
            "status": "unavailable",
            "message": f"Ollama is reachable, but the configured VLM model '{SETTINGS.vlm_model}' was not listed.",
            "actionHint": "Install a local Ollama vision model such as llava or set SAFETRACE_VLM_MODEL.",
            "details": {**details, "runtimeAvailable": True, "availableModels": names},
        }
    return {
        "status": "available",
        "message": f"VLM available via Ollama provider using '{SETTINGS.vlm_model}'.",
        "actionHint": None,
        "details": {**details, "runtimeAvailable": True, "availableModels": names},
    }


def _with_auto_summary(
    payload: dict[str, Any],
    *,
    requested_provider: str,
    selected_provider: str,
    available_providers: list[str],
    local_status: dict[str, Any],
    ollama_status: dict[str, Any],
) -> dict[str, Any]:
    details = {
        **payload.get("details", {}),
        "provider": requested_provider,
        "selectedProvider": selected_provider,
        "availableProviders": available_providers,
        "providers": {
            "local": {
                "status": local_status.get("status"),
                "message": local_status.get("message"),
                "details": local_status.get("details"),
            },
            "ollama": {
                "status": ollama_status.get("status"),
                "message": ollama_status.get("message"),
                "details": ollama_status.get("details"),
            },
        },
    }
    return {**payload, "details": details}


def vlm_status_payload(*, timeout_seconds: float = 1.5) -> dict[str, Any]:  # noqa: ARG001
    """Return a lightweight local VLM status payload without loading a model."""
    if bool(getattr(SETTINGS, "analysis_safe_mode", False)):
        return {
            "status": "disabled",
            "message": "Local runtime guard is active. VLM is not checked or loaded; Fast Local Analysis remains available.",
            "actionHint": "Unset SAFETRACE_ANALYSIS_SAFE_MODE to inspect or activate local VLM profiles.",
            "details": {
                "mode": "safe_mode",
                "requestedProvider": _normalized_provider(getattr(SETTINGS, "vlm_provider", AUTO_PROVIDER)),
                "selectedProvider": RULE_BASED_PROVIDER,
                "availableProviders": [],
                "vlmSuppressedReason": "safe_mode",
            },
        }

    mode = _normalized_mode(getattr(SETTINGS, "vlm_enabled", "auto"))
    requested_provider = _normalized_provider(getattr(SETTINGS, "vlm_provider", AUTO_PROVIDER))
    base_details = _base_details(mode=mode, requested_provider=requested_provider)

    if mode == "disabled":
        return {
            "status": "disabled",
            "message": "Enhanced VLM is disabled. Fast Local Analysis remains available.",
            "actionHint": "Set SAFETRACE_VLM_ENABLED=auto or enabled to use a local VLM.",
            "details": {
                **base_details,
                "selectedProvider": RULE_BASED_PROVIDER,
                "availableProviders": [],
            },
        }

    if requested_provider == LOCAL_PROVIDER:
        return _local_provider_status(base_details)
    if requested_provider == OLLAMA_PROVIDER:
        return _ollama_provider_status(base_details)
    if requested_provider != AUTO_PROVIDER:
        return {
            "status": "unavailable",
            "message": f"Unsupported local VLM provider '{requested_provider}'.",
            "actionHint": "Set SAFETRACE_VLM_PROVIDER=auto, local, or ollama.",
            "details": {**base_details, "selectedProvider": RULE_BASED_PROVIDER, "availableProviders": []},
        }

    local_status = _local_provider_status(base_details)
    ollama_status = _ollama_provider_status(base_details)
    available_providers = [
        provider
        for provider, status in (
            (LOCAL_PROVIDER, local_status),
            (OLLAMA_PROVIDER, ollama_status),
        )
        if status.get("status") == "available"
    ]
    if local_status.get("status") == "available":
        return _with_auto_summary(
            {
                **local_status,
                "message": "VLM available via local provider.",
            },
            requested_provider=AUTO_PROVIDER,
            selected_provider=LOCAL_PROVIDER,
            available_providers=available_providers,
            local_status=local_status,
            ollama_status=ollama_status,
        )
    if ollama_status.get("status") == "available":
        return _with_auto_summary(
            {
                **ollama_status,
                "message": "VLM available via Ollama provider.",
            },
            requested_provider=AUTO_PROVIDER,
            selected_provider=OLLAMA_PROVIDER,
            available_providers=available_providers,
            local_status=local_status,
            ollama_status=ollama_status,
        )
    return _with_auto_summary(
        {
            "status": "unavailable",
            "message": "VLM unavailable. SafeTrace will use rule-based explanations.",
            "actionHint": "Place packaged local VLM assets at models/vlm, or explicitly configure local Ollama.",
            "details": base_details,
        },
        requested_provider=AUTO_PROVIDER,
        selected_provider=RULE_BASED_PROVIDER,
        available_providers=available_providers,
        local_status=local_status,
        ollama_status=ollama_status,
    )


class VlmReasoner:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.device = resolve_device(device or SETTINGS.device)
        self.model_dir = Path(model_dir or SETTINGS.vlm_model_dir)
        self.requested_provider = _normalized_provider(getattr(SETTINGS, "vlm_provider", AUTO_PROVIDER))
        self.provider = RULE_BASED_PROVIDER
        self.enabled_mode = _normalized_mode(getattr(SETTINGS, "vlm_enabled", "auto"))
        request_enabled = SETTINGS.enable_vlm if enabled is None else enabled
        safe_mode = bool(getattr(SETTINGS, "analysis_safe_mode", False))
        self.enabled = bool(request_enabled) and self.enabled_mode != "disabled" and not safe_mode
        self._model = None
        self._processor = None
        self._loaded = False
        self.last_explanation_source = RULE_BASED_PROVIDER
        self.last_fallback_reason: str | None = None
        self.last_raw_vlm_text: str | None = None
        self.last_clean_vlm_text: str | None = None
        self.last_quality_issue: str | None = None

        if not self.enabled:
            if safe_mode:
                logger.info("VLM disabled by SafeTrace safe local mode.")
                return
            logger.info("VLM disabled by request or configuration.")
            return

        selected_provider = self._select_provider()
        if selected_provider == OLLAMA_PROVIDER:
            if not is_local_vlm_base_url(SETTINGS.vlm_ollama_base_url):
                logger.warning("VLM Ollama base URL is not local; using rule-based fallback.")
                self.enabled = False
                return
            self.provider = OLLAMA_PROVIDER
            self._loaded = True
            return

        if selected_provider == LOCAL_PROVIDER:
            self.provider = LOCAL_PROVIDER
            self._load_transformer_provider()
            return

        logger.info("No local VLM provider available; using rule-based fallback.")
        self.enabled = False

    # ------------------------------------------------------------------ #
    def _select_provider(self) -> str:
        if self.requested_provider == LOCAL_PROVIDER:
            return LOCAL_PROVIDER
        if self.requested_provider == OLLAMA_PROVIDER:
            return OLLAMA_PROVIDER
        if self.requested_provider != AUTO_PROVIDER:
            logger.warning("Unsupported VLM provider %s; using rule-based fallback.", self.requested_provider)
            return RULE_BASED_PROVIDER
        if _local_transformer_available(self.model_dir):
            return LOCAL_PROVIDER
        base_details = _base_details(mode=self.enabled_mode, requested_provider=AUTO_PROVIDER)
        if _ollama_provider_status(base_details).get("status") == "available":
            return OLLAMA_PROVIDER
        return RULE_BASED_PROVIDER

    def _load_transformer_provider(self) -> None:
        if not path_has_direct_vlm_model_files(self.model_dir):
            logger.warning("VLM model dir empty/missing at %s; disabling.", self.model_dir)
            self.enabled = False
            return
        try:
            import torch
            import transformers

            logger.info("Loading VLM from %s on %s", self.model_dir, self.device)
            self._processor = transformers.AutoProcessor.from_pretrained(
                str(self.model_dir), trust_remote_code=True, local_files_only=True
            )
            model_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            errors: list[str] = []
            for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
                model_class = getattr(transformers, class_name, None)
                if model_class is None:
                    continue
                try:
                    self._model = model_class.from_pretrained(str(self.model_dir), **model_kwargs).to(self.device).eval()
                    break
                except Exception as exc:  # pragma: no cover - optional runtime compatibility
                    errors.append(f"{class_name}: {exc}")
            if self._model is None:
                raise RuntimeError("; ".join(errors) or "No compatible transformers AutoModel class is available.")
            self._loaded = True
        except Exception as exc:  # pragma: no cover - optional path
            logger.warning("Failed to load local VLM (%s); falling back to rule-based explanation.", exc)
            self.enabled = False
            self.provider = RULE_BASED_PROVIDER

    def explain_violation(
        self,
        image: np.ndarray,
        violations: Sequence[Violation],
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        self.last_explanation_source = RULE_BASED_PROVIDER
        self.last_fallback_reason = None
        self.last_raw_vlm_text = None
        self.last_clean_vlm_text = None
        self.last_quality_issue = None
        if not self.enabled or not self._loaded:
            self.last_fallback_reason = "disabled_or_unloaded"
            return _fallback_explanation(violations)

        if self.provider == OLLAMA_PROVIDER:
            return self._explain_with_ollama(image, violations, context=context)
        if self.provider == LOCAL_PROVIDER:
            return self._explain_with_transformers(image, violations, context=context)
        return _fallback_explanation(violations)

    def _prompt_context(self, context: dict[str, Any] | None) -> str:
        if not context:
            return ""
        parts: list[str] = []
        region = context.get("imageRegion")
        if isinstance(region, dict):
            source = str(region.get("source") or "full_frame")
            labels = region.get("matchedLabels")
            label_text = ", ".join(str(label) for label in labels) if isinstance(labels, list) else ""
            if source == "detector_crop":
                parts.append(
                    "Image region: detector crop around relevant safety subject"
                    + (f" ({label_text})." if label_text else ".")
                )
            else:
                reason = str(region.get("reason") or "").replace("_", " ")
                parts.append(f"Image region: full frame{f' ({reason})' if reason else ''}.")
        detections = context.get("detections")
        if isinstance(detections, list) and detections:
            summary = []
            for item in detections[:6]:
                if isinstance(item, dict):
                    label = str(item.get("label") or item.get("rawLabel") or "object")
                    confidence = item.get("confidence")
                    if isinstance(confidence, (int, float)):
                        summary.append(f"{label} {float(confidence):.2f}")
                    else:
                        summary.append(label)
            if summary:
                parts.append(f"Detector context: {', '.join(summary)}.")
        analysis_context = context.get("analysisContext")
        if isinstance(analysis_context, dict):
            profile_label = analysis_context.get("profileLabel") or analysis_context.get("selectedUseCaseProfile")
            effective_query = analysis_context.get("effectiveQuery") or analysis_context.get("userQuery")
            finding = analysis_context.get("friendlyFindingName") or analysis_context.get("findingName")
            rule_support = analysis_context.get("ruleSupport")
            confidence_reason = analysis_context.get("confidenceReason")
            review_level = analysis_context.get("reviewLevel")
            if profile_label:
                parts.append(f"Selected use-case profile: {profile_label}.")
            if effective_query:
                parts.append(f"User query: {effective_query}.")
            if finding:
                parts.append(f"Rule finding under review: {finding}.")
            rule_bits = [
                f"review level {review_level}" if review_level else "",
                f"rule support {rule_support}" if rule_support else "",
                f"confidence reason {confidence_reason}" if confidence_reason else "",
            ]
            rule_text = "; ".join(bit for bit in rule_bits if bit)
            if rule_text:
                parts.append(f"Base finding context: {rule_text}.")
        return "\n".join(parts)

    def _prompt_for(self, violations: Sequence[Violation], context: dict[str, Any] | None = None) -> str:
        profile = str(getattr(SETTINGS, "vlm_profile", "rule_based") or "rule_based").strip().lower()
        names = ", ".join(v.name for v in violations) or "no obvious violations"
        context_text = self._prompt_context(context)
        context_block = f"\n{context_text}" if context_text else ""
        strict_intro = (
            "Look only at the image crop. Use visible evidence only. "
            "If the safety detail is unclear, say unclear. Do not list unrelated objects. "
            "Do not repeat the prompt. Keep every field short. "
            "Use the selected profile, user query, rule finding, and detector context only to focus the answer."
        )
        if profile in {"lightweight_256m", "lightweight_512m", "enhanced_2b", "enhanced_3b"}:
            lowered_names = names.lower()
            if "seatbelt" in lowered_names:
                return (
                    f"{strict_intro}{context_block}\n"
                    "Focus: torso area and possible belt path.\n"
                    "State whether a belt path is visible across the torso. If torso or belt path is occluded, say unclear.\n"
                    "Reply with exactly:\n"
                    "visible_evidence:\n"
                    "visual_status:\n"
                    "short_reason:\n"
                    "confidence_hint:"
                )
            if "helmet" in lowered_names or "ppe" in lowered_names:
                return (
                    f"{strict_intro}{context_block}\n"
                    "Focus: head and upper-body area.\n"
                    "State whether the head is visible and whether a helmet or PPE is visible. If the head is not visible, say unclear.\n"
                    "Reply with exactly:\n"
                    "visible_evidence:\n"
                    "visual_status:\n"
                    "short_reason:\n"
                    "confidence_hint:"
                )
            if "phone" in lowered_names or "distract" in lowered_names:
                return (
                    f"{strict_intro}{context_block}\n"
                    "Focus: hand, face, and driver area.\n"
                    "State whether a phone is visibly near a hand, face, or driver area. If active use is unclear, say unclear.\n"
                    "Reply with exactly:\n"
                    "visible_evidence:\n"
                    "visual_status:\n"
                    "short_reason:\n"
                    "confidence_hint:"
                )
            return (
                f"{strict_intro}{context_block}\n"
                f"Finding(s): {names}.\n"
                "State what visible safety evidence supports or weakens this finding.\n"
                "Reply with exactly:\n"
                "visible_evidence:\n"
                "visual_status:\n"
                "short_reason:\n"
                "confidence_hint:"
            )
        return f"{VLM_PROMPT}\nFindings to inspect: {names}."

    def _image_base64(self, image: np.ndarray) -> str:
        pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
        with io.BytesIO() as buffer:
            pil.convert("RGB").save(buffer, format="JPEG", quality=88)
            return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _processor_inputs_for_image(self, prompt_text: str, image: Image.Image):
        apply_chat_template = getattr(self._processor, "apply_chat_template", None)
        if callable(apply_chat_template):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            try:
                try:
                    tokenized = apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    if isinstance(tokenized, dict) or (
                        hasattr(tokenized, "to") and (hasattr(tokenized, "get") or hasattr(tokenized, "input_ids"))
                    ):
                        return tokenized
                except TypeError:
                    pass
                try:
                    prompt = apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                except TypeError:
                    prompt = apply_chat_template(messages, add_generation_prompt=True)
                if isinstance(prompt, list):
                    prompt = prompt[0] if prompt else ""
                if isinstance(prompt, str) and prompt.strip():
                    return self._processor(text=prompt, images=[image], return_tensors="pt")
            except Exception as exc:
                logger.debug(
                    "VLM chat-template prompt construction failed (%s); using image-token fallback.",
                    exc,
                )

        fallback_prompt = prompt_text if "<image>" in prompt_text else f"<image>\n{prompt_text}"
        try:
            return self._processor(text=fallback_prompt, images=[image], return_tensors="pt")
        except TypeError:
            return self._processor(text=fallback_prompt, images=image, return_tensors="pt")

    def _inputs_to_device(self, inputs):
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        if isinstance(inputs, dict):
            return {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        return inputs

    def _input_token_length(self, inputs) -> int | None:
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
        if input_ids is None:
            return None
        shape = getattr(input_ids, "shape", None)
        if shape is not None and len(shape):
            return int(shape[-1])
        if isinstance(input_ids, (list, tuple)):
            if input_ids and isinstance(input_ids[0], (list, tuple)):
                return len(input_ids[0])
            return len(input_ids)
        return None

    def _generated_tokens_only(self, output_ids, inputs):
        input_len = self._input_token_length(inputs)
        if input_len is None:
            return output_ids
        try:
            output_len = int(output_ids.shape[-1])
            if output_len > input_len:
                return output_ids[:, input_len:]
            return output_ids
        except Exception:
            pass
        if isinstance(output_ids, (list, tuple)):
            if output_ids and isinstance(output_ids[0], (list, tuple)):
                return [
                    list(row[input_len:]) if len(row) > input_len else list(row)
                    for row in output_ids
                ]
            return list(output_ids[input_len:]) if len(output_ids) > input_len else output_ids
        return output_ids

    def _accepted_vlm_text(self, raw_text: str, prompt_text: str, provider_label: str) -> str | None:
        self.last_raw_vlm_text = raw_text
        clean_text = sanitize_vlm_output(raw_text, prompt_text)
        self.last_clean_vlm_text = clean_text
        quality_issue = vlm_output_quality_issue(clean_text)
        self.last_quality_issue = quality_issue
        if quality_issue:
            self.last_fallback_reason = f"quality:{quality_issue}"
            logger.warning(
                "%s VLM output rejected (%s); using rule-based fallback.",
                provider_label,
                quality_issue,
            )
            return None
        return clean_text

    def _explain_with_ollama(
        self,
        image: np.ndarray,
        violations: Sequence[Violation],
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        try:
            prompt_text = self._prompt_for(violations, context=context)
            response = httpx.post(
                _ollama_endpoint("/api/generate"),
                json={
                    "model": SETTINGS.vlm_model,
                    "prompt": prompt_text,
                    "images": [self._image_base64(image)],
                    "stream": False,
                    "options": {
                        "num_predict": SETTINGS.vlm_max_tokens,
                        "temperature": 0.1,
                    },
                },
                timeout=SETTINGS.vlm_timeout_seconds,
            )
            response.raise_for_status()
            text = self._accepted_vlm_text(
                str(response.json().get("response") or ""),
                prompt_text,
                "Ollama",
            )
            if text:
                self.last_fallback_reason = None
                self.last_explanation_source = "vlm_ollama"
                return text
        except Exception as exc:  # pragma: no cover - exercised by fallback tests
            self.last_fallback_reason = f"ollama_error:{type(exc).__name__}"
            logger.warning("Ollama VLM generation failed (%s); using rule-based fallback.", exc)
        return _fallback_explanation(violations)

    def _explain_with_transformers(
        self,
        image: np.ndarray,
        violations: Sequence[Violation],
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        try:
            def generate_text():
                import torch

                pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
                prompt_text = self._prompt_for(violations, context=context)
                inputs = self._inputs_to_device(
                    self._processor_inputs_for_image(prompt_text, pil)
                )
                with torch.inference_mode():
                    output_ids = self._model.generate(
                        **inputs, max_new_tokens=SETTINGS.vlm_max_tokens, do_sample=False
                    )
                new_token_ids = self._generated_tokens_only(output_ids, inputs)
                raw_text = self._processor.batch_decode(
                    new_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )[0]
                return raw_text, prompt_text

            text, prompt_text = _run_with_timeout(generate_text, SETTINGS.vlm_timeout_seconds)
            text = self._accepted_vlm_text(text, prompt_text, "Local")
            if text:
                self.last_fallback_reason = None
                self.last_explanation_source = "vlm_local"
                return text
        except TimeoutError as exc:  # pragma: no cover
            self.last_fallback_reason = "generation_timeout"
            logger.warning("Local VLM generation failed (%s); using rule-based fallback.", exc)
        except Exception as exc:  # pragma: no cover
            self.last_fallback_reason = f"generation_error:{type(exc).__name__}"
            logger.warning("Local VLM generation failed (%s); using rule-based fallback.", exc)
        return _fallback_explanation(violations)


class RuleBasedReasoner:
    """Rule-based explanation provider used when VLM is disabled or inactive."""

    provider = RULE_BASED_PROVIDER
    enabled = False
    _loaded = False

    def __init__(self) -> None:
        self.last_explanation_source = RULE_BASED_PROVIDER

    def explain_violation(self, image: np.ndarray, violations: Sequence[Violation]) -> str:  # noqa: ARG002
        self.last_explanation_source = RULE_BASED_PROVIDER
        return _fallback_explanation(violations)
