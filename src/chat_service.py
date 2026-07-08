"""Optional SafeTrace-only assistant service."""
from __future__ import annotations

import importlib
import importlib.util
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from src.chat_guardrails import is_safetrace_question, safetrace_refusal
from src.config import SETTINGS


SAFETRACE_HELP_TEXT = """
SafeTrace is a local safety-video review tool. It accepts single image/video
uploads and ZIP or multi-video batches through the FastAPI backend. Results
contain an analysis summary, grouped violation events, evidence frames, and
technical JSON. Automated findings are evidence to review, not final legal or
operational truth. Confidence depends on video quality, sampled frames, detector
coverage, and configured local models.

Common UI areas:
- Analysis Summary explains the overall decision, confidence, grouped events,
  supporting frame count, sampling settings, and review guidance.
- Video Violation Overview groups findings by violation type and links to
  supporting evidence frames.
- Evidence Frames show annotated frame media, frame findings, optional VLM
  visual explanations, and collapsed technical evidence.
- The VLM explanation describes detected evidence for a result or frame. It is
  separate from SafeTrace Assistant, which is interactive and can answer usage,
  interpretation, troubleshooting, API, and selected-result questions.

Upload and export behavior:
- Single-video or image analysis is submitted with POST /api/analyze.
- ZIP or multi-video batch analysis is submitted with POST /api/batches/analyze.
- Job status is available at GET /api/jobs/{job_id}; completed results at
  GET /api/jobs/{job_id}/result.
- Batch status is available at GET /api/batches/{batch_id}.
- Technical JSON is downloaded from GET /api/reports/{job_id}/technical-json.
- Evidence media is served from GET /api/media/{job_id}/{filename}.

Troubleshooting:
- If the backend is unavailable, start the FastAPI app and retry the frontend
  connection. SafeTrace analysis requires the backend outside preview mode.
- If SafeTrace Assistant says missing_model, the packaged GGUF is not installed
  at the configured model path.
- If it says missing_runtime, install the llama-cpp-python runtime.
- Ollama is optional and can be selected with SAFETRACE_CHAT_PROVIDER=ollama.
- Main SafeTrace analysis still works when chat is disabled or unavailable.

Developer locations:
- React shell: frontend-react/src/App.tsx.
- Floating assistant UI: frontend-react/src/components/SafeTraceAssistant.tsx.
- Chat frontend client/types: frontend-react/src/services/chatService.ts and
  frontend-react/src/types/chat.ts.
- FastAPI routes: src/api/server.py.
- Chat provider and context: src/chat_service.py.
- Chat scope guardrails: src/chat_guardrails.py.
- Batch upload backend: src/api/batches.py and the /api/batches/analyze route.
- Job queue and analysis execution: src/api/jobs.py.
""".strip()


class ChatDisabledError(RuntimeError):
    pass


class ChatProviderUnavailableError(RuntimeError):
    pass


DEFAULT_PACKAGED_MODEL_PATH = Path("models") / "chat" / "safetrace-assistant-qwen2.5-1.5b-instruct-q4.gguf"
CHAT_DISABLED_VALUES = {"0", "false", "no", "n", "off", "disabled"}
CHAT_TRUE_VALUES = {"1", "true", "yes", "y", "on", "enabled"}
CHAT_ENABLE_HINT = (
    "Restart backend with SAFETRACE_CHAT_ENABLED=auto or SAFETRACE_CHAT_ENABLED=true."
)
PACKAGED_MODEL_HINT = (
    "Place the GGUF model at models/chat/safetrace-assistant-qwen2.5-1.5b-instruct-q4.gguf."
)
PACKAGED_RUNTIME_HINT = (
    "Install llama-cpp-python in the SafeTrace virtual environment with "
    ".venv\\Scripts\\python.exe -m pip install llama-cpp-python, then restart the backend with "
    ".venv\\Scripts\\python.exe -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --log-level info."
)
PACKAGED_RUNTIME_SETUP_COMMAND = ".venv\\Scripts\\python.exe -m pip install llama-cpp-python"
PACKAGED_RUNTIME_RESTART_REQUIRED = "Restart the SafeTrace backend after installing llama-cpp-python."

_PACKAGED_MODEL: Any | None = None
_PACKAGED_MODEL_PATH: Path | None = None
_PACKAGED_MODEL_LOCK = threading.Lock()
_PACKAGED_MODEL_LOADING = False


@dataclass
class ChatContext:
    text: str
    sources: list[str]


def chat_status_payload(*, allow_model_load: bool = True) -> Dict[str, Any]:
    provider = _configured_provider()
    enabled, mode = _chat_enabled()
    if not enabled:
        model_path = _packaged_model_path() if provider == "packaged_llamacpp" else None
        runtime_diagnostics = _llama_cpp_runtime_diagnostics() if provider == "packaged_llamacpp" else None
        reason = f"SafeTrace Assistant disabled by SAFETRACE_CHAT_ENABLED={mode}."
        return _status_payload(
            state="disabled",
            enabled=False,
            available=False,
            enabled_mode=mode,
            provider=provider,
            model=_provider_model_name(provider),
            model_path=_display_model_path(model_path) if model_path else None,
            model_exists=model_path.is_file() if model_path else None,
            runtime_available=runtime_diagnostics["importOk"] if runtime_diagnostics else None,
            runtime_diagnostics=runtime_diagnostics,
            message=reason,
            reason=reason,
            action_hint=CHAT_ENABLE_HINT,
        )
    if provider == "mock":
        return _status_payload(
            state="available",
            enabled=True,
            available=True,
            enabled_mode=mode,
            provider=provider,
            model="mock",
            message="SafeTrace Assistant is using the test provider.",
            reason="Mock chat provider is configured for tests or preview.",
            action_hint=None,
        )
    if provider == "packaged_llamacpp":
        return _packaged_status_payload(mode, allow_model_load=allow_model_load)
    if provider == "ollama":
        return _ollama_status_payload(mode)
    reason = f"Unsupported chat provider: {provider}"
    return _status_payload(
        state="unavailable",
        enabled=True,
        available=False,
        enabled_mode=mode,
        provider=provider,
        model=None,
        message=reason,
        reason=reason,
        action_hint="Set SAFETRACE_CHAT_PROVIDER=packaged_llamacpp or SAFETRACE_CHAT_PROVIDER=ollama.",
    )


def _status_payload(
    *,
    state: str,
    enabled: bool,
    available: bool,
    enabled_mode: str,
    provider: str,
    model: Optional[str],
    message: str,
    reason: Optional[str],
    action_hint: Optional[str],
    model_path: Optional[str] = None,
    model_exists: Optional[bool] = None,
    runtime_available: Optional[bool] = None,
    fallback_available: bool = False,
    fallback_label: Optional[str] = None,
    runtime_diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    diagnostics = runtime_diagnostics or {}
    return {
        "enabled": enabled,
        "available": available,
        "state": state,
        "status": state,
        "enabled_mode": enabled_mode,
        "provider": provider,
        "model": model,
        "model_path": model_path,
        "model_exists": model_exists,
        "runtime_available": runtime_available,
        "speed_profile": _speed_profile(),
        "warmup_on_open": bool(getattr(SETTINGS, "chat_warmup_on_open", False)),
        "fallback_available": fallback_available,
        "fallback_label": fallback_label,
        "runtime_diagnostics": diagnostics or None,
        "python_executable": diagnostics.get("backendPythonExecutable"),
        "expected_venv_python": diagnostics.get("expectedVenvPython"),
        "running_in_expected_venv": diagnostics.get("runningInExpectedVenv"),
        "llama_cpp_import_status": diagnostics.get("importStatus"),
        "llama_cpp_spec_found": diagnostics.get("specFound"),
        "llama_cpp_import_error_type": diagnostics.get("importErrorType"),
        "llama_cpp_import_error_message": diagnostics.get("importErrorMessage"),
        "setup_command": diagnostics.get("setupCommand"),
        "restart_required": diagnostics.get("restartRequired"),
        "message": message,
        "reason": reason or message,
        "action_hint": action_hint,
    }


def build_chat_context(
    *,
    job_store,
    batch_store,
    message: str = "",
    job_id: Optional[str],
    batch_id: Optional[str],
    include_current_result: bool,
) -> ChatContext:
    parts = [f"SafeTrace help:\n{SAFETRACE_HELP_TEXT}"]
    sources = ["docs"]

    if job_id and include_current_result and _question_needs_result_context(message):
        record = job_store.get(job_id)
        if record is not None:
            sources.append("job_result")
            parts.append(_summarize_job(record, include_technical=_question_asks_technical(message)))
        else:
            parts.append(f"Selected job {job_id} was not found.")

    if batch_id:
        batch = batch_store.get(batch_id, job_store)
        if batch is not None:
            sources.append("batch_manifest")
            parts.append(_summarize_batch(batch))
        else:
            parts.append(f"Selected batch {batch_id} was not found.")

    return ChatContext(text=_truncate_context("\n\n".join(parts)), sources=sources)


def answer_chat(
    *,
    message: str,
    job_store,
    batch_store,
    job_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    include_current_result: bool = True,
) -> Dict[str, Any]:
    provider = _configured_provider()
    enabled, mode = _chat_enabled()
    if not enabled:
        raise ChatDisabledError(f"SafeTrace Assistant disabled by SAFETRACE_CHAT_ENABLED={mode}. {CHAT_ENABLE_HINT}")

    has_context = bool(job_id or batch_id)
    if not is_safetrace_question(message, has_context=has_context):
        return {
            "answer": safetrace_refusal(),
            "sources": [],
            "safeTraceOnly": True,
            "modelProvider": provider,
        }

    selected_record = job_store.get(job_id) if job_id and include_current_result else None
    selected_batch = batch_store.get(batch_id, job_store) if batch_id else None

    result_answer = _result_aware_answer(message=message, record=selected_record)
    if result_answer:
        sources = ["docs"]
        if selected_record is not None and _record_result(selected_record):
            sources.append("job_result")
        return {
            "answer": _postprocess_answer(result_answer),
            "sources": sources,
            "safeTraceOnly": True,
            "modelProvider": provider,
        }

    provider_ready = True
    provider_error: ChatProviderUnavailableError | None = None
    try:
        _ensure_provider_ready(provider)
    except ChatProviderUnavailableError as exc:
        if not _deterministic_fallback_available(provider):
            raise
        provider_ready = False
        provider_error = exc

    templated_answer = _template_answer(message=message, record=selected_record, batch=selected_batch)
    if templated_answer:
        sources = ["docs"]
        if selected_record is not None and _question_needs_result_context(message):
            sources.append("job_result")
        if selected_batch is not None:
            sources.append("batch_manifest")
        return {
            "answer": _postprocess_answer(templated_answer),
            "sources": sources,
            "safeTraceOnly": True,
            "modelProvider": provider if provider_ready else _fallback_provider_name(provider),
        }

    if not provider_ready:
        return {
            "answer": _postprocess_answer(
                _limited_deterministic_fallback_answer(message=message, provider_error=provider_error)
            ),
            "sources": ["docs"],
            "safeTraceOnly": True,
            "modelProvider": _fallback_provider_name(provider),
        }

    context = build_chat_context(
        job_store=job_store,
        batch_store=batch_store,
        message=message,
        job_id=job_id,
        batch_id=batch_id,
        include_current_result=include_current_result,
    )

    if provider == "mock":
        answer = _mock_answer(message=message, context=context)
    elif provider == "packaged_llamacpp":
        answer = _packaged_llamacpp_answer(message=message, context=context)
    elif provider == "ollama":
        answer = _ollama_answer(message=message, context=context)
    else:
        raise ChatProviderUnavailableError(f"Unsupported chat provider: {provider}")

    return {
        "answer": _postprocess_answer(answer),
        "sources": context.sources,
        "safeTraceOnly": True,
        "modelProvider": provider,
    }


def _deterministic_fallback_available(provider: str) -> bool:
    return (
        provider == "packaged_llamacpp"
        and _packaged_model_path().is_file()
        and not _llama_cpp_runtime_available()
    )


def _fallback_provider_name(provider: str) -> str:
    return f"{provider}_deterministic_fallback"


def _limited_deterministic_fallback_answer(
    *,
    message: str,  # noqa: ARG001
    provider_error: ChatProviderUnavailableError | None,
) -> str:
    detail = str(provider_error or "The packaged llama.cpp runtime is not available.")
    return f"""
Limited SafeTrace help is available, but the packaged local chat model is not running.

- Runtime status: {detail}
- Analysis, uploads, batch jobs, exports, and rule-based explanations still work.
- For full local assistant answers, install the runtime with `.venv\\Scripts\\python.exe -m pip install llama-cpp-python`, then restart the backend.
- I can still answer built-in SafeTrace usage questions such as confidence, ZIP upload, evidence frames, exports, progress, and assistant setup.

Next: use `/api/chat/status` to confirm `runtime_available`, then retry after installing the runtime.
""".strip()


def _chat_enabled() -> tuple[bool, str]:
    raw = getattr(SETTINGS, "chat_enabled", "auto")
    if isinstance(raw, bool):
        return raw, "true" if raw else "false"
    mode = str(raw or "auto").strip().lower()
    if mode in CHAT_DISABLED_VALUES:
        return False, mode
    if mode in CHAT_TRUE_VALUES or mode == "auto":
        return True, mode
    return True, mode


def _speed_profile() -> str:
    return str(getattr(SETTINGS, "chat_speed_profile", "balanced") or "balanced").strip().lower()


def _summary_limits() -> tuple[int, int, int, int]:
    if _speed_profile() == "fast":
        return 3, 3, 3, 4
    return 5, 5, 6, 8


def _context_char_limit() -> int:
    return 6000 if _speed_profile() == "fast" else 10000


def _truncate_context(text: str) -> str:
    limit = _context_char_limit()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[Context truncated for the configured SafeTrace Assistant speed profile.]"


def _configured_provider() -> str:
    return str(getattr(SETTINGS, "chat_provider", "") or "packaged_llamacpp").strip().lower()


def _provider_model_name(provider: str) -> Optional[str]:
    if provider == "ollama":
        return SETTINGS.ollama_model
    if provider == "packaged_llamacpp":
        return _display_model_path(_packaged_model_path())
    if provider == "mock":
        return "mock"
    return None


def _display_model_path(path: Path) -> str:
    configured_path = Path(getattr(SETTINGS, "chat_model_path", DEFAULT_PACKAGED_MODEL_PATH))
    if configured_path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(SETTINGS.project_root))
    except ValueError:
        return str(path)


def _packaged_model_path() -> Path:
    raw_path = Path(getattr(SETTINGS, "chat_model_path", DEFAULT_PACKAGED_MODEL_PATH))
    if raw_path.is_absolute():
        return raw_path
    return SETTINGS.project_root / raw_path


def _expected_venv_python() -> Path:
    if sys.platform.startswith("win"):
        return SETTINGS.project_root / ".venv" / "Scripts" / "python.exe"
    return SETTINGS.project_root / ".venv" / "bin" / "python"


def _same_resolved_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left).lower() == str(right).lower()


def _llama_cpp_runtime_diagnostics() -> Dict[str, Any]:
    backend_python = Path(sys.executable)
    expected_python = _expected_venv_python()
    spec_found = importlib.util.find_spec("llama_cpp") is not None
    import_ok = False
    import_error_type: str | None = None
    import_error_message: str | None = None

    try:
        importlib.import_module("llama_cpp")
        import_ok = True
    except Exception as exc:  # pragma: no cover - native DLL failures are environment-specific
        import_error_type = type(exc).__name__
        import_error_message = str(exc) or repr(exc)

    if import_ok:
        import_status = "ok"
    elif spec_found:
        import_status = "import_error"
    else:
        import_status = "missing"

    return {
        "backendPythonExecutable": str(backend_python),
        "expectedVenvPython": str(expected_python),
        "expectedVenvPythonExists": expected_python.is_file(),
        "runningInExpectedVenv": expected_python.is_file() and _same_resolved_path(backend_python, expected_python),
        "specFound": spec_found,
        "importOk": import_ok,
        "importStatus": import_status,
        "importErrorType": import_error_type,
        "importErrorMessage": import_error_message,
        "setupCommand": PACKAGED_RUNTIME_SETUP_COMMAND,
        "restartRequired": PACKAGED_RUNTIME_RESTART_REQUIRED,
    }


def _llama_cpp_runtime_available() -> bool:
    return bool(_llama_cpp_runtime_diagnostics()["importOk"])


def _llama_cpp_unavailable_reason(diagnostics: Dict[str, Any]) -> str:
    if diagnostics.get("importStatus") == "import_error":
        error_type = diagnostics.get("importErrorType") or "ImportError"
        error_message = diagnostics.get("importErrorMessage") or "No import error message was provided."
        return (
            "llama-cpp-python is visible to this backend Python, but importing llama_cpp failed "
            f"with {error_type}: {error_message}. This is often a native DLL/runtime issue."
        )
    if diagnostics.get("expectedVenvPythonExists") and not diagnostics.get("runningInExpectedVenv"):
        return (
            "llama_cpp is not importable from the Python running this backend. The repo .venv exists, "
            "so start the backend with .venv\\Scripts\\python.exe and restart after installing llama-cpp-python."
        )
    return (
        "Packaged chat runtime is missing. Limited deterministic SafeTrace help is available, "
        "but full local model chat requires llama-cpp-python."
    )


def _packaged_status_payload(enabled_mode: str, *, allow_model_load: bool = True) -> Dict[str, Any]:
    model_path = _packaged_model_path()
    model_name = _display_model_path(model_path)
    model_exists = model_path.is_file()
    runtime_diagnostics = _llama_cpp_runtime_diagnostics()
    runtime_available = bool(runtime_diagnostics["importOk"])
    if not model_exists:
        reason = f"Packaged chat model is missing at {model_name}."
        return _status_payload(
            state="missing_model",
            enabled=True,
            available=False,
            enabled_mode=enabled_mode,
            provider="packaged_llamacpp",
            model=model_name,
            model_path=model_name,
            model_exists=False,
            runtime_available=runtime_available,
            runtime_diagnostics=runtime_diagnostics,
            message=reason,
            reason=reason,
            action_hint=PACKAGED_MODEL_HINT,
        )
    if not runtime_available:
        reason = _llama_cpp_unavailable_reason(runtime_diagnostics)
        return _status_payload(
            state="missing_runtime",
            enabled=True,
            available=False,
            enabled_mode=enabled_mode,
            provider="packaged_llamacpp",
            model=model_name,
            model_path=model_name,
            model_exists=True,
            runtime_available=False,
            runtime_diagnostics=runtime_diagnostics,
            message=reason,
            reason=reason,
            action_hint=PACKAGED_RUNTIME_HINT,
            fallback_available=True,
            fallback_label="Limited SafeTrace help",
        )
    if _PACKAGED_MODEL_LOADING:
        return _status_payload(
            state="loading",
            enabled=True,
            available=False,
            enabled_mode=enabled_mode,
            provider="packaged_llamacpp",
            model=model_name,
            model_path=model_name,
            model_exists=True,
            runtime_available=True,
            runtime_diagnostics=runtime_diagnostics,
            message="Packaged SafeTrace Assistant model is loading.",
            reason="The local GGUF is present and llama-cpp runtime is loading it.",
            action_hint="Wait a moment, then retry the assistant.",
        )
    if allow_model_load and bool(getattr(SETTINGS, "chat_autoload", False)) and _PACKAGED_MODEL is None:
        try:
            _get_packaged_model()
        except ChatProviderUnavailableError as exc:
            return _status_payload(
                state="unavailable",
                enabled=True,
                available=False,
                enabled_mode=enabled_mode,
                provider="packaged_llamacpp",
                model=model_name,
                model_path=model_name,
                model_exists=True,
                runtime_available=True,
                runtime_diagnostics=runtime_diagnostics,
                message=str(exc),
                reason=str(exc),
                action_hint="Check the configured GGUF file and llama-cpp-python installation.",
            )
    return _status_payload(
        state="available",
        enabled=True,
        available=True,
        enabled_mode=enabled_mode,
        provider="packaged_llamacpp",
        model=model_name,
        model_path=model_name,
        model_exists=True,
        runtime_available=True,
        runtime_diagnostics=runtime_diagnostics,
        message="SafeTrace Assistant packaged local model is available.",
        reason="Packaged model file and llama-cpp runtime are available.",
        action_hint=None,
    )


def warmup_chat_provider() -> Dict[str, Any]:
    provider = _configured_provider()
    enabled, mode = _chat_enabled()
    if not enabled:
        raise ChatDisabledError(f"SafeTrace Assistant disabled by SAFETRACE_CHAT_ENABLED={mode}. {CHAT_ENABLE_HINT}")
    if not bool(getattr(SETTINGS, "chat_warmup_on_open", False)):
        return chat_status_payload(allow_model_load=False)
    if provider == "mock":
        return chat_status_payload(allow_model_load=False)
    if provider == "packaged_llamacpp":
        _ensure_provider_ready(provider)
        _get_packaged_model()
        return chat_status_payload(allow_model_load=False)
    if provider == "ollama":
        _ensure_provider_ready(provider)
        return chat_status_payload(allow_model_load=False)
    raise ChatProviderUnavailableError(f"Unsupported chat provider: {provider}")


def _ollama_status_payload(enabled_mode: str) -> Dict[str, Any]:
    try:
        response = httpx.get(
            f"{SETTINGS.ollama_base_url.rstrip('/')}/api/tags",
            timeout=min(float(SETTINGS.chat_timeout_seconds), 3.0),
        )
        response.raise_for_status()
    except Exception:
        reason = "Ollama is not reachable. Start Ollama or choose the packaged local provider."
        return _status_payload(
            state="unavailable",
            enabled=True,
            available=False,
            enabled_mode=enabled_mode,
            provider="ollama",
            model=SETTINGS.ollama_model,
            message=reason,
            reason=reason,
            action_hint="Start Ollama or set SAFETRACE_CHAT_PROVIDER=packaged_llamacpp.",
        )
    return _status_payload(
        state="available",
        enabled=True,
        available=True,
        enabled_mode=enabled_mode,
        provider="ollama",
        model=SETTINGS.ollama_model,
        message="SafeTrace Assistant is available.",
        reason="Ollama responded to the local tags endpoint.",
        action_hint=None,
    )


def _ensure_provider_ready(provider: str) -> None:
    if provider == "mock":
        return
    if provider == "packaged_llamacpp":
        model_path = _packaged_model_path()
        if not model_path.is_file():
            raise ChatProviderUnavailableError(
                f"Packaged chat model is missing at {_display_model_path(model_path)}. {PACKAGED_MODEL_HINT}"
            )
        if not _llama_cpp_runtime_available():
            raise ChatProviderUnavailableError(
                f"Packaged chat runtime is missing. {PACKAGED_RUNTIME_HINT}"
            )
        if _PACKAGED_MODEL_LOADING:
            raise ChatProviderUnavailableError("Packaged SafeTrace Assistant model is loading. Retry shortly.")
        return
    if provider == "ollama":
        try:
            response = httpx.get(
                f"{SETTINGS.ollama_base_url.rstrip('/')}/api/tags",
                timeout=min(float(SETTINGS.chat_timeout_seconds), 3.0),
            )
            response.raise_for_status()
        except Exception as exc:
            raise ChatProviderUnavailableError(
                "Ollama is not reachable. Start Ollama or choose the packaged local provider."
            ) from exc
        return
    raise ChatProviderUnavailableError(f"Unsupported chat provider: {provider}")


def _summarize_job(record, *, include_technical: bool = False) -> str:
    event_limit, violation_limit, frame_limit, technical_limit = _summary_limits()
    lines = [
        f"Job {record.job_id}:",
        f"- status: {record.status}",
        f"- media: {record.original_filename}",
        f"- query: {record.query}",
    ]
    if record.result:
        summary = dict(record.result.get("summary") or {})
        lines.extend(
            [
                f"- frames analyzed: {summary.get('framesAnalyzed', 0)}",
                f"- frames with findings: {summary.get('framesWithViolations', 0)}",
                f"- violation types: {summary.get('uniqueViolationTypes', 0)}",
                f"- grouped events: {summary.get('potentialEventCount', 0)}",
                f"- overall confidence: {summary.get('overallConfidence', 'unknown')}",
            ]
        )
        events = list(record.result.get("events") or [])
        for event in events[:event_limit]:
            lines.append(
                f"- event {event.get('name', event.get('type', 'Finding'))}: "
                f"{event.get('severity', 'unknown')} severity from "
                f"{event.get('startTimestamp', 'unknown')} to {event.get('endTimestamp', 'unknown')}, "
                f"representative confidence {event.get('representativeConfidence', 'unknown')}, "
                f"supporting frames {event.get('supportingFrameCount', 'unknown')}"
            )
        for violation in list(record.result.get("violations") or [])[:violation_limit]:
            frames = ", ".join(frame.get("timestamp", "unknown") for frame in violation.get("affectedFrames", [])[:5])
            lines.append(
                f"- {violation.get('name', violation.get('id', 'Finding'))}: "
                f"{violation.get('severity', 'unknown')} severity, "
                f"confidence {violation.get('confidenceMax', 'unknown')}, frames {frames}"
            )
        for frame in list(record.result.get("frames") or [])[:frame_limit]:
            finding_names = ", ".join(v.get("name", v.get("id", "finding")) for v in frame.get("violations", [])[:4])
            lines.append(
                f"- frame {frame.get('frameNumber', '?')} at {frame.get('timestamp', 'unknown')}: "
                f"status {frame.get('status', 'unknown')}, "
                f"query relevance {frame.get('queryRelevance', 'unknown')}, "
                f"findings {finding_names or 'none'}"
            )
        technical = dict(record.result.get("technicalDetails") or {})
        if include_technical and technical:
            lines.append(f"- technical JSON sections: {', '.join(list(technical.keys())[:technical_limit])}")
    elif record.error:
        lines.append(f"- error: {record.error}")
    return "\n".join(lines)


def _summarize_batch(batch) -> str:
    accepted = batch.accepted_files
    rejected = batch.rejected_files
    lines = [
        f"Batch {batch.batch_id}:",
        f"- status: {batch.status}",
        f"- source upload: {batch.source_filename}",
        f"- accepted videos: {len(accepted)}",
        f"- rejected files: {len(rejected)}",
        f"- status counts: {batch.status_counts}",
    ]
    for item in accepted[:8]:
        lines.append(f"- accepted: {item.filename}, job {item.job_id}, status {item.status}")
    for item in rejected[:8]:
        lines.append(f"- rejected: {item.filename}, reason {item.reason}")
    return "\n".join(lines)


def _template_answer(*, message: str, record=None, batch=None) -> Optional[str]:
    question = _normalized_question(message)
    if _is_batch_implementation_question(question):
        return _developer_batch_upload_answer()
    if _is_zip_upload_question(question):
        return _zip_upload_answer()
    if _is_assistant_unavailable_question(question):
        return _assistant_unavailable_answer()
    if _is_confidence_question(question):
        return _confidence_answer()
    if _is_supporting_frames_question(question):
        return _supporting_frames_answer(record)
    if _is_explain_result_question(question):
        return _explain_result_answer(record)
    return None


def _result_aware_answer(*, message: str, record=None) -> Optional[str]:
    question = _normalized_question(message)
    if not _is_result_aware_question(question):
        return None

    result = _record_result(record)
    if not result:
        return _no_selected_result_answer()

    if _is_phone_question(question):
        return _specific_violation_answer(record, result, labels=("phone", "mobile", "cell phone"))
    if _is_seatbelt_question(question):
        return _specific_violation_answer(record, result, labels=("seatbelt", "seat belt", "belt"))
    if _is_helmet_question(question):
        return _specific_violation_answer(record, result, labels=("helmet", "hard hat", "hardhat"))
    if _is_safe_or_unsafe_question(question):
        return _safe_or_unsafe_answer(record, result)
    if _is_frames_analyzed_question(question):
        return _frames_analyzed_answer(record, result)
    if _is_frame_or_timestamp_question(question):
        return _supporting_frames_answer(record)
    if _is_result_confidence_question(question):
        return _result_confidence_answer(record, result)
    if _is_violations_overview_question(question):
        return _violations_detected_answer(record, result)
    return None


def _normalized_question(message: str) -> str:
    return re.sub(r"\s+", " ", message.strip().lower())


def _is_explain_result_question(question: str) -> bool:
    return "result" in question and any(word in question for word in ("explain", "summarize", "summary"))


def _is_supporting_frames_question(question: str) -> bool:
    return "frame" in question and any(word in question for word in ("support", "supporting", "top finding", "evidence"))


def _is_result_aware_question(question: str) -> bool:
    if _is_zip_upload_question(question) or _is_batch_implementation_question(question):
        return False
    if _is_confidence_question(question) and "mean" in question:
        return False
    return (
        _is_seatbelt_question(question)
        or _is_helmet_question(question)
        or _is_phone_question(question)
        or _is_safe_or_unsafe_question(question)
        or _is_frames_analyzed_question(question)
        or _is_frame_or_timestamp_question(question)
        or _is_result_confidence_question(question)
        or _is_violations_overview_question(question)
    )


def _is_seatbelt_question(question: str) -> bool:
    return "seatbelt" in question or "seat belt" in question


def _is_helmet_question(question: str) -> bool:
    return "helmet" in question or "hard hat" in question or "hardhat" in question


def _is_phone_question(question: str) -> bool:
    return "phone" in question or "mobile" in question or "cell phone" in question


def _is_safe_or_unsafe_question(question: str) -> bool:
    return bool(re.search(r"\b(?:safe|unsafe)\b", question))


def _is_frames_analyzed_question(question: str) -> bool:
    return "frame" in question and any(term in question for term in ("analyzed", "analysed", "sampled", "checked"))


def _is_frame_or_timestamp_question(question: str) -> bool:
    if _is_supporting_frames_question(question):
        return True
    if "timestamp" in question or "time" in question:
        return any(term in question for term in ("violation", "finding", "occur", "happen", "detected"))
    return "frame" in question and any(term in question for term in ("violation", "finding", "detected", "had"))


def _is_result_confidence_question(question: str) -> bool:
    return "confidence" in question and "mean" not in question


def _is_violations_overview_question(question: str) -> bool:
    return any(term in question for term in ("violation", "violations", "detected", "findings", "finding"))


def _is_zip_upload_question(question: str) -> bool:
    return "zip" in question and any(word in question for word in ("upload", "batch", "archive"))


def _is_confidence_question(question: str) -> bool:
    return "confidence" in question and any(word in question for word in ("mean", "overall", "explain", "what"))


def _is_batch_implementation_question(question: str) -> bool:
    return "batch" in question and any(word in question for word in ("implemented", "file", "function", "code", "where"))


def _is_assistant_unavailable_question(question: str) -> bool:
    return "assistant" in question and any(
        word in question
        for word in ("unavailable", "disabled", "missing", "runtime", "model", "not working", "offline")
    )


def _question_needs_result_context(message: str) -> bool:
    question = _normalized_question(message)
    if _is_zip_upload_question(question) or _is_batch_implementation_question(question):
        return False
    return any(
        word in question
        for word in (
            "result",
            "finding",
            "frame",
            "evidence",
            "violation",
            "confidence",
            "seatbelt",
            "seat belt",
            "helmet",
            "phone",
            "timestamp",
            "unsafe",
        )
    )


def _question_asks_technical(message: str) -> bool:
    question = _normalized_question(message)
    return any(word in question for word in ("technical", "json", "debug", "export", "report", "api"))


def _zip_upload_answer() -> str:
    return """
To upload a ZIP batch in the frontend:

1. Use the upload panel or queue upload button.
2. Select a ZIP containing supported video files.
3. SafeTrace validates the ZIP and lists accepted or rejected files.
4. Each accepted video becomes its own job.
5. Open each completed job from the Video Queue.

For developers, the backend endpoint is POST /api/batches/analyze.
""".strip()


def _confidence_answer() -> str:
    return """
Overall confidence is SafeTrace's confidence in the detected findings, not certainty.

It depends on:
- model detections
- sampled frames
- evidence consistency
- video quality

Next: use it as a review signal, then confirm the evidence frames manually.
""".strip()


def _developer_batch_upload_answer() -> str:
    return """
Batch upload is implemented in these SafeTrace areas:

- Backend ZIP validation and batch records: src/api/batches.py.
- FastAPI route: src/api/server.py at POST /api/batches/analyze.
- Frontend selection flow: frontend-react/src/App.tsx and UploadPanel.
- Batch status rendering: the BatchStatusPanel in frontend-react/src/App.tsx.

Next: start with src/api/batches.py if you need the validation rules.
""".strip()


def _assistant_unavailable_answer() -> str:
    return """
SafeTrace analysis still works when the assistant is unavailable.

Check:
- disabled: restart with SAFETRACE_CHAT_ENABLED=auto
- missing model: place the GGUF under models/chat/
- missing runtime: install llama-cpp-python
- Ollama fallback: use SAFETRACE_CHAT_PROVIDER=ollama only if Ollama is running

Next: open /api/chat/status and use its action_hint field.
""".strip()


def _no_selected_result_answer() -> str:
    return """
I do not have a selected completed SafeTrace result to inspect yet.

- Open a completed job/result in the frontend, or pass its job_id to /api/chat.
- Then ask about violations, seatbelts, helmets, phone use, frames, timestamps, or confidence.
- I will answer from that result JSON first.

Next: select a completed result and ask again.
""".strip()


def _specific_violation_answer(record, result: Dict[str, Any], *, labels: tuple[str, ...]) -> str:
    topic = _violation_topic(labels)
    findings = _findings_matching(result, labels)
    if not findings:
        return _missing_specific_violation_answer(record, result, topic=topic)

    finding = findings[0]
    frames = _unique_frames(finding["frames"])
    frame_text = _frame_list_text(frames)
    lines = _result_identity_lines(record, result)
    lines.append(f"SafeTrace flagged {finding['name']}.")
    lines.extend(
        [
            f"- Violation type: {finding['name']}",
            f"- Severity: {_display_value(finding.get('severity'))}",
            f"- Confidence: {_format_confidence(finding.get('confidence'))}",
            f"- Evidence count: {finding.get('support_count', len(frames))} supporting frame"
            f"{'' if int(finding.get('support_count') or len(frames) or 0) == 1 else 's'}",
        ]
    )
    if frame_text:
        lines.append(f"- Evidence: {frame_text}")

    if topic == "seatbelt":
        lines.append(
            "This suggests a seatbelt may not be visible in the sampled evidence, but it requires manual confirmation."
        )
    elif topic == "helmet":
        lines.append(
            "This suggests a helmet may not be visible in the sampled evidence, but it requires manual confirmation."
        )
    elif topic == "phone":
        lines.append(
            "This suggests possible phone-use evidence in the sampled frames, but it requires manual confirmation."
        )
    else:
        lines.append("Treat this as an automated review aid and confirm against the original footage.")
    return "\n".join(lines)


def _missing_specific_violation_answer(record, result: Dict[str, Any], *, topic: str) -> str:
    lines = _result_identity_lines(record, result)
    frames_analyzed = _summary_value(result, "framesAnalyzed", 0)
    frames_with_findings = _summary_value(result, "framesWithViolations", 0)
    if topic == "phone":
        lines.extend(
            [
                "SafeTrace did not detect a phone-use violation in this result.",
                f"- Evidence checked: {frames_analyzed} sampled frame{'' if frames_analyzed == 1 else 's'}; "
                f"{frames_with_findings} frame{'' if frames_with_findings == 1 else 's'} had findings.",
                "- Current detector/rules may not reliably support phone-use detection unless a configured violation type/evidence is present.",
                "- Recommendation: manually review the original footage, or add future custom model support for phone-use evidence.",
                "SafeTrace is an automated review aid, not final proof.",
            ]
        )
        return "\n".join(lines)

    if topic == "seatbelt":
        label = "Missing Seatbelt"
        caveat = "A seatbelt could still be hidden by camera angle, clothing, blur, glare, or occlusion."
    elif topic == "helmet":
        label = "Missing Helmet"
        caveat = "A helmet could still be hidden by camera angle, blur, glare, or occlusion."
    else:
        label = "the requested violation"
        caveat = "Important evidence can still be missed by sampling, camera angle, blur, glare, or occlusion."
    lines.extend(
        [
            f"SafeTrace did not find a {label} violation in the sampled evidence for this result.",
            f"- Evidence checked: {frames_analyzed} sampled frame{'' if frames_analyzed == 1 else 's'}; "
            f"{frames_with_findings} frame{'' if frames_with_findings == 1 else 's'} had findings.",
            f"- Caveat: {caveat}",
            "SafeTrace is an automated review aid; confirm important findings manually.",
        ]
    )
    return "\n".join(lines)


def _violations_detected_answer(record, result: Dict[str, Any]) -> str:
    findings = _result_findings(result)
    lines = _result_identity_lines(record, result)
    if not findings:
        lines.extend(
            [
                "SafeTrace did not detect matching violation findings in this result.",
                f"- Frames analyzed: {_summary_value(result, 'framesAnalyzed', 0)}",
                "SafeTrace is an automated review aid; confirm against the original footage if the scene is unclear.",
            ]
        )
        return "\n".join(lines)

    lines.append(f"SafeTrace detected {len(findings)} violation type{'' if len(findings) == 1 else 's'}:")
    for finding in findings[:6]:
        frames = _unique_frames(finding["frames"])
        lines.append(
            f"- {finding['name']}: {finding.get('severity', 'unknown')} severity, "
            f"{_format_confidence(finding.get('confidence'))}, "
            f"{finding.get('support_count', len(frames))} supporting frame"
            f"{'' if int(finding.get('support_count') or len(frames) or 0) == 1 else 's'}"
        )
    lines.append("SafeTrace is an automated review aid; confirm important findings manually.")
    return "\n".join(lines)


def _frames_analyzed_answer(record, result: Dict[str, Any]) -> str:
    lines = _result_identity_lines(record, result)
    frames_analyzed = _summary_value(result, "framesAnalyzed", 0)
    frames_with_findings = _summary_value(result, "framesWithViolations", 0)
    lines.extend(
        [
            f"SafeTrace analyzed {frames_analyzed} sampled frame{'' if frames_analyzed == 1 else 's'} for this result.",
            f"- Frames with findings: {frames_with_findings}",
            f"- Evidence count: {len(_all_unique_finding_frames(result))} unique supporting frame"
            f"{'' if len(_all_unique_finding_frames(result)) == 1 else 's'}",
            "SafeTrace samples evidence frames, so confirm important findings against the original footage.",
        ]
    )
    return "\n".join(lines)


def _result_confidence_answer(record, result: Dict[str, Any]) -> str:
    lines = _result_identity_lines(record, result)
    findings = _result_findings(result)
    overall = _summary_confidence(result, findings)
    lines.append(f"Overall confidence for this result is {_format_confidence(overall)}.")
    if findings:
        top = findings[0]
        lines.append(
            f"- Top finding: {top['name']} at {_format_confidence(top.get('confidence'))} "
            f"with {top.get('support_count', len(top['frames']))} supporting frame"
            f"{'' if int(top.get('support_count') or len(top['frames']) or 0) == 1 else 's'}."
        )
        frame_text = _frame_list_text(_unique_frames(top["frames"])[:4])
        if frame_text:
            lines.append(f"- Evidence: {frame_text}")
    lines.append("Confidence is a review signal, not certainty; confirm the evidence manually.")
    return "\n".join(lines)


def _safe_or_unsafe_answer(record, result: Dict[str, Any]) -> str:
    findings = _result_findings(result)
    lines = _result_identity_lines(record, result)
    if findings:
        names = ", ".join(finding["name"] for finding in findings[:4])
        lines.extend(
            [
                "Treat this video as unsafe or requiring review based on the sampled evidence.",
                f"- Detected finding{'' if len(findings) == 1 else 's'}: {names}",
                f"- Evidence count: {len(_all_unique_finding_frames(result))} unique supporting frame"
                f"{'' if len(_all_unique_finding_frames(result)) == 1 else 's'}",
                "SafeTrace is an automated review aid; confirm before taking operational action.",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "SafeTrace did not detect matching violation findings in the sampled evidence.",
            "- Do not treat that as a guarantee that the full video is safe.",
            "SafeTrace is an automated review aid; manually confirm unclear or high-risk scenes.",
        ]
    )
    return "\n".join(lines)


def _result_identity_lines(record, result: Dict[str, Any]) -> list[str]:
    job_id = str(result.get("jobId") or getattr(record, "job_id", "") or "unknown")
    short_job = _short_job_id(job_id)
    return [
        f"Video: {_result_media_name(record, result)}",
        f"Job: {job_id}" + (f" ({short_job})" if short_job != job_id else ""),
    ]


def _result_media_name(record, result: Dict[str, Any]) -> str:
    media = dict(result.get("media") or {})
    return str(media.get("name") or media.get("filename") or getattr(record, "original_filename", "") or "selected media")


def _short_job_id(job_id: str) -> str:
    match = re.match(r"^job_\d{8}_(.+)$", job_id)
    if match:
        return match.group(1)
    if len(job_id) > 18:
        return f"job_...{job_id[-8:]}"
    return job_id


def _summary_value(result: Dict[str, Any], key: str, default: int) -> int:
    value = dict(result.get("summary") or {}).get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _display_value(value: Any) -> str:
    text = str(value or "unknown").strip()
    return text or "unknown"


def _violation_topic(labels: tuple[str, ...]) -> str:
    joined = " ".join(labels).lower()
    if "seat" in joined or "belt" in joined:
        return "seatbelt"
    if "helmet" in joined or "hard" in joined:
        return "helmet"
    if "phone" in joined or "mobile" in joined or "cell" in joined:
        return "phone"
    return "violation"


def _findings_matching(result: Dict[str, Any], labels: tuple[str, ...]) -> list[Dict[str, Any]]:
    return [finding for finding in _result_findings(result) if _finding_matches(finding, labels)]


def _finding_matches(finding: Dict[str, Any], labels: tuple[str, ...]) -> bool:
    haystack = " ".join(
        str(finding.get(key) or "")
        for key in ("key", "name", "description")
    ).lower().replace("_", " ").replace("-", " ")
    return any(label.lower().replace("_", " ") in haystack for label in labels)


def _all_unique_finding_frames(result: Dict[str, Any]) -> list[Dict[str, Any]]:
    frames = []
    for finding in _result_findings(result):
        frames.extend(finding.get("frames") or [])
    return _unique_frames(frames)


def _frame_list_text(frames: list[Dict[str, Any]], *, limit: int = 5) -> str:
    pieces = []
    for frame in frames[:limit]:
        frame_number = frame.get("frame_number", "?")
        timestamp = frame.get("timestamp", "unknown")
        confidence = _format_confidence(frame.get("confidence"))
        if confidence == "not reported":
            pieces.append(f"Frame {frame_number} at {timestamp}")
        else:
            pieces.append(f"Frame {frame_number} at {timestamp} ({confidence})")
    if len(frames) > limit:
        pieces.append(f"{len(frames) - limit} more")
    return "; ".join(pieces)


def _format_confidence(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "not reported"
    if numeric <= 1.0:
        numeric *= 100.0
    return f"{round(numeric)}%"


def _explain_result_answer(record) -> str:
    result = _record_result(record)
    if not result:
        return """
I do not have a completed result selected yet.

- Upload media or open a completed job.
- Run analysis with a clear safety query.
- Then ask me to explain the result.

Next: select a completed SafeTrace result first.
""".strip()

    findings = _result_findings(result)
    if not findings:
        return """
This result did not find matching violation types in the selected evidence frames.

- Treat this as a review aid, not final proof.
- Check whether the query matches the scene.
- Review sampled frames for missed or unclear evidence.

Next: inspect the evidence frames manually.
""".strip()

    strongest_start, strongest_end = _strongest_evidence_range(findings)
    confidence = _confidence_label(_summary_confidence(result, findings))
    lines = [f"This result found {len(findings)} violation type{'' if len(findings) == 1 else 's'} that need review:", ""]
    for finding in findings[:4]:
        lines.append(
            f"- {finding['name']}: {finding['event_count']} grouped event"
            f"{'' if finding['event_count'] == 1 else 's'}, supported by {finding['support_count']} frame"
            f"{'' if finding['support_count'] == 1 else 's'}."
        )
    if strongest_start:
        when = strongest_start if strongest_start == strongest_end else f"{strongest_start} to {strongest_end}"
        lines.append(f"- Strongest evidence appears around {when}.")
    lines.append(f"- Overall confidence is {confidence}, but it is not final proof.")
    lines.extend(["", "Next: open the evidence frames and confirm the annotated regions visually."])
    return "\n".join(lines)


def _supporting_frames_answer(record) -> str:
    result = _record_result(record)
    findings = _result_findings(result) if result else []
    if not findings:
        return _no_selected_result_answer()

    top = findings[0]
    frames = _unique_frames(top["frames"])[:5]
    lines = _result_identity_lines(record, result)
    lines.extend(
        [
            f"The top finding is {top['name']}.",
            f"- Severity: {_display_value(top.get('severity'))}",
            f"- Confidence: {_format_confidence(top.get('confidence'))}",
            f"- Evidence count: {top.get('support_count', len(frames))} supporting frame"
            f"{'' if int(top.get('support_count') or len(frames) or 0) == 1 else 's'}",
            "Supporting frames:",
        ]
    )
    if frames:
        for frame in frames:
            confidence = _format_confidence(frame.get("confidence"))
            suffix = "" if confidence == "not reported" else f" ({confidence})"
            lines.append(f"- Frame {frame['frame_number']} at {frame['timestamp']}{suffix}")
    else:
        lines.append("- No supporting frame timestamps were reported.")
    lines.append("SafeTrace is an automated review aid; inspect the annotated evidence manually.")
    return "\n".join(lines)


def _record_result(record) -> Optional[Dict[str, Any]]:
    if record is None or not getattr(record, "result", None):
        return None
    return dict(record.result)


def _result_findings(result: Dict[str, Any]) -> list[Dict[str, Any]]:
    findings: Dict[str, Dict[str, Any]] = {}

    def finding_for(key: str, name: str, severity: str = "unknown") -> Dict[str, Any]:
        if key not in findings:
            findings[key] = {
                "key": key,
                "name": name or key,
                "severity": severity or "unknown",
                "event_count": 0,
                "support_count": 0,
                "confidence": 0.0,
                "frames": [],
            }
        return findings[key]

    events = list(result.get("events") or [])
    if events:
        for event in events:
            key = str(event.get("type") or event.get("id") or event.get("name") or "finding")
            item = finding_for(key, str(event.get("name") or key), str(event.get("severity") or "unknown"))
            item["event_count"] += 1
            item["support_count"] += int(event.get("supportingFrameCount") or len(event.get("supportingFrames") or []))
            item["confidence"] = max(float(event.get("representativeConfidence") or 0.0), item["confidence"])
            item["description"] = str(event.get("description") or item.get("description") or "")
            for frame in event.get("supportingFrames") or []:
                item["frames"].append(_frame_ref(frame))
        return _sort_findings(list(findings.values()))

    for violation in list(result.get("violations") or []):
        key = str(violation.get("id") or violation.get("name") or "finding")
        item = finding_for(key, str(violation.get("name") or key), str(violation.get("severity") or "unknown"))
        frames = list(violation.get("affectedFrames") or [])
        item["event_count"] += 1
        item["support_count"] += len(frames)
        item["confidence"] = max(float(violation.get("confidenceMax") or 0.0), item["confidence"])
        item["description"] = str(violation.get("description") or item.get("description") or "")
        for frame in frames:
            item["frames"].append(_frame_ref(frame))

    if findings:
        return _sort_findings(list(findings.values()))

    for frame in list(result.get("frames") or []):
        for violation in frame.get("violations") or []:
            key = str(violation.get("type") or violation.get("id") or violation.get("name") or "finding")
            item = finding_for(key, str(violation.get("name") or key), str(violation.get("severity") or "unknown"))
            item["event_count"] = max(1, item["event_count"])
            item["support_count"] += 1
            item["confidence"] = max(float(violation.get("confidence") or 0.0), item["confidence"])
            item["description"] = str(violation.get("description") or item.get("description") or "")
            item["frames"].append(_frame_ref({**frame, "confidence": violation.get("confidence")}))

    return _sort_findings(list(findings.values()))


def _sort_findings(findings: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    return sorted(
        findings,
        key=lambda item: (
            severity_rank.get(str(item.get("severity", "")).lower(), 0),
            float(item.get("confidence") or 0.0),
            int(item.get("support_count") or 0),
        ),
        reverse=True,
    )


def _frame_ref(frame: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "frame_number": frame.get("frameNumber", "?"),
        "timestamp": frame.get("timestamp", "unknown"),
        "confidence": frame.get("confidence"),
    }


def _unique_frames(frames: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    seen = set()
    unique = []
    for frame in frames:
        key = (str(frame.get("frame_number")), str(frame.get("timestamp")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(frame)
    return unique


def _strongest_evidence_range(findings: list[Dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    top_frames = _unique_frames(findings[0]["frames"]) if findings else []
    timestamps = [str(frame["timestamp"]) for frame in top_frames if frame.get("timestamp") and frame["timestamp"] != "unknown"]
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def _summary_confidence(result: Dict[str, Any], findings: list[Dict[str, Any]]) -> Optional[float]:
    summary = dict(result.get("summary") or {})
    if isinstance(summary.get("overallConfidence"), (int, float)):
        return float(summary["overallConfidence"])
    if findings:
        return float(findings[0].get("confidence") or 0.0)
    return None


def _confidence_label(value: Optional[float]) -> str:
    if value is None:
        return "not reported"
    if value >= 0.85:
        return "high"
    if value >= 0.6:
        return "moderate"
    return "low"


def _mock_answer(*, message: str, context: ChatContext) -> str:
    return (
        "SafeTrace Assistant test response. Based on the available SafeTrace context, "
        "review the analysis summary first, then inspect supporting evidence frames and "
        "technical JSON when confidence or detections need verification."
    )


def _build_prompt(*, message: str, context: ChatContext) -> str:
    return f"""
You are SafeTrace Assistant. Answer only from the SafeTrace context below.
If the answer is not supported by the context, say what can be checked in SafeTrace instead.
When selected-result facts are present, treat them as authoritative and do not override them.
Do not answer unrelated questions.

Answer style rules:
1. Be concise.
2. Do not repeat the same point.
3. Prefer bullets over long paragraphs.
4. Use at most 5 bullets unless the user asks for detail.
5. Keep most answers under 120 words.
6. For result interpretation, include only the most important evidence.
7. Use only the provided result facts for violation type, severity, frame, timestamp, confidence, and job ID.
8. For how-to questions, explain frontend steps first, API second only if useful.
9. Do not include raw floating point values unless they are meaningful to the user.
10. Do not mention technical JSON unless the user asks for debugging/export details.
11. End with one practical next step.

Context:
{context.text}

Question:
{message}
""".strip()


def _packaged_llamacpp_answer(*, message: str, context: ChatContext) -> str:
    llm = _get_packaged_model()
    prompt = _build_prompt(message=message, context=context)
    try:
        payload = llm(
            prompt,
            max_tokens=int(getattr(SETTINGS, "chat_max_tokens", 512)),
            temperature=float(getattr(SETTINGS, "chat_temperature", 0.2)),
            top_p=float(getattr(SETTINGS, "chat_top_p", 0.9)),
            repeat_penalty=float(getattr(SETTINGS, "chat_repeat_penalty", 1.15)),
            stop=["</s>", "\nQuestion:", "\nContext:", "\nUser:", "\n###"],
        )
    except Exception as exc:
        raise ChatProviderUnavailableError("Packaged SafeTrace Assistant model could not generate a response.") from exc
    answer = _extract_llama_text(payload)
    if not answer:
        raise ChatProviderUnavailableError("Packaged SafeTrace Assistant model returned an empty response.")
    return answer


def _get_packaged_model():
    global _PACKAGED_MODEL, _PACKAGED_MODEL_LOADING, _PACKAGED_MODEL_PATH

    model_path = _packaged_model_path()
    model_name = _display_model_path(model_path)
    if not model_path.is_file():
        raise ChatProviderUnavailableError(f"Packaged chat model is missing at {model_name}.")
    runtime_diagnostics = _llama_cpp_runtime_diagnostics()
    if not runtime_diagnostics["importOk"]:
        raise ChatProviderUnavailableError(_llama_cpp_unavailable_reason(runtime_diagnostics))

    with _PACKAGED_MODEL_LOCK:
        if _PACKAGED_MODEL is not None and _PACKAGED_MODEL_PATH == model_path:
            return _PACKAGED_MODEL
        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise ChatProviderUnavailableError(
                "Packaged chat runtime import failed after diagnostics succeeded "
                f"with {type(exc).__name__}: {exc}. {PACKAGED_RUNTIME_RESTART_REQUIRED}"
            ) from exc

        _PACKAGED_MODEL_LOADING = True
        try:
            _PACKAGED_MODEL = Llama(
                model_path=str(model_path),
                n_ctx=int(getattr(SETTINGS, "chat_context_window", 4096)),
                verbose=False,
            )
            _PACKAGED_MODEL_PATH = model_path
        except Exception as exc:
            _PACKAGED_MODEL = None
            _PACKAGED_MODEL_PATH = None
            raise ChatProviderUnavailableError(
                "Packaged SafeTrace Assistant model could not be loaded. Check the GGUF file and llama-cpp runtime."
            ) from exc
        finally:
            _PACKAGED_MODEL_LOADING = False
        return _PACKAGED_MODEL


def _extract_llama_text(payload: Any) -> str:
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] or {}
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if content:
                        return str(content).strip()
                text = first.get("text")
                if text:
                    return str(text).strip()
        response = payload.get("response")
        if response:
            return str(response).strip()
    text = getattr(payload, "text", None)
    return str(text or "").strip()


def _postprocess_answer(answer: str) -> str:
    text = str(answer or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return text
    text = _remove_repeated_paragraphs(text)
    text = _remove_repeated_sentences(text)
    text = _split_giant_paragraph(text)
    return _trim_answer(text).strip()


def _remove_repeated_paragraphs(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text)
    output = []
    seen = set()
    previous = None
    for paragraph in paragraphs:
        clean = paragraph.strip()
        if not clean:
            continue
        key = _repeat_key(clean)
        if key == previous or key in seen:
            continue
        seen.add(key)
        previous = key
        output.append(clean)
    return "\n\n".join(output)


def _remove_repeated_sentences(text: str) -> str:
    lines = []
    seen_sentences = set()
    seen_bullets = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if _is_list_line(stripped):
            key = _repeat_key(stripped)
            if key in seen_bullets:
                continue
            seen_bullets.add(key)
            lines.append(line.rstrip())
            continue
        pieces = re.split(r"(?<=[.!?])\s+", stripped)
        kept = []
        for piece in pieces:
            clean = piece.strip()
            if not clean:
                continue
            key = _repeat_key(clean)
            if key in seen_sentences:
                continue
            seen_sentences.add(key)
            kept.append(clean)
        if kept:
            lines.append(" ".join(kept))
    return "\n".join(lines)


def _split_giant_paragraph(text: str) -> str:
    if "\n" in text or len(text) < 360:
        return text
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentences) < 4:
        return text
    lead = sentences[0]
    bullets = [f"- {sentence}" for sentence in sentences[1:5]]
    return "\n\n".join([lead, "\n".join(bullets)])


def _trim_answer(text: str) -> str:
    limit = min(max(int(getattr(SETTINGS, "chat_max_tokens", 512)) * 5, 900), 1800)
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    for marker in ("\n\n", "\n", ". ", "! ", "? "):
        index = clipped.rfind(marker)
        if index > limit * 0.55:
            return clipped[: index + len(marker.strip())].strip()
    return clipped.rsplit(" ", 1)[0].strip()


def _repeat_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_list_line(line: str) -> bool:
    return bool(re.match(r"^(?:[-*]|\d+[.)])\s+", line))


def _ollama_answer(*, message: str, context: ChatContext) -> str:
    prompt = _build_prompt(message=message, context=context)
    try:
        response = httpx.post(
            f"{SETTINGS.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": SETTINGS.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": int(getattr(SETTINGS, "chat_max_tokens", 512)),
                    "temperature": float(getattr(SETTINGS, "chat_temperature", 0.2)),
                    "top_p": float(getattr(SETTINGS, "chat_top_p", 0.9)),
                    "repeat_penalty": float(getattr(SETTINGS, "chat_repeat_penalty", 1.15)),
                    "stop": ["</s>", "\nQuestion:", "\nContext:", "\nUser:", "\n###"],
                },
            },
            timeout=float(SETTINGS.chat_timeout_seconds),
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise ChatProviderUnavailableError(
            "Ollama is unavailable. Start Ollama, choose the packaged local provider, or disable SafeTrace Assistant."
        ) from exc
    answer = str(payload.get("response") or "").strip()
    if not answer:
        raise ChatProviderUnavailableError("Ollama returned an empty response.")
    return answer
