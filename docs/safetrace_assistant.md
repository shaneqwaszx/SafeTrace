# SafeTrace Assistant

SafeTrace Assistant is optional and limited to SafeTrace questions about the
app, local backend status, current analysis results, batch manifests, evidence
frames, exports, and known limitations. Guardrails run before any provider call,
so out-of-scope questions are refused without invoking a model.

## Default Packaged Provider

The default backend configuration uses a packaged local llama.cpp provider:

```cmd
set SAFETRACE_CHAT_ENABLED=auto
set SAFETRACE_CHAT_PROVIDER=packaged_llamacpp
set SAFETRACE_CHAT_MODEL_PATH=models/chat/safetrace-assistant-qwen2.5-1.5b-instruct-q4.gguf
set SAFETRACE_CHAT_CONTEXT_WINDOW=4096
set SAFETRACE_CHAT_MAX_TOKENS=512
set SAFETRACE_CHAT_TEMPERATURE=0.2
set SAFETRACE_CHAT_TOP_P=0.9
set SAFETRACE_CHAT_AUTOLOAD=false
set SAFETRACE_CHAT_WARMUP_ON_OPEN=false
```

In packaged mode, users do not need to run Ollama manually. The backend checks
for `llama-cpp-python` and the configured GGUF file. The model is lazy-loaded on
the first chat request by default, then cached for later requests.

Place local model files under:

```text
models/chat/
```

The approved default filename is:

```text
safetrace-assistant-qwen2.5-1.5b-instruct-q4.gguf
```

Model files are ignored by git and must not be committed.

## Fast Local Profile

For a lighter, more mobile-feeling assistant experience, use the fast profile:

```cmd
set SAFETRACE_CHAT_SPEED_PROFILE=fast
set SAFETRACE_CHAT_CONTEXT_WINDOW=2048
set SAFETRACE_CHAT_MAX_TOKENS=200
set SAFETRACE_CHAT_TEMPERATURE=0.1
set SAFETRACE_CHAT_TOP_P=0.82
set SAFETRACE_CHAT_REPEAT_PENALTY=1.15
```

`SAFETRACE_CHAT_SPEED_PROFILE=fast` also makes the backend send shorter,
summarized context to the local model by default. Explicit environment values
for context window, max tokens, temperature, top-p, or repeat penalty still take
precedence.

Recommended quality/speed balance:

```text
Qwen2.5-1.5B-Instruct Q4 GGUF
```

Recommended faster/mobile option:

```text
Qwen2.5-0.5B-Instruct Q4 GGUF
```

Alternative fast option:

```text
Llama-3.2-1B-Instruct Q4 GGUF
```

Do not download or commit model files automatically. Place approved local model
files manually or through a future installer/release package.

## Status Endpoint

Endpoints:

- `GET /api/chat/status`
- `POST /api/chat`

`GET /api/chat/status` reports one of these states:

- `available`: provider and model are ready.
- `disabled`: `SAFETRACE_CHAT_ENABLED` disables chat.
- `missing_model`: packaged provider is enabled but the GGUF file is absent.
- `missing_runtime`: packaged provider is enabled but `llama_cpp` is not installed.
- `loading`: the packaged local model is currently loading.
- `unavailable`: the selected provider is configured but cannot answer.

The status response also includes diagnostic fields such as `enabled_mode`,
`provider`, `model_path`, `model_exists`, `runtime_available`, `reason`, and
`action_hint`. It also reports local runtime diagnostics:

- `python_executable`: Python executable running the backend.
- `expected_venv_python`: repo `.venv` Python expected for local development.
- `running_in_expected_venv`: whether those paths match.
- `llama_cpp_import_status`: `ok`, `missing`, or `import_error`.
- `llama_cpp_import_error_type` / `llama_cpp_import_error_message`: native import
  failure details when the package exists but cannot load.
- `setup_command`: `.venv\Scripts\python.exe -m pip install llama-cpp-python`.
- `restart_required`: reminder to restart the backend after installing.

If `SAFETRACE_CHAT_ENABLED=false`, restart the backend with:

```cmd
set SAFETRACE_CHAT_ENABLED=auto
set SAFETRACE_CHAT_PROVIDER=packaged_llamacpp
set SAFETRACE_CHAT_MODEL_PATH=models/chat/safetrace-assistant-qwen2.5-1.5b-instruct-q4.gguf
```

If chat is disabled, unavailable, or missing the model, `POST /api/chat` returns
a structured `503` error instead of crashing. If the packaged model is present
but `llama_cpp` is missing, SafeTrace exposes a limited deterministic help mode
for built-in SafeTrace usage and troubleshooting answers. The status endpoint
still reports `missing_runtime`, `available=false`, and `fallback_available=true`
so the frontend can label this mode honestly.

To install the runtime in local development:

```cmd
.venv\Scripts\python.exe -m pip install llama-cpp-python
```

Verify the same interpreter before starting the backend:

```cmd
.venv\Scripts\python.exe -c "import sys; print(sys.executable); import llama_cpp; print('llama_cpp ok')"
```

Start local development with `.venv` Python, not plain `python`, so the backend
uses the environment where `llama-cpp-python` is installed:

```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1
.venv\Scripts\python.exe -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --log-level info
```

For the source backend profile that also enables the Lightweight VLM worker,
use the checked startup helper:

```cmd
scripts\start_safetrace_dev_backend.bat
```

That helper prefers `.venv\Scripts\python.exe`, starts CPU Safe Mode, sets
`SAFETRACE_CHAT_PROVIDER=packaged_llamacpp`, enables the local Lightweight VLM
worker with a 60-second subprocess timeout and a local visual review frame limit,
checks `llama_cpp`, and prints the exact
`llama-cpp-python` install command if the runtime is missing. Starting uvicorn
manually without those environment variables can leave the assistant in limited
mode and make Lightweight VLM look selected while evidence still falls back to
rule-based explanations because the worker was never enabled.

The helper sets `SAFETRACE_VLM_FRAME_LIMIT=5`, keeps
`SAFETRACE_VLM_MAX_EVIDENCE_FRAMES=5` for compatibility, and sets
`SAFETRACE_VLM_JOB_TIMEOUT_SECONDS=0` so local installed VLMs use per-attempt
watchdogs rather than a shared user-visible job budget.
Lightweight VLM answers are attached per evidence frame only when the worker
runs and the output passes the safety-specific quality gate. Frames skipped by
frame-limit policy, missing eligible violations, worker failures, timeouts, or
generic output remain rule-based and should show the specific skip or fallback
reason in technical evidence. If one VLM attempt times out or the output fails
quality checks, SafeTrace keeps the result available and records the runtime
guard reason without showing raw budget wording in the normal UI.

Restart the backend after installing. Main SafeTrace analysis, single-video
upload, ZIP/batch upload, and rule-based visual explanation fallback continue
to work while chat is unavailable or in limited help mode.

For packaged Windows builds, the backend executable must be built with the
Python environment that has `llama-cpp-python` installed. If the release package
finds the GGUF model but reports `missing_runtime`, rebuild the backend with:

```cmd
.venv\Scripts\python.exe scripts\build_backend_exe.py --run
```

Building with a global Python that lacks `llama_cpp` can produce an executable
where the assistant model is present but the packaged assistant runtime is not.
This does not make chat required for analysis; uploads, single-video analysis,
ZIP/batch analysis, and rule-based visual explanation fallback still work.

## Optional Warmup On Open

By default, the packaged local model is lazy-loaded only when the first in-scope
assistant question is submitted. To warm the model when the floating assistant
panel opens, set:

```cmd
set SAFETRACE_CHAT_WARMUP_ON_OPEN=true
```

The React assistant checks `GET /api/chat/status`; if `warmup_on_open` is true,
it calls `POST /api/chat/warmup` after the panel opens. This does not run during
backend startup and does not run during normal SafeTrace analysis.

## Optional Ollama Fallback

Ollama remains available as an explicit optional provider:

```cmd
set SAFETRACE_CHAT_ENABLED=true
set SAFETRACE_CHAT_PROVIDER=ollama
set SAFETRACE_OLLAMA_BASE_URL=http://127.0.0.1:11434
set SAFETRACE_OLLAMA_MODEL=llama3.2:3b
```

No Ollama binaries, GGUF files, or model weights are stored in this repository.
