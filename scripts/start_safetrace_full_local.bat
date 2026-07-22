@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%" || exit /b 1

set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [SafeTrace Full Local] ERROR: .venv\Scripts\python.exe was not found.
  exit /b 1
)

set "KMP_DUPLICATE_LIB_OK=TRUE"
set "OMP_NUM_THREADS=1"
set "SAFETRACE_RUNTIME_PROFILE=local_full"
set "SAFETRACE_REQUIRE_GPU=1"
set "SAFETRACE_REQUIRE_CHAT=1"
set "SAFETRACE_REQUIRE_MOBILESAM=1"
set "SAFETRACE_ALLOW_CPU_FALLBACK=0"
set "SAFETRACE_DEVICE=cuda"
set "SAFETRACE_ENABLE_GPU_AUTO=true"
set "SAFETRACE_ANALYSIS_SAFE_MODE=true"
set "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM=true"
set "SAFETRACE_MOBILESAM_ENABLED=auto"
set "SAFETRACE_MOBILESAM_WORKER_ENABLED=true"
set "SAFETRACE_VLM_ENABLED=auto"
set "SAFETRACE_ENABLE_VLM=true"
set "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED=true"
set "SAFETRACE_CHAT_ENABLED=auto"
set "SAFETRACE_CHAT_AUTOLOAD=true"
set "SAFETRACE_JOB_CONCURRENCY=2"
set "SAFETRACE_PREPROCESS_CONCURRENCY=2"
rem RTX 4080 model-backed Phase U validation admitted a bounded third worker.
rem Other profiles retain their conservative defaults.
set "SAFETRACE_ADAPTIVE_WORKERS_ENABLED=true"
set "SAFETRACE_WORKER_MIN=2"
set "SAFETRACE_WORKER_MAX=3"
set "SAFETRACE_THIRD_WORKER_VALIDATED=true"
set "SAFETRACE_GPU_INFERENCE_CONCURRENCY=3"
set "SAFETRACE_MOBILESAM_CONCURRENCY=1"
set "SAFETRACE_VLM_CONCURRENCY=1"
set "SAFETRACE_PER_BATCH_CONCURRENCY=3"
set "SAFETRACE_BATCH_MAX_ACTIVE_JOBS=3"
set "SAFETRACE_SCHEDULER_POLICY=oldest_batch_first"

echo [SafeTrace Full Local] Running strict local runtime preflight...
"%PYTHON_EXE%" scripts\preflight_full_local.py --output data\runtime_preflight_local_full.json
if errorlevel 1 (
  echo [SafeTrace Full Local] ERROR: Preflight failed. Backend was not started.
  exit /b 1
)

echo [SafeTrace Full Local] Preflight passed. Starting backend on http://127.0.0.1:8000
"%PYTHON_EXE%" -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000
