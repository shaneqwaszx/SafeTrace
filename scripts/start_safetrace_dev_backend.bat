@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

cd /d "%REPO_ROOT%" || (
  echo [SafeTrace Dev] ERROR: Could not change to repo root: "%REPO_ROOT%"
  exit /b 1
)

if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
  set "PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe"
  echo [SafeTrace Dev] Using .venv Python: "%REPO_ROOT%\.venv\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
  echo [SafeTrace Dev] WARNING: .venv\Scripts\python.exe was not found.
  echo [SafeTrace Dev] WARNING: Falling back to system Python from PATH.
  echo [SafeTrace Dev] WARNING: /api/system/status may report runtime gaps if system Python lacks SafeTrace dev dependencies.
)

set "KMP_DUPLICATE_LIB_OK=TRUE"
set "OMP_NUM_THREADS=1"
set "SAFETRACE_ANALYSIS_SAFE_MODE=true"
set "SAFETRACE_DEVICE=auto"
set "SAFETRACE_ENABLE_GPU_AUTO=true"
set "SAFETRACE_LIGHTWEIGHT_VLM_DEVICE=auto"
set "SAFETRACE_ENHANCED_VLM_DEVICE=cuda"
set "SAFETRACE_MOBILESAM_DEVICE=auto"
set "SAFETRACE_MOBILESAM_ENABLED=auto"
set "SAFETRACE_SAFE_MODE_ALLOW_MOBILESAM=true"
set "SAFETRACE_MOBILESAM_WORKER_ENABLED=true"
set "SAFETRACE_MOBILESAM_WORKER_TIMEOUT_SECONDS=60"
set "SAFETRACE_VLM_ENABLED=true"
set "SAFETRACE_VLM_PROFILE=lightweight_512m"
set "SAFETRACE_VLM_PROVIDER=auto"
set "SAFETRACE_VLM_MODEL_PATH=models\vlm\lightweight-512m"
set "SAFETRACE_VLM_DIR=models\vlm"
set "SAFETRACE_VLM_LIGHTWEIGHT_MODEL_PATH=models\vlm\lightweight-256m"
set "SAFETRACE_VLM_LIGHTWEIGHT_512M_MODEL_PATH=models\vlm\lightweight-512m"
set "SAFETRACE_VLM_ENHANCED_MODEL_PATH=models\vlm\enhanced-2b"
set "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED=true"
set "SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS=60"
set "SAFETRACE_LIGHTWEIGHT_VLM_PRIMARY=auto"
set "SAFETRACE_LIGHTWEIGHT_VLM_FALLBACK=256m"
set "SAFETRACE_LIGHTWEIGHT_VLM_CPU_PREFER_256M=true"
set "SAFETRACE_LIGHTWEIGHT_VLM_TIMEOUT_SECONDS=90"
set "SAFETRACE_LIGHTWEIGHT_VLM_TOTAL_BUDGET_SECONDS=0"
set "SAFETRACE_VLM_FRAME_LIMIT=5"
set "SAFETRACE_VLM_MAX_EVIDENCE_FRAMES=5"
set "SAFETRACE_VLM_JOB_TIMEOUT_SECONDS=0"
set "SAFETRACE_VLM_MAX_QUALITY_FAILURES=1"
set "SAFETRACE_VLM_DISABLE_AFTER_TIMEOUT=true"
set "SAFETRACE_VLM_MAX_TOKENS=40"
set "SAFETRACE_CHAT_ENABLED=auto"
set "SAFETRACE_CHAT_PROVIDER=packaged_llamacpp"
set "SAFETRACE_CHAT_MODEL_PATH=models\chat\safetrace-assistant-qwen2.5-1.5b-instruct-q4.gguf"

echo [SafeTrace Dev] Repo root: "%REPO_ROOT%"
echo [SafeTrace Dev] Backend URL: http://127.0.0.1:8000/api/health
echo [SafeTrace Dev] Safe Mode: %SAFETRACE_ANALYSIS_SAFE_MODE%
echo [SafeTrace Dev] Device: %SAFETRACE_DEVICE%
echo [SafeTrace Dev] GPU auto routing: %SAFETRACE_ENABLE_GPU_AUTO%
echo [SafeTrace Dev] MobileSAM device: %SAFETRACE_MOBILESAM_DEVICE%
echo [SafeTrace Dev] MobileSAM enabled: %SAFETRACE_MOBILESAM_ENABLED%
echo [SafeTrace Dev] MobileSAM worker enabled: %SAFETRACE_MOBILESAM_WORKER_ENABLED%
echo [SafeTrace Dev] VLM enabled: %SAFETRACE_VLM_ENABLED%
echo [SafeTrace Dev] VLM profile: %SAFETRACE_VLM_PROFILE%
echo [SafeTrace Dev] Lightweight VLM device: %SAFETRACE_LIGHTWEIGHT_VLM_DEVICE%
echo [SafeTrace Dev] Lightweight VLM primary policy: %SAFETRACE_LIGHTWEIGHT_VLM_PRIMARY%
echo [SafeTrace Dev] Lightweight VLM fallback policy: %SAFETRACE_LIGHTWEIGHT_VLM_FALLBACK%
echo [SafeTrace Dev] Lightweight VLM worker enabled: %SAFETRACE_LIGHTWEIGHT_VLM_WORKER_ENABLED%
echo [SafeTrace Dev] Lightweight VLM worker timeout: %SAFETRACE_LIGHTWEIGHT_VLM_WORKER_TIMEOUT_SECONDS%s
echo [SafeTrace Dev] Local visual review frame limit: %SAFETRACE_VLM_FRAME_LIMIT%
echo [SafeTrace Dev] Local visual review shared time budget: disabled
echo [SafeTrace Dev] Lightweight VLM quality failure limit: %SAFETRACE_VLM_MAX_QUALITY_FAILURES%
echo [SafeTrace Dev] Chat provider: %SAFETRACE_CHAT_PROVIDER%
echo [SafeTrace Dev] Chat model: %SAFETRACE_CHAT_MODEL_PATH%

if not exist "%SAFETRACE_CHAT_MODEL_PATH%" (
  echo [SafeTrace Dev] WARNING: Chat model file was not found at "%SAFETRACE_CHAT_MODEL_PATH%".
  echo [SafeTrace Dev] WARNING: Assistant will be limited unless the packaged GGUF model is present.
)

echo [SafeTrace Dev] Checking llama_cpp import with the selected Python...
"%PYTHON_EXE%" -c "import sys; print(sys.executable); import llama_cpp; print('llama_cpp ok')" >nul 2>nul
if errorlevel 1 (
  echo [SafeTrace Dev] WARNING: llama_cpp could not be imported by the selected Python.
  echo [SafeTrace Dev] WARNING: SafeTrace Assistant will run in limited local help mode until the runtime is installed.
  echo [SafeTrace Dev] Install command:
  echo [SafeTrace Dev] "%PYTHON_EXE%" -m pip install llama-cpp-python
) else (
  echo [SafeTrace Dev] llama_cpp import ok.
)

echo [SafeTrace Dev] Checking backend runtime imports...
"%PYTHON_EXE%" -c "import numpy; import uvicorn; import fastapi; import faiss; import cv2; import ultralytics; print('backend imports ok')"
if errorlevel 1 (
  echo [SafeTrace Dev] ERROR: Selected Python cannot import required SafeTrace backend dependencies.
  echo [SafeTrace Dev] ERROR: If FAISS reports a NumPy 2.x compatibility error, run:
  echo [SafeTrace Dev] "%PYTHON_EXE%" -m pip install "numpy<2"
  echo [SafeTrace Dev] ERROR: Backend not started because analysis would fail in this environment.
  exit /b 1
)

echo.
echo [SafeTrace Dev] Starting source FastAPI backend...
echo [SafeTrace Dev] Command: "%PYTHON_EXE%" -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --log-level info
"%PYTHON_EXE%" -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --log-level info

endlocal
