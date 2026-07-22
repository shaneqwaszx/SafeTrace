@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%" || exit /b 1

if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv\Scripts\python.exe. Create the environment from README_SOURCE_HANDOFF.md first.
  exit /b 1
)

if not exist "config\safetrace.env" (
  echo Missing config\safetrace.env.
  echo Copy config\safetrace.fastlocal.env.example to config\safetrace.env first.
  exit /b 1
)

set "KMP_DUPLICATE_LIB_OK=TRUE"
set "OMP_NUM_THREADS=1"
"%REPO_ROOT%\.venv\Scripts\python.exe" -m uvicorn src.api.server:app --env-file config\safetrace.env --host 0.0.0.0 --port 8000 --log-level info

endlocal
