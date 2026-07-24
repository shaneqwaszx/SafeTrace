@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "SAFETRACE_ROOT=%CD%"
set "SAFETRACE_RUNTIME_PROFILE=local_full"
set "SAFETRACE_ALLOWED_FRONTEND_ORIGINS=https://safetrace-iota.vercel.app,http://127.0.0.1:5173,http://localhost:5173"
set "SAFETRACE_ALLOWED_ORIGINS=%SAFETRACE_ALLOWED_FRONTEND_ORIGINS%"
set "SAFETRACE_INCLUDE_LOCAL_CORS_ORIGINS=true"
set "SAFETRACE_CONNECTOR_SESSION_TTL_SECONDS=3600"
set "SAFETRACE_CHAT_AUTOLOAD=true"

if not exist ".venv\Scripts\python.exe" (
  echo [SafeTrace] Missing .venv\Scripts\python.exe
  echo Run the local setup first, then try this launcher again.
  pause
  exit /b 1
)

echo [SafeTrace] Checking strict full-local runtime readiness...
".venv\Scripts\python.exe" "scripts\preflight_full_local.py"
if errorlevel 1 (
  echo [SafeTrace] Preflight failed. The backend was not started.
  pause
  exit /b 1
)

powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $h=Invoke-RestMethod 'http://127.0.0.1:8000/api/health' -TimeoutSec 2; if($h.api -ne 'safetrace-local' -or -not $h.bridgeSupported){exit 2}" >nul 2>&1
if not errorlevel 1 (
  echo [SafeTrace] Reusing the healthy local-only backend on 127.0.0.1:8000.
  goto :OPEN_FRONTEND
)

echo [SafeTrace] Starting local-only backend on 127.0.0.1:8000...
start "SafeTrace Local Backend" /D "%SAFETRACE_ROOT%" cmd /k ""%SAFETRACE_ROOT%\.venv\Scripts\python.exe" -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --log-level info"

for /L %%I in (1,1,60) do (
  powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $h=Invoke-RestMethod 'http://127.0.0.1:8000/api/health' -TimeoutSec 2; if($h.api -ne 'safetrace-local' -or -not $h.bridgeSupported){exit 2}" >nul 2>&1
  if not errorlevel 1 goto :OPEN_FRONTEND
  timeout /t 1 /nobreak >nul
)

echo [SafeTrace] Backend health check did not become ready.
echo Review the SafeTrace Local Backend window for the exact error.
pause
exit /b 1

:OPEN_FRONTEND
echo [SafeTrace] Opening the hosted frontend. Choose Connect Local Backend in the browser.
start "" "https://safetrace-iota.vercel.app/?backend=local"
exit /b 0
