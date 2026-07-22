@echo off
setlocal

set "REPO_ROOT=%~dp0"
call "%REPO_ROOT%scripts\start_safetrace_full_local.bat"
exit /b %errorlevel%
