@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m straintrace_capture.gui %*
) else (
    py -3 -m straintrace_capture.gui %*
)
exit /b %errorlevel%
