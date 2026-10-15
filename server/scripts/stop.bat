@echo off
REM ============================================================================
REM KioskAvatar — full stack stop
REM
REM Delegates to stop-stack.ps1, which stops the stack in this order:
REM   1. Graceful orchestrator shutdown via WS ({"type":"shutdown"} on :8765),
REM      with a Stop-Process -Force fallback after 8s if the graceful path
REM      doesn't exit. Graceful path is required because force-killing the
REM      orchestrator while OpenCV holds the C920 wedges Windows Media
REM      Foundation in MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED — only USB
REM      replug or reboot recovers from that.
REM   2. llama-server Stop-Process (port 8080).
REM   3. FastAPI Stop-Process (port 8100), last and after a short delay, so
REM      the admin dashboard can show the other components going down.
REM   4. Orphan sweep — any python.exe / cmd.exe still running orchestrator,
REM      uvicorn, or the start-* wrapper commands.
REM
REM Pass-through args (forwarded to stop-stack.ps1):
REM   --KeepLlama       Leave llama-server running (warm cache for re-start).
REM   --KeepFastAPI     Leave FastAPI running.
REM   --GracefulTimeoutSeconds N    Override the 8s graceful-exit timeout.
REM ============================================================================

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-stack.ps1" %*
