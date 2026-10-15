@echo off
REM ============================================================================
REM KioskAvatar - full stack start
REM
REM Sequenced startup of llama-server -> FastAPI -> orchestrator. Each
REM component launches in its own minimized cmd window so it persists past
REM this script. Per-component logs go to server\logs\{component}.{out,err}.log.
REM
REM Idempotent: each step checks whether the corresponding port is already
REM listening and skips launching if so. Health is still verified for each
REM component, whether this script launched it or it was already up.
REM
REM Timeouts are sized for cold starts:
REM   llama        480s (cold model load into VRAM)
REM   FastAPI      180s (Chatterbox TTS + emotion2vec init, ~75s measured)
REM   orchestrator 90s  (MediaPipe + faster-whisper init)
REM
REM UE5 (KioskAvatar.uproject) is launched manually after this script
REM completes. Stop the stack with stop.bat.
REM ============================================================================
setlocal

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set SCRIPTS=%ROOT%\scripts
set LOGS=%ROOT%\logs

if not exist "%LOGS%" mkdir "%LOGS%"

echo === stack start @ %TIME% ===

REM --- 1/3: llama-server ---
REM Skip-if-listening: errorlevel 1 means port already in use, errorlevel 0 means free.
powershell -NoProfile -Command "$listening = @(Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue); if ($listening.Count -gt 0) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    echo [1/3] llama-server already listening on :8080, skipping launch
) else (
    echo [1/3] llama-server -^> launching...
    start "kiosk-llama" /MIN "%SCRIPTS%\start-llama.bat"
)
echo       verifying /health on :8080 ^(up to 480s, cold model load^)...
powershell -NoProfile -Command "$d=(Get-Date).AddSeconds(480); while($true){ try { $r=Invoke-WebRequest -Uri 'http://127.0.0.1:8080/health' -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop; if($r.StatusCode -eq 200){ Write-Host ('       ready @ ' + (Get-Date -Format HH:mm:ss)); exit 0 } } catch {} if((Get-Date) -gt $d){ exit 1 } Start-Sleep -Milliseconds 1500 }"
if errorlevel 1 (
    echo TIMEOUT waiting for llama-server. Tail %LOGS%\llama-server.err.log for cause.
    exit /b 1
)

REM --- 2/3: FastAPI ---
powershell -NoProfile -Command "$listening = @(Get-NetTCPConnection -LocalPort 8100 -State Listen -ErrorAction SilentlyContinue); if ($listening.Count -gt 0) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    echo [2/3] FastAPI already listening on :8100, skipping launch
) else (
    echo [2/3] FastAPI -^> launching...
    start "kiosk-fastapi" /MIN "%SCRIPTS%\start-fastapi.bat"
)
echo       verifying /health on :8100 ^(up to 180s, TTS+A2E init^)...
powershell -NoProfile -Command "$d=(Get-Date).AddSeconds(180); while($true){ try { $r=Invoke-WebRequest -Uri 'http://127.0.0.1:8100/health' -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop; if($r.StatusCode -eq 200){ Write-Host ('       ready @ ' + (Get-Date -Format HH:mm:ss)); exit 0 } } catch {} if((Get-Date) -gt $d){ exit 1 } Start-Sleep -Milliseconds 1500 }"
if errorlevel 1 (
    echo TIMEOUT waiting for FastAPI. Tail %LOGS%\fastapi.err.log for cause.
    exit /b 1
)

REM --- 3/3: orchestrator ---
powershell -NoProfile -Command "$listening = @(Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue); if ($listening.Count -gt 0) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    echo [3/3] orchestrator already listening on :8765, skipping launch
) else (
    echo [3/3] orchestrator -^> launching...
    start "kiosk-orchestrator" /MIN "%SCRIPTS%\start-orchestrator.bat"
)
echo       verifying WS port :8765 ^(up to 90s, MediaPipe+Whisper init^)...
powershell -NoProfile -Command "$d=(Get-Date).AddSeconds(90); while($true){ if(@(Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue).Count -gt 0){ Write-Host ('       ready @ ' + (Get-Date -Format HH:mm:ss)); exit 0 } if((Get-Date) -gt $d){ exit 1 } Start-Sleep -Milliseconds 1500 }"
if errorlevel 1 (
    echo TIMEOUT waiting for orchestrator. Tail %LOGS%\orchestrator.err.log for cause.
    exit /b 1
)

echo.
echo === stack ready ===
echo   llama-server  http://127.0.0.1:8080
echo   FastAPI       http://127.0.0.1:8100   ^(dashboards: /dashboard, /admin^)
echo   orchestrator  ws://localhost:8765     ^(UE5 connects here^)
echo.
echo Now open KioskAvatar.uproject in UE 5.7.4 and hit Play.
echo Stop the stack with stop.bat ^(or stop-stack.ps1 with --KeepLlama / --KeepFastAPI^).

endlocal
