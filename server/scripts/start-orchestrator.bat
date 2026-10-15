@echo off
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
REM The Jabra keeper holds a second listening stream so orchestrator restarts never reset the
REM Jabra's mute (scripts\jabra_keeper.py); if one is already running, this copy exits. It is started through
REM PowerShell so it inherits none of this window's open files: started with "start", it would keep start.bat's
REM redirected log open for its whole life, and the next start.bat writing to that log would silently never run.
powershell -NoProfile -Command "Start-Process -FilePath '.\.venv\Scripts\pythonw.exe' -ArgumentList 'scripts\jabra_keeper.py' -WorkingDirectory '%CD%'"
REM Keep the previous run's logs: how it ended (or crashed) is only readable there.
if exist logs\orchestrator.err.log move /y logs\orchestrator.err.log logs\orchestrator.err.prev.log >nul
if exist logs\orchestrator.out.log move /y logs\orchestrator.out.log logs\orchestrator.out.prev.log >nul
.\.venv\Scripts\python.exe -u -m orchestrator.orchestrator > logs\orchestrator.out.log 2> logs\orchestrator.err.log
