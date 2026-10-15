@echo off
REM ============================================================================
REM KioskAvatar - full stack restart
REM
REM Sequenced stop-then-start. Uses stop.bat (which calls stop-stack.ps1
REM with the FastAPI-last delay) and then start.bat. A 2s buffer between
REM the two lets ports clear cleanly before the new processes try to bind.
REM ============================================================================

call "%~dp0stop.bat" %*
echo.
echo === restart bridge: waiting 2s for ports to clear ===
timeout /t 2 /nobreak > nul
echo.
call "%~dp0start.bat"
