@echo off
REM Package the kiosk as a standalone build; output goes under %OUT% below.
REM
REM Two launch pitfalls:
REM  1. Passing this command line through PowerShell -ArgumentList mangles the
REM     embedded quotes and the run silently never starts. Use this .bat.
REM  2. Do NOT launch with -WindowStyle Hidden and do NOT redirect stdout.
REM     AutomationTool sets console colours while logging; with no console
REM     buffer it fails mid-pak with
REM     "IOException: No process is on the other end of the pipe" AFTER a
REM     successful cook. Run it with a real (minimized) console and let
REM     UAT write its own log.
set UE=C:\Program Files\Epic Games\UE_5.7
for %%I in ("%~dp0..\..\unreal\KioskAvatar.uproject") do set "PROJ=%%~fI"
for %%I in ("%~dp0..\..\packaged") do set "OUT=%%~fI"

if not exist "%OUT%" mkdir "%OUT%"
del "%OUT%\package.DONE" 2>nul

call "%UE%\Engine\Build\BatchFiles\RunUAT.bat" BuildCookRun ^
 -project="%PROJ%" ^
 -noP4 -platform=Win64 -clientconfig=Development ^
 -cook -build -stage -pak -archive ^
 -archivedirectory="%OUT%" ^
 -utf8output -nocompileeditor

echo DONE_%ERRORLEVEL% > "%OUT%\package.DONE"
