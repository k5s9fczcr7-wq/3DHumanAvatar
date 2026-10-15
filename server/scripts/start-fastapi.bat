@echo off
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
.\.venv\Scripts\python.exe -u -m uvicorn server:app --host 127.0.0.1 --port 8100 --log-level warning > logs\fastapi.out.log 2> logs\fastapi.err.log
