@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=src
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m lmft_lab.sanctuary_club_app %*
) else (
    python -m lmft_lab.sanctuary_club_app %*
)
