@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0LMFT-Architecture\src"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"
set "TQDM_DISABLE=1"

if exist "LMFT-Architecture\.venv\Scripts\pythonw.exe" (
    start "" /d "%~dp0LMFT-Architecture" "LMFT-Architecture\.venv\Scripts\pythonw.exe" -m lmft_architecture.bloodmoon_app --single-window %*
    exit /b 0
)

if exist "LMFT-Architecture\.venv\Scripts\python.exe" (
    start "" /d "%~dp0LMFT-Architecture" "LMFT-Architecture\.venv\Scripts\python.exe" -m lmft_architecture.bloodmoon_app --single-window %*
    exit /b 0
)

echo [Celestriad 1.0] Python .venv not found yet — opening the 5-Page Setup Wizard...
call "%~dp0Launch-Setup-Wizard.bat"
