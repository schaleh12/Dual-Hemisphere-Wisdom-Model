@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0LMFT-Architecture\src"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"
set "TQDM_DISABLE=1"

if exist "LMFT-Architecture\.venv\Scripts\pythonw.exe" (
    start "" "LMFT-Architecture\.venv\Scripts\pythonw.exe" "LMFT-Architecture\src\lmft_architecture\installer_wizard.py"
    exit /b 0
)

where pythonw.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start "" pythonw.exe "LMFT-Architecture\src\lmft_architecture\installer_wizard.py"
    exit /b 0
)

python "LMFT-Architecture\src\lmft_architecture\installer_wizard.py"
