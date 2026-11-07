@echo off
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%LMFT-Architecture\src\lmft_architecture\bloodmoon_app.py" (
    set "ROOT=%LOCALAPPDATA%\LMFT-Celestriad\"
    if not exist "%LOCALAPPDATA%\LMFT-Celestriad\LMFT-Architecture\src\lmft_architecture\bloodmoon_app.py" (
        echo [Celestriad 1.0] Downloading Celestriad 1.0 release bundle to "%LOCALAPPDATA%\LMFT-Celestriad"...
        mkdir "%LOCALAPPDATA%\LMFT-Celestriad" 2>nul
        curl.exe -L "https://github.com/schaleh12/Dual-Hemisphere-Wisdom-Model/releases/latest/download/lmft-sanctuary-club-v0.1.0-win64.zip" -o "%TEMP%\lmft-celestriad-bundle.zip"
        tar.exe -xf "%TEMP%\lmft-celestriad-bundle.zip" -C "%LOCALAPPDATA%\LMFT-Celestriad"
    )
)
cd /d "%ROOT%"
set "PYTHONPATH=%ROOT%LMFT-Architecture\src"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"
set "TQDM_DISABLE=1"

if exist "LMFT-Architecture\.venv\Scripts\pythonw.exe" (
    start "" /d "%ROOT%LMFT-Architecture" "LMFT-Architecture\.venv\Scripts\pythonw.exe" -m lmft_architecture.bloodmoon_app --single-window %*
    exit /b 0
)

if exist "LMFT-Architecture\.venv\Scripts\python.exe" (
    start "" /d "%ROOT%LMFT-Architecture" "LMFT-Architecture\.venv\Scripts\python.exe" -m lmft_architecture.bloodmoon_app --single-window %*
    exit /b 0
)

echo [Celestriad 1.0] Python .venv not found yet — opening the 5-Page Setup Wizard...
call "%ROOT%Launch-Setup-Wizard.bat"
