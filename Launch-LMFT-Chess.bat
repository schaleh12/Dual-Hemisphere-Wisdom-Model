@echo off
setlocal
set "ARCH_DIR="
if exist "%~dp0src\lmft_architecture\chess_app.py" (
    set "ARCH_DIR=%~dp0"
) else if exist "%~dp0LMFT-Architecture\src\lmft_architecture\chess_app.py" (
    set "ARCH_DIR=%~dp0LMFT-Architecture\"
) else (
    set "ARCH_DIR=%LOCALAPPDATA%\LMFT-Celestriad\"
    if not exist "%LOCALAPPDATA%\LMFT-Celestriad\src\lmft_architecture\chess_app.py" (
        echo [Celestriad 1.0] Downloading Celestriad 1.0 release bundle to "%LOCALAPPDATA%\LMFT-Celestriad"...
        mkdir "%LOCALAPPDATA%\LMFT-Celestriad" 2>nul
        curl.exe -L "https://github.com/schaleh12/Dual-Hemisphere-Wisdom-Model/releases/latest/download/lmft-sanctuary-club-v0.1.0-win64.zip" -o "%TEMP%\lmft-celestriad-bundle.zip"
        tar.exe -xf "%TEMP%\lmft-celestriad-bundle.zip" -C "%LOCALAPPDATA%\LMFT-Celestriad"
    )
)
cd /d "%ARCH_DIR%"
set "PYTHONPATH=%ARCH_DIR%src"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"
set "TQDM_DISABLE=1"

if exist "%ARCH_DIR%.venv\Scripts\pythonw.exe" (
    start "" /d "%ARCH_DIR%" "%ARCH_DIR%.venv\Scripts\pythonw.exe" -m lmft_architecture.chess_app --single-window %*
    exit /b 0
)

if exist "%ARCH_DIR%.venv\Scripts\python.exe" (
    start "" /d "%ARCH_DIR%" "%ARCH_DIR%.venv\Scripts\python.exe" -m lmft_architecture.chess_app --single-window %*
    exit /b 0
)

echo [Celestriad 1.0] Python .venv not found in this folder yet — opening the 5-Step Setup Wizard...
if exist "%~dp0Launch-Setup-Wizard.bat" (
    call "%~dp0Launch-Setup-Wizard.bat"
) else (
    call "%ARCH_DIR%Launch-Setup-Wizard.bat"
)
