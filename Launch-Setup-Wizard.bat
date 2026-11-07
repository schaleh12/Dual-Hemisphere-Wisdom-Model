@echo off
setlocal
set "ARCH_DIR="
if exist "%~dp0src\lmft_architecture\installer_wizard.py" (
    set "ARCH_DIR=%~dp0"
) else if exist "%~dp0LMFT-Architecture\src\lmft_architecture\installer_wizard.py" (
    set "ARCH_DIR=%~dp0LMFT-Architecture\"
) else (
    set "ARCH_DIR=%LOCALAPPDATA%\LMFT-Celestriad\"
    if not exist "%LOCALAPPDATA%\LMFT-Celestriad\src\lmft_architecture\installer_wizard.py" (
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
    start "" "%ARCH_DIR%.venv\Scripts\pythonw.exe" "%ARCH_DIR%src\lmft_architecture\installer_wizard.py"
    exit /b 0
)

for %%P in (
    "C:\Program Files\Python312\pythonw.exe"
    "C:\Program Files\Python311\pythonw.exe"
    "C:\Program Files\Python313\pythonw.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\pythonw.exe"
) do (
    if exist %%P (
        start "" %%P "%ARCH_DIR%src\lmft_architecture\installer_wizard.py"
        exit /b 0
    )
)

python -c "import tkinter" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start "" pythonw.exe "%ARCH_DIR%src\lmft_architecture\installer_wizard.py"
    exit /b 0
)

echo [Celestriad 1.0] Installing official Python 3.11 via Windows Package Manager (winget)...
winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
if exist "%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe" (
    start "" "%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe" "%ARCH_DIR%src\lmft_architecture\installer_wizard.py"
    exit /b 0
)
python "%ARCH_DIR%src\lmft_architecture\installer_wizard.py"
