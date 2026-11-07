@echo off
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%LMFT-Architecture\src\lmft_architecture\installer_wizard.py" (
    set "ROOT=%LOCALAPPDATA%\LMFT-Celestriad\"
    if not exist "%LOCALAPPDATA%\LMFT-Celestriad\LMFT-Architecture\src\lmft_architecture\installer_wizard.py" (
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
    start "" "LMFT-Architecture\.venv\Scripts\pythonw.exe" "LMFT-Architecture\src\lmft_architecture\installer_wizard.py"
    exit /b 0
)

where pythonw.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start "" pythonw.exe "LMFT-Architecture\src\lmft_architecture\installer_wizard.py"
    exit /b 0
)

where python.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python.exe "LMFT-Architecture\src\lmft_architecture\installer_wizard.py"
    exit /b 0
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe" (
    start "" "%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe" "LMFT-Architecture\src\lmft_architecture\installer_wizard.py"
    exit /b 0
)

echo [Celestriad 1.0] Installing official Python 3.11 via Windows Package Manager (winget)...
winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
if exist "%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe" (
    start "" "%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe" "LMFT-Architecture\src\lmft_architecture\installer_wizard.py"
    exit /b 0
)
python "LMFT-Architecture\src\lmft_architecture\installer_wizard.py"
