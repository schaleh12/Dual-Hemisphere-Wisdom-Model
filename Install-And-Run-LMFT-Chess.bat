@echo off
setlocal
cd /d "%~dp0"
title LMFT-Chess — 1-Click Installer, Auto-Sync & Launcher

set "ARCH_DIR=%~dp0LMFT-Architecture"
if not exist "%ARCH_DIR%\pyproject.toml" set "ARCH_DIR=%~dp0"
set "VENV_PY=%ARCH_DIR%\.venv\Scripts\python.exe"

if /i not "%~1"=="--venv-only" if /i not "%~1"=="--direct" (
    if exist "%~dp0LMFT-Installer-Wizard.exe" (
        start "" "%~dp0LMFT-Installer-Wizard.exe"
        exit /b 0
    )
)

if not exist "%VENV_PY%" (
    echo ============================================================================
    echo   LMFT-Chess First-Time Setup — Creating Python Environment (.venv)
    echo ============================================================================
    python -m venv "%ARCH_DIR%\.venv"
    if errorlevel 1 (
        echo [ERROR] Python 3.10+ was not found on PATH. Please install Python 3.11+ from python.org.
        pause
        exit /b 1
    )
    "%VENV_PY%" -m pip install --upgrade pip
    "%VENV_PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    "%VENV_PY%" -m pip install -e "%ARCH_DIR%"
)

set "PYTHONPATH=%ARCH_DIR%\src"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"
set "TQDM_DISABLE=1"

echo Syncing latest code and shared chess wisdom with GitHub (schaleh12/Dual-Hemisphere-Wisdom-Model)...
"%VENV_PY%" -c "from lmft_architecture.github_auto_sync import fetch_remote_github_manifest, verify_or_sync_against_manifest; m = fetch_remote_github_manifest(timeout_sec=5.0); res = verify_or_sync_against_manifest('.', m, apply_sync=True); print('GitHub Sync Status:', res['status'], '| Updated files:', len(res['synced_files']))" 2>nul

if /i "%~1"=="--venv-only" (
    echo Python .venv and GitHub sync complete!
    exit /b 0
)

echo Launching LMFT-Chess...
start "" "%VENV_PY%" -m lmft_architecture.chess_app --single-window %*
exit /b 0
