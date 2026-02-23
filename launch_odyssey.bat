@echo off
setlocal

echo ===================================================
echo   ManifoldGL Phaser 9: The Odyssey Launcher
echo ===================================================
echo.

:: 1. Environment Setup
echo [*] Activating Unsloth Environment...
if exist "unsloth_env\Scripts\activate.bat" (
    call unsloth_env\Scripts\activate.bat
    echo [OK] Active Virtual Environment: unsloth_env
) else (
    echo [ERROR] unsloth_env not found! Training cannot proceed.
    pause
    exit /b 1
)

:: 2. Launch Training
echo.
echo [*] Launching The Odyssey Training Run...
echo     - Script: train_odyssey.py
echo     - Log: train_odyssey.log
echo.

python train_odyssey.py

echo.
echo [!] Training script exited.
pause
