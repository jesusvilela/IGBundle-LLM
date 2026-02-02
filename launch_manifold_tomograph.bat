@echo off
title IGBundle Manifold Tomograph
color 0A

cd /d "%~dp0"

echo.
echo  ====================================================
echo        IGBUNDLE // MANIFOLD TOMOGRAPH
echo  ====================================================
echo.

REM Kill any existing Python processes for tomograph
taskkill /F /FI "WINDOWTITLE eq Telemetry*" >nul 2>&1
del /F /Q tomograph_state.json >nul 2>&1

REM Activate environment
if exist "unsloth_env\Scripts\activate.bat" (
    call unsloth_env\Scripts\activate.bat
    echo  [OK] Virtual environment activated
)

echo.
echo  LM Studio: http://192.168.56.1:1234
echo  Dashboard: http://localhost:8050
echo  State:     %CD%\tomograph_state.json
echo.
echo  ====================================================
echo.

REM Start telemetry in background
start "Telemetry" /MIN cmd /c "python run_realtime_tomograph.py --checkpoint output/igbundle_v2_cp300_merged --lmstudio http://192.168.56.1:1234 --http-output tomograph_state.json --poll-interval 0.3"

REM Wait for telemetry to write first frame
echo  Waiting for telemetry to start...
timeout /t 3 /nobreak >nul

REM Start dashboard
echo  Starting dashboard...
python run_manifold_tomograph.py --lmstudio http://192.168.56.1:1234 --state-file tomograph_state.json
