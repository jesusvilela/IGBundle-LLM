@echo off
title IGBundle Tomograph - Live Telemetry
color 0A

echo.
echo  ====================================================
echo        IGBUNDLE // TOMOGRAPH
echo        Real-Time Model Telemetry
echo  ====================================================
echo.
echo  LM Studio: http://192.168.56.1:1234
echo  Dashboard: http://localhost:8050
echo.
echo  ====================================================
echo.

REM Activate environment
if exist "unsloth_env\Scripts\activate.bat" (
    call unsloth_env\Scripts\activate.bat
)

REM Install deps silently
pip install websockets websocket-client dash dash-bootstrap-components requests >nul 2>&1

REM Start tomograph server (background)
start /B "Tomograph" python run_realtime_tomograph.py ^
    --checkpoint output/igbundle_v2_cp300_merged ^
    --lmstudio http://192.168.56.1:1234 ^
    --poll-interval 0.3

REM Wait for server
timeout /t 2 /nobreak >nul

REM Start dashboard (foreground)
python run_tomograph_dashboard.py --lmstudio http://192.168.56.1:1234
