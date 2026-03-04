@echo off
title IACS - Inter-Agent Communication System
color 0B

cd /d "%~dp0"

echo.
echo  ====================================================
echo        IACS // Inter-Agent Communication System
echo  ====================================================
echo.

REM Start Python IACS server in background (must run from project root for package imports)
echo  [1/2] Starting IACS Server on http://localhost:9100 ...
start "IACS Server" /MIN cmd /c "cd /d "%~dp0\.." && python -m uvicorn iacs.server.main:app --host localhost --port 9100 --log-level info"

timeout /t 3 /nobreak >nul

REM Start Node.js dashboard
echo  [2/2] Starting Dashboard on http://localhost:9110 ...
cd dashboard
if not exist node_modules (
    echo  Installing dashboard dependencies...
    call npm install --production
)
node server.js
