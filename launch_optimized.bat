@echo off
setlocal

echo ===================================================
echo   ManifoldGL Optimized Launcher
echo ===================================================
echo.

:: 1. Environment Setup
echo [*] Checking Environment...
if exist "unsloth_env\Scripts\activate.bat" (
    call unsloth_env\Scripts\activate.bat
    echo [OK] Active Virtual Environment: unsloth_env
) else (
    echo [WARN] unsloth_env not found. Using system python...
)

:: 2. Check CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')" > cuda_check.tmp
set /p CUDA_STATUS=<cuda_check.tmp
del cuda_check.tmp
echo [*] %CUDA_STATUS%

:: 3. Open Documentation Site
echo.
echo [*] Opening Documentation Portal...
start "" "docs\index.html"

:: 4. Launch Backend
echo.
echo [*] Launching Neural Backend (Qwen 7B + IGBundle)...
echo     - Mode: 4-bit Quantization (Unsloth)
echo     - Checkpoint: Step 50
echo.
echo [!] Please wait for model loading...
echo.

python app.py --checkpoint output/igbundle_qwen7b_riemannian/checkpoint-50

pause
