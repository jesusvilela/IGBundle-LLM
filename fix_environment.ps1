# IGBundle Environment Fix Script
# Run this in PowerShell as Administrator

Write-Host "=== IGBundle Environment Fix ===" -ForegroundColor Cyan

# Step 1: Check current Python
Write-Host "`n[1/6] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Current: $pythonVersion"

if ($pythonVersion -match "3\.13") {
    Write-Host "WARNING: Python 3.13 detected - this is too new for many ML packages!" -ForegroundColor Red
    Write-Host "Recommendation: Install Python 3.11.x from python.org" -ForegroundColor Yellow
}

# Step 2: Create/recreate virtual environment with correct Python
Write-Host "`n[2/6] Setting up virtual environment..." -ForegroundColor Yellow

$envPath = "H:\LLM-MANIFOLD\igbundle-llm\venv_fixed"
if (Test-Path $envPath) {
    Write-Host "Removing old environment..."
    Remove-Item -Recurse -Force $envPath
}

# Try to find Python 3.11 or 3.10
$pythonPaths = @(
    "C:\Python311\python.exe",
    "C:\Python310\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
)

$selectedPython = $null
foreach ($p in $pythonPaths) {
    if (Test-Path $p) {
        $ver = & $p --version 2>&1
        Write-Host "Found: $p ($ver)"
        $selectedPython = $p
        break
    }
}

if ($selectedPython) {
    Write-Host "Creating venv with $selectedPython..."
    & $selectedPython -m venv $envPath
} else {
    Write-Host "Python 3.10/3.11 not found. Please install Python 3.11 first." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/release/python-3119/" -ForegroundColor Cyan
    exit 1
}

# Step 3: Activate and install packages
Write-Host "`n[3/6] Installing compatible packages..." -ForegroundColor Yellow
$activateScript = "$envPath\Scripts\Activate.ps1"

# Create requirements file with pinned versions
$requirements = @"
# Core ML Stack - Compatible versions for Windows + CUDA 12.4
torch==2.4.1+cu124
torchvision==0.19.1+cu124
torchaudio==2.4.1+cu124
--extra-index-url https://download.pytorch.org/whl/cu124

# Transformers ecosystem
transformers==4.44.2
accelerate==0.33.0
peft==0.12.0
datasets==2.20.0
safetensors==0.4.4

# Quantization (use older stable version)
bitsandbytes==0.43.3

# Do NOT install triton on Windows - it's Linux only
# torchao is optional and problematic

# Training utilities
pyyaml>=6.0
tqdm>=4.66.0
sentencepiece>=0.2.0
"@

$requirements | Out-File -FilePath "$envPath\requirements_fixed.txt" -Encoding UTF8

Write-Host "Requirements file created at: $envPath\requirements_fixed.txt"
Write-Host "`nTo complete setup, run these commands manually:" -ForegroundColor Green
Write-Host "  cd H:\LLM-MANIFOLD\igbundle-llm" -ForegroundColor White
Write-Host "  .\venv_fixed\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  pip install --upgrade pip" -ForegroundColor White
Write-Host "  pip install -r venv_fixed\requirements_fixed.txt" -ForegroundColor White

# Step 4: Network fix
Write-Host "`n[4/6] Applying network port exhaustion fix..." -ForegroundColor Yellow
Write-Host "This requires Administrator privileges."

try {
    # Increase ephemeral port range
    netsh int ipv4 set dynamicport tcp start=1025 num=64510
    netsh int ipv4 set dynamicport udp start=1025 num=64510
    Write-Host "Port range extended successfully." -ForegroundColor Green
} catch {
    Write-Host "Failed to extend port range. Run as Administrator." -ForegroundColor Red
}

# Step 5: GPU stability recommendations
Write-Host "`n[5/6] GPU Stability Recommendations..." -ForegroundColor Yellow
Write-Host @"
Your RTX 3060 Ti had 8 driver crashes this week. Recommendations:
1. Update NVIDIA drivers to latest Game Ready or Studio driver
2. Disable NVIDIA Broadcast during training (it uses GPU resources)
3. Set power management to 'Prefer Maximum Performance' in NVIDIA Control Panel
4. Consider underclocking GPU slightly if thermal throttling occurs
5. The 5-10 second sleep in compute_loss helps prevent PSU trips - keep it
"@

# Step 6: Create launch script
Write-Host "`n[6/6] Creating launch script..." -ForegroundColor Yellow

$launchScript = @'
# launch_training.ps1 - Use this to start training

# Activate environment
& "H:\LLM-MANIFOLD\igbundle-llm\venv_fixed\Scripts\Activate.ps1"

# Set environment variables for stability
$env:CUDA_LAUNCH_BLOCKING = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# Disable Triton (causes issues on Windows)
$env:TRITON_ENABLED = "0"
$env:USE_TRITON = "0"

# Change to project directory
cd H:\LLM-MANIFOLD\igbundle-llm

# Run training
python train.py --config configs/qwen25_7b_igbundle_lora.yaml

# For smoke test:
# python train.py --config configs/qwen25_7b_igbundle_lora.yaml --smoke_test
'@

$launchScript | Out-File -FilePath "H:\LLM-MANIFOLD\igbundle-llm\launch_training.ps1" -Encoding UTF8

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Install Python 3.11 if not already installed"
Write-Host "2. Run the pip install commands shown above"
Write-Host "3. Use .\launch_training.ps1 to start training"
