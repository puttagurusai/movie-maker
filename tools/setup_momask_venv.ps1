# Setup MoMask with normal venv (NO conda)
# Run from project root:  powershell -ExecutionPolicy Bypass -File tools\setup_momask_venv.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$MoMask = Join-Path $Root "third_party\momask-codes"
$Venv = Join-Path $Root "third_party\momask_venv"

if (-not (Test-Path $MoMask)) {
    Write-Error "Clone missing: $MoMask — run git clone https://github.com/EricGuo5513/momask-codes.git third_party/momask-codes"
}

Write-Host "Creating venv: $Venv"
python -m venv $Venv
$py = Join-Path $Venv "Scripts\python.exe"
$pip = Join-Path $Venv "Scripts\pip.exe"

& $py -m pip install -U pip setuptools wheel

# Flexible deps (modern torch; MoMask tested on 1.12 but works with newer often)
Write-Host "Installing PyTorch + deps (this may take a few minutes)..."
& $pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Host "CUDA 12.1 wheels failed; trying default torch..."
    & $pip install torch torchvision torchaudio
}

& $pip install numpy scipy matplotlib tqdm PyYAML einops ftfy regex packaging gdown
& $pip install "vector-quantize-pytorch==1.6.30"
& $pip install git+https://github.com/openai/CLIP.git

# Optional viz (not required for pipeline npz)
& $pip install smplx trimesh Pillow 2>$null

Write-Host "Downloading pretrained models into momask-codes/checkpoints ..."
Push-Location $MoMask
# Use bash script if git-bash exists; else python gdown helper
if (Get-Command bash -ErrorAction SilentlyContinue) {
    bash prepare/download_models.sh
} else {
    Write-Host "bash not found — run download via Python gdown..."
    & $py -c @"
import os, gdown, zipfile
from pathlib import Path
os.chdir(r'$MoMask')
Path('checkpoints').mkdir(exist_ok=True)
# Official models zip from README google drive folder helper
# Fallback: user runs prepare/download_models.sh in Git Bash / WSL
print('If auto-download fails, open README and download models to third_party/momask-codes/checkpoints/')
print('Script: third_party/momask-codes/prepare/download_models.sh')
"@
    if (Test-Path "prepare\download_models.sh") {
        # Try invoking with sh if available
        if (Get-Command sh -ErrorAction SilentlyContinue) {
            sh prepare/download_models.sh
        }
    }
}
Pop-Location

Write-Host ""
Write-Host "DONE. Activate:"
Write-Host "  $Venv\Scripts\Activate.ps1"
Write-Host "Test:"
Write-Host "  $py $Root\tools\momask_infer.py --prompt `"a man is walking`" --out `"lmm train\momask_walk.npz`""
