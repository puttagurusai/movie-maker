# Download MoMask HumanML3D pretrained weights (no conda)
# Requires: third_party\momask_venv with gdown, or system python with gdown

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$MoMask = Join-Path $Root "third_party\momask-codes"
$VenvPy = Join-Path $Root "third_party\momask_venv\Scripts\python.exe"
$py = if (Test-Path $VenvPy) { $VenvPy } else { "python" }

$ckpt = Join-Path $MoMask "checkpoints\t2m"
New-Item -ItemType Directory -Force -Path $ckpt | Out-Null
Set-Location $ckpt

Write-Host "Using $py"
& $py -m pip install -q "gdown>=4.7.1"
Write-Host "Downloading humanml3d_models.zip ..."
& $py -m gdown --fuzzy "https://drive.google.com/file/d/1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0/view?usp=sharing"
if (-not (Test-Path "humanml3d_models.zip")) {
    Write-Error "Download failed. Manually download from MoMask README Google Drive into $ckpt"
}
Write-Host "Unzipping..."
Expand-Archive -Path "humanml3d_models.zip" -DestinationPath "." -Force
Remove-Item "humanml3d_models.zip" -ErrorAction SilentlyContinue
Write-Host "Done. Contents:"
Get-ChildItem -Recurse -Depth 2 | Select-Object FullName
