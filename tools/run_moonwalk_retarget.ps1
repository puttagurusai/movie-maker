# run_moonwalk_retarget.ps1
# Retargets moonwalk.npz onto whole body.blend → saves as body_motion\_moonwalk_applied.blend
# Original blend file is NEVER modified.
#
# Usage:
#   cd C:\me\proj\projface_v1
#   .\tools\run_moonwalk_retarget.ps1
#
# Or with a custom NPZ path:
#   .\tools\run_moonwalk_retarget.ps1 -Npz "C:\path\to\moonwalk.npz"
#
# Or with a custom output name:
#   .\tools\run_moonwalk_retarget.ps1 -Npz "lmm train\moonwalk.npz" -Action "moonwalk" -Out "body_motion\_moonwalk_applied.blend"

param(
    [string]$Npz    = "lmm train\moonwalk.npz",
    [string]$Action = "moonwalk",
    [string]$Out    = "body_motion\_moonwalk_applied.blend",
    [string]$Source = "whole body.blend"
)

$Root    = "C:\me\proj\projface_v1"
$Blender = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"

# Resolve absolute NPZ path
if (-not [System.IO.Path]::IsPathRooted($Npz)) {
    $NpzAbs = Join-Path $Root $Npz
} else {
    $NpzAbs = $Npz
}

if (-not (Test-Path $NpzAbs)) {
    Write-Error "NPZ file not found: $NpzAbs"
    Write-Host "Place moonwalk.npz in: $Root\lmm train\"
    exit 1
}

$BlendSrc  = Join-Path $Root $Source
$OutAbs    = Join-Path $Root $Out
$Script    = Join-Path $Root "tools\apply_npz_to_blend.py"

Write-Host "=== Moonwalk Retarget ===" -ForegroundColor Cyan
Write-Host "Source blend : $BlendSrc"
Write-Host "NPZ          : $NpzAbs"
Write-Host "Action name  : $Action"
Write-Host "Output blend : $OutAbs"
Write-Host ""

if (-not (Test-Path $BlendSrc)) {
    Write-Error "Source blend not found: $BlendSrc"
    exit 1
}

# Run Blender in background (original file untouched - Blender opens a copy)
& $Blender --background $BlendSrc `
    --python $Script `
    -- `
    --npz $NpzAbs `
    --action $Action `
    --out $OutAbs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "SUCCESS! Output saved to: $OutAbs" -ForegroundColor Green
    Write-Host ""
    Write-Host "To open result in Blender:"
    Write-Host "  & `"$Blender`" `"$OutAbs`""
} else {
    Write-Host ""
    Write-Host "ERROR: Blender exited with code $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Check output above for [npz_retarget] log lines"
}
