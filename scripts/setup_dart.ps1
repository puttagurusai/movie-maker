# Bootstrap DART + SMPL-X paths for projface_v1
# Phase 1: body only (default SMPL-X head). Custom ARKit face = Phase 2.
#
# Usage (from repo root):
#   .\scripts\setup_dart.ps1
#   .\scripts\setup_dart.ps1 -DownloadCheckpoints   # needs gdown + network

param(
    [switch]$DownloadCheckpoints
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not (Test-Path (Join-Path $Root "third_party\DART\README.md"))) {
    $Root = Get-Location
}
Set-Location $Root

Write-Host "=== DART + SMPL-X setup ===" -ForegroundColor Cyan
Write-Host "Root: $Root"

$Dart = Join-Path $Root "third_party\DART"
if (-not (Test-Path (Join-Path $Dart "README.md"))) {
    Write-Host "Cloning DART..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path (Join-Path $Root "third_party") | Out-Null
    git clone --depth 1 https://github.com/zkf1997/DART.git (Join-Path $Root "third_party\DART")
}

$SmplxDst = Join-Path $Dart "data\smplx_lockedhead_20230207\models_lockedhead\smplx"
New-Item -ItemType Directory -Force -Path $SmplxDst | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Dart "data\smplx_lockedhead_20230207\models_lockedhead\smplh") | Out-Null

function Copy-Model($src, $dstName) {
    if (-not (Test-Path $src)) {
        Write-Host "  MISSING source: $src" -ForegroundColor Red
        return $false
    }
    $dst = Join-Path $SmplxDst $dstName
    Copy-Item $src $dst -Force
    $len = (Get-Item $dst).Length
    Write-Host "  OK $dstName ($len bytes) <- $src" -ForegroundColor Green
    return $true
}

Write-Host "`n[1] SMPL-X body models (locked-head layout for DART)"
$ok = $true
$ok = (Copy-Model (Join-Path $Root "models\smplx\SMPLX_MALE.npz") "SMPLX_MALE.npz") -and $ok
# female: prefer dedicated file, else locked_head tree
$fCandidates = @(
    (Join-Path $Root "models\smplx\SMPLX_FEMALE.npz"),
    (Join-Path $Root "models\smplx\smplx_locked_head.tar\smplx_locked_head\female\model.npz"),
    (Join-Path $Root "models\smplx\female\model.npz")
)
$fSrc = $fCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($fSrc) { $ok = (Copy-Model $fSrc "SMPLX_FEMALE.npz") -and $ok } else { Write-Host "  WARN no FEMALE model" -ForegroundColor Yellow }

$nCandidates = @(
    (Join-Path $Root "models\smplx\SMPLX_NEUTRAL.npz"),
    (Join-Path $Root "models\smplx\smplx_locked_head.tar\smplx_locked_head\neutral\model.npz")
)
$nSrc = $nCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($nSrc) { $ok = (Copy-Model $nSrc "SMPLX_NEUTRAL.npz") -and $ok }

Write-Host "`n[1b] Patch incomplete NPZ for smplx package"
$patch = Join-Path $Root "scripts\patch_smplx_for_dart.py"
if (Test-Path $patch) {
    try {
        py -3.11 $patch
    } catch {
        Write-Host "  patch failed (install smplx on py3.11): $_" -ForegroundColor Yellow
    }
}

Write-Host "`n[2] Checkpoints"
$ckpt = Join-Path $Dart "mld_denoiser\mld_fps_clip_repeat_euler\checkpoint_300000.pt"
if (Test-Path $ckpt) {
    Write-Host "  OK found $ckpt" -ForegroundColor Green
} else {
    Write-Host "  MISSING checkpoint_300000.pt" -ForegroundColor Yellow
    Write-Host "  Download Google Drive folder from DART README and merge into third_party\DART\"
    Write-Host "  https://drive.google.com/drive/folders/1vJg3GFVPT6kr6cA0HrQGmiAEBE2dkaps"
    if ($DownloadCheckpoints) {
        Write-Host "  Trying gdown --folder ..." -ForegroundColor Cyan
        $gdown = Get-Command gdown -ErrorAction SilentlyContinue
        if (-not $gdown) {
            py -3.11 -m pip install gdown -q
        }
        Push-Location $Dart
        try {
            py -3.11 -m gdown --folder "https://drive.google.com/drive/folders/1vJg3GFVPT6kr6cA0HrQGmiAEBE2dkaps" -O "_gdrive_dart" --remaining-ok
        } catch {
            Write-Host "  gdown failed: $_" -ForegroundColor Red
        }
        Pop-Location
    }
}

Write-Host "`n[3] Environment"
Write-Host "  Official DART uses conda env from third_party\DART\environment.yml (Linux-oriented)."
Write-Host "  Recommended: WSL2 Ubuntu + conda create -f environment.yml"
Write-Host "  Do NOT use system Python 3.13 for DART."

Write-Host "`n[4] Head policy"
Write-Host "  NOW:  default SMPL-X head (from body model)"
Write-Host "  LATER: attach project ARKit face at neck (Phase 2)"

New-Item -ItemType Directory -Force -Path (Join-Path $Root "body_motion\dart") | Out-Null
$status = @"
# DART setup status
generated: $(Get-Date -Format o)
smplx_dir: $SmplxDst
male: $(Test-Path (Join-Path $SmplxDst 'SMPLX_MALE.npz'))
female: $(Test-Path (Join-Path $SmplxDst 'SMPLX_FEMALE.npz'))
neutral: $(Test-Path (Join-Path $SmplxDst 'SMPLX_NEUTRAL.npz'))
checkpoint_default: $(Test-Path $ckpt)
phase: 1_body_default_head
next: install conda/WSL env + download checkpoints + run demos/run_demo.sh
"@
$statusPath = Join-Path $Root "body_motion\dart\SETUP_STATUS.txt"
Set-Content -Path $statusPath -Value $status -Encoding UTF8
Write-Host "`nWrote $statusPath" -ForegroundColor Cyan
Write-Host "Full guide: docs\DART_SMPL_X_SETUP.md" -ForegroundColor Cyan
Write-Host "=== done ===" -ForegroundColor Cyan
