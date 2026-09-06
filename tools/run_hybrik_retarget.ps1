# OFFICIAL HybrIK: joints → SMPLXLayer.hybrik() → SMPL-X params → our armature
# Usage:
#   powershell -File tools/run_hybrik_retarget.ps1 `
#     -Joints "third_party/momask-codes/generation/official_t2m_1786113934/joints/0/sample0_repeat0_len80.npy" `
#     -OutDir "body_motion/momask_cache/official_t2m_scale" `
#     -Name "walk_hybrik_official"

param(
    [Parameter(Mandatory = $true)][string]$Joints,
    [string]$OutDir = "body_motion/momask_cache/official_t2m_scale",
    [string]$Name = "walk_hybrik_official",
    [string]$Blend = "whole_body_retargeted.blend",
    [string]$Blender = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$pose = Join-Path $OutDir "${Name}_pose.npz"
$outBlend = Join-Path $OutDir "${Name}.blend"

Write-Host "[1/2] OFFICIAL HybrIK SMPLXLayer.hybrik joints → SMPL-X pose.npz"
python tools/hybrik_official_joints_to_pose.py --joints $Joints --out $pose
if ($LASTEXITCODE -ne 0) { throw "hybrik_official_joints_to_pose failed" }

Write-Host "[2/2] Bake official FK joints → SMPL-X_Armature"
& $Blender $Blend --background --python tools/hybrik_apply_to_smplx.py -- `
    --pose $pose --action $Name --root absolute --source fk --arm-aim src_target --out $outBlend
if ($LASTEXITCODE -ne 0) { throw "hybrik_apply_to_smplx failed" }

Write-Host "OK: $outBlend"
Write-Host "    pose: $pose (method=official_hybrik_SMPLXLayer.hybrik)"
