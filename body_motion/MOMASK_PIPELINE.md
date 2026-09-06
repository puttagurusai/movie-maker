# MoMask → projface pipeline

**Repo:** [EricGuo5513/momask-codes](https://github.com/EricGuo5513/momask-codes) (CVPR 2024)  
**Why:** Fast, high-quality text-to-motion on HumanML3D; **pretrained**; output is already **`(T, 22, 3)` joints**.

## Fit for our pipeline

| Need | MoMask | Ours |
|------|--------|------|
| Text → body motion | Yes | Yes |
| HumanML3D 22 joints | Yes `(nframe, 22, 3)` | `humanml3d_joints22` |
| Fast inference | Masked transform (not slow diffusion) | Wanted |
| Pretrained | Yes (`prepare/download_models.sh`) | Prefer |
| No conda | `pip install -r requirements` (Py 3.10) | Yes |
| Face / mouth | Not involved | Still face pipeline only |
| Blender skeleton | Export joints → our adapter | `apply_lmm_npz_to_smplx.py` |

**Verdict: YES — integrate as primary text→body engine.**  
Replace weak custom LMM for open text; keep Mixamo **clips** for known actions if desired.

```text
prompt
  → MoMask gen_t2m
  → joints (T,22,3) @ 20fps
  → tools/momask_infer.py  →  lmm_out.npz (our schema, resample 30fps)
  → tools/apply_lmm_npz_to_smplx.py
  → Action on SMPL-X_Armature
  → blender_receiver play
```

## Setup (normal venv, no conda) — already scaffolded

```powershell
cd C:\me\proj\projface_v1

# 1) Clone (done if third_party\momask-codes exists)
# git clone --depth 1 https://github.com/EricGuo5513/momask-codes.git third_party\momask-codes

# 2) Venv + deps (torch CPU or CUDA — use your preferred wheel)
python -m venv third_party\momask_venv
.\third_party\momask_venv\Scripts\pip.exe install -U pip
.\third_party\momask_venv\Scripts\pip.exe install torch torchvision torchaudio
.\third_party\momask_venv\Scripts\pip.exe install -r third_party\momask_requirements.txt
.\third_party\momask_venv\Scripts\pip.exe install git+https://github.com/openai/CLIP.git

# 3) Pretrained weights → third_party\momask-codes\checkpoints\t2m\
powershell -ExecutionPolicy Bypass -File tools\download_momask_models.ps1
```

NumPy 2.x patches applied under `third_party/momask-codes` (`np.float` → `np.float64`).

## Commands

```powershell
# Infer → pipeline NPZ (20fps → 30fps, humanml3d_joints22)
.\third_party\momask_venv\Scripts\python.exe tools\momask_infer.py `
  --prompt "a man is walking" --out "lmm train\momask_walk.npz" --motion_length 80 --no_ik

# Bake onto SMPL-X_Armature
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" `
  whole_body_retargeted.blend --python tools\apply_lmm_npz_to_smplx.py -- `
  --npz "lmm train\momask_walk.npz" --action momask_walk `
  --out body_motion\_momask_applied.blend
```

Or one-shot with Blender (if torch CUDA in venv + blender path set):

```powershell
.\third_party\momask_venv\Scripts\python.exe tools\momask_infer.py `
  --prompt "a man is walking" --out "lmm train\momask_walk.npz" --apply_blender
```

## Quality note

Custom LMM walk had bone-length CV ~0.2–0.3 (broken).  
MoMask walk sample: bone-length CV **~0.017** (good rigid skeleton).
