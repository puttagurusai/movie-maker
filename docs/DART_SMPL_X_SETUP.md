# DART + SMPL-X setup (body first; custom head later)

**Decision (locked):**

| Item | Choice |
|------|--------|
| Body | **SMPL-X male** only for DART |
| Motion engine | **DART** (text → SMPL-X pose) |
| Face now | **Default SMPL-X head** (from model) |
| Face later | Attach **our ARKit face** at neck (phase 2) |
| MotionLCM | Out of scope for this path |

Hybrid later:

```text
DART → SMPL-X body pose
our face agents → ARKit head (attached to neck)
```

---

## What is already in this repo

| Path | Role |
|------|------|
| `third_party/DART/` | Official DART code (git clone) |
| `models/smplx/` | Your SMPL-X assets |
| `third_party/DART/data/smplx_lockedhead_20230207/models_lockedhead/smplx/` | DART-expected body model layout (filled by setup script) |
| `scripts/setup_dart.ps1` | Bootstrap paths + checks |
| `body_motion/dart/` | Outputs / notes for our pipeline |

Upstream: https://github.com/zkf1997/DART  
Paper: DartControl (ICLR 2025)

---

## Phase plan

### Phase 1 — DART + SMPL-X body (NOW)

1. Install DART environment (GPU)  
2. Place pretrained **checkpoints** (Google Drive from DART README)  
3. Run text → motion demo  
4. Visualize SMPL-X body (pyrender or Blender add-on)  
5. **Keep default head** — do not attach custom face yet  

### Phase 2 — Our head (LATER)

1. Export / parent ARKit face to **neck** of DART-driven SMPL-X  
2. Body pose from DART; expressions from face agents  
3. Sync with TTS  

### Phase 3 — Product glue

1. Director: clip vs DART generate  
2. UDP/WS stream of SMPL-X params or bone eulers to viewer  

---

## Environment (important on Windows)

DART’s official `environment.yml` is **Linux + conda** (CUDA 11.8, PyTorch, pytorch3d).

**Recommended on this machine:**

### Option A — WSL2 Ubuntu (best match to upstream)

```bash
# inside WSL, from repo
cd /mnt/c/me/proj/projface_v1/third_party/DART
conda env create -f environment.yml
conda activate DART
```

### Option B — Native Windows (harder)

- Install **Miniconda** / Anaconda  
- Prefer a **Python 3.10** env (not 3.13)  
- Install PyTorch with CUDA matching your driver  
- `pytorch3d` on Windows is fragile — WSL is safer  

You currently have Python 3.13 / 3.11; DART expects older stack. Use a dedicated conda env.

---

## One-time data layout

DART reads:

```text
third_party/DART/data/
  smplx_lockedhead_20230207/
    models_lockedhead/
      smplx/
        SMPLX_MALE.npz
        SMPLX_FEMALE.npz   # DART loads both genders
        SMPLX_NEUTRAL.npz
```

Run from project root:

```powershell
.\scripts\setup_dart.ps1
py -3.11 scripts\patch_smplx_for_dart.py
```

This copies from `models/smplx/` into the DART tree, then pads missing hand-PCA/landmark keys so `smplx` can load (body motion works; hands use zero PCA for now).

**Preferred long-term:** download the official full locked-head pack from  
https://smpl-x.is.tue.mpg.de/ — zip `smplx_lockedhead_20230207` — and replace the patched files (no hand-PCA hacks).

---

## Checkpoints (required for demos)

**Status: merged from your download**  
`DART-20260725T103938Z-1-001/DART` → `third_party/DART/`

Primary BABEL / SMPL-X demo checkpoint:

```text
third_party/DART/mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt
```

Also present: `mvae/`, `policy_train/`, `data/seq_data_zero_male/`, `data/stand.pkl`, etc.

You can delete or archive the zip folder after verify:

```text
DART-20260725T103938Z-1-001/
```

Original Drive (if re-download needed):  
https://drive.google.com/drive/folders/1vJg3GFVPT6kr6cA0HrQGmiAEBE2dkaps

---

## Run first demo (after env + checkpoints)

```bash
cd third_party/DART
conda activate DART
# interactive text → motion (SMPL-X body + default head)
source ./demos/run_demo.sh
# or headless composition:
source ./demos/rollout.sh
```

Visualize:

```bash
python -m visualize.vis_seq --add_floor 1 --body_type smplx --translate_body 1 \
  --seq_path './path/to/generated/*.pkl'
```

Blender: install [SMPL-X Blender add-on](https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon), import DART’s exported `.npz` sequence (“add animation”).

---

## Head policy (explicit)

| Phase | Head |
|-------|------|
| **Now** | SMPL-X built-in head vertices (locked-head model) |
| **Later** | Strip / ignore body head region; parent **our** face to `neck` |

Do **not** mix ARKit face until DART body motion is verified on default head.

---

## Success criteria for Phase 1

- [ ] `setup_dart.ps1` reports SMPL-X files present  
- [ ] DART conda/WSL env imports `torch`, `smplx`, runs without crash  
- [ ] Checkpoint `checkpoint_300000.pt` present  
- [ ] Text prompt e.g. `wave` / `walk` produces `.pkl` sequence  
- [ ] Viewer shows **male SMPL-X** body moving (default head OK)  

Only then: Phase 2 custom head.

---

## Out of scope for this setup

- MotionLCM  
- MetaHuman  
- Gaussian splats  
- Attaching `face.blend` / ARKit yet  
- Web viewer stream (after body works)  
