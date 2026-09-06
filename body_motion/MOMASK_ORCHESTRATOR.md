# MoMask + Orchestrator integration

Face pipeline (lipsync + expressions) is **unchanged**.  
MoMask only drives **body** via SMPL-X Actions.

## Enable (testing: MoMask for ALL body)

```powershell
$env:USE_MOMASK = "1"      # default on
$env:MOMASK_ALL = "1"      # every beat uses MoMask (not catalog talk_open)
$env:MOMASK_GPU_ID = "0"   # gen_t2m on GPU
# MOMASK_SYNC defaults ON when USE_MOMASK=1

python orchestrator_agents.py
```

**Speed:**
- **gen_t2m alone** = text→BVH (GPU, relatively fast)
- **Integration** previously = gen + **2× Blender** cold starts (slow)
- **Now** = gen + **1× Blender** (`momask_bvh_to_smplx_fast.py`); **2nd run same prompt = cache**

Catalog-only again: `$env:USE_MOMASK="0"`

## Data flow

```text
User chat
  → BodyDirector
       body_mode: catalog | momask | auto
       humanml_prompt: "a person walks forward …"   # HumanML3D style
       dialogue_text + emotion  → FACE (TTS, lips, brows, …)
  → if momask:
       gen_t2m.py (official) → *_ik.bvh
       KeeMap → Mixamo
       working Rokoko map → SMPL-X Action
       cache: body_motion/momask_cache/
  → UDP
       type=viseme / emotion / head   (face)
       type=body action=momask_<hash> library_blend=…  (body)
  → blender_receiver plays Action on SMPL-X_Armature
```

## When MoMask runs

| Beat | Engine |
|------|--------|
| wave / talk while standing | **catalog** |
| walking / dancing / open caption | **momask** (if `USE_MOMASK=1`) |
| explicit `body_mode=momask` + caption | **momask** |

## Files

| Path | Role |
|------|------|
| `face_agents/momask_body_pipeline.py` | gen + retarget + cache |
| `tools/momask_keemap_exact.py` | BVH → Mixamo (KeeMap) |
| `tools/retarget_keemap_mixamo_rsl.py` | Mixamo → SMPL-X (working map) |
| `final hml3dto smpl.json` | **Canonical** Rokoko custom names (HML3D/Mixamo → SMPL-X) |
| `body_motion/momask_cache/` | cached Actions + meta JSON |

## Face stays separate

- Mouth/jaw: `lips_agent` + wav2arkit only  
- Expression: Brain + emotion_map  
- MoMask Actions must **not** include face shape keys (adapter leaves face rest)

## First-run note

First MoMask beat is **slow** (model + Blender retarget). Later identical prompts hit cache.
