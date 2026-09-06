# Final Mixamo → SMPL-X bone map

**Status: FINAL** (validated: walk OK with Rokoko after hand/collar fix)

## Files

| File | Purpose |
|------|---------|
| `FINAL_BONE_MAP.json` | Canonical map (52 pairs) |
| `mixamo_to_smplx_bone_map.json` | Same map (Rokoko / tools) |
| `tools/retarget_final_one.py` | Batch one FBX with this map |
| `_proof_wave_final.blend` | Second proof clip (wave) |

## Arm chain (critical)

| Mixamo | SMPL-X |
|--------|--------|
| Shoulder | collar |
| Arm | shoulder |
| ForeArm | elbow |
| Hand | wrist |

Fingers: Mixamo `*1,*2,*3` → SMPL-X `*1,*2,*3`. Tips (`*4`) unmapped.

## Production file (all clips)

- **`whole_body_retargeted.blend`** (also copied to `whole body.blend`)
- **23/23** catalog Actions on `SMPL-X_Armature` via `tools/retarget_final_all.py`
- Report: `body_motion/retarget_all_report.json`
- Scene FPS set to **30**

## Proof clips

1. **walk** — user OK in Rokoko with this map  
2. **wave** — inplace root fix: `body_motion/_proof_wave_inplace.blend`

## Check wave in Blender

1. Open `body_motion/_proof_wave_final.blend`
2. Select `SMPL-X_Armature` → Action Editor → action **wave**
3. Play — arm should wave, fingers mapped, no T-pose arms

## When user says “do all”

```powershell
# Example loop (after wave OK):
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" `
  "C:\me\proj\projface_v1\whole body.blend" --background `
  --python tools\retarget_final_one.py -- Idle.fbx idle
```

Or tell the agent: **do all clips**.

## Rokoko note

Do **not** click **Build Bone List** without re-applying this map (auto-detect breaks arms).
