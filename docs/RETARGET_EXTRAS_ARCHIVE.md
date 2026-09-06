# Archived retarget / BVH extras (reference only)

**Do not use on the default path.** Manual Rokoko after BVH is the source of truth.
These were pipeline experiments that hurt limb quality, root height, and feet.
Code remains in `tools/bvh_rokoko_direct_to_smplx.py` and
`third_party/momask-codes/visualization/joints2bvh.py` but is **off by default**.

## BVH formation (`joints2bvh.py`) — archived extras

| Extra | What it did | Why archived |
|-------|-------------|--------------|
| `canonicalize_facing_joints` | Force face +Z | NaN on 180° face; random line |
| `ease_rest_in_out` + custom rest quats as BVH | Rest pads in BVH | Wrong bone axes → limbs collapse |
| `push_knees_lateral` | Force knees out | Looked artificial vs manual |
| `push_elbows_forward` / `push_elbows_lateral` | Elbow plane fixes | Inward/outward collapse |
| `strip_bone_axis_twist` on arms/legs | Remove twist | Broke poles / hands (if on legs/arms) |

**Current default convert:** stock MoMask (IK + foot_ik) + `our_momask_template.bvh` only.

**On by default after bake (our side, not MoMask):**
- Heading align: rotate pelvis path so travel faces SMPL-X −Y (MoMask heading is random).
- Constant floor + jump-safe penetration lift. Disable with `MOMASK_ALIGN=0` / `MOMASK_FLOOR_PLANT=0`.

## Retarget (`bvh_rokoko_direct_to_smplx.py`) — archived extras

| Extra | Env / flag to re-enable | Why archived |
|-------|-------------------------|--------------|
| Auto scale hips→head | `--auto-scale` | Manual Rokoko does not; mismatches size |
| Floor plant (jump-safe) | `--floor-plant` | Manual does not; can fight feet |
| Knock-knee hip bias | `MOMASK_KNOCK_KNEE=1` | Manual does not |
| Custom rest pads on Action | `MOMASK_REST_PADS=1` | Idle rest is blender_receiver JSON only |
| Shoulder forward bias | `SHOULDER_FWD_DEG` / `--shoulder-fwd-deg` | Manual does not (default 0) |
| Horizontal-only root (Z=0) | was always on briefly | Prefer full local delta like constraint bake |

## Manual Rokoko path (default again)

1. Import BVH (`global_scale=1`, no auto scale)
2. Optional hips→pelvis align only if needed for constraint setup
3. Helper bones from SMPL-X rest → source parents (Rokoko style)
4. `COPY_ROTATION` (all pairs) + `COPY_LOCATION` on pelvis
5. Bake Action
6. Clear constraints, slim Action-only save

Updated: 2026-08-14
