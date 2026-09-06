# Blender-only body motion (phase 1)

**Goal:** Author realistic body actions on the real rig in Blender.  
Web/viewer procedural eulers are **paused** for motion quality — director JSON can stay later.

## Source of truth

| Item | Path |
|------|------|
| Original scene | `whole body.blend` |
| Working motion file | `body_motion/whole_body_motions.blend` |
| Action list | `body_motion/ACTIONS.md` |
| Preview stills | `body_motion/previews/*.png` |
| Builder script | `tools/blender_body_motion.py` |

Armature: **`SMPL-X_Armature`** (SMPL-X bone names).  
Face stays on `FaceRoot` → neck; body deforms via Armature modifier on `BodyMesh`.

## Generate / refresh actions

```powershell
cd C:\me\proj\projface_v1
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" `
  --background "whole body.blend" `
  --python tools/blender_body_motion.py
```

Creates Actions: `rest`, `wave`, `shrug`, `talk_open`, `point`, `think`, `recoil`, `celebrate`, `sit`, `nod`.

## Polish in Blender UI (main work)

1. Open `body_motion/whole_body_motions.blend`
2. Select **SMPL-X_Armature** → **Pose Mode** (`Ctrl+Tab`)
3. **Dope Sheet** editor → mode **Action Editor**
4. Dropdown: pick `wave` / `shrug` / …
5. Pose bones (G/R on bones), **I → Rotation** to key
6. **Space** play; scrub timeline
7. **File → Save**

### Tips for realism

- Start from **rest** (arms slightly down, soft elbows) — not stiff T-pose.
- Animate **spine + collar + shoulder + elbow + wrist** together (not arm only).
- Contact poses (`think` → chin): pose hand near face by eye; fine IK later.
- Sit: hips + knees + ankles + slight spine bend.
- Prefer fewer strong keys with Bézier over many tiny eulers.

### Bone axes (this rig)

| Bone | Raise arm | Elbow flex |
|------|-----------|------------|
| `right_shoulder` | local **Z −** | — |
| `left_shoulder` | local **Z +** | — |
| `right_elbow` | — | local **Z −** |
| `left_elbow` | — | local **Z +** |

Script helpers match this; if a pose looks mirrored, flip sign in Action Editor.

## Optional: Mixamo / mocap into same file

1. Import FBX with animation into a **second** armature.
2. Retarget onto `SMPL-X_Armature` (Blender retarget / Rokoko / manual copy).
3. **Bake Action** → rename to `wave`, `walk`, etc.
4. Delete helper armature.

Same Action names = drop-in later for web export.

## Later (not now): export for web

When motions look good:

1. NLA or per-Action export  
2. **File → Export → glTF 2.0** → animation only / skinned body  
3. Drop into `web_viewer/body_assets/animations/`  
4. Wire manifest (existing viewer path)

Until then: **all body motion work stays in Blender.**

## Face + body together

Face morphs still live in `face.blend` / ARKit pipeline.  
For expression + body shot in one scene: keep using `whole body.blend` / motion file with shape keys on head meshes (UDP receiver optional later).

## Checklist

- [ ] Run builder script once  
- [ ] Open motion blend, fix `wave` / `shrug` / `think` by eye  
- [ ] Re-render previews (re-run script **or** F12 stills)  
- [ ] Add more Actions as needed (`hands_reject`, `walk`, …)  
- [ ] Only after happy: export GLB for viewer  
