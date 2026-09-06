# Body motion Actions (Blender)

File: `body_motion/whole_body_motions.blend`
Armature: `SMPL-X_Armature`

| Action | Frames | Use for |
|--------|--------|---------|
| `rest` | 1–24 | director / polish |
| `wave` | 1–48 | director / polish |
| `shrug` | 1–36 | director / polish |
| `talk_open` | 1–36 | director / polish |
| `point` | 1–40 | director / polish |
| `think` | 1–56 | director / polish |
| `recoil` | 1–32 | director / polish |
| `celebrate` | 1–40 | director / polish |
| `sit` | 1–48 | director / polish |
| `nod` | 1–32 | director / polish |

## How to edit in Blender

1. Open `body_motion/whole_body_motions.blend`
2. Select **SMPL-X_Armature** → **Pose Mode**
3. **Dope Sheet → Action Editor** → dropdown pick action (wave, shrug, …)
4. Pose bones, **I → Rotation** to keyframe
5. Play timeline (Space) to preview
6. **File → Save**

Later export (not now): Action → NLA strip → glTF with animation.
