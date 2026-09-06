# MoMask → clean body (same path as official demos)

## What MoMask GitHub Visualization actually does

From [EricGuo5513/momask-codes](https://github.com/EricGuo5513/momask-codes) **Visualization** section:

1. Motions are rendered in **Blender** on **Mixamo** characters (T-pose + skeleton).
2. They **do not** drive SMPL-X directly for demos.
3. Retarget tool:
   - **Rokoko** → they report **large foot error**
   - **KeeMap Rig Transfer** → preferred, more precise  
     https://github.com/nkeeline/Keemap-Blender-Rig-ReTargeting-Addon/releases  
     Tutorial: https://www.youtube.com/watch?v=EG-VCMkVpxg
4. Bone map + **per-bone correction factors** (not bare name mapping):  
   `third_party/momask-codes/assets/mapping.json`  
   (or `mapping6.json` if needed)

That is why their Mixamo demos look clean: **MoMask BVH ≈ same bone language as Mixamo**, plus KeeMap’s **CorrectionFactor / QuatCorrectionFactor** per bone.

---

## Why our direct MoMask → SMPL-X twists

| Hop | Skeletons | Result |
|-----|-----------|--------|
| MoMask BVH → Mixamo | Same naming family, similar axes | Works in their demos (KeeMap) |
| Mixamo → SMPL-X | We already solved with `FINAL_BONE_MAP` + helper-bone retarget | **User OK (walk)** |
| MoMask BVH → SMPL-X **direct** | Different rest axes + collar lengths | Twist stomach / collars |

So: **do not fight SMPL-X with Rokoko from MoMask BVH.**  
Use the **same intermediate as the paper demos**, then our **already-perfect Mixamo→SMPL-X**.

```text
prompt
  → MoMask joints (T,22,3)
  → momask_walk.bvh          (official MoMask IK)
  → KeeMap + mapping.json    (official demo retarget)
  → Mixamo character posed
  → tools/retarget_final_one.py  (our FINAL_BONE_MAP)
  → Action on SMPL-X_Armature
```

---

## Path A — Official demo style (recommended)

### 1) MoMask BVH (you already have this)

```powershell
# if needed: joints → BVH
.\third_party\momask_venv\Scripts\python.exe tools\momask_joints_to_bvh.py `
  --npz "lmm train\momask_walk.npz" --out "lmm train\momask_walk.bvh" --ik
```

### 2) Install KeeMap (once)

1. Download release: https://github.com/nkeeline/Keemap-Blender-Rig-ReTargeting-Addon/releases  
2. Blender → Edit → Preferences → Add-ons → Install → enable **KeeMapRig**

### 3) Mixamo character (T-pose with skin)

- Mixamo → any character → **T-Pose** → download **FBX with skin**  
- Or reuse a T-pose FBX if you have one (not only animation clips)

### 4) KeeMap transfer (exactly their README)

1. Import `momask_walk.bvh`  
2. Import Mixamo T-pose FBX  
3. **Shift-select** source BVH armature + Mixamo armature  
4. **Pose Mode** → KeeMapRig panel (N-panel)  
5. Bone mapping file:  
   `C:\me\proj\projface_v1\third_party\momask-codes\assets\mapping.json`  
   → **Read In Bone Mapping File**  
6. Set Source Rig / Destination Rig names  
7. Number of Samples ≈ clip length (e.g. 119)  
8. **Transfer Animation from Source Destination**  
9. Play Mixamo — should match their demo quality (no collar/stomach mess)

### 5) Bake Mixamo action → export FBX (or keep in scene)

Export animated Mixamo as FBX (or leave armature in `.blend`).

### 6) Our proven hop → SMPL-X

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" `
  "C:\me\proj\projface_v1\whole_body_retargeted.blend" --background `
  --python tools\retarget_final_one.py -- `
  YourMixamoClip.fbx momask_walk
```

Uses `body_motion/FINAL_BONE_MAP.json` (walk already validated).

---

## Path B — KeeMap straight to SMPL-X (optional later)

Same KeeMap, but Destination = `SMPL-X_Armature`, with a **new** mapping file:

- Source: `Hips`, `LeftArm`, …  
- Dest: `pelvis`, `left_shoulder`, …  
- Plus hand-tuned **CorrectionFactor** / **QuatCorrectionFactor** (like their Mixamo map)

This is more work once; Path A reuses maps that already exist.

---

## Why Rokoko failed on MoMask but worked on Mixamo clips

| Source | Our experience | MoMask authors |
|--------|----------------|----------------|
| Mixamo FBX clips → SMPL-X | OK with FINAL map + helpers | n/a |
| MoMask BVH → character | Twist / feet issues | **Rokoko foot error; use KeeMap** |

Rokoko is fine when **rest poses already match** (Mixamo→SMPL-X with helpers).  
MoMask BVH needs **KeeMap-style correction factors**, which their `mapping.json` stores.

---

## Files you already have

| File | Role |
|------|------|
| `third_party/momask-codes/assets/mapping.json` | MoMask BVH → Mixamo (official) |
| `body_motion/FINAL_BONE_MAP.json` | Mixamo → SMPL-X (ours, validated) |
| `tools/retarget_final_one.py` | Automated Mixamo → SMPL-X bake |
| `lmm train/momask_walk.bvh` | Source motion |

---

## Bottom line

**Yes — we can solve it the same way their Mixamo demos do:**

1. **KeeMap + their mapping** → clean motion on Mixamo  
2. **Our Mixamo→SMPL-X retarget** → clean motion on our body  

Do **not** expect Rokoko alone on MoMask BVH → SMPL-X to look like their gallery.
