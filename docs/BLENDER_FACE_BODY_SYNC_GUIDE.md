# Blender guide: sync face + SMPL-X male body (manual)

Goal: one scene where **your ARKit face** sits correctly on the **SMPL-X male body** (scale, neck, rotation), then export a combined or paired setup for the viewer.

Use **Blender 4.5 or 5.1**.

---

## Files you use

| Asset | Path |
|--------|------|
| Face (preferred for morphs) | `face.blend` **or** `web_viewer/face.glb` |
| Body (skinned, head collapsed) | `web_viewer/body.glb` |
| Official male source (optional) | `models/smplx/male/model.npz` |
| Project root | `C:\me\proj\projface_v1` |

Start Blender:

```
"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
```

---

## Part A — New empty scene

1. **File → New → General**
2. Delete default cube (select → X → Delete)
3. Set units: **Scene properties** (scene icon) → **Units**
   - Unit System: **Metric**
   - Unit Scale: **1.0**
   - Length: **Meters**

---

## Part B — Import the body

1. **File → Import → glTF 2.0 (.glb/.gltf)**
2. Choose:  
   `C:\me\proj\projface_v1\web_viewer\body.glb`
3. Import options (if shown):
   - **Bone Dir:** Blender (Y Bone) or keep default
   - **Guess Original Bind Pose:** off is usually fine
4. In Outliner you should see something like:
   - `SMPL-X_Armature` (or armature + `BodyMesh`)
   - Bones including **`neck`**, **`head`**, **`pelvis`**, shoulders, etc.

### Check body orientation

1. Press **Numpad 1** (front view), **Numpad 7** (top)
2. Body should stand with **head up (+Z in Blender)**  
   - glTF often imports as **Y-up** converted to Blender **Z-up** automatically
3. Feet near **Z = 0** (or slightly above ground)

If the body is lying down or huge/tiny, fix **before** attaching the face (Part D scale).

---

## Part C — Import the face

### Option 1 — From GLB (same as viewer)

1. **File → Import → glTF 2.0**
2. Choose:  
   `C:\me\proj\projface_v1\web_viewer\face.glb`
3. You should see meshes like:
   - `head_lod0_ORIGINAL` (or similar)
   - eyes, teeth, tongue

### Option 2 — From face.blend (better if you edit morphs)

1. **File → Append** (not Open — Append keeps body in scene)
2. Open `C:\me\proj\projface_v1\face.blend`
3. Go into **Object/**
4. Select face-related objects (head, eyes, teeth, tongue) → **Append**

---

## Part D — Align face to body neck (manual sync)

### D1 — Find neck target

1. Select the **armature**
2. Go to **Pose Mode** (Ctrl+Tab → Pose Mode)
3. Select bone **`neck`** (or **`head`** if neck is hard to see)
4. **Viewport Overlays** → enable **Bones** / **Names** if needed  
   (Armature properties → Viewport Display → **Names**, **In Front**)
5. Note where the neck stump is (top of torso)

Tip: **Pose Mode → select neck → Shift+S → Cursor to Selected**  
This puts the 3D cursor on the neck joint — useful as a snap target.

### D2 — Center and scale the face

1. Object Mode
2. Select **all face meshes** (head + eyes + teeth + tongue)
3. **Ctrl+J** only if you want one mesh (optional; better keep separate and parent to empty)
4. Recommended: create an empty as face root
   - **Shift+A → Empty → Plain Axes** → name it `FaceRoot`
   - Select all face meshes → then **Shift+select FaceRoot** → **Ctrl+P → Object** (parent to empty)

5. Select `FaceRoot`
6. Scale roughly so head height ≈ **0.22–0.28 m** (SMPL-X head size)
   - Select face, **N panel → Item → Dimensions**  
   - Head height (Z) should be about **0.25 m** when aligned
   - Scale uniformly: **S** then type e.g. `0.15` or drag until it looks right next to shoulders

**Rule of thumb:** face width should match neck stump width (not wider than shoulders).

### D3 — Position on neck

1. Select `FaceRoot`
2. **G** grab, move so:
   - Chin sits just above the body neck stump  
   - Neck of face blends into body neck (no big gap, no sinking into chest)
3. Front view (**Numpad 1**): center left–right on body midline  
4. Side view (**Numpad 3**): head not too far forward/back  
5. Fine tune:
   - **G → Z** height  
   - **G → Y** depth (front/back in Blender)  
   - **R → Z** yaw if facing wrong way  
   - **R → X** pitch if looking up/down  

### D4 — Parent face to neck bone (sync when body moves)

This is what keeps face and body **in sync** when the skeleton poses.

1. Select `FaceRoot` (Object Mode)
2. **Shift+select** the armature
3. Go to **Pose Mode**, select bone **`neck`**
4. **Ctrl+P → Bone**  
   (or: parent with `Bone` so FaceRoot follows neck)

Alternative (constraint, easier to undo):

1. Select `FaceRoot`
2. **Bone Constraint** is wrong object type — use **Object Constraint**:
   - **Object Constraint Properties → Add Object Constraint → Child Of**
   - Target: armature  
   - Bone: `neck`  
   - Click **Set Inverse**  
   - Adjust until face stays put on neck  

**Child Of** is good for testing; **Parent to Bone** is cleaner for export.

### D5 — Hide leftover body head (if any)

Your export already collapses the SMPL-X head. If you still see a blob:

1. Edit Mode on `BodyMesh`
2. Select upper head vertices → delete or hide  
**Or** scale bone `head` to 0.001 in Pose Mode (for preview only).

---

## Part E — Quick pose test (sync check)

1. Select armature → **Pose Mode**
2. Rotate **`spine3`** or **`left_shoulder`** slightly
3. Face should move with the torso/neck  
4. **Pose → Clear Transform → All** to reset

If face detaches or drifts → redo **Set Inverse** on Child Of, or re-parent to `neck`.

---

## Part F — Save working scene

**File → Save As:**

```
C:\me\proj\projface_v1\web_viewer\avatar_face_body_sync.blend
```

Keep this as the **source of truth** for alignment.

---

## Part G — Export for the web viewer

### Option 1 — Export body + face as two files (recommended first)

Keeps ARKit morphs simpler.

**Face only**

1. Select `FaceRoot` + all face children  
2. **File → Export → glTF 2.0**  
3. Path: `web_viewer/face.glb`  
4. Options:
   - Format: **glTF Binary (.glb)**
   - **Include → Selected Objects**
   - **Transform → +Y Up**
   - **Data → Mesh → Shape Keys** = ON (ARKit morphs)
   - Apply modifiers if any  

**Body only** (if you edited body)

1. Select armature + body mesh  
2. Export → `web_viewer/body.glb`  
3. Options:
   - Selected Objects  
   - **Skinning** / Armature = ON  
   - Shape Keys off for body  
   - +Y Up  

Then keep using the existing `face_viewer.html` (loads both and attaches face to neck).  
After manual parent in Blender, viewer attach is a backup; export face in neck-local space if you bake parent.

### Option 2 — One combined avatar (advanced)

1. Select face + body + armature  
2. Export single `web_viewer/avatar.glb`  
3. Viewer would need a small update to load one file (do later)

For now **Option 1** is safer for lipsync morphs.

---

## Part H — Checklist (done when all true)

- [ ] Body standing, meters, feet near ground  
- [ ] Face scale matches neck/shoulders  
- [ ] No large gap or sinking at neck  
- [ ] Face faces same direction as body  
- [ ] Parent/constraint to **`neck`** bone  
- [ ] Pose spine/shoulder → face follows  
- [ ] Scene saved: `avatar_face_body_sync.blend`  
- [ ] Face morphs still exist on head mesh (shape keys)  
- [ ] Export face.glb with shape keys ON  

---

## Common problems

| Problem | Fix |
|---------|-----|
| Face huge or tiny | Uniform scale on `FaceRoot` until height ~0.25 m |
| Face backwards | R → Z → 180° |
| Face on wrong axis after glTF | Apply Rotation: Object → Apply → Rotation |
| Morphs missing after export | Export with **Shape Keys** enabled; don’t “Apply” basis-only |
| Body invisible | Check materials / solid viewport shading |
| Neck bone not found | Armature Display → Names; bone is lowercase `neck` |
| Face jumps when parenting | Use Child Of → **Set Inverse**, or clear parent inverse (Alt+P) |

---

## After manual sync

1. Tell the project alignment is saved in `avatar_face_body_sync.blend`  
2. We can copy your Blender **location/scale/rotation** numbers into `face_viewer.html` attach offsets so the web view matches  
3. Next: body motion (idle / actions) on the same armature  

---

## Numbers to write down (optional but helpful)

In Blender, with `FaceRoot` selected, copy **N-panel** values after alignment:

```
Location X: ____  Y: ____  Z: ____
Rotation  X: ____  Y: ____  Z: ____
Scale     X: ____  Y: ____  Z: ____
Parent bone: neck
```

Paste those here later so the web viewer can match your manual sync exactly.
