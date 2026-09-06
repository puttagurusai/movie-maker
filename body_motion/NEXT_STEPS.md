# Next steps (clips done)

## Done

- Final Mixamo → SMPL-X bone map (`FINAL_BONE_MAP.json`)
- All catalog clips retargeted → `whole_body_retargeted.blend` / `whole body.blend`
- Inplace root for gestures (no body slide on wave)
- Face/mouth shape-key scene animation stripped (mouth = face pipeline only)

## Open this file

`whole_body_retargeted.blend` (or `whole body.blend`)

## Live run (next step)

### 1. Blender

1. Open `whole_body_retargeted.blend`
2. Text Editor → open project root `blender_receiver.py` → **Run Script**
3. Console:

```python
bpy.ops.face.stream_receiver()
```

Expect: `[body_receiver] SMPL-X Actions ready: …` and face mesh found.

### 2. Agents

```powershell
cd C:\me\proj\projface_v1
python orchestrator_agents.py
```

Send a line or JSON so director picks `wave` / `talk_open` / `idle` while face gets visemes.

### 3. Verify

| Check | Pass |
|--------|------|
| Body Action plays (wave, walk) | arms/legs move |
| Mouth closed when no speech | no random jaw |
| Speaking | mouth from face packets only |
| Playback | ~30 FPS with Frame Dropping |

## FPS checklist (viewport)

1. Timeline → Playback → **Frame Dropping**
2. Viewport shading: **Solid** (not Material / EEVEE while testing)
3. Hide objects you don’t need
4. Subsurf / Data Transfer already off in viewport on cleaned file
5. Don’t leave heavy addons polling if unused

Body Actions are dense (1 key/frame). That is normal for retarget quality; Frame Dropping keeps wall-clock speed.

## Mouth ownership

| System | Controls |
|--------|----------|
| Body Actions | spine, limbs, fingers, root — **not** jaw/mouth shapes |
| Face pipeline (receiver) | ARKit blendshapes / visemes / emotion |

If mouth moves again without speech: re-run  
`tools/strip_face_from_body_actions.py` on the blend.

## Optional later

- NLA blend upper (talk) over base (idle/walk)
- Export GLB per clip for web
- Drop key density on long idles if file size matters

## LMM training (optional engine — not required for good clips)

Full guide: **`body_motion/train_lmm.txt`**

- LMM = optional engine under the same controller (`engine: clip | lmm`)
- Prefer finetune pretrained text-to-motion; do not block Phase A
- Face / mouth stay out of LMM
