# Blender-only pipeline (face + body clips)

**Scene:** `whole_body_retargeted.blend` (or `whole body.blend`)  
**Clips:** Mixamo retargeted Actions on `SMPL-X_Armature` (from `source_fbx/`)  
**Agents:** `orchestrator_agents.py` → UDP `:9001`  
**Receiver:** `blender_receiver.py` (face + body Action playback)  
**Next steps:** see `body_motion/NEXT_STEPS.md`

---

## One-time (already done if retarget ran)

```powershell
cd C:\me\proj\projface_v1
# FBXs in body_motion\source_fbx\
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" `
  --background "body_motion\whole_body_motions_face_attached.blend" `
  --python tools\retarget_mixamo_to_smplx.py -- --export-glb
```

Creates Actions: `idle`, `walk`, `wave`, `talk_open`, … and GLBs under `web_viewer/body_assets/animations/`.

---

## Run live (every session)

### 1) Blender

1. Open  
   `body_motion/whole_body_motions_face_attached.blend`
2. **Text Editor** → Open  
   `blender_receiver.py` (project root) → **Run Script**
3. Console / Python:

```python
bpy.ops.face.stream_receiver()
```

You should see:

```text
[body_receiver] SMPL-X Actions ready: N/...
[face_receiver] Target mesh: ...
```

Leave Blender running.

### 2) Agents (system Python)

```powershell
cd C:\me\proj\projface_v1
python orchestrator_agents.py
```

Paste chat or JSON, e.g.:

```json
[
  {
    "text": "Hello, welcome!",
    "emotion": "happy",
    "intensity": 0.85,
    "state": "standing",
    "actions": ["wave", "talk_open"]
  }
]
```

Or free chat if director LLM is configured.

### 3) What happens

```text
orchestrator_agents
  → Parler TTS + wav2arkit + face agents
  → body: MotionController → clip_id (wave, talk_open, …)
  → UDP type=body { action, clip_id, state, intensity, speed, loop }
  → blender_receiver plays Action on SMPL-X_Armature
  → face blendshapes + head as before
```

---

## Catalog

| File | Role |
|------|------|
| `body_motion/catalog.json` | clip_id → FBX / Action / director aliases |
| `face_agents/motion_controller.py` | Policy: actions/state → clip + variation |
| `tools/retarget_mixamo_to_smplx.py` | Re-bake if you add FBXs |

---

## Stop

Blender console:

```python
bpy.ops.face.stream_stop()
```

---

## Troubleshooting

| Issue | Fix |
|--------|-----|
| No body motion | Confirm Actions exist (Action Editor on armature). Re-run retarget. |
| Face only | Receiver must be the updated `blender_receiver.py` with body support. |
| Wrong Action | Check UDP `action` name matches Action data-block (`wave`, `idle`, …). |
| Stiff face attach | Use `whole_body_motions_face_attached.blend` (your 70% body). |

**Web viewer is optional later** — for now everything is Blender-only.
