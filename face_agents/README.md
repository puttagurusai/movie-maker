# face_agents — multi-agent talking face

Realistic **expression while talking** by splitting the face into region agents.

**Does not replace** `orchestrator.py` or `orchestrator_brain.py`.

## Run

```powershell
# Blender first
#   open face.blend → run blender_receiver.py → stream_receiver

python orchestrator_agents.py
```

Disable Brain expression bake (presets only):

```powershell
$env:USE_BRAIN="0"
python orchestrator_agents.py
```

Lip hold if mouth leads audio (ms):

```powershell
$env:FACE_LIP_HOLD_MS="120"
```

## Architecture

```
JSON {text, emotion, intensity}
        │
        ▼
   Parler TTS → WAV
        │
        ▼
 FaceCoordinator.prepare()
   ├─ energy envelope (speech RMS @ 30fps)
   ├─ Brain encoder bake → upper-face track
   ├─ lips_agent   → wav2arkit mouth/jaw
   ├─ eyes_agent   → blink + gaze + Brain eyes
   ├─ brows_agent  → emotion_map + Brain + energy pulse
   ├─ cheeks_agent → smile/frown/nose + Brain
   └─ head_agent   → nod / breath / energy
        │
        ▼
 play @ 30fps (audio clock + lip hold)
   viseme  ← lips
   emotion ← eyes + brows + cheeks
   head    ← head
        │
        ▼
 blender_receiver.py
```

## Agents

| Agent | Owns | While talking |
|-------|------|----------------|
| **lips** | jaw / mouth | wav2arkit track |
| **eyes** | blink, look, squint, wide | procedural + Brain micro |
| **brows** | brow shapes | preset + Brain + speech energy |
| **cheeks** | cheeks, nose, smile/frown | preset + Brain + energy |
| **head** | pitch/yaw/roll | nod + breath + energy peaks |

Key ownership is enforced: agents cannot write outside `owned_keys`.

## Brain encoder

Uses the same weights as the Brain pipeline:

```
models/brain/shared_encoder.pt
models/brain/face_head.pt
models/brain/character_adapter.pt
models/brain/hubert-base-ls960/
```

Lips still use **wav2arkit** (best lip-sync). Brain drives **expression** only.

## JSON example

```json
{
  "sentences": [
    {"text": "Hello, great to see you.", "emotion": "happy", "intensity": 0.85},
    {"text": "I am worried about the deadline.", "emotion": "fearful", "intensity": 0.8},
    {"text": "That was completely wrong.", "emotion": "angry", "intensity": 0.9}
  ]
}
```

## Files

| File | Role |
|------|------|
| `base.py` | `FaceContext`, `BaseFaceAgent` |
| `brain_track.py` | Bake Brain expression + energy |
| `lips_agent.py` | Mouth / jaw |
| `eyes_agent.py` | Blink / gaze / Brain eyes |
| `brows_agent.py` | Brows |
| `cheeks_agent.py` | Cheeks / nose / smile |
| `head_agent.py` | Head pose |
| `coordinator.py` | Merge + UDP + play |
| `policy_bridge.py` | Optional LLM FACE_POLICY |
| `../orchestrator_agents.py` | Entry point |
