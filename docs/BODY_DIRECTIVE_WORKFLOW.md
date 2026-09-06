# Procedural body directive workflow

Integrated from the Technical Directive (2-layer bone masking + LLM director).

## End-to-end flow

```
User text / JSON script
        │
        ▼
BodyDirectorAgent  (LLM menus OR rule fallback)
  → dialogue_text, emotion_label, base_state,
    upper_gesture_target, hand, actions[]
        │  normalize (director_schema)
        ▼
orchestrator_agents.py
  → Parler TTS → FaceCoordinator (face agents + body_agent eulers)
  → UDP type=body {
       state, gesture_target, hand, bones, energy, intensity
     }
        │
        ▼
ws_bridge → browser
        │
        ▼
face_viewer.html  updateBodyLayers()  **golden order**
  1. BASE AnimationMixer (walk/sit/idle clips or procedural tracks)
     — crossFade 0.5s on state change; IK paused during fade
  2. Overlay actions (upper eulers, bone-masked)
  3. IK reach: Bézier arc → spring-damper → 2-bone arm IK
  4. Breath noise + speech energy jitter
  Face blendshapes applied on their own WS path (not overwritten by mixer)
```

Optional Mixamo GLBs: `web_viewer/body_assets/animations/` + `manifest.json`


## Component A — Director JSON (LLM)

```json
{
  "beats": [{
    "dialogue_text": "Eww, that is disgusting!",
    "base_state": "standing",
    "upper_gesture_target": "hand_to_chest",
    "emotion_label": "disgusted",
    "intensity": 0.9,
    "hand": "right",
    "actions": ["recoil", "talk_open"]
  }]
}
```

Aliases still work: `text`, `emotion`, `state`, `gesture_target`.

## Component B — Upper IK (viewer)

1. Target bone lookup + offset  
2. Quadratic Bézier arc  
3. Spring-damper smooth  
4. CCD 2-bone IK (shoulder→elbow→wrist) + rotation clamps  
5. Breathing noise + RMS energy jitter  

## Complex motions

| Intent | Mechanism |
|--------|-----------|
| Sit / walk / dance | `base_state` only |
| Hand to chin/chest | `upper_gesture_target` + IK |
| Wave / celebrate | `actions[]` procedural clips |

## Run

```powershell
.\start.bat
# chat mode (LLM director) or:
# @temp/director_actions_test.json
python tools/verify_body_layer.py
```

## Files

| File | Role |
|------|------|
| `face_agents/body_director_agent.py` | LLM / rule director |
| `face_agents/director_schema.py` | Menus + normalize |
| `face_agents/body_agent.py` | Procedural upper eulers |
| `face_agents/coordinator.py` | UDP body packet |
| `web_viewer/face_viewer.html` | Layers + IK + spring |
| `web_viewer/ws_bridge.py` | Forward fields |
