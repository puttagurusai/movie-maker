# Body actions — layered architecture

Full workflow: **`docs/BODY_DIRECTIVE_WORKFLOW.md`**

## Model

```
BodyDirectorAgent (LLM menus or rules)
  base_state / state              → BASE layer
  upper_gesture_target / gesture  → OVERLAY IK
  hand, actions[]                 → upper verbs
        │
        ▼
body_agent  → procedural upper eulers (masked)
coordinator → UDP type=body { bones, state, gesture_target, hand, energy, … }
ws_bridge   → WebSocket
        │
        ▼
face_viewer.html  updateBodyLayers()
  1. applyBaseState      (legs / hips; pause IK on transition)
  2. applyOverlayBones   (upper eulers; skip IK arms)
  3. updateIKGesture     (Bézier + spring-damper + 2-bone IK + clamps)
  4. applyBreathSpeech   (noise breath + speech energy)
```

Face blendshapes stay on a separate path (agents / Brain).

## Director beat fields

```json
{
  "text": "Hmm, let me think.",
  "emotion": "thinking",
  "intensity": 0.7,
  "state": "standing",
  "gesture_target": "chin",
  "hand": "right",
  "actions": ["talk_open"],
  "action_timing": "during"
}
```

Also accepted:
- `"body_target": {"target": "chin", "hand": "right"}`
- Actions like `think_chin` / `point_forward` auto-map to a gesture if target omitted.

## Assets

| File | Role |
|------|------|
| `web_viewer/body.glb` | Skinned SMPL-X (IK deforms mesh) |
| `web_viewer/whole_body.glb` | Aligned face only (body mesh stripped) |

## Run

```powershell
.\start.bat
# or manually: ws_bridge + http.server + orchestrator_agents.py
```

Test script: `temp/director_actions_test.json`

Verify (no GPU):

```powershell
python tools/verify_body_layer.py
```

## Later

- Replace procedural base states with real clips (`walk_loop`, `stand_to_sit`, …)
- Additive complex dances as GLB clips under base/dancing
