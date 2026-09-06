# Movie camera (live pipeline)

Additive cinematic camera track on top of face + body.  
**Does not change** lips, expression, or MoMask/catalog body.

## Enable / disable

```powershell
# default ON
$env:USE_MOVIE_CAMERA = "1"

# off (old free-view / fixed cam behavior)
$env:USE_MOVIE_CAMERA = "0"
```

## Run

1. **Reload** `blender_receiver.py` in Blender (required once for camera support)  
2. `bpy.ops.face.stream_receiver()`  
3. `python orchestrator_agents.py`  
4. Speak / type a line as usual  

Console should show:

```text
[movie] timeline → movie_timeline_beat_1.json  camera=MS/dolly_in keys=3
[coordinator] CAMERA shot=MS move=dolly_in keys=3 ...
[camera_receiver] PLAN live shot=MS move=dolly_in ...
```

Blender creates **`MovieCam`** + **`MovieCam_LookAt`**, sets it as `scene.camera`, flies during speech, bakes keyframes for scrub.

## Shot rules (v2 — safe + multi-cam)

| Situation | Shot | Move | Role |
|-----------|------|------|------|
| Standing talk (short) | MS | static | A_cam |
| Standing talk (long) | MS | dolly_in / orbit | A_cam |
| Happy / wave | MS | static / arc | A_cam |
| Sad | MCU | dolly_out | A_cam |
| Angry / high intensity | **CU** (not ECU crash) | dolly_in / orbit | A_cam / B_cam |
| Detail surprise | CU static | static | C_cam |
| Walking | WS | follow / truck | env_cam / A_cam |

**Safety:** ECU/CU Y distance never closer than ~1.15m; Blender clamps min cam–look dist.  
**Multi-cam:** Director picks `camera_role`; pipeline creates `MovieCam_A/B/C/Env` on demand.  
See also: `docs/MOVIE_DIRECTOR_INTELLIGENCE.md`.

## Files

| Path | Role |
|------|------|
| `face_agents/camera_agent.py` | rules → keyframes |
| `face_agents/movie_timeline.py` | beat timeline JSON |
| `blender_receiver.py` | `type=camera` live + bake |
| `temp/movie_timeline_beat_N.json` | last plan dump |
| `test_movie_camera.py` | unit tests |

## Force a shot (optional)

Director beat fields:

```json
{
  "text": "Hello everyone.",
  "emotion": "happy",
  "intensity": 0.8,
  "state": "standing",
  "camera_shot": "CU",
  "camera_move": "dolly_in"
}
```

## Next upgrades

- Multi-beat playlist (cuts between lines)
- Offline EEVEE render → MP4
- LightingAgent
- Pelvis-true truck for long walks
