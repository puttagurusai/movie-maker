# Intelligent Movie Director (memory-less LLM safe)

## Problem from the 45s review

| Issue | Fix |
|-------|-----|
| Camera inside head | Safer ECU/CU distances + `clamp_camera_pair` + Blender min dist |
| Reels-style micro cuts | Director `target_duration_s` + holds; min take ~5.5s |
| Only zoom in/out | Expanded moves: orbit, arc, pan, truck, crane, follow, reveal, handheld… |
| Single camera | Multi-cam roles `A_cam` / `B_cam` / `C_cam` / `env_cam` created on demand |
| LLM has no memory | **ContinuityBoard** injected every call; one Director plan preferred |

## Architecture

```
Story / script
    │
    ▼
┌─────────────────────────────────────────┐
│ ContinuityBoard (EXTERNAL MEMORY)       │
│  character state, last cam, arc, shots  │
└─────────────────────────────────────────┘
    │ full inject every time
    ▼
┌─────────────────────────────────────────┐
│ Movie Director (ONE LLM call or rules)  │
│  take length, camera role/size/move,    │
│  pace, clip_policy, body intent         │
└─────────────────────────────────────────┘
    │ structured JSON
    ▼
Workers (no LLM memory needed)
  camera_agent → multi-cam keyframes (safe)
  motion_router / MoMask / catalog
  TTS + pad holds
  face bake
    │
    ▼
WRITE bake results → ContinuityBoard + package
```

**Do not** run separate lips/camera/body LLM agents each with empty history.  
**Do** one Director + durable board + deterministic workers.

## Run

```powershell
# Rules director (no API)
python run_movie_pipeline.py --script tests/movie_e2e_1min_script.json --no-brain --target-s 60

# LLM director (llm_fw/config.json) — board injected, model has no memory
python run_movie_pipeline.py --script tests/movie_e2e_1min_script.json --llm --no-brain

# Full E2E + log + preview
python run_e2e_movie_test.py --no-brain --res 640x360
python run_e2e_movie_test.py --llm --no-brain --res 640x360
```

Outputs per run:

- `movie_package.json`
- `master_timeline.json`
- `continuity_board.json`  ← external memory dump
- `E2E_PIPELINE_LOG.txt` / `.json` (e2e only)

## Director fields (per take)

| Field | Owner | Meaning |
|-------|--------|---------|
| `target_duration_s` | Director | Desired take length (pad if speech shorter) |
| `hold_before_s` / `hold_after_s` | Director | Silence / breathing room |
| `camera_role` | Director | Which cam to create/use |
| `camera_shot` | Director | ECU…WS |
| `camera_move` | Director | Creative move menu |
| `pace` | Director | slow/medium/fast path amplitude |
| `clip_policy` | Director | loop / hold_end / stretch / momask_match |
| Keyframe XYZ | **Worker** | Safe geometry only |

## Multi-cam

| Role | Object name | Use |
|------|-------------|-----|
| A_cam | MovieCam_A | Main coverage |
| B_cam | MovieCam_B | Side / angle |
| C_cam | MovieCam_C | Detail |
| env_cam | MovieCam_Env | Establish / wide |

Blender receiver creates missing cameras automatically and switches `scene.camera`.

## Safety env

```powershell
$env:MOVIE_CAM_MIN_DIST = "1.08"
$env:MOVIE_CAM_MIN_Y = "1.15"
$env:MOVIE_MIN_TAKE_S = "5.5"
$env:MOVIE_DIRECTOR_LLM = "1"   # allow LLM when provider present
```
