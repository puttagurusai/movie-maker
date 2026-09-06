# File guide — ProjFace pack

Short map of what each area is for. Setup steps live in **README.md**.

## Entry points

| File | Role |
|------|------|
| `whole_body_production_ready.blend` | Main Blender scene (SMPL-X hero) |
| `face.blend` | Face-only scene |
| `blender_receiver.py` | Blender UDP receiver + Session / Look / Play UI |
| `run_movie_pipeline.py` | Story → bake → Session install (product CLI) |
| `run_agentic_movie.py` | Any LLM + Movie OS tools |
| `orchestrator.py` / `orchestrator_agents.py` | Live chat / generate loop |
| `check_setup.py` | Sanity check after install |
| `download_assets.ps1` | Small model downloads |
| `start.bat` | Convenience launcher (if present) |

## Packages

| Path | Role |
|------|------|
| `face_agents/` | Lips, eyes, brows, body, camera, look, MoMask bridge, QC |
| `face_agents/movie_production/` | Multi-shot director, baker, `session_install` |
| `llm_fw/` | LLM providers + agentic tool belt (`movie_os_tools`, `agent_tools`, `blender_tools`) |
| `body_motion/` | Catalog, bone maps, `retarget_host.blend`, Mixamo FBX sources |
| `assets/looks/` | HDRI + look presets |
| `assets/cast/` | NPC / wardrobe assets |
| `tools/` | Retarget / MoMask helper scripts (no venvs) |
| `docs/` | Design and pipeline runbooks |
| `models/` | Weight download targets (see README) |
| `third_party/` | MoMask clone target |

## Speech / face helpers

| File | Role |
|------|------|
| `parler_voice.py` | Parler-TTS wrapper |
| `wav2arkit.py` | Audio → ARKit mouth |
| `audio2emotion.py` | NVIDIA A2E wrapper |
| `brain_inference.py` / `brain_model.py` | Brain face path |
| `emotion_map.py` / `emotion_manager.py` | Emotion presets / 26-D |
| `mediapipe_face_capture.py` | Live webcam → blendshapes |
| `prosody_gpu.py` | Pitch/energy features |

## Config

| File | Role |
|------|------|
| `llm_fw/config.example.json` | Template — copy to `config.json` locally |
| `requirements.txt` | Core pip deps |
| `requirements-movie.txt` | Extra movie/export deps |
