# Product goal (locked)

**Beat T2V/Higgsfield as an editable 3D movie OS in Blender** — not as an MP4 factory.

## What “done” means

After a story runs, **Play Session in Blender** shows:

- Look_Set + NPCs  
- Hero body motion (wave/walk/talk on the right frames)  
- Lips + face  
- Stable SessionCam (no zoom fight)  

MP4 is optional and only after Play looks right (`--export-mp4`).

## Agentic AGI control model (ChatGPT pattern)

ChatGPT is an LLM **plus tools**. This product is the same: user sets **any** LLM
in `llm_fw/config.json`; the model gets a tool belt for film + Blender.

| Layer | Tools (examples) | Backing |
|-------|------------------|---------|
| Film | `movie_run_story`, `movie_play` | pipeline + Session install |
| Performance agents | `body_direct`, `speech_say`, `camera_direct`, `face_emotion` | BodyDirector / TTS / CameraAgent / face policy |
| Session | `body_append`, `session_bind`, `session_inspect` | UDP Session SoT on SMPL-X |
| Scene kits | `look_plan_from_text`, `look_apply`, `cast_spawn`, `wardrobe_apply` | LookAgent + fit gate |
| Lights / materials | `light_set`, `material_set`, `prop_add` | Look_Set typed + bpy helpers |
| Escape hatch | `bpy_exec` | Sandboxed bpy (kits cannot express) |

Hybrid rule: **typed agent tools first**; bpy/MCP-style only for missing sets/props.
Native OpenAI `tool_calls` when the provider supports them; JSON fallback otherwise.

```bash
# Deterministic product path (LLM director if configured)
python run_movie_pipeline.py --story "Walk on a sunny street, wave hello, say goodbye."

# Agentic attach — any llm_fw model drives the same tools
python run_agentic_movie.py --info
python run_agentic_movie.py --list-tools
python run_agentic_movie.py --story "Walk on a sunny street, wave hello, say goodbye."
python run_agentic_movie.py --ask "Sunny street, 3 NPCs, hero waves and says hello."
```

Defaults: agentic LLM director (if configured), Session install ON, **session_bind** after install, export OFF.

## Body motion contract

1. Multi-beat install: walking state + `[wave, talk_open]` → separate Session clips (not one frozen walk).  
2. After install, UDP `body.op=session_bind` binds `Session_<id>` to **SMPL-X_Armature**.  
3. Play/Space scrub must move legs/arms — lips+camera alone is a fail.
