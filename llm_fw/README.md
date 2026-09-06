# llm_fw — attach any LLM to Movie OS tools

Same idea as ChatGPT: the model is a brain; **tools** do the work.

## Configure a provider

```powershell
copy llm_fw\config.example.json llm_fw\config.json
# Prefer env vars (GROQ_API_KEY / OPENAI_API_KEY / XAI_API_KEY) — never commit real keys
```

Example shape (see `config.example.json` and `config.groq.json`):

- `provider`: `openai_compat` or `anthropic`
- `model`: e.g. `openai/gpt-oss-20b` (Groq), `gpt-4o-mini`, `grok-2-latest`
- `base_url`: e.g. `https://api.groq.com/openai/v1`
- `api_key_env`: name of the environment variable holding the key

Works with Groq, OpenAI, xAI Grok, OpenRouter, Ollama, etc.  
Swap model + base_url + env — **tool belt stays the same**.

## Run

```bash
# Blender: run blender_receiver.py → Start face stream
python run_agentic_movie.py --info
python run_agentic_movie.py --list-tools
python run_agentic_movie.py --ask "Sunny street, wave hello, say hi."
python run_agentic_movie.py --story "Walk on a sunny street, wave hello, say goodbye."
```

Native OpenAI `tool_calls` when supported; JSON tool protocol as fallback.

## Tool groups

- **Film:** `movie_run_story`, `movie_play`
- **Agents:** `look_plan_from_text`, `body_direct`, `camera_direct`, `speech_say`, `face_emotion`
- **Session:** `body_append`, `session_bind`, `session_inspect`
- **Scene:** `look_apply`, `cast_spawn`, `wardrobe_apply`, `light_set`, `material_set`, `prop_add`, `bpy_exec`
