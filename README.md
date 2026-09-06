# ProjFace — editable 3D performance pipeline for Blender

**ProjFace** turns a story or chat line into a **controllable 3D Session** in Blender: speech + lips, body motion, camera, look/set, and NPCs.  

The deliverable of *this* pack is the **skeleton take** you can Play and scrub in Blender (correct motion, audio, layout).  
A separate **beauty layer** (Wan, Seedance, LTX, … — your choice) can skin that playblast for photoreal look. We do **not** require paid beauty APIs to run the core product.

---

## What this project is

| Layer | Role |
|-------|------|
| **Director** | Story / chat → shots (stage vs spoken, camera, look) — rules or any LLM via `llm_fw` |
| **Speech** | Parler-TTS → WAV |
| **Lips** | wav2arkit → ARKit mouth shapes on the hero head |
| **Face (optional Brain)** | Upper-face emotion from audio + HuBERT |
| **Body** | Catalog Actions and/or MoMask → SMPL-X Session timeline |
| **Look / cast** | Street/studio kits, HDRI, extras NPCs, wardrobe fit gate |
| **Camera** | SessionCam bake (no live zoom fight on install) |
| **Agentic tools** | ChatGPT-style: any LLM calls `movie_run_story`, `body_direct`, `look_apply`, … |

**Primary loop:** open the main `.blend` → start `blender_receiver` → run a story → **Play Session** in Blender.

Optional later: export a fast EEVEE/viewport playblast as the skeleton for a beauty model.

---

## Folder map

```
git_upload4/
├── README.md                          ← you are here
├── FILE_GUIDE.md                      ← file-by-file roles
├── requirements.txt
├── requirements-movie.txt
├── download_assets.ps1                ← small assets helper
├── check_setup.py
├── start.bat
│
├── whole_body_production_ready.blend  ← MAIN scene (SMPL-X hero)
├── face.blend                         ← face-only scene (optional)
│
├── blender_receiver.py                ← run INSIDE Blender (UDP + Session UI)
├── run_movie_pipeline.py              ← story → bake → Session install
├── run_agentic_movie.py               ← any LLM + Movie OS tools
├── orchestrator.py / orchestrator_agents.py
├── parler_voice.py / wav2arkit.py / audio2emotion.py / …
│
├── face_agents/                       ← face, body, camera, look, movie_production
├── llm_fw/                            ← providers + Movie OS tool belt
├── body_motion/                       ← catalog, bone maps, retarget host blend
├── assets/
│   ├── looks/hdri/                    ← Poly Haven HDRIs + presets
│   └── cast/                          ← NPC / wardrobe assets
├── tools/                             ← retarget / MoMask helpers (scripts only)
├── docs/                              ← design + runbooks
├── models/                            ← DOWNLOAD weights here (see below)
└── third_party/                       ← clone MoMask here (see below)
```

Large neural weights are **not** shipped in this folder (multi‑GB). Download each model with the **one link per model** table below into the paths shown.

---

## Requirements

| Tool | Version / notes |
|------|-----------------|
| **OS** | Windows 10/11 (PowerShell scripts) |
| **Python** | 3.11+ **system** Python (not Blender’s) |
| **Blender** | 4.2+ (5.x OK); EEVEE for previews |
| **ffmpeg** | On PATH, or `imageio-ffmpeg` via pip |
| **GPU** | Optional but recommended (Parler, MoMask, Brain) |

```powershell
cd git_upload4
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-movie.txt
# Parler-TTS package:
pip install git+https://github.com/huggingface/parler-tts.git
# Torch: install the build that matches your CUDA from https://pytorch.org
python check_setup.py
```

Optional LLM director / agentic tools:

```powershell
copy llm_fw\config.example.json llm_fw\config.json
# Set api_key via env var (GROQ_API_KEY / OPENAI_API_KEY / …) — do not commit keys
```

---

## Model downloads (one link per model)

Place files under `models/` (or `third_party/` for MoMask) as indicated.  
**One official link per model** — use Hugging Face CLI, browser, or `gdown` as noted.

### 1) Parler-TTS Mini v1 (speech)

| | |
|--|--|
| **Link** | https://huggingface.co/parler-tts/parler-tts-mini-v1 |
| **Install to** | `models/parler-tts-mini-v1/` |
| **Command** | `huggingface-cli download parler-tts/parler-tts-mini-v1 --local-dir models/parler-tts-mini-v1` |
| **Size** | ~1 GB |
| **Used by** | `parler_voice.py`, movie baker TTS |

### 2) wav2arkit (lips / mouth)

| | |
|--|--|
| **Link** | https://huggingface.co/myned-ai/wav2arkit_cpu |
| **Install to** | `models/wav2arkit_cpu/` |
| **Command** | `huggingface-cli download myned-ai/wav2arkit_cpu --local-dir models/wav2arkit_cpu` |
| **Size** | ~385 MB (`wav2arkit_cpu.onnx` + `.onnx.data`) |
| **Used by** | `wav2arkit.py`, lips agent |

### 3) HuBERT base (Brain audio encoder)

| | |
|--|--|
| **Link** | https://huggingface.co/facebook/hubert-base-ls960 |
| **Install to** | `models/brain/hubert-base-ls960/` |
| **Command** | `huggingface-cli download facebook/hubert-base-ls960 --local-dir models/brain/hubert-base-ls960` |
| **Size** | ~360 MB |
| **Used by** | Brain face path (`orchestrator_brain.py`) — optional if you only need lips |

### 4) NVIDIA Audio2Emotion (emotion from audio)

| | |
|--|--|
| **Link** | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/maxine/resources/audio2emotion-v2.2 (NGC Maxine Audio2Emotion) |
| **Alt docs** | https://docs.nvidia.com/ace/audio2emotion-microservice/latest/index.html |
| **Install to** | `models/audio2emotion_v2.2/` (ONNX + `model.json` / config from the package) |
| **Used by** | `audio2emotion.py` — optional Brain emotion input |

> If you skip Audio2Emotion + Brain weights, lips + catalog body + camera still work; upper-face Brain expression will be limited.

### 5) Brain face modules (project-trained)

| | |
|--|--|
| **Link** | *Not a public HF repo* — copy from your training export: `shared_encoder.pt`, `face_head.pt`, `character_adapter.pt` / `adapter.pt`, `brain.pt` |
| **Install to** | `models/brain/` |
| **Also keep** | `emotion_stats.json` (already included in this pack when present) |
| **Used by** | Brain upper-face track |

Ship these yourself next to the pack if you distribute Brain-enabled builds.

### 6) MediaPipe Face Landmarker (live webcam mode)

| | |
|--|--|
| **Link** | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task |
| **Install to** | `models/face_landmarker.task` |
| **Command** | `.\download_assets.ps1` (downloads this + optional Kokoro) |
| **Size** | ~4 MB |
| **Used by** | `mediapipe_face_capture.py` |

### 7) MoMask text-to-motion (open body generation)

| | |
|--|--|
| **Code** | https://github.com/EricGuo5513/momask-codes |
| **Weights (Drive)** | https://drive.google.com/file/d/1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0/view?usp=sharing |
| **Install code to** | `third_party/momask-codes/` |
| **Install weights to** | `third_party/momask-codes/checkpoints/t2m/` |
| **Helper** | `powershell -File tools\download_momask_models.ps1` (after clone + venv) |
| **Used by** | `face_agents/momask_body_pipeline.py` |

```powershell
git clone --depth 1 https://github.com/EricGuo5513/momask-codes.git third_party\momask-codes
python -m venv third_party\momask_venv
.\third_party\momask_venv\Scripts\pip.exe install -U pip torch
.\third_party\momask_venv\Scripts\pip.exe install -r third_party\momask-codes\requirements.txt
# then download_momask_models.ps1
```

Catalog-only body (wave/walk/talk from `body_motion`) works **without** MoMask.

### 8) Optional legacy Kokoro TTS

| | |
|--|--|
| **Link (model)** | https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx |
| **Link (voices)** | https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin |
| **Install to** | `models/kokoro-v1.0.onnx`, `models/voices-v1.0.bin` |
| **Note** | Orchestrator prefers Parler; Kokoro is optional |

### 9) Optional beauty layer (not required to run ProjFace)

These skin your **skeleton playblast** — pick one later:

| Model | Link | Cost |
|-------|------|------|
| Wan 2.1 / 2.2 | https://github.com/Wan-Video/Wan2.1 | Free local |
| Wan2GP (UI) | https://github.com/deepbeepmeep/Wan2GP | Free local |
| LTX-Video | https://huggingface.co/Lightricks/LTX-Video | Free (check license) |
| HunyuanVideo | https://huggingface.co/tencent/HunyuanVideo | Free local |
| Seedance (Higgsfield) | https://higgsfield.ai | Paid credits |

---

## Quick start (first Session)

### 1. Open the main scene

1. Start **Blender**.  
2. Open **`whole_body_production_ready.blend`**.  
3. Text Editor → Open **`blender_receiver.py`** → **Run Script**.  
4. In the 3D View sidebar (**N** → **Avatar**): **Start face stream**  
   (or console: `bpy.ops.face.stream_receiver()`).

### 2. Run a story (system Python)

```powershell
.\.venv\Scripts\Activate.ps1
python run_movie_pipeline.py --story "Walk on a sunny street, wave hello, say goodbye."
```

Defaults: bake → **Session install** → `session_bind`. Export MP4 is **off**.

### 3. Play in Blender

**Avatar panel → Play Session** (or Space after bind).  
You should see body motion + lips + camera on the Session timeline.

### 4. Agentic LLM (optional)

```powershell
python run_agentic_movie.py --info
python run_agentic_movie.py --ask "Sunny street, wave hello, then session_bind."
```

Any OpenAI-compatible provider in `llm_fw/config.json` uses the same tool belt.

---

## Product modes

| Mode | Command / UI | Output |
|------|----------------|--------|
| **Session Play (primary)** | `run_movie_pipeline.py` + Play Session | Editable 3D take |
| **Agentic director** | `run_agentic_movie.py` | Same Session via tools |
| **Chat in Blender** | Avatar chat panel | Incremental Session clips |
| **Skeleton playblast** | Preview / EEVEE export (optional) | MP4 + audio for beauty models |
| **Beauty** | User’s Wan / Seedance / … | Photoreal skin on your skeleton |

---

## Setup checklist

- [ ] Python venv + `requirements.txt` / `requirements-movie.txt`  
- [ ] Parler downloaded into `models/parler-tts-mini-v1/`  
- [ ] wav2arkit into `models/wav2arkit_cpu/`  
- [ ] (Optional) HuBERT + Brain `.pt` + Audio2Emotion  
- [ ] (Optional) MoMask clone + checkpoints  
- [ ] Open `whole_body_production_ready.blend`, run `blender_receiver.py`, start stream  
- [ ] `python check_setup.py` reports OK  
- [ ] One story run + Play Session shows motion  

---

## Security

- Do **not** commit `llm_fw/config.json` with API keys. Use `config.example.json` + env vars.  
- Do **not** commit multi‑GB weight folders; document links only (as above).

---

## Docs

| Doc | Topic |
|-----|--------|
| [FILE_GUIDE.md](FILE_GUIDE.md) | What each file does |
| [docs/GOAL_PIPELINE.md](docs/GOAL_PIPELINE.md) | Product goal + agentic model |
| [docs/MOVIE_PIPELINE_GUIDE.md](docs/MOVIE_PIPELINE_GUIDE.md) | Movie bake path |
| [docs/SESSION_TIMELINE.md](docs/SESSION_TIMELINE.md) | Session SoT |
| [models/README.md](models/README.md) | Models layout |
| [third_party/README.md](third_party/README.md) | MoMask / extras |
| [llm_fw/README.md](llm_fw/README.md) | Attach any LLM |

---

## License notes

- Poly Haven HDRIs: see `assets/looks/LICENSE_POLYHAVEN.txt` (CC0).  
- Third-party models: follow each model’s license (Parler, wav2arkit, HuBERT, MoMask, NVIDIA, etc.).  
- ProjFace application code in this pack: use per your project license.
