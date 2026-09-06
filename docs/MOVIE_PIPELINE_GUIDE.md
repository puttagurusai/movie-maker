# Agentic Movie Generation Pipeline — Complete Guide

> **Goal**: User types a story → pipeline produces a movie-level MP4 clip in seconds.  
> **Philosophy**: Simple LLM prompts + rule-based agents + existing tools. No heavy frameworks. No MCP overhead.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Pipeline Flow (End to End)](#pipeline-flow)
3. [Agent Specifications](#agent-specifications)
   - [StoryAgent](#1-storyagent)
   - [DialogueAgent](#2-dialogueagent)
   - [BeatAgent (DirectorAgent)](#3-beatagent-directoragent)
   - [CameraAgent](#4-cameraagent)
   - [LightingAgent](#5-lightingagent)
   - [BodyAgent](#6-bodyagent)
   - [FaceAgent](#7-faceagent)
   - [VoiceAgent](#8-voiceagent)
   - [LipSyncAgent](#9-lipsyncagent)
   - [RenderAgent](#10-renderagent)
4. [Data Schemas](#data-schemas)
5. [Blender Director Server API](#blender-director-server-api)
6. [Camera Move Vocabulary](#camera-move-vocabulary)
7. [LLM Prompt Templates](#llm-prompt-templates)
8. [Configuration](#configuration)
9. [File Structure](#file-structure)
10. [Performance Targets](#performance-targets)
11. [Implementation Roadmap](#implementation-roadmap)
12. [Running the Pipeline](#running-the-pipeline)

---

## Architecture Overview

```
USER INPUT (plain text story)
        │
        ▼
┌───────────────────┐
│    StoryAgent     │  1 LLM call → breaks story into scenes
│  (llm_fw + Groq)  │
└────────┬──────────┘
         │  scenes[]
         ▼
┌───────────────────┐
│  DialogueAgent    │  1 LLM call per scene → creates dialogue + emotion tags
│  (llm_fw + Groq)  │  (runs parallel for all scenes)
└────────┬──────────┘
         │  dialogue[]
    ┌────┴─────────────────────────────────────────┐
    │                                              │
    ▼                                              ▼
┌──────────────┐                         ┌──────────────────┐
│  BeatAgent   │  rule-based per         │   VoiceAgent     │
│(DirectorAgent│  sentence → beats[]     │  (Parler TTS)    │
└──────┬───────┘                         │  text → WAV      │
       │                                 └────────┬─────────┘
  ┌────┼───────────┐                              │
  ▼    ▼           ▼                              ▼
Camera Body     Face                     ┌──────────────────┐
Agent  Agent    Agent                    │  LipSyncAgent    │
(rules)(exists) (exists)                 │  WAV → ARKit     │
  │      │        │                      │  blendshapes/f   │
  └──────┴────────┴──────────────────────┘
                  │ All data ready
                  ▼
         ┌─────────────────┐
         │   RenderAgent   │  → blender_director_server.py
         │ (Blender EEVEE) │  → ffmpeg composite
         └────────┬────────┘
                  │
                  ▼
              final.mp4
```

**Key principle**: All LLM calls happen in parallel. All rule-based agents are instant. Blender renders asynchronously. Total latency = max(LLM time, TTS time) + render time.

---

## Pipeline Flow

### Step-by-step execution of a 2-minute, 3-scene movie

```
T=0.0s   MoviePipeline.generate(story_text)
T=0.0s     → StoryAgent.run(story_text)           [LLM call, async]
T=0.0s     → VoiceAgent.synthesize(full_text)     [TTS, parallel]
T=0.5s   StoryAgent returns: scenes[3]
T=0.5s     → DialogueAgent.run(scene) × 3         [3 parallel LLM calls]
T=1.0s   DialogueAgent returns: dialogue per scene
T=1.0s     → BeatAgent.run(dialogue) per scene    [rule-based, instant]
T=1.0s     → CameraAgent.plan(beats) per scene    [rule-based, instant]
T=1.0s     → LightingAgent.plan(scenes)           [rule-based, instant]
T=1.5s   VoiceAgent returns: scene_1.wav, scene_2.wav, scene_3.wav
T=1.5s     → LipSyncAgent.run(wav) × 3            [wav2arkit, parallel]
T=2.0s   All data assembled → send to Blender
T=2.0s     → RenderAgent.load_scene(all_data)
T=2.0s     → blender_director_server: set camera keyframes
T=2.0s     → blender_director_server: set lighting
T=2.0s     → blender_director_server: set face blendshapes
T=2.0s     → blender_director_server: set body actions
T=2.5s     → blender_director_server: render_clip(0, end, 'scene_1.mp4')
T=12s    scene_1.mp4 done (120 frames × GPU EEVEE ~10ms/frame)
T=22s    scene_2.mp4, scene_3.mp4 done
T=23s    ffmpeg: stitch scenes + mux audio → final.mp4
T=23s    DONE ✓
```

---

## Agent Specifications

### 1. StoryAgent

**File**: `face_agents/story_agent.py`  
**Type**: LLM (1 call per movie)  
**Speed**: ~0.5-1.0 seconds

**Input**:
```python
story_text: str  # plain English story, any length
```

**Output** (list of SceneDict):
```json
[
  {
    "scene_id": 1,
    "summary": "Scientist discovers cure, alone in lab at night",
    "location": "lab",
    "time_of_day": "night",
    "mood": "awe",
    "characters": ["scientist"],
    "duration_hint_seconds": 30,
    "key_action": "discovery moment"
  },
  ...
]
```

**LLM Prompt** (system):
```
You are a film scene breakdown assistant.
Break the given story into 1-5 scenes suitable for a single-character 3D animation.
Output ONLY a JSON array of scene objects with these exact keys:
scene_id, summary, location, time_of_day (day/night/golden_hour/overcast),
mood (one of: awe, excited, sad, angry, neutral, anxious, joy, love),
characters (list of character names), duration_hint_seconds (int 10-120),
key_action (short string, the main physical/emotional action in the scene).
```

**LLM Prompt** (user):
```
Story: {story_text}

Output only the JSON array, no explanation.
```

---

### 2. DialogueAgent

**File**: `face_agents/dialogue_agent.py`  
**Type**: LLM (1 call per scene, all parallel)  
**Speed**: ~0.3-0.5 seconds each

**Input**: `SceneDict` from StoryAgent

**Output** (list of DialogueBeatDict):
```json
[
  {
    "sentence": "Oh my god... it actually works.",
    "emotion": "awe",
    "intensity": 0.9,
    "pause_before_ms": 500,
    "pause_after_ms": 1000,
    "gesture": "look_at_object",
    "state": "stand"
  },
  {
    "sentence": "I need to call Sarah right now.",
    "emotion": "excited",
    "intensity": 0.7,
    "pause_before_ms": 0,
    "pause_after_ms": 300,
    "gesture": "reach_forward",
    "state": "stand"
  }
]
```

**LLM Prompt** (system):
```
You are a screenplay writer for a single-character 3D animated film.
Write natural spoken dialogue for the given scene.
Each sentence is a separate beat object. Output ONLY a JSON array.

Beat object keys:
  sentence: str (what the character says, 5-25 words)
  emotion: one of [neutral, happy, sad, angry, surprised, awe, excited, anxious, love, disgust]
  intensity: float 0.0-1.0 (how strongly the emotion shows)
  pause_before_ms: int (silence before speaking, 0-2000)
  pause_after_ms: int (silence after speaking, 0-2000)
  gesture: one of [none, nod, shake_head, look_around, reach_forward, look_at_object,
                   hand_to_chin, point_forward, wave, shrug, celebrate]
  state: one of [stand, walk, sit, lean_forward]

Keep sentences short and natural. 4-10 beats per scene.
```

**LLM Prompt** (user):
```
Scene: {scene['summary']}
Mood: {scene['mood']}
Key action: {scene['key_action']}
Duration hint: {scene['duration_hint_seconds']} seconds

Write the dialogue. Output only the JSON array.
```

---

### 3. BeatAgent (DirectorAgent)

**File**: `face_agents/director_agent.py` (EXISTS — already working)  
**Type**: Rule-based + existing LLM logic  
**Speed**: ~0 seconds (synchronous, in-process)

**Input**: `DialogueBeatDict` from DialogueAgent  
**Output**: `BeatDict` — extends DialogueBeatDict with face/body specifics

```json
{
  "sentence": "Oh my god... it actually works.",
  "emotion": "awe",
  "intensity": 0.9,
  "gesture": "look_at_object",
  "state": "stand",
  "body_action": "idle_breathe",
  "gesture_target": "forward",
  "blink_rate": "slow",
  "eye_dart": false,
  "head_tilt": 0.1,
  "face_blendshapes": {
    "browInnerUp": 0.85,
    "eyeWideLeft": 0.7,
    "eyeWideRight": 0.7,
    "mouthOpen": 0.3,
    "jawOpen": 0.25
  }
}
```

**Note**: This agent already exists and drives the current real-time pipeline. For movie mode it runs offline, processing all beats before render.

---

### 4. CameraAgent

**File**: `face_agents/camera_agent.py`  
**Type**: Rule-based (no LLM needed)  
**Speed**: ~0 seconds

**Input**: `BeatDict[]` for a scene  
**Output**: `CameraKeyframeList` — list of keyframe dicts for Blender

```json
[
  {
    "frame": 0,
    "location": [0, 2.5, 1.5],
    "look_at": [0, 0, 1.6],
    "fov_deg": 40,
    "interpolation": "BEZIER",
    "move_type": "static"
  },
  {
    "frame": 48,
    "location": [0, 1.8, 1.55],
    "look_at": [0, 0, 1.6],
    "fov_deg": 38,
    "interpolation": "BEZIER",
    "move_type": "dolly_in"
  }
]
```

**Shot Selection Rules**:

| Emotion | Intensity | State | Shot Type | Camera Move |
|---------|-----------|-------|-----------|-------------|
| awe | >0.7 | stand | ECU (extreme close-up) | slow dolly in |
| excited | >0.6 | stand | CU (close-up) | slight push |
| angry | >0.7 | stand | CU | fast push in |
| sad | any | stand | MCU | slow pull back |
| happy, joy | any | stand | MS (medium) | static or slow pan |
| neutral | any | stand | MS | static |
| any | any | walk | WS (wide) | truck (follow) |
| celebrate | any | stand | WS | crane up |
| anxious | any | stand | MCU | slight handheld wobble |
| any | >0.8 | any | ECU | slow push |

**Shot Parameters**:

| Shot | Location (x,y,z) | FOV | Look-at |
|------|------------------|-----|---------|
| ECU | [0, 0.6, 1.65] | 28° | [0, 0, 1.65] |
| CU | [0, 1.2, 1.6] | 35° | [0, 0, 1.6] |
| MCU | [0, 1.8, 1.5] | 40° | [0, 0, 1.5] |
| MS | [0, 2.5, 1.4] | 45° | [0, 0, 1.4] |
| MLS | [0, 3.2, 1.3] | 50° | [0, 0, 1.2] |
| WS | [0, 4.5, 1.5] | 55° | [0, 0, 1.0] |

**Camera Move Speeds**:

| Move type | Duration (frames at 24fps) | Distance |
|-----------|---------------------------|----------|
| fast push | 12 | 0.3m |
| slow push | 48 | 0.15m |
| truck | beat_frames | walk speed = 0.05m/f |
| crane up | 60 | +0.5m Y |
| handheld | continuous | ±0.02m random walk |
| cut | 0 | jump to new position |

**Cut vs Move Logic**:
- Between beats: always cut (hard cut = cinematic)
- Within long beat (>3 seconds): add move mid-beat
- Same emotion consecutive beats: push slightly, no full cut

---

### 5. LightingAgent

**File**: `face_agents/lighting_agent.py`  
**Type**: Rule-based (no LLM)  
**Speed**: ~0 seconds

**Input**: `SceneDict[]` (mood + time_of_day per scene)  
**Output**: `LightingPreset` per scene

**Presets** (sent as bpy commands via Blender server):

| time_of_day | mood | Key Light | Fill | Rim | Ambient |
|-------------|------|-----------|------|-----|---------|
| night | any | warm orange, 45° left, hard | none | cool blue rim | very dark |
| day | neutral | white, 30° left, soft | 0.3x | white rim | bright |
| day | excited/joy | warm yellow, 40° left | 0.4x white | golden rim | bright |
| day | sad | cool white, overhead, diffuse | 0.2x | no rim | grey |
| golden_hour | any | deep orange, 15° left, hard | none | orange rim | warm |
| overcast | angry/anxious | grey, 90° overhead | 0.3x cool | none | flat grey |

**Blender Output Format**:
```json
{
  "key_light": {"energy": 800, "color": [1.0, 0.85, 0.7], "location": [-2, 3, 4], "type": "SPOT"},
  "fill_light": {"energy": 200, "color": [0.8, 0.9, 1.0], "location": [3, 2, 2], "type": "AREA"},
  "rim_light":  {"energy": 400, "color": [0.6, 0.7, 1.0], "location": [0, -3, 2], "type": "SPOT"},
  "world_strength": 0.05
}
```

---

### 6. BodyAgent

**File**: `face_agents/body_director_agent.py` (EXISTS — already working)  
**Type**: LLM (optional) + rule-based  
**Speed**: ~0 seconds (rule-based mode for movie)

**Input**: `BeatDict[]`  
**Output**: `BodyActionDict[]` — which FBX action to play per beat

```json
{
  "action_name": "Head Nod Yes",
  "start_frame": 0,
  "speed": 1.0,
  "loop": false,
  "transition_frames": 12
}
```

**Available actions** (from `source_fbx/`):
Agreeing, Clapping, Head Nod Yes, Shrug, Wave, Pointing, Idle Breathing, Walking, Surprised, etc.

---

### 7. FaceAgent

**File**: `face_agents/` — `FaceCoordinator` (EXISTS — already working)  
**Type**: Multi-sub-agent (lips, eyes, brows, cheeks, head)  
**Speed**: ~0 seconds per frame

**Input**: `BeatDict` + lip sync blendshapes per frame  
**Output**: Per-frame ARKit blendshape dict (52 values, 0.0-1.0)

**Sub-agents**:
- `lips_agent.py` — mouth shape from emotion + lip sync frames
- `eyes_agent.py` — blink rate, eye dart, eye wide from emotion
- `brows_agent.py` — brow position from emotion + intensity
- `cheeks_agent.py` — cheek puff, squint from happiness
- `head_agent.py` — head tilt, nod from gesture

---

### 8. VoiceAgent

**File**: `parler_voice.py` (EXISTS — already working)  
**Type**: Neural TTS (Parler TTS Mini)  
**Speed**: ~0.5-2 seconds per sentence

**Input**: `sentence: str`, `emotion: str`, `intensity: float`  
**Output**: `audio_path: str` (WAV file, 22050 Hz mono)

**Emotion → voice style mapping** (in parler prompt):
- happy/excited: "enthusiastic, upbeat, warm voice"
- sad: "slow, soft, emotional voice"
- angry: "firm, tense, clipped voice"
- awe: "slow, breathless, hushed voice"
- neutral: "clear, natural, conversational"

---

### 9. LipSyncAgent

**File**: `wav2arkit.py` (EXISTS — already working)  
**Type**: ONNX model inference  
**Speed**: ~0.1-0.5 seconds per WAV

**Input**: `wav_path: str`  
**Output**: `blendshapes_per_frame: list[dict]` — ARKit blendshapes at 30fps

**Key blendshapes for lip sync**:
- jawOpen, mouthFunnel, mouthPucker, mouthShrugUpper, mouthShrugLower
- mouthUpperUpLeft, mouthUpperUpRight, mouthLowerDownLeft, mouthLowerDownRight
- mouthClose, mouthRollUpper, mouthRollLower

---

### 10. RenderAgent

**File**: `face_agents/render_agent.py` (TO BUILD)  
**Type**: Blender EEVEE + ffmpeg  
**Speed**: ~4-15 seconds per 30-second scene (with GPU)

**Input**:
```python
scene_data: dict  # all keyframes, blendshapes, lighting, audio
output_path: str  # where to save MP4
fps: int = 24
resolution: tuple = (1920, 1080)
renderer: str = 'EEVEE'  # or 'CYCLES'
```

**Output**: `mp4_path: str`

**Process**:
1. HTTP POST to `blender_director_server.py`: load all keyframes + settings
2. HTTP POST: trigger render (async)
3. Poll until render complete
4. ffmpeg: composite rendered frames + WAV → MP4

**Performance**:
- EEVEE + GPU: ~10ms/frame = 100 FPS = 30s scene renders in 3 seconds
- EEVEE + CPU: ~100ms/frame = 10 FPS = 30s scene renders in 30 seconds
- Cycles + GPU: ~500ms/frame = 2 FPS = 30s scene renders in 180 seconds (high quality)

**Recommendation**: EEVEE for fast iteration, Cycles only for final export.

---

## Data Schemas

### MovieRequest
```json
{
  "story": "A scientist discovers a cure...",
  "character_name": "person01",
  "output_dir": "output/movies/",
  "fps": 24,
  "resolution": [1920, 1080],
  "renderer": "EEVEE",
  "llm_config": {"provider": "groq", "model": "llama-3.1-8b-instant"}
}
```

### SceneDict
```json
{
  "scene_id": 1,
  "summary": "string",
  "location": "lab|office|outdoor|home",
  "time_of_day": "day|night|golden_hour|overcast",
  "mood": "awe|excited|sad|angry|neutral|anxious|joy|love",
  "characters": ["scientist"],
  "duration_hint_seconds": 30,
  "key_action": "string"
}
```

### DialogueBeatDict
```json
{
  "sentence": "string",
  "emotion": "neutral|happy|sad|angry|surprised|awe|excited|anxious|love|disgust",
  "intensity": 0.0,
  "pause_before_ms": 0,
  "pause_after_ms": 300,
  "gesture": "none|nod|shake_head|look_around|reach_forward|look_at_object|hand_to_chin|point_forward|wave|shrug|celebrate",
  "state": "stand|walk|sit|lean_forward"
}
```

### FullBeatDict (after all agents)
```json
{
  "beat_id": 1,
  "scene_id": 1,
  "sentence": "string",
  "emotion": "string",
  "intensity": 0.8,
  "gesture": "string",
  "state": "string",
  "start_frame": 0,
  "end_frame": 72,
  "audio_path": "temp/scene_1_beat_1.wav",
  "face_blendshapes_per_frame": [{"jawOpen": 0.3, ...}, ...],
  "body_action": {"action_name": "idle", "speed": 1.0},
  "camera_keyframes": [{"frame": 0, "location": [...], ...}],
  "lighting": {"key_light": {...}, ...}
}
```

---

## Blender Director Server API

**File**: `blender_director_server.py`  
**Runs**: Inside Blender Python environment (bpy)  
**Protocol**: HTTP on `localhost:7500`  
**Start**: Add to Blender startup scripts or launch via `blender --background --python blender_director_server.py`

### Endpoints

#### POST `/camera/keyframes`
Set camera animation keyframes.
```json
{
  "keyframes": [
    {"frame": 0,  "location": [0, 2.5, 1.5], "look_at": [0, 0, 1.6], "fov_deg": 40},
    {"frame": 48, "location": [0, 1.8, 1.55], "look_at": [0, 0, 1.6], "fov_deg": 38}
  ],
  "interpolation": "BEZIER"
}
```

#### POST `/lighting/set`
Set scene lighting.
```json
{
  "key_light":  {"energy": 800, "color": [1.0, 0.85, 0.7], "location": [-2, 3, 4]},
  "fill_light": {"energy": 200, "color": [0.8, 0.9, 1.0], "location": [3, 2, 2]},
  "rim_light":  {"energy": 400, "color": [0.6, 0.7, 1.0], "location": [0, -3, 2]},
  "world_strength": 0.05
}
```

#### POST `/face/blendshapes`
Set face blendshapes per frame (bulk, entire scene at once).
```json
{
  "start_frame": 0,
  "fps": 24,
  "frames": [
    {"jawOpen": 0.3, "mouthSmileLeft": 0.5, "browInnerUp": 0.4, ...},
    {"jawOpen": 0.5, "mouthSmileLeft": 0.6, "browInnerUp": 0.5, ...}
  ]
}
```

#### POST `/body/action`
Set body animation action.
```json
{
  "action_name": "Head Nod Yes",
  "start_frame": 0,
  "speed": 1.0,
  "loop": false
}
```

#### POST `/render/start`
Start render.
```json
{
  "start_frame": 0,
  "end_frame": 576,
  "fps": 24,
  "output_path": "C:/output/scene_1/frame_####.png",
  "renderer": "EEVEE",
  "resolution_x": 1920,
  "resolution_y": 1080,
  "samples": 64
}
```
Response: `{"job_id": "abc123", "status": "started"}`

#### GET `/render/status?job_id=abc123`
Poll render progress.
```json
{"job_id": "abc123", "status": "rendering", "progress": 0.45, "frames_done": 260, "total": 576}
```

#### POST `/scene/reset`
Clear all animation data and return to base pose.

---

## Camera Move Vocabulary

### Static Shots (cut between them)
```
ECU  Extreme Close-Up  — fills frame with eyes/mouth only
CU   Close-Up          — head and shoulders
MCU  Medium Close-Up   — chest up
MS   Medium Shot       — waist up
MLS  Medium Long Shot  — knee up
WS   Wide Shot         — full body + space around
EWS  Extreme Wide      — figure small in environment
```

### Moving Shots
```
DOLLY IN   — camera moves forward toward subject (builds tension/intimacy)
DOLLY OUT  — camera moves backward (reveals context, creates distance)
PAN        — camera rotates horizontally (follows action or reveals)
TILT UP    — camera rotates up (character looks powerful/hopeful)
TILT DOWN  — camera rotates down (vulnerability, looking down at something)
TRUCK      — camera moves sideways parallel to subject (tracking walk)
CRANE UP   — camera rises vertically (God POV, epic reveal)
PUSH/ZOOM  — lens zoom while dollying opposite (Vertigo: focus stays, bg distorts)
HANDHELD   — subtle random walk ±0.02m (intimacy, anxiety, realism)
```

### Blender Implementation (bpy)
```python
# Set keyframe for camera move
cam = bpy.data.objects['Camera']
cam.location = (x, y, z)
cam.keyframe_insert(data_path='location', frame=frame_num)

# Point camera at target
direction = target_pos - cam.location
rot_quat = direction.to_track_quat('-Z', 'Y')
cam.rotation_euler = rot_quat.to_euler()
cam.keyframe_insert(data_path='rotation_euler', frame=frame_num)

# Set FOV
cam.data.lens = 36 / math.tan(math.radians(fov_deg / 2))  # 36mm sensor
cam.data.keyframe_insert(data_path='lens', frame=frame_num)

# Set interpolation to smooth
for fcurve in cam.animation_data.action.fcurves:
    for kp in fcurve.keyframe_points:
        kp.interpolation = 'BEZIER'
        kp.easing = 'EASE_IN_OUT'
```

---

## LLM Prompt Templates

### StoryAgent System Prompt
```
You are a film scene breakdown assistant for a single-character 3D animation system.
Break the given story into 1-5 distinct scenes.

Rules:
- Each scene should flow naturally to the next
- Scenes should be 10-90 seconds each
- Only one character is available (no other characters appear on screen)
- Phone calls / conversations with others are OK (other voice is off-screen)
- Locations are abstract (lab, office, outdoor, home) — no complex environments

Output ONLY valid JSON array, no markdown, no explanation.
```

### DialogueAgent System Prompt
```
You are a screenplay writer for single-character 3D animation.
Write natural dialogue for the given scene.

Character voice: natural, conversational, emotionally authentic.
Sentence length: 5-20 words each. Keep it concise.
Beats: 3-8 per scene.
Pacing: Use pause_before_ms and pause_after_ms for natural rhythm.

Gesture options: none, nod, shake_head, look_around, reach_forward,
look_at_object, hand_to_chin, point_forward, wave, shrug, celebrate

State options: stand, walk, sit, lean_forward

Emotion options: neutral, happy, sad, angry, surprised, awe, excited, anxious, love, disgust

Output ONLY valid JSON array of beat objects, no markdown, no explanation.
```

### Minimal DirectorAgent Prompt (when LLM override needed)
```
Given this character beat, output the body action name.
Available actions: idle_breathe, walk, head_nod, head_shake, shrug, point, wave,
reach_forward, look_around, celebrate, sit_lean_forward
Beat: {json}
Output ONLY the action_name string.
```

---

## Configuration

### `llm_fw/config.json` (user edits this)
```json
{
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "api_key": "gsk_YOUR_KEY_HERE",
  "base_url": "https://api.groq.com/openai/v1",
  "temperature": 0.5,
  "max_tokens": 2048
}
```

**Provider options**:
| Provider | base_url | api_key source |
|----------|----------|---------------|
| Groq (recommended, fast, free) | `https://api.groq.com/openai/v1` | console.groq.com |
| OpenAI | `https://api.openai.com/v1` | platform.openai.com |
| Anthropic | (no base_url needed) | console.anthropic.com |
| Ollama (local, free) | `http://localhost:11434/v1` | none needed |
| OpenRouter | `https://openrouter.ai/api/v1` | openrouter.ai |

### `movie_config.json` (movie pipeline settings)
```json
{
  "blender_server_url": "http://localhost:7500",
  "blender_scene_file": "C:/me/proj/projface_v1/whole_body_production_ready.blend",
  "output_dir": "C:/me/proj/projface_v1/output/movies",
  "temp_dir": "C:/me/proj/projface_v1/temp/movie_pipeline",
  "fps": 24,
  "resolution": [1920, 1080],
  "renderer": "EEVEE",
  "eevee_samples": 64,
  "cycles_samples": 128,
  "audio_sample_rate": 22050,
  "character_name": "person01",
  "face_mesh_object": "Head_Mesh",
  "body_armature": "Armature"
}
```

---

## File Structure

```
projface_v1/
├── face_agents/
│   ├── __init__.py
│   ├── base.py
│   ├── agent_comm.py              (EXISTS — UDP comms)
│   ├── director_agent.py          (EXISTS — BeatAgent)
│   ├── body_director_agent.py     (EXISTS — BodyAgent)
│   ├── FaceCoordinator            (EXISTS — FaceAgent)
│   ├── story_agent.py             ← BUILD THIS FIRST
│   ├── dialogue_agent.py          ← BUILD SECOND
│   ├── camera_agent.py            ← BUILD THIRD
│   ├── lighting_agent.py          ← BUILD FOURTH
│   └── render_agent.py            ← BUILD FIFTH
│
├── llm_fw/
│   ├── providers/
│   │   ├── registry.py            (EXISTS — supports groq/openai/anthropic/ollama)
│   │   ├── openai_compat.py       (EXISTS — handles Groq natively)
│   │   └── anthropic_provider.py  (EXISTS)
│   └── agents/
│       └── director_agent.py      (EXISTS)
│
├── blender_director_server.py     ← BUILD (runs inside Blender)
├── movie_pipeline.py              ← BUILD (main orchestrator)
├── movie_config.json              ← CREATE (user edits)
│
├── wav2arkit.py                   (EXISTS — LipSyncAgent)
├── parler_voice.py                (EXISTS — VoiceAgent)
├── brain_inference.py             (EXISTS — emotion)
│
├── body_motion/
│   ├── flashavatar_colab.ipynb    (GS face, Phase 2)
│   └── questions.txt              (this session log)
│
├── docs/
│   └── MOVIE_PIPELINE_GUIDE.md    ← THIS FILE
│
└── whole_body_production_ready.blend  (EXISTS — main character scene)
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Story → scenes | <1 second | 1 Groq LLM call |
| Scene → dialogue | <0.5 sec | parallel LLM calls |
| Rule-based agents | <10ms total | camera, lighting, body |
| TTS (30s dialogue) | <2 seconds | Parler TTS Mini |
| Lip sync (30s) | <0.5 seconds | wav2arkit ONNX |
| Blender scene load | <2 seconds | pre-loaded server |
| Render 30s EEVEE GPU | <3 seconds | ~100 FPS on RTX |
| Render 30s EEVEE CPU | <30 seconds | ~10 FPS on CPU |
| ffmpeg stitch | <2 seconds | per scene |
| **Total 2-min movie** | **<25 seconds** | GPU, 3 scenes |
| **Total 2-min movie** | **<5 minutes** | CPU only |

**Groq free tier**: 6000 req/min, 30 req/min on llama-3.3-70b → use 8b for speed.  
**Cost per movie**: ~$0.001-0.005 on Groq, ~$0 on Ollama local.

---

## Implementation Roadmap

### Phase 0 — EXISTS AND WORKING ✓
- [x] `wav2arkit.py` — lip sync from audio
- [x] `parler_voice.py` — text to speech
- [x] `brain_inference.py` — emotion from audio
- [x] `face_agents/director_agent.py` — beat planning
- [x] `face_agents/body_director_agent.py` — body action selection
- [x] `face_agents/FaceCoordinator` — face blendshapes (lips/eyes/brows/cheeks/head)
- [x] `llm_fw/` — LLM provider framework (Groq/OpenAI/Anthropic/Ollama)
- [x] `blender_receiver.py` — receives UDP and applies to Blender
- [x] `whole_body_production_ready.blend` — character scene

### Phase 1 — MOVIE PIPELINE (Build Now)
- [ ] `blender_director_server.py` — HTTP server in Blender (~150 lines)
  - camera keyframes, lighting, face bulk blendshapes, body action, render trigger
- [ ] `face_agents/story_agent.py` — story → scenes (~50 lines)
- [ ] `face_agents/dialogue_agent.py` — scene → dialogue beats (~60 lines)
- [ ] `face_agents/camera_agent.py` — beats → camera keyframes (~80 lines)
- [ ] `face_agents/lighting_agent.py` — mood → lighting preset (~40 lines)
- [ ] `face_agents/render_agent.py` — trigger render + ffmpeg (~80 lines)
- [ ] `movie_pipeline.py` — orchestrates all agents (~120 lines)
- [ ] `movie_config.json` — user configuration file

### Phase 2 — QUALITY UPGRADE
- [ ] GS face avatar (FlashAvatar) — replace Blender mesh face with photorealistic GS
  - Train in Colab (flashavatar_colab.ipynb)
  - Render GS face frames separately
  - Composite onto body render
- [ ] Background scenes — HDRI + set dressing in Blender
- [ ] Multi-character support — second character for off-screen dialogue
- [ ] MoMask body motion — text → natural body motion (replaces FBX clip selection)

### Phase 3 — PRODUCTION POLISH
- [ ] Web UI — simple input box, download button
- [ ] Batch processing — generate 10 clips at once
- [ ] Style control — "cinematic", "documentary", "vlog" presets
- [ ] Music + SFX layer — background music selection by mood

---

## Running the Pipeline

### Quick Start (Phase 1, when built)

```bash
# Terminal 1: Start Blender with director server
blender --background --python blender_director_server.py -- --scene whole_body_production_ready.blend --port 7500

# Terminal 2: Generate a movie
python movie_pipeline.py --story "A scientist discovers a breakthrough and calls his family to share the news" --output output/test_movie.mp4
```

### Current Working Pipeline (Phase 0)
```bash
# Real-time interactive mode (existing)
python orchestrator_agents.py

# Then in Blender, run blender_receiver.py
# Speak or type to drive the avatar live
```

### Config for Groq (edit before running)
```json
// llm_fw/config.json
{
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "api_key": "gsk_YOUR_GROQ_KEY",
  "base_url": "https://api.groq.com/openai/v1"
}
```

---

## Design Principles

1. **One LLM call per level** — not one per beat. Process whole story at once.
2. **Rule-based where possible** — camera, lighting, body action don't need LLM.
3. **Parallel execution** — TTS runs while LLM runs. LipSync runs while camera plans.
4. **Existing tools first** — never replace working code. Extend it.
5. **LLM agnostic** — config.json switches provider. Agents never hardcode a model.
6. **Blender as renderer** — no GPU training, no GS until Phase 2. Use existing scene.
7. **Simple prompts** — closed-domain JSON output. 8B models handle it perfectly.
8. **Movie-level quality** — cinematic camera motion, correct lighting mood, natural pacing.

---

---

## Power Upgrades — AI + Internet + Unlimited Blender Control

> These upgrades transform the pipeline from a fixed-command system into a fully
> open, AI-driven creative engine. Add any or all independently.

---

### UPGRADE 1 — BlenderCodeAgent (Unlimited Blender Power)

**The problem with fixed API endpoints**: `/camera/keyframes` only does cameras.
Want to add a new prop? Change material? Build a scene from scratch? Need a new endpoint each time.

**The solution**: LLM generates raw bpy Python code → Blender executes it directly.
**Result**: AI can do ANYTHING in Blender. No fixed endpoints needed.

**New endpoint** in `blender_director_server.py`:
```python
@app.route('/execute_python', methods=['POST'])
def execute_python():
    code = request.json['code']
    result = {}
    try:
        exec(compile(code, '<llm_generated>', 'exec'),
             {'bpy': bpy, 'mathutils': mathutils, 'result': result})
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'code': code}), 400
```

**New agent** `face_agents/blender_code_agent.py`:
```python
class BlenderCodeAgent:
    """LLM generates bpy Python code for any Blender operation."""

    SYSTEM = """You are a Blender Python API expert.
Generate executable bpy code for the given task.
The code runs inside Blender's Python environment.
bpy and mathutils are already imported.
Store results in the 'result' dict for return values.
Output ONLY the Python code, no markdown fences, no explanation."""

    def run(self, task_description: str) -> str:
        """Returns bpy code string for any Blender task."""
        response = self.llm.chat([LLMMessage('user', task_description)],
                                  system=self.SYSTEM)
        code = response.content.strip()
        # Execute in Blender
        resp = requests.post(f'{BLENDER_URL}/execute_python', json={'code': code})
        return resp.json()
```

**Example tasks the BlenderCodeAgent can now do**:
```python
# Build a lab scene from scratch
blender_code_agent.run("Create a simple lab: add a desk (cube scaled 2x0.8x0.7), 
    a computer monitor (cube 0.4x0.05x0.3), three lab flasks (UV spheres 0.05 radius),
    position camera at [0,3,1.5] looking at origin, 3-point lighting setup")

# Create procedural background
blender_code_agent.run("Create a dark background wall behind character, 
    add subtle bokeh depth of field to camera with focus on character face")

# Add atmospheric particles
blender_code_agent.run("Add particle system to create floating dust motes,
    100 particles, random motion, 0.01 size, emission strength 0.5")

# Material changes
blender_code_agent.run("Change the character shirt material to dark blue,
    add slight roughness 0.8 so it looks like fabric")
```

**Power level**: With BlenderCodeAgent, the AI can build entire scenes from text,
add props, change materials, create particle effects, set up node trees — anything bpy supports.

---

### UPGRADE 2 — WebSearchAgent (Internet-Connected Agents)

**Why**: Agents that can search the internet produce more accurate, contextual content.
- ScriptAgent searches: "how does a PCR discovery actually happen in a lab?"
- DialogueAgent searches: "what would a scientist actually say when they discover a cure?"
- StyleAgent searches: "cinematography of 'First Man' lab scenes — camera style"
- LightingAgent searches: "how is a research lab lit at night — reference images"

**Implementation** (zero cost, no API key needed):

```python
# face_agents/web_search_agent.py

import urllib.request, urllib.parse, json

class WebSearchAgent:
    """Free internet search using DuckDuckGo Instant Answer API."""

    DDG_URL = 'https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1'

    def search(self, query: str, n_results: int = 3) -> list[str]:
        """Returns list of text snippets relevant to query. Free, no API key."""
        url = self.DDG_URL.format(query=urllib.parse.quote(query))
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
            results = []
            if data.get('AbstractText'):
                results.append(data['AbstractText'])
            for item in data.get('RelatedTopics', [])[:n_results]:
                if isinstance(item, dict) and item.get('Text'):
                    results.append(item['Text'])
            return results[:n_results]
        except Exception:
            return []  # fail silently, continue without search

    def search_summary(self, query: str) -> str:
        """Returns single paragraph summary for LLM context injection."""
        results = self.search(query)
        return ' '.join(results)[:500] if results else ''
```

**How agents use it — context injection**:
```python
class DialogueAgent:
    def run(self, scene: SceneDict) -> list[DialogueBeatDict]:
        # Search for contextual accuracy before writing dialogue
        search_context = self.web_search.search_summary(
            f"{scene['key_action']} realistic dialogue scene"
        )
        user_prompt = f"""
Scene: {scene['summary']}
Key action: {scene['key_action']}
Research context: {search_context}
Write the dialogue beats as JSON array.
"""
        # LLM now has real-world context → more accurate, natural dialogue
```

**StyleAgent** (uses image search for visual references):
```python
class StyleAgent:
    """Searches for cinematic references to inform camera + lighting decisions."""

    def get_style_context(self, mood: str, location: str) -> dict:
        search = self.web_search.search_summary(
            f"cinematography {mood} scene {location} camera angles lighting"
        )
        # Feed to CameraAgent and LightingAgent as context
        return {'style_notes': search}
```

---

### UPGRADE 3 — ResearchAgent (Story Accuracy)

**Why**: Before writing a movie about a scientist, doctor, lawyer — the AI should know
the real facts so dialogue sounds authentic, not generic.

```python
# face_agents/research_agent.py

class ResearchAgent:
    """Researches story topics to inject accurate context into DialogueAgent."""

    SYSTEM = """You are a research assistant. Given a story summary, identify
the 2-3 key topics that need real-world accuracy (technical terms, procedures,
authentic reactions). Search context will be provided. Output JSON:
{"topics": ["topic1", "topic2"], "key_facts": ["fact1", "fact2"], 
 "authentic_phrases": ["phrase1", "phrase2"]}"""

    def run(self, story_text: str) -> dict:
        # Step 1: LLM identifies what to research
        topics_response = self.llm.chat(
            [LLMMessage('user', f'Story: {story_text}\nWhat needs research?')],
            system=self.SYSTEM
        )
        topics = json.loads(topics_response.content)

        # Step 2: Search each topic
        facts = []
        for topic in topics.get('topics', []):
            facts.append(self.web_search.search_summary(topic))

        # Step 3: Return enriched context for DialogueAgent
        return {
            'key_facts': topics.get('key_facts', []),
            'authentic_phrases': topics.get('authentic_phrases', []),
            'search_context': ' '.join(facts)[:800]
        }
```

**Effect**: A story about a scientist discovering penicillin will have the AI
actually look up what a discovery moment feels like, what scientists say,
what the technical steps are — producing authentic, believable dialogue.

---

### UPGRADE 4 — MusicAgent (Automatic Background Score)

**Why**: A movie without music feels empty. Free royalty-free music exists.

```python
# face_agents/music_agent.py

class MusicAgent:
    """Selects and downloads royalty-free background music by mood."""

    # Curated free music by mood (Pixabay — truly free, no attribution needed)
    MOOD_TRACKS = {
        'awe':      'https://pixabay.com/music/search/cinematic%20discovery/',
        'excited':  'https://pixabay.com/music/search/upbeat%20motivational/',
        'sad':      'https://pixabay.com/music/search/emotional%20piano/',
        'angry':    'https://pixabay.com/music/search/dramatic%20tense/',
        'happy':    'https://pixabay.com/music/search/happy%20uplifting/',
        'neutral':  'https://pixabay.com/music/search/ambient%20background/',
        'anxious':  'https://pixabay.com/music/search/suspense%20tension/',
        'love':     'https://pixabay.com/music/search/romantic%20gentle/',
    }

    # Hardcoded reliable fallback tracks (direct MP3 links, always available)
    FALLBACK_TRACKS = {
        'awe':     'cinematic_discovery_01.mp3',
        'excited': 'upbeat_motivational_01.mp3',
        'sad':     'emotional_piano_01.mp3',
        'angry':   'dramatic_tension_01.mp3',
        'neutral': 'ambient_background_01.mp3',
    }

    def select_track(self, dominant_mood: str, duration_seconds: float) -> str:
        """Returns path to background music WAV file, looped/trimmed to duration."""
        track = self.FALLBACK_TRACKS.get(dominant_mood, self.FALLBACK_TRACKS['neutral'])
        # ffmpeg: loop track to fill duration, fade out last 3s
        output = f'temp/music_{dominant_mood}.wav'
        subprocess.run([
            'ffmpeg', '-y', '-stream_loop', '-1', '-i', track,
            '-t', str(duration_seconds), '-af', 'afade=t=out:st=' + str(duration_seconds-3) + ':d=3',
            output
        ], capture_output=True)
        return output
```

**RenderAgent uses MusicAgent**:
```python
# In RenderAgent.compose_final_mp4():
music = music_agent.select_track(dominant_mood, total_duration)
# ffmpeg: mix voice (0dB) + music (-18dB) + video
ffmpeg -i rendered_video.mp4 -i voice.wav -i music.wav \
    -filter_complex "[1:a]volume=1.0[voice];[2:a]volume=0.12[music];[voice][music]amix=inputs=2[a]" \
    -map 0:v -map '[a]' -c:v copy -c:a aac final_movie.mp4
```

---

### UPGRADE 5 — Enhanced Architecture (All Upgrades Combined)

```
USER INPUT
    │
    ▼
ResearchAgent ──────────────────────── WebSearchAgent
    │ key_facts + authentic_phrases          │ free DDG search
    ▼                                        │
StoryAgent ─────────────────────────────────┘
    │ scenes[]
    ├──────────────────── StyleAgent ── WebSearchAgent
    │                         │ visual references
    ▼                         ▼
DialogueAgent ◄──── research_context + style_notes
    │ dialogue beats (accurate, contextual)
    ▼
BeatAgent ──── CameraAgent ──── LightingAgent
    │               │                │
    │         keyframe curves    lighting presets
    ▼               │                │
VoiceAgent          └────────────────┘
    │                        │
LipSyncAgent                 ▼
    │              BlenderCodeAgent ◄─── LLM generates bpy code
    │               (scene building,         for any task
    │                props, materials,
    └───────────────►FX, particles)
                             │
                             ▼
                       RenderAgent
                      (EEVEE render)
                             │
                       MusicAgent ── mood → background score
                             │
                         ffmpeg
                       (video + voice + music)
                             │
                         final.mp4  ◄── movie-level output
```

---

### UPGRADE 6 — Agent Self-Improvement Loop

**Most powerful upgrade**: Agents evaluate their own output and retry if quality is low.

```python
class QualityAgent:
    """Evaluates output of other agents and requests retry if below threshold."""

    SYSTEM = """Rate the quality of this movie scene dialogue on 4 criteria:
    naturalness (0-10), emotional authenticity (0-10),
    pacing (0-10), cinematic potential (0-10).
    Output JSON: {"scores": {...}, "average": 0.0, "issues": [...], "pass": true/false}
    Pass if average >= 7.0."""

    def evaluate_dialogue(self, dialogue: list, scene: SceneDict) -> dict:
        prompt = f"Scene: {scene['summary']}\nDialogue: {json.dumps(dialogue)}"
        resp = self.llm.chat([LLMMessage('user', prompt)], system=self.SYSTEM)
        return json.loads(resp.content)

# In MoviePipeline:
for attempt in range(3):  # max 3 retries
    dialogue = dialogue_agent.run(scene, research_context)
    quality = quality_agent.evaluate_dialogue(dialogue, scene)
    if quality['pass']:
        break
    # Inject quality issues as feedback for next attempt
    scene['quality_feedback'] = quality['issues']
```

---

### Power Upgrade Summary

| Upgrade | What it adds | Cost | Complexity |
|---------|-------------|------|------------|
| BlenderCodeAgent | AI controls ALL of Blender via code | ~$0.001/scene | Medium |
| WebSearchAgent | Internet context for accuracy | $0 (DuckDuckGo free) | Low |
| ResearchAgent | Story topic research → authentic dialogue | ~$0.001/movie | Low |
| StyleAgent | Cinematic reference search → better camera | $0 | Low |
| MusicAgent | Auto background score by mood | $0 (royalty-free) | Low |
| QualityAgent | Self-improving retry loop | ~$0.002/movie | Medium |

**Total added cost per movie**: ~$0.005 maximum (Groq free tier covers hundreds/day)

---

### Updated Design Principles (with Power Upgrades)

1. **One LLM call per level** — batch process whole story, not per beat
2. **Rule-based where possible** — camera, lighting, body don't need LLM
3. **Parallel execution** — TTS, search, LLM all run simultaneously
4. **Existing tools first** — extend, never replace working code
5. **LLM agnostic** — config.json switches any provider
6. **Code generation > fixed endpoints** — BlenderCodeAgent = unlimited power
7. **Internet-grounded** — WebSearch gives agents real-world context
8. **Self-improving** — QualityAgent retries poor outputs automatically
9. **Simple prompts** — closed JSON output, 8B models sufficient
10. **Movie-level quality** — research → authentic story + cinematic camera + music

---

*Guide version: Aug 1 2026 | Pipeline: projface_v1 | Author: Cascade*
