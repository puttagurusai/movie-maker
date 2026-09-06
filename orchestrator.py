"""
orchestrator.py

Main "brain" script.
Pipeline:
- Accept sentences (temporarily: you paste raw JSON)
- For each sentence:
    1. Parler-TTS (emotion-styled voice) → temp/sentence_N.wav
    2. Rhubarb lip-sync → temp/sentence_N.json
    3. Send emotion (upper face) over UDP as type="emotion"
    4. Play audio + send timed visemes over UDP as type="viseme"

TEMPORARY MODE: Paste raw JSON instead of live LLM call.
Full format example:
{"sentences": [{"text": "Hello world.", "emotion": "happy", "intensity": 0.75}]}

Run:
    python orchestrator.py

Requirements:
    - parler-tts package + HuggingFace model cache (auto-download mini-v1)
    - rhubarb.exe (+ res/) in project root
    - (ANTHROPIC_API_KEY optional for now)
"""

import os
import sys
import time
import json
import socket
import subprocess
import tempfile
import threading
from pathlib import Path

import soundfile as sf
import sounddevice as sd
from anthropic import Anthropic

import emotion_map
from parler_voice import (
    build_voice_style,
    ensure_filler_sounds,
    generate_speech,
    load_parler,
)
import wav2arkit

# Lip-sync / face-from-audio backend:
#   "wav2arkit" — ONNX audio→ARKit mouth (default)
#   "rhubarb"   — phoneme visemes
#   "brain"     — trained Brain (audio + emotion + emotion_26d + prosody)
LIPSYNC_ENGINE = "wav2arkit"

# When LIPSYNC_ENGINE == "brain":
#   allow_stub=True uses a random MLP if models/brain/*.pt missing (dev only)
BRAIN_ALLOW_STUB = False


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 9001

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# Rhubarb binary name (Windows vs others)
if os.name == "nt":
    RHUBARB_CMD = "rhubarb.exe"
else:
    RHUBARB_CMD = "./rhubarb"

# IMPORTANT for -r phonetic:
# Rhubarb needs the 'res' folder (containing sphinx models) from the release zip
# placed next to rhubarb.exe. Otherwise you get "could not find resource file" error.

# Claude settings
CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a conversational AI assistant. Always respond ONLY with valid JSON in this exact format, no other text:
{"sentences": [{"text": "sentence text here", "emotion": "happy", "intensity": 0.7}]}

Emotions must be one of: neutral, happy, sad, angry, surprised, disgusted, fearful, sarcastic, thinking.
Intensity is a float from 0.0 to 1.0.
Split your response into natural spoken sentences. Keep sentences reasonably short."""

# Viseme (Rhubarb A-X) to ARKit blendshape mapping (mouth / jaw area)
# These values are tuned for distinct visible phoneme shapes on ARKit rigs.
VISEME_MAP = {

    # X — silence / rest
    "X": {
        "jawOpen":0.0, "mouthClose":0.0,
        "mouthPucker":0.0, "mouthFunnel":0.0,
        "mouthRollLower":0.05, "mouthRollUpper":0.02,
        "mouthShrugLower":0.0, "mouthShrugUpper":0.0,
        "mouthSmileLeft":0.0, "mouthSmileRight":0.0,
        "mouthFrownLeft":0.0, "mouthFrownRight":0.0,
        "mouthDimpleLeft":0.0, "mouthDimpleRight":0.0,
        "mouthUpperUpLeft":0.0, "mouthUpperUpRight":0.0,
        "mouthLowerDownLeft":0.0, "mouthLowerDownRight":0.0,
        "mouthPressLeft":0.0, "mouthPressRight":0.0,
        "mouthStretchLeft":0.0, "mouthStretchRight":0.0,
        "tongueOut":0.0,
    },

    # A — P B M — lips pressed together
    "A": {
        "jawOpen":0.0, "mouthClose":1.0,
        "mouthPucker":0.0, "mouthFunnel":0.0,
        "mouthRollLower":0.3, "mouthRollUpper":0.3,
        "mouthShrugLower":0.0, "mouthShrugUpper":0.0,
        "mouthSmileLeft":0.0, "mouthSmileRight":0.0,
        "mouthFrownLeft":0.0, "mouthFrownRight":0.0,
        "mouthDimpleLeft":0.0, "mouthDimpleRight":0.0,
        "mouthUpperUpLeft":0.0, "mouthUpperUpRight":0.0,
        "mouthLowerDownLeft":0.0, "mouthLowerDownRight":0.0,
        "mouthPressLeft":0.8, "mouthPressRight":0.8,
        "mouthStretchLeft":0.0, "mouthStretchRight":0.0,
        "tongueOut":0.0,
    },

    # B — K G NG — back of throat, jaw open, mouth open
    "B": {
        "jawOpen":0.6, "mouthClose":0.0,
        "mouthPucker":0.0, "mouthFunnel":0.0,
        "mouthRollLower":0.0, "mouthRollUpper":0.0,
        "mouthShrugLower":0.1, "mouthShrugUpper":0.1,
        "mouthSmileLeft":0.0, "mouthSmileRight":0.0,
        "mouthFrownLeft":0.0, "mouthFrownRight":0.0,
        "mouthDimpleLeft":0.0, "mouthDimpleRight":0.0,
        "mouthUpperUpLeft":0.2, "mouthUpperUpRight":0.2,
        "mouthLowerDownLeft":0.2, "mouthLowerDownRight":0.2,
        "mouthPressLeft":0.0, "mouthPressRight":0.0,
        "mouthStretchLeft":0.0, "mouthStretchRight":0.0,
        "tongueOut":0.0,
    },

    # C — EH IH IY — wide smile shape, teeth visible
    "C": {
        "jawOpen":0.3, "mouthClose":0.0,
        "mouthPucker":0.0, "mouthFunnel":0.0,
        "mouthRollLower":0.0, "mouthRollUpper":0.0,
        "mouthShrugLower":0.0, "mouthShrugUpper":0.0,
        "mouthSmileLeft":0.7, "mouthSmileRight":0.7,
        "mouthFrownLeft":0.0, "mouthFrownRight":0.0,
        "mouthDimpleLeft":0.3, "mouthDimpleRight":0.3,
        "mouthUpperUpLeft":0.5, "mouthUpperUpRight":0.5,
        "mouthLowerDownLeft":0.3, "mouthLowerDownRight":0.3,
        "mouthPressLeft":0.0, "mouthPressRight":0.0,
        "mouthStretchLeft":0.5, "mouthStretchRight":0.5,
        "tongueOut":0.0,
    },

    # D — AH AA — wide open jaw, natural open mouth
    "D": {
        "jawOpen":0.85, "mouthClose":0.0,
        "mouthPucker":0.0, "mouthFunnel":0.0,
        "mouthRollLower":0.0, "mouthRollUpper":0.0,
        "mouthShrugLower":0.2, "mouthShrugUpper":0.0,
        "mouthSmileLeft":0.0, "mouthSmileRight":0.0,
        "mouthFrownLeft":0.0, "mouthFrownRight":0.0,
        "mouthDimpleLeft":0.0, "mouthDimpleRight":0.0,
        "mouthUpperUpLeft":0.2, "mouthUpperUpRight":0.2,
        "mouthLowerDownLeft":0.5, "mouthLowerDownRight":0.5,
        "mouthPressLeft":0.0, "mouthPressRight":0.0,
        "mouthStretchLeft":0.0, "mouthStretchRight":0.0,
        "tongueOut":0.0,
    },

    # E — OW OH — mid open, slightly rounded
    "E": {
        "jawOpen":0.5, "mouthClose":0.0,
        "mouthPucker":0.2, "mouthFunnel":0.3,
        "mouthRollLower":0.1, "mouthRollUpper":0.0,
        "mouthShrugLower":0.1, "mouthShrugUpper":0.1,
        "mouthSmileLeft":0.0, "mouthSmileRight":0.0,
        "mouthFrownLeft":0.0, "mouthFrownRight":0.0,
        "mouthDimpleLeft":0.0, "mouthDimpleRight":0.0,
        "mouthUpperUpLeft":0.1, "mouthUpperUpRight":0.1,
        "mouthLowerDownLeft":0.3, "mouthLowerDownRight":0.3,
        "mouthPressLeft":0.0, "mouthPressRight":0.0,
        "mouthStretchLeft":0.0, "mouthStretchRight":0.0,
        "tongueOut":0.0,
    },

    # F — UW UH OO — tight round pucker
    "F": {
        "jawOpen":0.1, "mouthClose":0.0,
        "mouthPucker":0.95, "mouthFunnel":0.6,
        "mouthRollLower":0.2, "mouthRollUpper":0.1,
        "mouthShrugLower":0.0, "mouthShrugUpper":0.0,
        "mouthSmileLeft":0.0, "mouthSmileRight":0.0,
        "mouthFrownLeft":0.0, "mouthFrownRight":0.0,
        "mouthDimpleLeft":0.0, "mouthDimpleRight":0.0,
        "mouthUpperUpLeft":0.0, "mouthUpperUpRight":0.0,
        "mouthLowerDownLeft":0.0, "mouthLowerDownRight":0.0,
        "mouthPressLeft":0.0, "mouthPressRight":0.0,
        "mouthStretchLeft":0.0, "mouthStretchRight":0.0,
        "tongueOut":0.0,
    },

    # G — F V — lower lip under upper teeth
    "G": {
        "jawOpen":0.15, "mouthClose":0.0,
        "mouthPucker":0.0, "mouthFunnel":0.0,
        "mouthRollLower":0.0, "mouthRollUpper":0.0,
        "mouthShrugLower":0.0, "mouthShrugUpper":0.0,
        "mouthSmileLeft":0.0, "mouthSmileRight":0.0,
        "mouthFrownLeft":0.0, "mouthFrownRight":0.0,
        "mouthDimpleLeft":0.0, "mouthDimpleRight":0.0,
        "mouthUpperUpLeft":0.5, "mouthUpperUpRight":0.5,
        "mouthLowerDownLeft":0.8, "mouthLowerDownRight":0.8,
        "mouthPressLeft":0.0, "mouthPressRight":0.0,
        "mouthStretchLeft":0.0, "mouthStretchRight":0.0,
        "tongueOut":0.0,
    },

    # H — L TH — tongue tip, slightly open
    "H": {
        "jawOpen":0.35, "mouthClose":0.0,
        "mouthPucker":0.0, "mouthFunnel":0.0,
        "mouthRollLower":0.0, "mouthRollUpper":0.0,
        "mouthShrugLower":0.0, "mouthShrugUpper":0.0,
        "mouthSmileLeft":0.2, "mouthSmileRight":0.2,
        "mouthFrownLeft":0.0, "mouthFrownRight":0.0,
        "mouthDimpleLeft":0.0, "mouthDimpleRight":0.0,
        "mouthUpperUpLeft":0.3, "mouthUpperUpRight":0.3,
        "mouthLowerDownLeft":0.2, "mouthLowerDownRight":0.2,
        "mouthPressLeft":0.0, "mouthPressRight":0.0,
        "mouthStretchLeft":0.2, "mouthStretchRight":0.2,
        "tongueOut":0.3,
    },
}

# Default silence viseme
DEFAULT_VISEME = "X"

# Modifiers to merge into emotion during playback based on current viseme (Step 2)
VISEME_BROW_MODIFIERS = {
    "X": {"browInnerUp": 0.0,  "eyeSquintLeft": 0.0,  "eyeSquintRight": 0.0},
    "A": {"browInnerUp": 0.0,  "eyeSquintLeft": 0.0,  "eyeSquintRight": 0.0},
    "B": {"browInnerUp": 0.0,  "eyeSquintLeft": 0.05, "eyeSquintRight": 0.05},
    "C": {"browInnerUp": 0.05, "eyeSquintLeft": 0.08, "eyeSquintRight": 0.08},
    "D": {"browInnerUp": 0.04, "eyeSquintLeft": 0.06, "eyeSquintRight": 0.06},
    "E": {"browInnerUp": 0.06, "eyeSquintLeft": 0.0,  "eyeSquintRight": 0.0},
    "F": {"browInnerUp": 0.03, "eyeSquintLeft": 0.07, "eyeSquintRight": 0.07},
    "G": {"browInnerUp": 0.04, "eyeSquintLeft": 0.05, "eyeSquintRight": 0.05},
    "H": {"browInnerUp": 0.02, "eyeSquintLeft": 0.0,  "eyeSquintRight": 0.0},
}

# Current emotion state (used by re-sender thread)
current_emotion = "neutral"
current_intensity = 0.0
_current_viseme = "X"
_emotion_thread = None
_stop_emotion_thread = False

# For onset/decay (Step D)
current_emotion_values = {}


# ---------------------------------------------------------------------------
# UDP HELPERS
# ---------------------------------------------------------------------------
_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_udp(packet: dict):
    """Send a JSON packet over UDP to Blender."""
    try:
        data = json.dumps(packet).encode("utf-8")
        _udp_sock.sendto(data, (UDP_IP, UDP_PORT))
    except Exception as e:
        print(f"[orchestrator] UDP send failed: {e}")


def send_emotion(emotion: str, intensity: float):
    """Send upper-face emotion blendshapes. Also updates current state for re-sender."""
    global current_emotion, current_intensity
    current_emotion = emotion
    current_intensity = intensity
    blendshapes = emotion_map.get_blendshapes(emotion, intensity)
    packet = {
        "type": "emotion",
        "emotion": emotion,
        "blendshapes": blendshapes
    }
    send_udp(packet)
    print(f"  [UDP] Sent EMOTION packet: {emotion} @ {intensity:.2f}  → {blendshapes}")


def send_viseme(viseme: str):
    """Convert a Rhubarb viseme letter to blendshapes and send as type 'viseme'."""
    viseme = viseme.upper() if viseme else DEFAULT_VISEME
    if viseme not in VISEME_MAP:
        viseme = DEFAULT_VISEME
    blendshapes = VISEME_MAP[viseme]
    packet = {
        "type": "viseme",
        "blendshapes": blendshapes
    }
    send_udp(packet)
    print(f"  [UDP] Sent VISEME '{viseme}' → {blendshapes}")


def ramp_emotion(target_blendshapes, duration_ms=300, steps=10):
    """Ramp the current emotion to target over duration_ms using steps.
    Blocks during the ramp (for onset before audio).
    """
    global current_emotion_values
    if not target_blendshapes:
        return
    interval = duration_ms / max(1, steps) / 1000.0
    start_vals = current_emotion_values.copy()
    all_keys = set(start_vals.keys()) | set(target_blendshapes.keys())
    for step in range(1, steps + 1):
        frac = step / steps
        interp = {}
        for k in all_keys:
            s = start_vals.get(k, 0.0)
            t = target_blendshapes.get(k, 0.0)
            interp[k] = s * (1 - frac) + t * frac
        # Merge viseme brow modifiers during the ramp (for mod during build)
        mods = VISEME_BROW_MODIFIERS.get(_current_viseme, VISEME_BROW_MODIFIERS["X"])
        for k, v in mods.items():
            if k in interp:
                interp[k] = max(0.0, min(1.0, interp[k] + v))
            else:
                interp[k] = v
        packet = {"type": "emotion", "emotion": current_emotion, "blendshapes": interp}
        send_udp(packet)
        current_emotion_values = interp.copy()
        time.sleep(interval)
    current_emotion_values = target_blendshapes.copy()


# ---------------------------------------------------------------------------
# CLAUDE
# ---------------------------------------------------------------------------
def get_claude_client() -> Anthropic | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("INFO: ANTHROPIC_API_KEY not set (this is OK in temporary JSON-paste mode).")
        print("      You can set it later when we reconnect the live LLM.")
        return None
    return Anthropic(api_key=api_key)


def get_sentences_from_claude(client: Anthropic, user_text: str) -> list[dict]:
    """Call Claude and return parsed list of {text, emotion, intensity}."""
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_text}
            ]
        )
        # Extract text from response
        content = response.content[0].text.strip() if response.content else ""
        
        # Clean up in case model adds markdown
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        data = json.loads(content)
        sentences = data.get("sentences", [])
        
        # Basic validation
        cleaned = []
        for s in sentences:
            text = s.get("text", "").strip()
            emotion = s.get("emotion", "neutral").lower()
            intensity = float(s.get("intensity", 0.7))
            if text:
                cleaned.append({
                    "text": text,
                    "emotion": emotion if emotion in emotion_map.EMOTION_MAP else "neutral",
                    "intensity": max(0.0, min(1.0, intensity))
                })
        return cleaned or [{"text": user_text, "emotion": "neutral", "intensity": 0.6}]

    except Exception as e:
        print(f"[Claude] Error: {e}")
        # Fallback
        return [{"text": user_text, "emotion": "neutral", "intensity": 0.6}]


def get_sentences_from_raw_json() -> list[dict] | None:
    """
    TEMP MODE: Paste raw JSON here.

    IMPORTANT: 
    - Paste the COMPLETE JSON (you can paste multiple lines at once)
    - Then press ENTER on a BLANK LINE to submit
    - The JSON must start with { and contain "sentences"

    Big example JSON with surprise, angry, happy/smile, fearful, sarcastic etc. (copy-paste this):
    {"sentences": [
      {"text": "Oh my god, what is that thing coming toward us right now?", "emotion": "surprised", "intensity": 0.92},
      {"text": "I'm so angry, this is completely unacceptable behavior!", "emotion": "angry", "intensity": 0.85},
      {"text": "Wait... actually the way it's bouncing around is kind of hilarious.", "emotion": "happy", "intensity": 0.7},
      {"text": "I'm starting to get really scared about what happens next.", "emotion": "fearful", "intensity": 0.78},
      {"text": "Honestly this entire situation has been totally ridiculous from the beginning.", "emotion": "sarcastic", "intensity": 0.65}
    ]}
    """
    print("\n" + "=" * 55)
    print("Paste the FULL JSON (multi-line OK).")
    print("When done pasting, press ENTER on an EMPTY line.")
    print("Type 'quit' alone on a line to exit.")
    print("=" * 55)

    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return None

        stripped = line.strip()
        if stripped.lower() in ("quit", "exit", "q"):
            return None

        if stripped == "":
            # blank line = finish input
            break

        lines.append(line)

    raw = "\n".join(lines).strip()

    if not raw:
        print("No content received. Try again.")
        return get_sentences_from_raw_json()

    try:
        data = json.loads(raw)
        sentences = data.get("sentences", [])

        if not isinstance(sentences, list) or len(sentences) == 0:
            print("JSON parsed but no valid 'sentences' array.")
            print("Make sure your JSON has this structure:")
            print('  {"sentences": [ {"text": "...", "emotion": "...", "intensity": 0.7} ] }')
            return get_sentences_from_raw_json()

        cleaned = []
        for s in sentences:
            text = str(s.get("text", "")).strip()
            emotion = str(s.get("emotion", "neutral")).lower().strip()
            try:
                intensity = float(s.get("intensity", 0.7))
            except (TypeError, ValueError):
                intensity = 0.7

            if text:
                cleaned.append({
                    "text": text,
                    "emotion": emotion if emotion in emotion_map.EMOTION_MAP else "neutral",
                    "intensity": max(0.0, min(1.0, intensity))
                })

        if cleaned:
            return cleaned
        else:
            print("No valid sentences after cleaning. Try again.")
            return get_sentences_from_raw_json()

    except json.JSONDecodeError as e:
        print(f"Invalid JSON ({e}).")
        print("Tip: Copy the entire JSON block at once, including { and }")
        return get_sentences_from_raw_json()
    except Exception as e:
        print(f"Error: {e}")
        return get_sentences_from_raw_json()


# ---------------------------------------------------------------------------
# TTS + LIP SYNC + PLAYBACK
# ---------------------------------------------------------------------------
def generate_tts(text: str, emotion: str, intensity: float, output_path: str):
    """
    Generate emotion-styled speech with Parler-TTS.
    Returns (audio_array, sample_rate). WAV is written to output_path.
    """
    print(
        f"  [TTS/Parler] \"{text[:60]}{'...' if len(text) > 60 else ''}\"  "
        f"emotion={emotion} intensity={intensity:.2f}"
    )
    style = build_voice_style(emotion, intensity)
    print(f"  [TTS/Parler] style: {style[:120]}...")
    audio, sample_rate = generate_speech(
        text=text,
        voice_style=style,
        output_path=output_path,
        play_audio=False,
    )
    return audio, sample_rate


def run_rhubarb(wav_path: str, sentence_idx: int) -> list[dict]:
    """Run Rhubarb lip sync and return list of mouth cues [{'start': float, 'value': str}, ...]."""
    json_path = TEMP_DIR / f"sentence_{sentence_idx}.json"
    
    cmd = [
        RHUBARB_CMD,
        "-f", "json",
        "-o", str(json_path),
        wav_path,
        "-r", "phonetic"
    ]
    
    print(f"  [Rhubarb] Running lip sync with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            print(f"  [Rhubarb] ERROR (return code {result.returncode}): {result.stderr.strip()}")
            print(f"  [Rhubarb] stdout: {result.stdout.strip()}")
            print("  [Rhubarb] TIP: Rhubarb needs its 'res' folder for phonetic recognition.")
            print("  Make sure you copied the 'res' folder from the Rhubarb zip next to rhubarb.exe")
        else:
            print("  [Rhubarb] Command completed successfully.")
    except FileNotFoundError:
        print(f"ERROR: Rhubarb binary not found at '{RHUBARB_CMD}'.")
        print("Make sure you placed rhubarb or rhubarb.exe in the project folder.")
        return []
    except Exception as e:
        print(f"  [Rhubarb] Error: {e}")
        return []

    # Parse Rhubarb output
    if not json_path.exists():
        print("  [Rhubarb] No output JSON created — lip sync failed.")
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Rhubarb JSON structure usually has "mouthCues"
        cues = data.get("mouthCues", [])
        # Normalize to our simple format
        normalized = []
        for cue in cues:
            start = float(cue.get("start", 0.0))
            value = cue.get("value", "X")
            normalized.append({"start": start, "value": value})
        
        print(f"  [Rhubarb] Got {len(normalized)} mouth cues.")
        return normalized
    except Exception as e:
        print(f"  [Rhubarb] Failed to parse JSON: {e}")
        return []


def _emotion_resender():
    """Background thread: re-send current emotion (merged with viseme brow modifiers) every 150ms."""
    global _stop_emotion_thread
    while not _stop_emotion_thread:
        base = emotion_map.get_blendshapes(current_emotion, current_intensity)
        mods = VISEME_BROW_MODIFIERS.get(_current_viseme, VISEME_BROW_MODIFIERS["X"])
        merged = base.copy()
        for k, v in mods.items():
            if k in merged:
                merged[k] = max(0.0, min(1.0, merged[k] + v))
            else:
                merged[k] = v
        # Send merged directly to avoid resetting globals
        packet = {"type": "emotion", "emotion": current_emotion, "blendshapes": merged}
        send_udp(packet)
        current_emotion_values = merged.copy()
        time.sleep(0.15)


def play_audio_and_visemes(
    wav_path: str,
    sample_rate: int,
    cues: list[dict] | None = None,
    emotion: str = "neutral",
    intensity: float = 0.0,
    arkit_frames: list[dict] | None = None,
    arkit_fps: float = 30.0,
):
    """
    Play the audio and stream mouth shapes to Blender at ~30 fps.

    Lip-sync sources (prefer arkit_frames when provided):
      - arkit_frames: list of blendshape dicts from wav2arkit (30 fps)
      - cues: Rhubarb mouth cues (legacy letter A–X → VISEME_MAP)
    """
    global _stop_emotion_thread, _emotion_thread, _current_viseme

    use_arkit = bool(arkit_frames)
    if not use_arkit and not cues:
        cues = [{"start": 0.0, "value": DEFAULT_VISEME}]

    # Load audio for playback info
    audio_data, _ = sf.read(wav_path, dtype="float32")
    duration = len(audio_data) / sample_rate

    mode = "wav2arkit" if use_arkit else "rhubarb"
    print(f"  [Playback] Starting audio ({duration:.2f}s) lipsync={mode} @ ~30fps → Blender")

    # Rhubarb path: prepare interpolated cue dicts
    prepared_cues = []
    if not use_arkit:
        for c in cues or []:
            val = c.get("value", DEFAULT_VISEME)
            v = val.upper() if val else DEFAULT_VISEME
            d = VISEME_MAP.get(v, VISEME_MAP[DEFAULT_VISEME]).copy()
            prepared_cues.append({"start": c["start"], "dict": d, "value": v})
        if not prepared_cues:
            prepared_cues = [{"start": 0.0, "dict": VISEME_MAP[DEFAULT_VISEME].copy(), "value": DEFAULT_VISEME}]

    # Start audio (non-blocking)
    sd.play(audio_data, sample_rate)

    # Nod for sentence start via head packet
    send_udp({"type": "head", "pitch": 0.12, "yaw": 0.0, "roll": 0.0})
    time.sleep(0.2)
    send_udp({"type": "head", "pitch": 0.0, "yaw": 0.0, "roll": 0.0})

    # Start emotion onset ramp as thread CONCURRENTLY with audio
    target = emotion_map.get_blendshapes(emotion, intensity)
    print(f"  [Emotion] Starting CONCURRENT onset ramp for {emotion} @ {intensity:.2f} over 300ms (while audio plays)")
    onset_thread = threading.Thread(target=lambda: ramp_emotion(target, duration_ms=300, steps=10), daemon=True)
    onset_thread.start()

    def delayed_resender_start():
        time.sleep(0.31)
        global _stop_emotion_thread, _emotion_thread
        _stop_emotion_thread = False
        _emotion_thread = threading.Thread(target=_emotion_resender, daemon=True)
        _emotion_thread.start()
    threading.Thread(target=delayed_resender_start, daemon=True).start()

    _current_viseme = DEFAULT_VISEME
    playback_start = time.time()
    send_viseme(DEFAULT_VISEME)
    last_mouth = VISEME_MAP[DEFAULT_VISEME].copy()

    frame_interval = 1.0 / 30.0
    last_frame_time = playback_start

    while True:
        now = time.time()
        current_t = now - playback_start
        if current_t >= duration:
            break

        if use_arkit:
            # Nearest-frame sample from wav2arkit sequence
            interp = wav2arkit.frame_at_time(arkit_frames, current_t, fps=arkit_fps)
            if not interp:
                interp = last_mouth
            _current_viseme = "X"  # no discrete letter; skip brow viseme mods
        else:
            current_idx = 0
            for i, c in enumerate(prepared_cues):
                if c["start"] <= current_t:
                    current_idx = i
                else:
                    break

            next_idx = min(current_idx + 1, len(prepared_cues) - 1)
            curr = prepared_cues[current_idx]
            nxt = prepared_cues[next_idx]
            cue_start = curr["start"]
            cue_end = nxt["start"]

            if cue_end > cue_start:
                cue_dur = cue_end - cue_start
                pos = current_t - cue_start
                local_frac = pos / cue_dur
                if local_frac < 0.2:
                    if current_idx > 0:
                        prev_dict = prepared_cues[current_idx - 1]["dict"]
                        sub_frac = local_frac / 0.2
                        all_keys = set(prev_dict.keys()) | set(curr["dict"].keys())
                        interp = {
                            k: prev_dict.get(k, 0.0) * (1 - sub_frac) + curr["dict"].get(k, 0.0) * sub_frac
                            for k in all_keys
                        }
                    else:
                        interp = curr["dict"].copy()
                elif local_frac > 0.8:
                    sub_frac = (local_frac - 0.8) / 0.2
                    all_keys = set(curr["dict"].keys()) | set(nxt["dict"].keys())
                    interp = {
                        k: curr["dict"].get(k, 0.0) * (1 - sub_frac) + nxt["dict"].get(k, 0.0) * sub_frac
                        for k in all_keys
                    }
                else:
                    interp = curr["dict"].copy()
            else:
                interp = curr["dict"].copy()
            _current_viseme = prepared_cues[current_idx].get("value", DEFAULT_VISEME)

        send_udp({"type": "viseme", "blendshapes": interp})
        last_mouth = interp

        sleep_time = frame_interval - (time.time() - last_frame_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_frame_time = time.time()

    sd.wait()

    _stop_emotion_thread = True
    if _emotion_thread and _emotion_thread.is_alive():
        _emotion_thread.join(timeout=0.5)

    print("  [Playback] Smoothly returning mouth/face to neutral (no snap)...")
    rest_mouth = VISEME_MAP[DEFAULT_VISEME]
    settle_ms = 380
    settle_steps = 14
    step_dt = settle_ms / settle_steps / 1000.0
    all_mouth_keys = set(last_mouth.keys()) | set(rest_mouth.keys())
    for step in range(1, settle_steps + 1):
        frac = step / settle_steps
        ease = 1.0 - (1.0 - frac) ** 2
        interp = {
            k: last_mouth.get(k, 0.0) * (1.0 - ease) + rest_mouth.get(k, 0.0) * ease
            for k in all_mouth_keys
        }
        send_udp({"type": "viseme", "blendshapes": interp})
        time.sleep(step_dt)

    _current_viseme = DEFAULT_VISEME
    send_udp({
        "type": "rest_pose",
        "smooth": True,
        "blendshapes": emotion_map.NEUTRAL_REST.copy(),
    })
    time.sleep(0.15)
    print("  [Playback] Done.")


def process_sentence(client: Anthropic, sentence: dict, idx: int):
    """Full pipeline for one sentence."""
    text = sentence["text"]
    emotion = sentence["emotion"]
    intensity = sentence["intensity"]

    print(f"\n{'='*50}")
    print(f"[Sentence {idx}] {text}")
    print(f"{'='*50}")

    # 1. Parler TTS (emotion-styled voice → WAV)
    wav_path = str(TEMP_DIR / f"sentence_{idx}.wav")
    audio_data, sr = generate_tts(text, emotion, intensity, wav_path)
    duration = len(audio_data) / float(sr)

    # 2. Face-from-audio: wav2arkit | rhubarb | brain
    cues = None
    arkit_frames = None
    arkit_fps = 30.0
    engine = (LIPSYNC_ENGINE or "wav2arkit").lower().strip()
    if engine == "brain":
        print("  [Brain] Building 4 inputs: audio + emotion + emotion_26d + prosody...")
        import brain_inference
        import prosody_gpu
        from emotion_manager import get_emotion_manager

        # 1) audio = wav_path
        # 2) emotion label + intensity (from JSON / Groq)
        # 3) emotion 26-D via EmotionManager (emotion_stats.json)
        emo_mgr = get_emotion_manager()
        device_26 = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        e26 = emo_mgr.get_emotion_26d_tensor(
            groq_emotion=emotion,
            intensity=intensity,
            device=device_26,
        )  # [1, 26]
        print(
            f"  [Brain] emotion={emotion} → {emo_mgr.resolve_label(emotion)} "
            f"intensity={intensity:.2f} emotion_26d={tuple(e26.shape)} "
            f"L2={e26.norm():.3f}"
        )
        # 4) prosody [T, 3]
        prosody = prosody_gpu.extract_prosody(wav_path)
        print(f"  [Brain] prosody={tuple(prosody.shape)} device={prosody.device}")

        # ONE entry: run_brain → T frames × 52 ARKit names (mouth keys for UDP viseme)
        arkit_frames, arkit_fps, _raw = brain_inference.run_brain(
            wav_path=wav_path,
            emotion_label=emotion,
            intensity=intensity,
            mouth_only=True,  # lips stream; upper face still from emotion_map UDP
        )
    elif engine == "wav2arkit":
        print("  [LipSync] Running wav2arkit (audio → ARKit blendshapes)...")
        arkit_frames, arkit_fps, _raw = wav2arkit.audio_file_to_frames(wav_path, mouth_only=True)
    else:
        print("  [LipSync] Running Rhubarb...")
        cues = run_rhubarb(wav_path, idx)

    # 3. Play + stream mouth + concurrent emotion onset
    global current_emotion, current_intensity
    current_emotion = emotion
    current_intensity = intensity
    play_audio_and_visemes(
        wav_path,
        sr,
        cues=cues,
        emotion=emotion,
        intensity=intensity,
        arkit_frames=arkit_frames,
        arkit_fps=arkit_fps,
    )

    # 5. Decay — ramp upper-face emotion DOWN to neutral after audio (Step D).
    # Mouth already eased closed in play_audio_and_visemes; this softens brows/eyes/smile.
    # Duration scales with sentence length (shorter sentences decay faster).
    decay_ms = max(350, min(700, int(duration * 1000 * 0.35)))
    neutral = emotion_map.get_blendshapes("neutral", 1.0)
    print(f"  [Emotion] Ramping DECAY to neutral rest over {decay_ms}ms")
    ramp_emotion(neutral, duration_ms=decay_ms, steps=14)

    # Final soft rest so any leftover keys also glide home (no hard snap)
    send_udp({
        "type": "rest_pose",
        "smooth": True,
        "blendshapes": emotion_map.NEUTRAL_REST.copy(),
    })


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(f"ORCHESTRATOR — Parler-TTS + lipsync={LIPSYNC_ENGINE} + Blender  (JSON mode)")
    print("=" * 60)
    print(">>> STEP 1: In Blender, run this first:")
    print("    bpy.ops.face.stream_receiver()")
    print(">>> STEP 2: Come back here and paste your JSON.")
    print("=" * 60)
    print("Paste the FULL JSON (multi-line supported).")
    print("Press ENTER on a blank line after pasting to submit.")
    print("Later we will reconnect the live Claude call.\n")

    # We still load the client for future use, but don't call it yet.
    try:
        client = get_claude_client()
        print("[OK] Claude client initialized (not used in temp mode).")
    except SystemExit:
        client = None  # Will continue without it

    # Load Parler once (GPU/CPU, half precision, warmup)
    try:
        load_parler()
        print("[OK] Parler-TTS loaded.")
        # Skip filler pre-gen at startup (each is a full Parler pass ~10s).
        # Call ensure_filler_sounds() only when you need LLM wait cover.
        print("[OK] Parler ready (fillers lazy — not pre-generated).")
    except Exception as e:
        print(f"ERROR loading Parler-TTS: {e}")
        print("Install: pip install git+https://github.com/huggingface/parler-tts.git")
        print("         pip install transformers accelerate torch")
        sys.exit(1)

    # Preload face-from-audio backend
    engine = (LIPSYNC_ENGINE or "wav2arkit").lower().strip()
    if engine == "brain":
        try:
            import brain_inference
            from emotion_manager import get_emotion_manager
            get_emotion_manager()  # requires models/brain/emotion_stats.json
            brain_inference.get_brain(allow_stub=BRAIN_ALLOW_STUB)
            print("[OK] Brain ready (audio + emotion + emotion_26d + prosody).")
        except Exception as e:
            print(f"ERROR loading Brain: {e}")
            print("  1) python extract_emotion_stats.py train.parquet models/brain/emotion_stats.json")
            print("  2) Place weights in models/brain/brain.pt (or .onnx)")
            print("  3) Or set BRAIN_ALLOW_STUB = True for plumbing tests only")
            print("  4) pip install torchcrepe torchaudio pandas pyarrow")
            sys.exit(1)
    elif engine == "wav2arkit":
        try:
            wav2arkit.load_session()
            print("[OK] wav2arkit ONNX loaded + warmed (audio → ARKit).")
        except Exception as e:
            print(f"ERROR loading wav2arkit: {e}")
            print("  pip install onnxruntime scipy")
            print("  Model should be in models/wav2arkit_cpu/")
            sys.exit(1)
    else:
        if not Path(RHUBARB_CMD).exists():
            print(f"WARNING: Rhubarb binary '{RHUBARB_CMD}' not found.")
            print("Lip sync will be skipped or may fail.")

    print("\nReady. Paste your JSON below (then blank line).")
    print(">>> No wait — start Blender stream_receiver first if you want live play.\n")

    # Send a heartbeat right before prompting for next JSON so the receiver knows we are still here
    # (prevents silence timeout during the time the user is thinking or typing the next JSON)
    send_viseme(DEFAULT_VISEME)
    send_emotion("neutral", 1.0)

    # Send connection test - use rest_pose to force full neutral rest on the face
    print("  [Startup] Sending rest_pose to force neutral rest on Blender...")
    rest_packet = {
        "type": "rest_pose",
        "blendshapes": emotion_map.NEUTRAL_REST.copy()
    }
    send_udp(rest_packet)
    print("[Orchestrator] Rest pose sent. Check Blender for 'Applying REST_POSE' and face should be at natural rest.")

    sentence_counter = 0

    while True:
        # Heartbeat / rest before asking for next JSON - send rest_pose to keep/return face to natural rest during user pause
        rest_packet = {
            "type": "rest_pose",
            "blendshapes": emotion_map.NEUTRAL_REST.copy()
        }
        send_udp(rest_packet)

        sentences = get_sentences_from_raw_json()

        if sentences is None:
            print("Goodbye!")
            break

        # Process each sentence in order (same pipeline)
        for i, sent in enumerate(sentences, 1):
            sentence_counter += 1
            process_sentence(client, sent, sentence_counter)

        # Small pause between full responses
        time.sleep(0.3)

    # Cleanup
    sd.stop()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()
