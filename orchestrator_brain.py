"""
orchestrator_brain.py

Brain + Parler Mini pipeline (does NOT gut orchestrator.py).

  JSON → Parler TTS → WAV
       → emotion_manager (emotion_26d)
       → Brain (upper face / expression)  +  wav2arkit (mouth lip-sync)   [default hybrid]
       → play_brain_output @ 30fps (viseme lower + emotion upper UDP)
       → blender_receiver.py

Lip modes (MOUTH_ENGINE):
  hybrid   — wav2arkit mouth (good lip-sync) + Brain upper + emotion_map  [DEFAULT]
  brain    — pure Brain mouth (softer / flatter; research mode)
  wav2arkit — wav2arkit mouth only + emotion_map upper (classic look)

Why hybrid: pure Brain lower-head motion is low-variance (jaw std ~0.01 vs
wav2arkit ~0.04) so lips look open but not speech-locked — “Rhubarb-like”.
wav2arkit was trained for continuous audio→ARKit lips; use that for mouth.

Run:
  1. Blender: blender_receiver.py → bpy.ops.face.stream_receiver()
  2. python orchestrator_brain.py
"""

from __future__ import annotations

import os
import re as _re
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

# Reuse shared config / UDP / TTS / JSON input from classic orchestrator
import orchestrator as base
import emotion_map
import brain_inference
import prosody_gpu
import wav2arkit
from emotion_manager import get_emotion_manager
from parler_voice import (
    load_parler,
    build_voice_style,
    generate_speech,
    ensure_vocal_sounds,
    VOCAL_SOUND_TEXTS,
)

# Hybrid is the production path (Brain still runs for upper / emotion_26d)
base.LIPSYNC_ENGINE = "brain"
base.BRAIN_ALLOW_STUB = False

# Speed: default to stats-based emotion_26d (fast, uses emotion_stats.json averages).
# A2E (nvidia/Audio2Emotion-v2.2, 1.27GB ONNX) adds 1-3s per sentence with minimal
# visual gain when using our new emotion_map preset blending (35-80% preset weight).
# Override with: EMOTION_26D_SOURCE=a2e python orchestrator_brain.py
os.environ.setdefault("EMOTION_26D_SOURCE", "stats")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEMP_DIR = base.TEMP_DIR
TEMP_DIR.mkdir(exist_ok=True)

# NVIDIA ARKit order (same as brain_inference)
NVIDIA_ARKIT_ORDER = brain_inference.NVIDIA_ARKIT_ORDER

# Match brain_model ARKIT_LOWER / ARKIT_UPPER (scatter layout) + blender_receiver.
# Lower (viseme): cheekPuff + jaw + full mouth interior + tongue — NOT smile/frown/noseSneer
# Upper (emotion): brows, cheek squints, eyes, noseSneer, mouthSmile/Frown (expression path)
from brain_model import ARKIT_LOWER_INDICES, ARKIT_UPPER_INDICES

LOWER_FACE_KEYS = {NVIDIA_ARKIT_ORDER[i] for i in ARKIT_LOWER_INDICES}
# Mouth keys owned by lip-sync engine (must match wav2arkit.MOUTH_JAW_KEYS spirit)
MOUTH_LIPSYNC_KEYS = set(wav2arkit.MOUTH_JAW_KEYS)

# blender_receiver applies these keys via emotion packets (slow expressive glide).
# mouthSmile/Frown + noseSneer were previously dropped — now included so the
# emotion_map presets and Brain upper predictions fully reach Blender.
BLENDER_UPPER_KEYS = (set(emotion_map.UPPER_FACE_KEYS) & {
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekSquintLeft", "cheekSquintRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthPressLeft", "mouthPressRight",   # sad lip tightening / droop
}) | {
    "noseSneerLeft", "noseSneerRight",  # Brain upper head predicts these; add directly
}
# Full Brain upper scatter set (for logging); playback uses BLENDER_UPPER_KEYS
UPPER_FACE_KEYS = {NVIDIA_ARKIT_ORDER[i] for i in ARKIT_UPPER_INDICES}

# How strongly to pull Brain upper toward emotion_map presets.
# 0.35 = Brain drives 65% (keeps micro-motion), preset anchors 35% (readable emotion).
# Previous 0.55 with max-bias formula was washing out Brain temporal dynamics.
EMOTION_MAP_BLEND = 0.35

# hybrid | brain | wav2arkit  (env MOUTH_ENGINE overrides)
MOUTH_ENGINE = os.environ.get("MOUTH_ENGINE", "hybrid").strip().lower()
if MOUTH_ENGINE not in ("hybrid", "brain", "wav2arkit"):
    MOUTH_ENGINE = "hybrid"

# A/V sync: hold lips BACK so mouth is not ahead of what you hear.
# Windows audio output latency is often 80–150ms; wav2arkit also anticipates
# phonemes slightly. Defaults are positive on purpose.
#   $env:LIP_DELAY_MS="160"   # raise if lips still lead
#   $env:LIP_DELAY_MS="80"    # lower if lips lag behind
# If unset, we auto-pick max(120, device_output_latency_ms).
_LIP_DELAY_ENV = os.environ.get("LIP_DELAY_MS")  # None = auto
LIP_MODEL_LEAD_MS = float(os.environ.get("LIP_MODEL_LEAD_MS", "40"))

# Extra silence appended only at playback (not in lip-frame generation)
PLAY_TAIL_PAD_S = float(os.environ.get("PLAY_TAIL_PAD_S", "0.12"))

# How long after last speech frame to ease mouth shut before full rest (seconds)
TAIL_EASE_S = float(os.environ.get("LIP_TAIL_EASE_S", "0.12"))


def _output_latency_s(sr: int = 44100) -> float:
    """Best-effort WASAPI/host output latency in seconds."""
    try:
        dev = sd.query_devices(kind="output")
        lat = dev.get("default_high_output_latency")
        if lat is None or float(lat) <= 0:
            lat = dev.get("default_low_output_latency")
        if lat is not None and float(lat) > 0:
            return float(lat)
    except Exception:
        pass
    return 0.10  # safe Windows default ~100ms


def _lip_hold_s(sr: int = 44100) -> float:
    """Seconds to hold lips behind audio wall-clock (lips sample earlier content)."""
    device_ms = _output_latency_s(sr) * 1000.0
    if _LIP_DELAY_ENV is not None and str(_LIP_DELAY_ENV).strip() != "":
        base_ms = float(_LIP_DELAY_ENV)
    else:
        # Auto: at least 120ms, or device high latency if larger
        base_ms = max(120.0, device_ms)
    total_ms = base_ms + float(LIP_MODEL_LEAD_MS)
    # Clamp to a useful range
    total_ms = max(40.0, min(280.0, total_ms))
    return total_ms / 1000.0


# ---------------------------------------------------------------------------
# VOCAL REACTIONS  — triggered by [token] markers in sentence text
# ---------------------------------------------------------------------------
# Blendshape burst shown when the tagged word is spoken.
# "emotion" sets the UDP emotion field; "blendshapes" are absolute overrides
# for UPPER_FACE keys only (lower/mouth still follows wav2arkit lip-sync).
VOCAL_REACTIONS: dict[str, dict] = {
    "eww": {
        "emotion": "disgusted",
        "duration_s": 0.70,
        "blendshapes": {
            "noseSneerLeft": 1.0, "noseSneerRight": 1.0,
            "eyeSquintLeft": 0.80, "eyeSquintRight": 0.80,
            "cheekSquintLeft": 0.60, "cheekSquintRight": 0.60,
            "mouthFrownLeft": 0.70, "mouthFrownRight": 0.70,
            "browDownLeft": 0.60, "browDownRight": 0.60,
        },
    },
    "ew": {  # alias
        "emotion": "disgusted",
        "duration_s": 0.50,
        "blendshapes": {
            "noseSneerLeft": 0.90, "noseSneerRight": 0.90,
            "eyeSquintLeft": 0.65, "eyeSquintRight": 0.65,
            "mouthFrownLeft": 0.50, "mouthFrownRight": 0.50,
        },
    },
    "laugh": {
        "emotion": "happy",
        "duration_s": 1.00,
        "blendshapes": {
            "mouthSmileLeft": 1.0, "mouthSmileRight": 1.0,
            "cheekSquintLeft": 1.0, "cheekSquintRight": 1.0,
            "eyeSquintLeft": 0.70, "eyeSquintRight": 0.70,
            "browOuterUpLeft": 0.30, "browOuterUpRight": 0.30,
        },
    },
    "chuckle": {
        "emotion": "happy",
        "duration_s": 0.60,
        "blendshapes": {
            "mouthSmileLeft": 0.80, "mouthSmileRight": 0.80,
            "cheekSquintLeft": 0.70, "cheekSquintRight": 0.70,
            "eyeSquintLeft": 0.50, "eyeSquintRight": 0.50,
        },
    },
    "sigh": {
        "emotion": "sad",
        "duration_s": 0.80,
        "blendshapes": {
            "browInnerUp": 0.70,
            "eyeBlinkLeft": 0.30, "eyeBlinkRight": 0.30,
            "mouthFrownLeft": 0.50, "mouthFrownRight": 0.50,
        },
    },
    "gasp": {
        "emotion": "surprised",
        "duration_s": 0.50,
        "blendshapes": {
            "eyeWideLeft": 1.0, "eyeWideRight": 1.0,
            "browInnerUp": 0.90,
            "browOuterUpLeft": 0.70, "browOuterUpRight": 0.70,
        },
    },
    "hmm": {
        "emotion": "thinking",
        "duration_s": 0.60,
        "blendshapes": {
            "browDownLeft": 0.35, "browInnerUp": 0.25,
            "eyeSquintLeft": 0.30, "eyeSquintRight": 0.20,
        },
    },
    "ugh": {
        "emotion": "disgusted",
        "duration_s": 0.60,
        "blendshapes": {
            "noseSneerLeft": 0.70, "noseSneerRight": 0.70,
            "browDownLeft": 0.55, "browDownRight": 0.55,
            "eyeSquintLeft": 0.55, "eyeSquintRight": 0.55,
        },
    },
    "wow": {
        "emotion": "surprised",
        "duration_s": 0.70,
        "blendshapes": {
            "eyeWideLeft": 0.90, "eyeWideRight": 0.90,
            "browInnerUp": 0.80, "browOuterUpLeft": 0.60, "browOuterUpRight": 0.60,
        },
    },
    # -----------------------------------------------------------------------
    # Additional tokens (sourced from ElevenLabs, Suno AI, Play.ai and
    # voice acting / animation script conventions — Jul 2026)
    # -----------------------------------------------------------------------
    "cry": {
        "emotion": "sad",
        "duration_s": 1.20,
        "blendshapes": {
            "browInnerUp": 0.90,
            "eyeBlinkLeft": 0.60, "eyeBlinkRight": 0.60,
            "eyeSquintLeft": 0.50, "eyeSquintRight": 0.50,
            "cheekSquintLeft": 0.40, "cheekSquintRight": 0.40,
            "mouthFrownLeft": 0.80, "mouthFrownRight": 0.80,
            "mouthPressLeft": 0.40, "mouthPressRight": 0.40,
        },
    },
    "crying": {  # alias
        "emotion": "sad",
        "duration_s": 1.20,
        "blendshapes": {
            "browInnerUp": 0.90,
            "eyeBlinkLeft": 0.60, "eyeBlinkRight": 0.60,
            "eyeSquintLeft": 0.50, "eyeSquintRight": 0.50,
            "mouthFrownLeft": 0.80, "mouthFrownRight": 0.80,
            "mouthPressLeft": 0.40, "mouthPressRight": 0.40,
        },
    },
    "sob": {
        "emotion": "sad",
        "duration_s": 1.50,
        "blendshapes": {
            "browInnerUp": 1.00,
            "browDownLeft": 0.20, "browDownRight": 0.20,
            "eyeBlinkLeft": 0.70, "eyeBlinkRight": 0.70,
            "eyeSquintLeft": 0.60, "eyeSquintRight": 0.60,
            "cheekSquintLeft": 0.60, "cheekSquintRight": 0.60,
            "mouthFrownLeft": 0.95, "mouthFrownRight": 0.95,
            "mouthPressLeft": 0.50, "mouthPressRight": 0.50,
        },
    },
    "scoff": {
        "emotion": "disgusted",
        "duration_s": 0.60,
        "blendshapes": {
            "browDownLeft": 0.35, "browDownRight": 0.20,
            "browOuterUpLeft": 0.25,
            "eyeSquintLeft": 0.45, "eyeSquintRight": 0.30,
            "noseSneerLeft": 0.55, "noseSneerRight": 0.35,
            "mouthSmileLeft": 0.25, "mouthSmileRight": 0.10,  # one-sided dismissive curl
        },
    },
    "groan": {
        "emotion": "disgusted",
        "duration_s": 0.80,
        "blendshapes": {
            "browDownLeft": 0.65, "browDownRight": 0.65,
            "eyeSquintLeft": 0.55, "eyeSquintRight": 0.55,
            "mouthFrownLeft": 0.60, "mouthFrownRight": 0.60,
            "noseSneerLeft": 0.35, "noseSneerRight": 0.35,
        },
    },
    "yawn": {
        "emotion": "neutral",
        "duration_s": 1.50,
        "blendshapes": {
            "eyeBlinkLeft": 0.70, "eyeBlinkRight": 0.70,
            "eyeSquintLeft": 0.40, "eyeSquintRight": 0.40,
            "browInnerUp": 0.30,
            "browOuterUpLeft": 0.20, "browOuterUpRight": 0.20,
            "cheekSquintLeft": 0.30, "cheekSquintRight": 0.30,
        },
    },
    "wince": {
        "emotion": "fearful",
        "duration_s": 0.50,
        "blendshapes": {
            "eyeBlinkLeft": 0.80, "eyeBlinkRight": 0.80,
            "eyeSquintLeft": 0.90, "eyeSquintRight": 0.90,
            "cheekSquintLeft": 0.70, "cheekSquintRight": 0.70,
            "noseSneerLeft": 0.50, "noseSneerRight": 0.50,
            "browDownLeft": 0.40, "browDownRight": 0.40,
        },
    },
    "smirk": {
        "emotion": "sarcastic",
        "duration_s": 0.80,
        "blendshapes": {
            "mouthSmileLeft": 0.60, "mouthSmileRight": 0.15,  # asymmetric
            "cheekSquintLeft": 0.40, "cheekSquintRight": 0.10,
            "browDownLeft": 0.20, "browDownRight": 0.15,
            "eyeSquintLeft": 0.30, "eyeSquintRight": 0.15,
        },
    },
    "grunt": {
        "emotion": "angry",
        "duration_s": 0.50,
        "blendshapes": {
            "browDownLeft": 0.85, "browDownRight": 0.85,
            "noseSneerLeft": 0.65, "noseSneerRight": 0.65,
            "eyeSquintLeft": 0.60, "eyeSquintRight": 0.60,
            "cheekSquintLeft": 0.30, "cheekSquintRight": 0.30,
        },
    },
    "sniff": {
        "emotion": "sad",
        "duration_s": 0.60,
        "blendshapes": {
            "browInnerUp": 0.50,
            "eyeSquintLeft": 0.30, "eyeSquintRight": 0.30,
            "noseSneerLeft": 0.30, "noseSneerRight": 0.25,
            "mouthFrownLeft": 0.35, "mouthFrownRight": 0.35,
        },
    },
    "nervous": {
        "emotion": "fearful",
        "duration_s": 0.90,
        "blendshapes": {
            "mouthSmileLeft": 0.35, "mouthSmileRight": 0.35,  # nervous smile
            "eyeSquintLeft": 0.25, "eyeSquintRight": 0.25,
            "browInnerUp": 0.40,
            "eyeWideLeft": 0.30, "eyeWideRight": 0.30,
        },
    },
    "sneeze": {
        "emotion": "surprised",
        "duration_s": 0.50,
        "blendshapes": {
            "eyeBlinkLeft": 0.90, "eyeBlinkRight": 0.90,
            "noseSneerLeft": 0.80, "noseSneerRight": 0.80,
            "cheekSquintLeft": 0.70, "cheekSquintRight": 0.70,
            "browDownLeft": 0.40, "browDownRight": 0.40,
        },
    },
    "cough": {
        "emotion": "neutral",
        "duration_s": 0.50,
        "blendshapes": {
            "eyeSquintLeft": 0.35, "eyeSquintRight": 0.35,
            "browDownLeft": 0.25, "browDownRight": 0.25,
            "cheekSquintLeft": 0.30, "cheekSquintRight": 0.30,
        },
    },
}


def parse_vocal_markers(text: str):
    """
    Extract [token] markers from text.
    Keeps the token word in the TTS text so Parler speaks it naturally.
    Returns (clean_text, markers) where markers = list of
      {"char_pos": int, "token": str, "reaction": dict}.
    """
    markers = []
    parts = []
    cursor = 0
    for m in _re.finditer(r'\[([^\]]+)\]', text):
        parts.append(text[cursor:m.start()])
        token = m.group(1).lower().strip()
        char_pos = sum(len(p) for p in parts)
        # Do NOT add the word to TTS text — the vocalization sound is generated
        # separately and spliced into the audio. Add a space to keep word boundaries.
        parts.append(" ")
        reaction = VOCAL_REACTIONS.get(token)
        if reaction:
            markers.append({"char_pos": char_pos, "token": token, "reaction": reaction})
        cursor = m.end()
    parts.append(text[cursor:])
    clean_text = "".join(parts)
    return clean_text, markers


def _splice_vocal_sounds(
    audio: np.ndarray,
    sr: int,
    vocal_markers: list[dict],
    duration: float,
) -> np.ndarray:
    """
    Mix pre-generated vocalization clips into the main audio at marker timestamps.
    The token word is NOT in the TTS text; the clip IS the sound.
    Uses additive mix (capped to [-1,1]) so speech and sound overlap naturally.
    """
    if not vocal_markers:
        return audio

    # Generate only the tokens this sentence needs (not all 20+ at startup)
    needed = list({vm["token"] for vm in vocal_markers})
    vocal_cache = ensure_vocal_sounds(
        temp_dir=str(TEMP_DIR),
        generate_missing=True,
        only_tokens=needed,
    )
    result = audio.astype(np.float32).copy()

    for vm in vocal_markers:
        token = vm["token"]
        if token not in vocal_cache:
            print(f"  [vocal_splice] No clip for [{token}] — skip sound (face reaction still runs)")
            continue
        clip_path, clip_sr = vocal_cache[token]
        try:
            clip, _ = sf.read(clip_path, dtype="float32")
        except Exception as e:
            print(f"  [vocal_splice] Cannot read {clip_path}: {e}")
            continue
        if clip.ndim > 1:
            clip = clip.mean(axis=1)

        # Trim leading silence from clip (Parler often adds ~200ms silence at start)
        _thresh = float(np.abs(clip).max()) * 0.02
        _nonzero = np.where(np.abs(clip) > _thresh)[0]
        if len(_nonzero):
            clip = clip[_nonzero[0]:]

        # Compute sample position from char ratio
        tc = max(1, int(vm.get("total_chars", 1)))
        t_marker = (vm["char_pos"] / tc) * duration
        sample_pos = int(t_marker * sr)

        # Pad result if clip extends beyond current audio end
        needed = sample_pos + len(clip)
        if needed > len(result):
            result = np.pad(result, (0, needed - len(result)), mode="constant")

        # Mix at 85% to blend with speech
        end_pos = sample_pos + len(clip)
        result[sample_pos:end_pos] += clip * 0.85
        result = np.clip(result, -1.0, 1.0)
        print(f"  [vocal_splice] [{token}] @ {t_marker:.2f}s  clip={len(clip)/sr:.2f}s")

    return result


def send_rest_pose(smooth: bool = True) -> None:
    base.send_udp({
        "type": "rest_pose",
        "smooth": smooth,
        "blendshapes": emotion_map.NEUTRAL_REST.copy(),
    })


def _blend_upper(brain_upper: dict, emotion: str, intensity: float, dynamic_inten: float = 1.0) -> dict:
    """
    Merge Brain upper-face channels with emotion_map presets.

    Pure weighted blend (no max-bias): Brain drives temporal micro-motion,
    preset anchors the readable emotion shape. dynamic_inten scales the preset
    contribution per-frame (energy envelope from speech modulates expression).
    """
    preset = emotion_map.get_blendshapes(emotion, intensity * dynamic_inten)
    out = {}
    a = float(EMOTION_MAP_BLEND)
    for k in BLENDER_UPPER_KEYS:
        b = float(brain_upper.get(k, 0.0))
        p = float(preset.get(k, 0.0))
        if "Smile" in k or "Frown" in k or "Sneer" in k or "Press" in k:
            # Brain lower head bypassed in hybrid; preset must dominate fully.
            a_k = 0.90
        elif p >= 0.70:
            # High-keyed preset (angry browDown=0.95, sad browInnerUp=0.80,
            # surprised eyeWide=0.85). Must reach target — give preset 0.80 weight.
            # Without this, Brain's low-variance output (typ 0.1-0.3) dilutes to ~0.45.
            a_k = 0.80
        elif p >= 0.40:
            # Medium-keyed preset (eyeSquint, cheekSquint for disgust/happy).
            a_k = 0.60
        else:
            # Subtle/neutral: Brain micro-motion drives naturally.
            a_k = a  # 0.35
        out[k] = b * (1.0 - a_k) + p * a_k
    return out


def _merge_mouth_upper(
    mouth_frames: list[dict],
    upper_frames: list[dict] | None,
    fps: float = 30.0,
) -> list[dict]:
    """
    Build per-frame dicts: mouth keys from mouth_frames, upper from upper_frames.
    Length follows mouth (lip-sync master clock). Upper is time-sampled if lengths differ.
    """
    if not mouth_frames:
        return list(upper_frames or [])

    n = len(mouth_frames)
    out: list[dict] = []
    for i in range(n):
        merged = {k: float(v) for k, v in mouth_frames[i].items() if k in MOUTH_LIPSYNC_KEYS}
        # Prefer dedicated mouth engine values; never carry chronic Brain mouthClose
        if "mouthClose" in merged:
            # wav2arkit already suppresses; keep its value (small). Zero if huge.
            if merged["mouthClose"] > 0.2:
                merged["mouthClose"] = 0.0

        if upper_frames:
            t = i / fps
            u = wav2arkit.frame_at_time(upper_frames, t, fps=fps)
            for k, v in u.items():
                if k in BLENDER_UPPER_KEYS or k in UPPER_FACE_KEYS:
                    if k not in MOUTH_LIPSYNC_KEYS:
                        merged[k] = float(v)
        out.append(merged)
    return out


def _fit_frames_to_audio_duration(
    frames: list[dict],
    audio_duration: float,
    fps: float,
) -> list[dict]:
    """
    Resample lip frames so timeline length == audio length.

    Fixes: lips freeze while trailing audio still plays (model returned fewer
    frames than audio_duration * fps). Time-warps the sequence to span the
    full WAV, then pads a short ease-to-closed tail if needed.
    """
    if not frames or audio_duration <= 0:
        return list(frames or [])

    target_n = max(1, int(round(audio_duration * fps)))
    src_n = len(frames)
    src_dur = max(src_n / fps, 1.0 / fps)

    # If already within ~1 frame of audio length, keep as-is
    if abs(src_n - target_n) <= 1:
        out = list(frames)
    else:
        out = []
        for i in range(target_n):
            # Wall time along full audio; sample source proportionally
            t_audio = i / fps
            t_src = t_audio * (src_dur / audio_duration)
            # Clamp into source span
            t_src = min(max(0.0, t_src), max(0.0, (src_n - 1) / fps))
            out.append(wav2arkit.frame_at_time(frames, t_src, fps=fps))

    # Do NOT ease mouth closed during the last part of the clip here.
    # That made lips look "finished" while speech was still playing.
    # Closing is handled only after full audio ends (settle in play_brain_output).
    return out


def _split_viseme_emotion(
    frame_dict: dict,
    emotion: str,
    intensity: float,
    dynamic_inten: float = 1.0,
) -> tuple[dict, dict]:
    """Build viseme (mouth) + emotion (upper) packets from one frame dict."""
    lower_keys = {
        k: float(v)
        for k, v in frame_dict.items()
        if k in MOUTH_LIPSYNC_KEYS or k in LOWER_FACE_KEYS
    }
    if lower_keys.get("mouthClose", 0.0) > 0.15:
        lower_keys["mouthClose"] = min(lower_keys["mouthClose"], 0.05)
    # Expression keys are sent via emotion packet only — remove from viseme.
    # Prevents wav2arkit's near-zero mouthSmile from fighting the emotion preset.
    for k in list(lower_keys.keys()):
        if k in BLENDER_UPPER_KEYS:
            lower_keys.pop(k, None)

    brain_upper = {k: float(v) for k, v in frame_dict.items() if k in BLENDER_UPPER_KEYS}
    upper_keys = _blend_upper(brain_upper, emotion, intensity, dynamic_inten=dynamic_inten)
    return lower_keys, upper_keys


def play_brain_output(
    wav_path: str,
    frames: list[dict],
    device=None,
    emotion: str | None = None,
    intensity: float | None = None,
    fps: float = 30.0,
    vocal_markers: list | None = None,
    audio_data: np.ndarray | None = None,
    sample_rate: int | None = None,
) -> None:
    """
    Play WAV and stream frames at ~30 fps with *time-based* sampling.

    Sync rules:
      • Fit frames to full audio duration (no frozen mouth while audio continues)
      • LIP_DELAY_MS holds lips back so they match heard audio (not ahead)
      • Keep sending until audio device finishes; rest only after playback ends
    """
    # Prefer in-memory audio from TTS (exact buffer that was saved) over re-read
    if audio_data is not None and sample_rate is not None:
        audio = np.asarray(audio_data, dtype=np.float32)
        sr = int(sample_rate)
    else:
        audio, sr = sf.read(wav_path, dtype="float32")
        sr = int(sr)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    duration = float(len(audio) / float(sr))
    emo = emotion if emotion is not None else base.current_emotion
    inten = float(intensity if intensity is not None else base.current_intensity)

    if not frames:
        print("  [play_brain] No frames — playing audio only")
        sd.play(audio, sr, blocking=True)
        send_rest_pose(smooth=True)
        return

    # Stretch/pad lips to cover entire WAV timeline
    n_before = len(frames)
    frames = _fit_frames_to_audio_duration(frames, duration, fps)
    n_frames = len(frames)
    frame_span = n_frames / float(fps)

    # Hold lips behind heard audio (fixes "mouth moves before voice")
    lip_hold = _lip_hold_s(sr)

    frame_interval = 1.0 / float(fps)
    print(
        f"  [play_brain] audio={duration:.2f}s frames={n_before}→{n_frames} "
        f"span={frame_span:.2f}s @ {fps:.0f}fps "
        f"lip_hold={lip_hold*1000:.0f}ms "
        f"(auto_device≈{_output_latency_s(sr)*1000:.0f}ms + model={LIP_MODEL_LEAD_MS:.0f}ms) "
        f"mouth_engine={MOUTH_ENGINE} emotion={emo}"
    )

    # Pre-compute per-frame energy envelope at 30fps for dynamic intensity modulation.
    # Maps speech energy peaks to stronger expressions, silences to softer ones.
    # This gives the face a "pulsing" aliveness tied to the stress pattern of speech.
    _hop = max(1, int(round(float(sr) / fps)))
    _n_env = n_frames
    energy_envelope = np.array([
        float(np.sqrt(np.mean(np.square(audio[i * _hop: (i + 1) * _hop])) + 1e-9))
        for i in range(_n_env)
    ], dtype=np.float32)
    _e_max = float(energy_envelope.max())
    if _e_max > 1e-6:
        energy_envelope /= _e_max
    # Smooth the envelope to avoid jitter (3-frame running avg — shorter = more dynamic)
    _kernel = np.ones(3, dtype=np.float32) / 3.0
    energy_envelope = np.convolve(energy_envelope, _kernel, mode="same")

    # Closed mouth packet (never leak first speech frame during hold-back)
    closed_mouth = {k: 0.0 for k in MOUTH_LIPSYNC_KEYS}
    closed_upper = _blend_upper({}, emo, inten * 0.35)

    # Track last *actually sent* packets so end settle continues from real pose
    # (not a rebuilt full emotion burst).
    last_lower_sent: dict = {}
    last_upper_sent: dict = {}

    def _send_pair(
        frame_dict: dict, e: str, inten_f: float,
        dynamic_inten: float = 1.0, vocal_burst: dict | None = None,
    ) -> None:
        nonlocal last_lower_sent, last_upper_sent
        lower_keys, upper_keys = _split_viseme_emotion(frame_dict, e, inten_f, dynamic_inten=dynamic_inten)
        if lower_keys:
            base.send_udp({"type": "viseme", "blendshapes": lower_keys})
            last_lower_sent = dict(lower_keys)
        # Vocal burst: override upper-face with reaction blendshapes
        if vocal_burst:
            burst_emo = vocal_burst.get("emotion", e)
            burst_upper = {k: float(v) for k, v in vocal_burst["blendshapes"].items()
                           if k in BLENDER_UPPER_KEYS}
            base.send_udp({"type": "emotion", "emotion": burst_emo, "blendshapes": burst_upper})
            last_upper_sent = dict(burst_upper)
        elif upper_keys:
            base.send_udp({"type": "emotion", "emotion": e, "blendshapes": upper_keys})
            last_upper_sent = dict(upper_keys)

    # Soft rest targets before speech (smooth — no mid-sentence snap)
    send_rest_pose(smooth=True)
    _send_pair(closed_mouth, emo, inten * 0.2)

    # Convert character-position markers → audio timestamps
    _vmarkers: list[dict] = []
    if vocal_markers:
        for _vm in vocal_markers:
            _tc = max(1, int(_vm.get("total_chars", 1)))
            _t = (_vm["char_pos"] / _tc) * duration
            _vmarkers.append({
                "t": _t,
                "end": _t + _vm["reaction"]["duration_s"],
                "reaction": _vm["reaction"],
            })

    # --- A/V with lip lag (mouth must not lead the voice) ---
    # current_t = time since sd.play started (audio buffer timeline).
    # You HEAR sample current_t only after device latency L.
    # So we sample lips at lip_t = current_t - lip_hold (lip_hold ≈ L + model lead).
    # Until lip_t >= 0 the mouth stays closed.
    speech_dur = duration
    pad_s = max(0.0, PLAY_TAIL_PAD_S)
    if pad_s > 0:
        audio_play = np.concatenate([
            audio,
            np.zeros(int(round(pad_s * sr)), dtype=np.float32),
        ])
    else:
        audio_play = audio
    play_dur = float(len(audio_play) / float(sr))

    sd.play(np.ascontiguousarray(audio_play, dtype=np.float32), int(sr), blocking=False)
    playback_start = time.perf_counter()
    last_frame_time = playback_start
    tick = 0
    last_sent: dict = dict(closed_mouth)
    max_lip_t = max(0.0, (n_frames - 1) / float(fps))

    print(
        f"  [play_brain] play speech={speech_dur:.2f}s pad={pad_s:.2f}s "
        f"lip_hold={lip_hold*1000:.0f}ms (lips lag audio to fix lead)"
    )

    while True:
        now = time.perf_counter()
        current_t = now - playback_start
        # Keep looping through speech + pad so last syllable is not dropped
        if current_t >= play_dur + lip_hold * 0.25:
            break

        # Trailing pad after speech samples finished
        if current_t >= speech_dur:
            frame_dict = dict(last_sent) if last_sent else dict(closed_mouth)
            for k in list(frame_dict.keys()):
                frame_dict[k] = float(frame_dict[k]) * 0.85
            _dyn = 0.45
            _active_vocal = None
        else:
            # CRITICAL: sample EARLIER frames so mouth is not ahead of heard audio
            lip_t = current_t - lip_hold
            if lip_t < 0.0:
                frame_dict = dict(closed_mouth)
                _dyn = 0.30
                _active_vocal = None
            else:
                lip_t_clamped = min(max(0.0, lip_t), max_lip_t)
                frame_dict = wav2arkit.frame_at_time(frames, lip_t_clamped, fps=fps)
                last_sent = frame_dict
                _env_idx = min(int(lip_t_clamped * fps), len(energy_envelope) - 1)
                _env_idx = max(0, _env_idx)
                _dyn = 0.50 + 0.70 * float(energy_envelope[_env_idx])
                _active_vocal = None
                for _vm in _vmarkers:
                    if _vm["t"] <= lip_t_clamped <= _vm["end"]:
                        _active_vocal = _vm["reaction"]
                        break

        _send_pair(frame_dict, emo, inten, dynamic_inten=_dyn, vocal_burst=_active_vocal)

        if tick % 15 == 0:
            _lt = current_t - lip_hold
            print(
                f"  [play_brain] audio_t={current_t:.2f}s "
                f"lip_t={max(0.0, _lt):.2f}s "
                f"jawOpen={frame_dict.get('jawOpen', 0):.3f}"
            )
        tick += 1

        sleep_time = frame_interval - (time.perf_counter() - last_frame_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_frame_time = time.perf_counter()

    sd.wait()

    # ---- One smooth settle: last sent pose → neutral rest (no emotion re-burst) ----
    # Old bug: after audio we re-sent full emotion_map for ~0.2s, then neutral,
    # then rest_pose — face went normal → emotion flash → neutral again.
    rest = emotion_map.NEUTRAL_REST
    start_lower = dict(last_lower_sent) if last_lower_sent else {
        k: float(last_sent.get(k, 0.0)) for k in MOUTH_LIPSYNC_KEYS if k in last_sent
    }
    start_upper = dict(last_upper_sent) if last_upper_sent else {
        k: float(last_sent.get(k, 0.0)) for k in BLENDER_UPPER_KEYS if k in last_sent
    }
    # Include any rest keys so we glide fully to rest pose
    all_lower_keys = set(start_lower) | {k for k in rest if k in MOUTH_LIPSYNC_KEYS or k in LOWER_FACE_KEYS}
    all_upper_keys = set(start_upper) | (set(BLENDER_UPPER_KEYS) & set(rest.keys()))

    settle_steps = 18  # ~0.6s at 33ms
    for step in range(1, settle_steps + 1):
        frac = step / float(settle_steps)
        # smoothstep ease-out
        ease = frac * frac * (3.0 - 2.0 * frac)
        lower = {}
        for k in all_lower_keys:
            a = float(start_lower.get(k, 0.0))
            b = float(rest.get(k, 0.0))
            lower[k] = a * (1.0 - ease) + b * ease
        upper = {}
        for k in all_upper_keys:
            a = float(start_upper.get(k, 0.0))
            b = float(rest.get(k, 0.0))
            upper[k] = a * (1.0 - ease) + b * ease
        if lower:
            base.send_udp({"type": "viseme", "blendshapes": lower})
        if upper:
            base.send_udp({
                "type": "emotion",
                "emotion": "neutral",
                "blendshapes": upper,
            })
        time.sleep(0.033)

    # Final rest target once (already near rest — no flash)
    send_rest_pose(smooth=True)
    print(f"  [play_brain] Done (full audio {duration:.2f}s, smooth settle).")


def process_sentence_brain(sentence: dict, idx: int) -> None:
    """
    Full sentence:
      1) Parler TTS
      2) Mouth: wav2arkit (hybrid/default) and/or Brain
      3) Upper: Brain + emotion_map blend
      4) play_brain_output (dual UDP + audio)
    """
    text = sentence["text"]
    emotion = sentence["emotion"]
    intensity = float(sentence["intensity"])

    # Parse [token] vocal markers from text before TTS.
    # Markers are removed from brackets but the word stays, so Parler speaks it.
    tts_text, vocal_markers = parse_vocal_markers(text)
    total_chars = max(1, len(tts_text))
    for _vm in vocal_markers:
        _vm["total_chars"] = total_chars
    if vocal_markers:
        print(f"  [vocal] {len(vocal_markers)} reaction(s): "
              f"{[m['token'] for m in vocal_markers]}")

    print(f"\n{'=' * 50}")
    print(f"[Brain Sentence {idx}] {tts_text}")
    print(
        f"  emotion={emotion} intensity={intensity:.2f} device={DEVICE} "
        f"mouth_engine={MOUTH_ENGINE}"
    )
    print(f"{'=' * 50}")

    base.current_emotion = emotion
    base.current_intensity = intensity

    # 1) Parler Mini → WAV  (use cleaned text — brackets stripped, word kept)
    wav_path = str(TEMP_DIR / f"brain_sentence_{idx}.wav")
    style = build_voice_style(emotion, intensity)
    print(f"  [TTS/Parler] style: {style[:100]}...")
    audio, sr = generate_speech(
        text=tts_text,
        voice_style=style,
        output_path=wav_path,
        play_audio=False,
    )
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration = len(audio) / float(sr)
    print(f"  [TTS] Saved {wav_path} ({duration:.2f}s @ {sr} Hz) samples={len(audio)}")

    # 1b) Splice pre-generated vocalization clips into audio at marker timestamps.
    # Token words are NOT in tts_text — the clip IS the sound (eww, heh heh, etc.).
    if vocal_markers:
        audio = _splice_vocal_sounds(audio, sr, vocal_markers, duration)
        duration = len(audio) / float(sr)
        sf.write(wav_path, audio, sr, subtype="PCM_16")
        print(f"  [vocal_splice] Audio after splice: {duration:.2f}s samples={len(audio)}")

    # 2) emotion_26d from NVIDIA Audio2Emotion-v2.2 on this WAV
    #    (falls back to emotion_stats averages only if A2E model missing)
    _e26_src = os.environ.get("EMOTION_26D_SOURCE", "stats")
    print(f"  [emotion_26d] source={_e26_src} emotion={emotion} intensity={intensity:.2f}")
    e26 = brain_inference.get_emotion_vector(
        emotion_label=emotion,
        intensity=intensity,
        device=str(DEVICE),
        wav_path=wav_path,
    )
    print(
        f"  [emotion_26d] shape={tuple(e26.shape)} L2={e26.norm():.4f} "
        f"mean={e26.mean():.4f} explicit[16:26]={e26[16:].detach().cpu().numpy().round(3).tolist()}"
    )
    if float(e26.abs().sum()) < 1e-8:
        raise RuntimeError(
            "emotion_26d is all zeros — install Audio2Emotion "
            "(python audio2emotion.py --download) or check emotion_stats.json fallback"
        )

    brain_frames: list[dict] = []
    brain_raw = None
    w2a_frames: list[dict] = []
    fps = 30.0

    need_brain = MOUTH_ENGINE in ("hybrid", "brain")
    need_w2a = MOUTH_ENGINE in ("hybrid", "wav2arkit")

    # 3a) Brain (upper / full face) — uses same A2E 26d inside run_brain
    if need_brain:
        print("  [Brain] run_brain (HuBERT + prosody + A2E 26d + weights)...")
        brain_frames, fps, brain_raw = brain_inference.run_brain(
            wav_path=wav_path,
            emotion_label=emotion,
            intensity=intensity,
            device=str(DEVICE),
            mouth_only=False,
        )
        print(
            f"  [Brain] frames={len(brain_frames)} fps={fps} raw={brain_raw.shape} "
            f"jawOpen_peak={brain_raw[:, 24].max():.3f} "
            f"jawOpen_std={brain_raw[:, 24].std():.4f} "
            f"browInnerUp_peak={brain_raw[:, 2].max():.3f}"
        )

    # 3b) wav2arkit mouth — continuous audio→ARKit lip-sync (what looked good before)
    if need_w2a:
        print("  [LipSync] wav2arkit mouth (enhanced)...")
        w2a_frames, w2a_fps, w2a_raw = wav2arkit.audio_file_to_frames(
            wav_path,
            mouth_only=True,
            enhance_mouth=True,
        )
        fps = float(w2a_fps) or fps
        print(
            f"  [LipSync] wav2arkit frames={len(w2a_frames)} "
            f"jawOpen_peak={w2a_raw[:, 24].max():.3f} "
            f"jawOpen_std={w2a_raw[:, 24].std():.4f}"
        )

    # 3c) Merge by mode
    if MOUTH_ENGINE == "hybrid":
        frames = _merge_mouth_upper(w2a_frames, brain_frames, fps=fps)
        print(
            f"  [Merge] hybrid: mouth=wav2arkit ({len(w2a_frames)} fr) "
            f"+ upper=Brain ({len(brain_frames)} fr) → {len(frames)} fr"
        )
    elif MOUTH_ENGINE == "wav2arkit":
        frames = list(w2a_frames)
        print(f"  [Merge] wav2arkit-only mouth frames={len(frames)}")
    else:
        frames = list(brain_frames)
        print(
            f"  [Merge] pure Brain mouth (low dynamics — set MOUTH_ENGINE=hybrid for good sync)"
        )

    # 4) Play + dual UDP  (use same in-memory audio buffer as TTS — full length)
    play_brain_output(
        wav_path,
        frames,
        device=DEVICE,
        emotion=emotion,
        intensity=intensity,
        fps=fps,
        vocal_markers=vocal_markers if vocal_markers else None,
        audio_data=audio,
        sample_rate=int(sr),
    )


def startup() -> None:
    print("=" * 60)
    print("ORCHESTRATOR_BRAIN — Parler Mini + Brain upper + lip-sync")
    print(f"  MOUTH_ENGINE={MOUTH_ENGINE}  "
          f"(hybrid=wav2arkit lips+Brain face | brain | wav2arkit)")
    print("=" * 60)

    if DEVICE.type == "cuda":
        name = torch.cuda.get_device_name(0)
        print(f"Using device: {DEVICE} ({name})")
        try:
            free, total = torch.cuda.mem_get_info()
            print(f"  VRAM free/total: {free/1e9:.2f} / {total/1e9:.2f} GB")
        except Exception:
            pass
    else:
        print("WARNING: No GPU found, using CPU")
        print("Expect slower inference (HuBERT + crepe + Parler)")

    print(">>> Blender first: blender_receiver.py → bpy.ops.face.stream_receiver()")
    print("=" * 60)

    # Audio2Emotion-v2.2 (live emotion_26d from audio — primary)
    try:
        import audio2emotion as a2e

        if not a2e.is_available():
            print("Audio2Emotion model not on disk yet — attempting download (~1.27 GB)...")
            print("  Need: huggingface-cli login + accept license on")
            print("  https://huggingface.co/nvidia/Audio2Emotion-v2.2")
            try:
                a2e.ensure_model()
            except Exception as e:
                print(f"  [WARN] A2E download failed: {e}")
                print("  Falling back to emotion_stats.json averages until model is installed.")
        if a2e.is_available():
            a2e.load_session()
            print("Audio2Emotion-v2.2 ready → emotion_26d from audio")
        else:
            print("Audio2Emotion NOT ready — using emotion_stats averages")
            emo_mgr = get_emotion_manager()
            print(f"  stats labels: {sorted(emo_mgr.stats.keys())}")
    except Exception as e:
        print(f"[WARN] audio2emotion import/load: {e}")
        emo_mgr = get_emotion_manager()
        print(f"  Fallback stats labels: {sorted(emo_mgr.stats.keys())}")

    # Brain three modules (upper face / full model)
    if MOUTH_ENGINE in ("hybrid", "brain"):
        brain_inference.load_brain_model(str(DEVICE))
        print("Brain model loaded (shared_encoder + face_head + character_adapter)")
        if DEVICE.type == "cuda":
            try:
                used = torch.cuda.memory_allocated() / 1e9
                print(f"  VRAM used after Brain load: {used:.2f} GB")
            except Exception:
                pass

    # wav2arkit mouth (ONNX CPU) — the continuous lip-sync you liked
    if MOUTH_ENGINE in ("hybrid", "wav2arkit"):
        wav2arkit.load_session()
        print("wav2arkit lip-sync ready (mouth engine)")

    # Parler Mini
    load_parler()
    print("Parler-TTS Mini loaded")
    if DEVICE.type == "cuda":
        try:
            used = torch.cuda.memory_allocated() / 1e9
            free, total = torch.cuda.mem_get_info()
            print(f"  VRAM after Parler: used={used:.2f} GB  free={free/1e9:.2f}/{total/1e9:.2f} GB")
            if free < 0.6e9:
                print("  [WARN] Very little free VRAM — keep sentences short; avoid loading extra models.")
        except Exception:
            pass

    # Only INDEX existing vocal clips (fast). Missing clips generate on first use of [token].
    # Full pre-gen of 20+ Parler clips on a 4GB laptop can take 10–30+ minutes.
    print("Indexing cached vocal clips (generate on demand for [eww]/[laugh]/…)...")
    try:
        cache = ensure_vocal_sounds(temp_dir=str(TEMP_DIR), generate_missing=False)
        print(f"  Cached vocal clips on disk: {len(cache)} / {len(VOCAL_SOUND_TEXTS)}")
    except Exception as e:
        print(f"  [WARN] Vocal index: {e}")

    send_rest_pose(smooth=False)
    print("Ready. Paste JSON (blank line to submit). Type quit to exit.\n")
    if MOUTH_ENGINE == "hybrid":
        print("  Tip: lips = wav2arkit | face = Brain + emotion_map | [eww] = face + sound on demand")
        print("  Flash-Attention-2 message is optional (warning only).")
        print("  Your GPU has ~4GB VRAM — Parler is the slow step; that is expected.\n")


def main() -> None:
    startup()
    print(">>> Starting (no wait). Ensure Blender stream_receiver is already running.")
    print()

    # Hook process path used by base main loop? We run our own loop for full control.
    sentence_counter = 0
    while True:
        sentences = base.get_sentences_from_raw_json()
        if sentences is None:
            print("Goodbye!")
            break
        for sent in sentences:
            sentence_counter += 1
            process_sentence_brain(sent, sentence_counter)
        time.sleep(0.2)

    sd.stop()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()
