"""
brain_track.py — bake Brain full-face ARKit track for expression agents.

Lips still come from wav2arkit (lips_agent). This track feeds brows / eyes /
cheeks with speech-correlated upper-face motion from the trained encoder.

Fails soft if Brain weights or CUDA are unavailable — agents fall back to
emotion_map presets only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Keys that expression agents may pull from Brain (not mouth/jaw speech core)
BRAIN_EXPRESSION_KEYS = {
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekSquintLeft", "cheekSquintRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "noseSneerLeft", "noseSneerRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthPressLeft", "mouthPressRight",
}


def compute_energy_envelope(
    audio: np.ndarray,
    sr: int,
    fps: float = 30.0,
) -> np.ndarray:
    """Per-frame RMS energy in [0, 1] at `fps`."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return np.zeros(1, dtype=np.float32)
    hop = max(1, int(round(sr / fps)))
    n = max(1, int(np.ceil(len(audio) / hop)))
    env = np.zeros(n, dtype=np.float32)
    for i in range(n):
        seg = audio[i * hop : (i + 1) * hop]
        env[i] = float(np.sqrt(np.mean(np.square(seg)) + 1e-9))
    peak = float(env.max())
    if peak > 1e-6:
        env /= peak
    # light smooth
    if n >= 3:
        k = np.ones(3, dtype=np.float32) / 3.0
        env = np.convolve(env, k, mode="same")
    return env


def energy_at(envelope: np.ndarray, t: float, fps: float = 30.0) -> float:
    if envelope is None or len(envelope) == 0:
        return 0.0
    idx = int(t * fps)
    idx = max(0, min(len(envelope) - 1, idx))
    return float(envelope[idx])


def bake_brain_expression(
    wav_path: str,
    emotion: str,
    intensity: float,
    device: Optional[str] = None,
) -> Tuple[List[Dict[str, float]], float, Optional[np.ndarray]]:
    """
    Run Brain once → list of upper/expression dicts + fps + raw [T,52] or None.

    Returns ([], 30.0, None) on failure.
    """
    try:
        import torch
        import brain_inference
    except Exception as e:
        print(f"[brain_track] Brain import failed: {e}")
        return [], 30.0, None

    if not Path(wav_path).is_file():
        print(f"[brain_track] Missing audio: {wav_path}")
        return [], 30.0, None

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        frames, fps, raw = brain_inference.run_brain(
            wav_path=wav_path,
            emotion_label=emotion,
            intensity=float(intensity),
            device=str(device),
            mouth_only=False,
            postprocess=True,
        )
    except Exception as e:
        print(f"[brain_track] run_brain failed: {e}")
        return [], 30.0, None

    cleaned: List[Dict[str, float]] = []
    for fr in frames:
        cleaned.append({
            k: float(v)
            for k, v in fr.items()
            if k in BRAIN_EXPRESSION_KEYS and float(v) > 1e-6
        })

    # Log a few peaks
    peaks = {}
    for k in ("browInnerUp", "browOuterUpLeft", "eyeWideLeft", "mouthSmileLeft", "noseSneerLeft"):
        if raw is not None and raw.ndim == 2:
            try:
                from brain_inference import NVIDIA_ARKIT_ORDER
                i = NVIDIA_ARKIT_ORDER.index(k)
                peaks[k] = float(raw[:, i].max())
            except Exception:
                pass
    print(
        f"[brain_track] Baked {len(cleaned)} frames @ {fps:.0f}fps "
        f"emotion={emotion} peaks={ {k: round(v, 3) for k, v in peaks.items()} }"
    )
    return cleaned, float(fps), raw


def sample_brain_frame(
    frames: List[Dict[str, float]],
    t: float,
    fps: float = 30.0,
) -> Dict[str, float]:
    """Linear blend between neighboring Brain frames (same idea as wav2arkit)."""
    if not frames:
        return {}
    try:
        import wav2arkit
        return wav2arkit.frame_at_time(frames, t, fps=fps)
    except Exception:
        idx = int(round(t * fps))
        idx = max(0, min(len(frames) - 1, idx))
        return dict(frames[idx])
