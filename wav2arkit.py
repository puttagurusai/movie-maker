"""
wav2arkit.py

Audio → 52 ARKit blendshapes using myned-ai/wav2arkit_cpu (ONNX, CPU).

Model: models/wav2arkit_cpu/
Docs:  https://huggingface.co/myned-ai/wav2arkit_cpu
Ref:   myned-ai/avatar-chat-server (warmup, 16 kHz, chunk processing)

Official config (models/wav2arkit_cpu/config.json):
  preprocessing.sample_rate = 16000
  preprocessing.channels    = 1
  preprocessing.normalize   = false   ← do NOT peak-normalize input
  output_fps                = 30
  value_range               = [0, 1]

Raw model peaks (esp. jawOpen) are often ~0.05–0.15 (MediaPipe-trained).
We post-process mouth/jaw for Faceit/ARKit rigs so lips open visibly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover
    ort = None  # type: ignore
    _ORT_ERR = e
else:
    _ORT_ERR = None

try:
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover
    resample_poly = None  # type: ignore

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "wav2arkit_cpu"
ONNX_PATH = MODELS_DIR / "wav2arkit_cpu.onnx"
CONFIG_PATH = MODELS_DIR / "config.json"
HF_REPO = "myned-ai/wav2arkit_cpu"

# From config.json / model card
DEFAULT_BLENDSHAPE_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]

# Mouth / jaw driven by audio (upper face still from emotion_map)
MOUTH_JAW_KEYS = {
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "tongueOut",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
}

# Keys that OPEN the mouth — amplify these for visible speech on Faceit rigs
OPEN_KEYS = {
    "jawOpen",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "mouthFunnel", "mouthPucker",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthShrugLower",
    "tongueOut",
}

# Tunables (rig-dependent)
# Auto-scale maps peak jawOpen → TARGET_JAW_PEAK (main fix for "lips barely open")
TARGET_JAW_PEAK = 0.20    # more visible jaw on Faceit / agent path
MAX_AUTO_SCALE = 10.0
OPENER_BOOST = 1.30       # punchier openers for clear lip-sync
MOUTH_CLOSE_SUPPRESS = 0.92
POWER_CURVE = 0.78        # <1 = punchier mid-range; 1.0 = linear
NORMALIZE_INPUT = False   # config: preprocessing.normalize = false

_session: Optional["ort.InferenceSession"] = None
_blendshape_names: List[str] = list(DEFAULT_BLENDSHAPE_NAMES)
_output_fps: float = 30.0
_warmed = False
_name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(DEFAULT_BLENDSHAPE_NAMES)}


def ensure_model() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if ONNX_PATH.is_file() and ONNX_PATH.stat().st_size > 1000:
        return ONNX_PATH

    print(f"[wav2arkit] Downloading {HF_REPO} → {MODELS_DIR} ...")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    if not ONNX_PATH.is_file():
        raise FileNotFoundError(f"Expected ONNX at {ONNX_PATH}")
    print(f"[wav2arkit] Ready: {ONNX_PATH}")
    return ONNX_PATH


def _fast_resample(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Fast resample. Prefer scipy (no Numba JIT cold-start like first librosa call)."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    # gcd polyphase — fast path for common rates (44100→16000, 48000→16000, 24000→16000)
    if resample_poly is not None:
        from math import gcd
        g = gcd(int(orig_sr), int(target_sr))
        up = int(target_sr) // g
        down = int(orig_sr) // g
        out = resample_poly(audio.astype(np.float64), up, down)
        return out.astype(np.float32)

    if librosa is not None:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)

    # Linear fallback
    n = int(round(len(audio) * target_sr / orig_sr))
    x = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    xi = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(xi, x, audio).astype(np.float32)


def load_session(force: bool = False) -> "ort.InferenceSession":
    """Load ONNX once + warmup (as recommended by avatar-chat-server)."""
    global _session, _blendshape_names, _output_fps, _name_to_idx, _warmed

    if _session is not None and not force:
        return _session

    if ort is None:
        raise ImportError(
            "onnxruntime required: pip install onnxruntime\n"
            f"Original: {_ORT_ERR}"
        )

    path = ensure_model()
    if CONFIG_PATH.is_file():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        names = cfg.get("blendshape_names")
        if isinstance(names, list) and len(names) == 52:
            _blendshape_names = names
            _name_to_idx = {n: i for i, n in enumerate(names)}
        _output_fps = float(cfg.get("output_fps", 30))
        # respect config preprocessing.normalize
        prep = cfg.get("preprocessing") or {}
        global NORMALIZE_INPUT
        NORMALIZE_INPUT = bool(prep.get("normalize", False))

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # avatar-chat-server: model is small; 2 threads avoid thrashing with Parler on GPU host
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1

    _session = ort.InferenceSession(
        str(path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    print(f"[wav2arkit] Loaded {path.name}  fps={_output_fps}  normalize={NORMALIZE_INPUT}")

    # Warmup: 1s @ 44.1k → 16k + infer (kills first-call 20s+ librosa/ORT stall)
    if not _warmed:
        t0 = time.time()
        dummy_44k = (np.random.randn(44100) * 0.02).astype(np.float32)
        a16 = _fast_resample(dummy_44k, 44100, 16000)
        _ = _infer_raw(a16)
        _warmed = True
        print(f"[wav2arkit] Warmup done in {(time.time() - t0) * 1000:.0f} ms")

    return _session


def _infer_raw(audio_16k: np.ndarray) -> np.ndarray:
    session = load_session()
    if audio_16k.ndim != 1:
        audio_16k = np.asarray(audio_16k).reshape(-1)
    audio_in = audio_16k.astype(np.float32).reshape(1, -1)
    out = session.run(None, {"audio_waveform": audio_in})[0]
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    # Match frame count to audio length so lips never end before the WAV does
    expected = int(round(len(audio_16k) / 16000.0 * _output_fps))
    if expected > 0:
        if arr.shape[0] > expected:
            arr = arr[:expected]
        elif arr.shape[0] < expected and arr.shape[0] > 0:
            # Pad with last frame (usually near rest) so timeline = audio
            pad = np.repeat(arr[-1:], expected - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
    return np.clip(arr, 0.0, 1.0)


def postprocess_mouth(frames: np.ndarray) -> np.ndarray:
    """
    Make lips open more realistically on ARKit/Faceit rigs.

    Raw wav2arkit peaks are often tiny (~0.1 jawOpen). We:
      1) Auto-scale so peak jawOpen ≈ target_jaw_peak (FACE_POLICY or default)
      2) mouth_gain * OPENER_BOOST
      3) Power curve for punchier mid-range
      4) Suppress mouthClose when jaw is open (otherwise lips stay shut)
    """
    jaw_peak = TARGET_JAW_PEAK
    mouth_gain = 1.0
    try:
        from face_agents.policy_bridge import get_policy
        pol = get_policy()
        jaw_peak = float(pol.get("target_jaw_peak", jaw_peak))
        mouth_gain = float(pol.get("mouth_gain", mouth_gain))
    except Exception:
        pass

    out = frames.astype(np.float32, copy=True)
    jaw_i = _name_to_idx.get("jawOpen", 24)
    close_i = _name_to_idx.get("mouthClose", 26)
    open_idx = [_name_to_idx[k] for k in OPEN_KEYS if k in _name_to_idx]
    mouth_idx = [_name_to_idx[k] for k in MOUTH_JAW_KEYS if k in _name_to_idx]

    peak = float(out[:, jaw_i].max()) if out.size else 0.0
    auto = min(MAX_AUTO_SCALE, jaw_peak / peak) if peak > 1e-4 else 1.0

    out[:, mouth_idx] *= auto * mouth_gain
    out[:, open_idx] = np.clip(out[:, open_idx] * OPENER_BOOST, 0.0, 1.0)

    if POWER_CURVE != 1.0:
        out[:, open_idx] = np.power(np.clip(out[:, open_idx], 0.0, 1.0), POWER_CURVE)

    # mouthClose fights open mouth on Faceit / ARKit rigs
    if close_i is not None:
        jaw = np.clip(out[:, jaw_i], 0.0, 1.0)
        out[:, close_i] *= (1.0 - MOUTH_CLOSE_SUPPRESS * jaw)
        out[jaw > 0.12, close_i] = np.minimum(out[jaw > 0.12, close_i], 0.05)

    return np.clip(out, 0.0, 1.0)


def load_audio_16k(path: str) -> np.ndarray:
    """Load mono float32 @ 16 kHz. normalize=false per config."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = _fast_resample(audio.astype(np.float32), int(sr), 16000)

    if NORMALIZE_INPUT:
        peak = float(np.max(np.abs(audio))) or 1.0
        audio = audio / peak

    return audio.astype(np.float32)


def infer_blendshapes(audio_16k: np.ndarray, enhance_mouth: bool = True) -> np.ndarray:
    """16 kHz mono → (frames, 52)."""
    arr = _infer_raw(audio_16k)
    if enhance_mouth:
        arr = postprocess_mouth(arr)
    return arr


def frames_to_dicts(
    frames: np.ndarray,
    mouth_only: bool = True,
) -> List[Dict[str, float]]:
    names = _blendshape_names
    result = []
    for row in frames:
        d = {}
        for i, name in enumerate(names):
            if i >= len(row):
                break
            if mouth_only and name not in MOUTH_JAW_KEYS:
                continue
            d[name] = float(row[i])
        result.append(d)
    return result


def audio_file_to_frames(
    wav_path: str,
    mouth_only: bool = True,
    enhance_mouth: bool = True,
) -> Tuple[List[Dict[str, float]], float, np.ndarray]:
    """WAV → mouth blendshape frames @ 30 fps."""
    t0 = time.time()
    audio = load_audio_16k(wav_path)
    t1 = time.time()
    raw = infer_blendshapes(audio, enhance_mouth=enhance_mouth)
    t2 = time.time()
    frames = frames_to_dicts(raw, mouth_only=mouth_only)

    jaw_i = _name_to_idx.get("jawOpen", 24)
    jaw_peak = float(raw[:, jaw_i].max()) if raw.size else 0.0
    print(
        f"[wav2arkit] {Path(wav_path).name}: {len(audio)/16000:.2f}s → {len(frames)} fr @ {_output_fps:.0f}fps  "
        f"jawOpen_peak={jaw_peak:.2f}  "
        f"resample={(t1-t0)*1000:.0f}ms infer={(t2-t1)*1000:.0f}ms"
    )
    return frames, _output_fps, raw


def frame_at_time(
    frames: List[Dict[str, float]],
    t: float,
    fps: float = 30.0,
) -> Dict[str, float]:
    if not frames:
        return {}
    # Linear blend between neighbors for smoother lips
    f = t * fps
    i0 = int(np.floor(f))
    i1 = min(i0 + 1, len(frames) - 1)
    i0 = max(0, min(i0, len(frames) - 1))
    if i0 == i1:
        return frames[i0]
    a = f - i0
    d0, d1 = frames[i0], frames[i1]
    keys = set(d0) | set(d1)
    return {k: d0.get(k, 0.0) * (1 - a) + d1.get(k, 0.0) * a for k in keys}


if __name__ == "__main__":
    import sys

    load_session()
    dummy = np.random.randn(16000).astype(np.float32) * 0.05
    out = infer_blendshapes(dummy)
    print(f"dummy {out.shape} jaw peak={out[:, 24].max():.3f}")

    if len(sys.argv) > 1:
        frames, fps, raw = audio_file_to_frames(sys.argv[1])
        print(f"enhanced jaw peak={raw[:, 24].max():.3f}")
