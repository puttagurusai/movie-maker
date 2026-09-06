"""
audio2emotion.py — NVIDIA Audio2Emotion-v2.2 → emotion_26d for Brain.

Replaces averaged emotion_stats.json lookup with live audio-driven conditioning
matching the dataset field:

  emotion_26d: list[float32] shape (26,)
  NVIDIA Audio2Emotion v2.2 vector (16 implicit + 10 explicit slots)

Model (gated HF):
  https://huggingface.co/nvidia/Audio2Emotion-v2.2
  network.onnx  ≈ 1.27 GB   |  ~310M params (Wav2Vec2-Large based)

Setup (one time):
  1. huggingface-cli login
  2. Accept license on the model page
  3. python audio2emotion.py --download
  4. python audio2emotion.py path/to.wav   # smoke test

Inference:
  from audio2emotion import infer_emotion_26d
  vec = infer_emotion_26d("temp/hello.wav")   # np.float32 [26]
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "audio2emotion_v2.2"
ONNX_PATH = MODEL_DIR / "network.onnx"
LAYOUT_PATH = PROJECT_ROOT / "models" / "brain" / "emotion_26d_layout.json"
HF_REPO = "nvidia/Audio2Emotion-v2.2"

# Official network_info.json emotion order (6 logits)
A2E6_NAMES = ("angry", "disgust", "fear", "happy", "neutral", "sad")

# model_config.json emotion_correspondence → index in 10-D explicit vector
# Full emotion_26d = [16 implicit | 10 explicit], so global index = 16 + local
# angry→1, disgust→3, fear→4, happy→6, sad→9, neutral→-1 (no slot)
DEFAULT_CORRESPONDENCE = {
    "angry": 1,
    "disgust": 3,
    "fear": 4,
    "happy": 6,
    "sad": 9,
    "neutral": -1,
}

# App / JSON labels → A2E6 name
LABEL_TO_A2E6 = {
    "angry": "angry",
    "anger": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happy",
    "joy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "happy",
    "surprised": "happy",
    "calm": "neutral",
    "thinking": "neutral",
    "apologetic": "sad",
    "assertive": "angry",
    "concerned": "sad",
    "encouraging": "happy",
    "sarcastic": "neutral",
}

_session = None
_input_name: Optional[str] = None
_output_names: List[str] = []
_layout: Optional[dict] = None
_model_cfg: Optional[dict] = None
_network_info: Optional[dict] = None
_warmed = False


def _load_layout() -> dict:
    global _layout
    if _layout is not None:
        return _layout
    if LAYOUT_PATH.is_file():
        _layout = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    else:
        _layout = {"implicit_mean": [0.0] * 16}
    return _layout


def _load_model_sidecars() -> Tuple[dict, dict]:
    """Load network_info.json + model_config.json shipped with the ONNX."""
    global _model_cfg, _network_info
    if _model_cfg is not None and _network_info is not None:
        return _network_info, _model_cfg
    ni_path = MODEL_DIR / "network_info.json"
    mc_path = MODEL_DIR / "model_config.json"
    _network_info = json.loads(ni_path.read_text(encoding="utf-8")) if ni_path.is_file() else {
        "emotions": list(A2E6_NAMES),
        "audio_params": {"samplerate": 16000},
    }
    _model_cfg = json.loads(mc_path.read_text(encoding="utf-8")) if mc_path.is_file() else {
        "post_processing_config": {
            "output_emotion_length": 10,
            "emotion_strength": 0.6,
            "emotion_correspondence": DEFAULT_CORRESPONDENCE,
        }
    }
    return _network_info, _model_cfg


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x - float(x.max())
    e = np.exp(x)
    return (e / max(float(e.sum()), 1e-8)).astype(np.float32)


def ensure_model(force: bool = False) -> Path:
    """
    Download network.onnx into models/audio2emotion_v2.2/ (~1.27 GB).
    Requires HF login + license acceptance for the gated repo.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if ONNX_PATH.is_file() and ONNX_PATH.stat().st_size > 1_000_000_000 and not force:
        return ONNX_PATH

    print(f"[audio2emotion] Downloading {HF_REPO} → {MODEL_DIR} (~1.27 GB) ...")
    print("  Requires: huggingface-cli login + accept model license on HF page")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("pip install huggingface_hub") from e

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=[
            "network.onnx",
            "network_info.json",
            "model_config.json",
            "model.json",
            "trt_info.json",
            "README.md",
            "LICENSE",
        ],
    )
    if not ONNX_PATH.is_file():
        raise FileNotFoundError(
            f"Download finished but {ONNX_PATH} missing. "
            "Accept the license at https://huggingface.co/nvidia/Audio2Emotion-v2.2 "
            "and run: huggingface-cli login"
        )
    print(f"[audio2emotion] Ready: {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e9:.2f} GB)")
    return ONNX_PATH


def load_session(providers: Optional[List[str]] = None):
    """Load ONNX Runtime session (cached). Prefers CUDA EP then CPU."""
    global _session, _input_name, _output_names, _warmed

    if _session is not None:
        return _session

    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("pip install onnxruntime-gpu  OR  onnxruntime") from e

    path = ensure_model()
    avail = ort.get_available_providers()
    if providers is None:
        providers = []
        if "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    print(f"[audio2emotion] Loading ONNX  providers={providers} ...")
    t0 = time.time()
    _session = ort.InferenceSession(str(path), sess_options=so, providers=providers)
    _input_name = _session.get_inputs()[0].name
    _output_names = [o.name for o in _session.get_outputs()]
    print(
        f"[audio2emotion] Loaded in {time.time()-t0:.1f}s  "
        f"in={_input_name} outs={_output_names}"
    )
    for inp in _session.get_inputs():
        print(f"  input  {inp.name}: {inp.shape} {inp.type}")
    for out in _session.get_outputs():
        print(f"  output {out.name}: {out.shape} {out.type}")

    # Warmup with 1s silence @ 16 kHz
    if not _warmed:
        dummy = np.zeros((1, 16000), dtype=np.float32)
        try:
            _run_raw(dummy)
            _warmed = True
            print("[audio2emotion] Warmup OK")
        except Exception as e:
            # try alternate shapes
            for shape in [(16000,), (1, 1, 16000), (1, 16000, 1)]:
                try:
                    _run_raw(np.zeros(shape, dtype=np.float32))
                    _warmed = True
                    print(f"[audio2emotion] Warmup OK with shape {shape}")
                    break
                except Exception:
                    continue
            if not _warmed:
                print(f"[audio2emotion] Warmup failed (will retry on real audio): {e}")

    return _session


def _prepare_audio(wav_path: str) -> np.ndarray:
    """Load mono float32 16 kHz waveform [N]."""
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        # linear resample
        n = int(round(len(audio) * 16000 / sr))
        if n < 1:
            n = 1
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    # pad very short clips (model often wants ≥0.5s)
    min_len = 8000  # 0.5 s
    if audio.shape[0] < min_len:
        audio = np.pad(audio, (0, min_len - audio.shape[0]))
    return audio.astype(np.float32)


def _run_raw(audio: np.ndarray) -> List[np.ndarray]:
    """Run session; try common input layouts."""
    sess = load_session()
    name = _input_name or sess.get_inputs()[0].name
    a = np.asarray(audio, dtype=np.float32)
    candidates = []
    if a.ndim == 1:
        candidates = [a[None, :], a[None, None, :], a]
    elif a.ndim == 2:
        candidates = [a, a[None, ...], a[:, None, :]]
    else:
        candidates = [a]

    last_err = None
    for cand in candidates:
        try:
            outs = sess.run(None, {name: cand.astype(np.float32)})
            return [np.asarray(o) for o in outs]
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Audio2Emotion ONNX failed for shapes tried: {last_err}")


def _reduce_to_vector(arr: np.ndarray) -> np.ndarray:
    """Collapse multi-frame / batch outputs to a single 1-D vector."""
    x = np.asarray(arr, dtype=np.float32)
    while x.ndim > 1:
        # mean over time/batch dims except last
        x = x.mean(axis=0)
    return x.reshape(-1)


def pack_6class_to_26d(
    logits_or_probs6: np.ndarray,
    preferred: Optional[str] = None,
    preferred_strength: float = 0.0,
    intensity: float = 1.0,
) -> np.ndarray:
    """
    Pack A2E 6-class logits/probs into dataset-style emotion_26d [26].

    Layout (NVIDIA model_config + train.parquet):
      [0:16]  implicit  — small continuous baseline from dataset mean
      [16:26] explicit (10) — slots via emotion_correspondence
              angry→1, disgust→3, fear→4, happy→6, sad→9, neutral→none
    """
    layout = _load_layout()
    _, mcfg = _load_model_sidecars()
    pp = mcfg.get("post_processing_config") or {}
    corr = pp.get("emotion_correspondence") or DEFAULT_CORRESPONDENCE
    emo_strength = float(pp.get("emotion_strength", 0.6))
    contrast = float(pp.get("emotion_contrast", 1.0))

    out = np.zeros(26, dtype=np.float32)
    imp = np.array(layout.get("implicit_mean", [0.0] * 16), dtype=np.float32)
    if imp.shape[0] >= 16:
        out[:16] = imp[:16]

    raw = np.asarray(logits_or_probs6, dtype=np.float32).reshape(-1)
    if raw.size < 6:
        raw = np.pad(raw, (0, 6 - raw.size))
    raw = raw[:6]

    # ONNX outputs logits (seen min/max outside [0,1]) → softmax
    if float(raw.min()) < -0.05 or float(raw.max()) > 1.05 or abs(float(raw.sum()) - 1.0) > 0.15:
        p = _softmax(raw * contrast)
    else:
        p = np.clip(raw, 0.0, None)
        p = p / max(float(p.sum()), 1e-8)

    # preferred emotion blend (JSON director label)
    if preferred and preferred_strength > 0:
        key = LABEL_TO_A2E6.get(preferred.lower().strip(), preferred.lower().strip())
        names = list(A2E6_NAMES)
        ni = _load_model_sidecars()[0]
        if "emotions" in ni:
            names = list(ni["emotions"])
        if key in names:
            idx = names.index(key)
            one = np.zeros(6, dtype=np.float32)
            one[idx] = 1.0
            w = float(np.clip(preferred_strength, 0.0, 1.0))
            p = (1.0 - w) * p + w * one
            p = p / max(float(p.sum()), 1e-8)

    # 10-D explicit vector → place into out[16:26]
    explicit = np.zeros(10, dtype=np.float32)
    names = list((_load_model_sidecars()[0]).get("emotions", A2E6_NAMES))
    for name, prob in zip(names, p):
        slot = corr.get(name, DEFAULT_CORRESPONDENCE.get(name, -1))
        if slot is None or int(slot) < 0:
            continue
        si = int(slot)
        if 0 <= si < 10:
            explicit[si] = float(prob)

    out[16:26] = explicit * emo_strength
    out = out * float(np.clip(intensity, 0.0, 1.5))
    return out.astype(np.float32)


def outs_to_emotion_26d(
    outs: List[np.ndarray],
    preferred: Optional[str] = None,
    preferred_strength: float = 0.0,
    intensity: float = 1.0,
) -> np.ndarray:
    """
    Convert raw ONNX outputs → [26] float32.

    Observed v2.2 ONNX: output shape (batch, 6) logits for
    [angry, disgust, fear, happy, neutral, sad].
    """
    # Native 26-D if ever present
    for o in outs:
        v = _reduce_to_vector(o)
        if v.size == 26:
            vec = v.astype(np.float32) * float(np.clip(intensity, 0.0, 1.5))
            if preferred and preferred_strength > 0.15:
                vec = _blend_prototype(vec, preferred, preferred_strength)
            return vec

    # 10-D explicit already
    for o in outs:
        v = _reduce_to_vector(o)
        if v.size == 10:
            out = np.zeros(26, dtype=np.float32)
            layout = _load_layout()
            imp = np.array(layout.get("implicit_mean", [0.0] * 16), dtype=np.float32)
            out[:16] = imp[:16] if imp.shape[0] >= 16 else 0.0
            out[16:26] = v
            return out * float(np.clip(intensity, 0.0, 1.5))

    best = None
    for o in outs:
        v = _reduce_to_vector(o)
        if v.size >= 6:
            best = v
            break
    if best is None:
        raise RuntimeError(f"Unexpected A2E outputs: {[o.shape for o in outs]}")

    return pack_6class_to_26d(
        best[:6] if best.size >= 6 else best,
        preferred=preferred,
        preferred_strength=preferred_strength,
        intensity=intensity,
    )


def _blend_prototype(vec: np.ndarray, label: str, strength: float) -> np.ndarray:
    layout = _load_layout()
    protos = layout.get("prototypes") or {}
    key = label.lower().strip()
    # map aliases
    aliases = {
        "surprised": "surprise",
        "fearful": "fear",
        "disgusted": "disgust",
        "thinking": "neutral",
        "sarcastic": "calm",
    }
    key = aliases.get(key, key)
    if key not in protos:
        return vec
    proto = np.array(protos[key], dtype=np.float32)
    w = float(np.clip(strength, 0.0, 1.0))
    return ((1.0 - w) * vec + w * proto).astype(np.float32)


def infer_emotion_26d(
    wav_path: str,
    preferred_emotion: Optional[str] = None,
    preferred_strength: float = 0.35,
    intensity: float = 1.0,
) -> np.ndarray:
    """
    WAV → emotion_26d [26] float32 (dataset-compatible conditioning vector).

    preferred_strength: how much to trust the JSON emotion label vs pure audio
      (0 = audio only, 1 = full label prototype / one-hot). Default 0.35 keeps
      audio primary while honouring director intent.
    """
    t0 = time.time()
    audio = _prepare_audio(wav_path)
    outs = _run_raw(audio)
    vec = outs_to_emotion_26d(
        outs,
        preferred=preferred_emotion,
        preferred_strength=preferred_strength,
        intensity=intensity,
    )
    dt = (time.time() - t0) * 1000
    print(
        f"[audio2emotion] {Path(wav_path).name}: 26d L2={np.linalg.norm(vec):.3f} "
        f"max={vec.max():.3f}  ({dt:.0f} ms) preferred={preferred_emotion!r}@{preferred_strength:.2f}"
    )
    return vec


def infer_emotion_26d_tensor(
    wav_path: str,
    preferred_emotion: Optional[str] = None,
    preferred_strength: float = 0.35,
    intensity: float = 1.0,
    device: str = "cuda",
):
    import torch

    v = infer_emotion_26d(
        wav_path,
        preferred_emotion=preferred_emotion,
        preferred_strength=preferred_strength,
        intensity=intensity,
    )
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return torch.tensor(v, dtype=torch.float32, device=device)


def is_available() -> bool:
    return ONNX_PATH.is_file() and ONNX_PATH.stat().st_size > 1_000_000_000


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="NVIDIA Audio2Emotion-v2.2 → emotion_26d")
    ap.add_argument("wav", nargs="?", help="Optional WAV to run")
    ap.add_argument("--download", action="store_true", help="Download model only")
    ap.add_argument("--emotion", default=None, help="Preferred emotion label")
    ap.add_argument("--strength", type=float, default=0.35)
    ap.add_argument("--intensity", type=float, default=1.0)
    args = ap.parse_args()

    if args.download or args.wav:
        ensure_model()
    if args.download and not args.wav:
        print("Download done.")
        raise SystemExit(0)
    if not args.wav:
        ap.print_help()
        raise SystemExit(0)

    load_session()
    vec = infer_emotion_26d(
        args.wav,
        preferred_emotion=args.emotion,
        preferred_strength=args.strength,
        intensity=args.intensity,
    )
    print("emotion_26d:", np.array2string(vec, precision=4, separator=", "))
    print("explicit[16:26]:", np.round(vec[16:], 4))
