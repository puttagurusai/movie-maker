"""
brain_inference.py — production Brain wrapper (matches training notebook).

Components:
  1) load_brain_model(device)     → shared_encoder, face_head, adapter  (3 .pt files)
  2) get_emotion_vector(...)     → emotion_26d [26] via emotion_manager
  3) extract_audio_features(...) → HuBERT+prosody → [T, 768]
  4) run_brain(...)              → ONE entry for orchestrator → ARKit frames

Module files (under models/brain/):
  shared_encoder.pt
  face_head.pt
  character_adapter.pt   (or adapter.pt)
  emotion_stats.json
  hubert-base-ls960/     (local HuBERT config/weights base)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

import prosody_gpu
from brain_model import (
    ARKIT_52_NAMES,
    CharacterAdapter,
    FaceHeadBundle,
    SharedEncoderBundle,
    _hubert_source,
)

# NVIDIA ARKit-52 order (hardcoded — index → blendshape name)
NVIDIA_ARKIT_ORDER = [
    "browDownLeft",        # 0
    "browDownRight",       # 1
    "browInnerUp",         # 2
    "browOuterUpLeft",     # 3
    "browOuterUpRight",    # 4
    "cheekPuff",           # 5
    "cheekSquintLeft",     # 6
    "cheekSquintRight",    # 7
    "eyeBlinkLeft",        # 8
    "eyeBlinkRight",       # 9
    "eyeLookDownLeft",     # 10
    "eyeLookDownRight",    # 11
    "eyeLookInLeft",       # 12
    "eyeLookInRight",      # 13
    "eyeLookOutLeft",      # 14
    "eyeLookOutRight",     # 15
    "eyeLookUpLeft",       # 16
    "eyeLookUpRight",      # 17
    "eyeSquintLeft",       # 18
    "eyeSquintRight",      # 19
    "eyeWideLeft",         # 20
    "eyeWideRight",        # 21
    "jawForward",          # 22
    "jawLeft",             # 23
    "jawOpen",             # 24
    "jawRight",            # 25
    "mouthClose",          # 26
    "mouthDimpleLeft",     # 27
    "mouthDimpleRight",    # 28
    "mouthFrownLeft",      # 29
    "mouthFrownRight",     # 30
    "mouthFunnel",         # 31
    "mouthLeft",           # 32
    "mouthLowerDownLeft",  # 33
    "mouthLowerDownRight", # 34
    "mouthPressLeft",      # 35
    "mouthPressRight",     # 36
    "mouthPucker",         # 37
    "mouthRight",          # 38
    "mouthRollLower",      # 39
    "mouthRollUpper",      # 40
    "mouthShrugLower",     # 41
    "mouthShrugUpper",     # 42
    "mouthSmileLeft",      # 43
    "mouthSmileRight",     # 44
    "mouthStretchLeft",    # 45
    "mouthStretchRight",   # 46
    "mouthUpperUpLeft",    # 47
    "mouthUpperUpRight",   # 48
    "noseSneerLeft",       # 49
    "noseSneerRight",      # 50
    "tongueOut",           # 51
]
assert len(NVIDIA_ARKIT_ORDER) == 52
assert NVIDIA_ARKIT_ORDER == list(ARKIT_52_NAMES)

try:
    from emotion_manager import EmotionManager, get_emotion_manager
except ImportError:
    EmotionManager = None  # type: ignore
    get_emotion_manager = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent
BRAIN_DIR = PROJECT_ROOT / "models" / "brain"

# ---------------------------------------------------------------------------
# Module-level caches (load once)
# ---------------------------------------------------------------------------
_encoder: Optional[SharedEncoderBundle] = None
_face_head: Optional[FaceHeadBundle] = None
_adapter: Optional[CharacterAdapter] = None
_device: Optional[str] = None
_emo_mgr = None

MOUTH_KEYS = {
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "tongueOut", "cheekPuff",
}

# Keys that open the mouth (same idea as wav2arkit postprocess)
_OPEN_KEYS = {
    "jawOpen",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "mouthFunnel", "mouthPucker",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthShrugLower",
    "tongueOut",
}

# Retarget defaults (Faceit / ARKit). Brain raw often has chronic mouthClose bias.
TARGET_JAW_PEAK = 0.22
MAX_AUTO_SCALE = 8.0
OPENER_BOOST = 1.25
MOUTH_CLOSE_FORCE = 0.0          # training data never uses mouthClose; force off
FROWN_DAMP_WHEN_SPEAKING = 0.35  # reduce frown while jaw open (avoids "downward lips")
POWER_CURVE = 0.85
NAME_TO_IDX = {n: i for i, n in enumerate(NVIDIA_ARKIT_ORDER)}


def postprocess_brain_arkit(
    frames: np.ndarray,
    *,
    target_jaw_peak: float = TARGET_JAW_PEAK,
    opener_boost: float = OPENER_BOOST,
    power_curve: float = POWER_CURVE,
) -> np.ndarray:
    """
    Fix common Brain→Faceit issues that made lips look downward / sealed vs wav2arkit.

    Not an ARKit *name* remapping bug — NVIDIA order matches training + wav2arkit.
    Problems observed on raw Brain output:
      • mouthClose stuck ~0.5–0.6 (train.parquet mouthClose ≈ 0) → sealed / pulled-down lips
      • weak jawOpen peaks vs mouthLowerDown → lower lip drops without open jaw
      • character_adapter raises a ~0.15 floor on all channels

    Steps mirror wav2arkit.postprocess_mouth, tuned for Brain biases.
    """
    if frames.size == 0:
        return frames
    out = np.asarray(frames, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] < 52:
        return np.clip(out, 0.0, 1.0)

    jaw_i = NAME_TO_IDX["jawOpen"]
    close_i = NAME_TO_IDX["mouthClose"]
    frown_l = NAME_TO_IDX["mouthFrownLeft"]
    frown_r = NAME_TO_IDX["mouthFrownRight"]
    open_idx = [NAME_TO_IDX[k] for k in _OPEN_KEYS if k in NAME_TO_IDX]
    mouth_idx = [NAME_TO_IDX[k] for k in MOUTH_KEYS if k in NAME_TO_IDX]

    # 1) Kill mouthClose — primary cause of "lips pointing down / sealed while talking"
    out[:, close_i] = MOUTH_CLOSE_FORCE

    # 2) Light baseline pull toward 0 on near-static floor (adapter residual ~0.15)
    #    Only shrink channels that barely vary (not speech openers).
    for i in range(out.shape[1]):
        if i in open_idx or i == jaw_i:
            continue
        col = out[:, i]
        if float(col.std()) < 0.02 and float(col.mean()) > 0.12:
            out[:, i] = np.clip(col - 0.10, 0.0, 1.0)

    # 2b) Amplify *temporal* dynamics on mouth channels.
    #     Brain often has high mean / tiny std → mouth looks stuck open (Rhubarb-ish).
    #     Re-center deviations around a lower baseline and boost frame-to-frame shape.
    for i in mouth_idx:
        if i == close_i:
            continue
        col = out[:, i].astype(np.float32)
        mean = float(col.mean())
        std = float(col.std())
        if std < 1e-5:
            continue
        # Keep a modest baseline so lips don't go fully dead, boost motion ~3–4×
        dyn_boost = 3.5 if std < 0.03 else 2.0
        baseline = min(mean * 0.35, 0.08)
        out[:, i] = np.clip(baseline + (col - mean) * dyn_boost + mean * 0.25, 0.0, 1.0)

    # 3) Auto-scale mouth/jaw so peak jawOpen ≈ target (visible speech)
    peak = float(out[:, jaw_i].max()) if out.shape[0] else 0.0
    auto = min(MAX_AUTO_SCALE, target_jaw_peak / peak) if peak > 1e-4 else 1.0
    # Never shrink dynamic range if peak already past target (preserve lip motion)
    if auto < 1.0:
        auto = 1.0
    out[:, mouth_idx] *= auto
    out[:, open_idx] = np.clip(out[:, open_idx] * opener_boost, 0.0, 1.0)

    if power_curve != 1.0:
        out[:, open_idx] = np.power(np.clip(out[:, open_idx], 0.0, 1.0), power_curve)

    # 4) When jaw is open, damp frowns (downward corners) so speech doesn't look sad
    jaw = np.clip(out[:, jaw_i], 0.0, 1.0)
    damp = 1.0 - FROWN_DAMP_WHEN_SPEAKING * jaw
    out[:, frown_l] *= damp
    out[:, frown_r] *= damp

    # 5) mouthClose stay off even after scale
    out[:, close_i] = MOUTH_CLOSE_FORCE

    return np.clip(out, 0.0, 1.0)


# ===========================================================================
# COMPONENT 1 — Model loader
# ===========================================================================
def load_brain_model(device: Optional[str] = None):
    """
    Loads THREE files:
      shared_encoder.pt  → encoder bundle (HuBERT + audio path + shared layers)
      face_head.pt       → face head + dual output heads
      character_adapter.pt → adapter weights
    Cached as module-level globals; eval() + used under no_grad at inference.
    """
    global _encoder, _face_head, _adapter, _device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if (
        _encoder is not None
        and _face_head is not None
        and _adapter is not None
        and _device == device
    ):
        return _encoder, _face_head, _adapter

    dev = torch.device(device)
    enc_path = BRAIN_DIR / "shared_encoder.pt"
    face_path = BRAIN_DIR / "face_head.pt"
    adapt_path = BRAIN_DIR / "character_adapter.pt"
    if not adapt_path.is_file():
        adapt_path = BRAIN_DIR / "adapter.pt"

    # Auto-export from monolithic brain.pt if split files missing
    if not enc_path.is_file() or not face_path.is_file() or not adapt_path.is_file():
        mono = BRAIN_DIR / "brain.pt"
        if mono.is_file():
            print("[brain] Split modules missing — exporting from brain.pt ...")
            from export_brain_modules import export
            export(mono, BRAIN_DIR)
        else:
            raise FileNotFoundError(
                f"Need {enc_path.name}, {face_path.name}, character_adapter.pt "
                f"or monolithic models/brain/brain.pt"
            )

    print(f"[brain] Loading modules on {device} ...")
    print(f"  HuBERT source: {_hubert_source()}")

    enc = SharedEncoderBundle()
    enc.load_state_dict(torch.load(enc_path, map_location="cpu", weights_only=False), strict=True)
    enc.to(dev).eval()

    face = FaceHeadBundle()
    face.load_state_dict(torch.load(face_path, map_location="cpu", weights_only=False), strict=True)
    face.to(dev).eval()

    adapt = CharacterAdapter()
    adapt.load_state_dict(torch.load(adapt_path, map_location="cpu", weights_only=False), strict=True)
    adapt.to(dev).eval()

    _encoder, _face_head, _adapter, _device = enc, face, adapt, device
    print("[brain] Loaded shared_encoder + face_head + character_adapter (cached)")
    return _encoder, _face_head, _adapter


# ===========================================================================
# COMPONENT 2 — emotion_26d resolver
# ===========================================================================
# "a2e"  = NVIDIA Audio2Emotion-v2.2 from audio (preferred, matches training source)
# "stats" = averaged emotion_stats.json (legacy fallback)
import os as _os
EMOTION_26D_SOURCE = _os.environ.get("EMOTION_26D_SOURCE", "a2e").strip().lower()
# How much JSON emotion label blends into A2E (0=pure audio, 1=label prototype)
A2E_PREFERRED_STRENGTH = float(_os.environ.get("A2E_PREFERRED_STRENGTH", "0.35"))


def get_emotion_vector(
    emotion_label: str,
    intensity: float,
    device: str = "cuda",
    noise_scale: float = 0.25,
    wav_path: Optional[str] = None,
    source: Optional[str] = None,
) -> torch.Tensor:
    """
    Return emotion_26d [26] float32 on device.

    Preferred path (source=\"a2e\"):
      NVIDIA Audio2Emotion-v2.2 ONNX on the sentence WAV → 26-D vector
      (16 implicit + 10 explicit), same conditioning family as train.parquet.

    Fallback (source=\"stats\" or A2E missing):
      emotion_stats.json mean/std averaging (old path).
    """
    global _emo_mgr
    src = (source or EMOTION_26D_SOURCE or "a2e").lower()
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # --- live Audio2Emotion from audio ---
    if src in ("a2e", "audio2emotion", "audio") and wav_path:
        try:
            import audio2emotion as a2e

            if not a2e.is_available():
                a2e.ensure_model()
            t = a2e.infer_emotion_26d_tensor(
                wav_path,
                preferred_emotion=emotion_label,
                preferred_strength=A2E_PREFERRED_STRENGTH,
                intensity=float(intensity),
                device=device,
            )
            print(
                f"[brain] emotion_26d source=Audio2Emotion-v2.2 "
                f"L2={t.norm():.3f} label={emotion_label!r}"
            )
            return t  # [26]
        except Exception as e:
            print(f"[brain] Audio2Emotion failed ({e}) — falling back to emotion_stats averages")

    # --- legacy averages ---
    if get_emotion_manager is None:
        raise ImportError("emotion_manager.py required")

    if _emo_mgr is None:
        stats = BRAIN_DIR / "emotion_stats.json"
        _emo_mgr = get_emotion_manager(stats)

    t = _emo_mgr.get_emotion_26d_tensor(
        groq_emotion=emotion_label,
        intensity=intensity,
        device=device,
        add_noise=True,
        noise_scale=noise_scale,
    )  # [1, 26]
    print(
        f"[brain] emotion_26d source=stats_avg "
        f"L2={t.norm():.3f} label={emotion_label!r}"
    )
    return t.squeeze(0)  # [26]


# ===========================================================================
# COMPONENT 3 — HuBERT feature extractor (+ prosody concat + project)
# ===========================================================================
def extract_audio_features(wav_path: str, device: str = "cuda") -> torch.Tensor:
    """
    WAV → mono 16k → HuBERT → [T50,768]
      → align prosody [T,3] → cat → Linear 771→768 → 30fps [T,768]
    """
    encoder, _, _ = load_brain_model(device)
    dev = torch.device(device if not (device.startswith("cuda") and not torch.cuda.is_available()) else "cpu")

    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.to(dev)  # [1, T]

    # Prosody on same device preference
    pros_dev = "cuda" if dev.type == "cuda" else "cpu"
    prosody = prosody_gpu.extract_prosody(wav_path, device=pros_dev)  # [T, 3]
    prosody = prosody.to(dev).unsqueeze(0)  # [1, T, 3]

    with torch.no_grad():
        # HuBERT runs inside encode_audio_features (frozen early layers)
        feats = encoder.encode_audio_features(wav, prosody)  # [1, T30, 768]
    return feats.squeeze(0)  # [T, 768]


# ===========================================================================
# COMPONENT 4 — Main inference (orchestrator calls this)
# ===========================================================================
@torch.inference_mode()
def run_brain(
    wav_path: str,
    emotion_label: str,
    intensity: float,
    device: Optional[str] = None,
    mouth_only: bool = False,
    postprocess: bool = True,
    use_adapter: bool = True,
) -> Tuple[List[Dict[str, float]], float, np.ndarray]:
    """
    ONE function the orchestrator calls.

    Steps:
      1. Load models (cached)
      2. emotion_26d = get_emotion_vector(...)
      3. audio_features = extract_audio_features(...)   [T × 768]
      4. Align time (features already on one T)
      5. shared_out, emotion_emb = fusion encoder      [T × 384], [384]
      6. face_out = face_head(shared_out, emotion_emb)
         HEAD_A lower 29 + HEAD_B upper 23 → scatter to [T × 52] NVIDIA order
         (trained weights are 29+23=52, not 28+24)
      7. final = character_adapter(face_out)             [T × 52]  (optional)
      8. postprocess_brain_arkit (mouthClose kill + jaw retarget)  (default on)
      9. Convert each frame to named dict via NVIDIA_ARKIT_ORDER
     10. Return frames list

    Returns:
      frames: list of T dicts (52 named values, or mouth-only if mouth_only=True)
      fps: 30.0
      raw: np.ndarray [T, 52]  (after postprocess if enabled)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # 1. Load models (cached after first call)
    encoder, face_head, adapter = load_brain_model(device)
    dev = torch.device(device)

    # 2. Get emotion_26d [26] — Audio2Emotion from WAV (not label averages)
    emotion_vector = get_emotion_vector(
        emotion_label,
        intensity,
        device=device,
        wav_path=wav_path,
    )
    print(
        f"[brain] emotion={emotion_label} intensity={intensity:.2f} "
        f"26d L2={emotion_vector.norm():.3f}"
    )

    # 3. Get audio features [T × 768]
    audio_features = extract_audio_features(wav_path, device=device)
    T = int(audio_features.shape[0])
    print(f"[brain] audio_features {tuple(audio_features.shape)}")

    # 4. Align time dimensions (audio stream already one T; ±2 ok)
    audio_b = audio_features.unsqueeze(0)  # [1, T, 768]
    emo_b = emotion_vector.unsqueeze(0)    # [1, 26]
    inten = torch.tensor([float(intensity)], device=dev, dtype=torch.float32)

    # 5. Run fusion / shared encoder → [1, T, 384]
    #    emotion_emb is the 384-D conditioned vector used by face_head cross-attn
    #    (built from emotion_26d [26] + intensity inside the encoder bundle)
    shared_out, emotion_emb = encoder.forward_fusion(audio_b, emo_b, inten)
    # shared_out: [1, T, 384]   emotion_emb: [1, 384]

    # 6. Run face head:
    #    face_head(shared_out [T×384], emotion path)
    #    → lower head 29 (HEAD_A / lip-sync) + upper head 23 (HEAD_B / expression)
    #    → scatter into [T × 52] NVIDIA ARKit-52 ordering (not plain concat)
    face_out = face_head(shared_out, emotion_emb)  # [1, T, 52]
    # (Internally: HEAD_A [T×29] + HEAD_B [T×23] scattered to ARKit indices)

    # 7. Character adapter (residual mix) — can raise a floor; postprocess compensates
    final = adapter(face_out) if use_adapter else face_out  # [1, T, 52]

    # 8. Numpy + speech retarget (critical for Faceit — same class of fixes as wav2arkit)
    raw = final.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [T, 52]
    jaw_i = 24
    close_i = 26
    pre_jaw = float(raw[:, jaw_i].max()) if raw.size else 0.0
    pre_close = float(raw[:, close_i].mean()) if raw.size else 0.0
    if postprocess:
        raw = postprocess_brain_arkit(raw)

    # 9. Convert to list of named dicts
    frames: List[Dict[str, float]] = []
    for t in range(raw.shape[0]):
        frame_dict: Dict[str, float] = {}
        for i in range(52):
            name = NVIDIA_ARKIT_ORDER[i]
            val = float(raw[t, i])
            if mouth_only and name not in MOUTH_KEYS:
                continue
            frame_dict[name] = val
        frames.append(frame_dict)

    # 10. Return frames list
    print(
        f"[brain] out T={raw.shape[0]} cols=52 "
        f"jawOpen_peak={raw[:, jaw_i].max():.3f} "
        f"(pre={pre_jaw:.3f}) mouthClose_mean={raw[:, close_i].mean():.3f} "
        f"(pre={pre_close:.3f}) postprocess={postprocess} mouth_only={mouth_only}"
    )
    return frames, 30.0, raw


def _to_dicts(arr: np.ndarray, mouth_only: bool = False) -> List[Dict[str, float]]:
    """Helper: raw [T,52] → list of named frames using NVIDIA_ARKIT_ORDER."""
    frames = []
    for t in range(arr.shape[0]):
        frame_dict = {}
        for i in range(min(52, arr.shape[1])):
            name = NVIDIA_ARKIT_ORDER[i]
            if mouth_only and name not in MOUTH_KEYS:
                continue
            frame_dict[name] = float(arr[t, i])
        frames.append(frame_dict)
    return frames


# ---------------------------------------------------------------------------
# Orchestrator-compatible API (legacy names)
# ---------------------------------------------------------------------------
class BrainModel:
    """Thin class wrapper so existing get_brain() / predict() still work."""

    def __init__(self, device: Optional[str] = None, allow_stub: bool = False, **kwargs):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fps = 30.0
        self.arkit_names = list(ARKIT_52_NAMES)
        load_brain_model(self.device)

    def predict(
        self,
        audio_path: str,
        emotion: str,
        intensity: float = 1.0,
        prosody=None,  # ignored — extracted inside
        emotion_26d=None,  # optional override
        mouth_only: bool = True,
    ):
        if emotion_26d is not None:
            # inject via temporary run path
            enc, face, adapt = load_brain_model(self.device)
            dev = torch.device(self.device)
            wav, sr = torchaudio.load(audio_path)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            wav = wav.to(dev)
            if prosody is None:
                prosody = prosody_gpu.extract_prosody(
                    audio_path, device="cuda" if dev.type == "cuda" else "cpu"
                )
            prosody = prosody.to(dev)
            if prosody.dim() == 2:
                prosody = prosody.unsqueeze(0)
            e26 = emotion_26d.to(dev).float()
            if e26.dim() == 1:
                e26 = e26.unsqueeze(0)
            with torch.no_grad():
                feats = enc.encode_audio_features(wav, prosody)
                inten = torch.tensor([float(intensity)], device=dev)
                shared, emo_emb = enc.forward_fusion(feats, e26, inten)
                arkit = face(shared, emo_emb)
                arkit = adapt(arkit)
            raw = arkit.squeeze(0).cpu().numpy().astype(np.float32)
            return _to_dicts(raw, mouth_only), 30.0, raw

        return run_brain(audio_path, emotion, intensity, device=self.device, mouth_only=mouth_only)


_BRAIN_WRAP: Optional[BrainModel] = None


def get_brain(allow_stub: bool = False, force_reload: bool = False, **kwargs) -> BrainModel:
    global _BRAIN_WRAP, _encoder, _face_head, _adapter, _device
    if force_reload:
        _BRAIN_WRAP = None
        _encoder = _face_head = _adapter = None
        _device = None
    if _BRAIN_WRAP is None:
        _BRAIN_WRAP = BrainModel(allow_stub=allow_stub)
    return _BRAIN_WRAP


def run_brain_on_wav(wav_path, emotion, intensity=1.0, mouth_only=True, allow_stub=False):
    return run_brain(wav_path, emotion, intensity, mouth_only=mouth_only)


if __name__ == "__main__":
    import sys
    print("BRAIN_DIR", BRAIN_DIR)
    for n in ("shared_encoder.pt", "face_head.pt", "character_adapter.pt", "brain.pt", "emotion_stats.json"):
        p = BRAIN_DIR / n
        print(f"  {n}: {'OK' if p.is_file() else 'MISSING'} {p}")
    if len(sys.argv) > 1:
        frames, fps, raw = run_brain(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "happy", 0.8)
        print("frames", len(frames), "raw", raw.shape)
    else:
        print("Usage: python brain_inference.py <wav> [emotion]")
        print("First run export if needed: python export_brain_modules.py")
