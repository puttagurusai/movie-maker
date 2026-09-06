"""
emotion_stats.py

Emotion → 26-D vector for Brain input #3.

Provides:
  - Canonical emotion id map (same family as myned / CREMA-style labels)
  - Precomputed mean 26-D vectors per emotion (defaults + optional .pt override)
  - get_emotion_26d(emotion, intensity) → torch.Tensor [26]

Override means by placing:
  models/brain/emotion_mean_vectors.pt
    dict: { "happy": Tensor[26], "sad": Tensor[26], ... }
  or
  models/brain/emotion_mean_vectors.json
    { "happy": [26 floats], ... }

Run to regenerate / print defaults:
  python emotion_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
BRAIN_DIR = PROJECT_ROOT / "models" / "brain"
MEAN_PT = BRAIN_DIR / "emotion_mean_vectors.pt"
MEAN_JSON = BRAIN_DIR / "emotion_mean_vectors.json"

# Canonical 7-class + extras used by orchestrator
EMOTION_ID = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "surprised": 3,
    "angry": 4,
    "fearful": 5,
    "fear": 5,
    "disgusted": 6,
    "disgust": 6,
    "sarcastic": 7,
    "thinking": 8,
}

EMOTION_DIM = 26

# Stable seed so default vectors are deterministic without training data
_SEED = 42


def _default_mean_table() -> Dict[str, torch.Tensor]:
    """
    Deterministic pseudo-mean vectors in R^26 for each emotion.
    Replace with real precomputed means from your teacher dataset when ready.
    """
    g = torch.Generator().manual_seed(_SEED)
    table: Dict[str, torch.Tensor] = {}
    # One-hot-ish bases + small noise for 9 labels
    labels = [
        "neutral", "happy", "sad", "surprised", "angry",
        "fearful", "disgusted", "sarcastic", "thinking",
    ]
    for i, lab in enumerate(labels):
        v = torch.zeros(EMOTION_DIM)
        # place a few peaks so emotions are linearly separable
        v[i % EMOTION_DIM] = 1.0
        v[(i * 3) % EMOTION_DIM] = 0.6
        v[(i * 5 + 1) % EMOTION_DIM] = 0.35
        noise = torch.randn(EMOTION_DIM, generator=g) * 0.02
        v = torch.clamp(v + noise, 0.0, 1.0)
        # neutral stays near zero
        if lab == "neutral":
            v = torch.zeros(EMOTION_DIM) + 0.05
        table[lab] = v.float()
    return table


_MEAN_CACHE: Optional[Dict[str, torch.Tensor]] = None


def load_emotion_means(force_reload: bool = False) -> Dict[str, torch.Tensor]:
    """Load mean vectors from disk if present, else defaults."""
    global _MEAN_CACHE
    if _MEAN_CACHE is not None and not force_reload:
        return _MEAN_CACHE

    table = _default_mean_table()

    if MEAN_PT.is_file():
        raw = torch.load(MEAN_PT, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            for k, v in raw.items():
                t = torch.as_tensor(v, dtype=torch.float32).flatten()
                if t.numel() == EMOTION_DIM:
                    table[str(k).lower()] = t
            print(f"[emotion_stats] Loaded means from {MEAN_PT}")
    elif MEAN_JSON.is_file():
        data = json.loads(MEAN_JSON.read_text(encoding="utf-8"))
        for k, v in data.items():
            t = torch.tensor(v, dtype=torch.float32).flatten()
            if t.numel() == EMOTION_DIM:
                table[str(k).lower()] = t
        print(f"[emotion_stats] Loaded means from {MEAN_JSON}")
    else:
        print("[emotion_stats] Using built-in default 26-D means "
              f"(place real stats at {MEAN_PT} when ready)")

    _MEAN_CACHE = table
    return table


def get_emotion_id(emotion: str) -> int:
    return EMOTION_ID.get((emotion or "neutral").lower().strip(), 0)


def get_emotion_26d(
    emotion: str,
    intensity: float = 1.0,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Brain input #3: emotion 26-D vector.

    Interpolates between neutral mean and emotion mean by intensity.
    Returns Tensor [26] on CPU or device.
    """
    means = load_emotion_means()
    emo = (emotion or "neutral").lower().strip()
    if emo == "fear":
        emo = "fearful"
    if emo == "disgust":
        emo = "disgusted"
    if emo not in means:
        emo = "neutral"

    intensity = float(max(0.0, min(1.0, intensity)))
    neutral = means["neutral"]
    target = means[emo]
    vec = neutral * (1.0 - intensity) + target * intensity

    if device is not None:
        vec = vec.to(device)
    return vec


def save_default_means(path: Optional[Path] = None) -> Path:
    """Write default means to models/brain/ for inspection / override."""
    path = path or MEAN_PT
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _default_mean_table()
    torch.save(table, path)
    # also JSON for easy viewing
    jpath = path.with_suffix(".json")
    jdata = {k: v.tolist() for k, v in table.items()}
    jpath.write_text(json.dumps(jdata, indent=2), encoding="utf-8")
    print(f"[emotion_stats] Wrote {path} and {jpath}")
    return path


def precompute_from_tensor_dict(data: Dict[str, List[List[float]]], out_path: Optional[Path] = None) -> Path:
    """
    Precompute mean vectors from raw samples.

    data: { "happy": [[26 floats], ...], "sad": [...], ... }
    """
    out_path = out_path or MEAN_PT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table: Dict[str, torch.Tensor] = {}
    for lab, rows in data.items():
        t = torch.tensor(rows, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.shape[-1] != EMOTION_DIM:
            raise ValueError(f"{lab}: expected last dim {EMOTION_DIM}, got {t.shape}")
        table[lab.lower()] = t.mean(dim=0)
    torch.save(table, out_path)
    print(f"[emotion_stats] Precomputed means → {out_path}")
    return out_path


if __name__ == "__main__":
    p = save_default_means()
    load_emotion_means(force_reload=True)
    for e in ["neutral", "happy", "sad", "angry", "surprised"]:
        v = get_emotion_26d(e, 0.8)
        print(f"  {e:12s} id={get_emotion_id(e)} L2={v.norm():.3f} first5={v[:5].tolist()}")
