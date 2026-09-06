"""
emotion_manager.py

Runtime manager for Brain input #3 (emotion_26d).

Loads emotion_stats.json (from extract_emotion_stats.py), maps incoming emotion
labels (e.g. from Groq / orchestrator JSON) to dataset categories, applies
std variation, scales by intensity, outputs [1, 26] tensor on GPU/CPU.

Usage:
  from emotion_manager import EmotionManager
  emo_mgr = EmotionManager("models/brain/emotion_stats.json")
  t = emo_mgr.get_emotion_26d_tensor("thinking", 0.75, device="cuda")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STATS = PROJECT_ROOT / "models" / "brain" / "emotion_stats.json"


class EmotionManager:
    def __init__(self, stats_json_path: Union[str, Path] = None):
        """Loads emotion_stats.json and sets up label → dataset key mappings."""
        path = Path(stats_json_path) if stats_json_path else DEFAULT_STATS
        if not path.is_file():
            raise FileNotFoundError(
                f"emotion_stats.json not found at: {path}\n"
                "Run once:\n"
                "  python extract_emotion_stats.py path/to/train.parquet "
                f"{path}"
            )
        with open(path, "r", encoding="utf-8") as f:
            self.stats = json.load(f)
        self.stats_path = path

        # Mappings from app/Groq labels to available dataset keys
        self.label_mapping = {
            "surprised": "surprise",
            "fearful": "fear",
            "disgusted": "disgust",
            "sarcastic": "calm",      # Closest available match
            "thinking": "neutral",    # Closest available match
            "happy": "happy",
            "sad": "sad",
            "angry": "angry",
            "neutral": "neutral",
            "calm": "calm",
            "surprise": "surprise",
            "fear": "fear",
            "disgust": "disgust",
        }
        print(f"[emotion_manager] Loaded stats from {path}  "
              f"labels={list(self.stats.keys())}")

    def resolve_label(self, groq_emotion: str) -> str:
        raw_label = (groq_emotion or "neutral").lower().strip()
        target_label = self.label_mapping.get(raw_label, "neutral")
        if target_label not in self.stats:
            # try raw label as-is
            if raw_label in self.stats:
                return raw_label
            target_label = "neutral" if "neutral" in self.stats else next(iter(self.stats))
        return target_label

    def get_emotion_26d_tensor(
        self,
        groq_emotion: str,
        intensity: float,
        device: str = "cuda",
        add_noise: bool = True,
        noise_scale: float = 0.25,
    ) -> torch.Tensor:
        """
        Generates a [1, 26] emotion_26d tensor.

        Formula: (mean + random_normal * std * noise_scale) * intensity
        noise_scale default 0.25 = subtle variation.
        """
        target_label = self.resolve_label(groq_emotion)

        mean_vec = np.array(self.stats[target_label]["mean"], dtype=np.float32)
        std_vec = np.array(self.stats[target_label]["std"], dtype=np.float32)

        intensity = float(max(0.0, min(1.0, intensity)))

        # 1. Add subtle Gaussian variation (scaled to keep animations stable)
        if add_noise:
            noise = np.random.normal(loc=0.0, scale=1.0, size=mean_vec.shape).astype(np.float32)
            varied_vec = mean_vec + (noise * std_vec * float(noise_scale))
        else:
            varied_vec = mean_vec.copy()

        # 2. Scale by intensity
        final_vec = varied_vec * intensity

        # 3. Convert to PyTorch Tensor [1, 26] on device
        if device is None or (isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available()):
            device = "cpu"

        tensor_26d = torch.tensor(final_vec, dtype=torch.float32, device=device).unsqueeze(0)
        return tensor_26d

    def get_emotion_26d(
        self,
        emotion: str,
        intensity: float = 1.0,
        device: Optional[str] = None,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """Convenience: returns [26] (squeezed) for brain_inference compatibility."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        t = self.get_emotion_26d_tensor(emotion, intensity, device=device, add_noise=add_noise)
        return t.squeeze(0)


# Process-wide singleton
_MANAGER: Optional[EmotionManager] = None


def get_emotion_manager(stats_json_path: Union[str, Path] = None, force: bool = False) -> EmotionManager:
    global _MANAGER
    if _MANAGER is None or force:
        _MANAGER = EmotionManager(stats_json_path or DEFAULT_STATS)
    return _MANAGER


if __name__ == "__main__":
    # Demo (needs emotion_stats.json)
    path = DEFAULT_STATS if DEFAULT_STATS.is_file() else Path("emotion_stats.json")
    if not path.is_file():
        print(f"No stats at {path}. Run extract_emotion_stats.py first.")
        raise SystemExit(1)
    emo_mgr = EmotionManager(path)
    for e, i in [("thinking", 0.75), ("happy", 0.9), ("surprised", 0.8)]:
        t = emo_mgr.get_emotion_26d_tensor(e, i, device="cpu")
        print(f"{e:12s} → {emo_mgr.resolve_label(e):10s} shape={tuple(t.shape)} "
              f"mean={t.mean().item():.4f} max={t.max().item():.4f}")
