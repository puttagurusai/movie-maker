"""
base.py — shared context and agent interface.

Runtime agents do NOT edit source code.
They only return blendshape dicts for the keys they own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np


@dataclass
class FaceContext:
    """
    Shared state for all face agents at time t (seconds into the sentence).
    Coordinator updates fields each frame before calling agent.sample().
    """

    # Timeline
    t: float = 0.0
    duration: float = 0.0
    is_speaking: bool = False

    # Semantic state (from LLM / JSON)
    emotion: str = "neutral"
    intensity: float = 0.5
    text: str = ""

    # Audio
    audio_path: Optional[str] = None
    sample_rate: int = 16000

    # Precomputed lip track (filled by LipsAgent.prepare)
    lips_frames: List[Dict[str, float]] = field(default_factory=list)
    lips_fps: float = 30.0

    # Precomputed Brain expression track (upper face / micro-expression)
    brain_frames: List[Dict[str, float]] = field(default_factory=list)
    brain_fps: float = 30.0
    brain_enabled: bool = False

    # Speech energy 0..1 at brain/lips fps (for alive expression while talking)
    energy_envelope: Optional[np.ndarray] = None

    # Optional extras any agent may set during prepare
    extras: Dict[str, Any] = field(default_factory=dict)

    def progress(self) -> float:
        """0..1 through the sentence."""
        if self.duration <= 1e-6:
            return 0.0
        return max(0.0, min(1.0, self.t / self.duration))

    def speech_energy(self) -> float:
        """RMS speech energy at current t, in [0, 1]."""
        if self.energy_envelope is None or len(self.energy_envelope) == 0:
            return 0.5 if self.is_speaking else 0.0
        fps = self.lips_fps or self.brain_fps or 30.0
        idx = int(self.t * fps)
        idx = max(0, min(len(self.energy_envelope) - 1, idx))
        return float(self.energy_envelope[idx])


class BaseFaceAgent:
    """
    One region controller.

    Subclasses must define:
      name: str
      owned_keys: set of ARKit names this agent may emit

    sample() must only return keys ⊆ owned_keys.
    """

    name: str = "base"
    owned_keys: Set[str] = set()

    def prepare(self, ctx: FaceContext) -> None:
        """Called once before a sentence plays (load models, bake tracks)."""
        pass

    def sample(self, ctx: FaceContext) -> Dict[str, float]:
        """Return blendshape values for time ctx.t. Keys must be ⊆ owned_keys."""
        return {}

    def on_sentence_end(self, ctx: FaceContext) -> Dict[str, float]:
        """Optional settle values when speech ends."""
        return {k: 0.0 for k in self.owned_keys}

    def _filter(self, values: Dict[str, float]) -> Dict[str, float]:
        """Drop any keys this agent is not allowed to control."""
        if not self.owned_keys:
            return dict(values)
        return {k: float(v) for k, v in values.items() if k in self.owned_keys}
