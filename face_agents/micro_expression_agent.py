"""
micro_expression_agent.py — fleeting micro-expressions for realism.

Generates brief (80–400 ms) sub-expression flashes that appear:
  • At speech-energy peaks (energy > threshold) — emotion-congruent flash
  • Randomly at low probability — biological variability
  • At sentence onset (first 0.3 s) — expression "set" micro-burst

Coordinator uses max-merge for this agent: output values are additive
overlays on top of existing expression agents (brows / eyes / cheeks).
Owned keys overlap intentionally; coordinator takes max(existing, micro)
so micro can only ADD expression, never subtract.
"""

from __future__ import annotations

import random
from typing import Dict

from .base import BaseFaceAgent, FaceContext
from .expression_mix import smoothstep_ease


# All upper-face expression keys this agent may push
MICRO_KEYS = {
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekSquintLeft", "cheekSquintRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "noseSneerLeft", "noseSneerRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
}

# Per-emotion micro-expression recipes (peak blendshape values at full intensity)
_MICRO_RECIPES: Dict[str, Dict[str, float]] = {
    "happy": {
        "cheekSquintLeft": 0.80, "cheekSquintRight": 0.80,
        "eyeSquintLeft": 0.65,   "eyeSquintRight": 0.65,
        "mouthSmileLeft": 0.70,  "mouthSmileRight": 0.70,
        "browOuterUpLeft": 0.30, "browOuterUpRight": 0.30,
    },
    "sad": {
        "browInnerUp": 0.85,
        "browDownLeft": 0.30, "browDownRight": 0.30,
        "eyeSquintLeft": 0.40, "eyeSquintRight": 0.40,
        "mouthFrownLeft": 0.80, "mouthFrownRight": 0.80,
    },
    "angry": {
        "browDownLeft": 0.90, "browDownRight": 0.90,
        "eyeSquintLeft": 0.70, "eyeSquintRight": 0.70,
        "noseSneerLeft": 0.55, "noseSneerRight": 0.55,
    },
    "surprised": {
        "browInnerUp": 0.90,
        "browOuterUpLeft": 0.65, "browOuterUpRight": 0.65,
        "eyeWideLeft": 0.80,     "eyeWideRight": 0.80,
    },
    "fearful": {
        "eyeWideLeft": 0.90,    "eyeWideRight": 0.90,
        "browInnerUp": 0.75,
        "browOuterUpLeft": 0.35, "browOuterUpRight": 0.35,
        "eyeSquintLeft": 0.20,  "eyeSquintRight": 0.20,
    },
    "disgusted": {
        "noseSneerLeft": 0.75,  "noseSneerRight": 0.70,
        "browDownLeft": 0.45,   "browDownRight": 0.45,
        "eyeSquintLeft": 0.55,  "eyeSquintRight": 0.55,
    },
    "sarcastic": {
        "browOuterUpLeft": 0.45,  # left brow cocked
        "browDownRight": 0.40,    # right brow skeptical
        "eyeSquintRight": 0.65,   # right eye narrowed
        "eyeSquintLeft": 0.30,
        "mouthSmileLeft": 0.55,   # left smirk
        "mouthFrownRight": 0.35,  # right corner down
        "cheekSquintLeft": 0.28,
    },
    "thinking": {
        "browDownLeft": 0.45,
        "browInnerUp": 0.40,
        "eyeSquintLeft": 0.40,
    },
    "calm": {
        "browInnerUp": 0.12,
        "eyeSquintLeft": 0.15, "eyeSquintRight": 0.15,
    },
    "apologetic": {
        "browInnerUp": 0.60,
        "browDownLeft": 0.25, "browDownRight": 0.25,
        "mouthFrownLeft": 0.30, "mouthFrownRight": 0.30,
    },
    "assertive": {
        "browDownLeft": 0.55, "browDownRight": 0.55,
        "eyeSquintLeft": 0.50, "eyeSquintRight": 0.50,
        "noseSneerLeft": 0.20, "noseSneerRight": 0.20,
    },
    "concerned": {
        "browInnerUp": 0.65,
        "browDownLeft": 0.35, "browDownRight": 0.35,
        "eyeSquintLeft": 0.30, "eyeSquintRight": 0.30,
    },
    "encouraging": {
        "browOuterUpLeft": 0.40, "browOuterUpRight": 0.40,
        "mouthSmileLeft": 0.55,  "mouthSmileRight": 0.55,
        "cheekSquintLeft": 0.50, "cheekSquintRight": 0.50,
    },
    "neutral": {
        "browInnerUp": 0.12,
        "eyeSquintLeft": 0.10, "eyeSquintRight": 0.10,
    },

    # ---- Extended emotions ----
    "calm": {
        "browOuterUpLeft": 0.15, "browOuterUpRight": 0.15,
        "eyeSquintLeft": 0.08,   "eyeSquintRight": 0.08,
        "mouthSmileLeft": 0.06,  "mouthSmileRight": 0.06,
    },
    "apologetic": {
        "browInnerUp": 0.75,
        "browDownLeft": 0.22, "browDownRight": 0.22,
        "eyeSquintLeft": 0.25, "eyeSquintRight": 0.25,
        "mouthFrownLeft": 0.35, "mouthFrownRight": 0.35,
    },
    "assertive": {
        "browDownLeft": 0.55, "browDownRight": 0.55,
        "eyeSquintLeft": 0.50, "eyeSquintRight": 0.50,
    },
    "concerned": {
        "browInnerUp": 0.65,
        "browDownLeft": 0.32, "browDownRight": 0.32,
        "eyeSquintLeft": 0.32, "eyeSquintRight": 0.32,
        "mouthFrownLeft": 0.22, "mouthFrownRight": 0.22,
    },
    "encouraging": {
        "browOuterUpLeft": 0.45, "browOuterUpRight": 0.45,
        "cheekSquintLeft": 0.50, "cheekSquintRight": 0.50,
        "mouthSmileLeft": 0.65,  "mouthSmileRight": 0.65,
        "eyeSquintLeft": 0.40,   "eyeSquintRight": 0.40,
    },
}

_DEFAULT_RECIPE: Dict[str, float] = {
    "browInnerUp": 0.18,
    "eyeSquintLeft": 0.12, "eyeSquintRight": 0.12,
}


class MicroExpressionAgent(BaseFaceAgent):
    """
    Generates brief overlaid micro-expressions during speech.

    Coordinator merges with max(existing, micro) per key, so this agent
    can only amplify expression — never reduce what other agents set.
    """

    name = "micro"
    owned_keys = set(MICRO_KEYS)

    def __init__(self) -> None:
        self._active: bool = False
        self._phase: int = 0          # 0=onset 1=hold 2=decay
        self._phase_val: float = 0.0  # 0..1 envelope value
        self._recipe: Dict[str, float] = {}
        self._intensity: float = 0.40
        self._rate: float = 0.25            # micro-exprs per second (avg)
        self._energy_threshold: float = 0.65
        self._onset_dur: float = 0.10
        self._hold_dur: float = 0.08
        self._decay_dur: float = 0.18
        self._frame_t: float = 0.0         # time inside current phase
        self._next_trigger_t: float = 0.0
        self._last_sample_t: float = -1.0

    def prepare(self, ctx: FaceContext) -> None:
        self._active = False
        self._phase_val = 0.0
        self._last_sample_t = -1.0
        self._frame_t = 0.0

        try:
            from .policy_bridge import get_policy
            pol = get_policy()
            self._intensity = float(pol.get("micro_intensity", self._intensity))
            self._rate = float(pol.get("micro_rate", self._rate))
            self._energy_threshold = float(pol.get("micro_energy_threshold", self._energy_threshold))
        except Exception:
            pass

        emo = (ctx.emotion or "neutral").lower()
        self._recipe = dict(_MICRO_RECIPES.get(emo, _DEFAULT_RECIPE))

        # Schedule first micro-expression shortly after speech onset
        self._next_trigger_t = random.uniform(0.15, 0.45)
        print(
            f"[micro_agent] {emo} intensity={self._intensity:.2f} "
            f"rate={self._rate:.2f}/s thresh={self._energy_threshold:.2f}"
        )

    def _trigger(self, ctx: FaceContext) -> None:
        self._active = True
        self._phase = 0
        self._frame_t = 0.0
        self._onset_dur = random.uniform(0.06, 0.14)
        self._hold_dur = random.uniform(0.00, 0.12)
        self._decay_dur = random.uniform(0.12, 0.26)
        total = self._onset_dur + self._hold_dur + self._decay_dur
        interval = (random.expovariate(self._rate)
                    if self._rate > 1e-4 else 5.0)
        self._next_trigger_t = ctx.t + max(total + 0.08, interval)

    def sample(self, ctx: FaceContext) -> Dict[str, float]:
        if not ctx.is_speaking:
            return {}

        dt = (ctx.t - self._last_sample_t
              if self._last_sample_t >= 0 else 1.0 / 30.0)
        self._last_sample_t = ctx.t

        energy = ctx.speech_energy()
        ease = smoothstep_ease(ctx.t, onset=0.10, duration=ctx.duration)
        if ease < 0.05:
            return {}

        # Trigger check
        if not self._active:
            energy_hit = (energy >= self._energy_threshold
                          and random.random() < 0.22 * dt * 30.0)
            time_hit = ctx.t >= self._next_trigger_t
            if energy_hit or time_hit:
                self._trigger(ctx)

        if not self._active:
            return {}

        # Advance phase envelope
        self._frame_t += dt
        if self._phase == 0:  # onset
            self._phase_val = (min(1.0, self._frame_t / self._onset_dur)
                               if self._onset_dur > 1e-4 else 1.0)
            if self._frame_t >= self._onset_dur:
                self._phase = 1
                self._frame_t = 0.0
        elif self._phase == 1:  # hold
            self._phase_val = 1.0
            if self._frame_t >= self._hold_dur:
                self._phase = 2
                self._frame_t = 0.0
        else:  # decay
            self._phase_val = (max(0.0, 1.0 - self._frame_t / self._decay_dur)
                               if self._decay_dur > 1e-4 else 0.0)
            if self._frame_t >= self._decay_dur:
                self._active = False
                self._phase_val = 0.0

        if self._phase_val <= 0.001:
            return {}

        # Energy boost at louder speech moments
        e_boost = 1.0 + 0.28 * max(0.0, energy - 0.50)
        intensity = self._intensity * self._phase_val * ease * e_boost

        out: Dict[str, float] = {}
        for k, v in self._recipe.items():
            if k in self.owned_keys:
                out[k] = min(1.0, float(v) * intensity)
        return self._filter(out)

    def on_sentence_end(self, ctx: FaceContext) -> Dict[str, float]:
        self._active = False
        self._phase_val = 0.0
        return {}
