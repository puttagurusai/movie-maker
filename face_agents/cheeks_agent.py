"""
cheeks_agent.py — smile / frown / nose / cheek squint (expression mouth corners).

Owns expression corners so lips_agent can focus on speech shapes only.
Strong preset floor + Brain + energy for realistic talking emotion.
"""

from __future__ import annotations

from typing import Dict

import emotion_map

from .base import BaseFaceAgent, FaceContext
from .brain_track import sample_brain_frame
from .expression_mix import mix_preset_brain, smooth_follow, smoothstep_ease, emotion_ease


class CheeksAgent(BaseFaceAgent):
    name = "cheeks"

    owned_keys = {
        "cheekSquintLeft", "cheekSquintRight",
        "noseSneerLeft", "noseSneerRight",
        "mouthSmileLeft", "mouthSmileRight",
        "mouthFrownLeft", "mouthFrownRight",
        "mouthPressLeft", "mouthPressRight",
    }

    def __init__(self) -> None:
        self._current: Dict[str, float] = {k: 0.0 for k in self.owned_keys}
        self._target: Dict[str, float] = {k: 0.0 for k in self.owned_keys}
        self._preset_w = 0.78
        self._brain_w = 0.22
        self._expr_scale = 1.20

    def prepare(self, ctx: FaceContext) -> None:
        i = min(1.0, float(ctx.intensity) * 1.15)
        full = emotion_map.get_blendshapes(ctx.emotion, i)
        try:
            from .policy_bridge import get_policy
            pol = get_policy()
            self._preset_w = float(pol.get("expr_preset_weight", self._preset_w))
            self._brain_w = float(pol.get("brain_cheek_weight", self._brain_w))
            self._expr_scale = float(pol.get("expr_scale", self._expr_scale))
        except Exception:
            pass

        self._target = {
            k: min(1.0, float(full.get(k, 0.0)) * self._expr_scale)
            for k in self.owned_keys
        }
        # Emotion-specific boosts (Faceit often needs stronger corners)
        e = (ctx.emotion or "").lower()
        if e == "happy":
            for k in ("mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft", "cheekSquintRight"):
                self._target[k] = min(1.0, max(self._target.get(k, 0.0), 0.75 * i * self._expr_scale / 1.2))
        elif e == "sad":
            for k in ("mouthFrownLeft", "mouthFrownRight"):
                self._target[k] = min(1.0, max(self._target.get(k, 0.0), 0.80 * i))
        elif e in ("angry", "disgusted"):
            for k in ("noseSneerLeft", "noseSneerRight"):
                self._target[k] = min(1.0, max(self._target.get(k, 0.0), 0.55 * i))

        self._current = {k: 0.0 for k in self.owned_keys}
        print(
            f"[cheeks_agent] {ctx.emotion}@{i:.2f} scale={self._expr_scale:.2f} "
            f"brain={'on' if ctx.brain_enabled else 'off'} "
            f"→ { {k: round(v, 2) for k, v in self._target.items() if v > 0.05} }"
        )

    def sample(self, ctx: FaceContext) -> Dict[str, float]:
        base_ease = smoothstep_ease(ctx.t, onset=0.16, duration=ctx.duration) if ctx.is_speaking else 0.0
        ease = emotion_ease(ctx.t, ctx.duration, ctx.emotion, base_ease)
        energy = ctx.speech_energy() if ctx.is_speaking else 0.0

        brain = {k: 0.0 for k in self.owned_keys}
        if ctx.brain_enabled and ctx.brain_frames:
            raw = sample_brain_frame(ctx.brain_frames, ctx.t, fps=ctx.brain_fps)
            brain = {k: float(raw.get(k, 0.0)) for k in self.owned_keys}

        target = mix_preset_brain(
            self.owned_keys,
            self._target,
            brain,
            ease=ease,
            energy=energy,
            preset_w=self._preset_w,
            brain_w=self._brain_w if ctx.brain_enabled else 0.0,
            energy_boost=0.18,
        )
        emo = (ctx.emotion or "").lower()
        # Hold readable floor for sustaining emotions only
        if ctx.is_speaking and ease > 0.1 and emo not in ("sarcastic", "surprised", "disgusted"):
            for k in self.owned_keys:
                floor = self._target.get(k, 0.0) * 0.45 * ease
                target[k] = max(target[k], floor)
        # Hard cap for pulse/decay emotions: energy_boost must NOT undo emotion_ease
        elif emo in ("sarcastic", "surprised", "disgusted"):
            for k in self.owned_keys:
                hard_cap = self._target.get(k, 0.0) * ease * 1.05  # 5% tolerance
                target[k] = min(target.get(k, 0.0), hard_cap)

        out = smooth_follow(self._current, target, self.owned_keys, alpha=0.38, energy=energy)
        return self._filter(out)

    def on_sentence_end(self, ctx: FaceContext) -> Dict[str, float]:
        return self._filter({k: self._current.get(k, 0.0) * 0.3 for k in self.owned_keys})
