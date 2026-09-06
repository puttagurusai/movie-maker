"""
brows_agent.py — brows only.

Readable emotion_map + Brain micro-motion + speech energy.
Strong preset floor so upper face never looks "empty" while talking.
"""

from __future__ import annotations

from typing import Dict

import emotion_map

from .base import BaseFaceAgent, FaceContext
from .brain_track import sample_brain_frame
from .expression_mix import mix_preset_brain, smooth_follow, smoothstep_ease, emotion_ease


class BrowsAgent(BaseFaceAgent):
    name = "brows"

    owned_keys = {
        "browDownLeft", "browDownRight",
        "browInnerUp",
        "browOuterUpLeft", "browOuterUpRight",
    }

    def __init__(self) -> None:
        self._current: Dict[str, float] = {k: 0.0 for k in self.owned_keys}
        self._target: Dict[str, float] = {k: 0.0 for k in self.owned_keys}
        self._brow_scale = 1.15  # slightly overdrive for Faceit readability
        self._preset_w = 0.75
        self._brain_w = 0.25

    def prepare(self, ctx: FaceContext) -> None:
        emotion, intensity = ctx.emotion, ctx.intensity
        try:
            from .policy_bridge import get_policy
            pol = get_policy()
            # Only override emotion/intensity if user explicitly set emotion_override
            # (pol["emotion"] defaults to "neutral" and must NOT silently shadow ctx.emotion)
            emo_ov = pol.get("emotion_override", "")
            if emo_ov:
                emotion = str(emo_ov)
                intensity = float(pol.get("intensity", intensity))
            self._brow_scale = float(pol.get("brow_scale", self._brow_scale))
            self._preset_w = float(pol.get("expr_preset_weight", self._preset_w))
            self._brain_w = float(pol.get("brain_brow_weight", self._brain_w))
        except Exception:
            pass

        # Boost intensity so emotion reads clearly on camera
        i = min(1.0, float(intensity) * 1.28)
        full = emotion_map.get_blendshapes(emotion, i)
        self._target = {
            k: min(1.0, float(full.get(k, 0.0)) * self._brow_scale)
            for k in self.owned_keys
        }
        self._current = {k: 0.0 for k in self.owned_keys}
        print(
            f"[brows_agent] {emotion}@{i:.2f} scale={self._brow_scale:.2f} "
            f"preset/brain={self._preset_w:.2f}/{self._brain_w:.2f} "
            f"brain_track={'yes' if ctx.brain_enabled else 'no'} "
            f"→ { {k: round(v, 2) for k, v in self._target.items() if v > 0.02} }"
        )

    def sample(self, ctx: FaceContext) -> Dict[str, float]:
        base_ease = smoothstep_ease(ctx.t, onset=0.14, duration=ctx.duration) if ctx.is_speaking else 0.0
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
            energy_boost=0.22,
        )
        emo = (ctx.emotion or "").lower()
        # Minimum floor for sustaining emotions (not transient ones)
        if ctx.is_speaking and ease > 0.1 and emo not in ("sarcastic", "surprised", "disgusted"):
            for k in self.owned_keys:
                floor = self._target.get(k, 0.0) * 0.40 * ease
                target[k] = max(target[k], floor)
        # Hard cap for pulse/decay emotions: energy_boost must NOT undo emotion_ease
        elif emo in ("sarcastic", "surprised", "disgusted"):
            for k in self.owned_keys:
                hard_cap = self._target.get(k, 0.0) * ease * 1.05  # 5% tolerance
                target[k] = min(target.get(k, 0.0), hard_cap)

        out = smooth_follow(self._current, target, self.owned_keys, alpha=0.40, energy=energy)
        return self._filter(out)

    def on_sentence_end(self, ctx: FaceContext) -> Dict[str, float]:
        return self._filter({k: self._current.get(k, 0.0) * 0.35 for k in self.owned_keys})
