"""
head_agent.py — owns head pose only (not blendshapes).

pitch / yaw / roll for type="head" packets, tied to speech energy + emotion.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Tuple

from .base import BaseFaceAgent, FaceContext


class HeadAgent(BaseFaceAgent):
    name = "head"
    owned_keys = set()

    def __init__(self) -> None:
        self._pitch = 0.0
        self._yaw = 0.0
        self._roll = 0.0
        self._did_nod = False
        self._nod_budget = 3         # max mid-sentence nods per sentence
        self._nod_cooldown = 0.0     # time before next nod allowed
        self._prev_energy = 0.0      # for peak detection

    def prepare(self, ctx: FaceContext) -> None:
        self._did_nod = False
        self._pitch = 0.0
        self._yaw = 0.0
        self._roll = 0.0
        self._nod_budget = max(1, min(4, int(ctx.duration / 1.5)))
        self._nod_cooldown = 0.0
        self._prev_energy = 0.0
        print("[head_agent] Ready (nod + breath + speech energy)")

    def sample(self, ctx: FaceContext) -> Dict[str, float]:
        return {}

    def get_head(self, ctx: FaceContext) -> Tuple[float, float, float]:
        energy = ctx.speech_energy() if ctx.is_speaking else 0.0
        breath = math.sin(ctx.t * 0.85) * (0.018 + 0.012 * energy)
        target_pitch = breath
        target_yaw = math.sin(ctx.t * 0.35) * 0.02 * (0.5 + energy)
        target_roll = 0.0

        if ctx.is_speaking:
            if not self._did_nod and 0.02 <= ctx.t < 0.28:
                target_pitch += 0.10
            elif ctx.t >= 0.28:
                self._did_nod = True

            # [nod] expression token: explicit nod from text author
            if ctx.extras.get("token_nod") and ctx.t > self._nod_cooldown:
                target_pitch += 0.12
                self._nod_cooldown = ctx.t + 0.40
                ctx.extras.pop("token_nod", None)

            # Mid-sentence nods: fire on energy peaks (rising edge > threshold)
            is_peak = (energy > 0.52
                       and energy > self._prev_energy + 0.08
                       and ctx.t > self._nod_cooldown
                       and self._nod_budget > 0)
            if is_peak:
                target_pitch += 0.06 + 0.04 * energy
                self._nod_cooldown = ctx.t + 0.55  # min gap between nods
                self._nod_budget -= 1
            # Continuous micro-modulation with energy
            target_pitch += 0.018 * energy

            emo = (ctx.emotion or "").lower()
            if emo == "surprised":
                target_pitch -= 0.08
            elif emo == "angry":
                target_pitch += 0.05
            elif emo == "thinking":
                target_roll = 0.06
            elif emo == "sad":
                target_pitch += 0.08
            elif emo == "happy":
                target_pitch -= 0.02
        else:
            target_pitch = breath + random.uniform(-0.015, 0.015)

        self._prev_energy = energy
        self._pitch = self._pitch * 0.82 + target_pitch * 0.18
        self._yaw = self._yaw * 0.88 + target_yaw * 0.12
        self._roll = self._roll * 0.88 + target_roll * 0.12
        return self._pitch, self._yaw, self._roll

    def on_sentence_end(self, ctx: FaceContext) -> Dict[str, float]:
        return {}
