"""
eyes_agent.py — blink, gaze, squint/wide with strong emotion + Brain.
"""

from __future__ import annotations

import random
from typing import Dict

import emotion_map

from .base import BaseFaceAgent, FaceContext
from .brain_track import sample_brain_frame
from .expression_mix import smoothstep_ease, emotion_ease


class EyesAgent(BaseFaceAgent):
    name = "eyes"

    owned_keys = {
        "eyeBlinkLeft", "eyeBlinkRight",
        "eyeSquintLeft", "eyeSquintRight",
        "eyeWideLeft", "eyeWideRight",
        "eyeLookDownLeft", "eyeLookDownRight",
        "eyeLookInLeft", "eyeLookInRight",
        "eyeLookOutLeft", "eyeLookOutRight",
        "eyeLookUpLeft", "eyeLookUpRight",
    }

    # Eye-contact state machine constants
    _CONTACT_DUR   = (2.0, 4.5)   # seconds holding eye contact
    _AWAY_DUR      = (0.8, 2.0)   # seconds looking away
    _CONTACT_RANGE = 0.07          # gaze radius during eye contact (near center)
    _AWAY_RANGE    = 0.20          # gaze radius when looking away

    def __init__(self) -> None:
        self._next_blink_t = 0.0
        self._blink_phase = 0.0
        self._blink_active = False
        self._gaze_x = 0.0
        self._gaze_y = 0.0
        self._gaze_target_x = 0.0
        self._gaze_target_y = 0.0
        self._next_gaze_t = 0.0
        # Eye-contact state machine
        self._gaze_state = "contact"  # "contact" | "away"
        self._gaze_state_end_t = random.uniform(*self._CONTACT_DUR)
        self._emo_target: Dict[str, float] = {}
        self._brain_w = 0.35

    def prepare(self, ctx: FaceContext) -> None:
        self._schedule_blink(ctx, force_soon=True)
        self._pick_gaze(ctx)
        i = min(1.0, float(ctx.intensity) * 1.15)
        full = emotion_map.get_blendshapes(ctx.emotion, i)
        self._emo_target = {
            k: float(full.get(k, 0.0))
            for k in ("eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight")
        }
        # Ensure minimum readable eye emotion
        e = (ctx.emotion or "").lower()
        if e == "happy":
            self._emo_target["eyeSquintLeft"] = max(self._emo_target.get("eyeSquintLeft", 0), 0.45 * i)
            self._emo_target["eyeSquintRight"] = max(self._emo_target.get("eyeSquintRight", 0), 0.45 * i)
        elif e == "surprised":
            self._emo_target["eyeWideLeft"] = max(self._emo_target.get("eyeWideLeft", 0), 0.75 * i)
            self._emo_target["eyeWideRight"] = max(self._emo_target.get("eyeWideRight", 0), 0.75 * i)
        elif e == "angry":
            self._emo_target["eyeSquintLeft"] = max(self._emo_target.get("eyeSquintLeft", 0), 0.50 * i)
            self._emo_target["eyeSquintRight"] = max(self._emo_target.get("eyeSquintRight", 0), 0.50 * i)
        elif e == "fearful":
            self._emo_target["eyeWideLeft"] = max(self._emo_target.get("eyeWideLeft", 0), 0.70 * i)
            self._emo_target["eyeWideRight"] = max(self._emo_target.get("eyeWideRight", 0), 0.70 * i)

        try:
            from .policy_bridge import get_policy
            pol = get_policy()
            self._brain_w = float(pol.get("brain_eye_weight", self._brain_w))
        except Exception:
            pass
        print(f"[eyes_agent] {ctx.emotion} brain={'on' if ctx.brain_enabled else 'off'}")

    def _schedule_blink(self, ctx: FaceContext, force_soon: bool = False) -> None:
        emo = (ctx.emotion or "neutral").lower()
        ranges = {
            "fearful": (1.5, 3.0), "angry": (2.0, 3.5), "thinking": (5.0, 9.0),
            "sad": (4.0, 8.0), "happy": (3.2, 6.5),
        }
        lo, hi = ranges.get(emo, (2.8, 5.5))
        delay = random.uniform(0.35, 1.0) if force_soon else random.uniform(lo, hi)
        self._next_blink_t = ctx.t + delay
        self._blink_active = False
        self._blink_phase = 0.0

    def _pick_gaze(self, ctx: FaceContext) -> None:
        energy = ctx.speech_energy() if ctx.is_speaking else 0.0

        # Advance state machine
        if ctx.t >= self._gaze_state_end_t:
            if self._gaze_state == "contact":
                self._gaze_state = "away"
                self._gaze_state_end_t = ctx.t + random.uniform(*self._AWAY_DUR)
            else:
                self._gaze_state = "contact"
                self._gaze_state_end_t = ctx.t + random.uniform(*self._CONTACT_DUR)

        if self._gaze_state == "contact":
            # Near-center: eye contact with listener
            r = self._CONTACT_RANGE * (1.0 + 0.4 * energy)
            self._gaze_target_x = random.uniform(-r, r)
            self._gaze_target_y = random.uniform(-r * 0.4, r * 0.4)
        else:
            # Away: look to side or up-left (classic "thinking" gaze)
            r = self._AWAY_RANGE * (0.8 + 0.4 * energy)
            # Bias toward up-left (recall direction) or side
            side = random.choice([-1.0, 1.0, -1.0])  # 2/3 chance left
            self._gaze_target_x = side * random.uniform(r * 0.5, r)
            self._gaze_target_y = random.uniform(0.0, r * 0.6)  # slightly up

        dt = random.uniform(0.5, 1.6 if ctx.is_speaking else 2.8)
        self._next_gaze_t = ctx.t + dt

    def sample(self, ctx: FaceContext) -> Dict[str, float]:
        out: Dict[str, float] = {k: 0.0 for k in self.owned_keys}
        base_ease = smoothstep_ease(ctx.t, onset=0.14, duration=ctx.duration) if ctx.is_speaking else 0.0
        # emotion_ease only applies to eyeWide (transient shock) — squint/blink use plain ease
        eye_wide_ease = emotion_ease(ctx.t, ctx.duration, ctx.emotion, base_ease)
        ease = base_ease
        energy = ctx.speech_energy() if ctx.is_speaking else 0.0

        if not self._blink_active and ctx.t >= self._next_blink_t:
            self._blink_active = True
            self._blink_phase = 0.0

        blink_val = 0.0
        if self._blink_active:
            # Faster close (~100ms) so lids fully seal like a real blink
            self._blink_phase += 1.0 / 5.5
            if self._blink_phase < 1.0:
                blink_val = self._blink_phase ** 0.7  # ease-in close
            elif self._blink_phase < 2.0:
                blink_val = (2.0 - self._blink_phase) ** 1.1
            else:
                self._blink_active = False
                self._schedule_blink(ctx)

        blink_scale = gaze_scale = 1.0
        try:
            from .policy_bridge import get_policy
            pol = get_policy()
            blink_scale = float(pol.get("blink_scale", 1.15))
            gaze_scale = float(pol.get("gaze_scale", 1.25))
        except Exception:
            blink_scale = 1.15
            gaze_scale = 1.25

        # Strong full lid close
        out["eyeBlinkLeft"] = min(1.0, blink_val * blink_scale * 1.05)
        out["eyeBlinkRight"] = min(1.0, blink_val * blink_scale * 1.05)

        if ctx.t >= self._next_gaze_t:
            self._pick_gaze(ctx)
        self._gaze_x = self._gaze_x * 0.82 + self._gaze_target_x * 0.18
        self._gaze_y = self._gaze_y * 0.82 + self._gaze_target_y * 0.18
        gx, gy = self._gaze_x * gaze_scale, self._gaze_y * gaze_scale
        # Slightly larger readable gaze (still natural)
        gx = max(-0.55, min(0.55, gx * 1.15))
        gy = max(-0.4, min(0.4, gy * 1.15))
        if gx < 0:
            out["eyeLookInLeft"] = -gx
            out["eyeLookOutRight"] = -gx
        else:
            out["eyeLookOutLeft"] = gx
            out["eyeLookInRight"] = gx
        if gy < 0:
            out["eyeLookDownLeft"] = -gy
            out["eyeLookDownRight"] = -gy
        else:
            out["eyeLookUpLeft"] = gy
            out["eyeLookUpRight"] = gy

        # Strong emotion eyes + energy
        emo = (ctx.emotion or "").lower()
        _transient_eye_emos = ("surprised", "fearful", "disgusted")
        e_mul = 1.0 + 0.15 * energy
        for k, v in self._emo_target.items():
            # eyeWide uses emotion_ease (decays for surprised/fearful) — squint uses plain ease
            if k in ("eyeWideLeft", "eyeWideRight") and emo in _transient_eye_emos:
                applied_ease = eye_wide_ease
                val = min(1.0, float(v) * applied_ease * e_mul)
                val = min(val, float(v) * applied_ease * 1.05)  # hard cap
            else:
                val = min(1.0, float(v) * ease * e_mul)
            out[k] = max(out.get(k, 0.0), val)

        if ctx.brain_enabled and ctx.brain_frames:
            raw = sample_brain_frame(ctx.brain_frames, ctx.t, fps=ctx.brain_fps)
            for k in ("eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight"):
                b = float(raw.get(k, 0.0)) * self._brain_w * ease
                out[k] = min(1.0, max(out.get(k, 0.0), b))

        if blink_val > 0.35:
            out["eyeSquintLeft"] *= 0.25
            out["eyeSquintRight"] *= 0.25
            out["eyeWideLeft"] *= 0.15
            out["eyeWideRight"] *= 0.15

        return self._filter(out)

    def on_sentence_end(self, ctx: FaceContext) -> Dict[str, float]:
        return self._filter({k: 0.0 for k in self.owned_keys})
