"""
emotion_map.py

Contains EMOTION_MAP: upper-face only blendshape targets for different emotions.
Jaw and full mouth interior shapes are deliberately excluded (they are controlled by lip-sync visemes).

Only brows, eyes, eye area, cheeks, and mouth corners are used here.
"""

NEUTRAL_REST = {
    # Jaw — natural slight opening, not clenched
    "jawOpen":              0.03,
    "jawForward":           0.0,
    "jawLeft":              0.0,
    "jawRight":             0.0,

    # Mouth — natural resting lip tone (very subtle)
    "mouthClose":           0.05,
    "mouthPucker":          0.0,
    "mouthFunnel":          0.0,
    "mouthRollLower":       0.02,
    "mouthRollUpper":       0.02,
    "mouthShrugLower":      0.0,
    "mouthShrugUpper":      0.0,
    "mouthSmileLeft":       0.0,
    "mouthSmileRight":      0.0,
    "mouthFrownLeft":       0.0,
    "mouthFrownRight":      0.0,
    "mouthDimpleLeft":      0.0,
    "mouthDimpleRight":     0.0,
    "mouthUpperUpLeft":     0.0,
    "mouthUpperUpRight":    0.0,
    "mouthLowerDownLeft":   0.0,
    "mouthLowerDownRight":  0.0,
    "mouthLeft":            0.0,
    "mouthRight":           0.0,
    "mouthStretchLeft":     0.0,
    "mouthStretchRight":    0.0,
    "mouthPressLeft":       0.0,
    "mouthPressRight":      0.0,

    # Brows — completely neutral
    "browDownLeft":         0.0,
    "browDownRight":        0.0,
    "browInnerUp":          0.0,
    "browOuterUpLeft":      0.0,
    "browOuterUpRight":     0.0,

    # Eyes — open but with tiny natural squint (not wide, not fully relaxed)
    "eyeBlinkLeft":         0.0,
    "eyeBlinkRight":        0.0,
    "eyeSquintLeft":        0.05,
    "eyeSquintRight":       0.05,
    "eyeWideLeft":          0.0,
    "eyeWideRight":         0.0,

    # Cheeks
    "cheekPuff":            0.0,
    "cheekSquintLeft":      0.0,
    "cheekSquintRight":     0.0,

    # Nose
    "noseSneerLeft":        0.0,
    "noseSneerRight":       0.0,

    # Tongue
    "tongueOut":            0.0,
}

EMOTION_MAP = {
    "neutral": NEUTRAL_REST.copy(),   # Use the natural rest pose as baseline

    "happy": {
        "browInnerUp": 0.15,
        "browOuterUpLeft": 0.35,
        "browOuterUpRight": 0.35,
        "eyeSquintLeft": 0.6,
        "eyeSquintRight": 0.6,
        "eyeBlinkLeft": 0.1,
        "eyeBlinkRight": 0.1,
        "cheekSquintLeft": 0.7,
        "cheekSquintRight": 0.7,
        "mouthSmileLeft": 0.85,
        "mouthSmileRight": 0.85,
    },

    "sad": {
        # browInnerUp high = inner brows raise — the "sad puppy" look.
        # browDown on outer = angled brow (outer down, inner up = sad arch).
        "browDownLeft": 0.30,
        "browDownRight": 0.30,
        "browInnerUp": 0.80,
        "eyeSquintLeft": 0.35,
        "eyeSquintRight": 0.35,
        "eyeBlinkLeft": 0.20,
        "eyeBlinkRight": 0.20,
        "mouthFrownLeft": 0.90,
        "mouthFrownRight": 0.90,
        "mouthPressLeft": 0.30,
        "mouthPressRight": 0.30,
    },

    "angry": {
        # browInnerUp intentionally REMOVED — it fights brow convergence.
        # browDownLeft/Right at max pulls brows down AND medially (inward).
        "browDownLeft": 0.95,
        "browDownRight": 0.95,
        "browInnerUp": 0.0,
        "eyeSquintLeft": 0.70,
        "eyeSquintRight": 0.70,
        "cheekSquintLeft": 0.25,
        "cheekSquintRight": 0.25,
        "mouthFrownLeft": 0.35,
        "mouthFrownRight": 0.35,
        "noseSneerLeft": 0.50,
        "noseSneerRight": 0.50,
    },

    "surprised": {
        "browInnerUp": 0.95,
        "browOuterUpLeft": 0.7,
        "browOuterUpRight": 0.7,
        "eyeWideLeft": 0.92,
        "eyeWideRight": 0.92,
        "eyeBlinkLeft": 0.0,
        "eyeBlinkRight": 0.0,
        # slight open mouth for "oh" — lips agent still owns speech jaw
        "mouthSmileLeft": 0.12,
        "mouthSmileRight": 0.12,
        "mouthUpperUpLeft": 0.25,
        "mouthUpperUpRight": 0.25,
        "cheekSquintLeft": 0.1,
        "cheekSquintRight": 0.1,
    },

    "disgusted": {
        "browDownLeft": 0.5,
        "browDownRight": 0.5,
        "eyeSquintLeft": 0.65,
        "eyeSquintRight": 0.65,
        "cheekSquintLeft": 0.5,
        "cheekSquintRight": 0.48,
        "mouthFrownLeft": 0.6,
        "mouthFrownRight": 0.6,
        "mouthUpperUpLeft": 0.45,
        "mouthUpperUpRight": 0.4,
        "noseSneerLeft": 0.85,
        "noseSneerRight": 0.8,
        "mouthPressLeft": 0.2,
        "mouthPressRight": 0.2,
    },

    "fearful": {
        "browInnerUp": 0.85,
        "browOuterUpLeft": 0.4,
        "browOuterUpRight": 0.4,
        "eyeWideLeft": 0.98,
        "eyeWideRight": 0.98,
        "eyeSquintLeft": 0.12,
        "eyeSquintRight": 0.12,
        "eyeBlinkLeft": 0.05,
        "eyeBlinkRight": 0.05,
        "mouthFrownLeft": 0.25,
        "mouthFrownRight": 0.25,
        "mouthStretchLeft": 0.2,
        "mouthStretchRight": 0.2,
    },

    "sarcastic": {
        # Classic one-sided smirk: left brow cocked, right squinted skeptically
        "browDownRight": 0.35,
        "browOuterUpLeft": 0.40,  # left brow raised (skeptical arch)
        "browInnerUp": 0.12,
        "eyeSquintLeft": 0.35,
        "eyeSquintRight": 0.60,   # right eye squinted more (skeptical)
        "mouthSmileLeft": 0.50,   # left side smirk
        "mouthSmileRight": 0.10,  # right side stays down
        "mouthFrownRight": 0.30,  # right corner down (half-frown, half-smirk)
        "cheekSquintLeft": 0.25,
    },

    "thinking": {
        "browDownLeft": 0.55,
        "browDownRight": 0.2,
        "browInnerUp": 0.45,
        "browOuterUpLeft": 0.2,
        "eyeSquintLeft": 0.45,
        "eyeSquintRight": 0.2,
        "eyeLookDownLeft": 0.15,
        "eyeLookDownRight": 0.15,
        "mouthSmileLeft": 0.06,
        "mouthSmileRight": 0.02,
        "mouthPressLeft": 0.12,
        "mouthPressRight": 0.08,
    },

    # ---- Extended emotions (previously fell back to neutral) ----

    "calm": {
        "browOuterUpLeft": 0.12,
        "browOuterUpRight": 0.12,
        "eyeSquintLeft": 0.08,
        "eyeSquintRight": 0.08,
        "mouthSmileLeft": 0.05,
        "mouthSmileRight": 0.05,
    },

    "apologetic": {
        "browInnerUp": 0.70,
        "browDownLeft": 0.20,
        "browDownRight": 0.20,
        "eyeSquintLeft": 0.22,
        "eyeSquintRight": 0.22,
        "mouthFrownLeft": 0.30,
        "mouthFrownRight": 0.30,
        "mouthPressLeft": 0.15,
        "mouthPressRight": 0.15,
    },

    "assertive": {
        "browDownLeft": 0.50,
        "browDownRight": 0.50,
        "browInnerUp": 0.10,
        "eyeSquintLeft": 0.45,
        "eyeSquintRight": 0.45,
        "mouthPressLeft": 0.20,
        "mouthPressRight": 0.20,
    },

    "concerned": {
        "browInnerUp": 0.60,
        "browDownLeft": 0.30,
        "browDownRight": 0.30,
        "eyeSquintLeft": 0.30,
        "eyeSquintRight": 0.30,
        "mouthFrownLeft": 0.20,
        "mouthFrownRight": 0.20,
    },

    "encouraging": {
        "browOuterUpLeft": 0.40,
        "browOuterUpRight": 0.40,
        "browInnerUp": 0.15,
        "eyeSquintLeft": 0.35,
        "eyeSquintRight": 0.35,
        "cheekSquintLeft": 0.45,
        "cheekSquintRight": 0.45,
        "mouthSmileLeft": 0.60,
        "mouthSmileRight": 0.60,
    },
}

UPPER_FACE_KEYS = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekSquintLeft", "cheekSquintRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthPressLeft", "mouthPressRight",
    "noseSneerLeft", "noseSneerRight",
]


# ─────────────────────────────────────────────────────────────────────────────
# HOLD_CURVES — how each emotion's expression evolves DURING a sentence.
#
# Types:
#   "hold"    — maintain full expression throughout (default for most emotions)
#   "pulse"   — flash at onset, drop to mid_floor, optionally pulse at end
#   "decay"   — full at onset, then exponential decay to floor
#   "partial" — never reaches full; capped at max_scale
#   "ramp_in" — starts low, builds over the sentence (excitement, building anger)
#
# All timing fields are in seconds; ratios are relative to sentence duration.
# ─────────────────────────────────────────────────────────────────────────────
HOLD_CURVES: dict = {
    # Transient reactions: smirk/disgust only at onset, barely hold mid-sentence
    "sarcastic":   {"type": "pulse",   "onset_s": 0.28, "hold_after_onset_s": 0.15, "mid_floor": 0.28, "end_window_s": 0.50},
    "disgusted":   {"type": "decay",   "peak_s":  0.30, "floor": 0.45,  "decay_ratio": 0.55},
    "surprised":   {"type": "decay",   "peak_s":  0.35, "floor": 0.55,  "decay_ratio": 0.45},
    # Contemplative: partial cap (never reaches full intensity)
    "thinking":    {"type": "partial", "max_scale": 0.80},
    "confused":    {"type": "partial", "max_scale": 0.75},
    # Builds over the sentence (enthusiasm / conviction growing)
    "excited":     {"type": "ramp_in", "start_scale": 0.55, "end_scale": 1.0},
    "assertive":   {"type": "ramp_in", "start_scale": 0.60, "end_scale": 1.0},
    # Hold full throughout — these emotions sustain correctly
    "happy":       {"type": "hold"},
    "sad":         {"type": "hold"},
    "angry":       {"type": "hold"},
    "fearful":     {"type": "hold"},
    "neutral":     {"type": "hold"},
    "calm":        {"type": "hold"},
    "encouraging": {"type": "hold"},
    "apologetic":  {"type": "hold"},
    "concerned":   {"type": "hold"},
    "playful":     {"type": "hold"},
    "empathetic":  {"type": "hold"},
    "bored":       {"type": "partial", "max_scale": 0.70},
}


def get_hold_curve(emotion: str) -> dict:
    """Return the hold-curve parameters for an emotion (defaults to 'hold' type)."""
    emo = (emotion or "neutral").lower().strip()
    return HOLD_CURVES.get(emo, {"type": "hold"})


def get_blendshapes(emotion: str, intensity: float = 1.0) -> dict:
    """
    Returns the blendshape values for the given emotion, scaled by intensity.
    Falls back to neutral if emotion is unknown.
    Only upper-face keys are present.
    For "neutral" we always return the full natural rest pose (intensity is ignored).
    """
    emotion = (emotion or "neutral").lower().strip()
    if emotion == "neutral":
        return NEUTRAL_REST.copy()

    base = EMOTION_MAP.get(emotion, EMOTION_MAP["neutral"])
    if intensity <= 0:
        intensity = 0.0
    return {k: round(v * intensity, 4) for k, v in base.items()}


if __name__ == "__main__":
    # Quick self-test
    print("Available emotions:", list(EMOTION_MAP.keys()))
    print("Happy @ 0.8:", get_blendshapes("happy", 0.8))
    print("Unknown -> neutral:", get_blendshapes("excited", 1.0))
