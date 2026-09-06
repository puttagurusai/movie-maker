"""
expression_mix.py — shared helpers for strong, readable, speech-linked expression.
"""

from __future__ import annotations

from typing import Dict, Iterable


def smoothstep_ease(t: float, onset: float = 0.18, duration: float = 1.0) -> float:
    """0→1 onset at speech start; soft release near end."""
    if duration <= 1e-6:
        return 0.0
    if t < 0:
        return 0.0
    # onset
    e = min(1.0, t / max(onset, 1e-3))
    e = e * e * (3.0 - 2.0 * e)
    # release last 0.28s
    if duration > 0.35 and t > duration - 0.28:
        r = max(0.0, (duration - t) / 0.28)
        e *= r * r * (3.0 - 2.0 * r)
    return float(max(0.0, min(1.0, e)))


def mix_preset_brain(
    keys: Iterable[str],
    preset: Dict[str, float],
    brain: Dict[str, float],
    *,
    ease: float,
    energy: float,
    preset_w: float = 0.72,
    brain_w: float = 0.28,
    energy_boost: float = 0.20,
) -> Dict[str, float]:
    """
    Strong preset (readable emotion) + Brain micro-motion + energy.

    Uses max-biased mix so weak Brain values cannot kill a strong happy/angry face,
    while still allowing Brain to add variation when it is higher.
    """
    preset_w = max(0.0, min(1.0, preset_w))
    brain_w = max(0.0, min(1.0, brain_w))
    s = preset_w + brain_w
    if s > 1e-6:
        preset_w, brain_w = preset_w / s, brain_w / s

    out: Dict[str, float] = {}
    e_norm = max(0.0, min(1.0, energy))
    # Primary energy multiplier (boosts on loud speech)
    e_mul = 1.0 + energy_boost * e_norm
    # Secondary: expression floor DROPS on quiet speech so expression "breathes"
    # At energy=0: floor=40% of preset  At energy=1: floor=70% of preset
    dynamic_floor = 0.40 + 0.30 * e_norm
    for k in keys:
        p = float(preset.get(k, 0.0)) * ease
        b = float(brain.get(k, 0.0)) * ease
        # weighted average
        avg = preset_w * p + brain_w * b
        # dynamic floor: varies with energy so expression pulses with speech
        val = max(avg, p * dynamic_floor)
        # allow brain peaks through
        val = max(val, b * 0.90)
        val = min(1.0, val * e_mul)
        out[k] = val
    return out


def smooth_follow(
    current: Dict[str, float],
    target: Dict[str, float],
    keys: Iterable[str],
    alpha: float = 0.35,
    energy: float = 0.5,
) -> Dict[str, float]:
    """
    Higher alpha = snappier. energy adapts alpha:
      high energy → snappier (expression attacks fast on loud speech)
      low energy  → slower  (expression lingers after quiet phonemes)
    """
    # Adaptive alpha: snappier on peaks, calmer on quiet
    e_norm = max(0.0, min(1.0, energy))
    adaptive = alpha * (0.75 + 0.50 * e_norm)  # range: alpha*0.75 to alpha*1.25
    a = max(0.05, min(0.95, adaptive))
    out = {}
    for k in keys:
        c = float(current.get(k, 0.0))
        t = float(target.get(k, 0.0))
        v = c * (1.0 - a) + t * a
        current[k] = v
        out[k] = v
    return out


def emotion_ease(
    t: float,
    duration: float,
    emotion: str,
    base_ease: float,
) -> float:
    """
    Apply emotion-specific hold curve on top of the base smoothstep_ease.

    Curve parameters come from emotion_map.HOLD_CURVES — fully data-driven,
    no hardcoded per-emotion logic. To change any emotion's hold behaviour,
    edit HOLD_CURVES in emotion_map.py.

    Curve types:
      hold    — full expression throughout (base_ease unchanged)
      pulse   — onset flash → mid_floor → optional end pulse
      decay   — full onset → decays to floor
      partial — capped at max_scale
      ramp_in — builds from start_scale to end_scale over the sentence
    """
    if duration <= 1e-6 or base_ease <= 0.0:
        return base_ease

    try:
        import emotion_map as _em
        curve = _em.get_hold_curve(emotion)
    except Exception:
        return base_ease

    ctype = curve.get("type", "hold")

    if ctype == "hold":
        return base_ease

    elif ctype == "pulse":
        onset_s       = float(curve.get("onset_s", 0.25))
        hold_onset_s  = float(curve.get("hold_after_onset_s", 0.0))  # optional hold at peak
        mid_floor     = float(curve.get("mid_floor", 0.30))
        end_win_s     = float(curve.get("end_window_s", 0.20))
        onset_dur     = min(onset_s, duration * 0.25)
        hold_end      = onset_dur + min(hold_onset_s, duration * 0.12)
        end_start     = max(hold_end, duration - end_win_s)

        if t < onset_dur:
            frac = t / max(onset_dur, 1e-4)
            frac = frac * frac * (3.0 - 2.0 * frac)  # smoothstep onset ramp
            return base_ease * frac
        elif t < hold_end:
            return base_ease  # brief hold at peak before dropping ("beat" landing)
        elif t >= end_start:
            # Jump immediately to full ease at end pulse so smooth_follow has
            # maximum lead time before the smoothstep base_ease release.
            return base_ease
        else:
            return base_ease * mid_floor

    elif ctype == "decay":
        peak_s      = float(curve.get("peak_s", 0.35))
        floor_      = float(curve.get("floor", 0.55))
        decay_ratio = float(curve.get("decay_ratio", 0.45))
        peak_t = min(peak_s, duration * 0.25)
        if t <= peak_t:
            return base_ease
        fade_frac = min(1.0, (t - peak_t) / max(0.5, duration - peak_t))
        scale = 1.0 - decay_ratio * fade_frac
        return base_ease * max(floor_, scale)

    elif ctype == "partial":
        max_scale = float(curve.get("max_scale", 0.80))
        return base_ease * max_scale

    elif ctype == "ramp_in":
        start = float(curve.get("start_scale", 0.50))
        end   = float(curve.get("end_scale",   1.00))
        frac  = min(1.0, t / max(duration, 1e-4))
        frac  = frac * frac * (3.0 - 2.0 * frac)  # smoothstep
        scale = start + (end - start) * frac
        return base_ease * scale

    else:
        return base_ease


def blend_emotions(
    prev: Dict[str, float],
    curr: Dict[str, float],
    t: float,
    blend_dur: float = 0.25,
) -> Dict[str, float]:
    """Smooth blend from prev emotion to curr over blend_dur seconds."""
    if blend_dur <= 1e-6 or t >= blend_dur:
        return dict(curr)
    frac = t / blend_dur
    ease = frac * frac * (3.0 - 2.0 * frac)  # smoothstep
    keys = set(prev) | set(curr)
    return {
        k: float(prev.get(k, 0.0)) * (1.0 - ease) + float(curr.get(k, 0.0)) * ease
        for k in keys
    }
