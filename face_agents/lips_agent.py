"""
lips_agent.py — speech mouth / jaw only (wav2arkit).

Does NOT own smile/frown (cheeks_agent owns expression corners) so emotion
and lip-sync do not fight.
"""

from __future__ import annotations

from typing import Dict

from .base import BaseFaceAgent, FaceContext

try:
    import wav2arkit
except ImportError:
    wav2arkit = None  # type: ignore


# Speech-only keys — no smile/frown (expression path owns those)
SPEECH_MOUTH_KEYS = {
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "tongueOut",
    "cheekPuff",
}


class LipsAgent(BaseFaceAgent):
    name = "lips"
    owned_keys = set(SPEECH_MOUTH_KEYS)

    def __init__(self) -> None:
        self._mouth_gain = 1.15
        self._open_boost = 1.25

    def prepare(self, ctx: FaceContext) -> None:
        try:
            from .policy_bridge import get_policy
            pol = get_policy()
            self._mouth_gain = float(pol.get("mouth_gain", self._mouth_gain))
            self._open_boost = float(pol.get("mouth_open_boost", self._open_boost))
            # Push jaw peak for Faceit
            if hasattr(wav2arkit, "TARGET_JAW_PEAK"):
                peak = float(pol.get("target_jaw_peak", getattr(wav2arkit, "TARGET_JAW_PEAK", 0.18)))
                wav2arkit.TARGET_JAW_PEAK = max(0.14, min(0.35, peak))
        except Exception:
            pass

        if not ctx.audio_path or wav2arkit is None:
            print("[lips_agent] No audio/wav2arkit — lips rest")
            ctx.lips_frames = []
            return

        print(f"[lips_agent] Baking lip track: {ctx.audio_path}")
        frames, fps, raw = wav2arkit.audio_file_to_frames(
            ctx.audio_path,
            mouth_only=True,
            enhance_mouth=True,
        )
        cleaned = []
        for fr in frames:
            cleaned.append({
                k: float(v) for k, v in fr.items() if k in self.owned_keys
            })
        ctx.lips_frames = cleaned
        ctx.lips_fps = float(fps)
        jaw_i = 24
        jaw_peak = float(raw[:, jaw_i].max()) if raw is not None and raw.size else 0.0
        print(
            f"[lips_agent] {len(cleaned)} fr @ {fps:.0f}fps "
            f"jawPeak={jaw_peak:.3f} gain={self._mouth_gain:.2f} openBoost={self._open_boost:.2f}"
        )

    def sample(self, ctx: FaceContext) -> Dict[str, float]:
        if not ctx.lips_frames:
            return self._filter({"jawOpen": 0.0, "mouthClose": 0.04})

        if wav2arkit is not None:
            d = wav2arkit.frame_at_time(ctx.lips_frames, ctx.t, fps=ctx.lips_fps)
        else:
            idx = max(0, min(len(ctx.lips_frames) - 1, int(round(ctx.t * ctx.lips_fps))))
            d = dict(ctx.lips_frames[idx])

        out = {}
        openers = {
            "jawOpen", "mouthLowerDownLeft", "mouthLowerDownRight",
            "mouthUpperUpLeft", "mouthUpperUpRight",
            "mouthFunnel", "mouthPucker",
            "mouthStretchLeft", "mouthStretchRight",
            "mouthShrugLower",
        }
        for k, v in d.items():
            if k not in self.owned_keys:
                continue
            val = float(v) * self._mouth_gain
            if k in openers:
                val *= self._open_boost
            if k == "mouthClose":
                # Never seal lips while talking
                jaw = float(d.get("jawOpen", 0.0))
                val = min(val * 0.35, 0.08)
                if jaw > 0.08:
                    val = min(val, 0.04)
            out[k] = min(1.0, max(0.0, val))

        return self._filter(out)

    def on_sentence_end(self, ctx: FaceContext) -> Dict[str, float]:
        return self._filter({
            "jawOpen": 0.03,
            "mouthClose": 0.05,
            "mouthRollLower": 0.02,
            "mouthRollUpper": 0.02,
        })
