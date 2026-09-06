"""
Read FACE_POLICY from llm_fw (if available) for runtime agents.
Also merges SET key=value overrides from agents.txt.
Safe if llm_fw or agents.txt are not used — returns defaults.
"""

from __future__ import annotations

from typing import Any, Dict


# Defaults from face-only triple test (wav2arkit + A2E + Brain):
#   lipsync → wav2arkit; emotion → A2E 26d + Brain upper mixed with emotion_map
#   See temp/face_triple_report.json (mean lip ~0.82, emotion signatures pass)
_DEFAULTS: Dict[str, Any] = {
    "emotion": "neutral",
    "intensity": 0.7,
    "mouth_gain": 1.15,
    "mouth_open_boost": 1.25,
    "target_jaw_peak": 0.22,
    # Mouth samples this many seconds BEHIND audio (not ahead)
    "face_lip_hold_s": 0.14,
    "face_time_lead_s": 0.0,
    "blink_scale": 1.0,
    "gaze_scale": 1.0,
    "brow_scale": 1.15,
    "expr_scale": 1.20,
    # Balanced preset + Brain (was preset-heavy 0.75 / brain 0.22–0.35)
    "expr_preset_weight": 0.60,
    "brain_brow_weight": 0.40,
    "brain_cheek_weight": 0.38,
    "brain_eye_weight": 0.40,
    # MicroExpressionAgent params
    "micro_intensity": 0.40,
    "micro_rate": 0.25,
    "micro_energy_threshold": 0.65,
    # Explicit emotion override for brows_agent (empty = use ctx.emotion from JSON)
    "emotion_override": "",
}


def get_policy() -> Dict[str, Any]:
    out = dict(_DEFAULTS)

    # Layer 1: llm_fw FACE_POLICY (if running run_llm_agents.py)
    try:
        from llm_fw.tools.face_tools import get_face_policy
        out.update(get_face_policy())
    except Exception:
        pass

    # Layer 2: SET key=value overrides written to agents.txt
    try:
        from .agent_comm import get_policy_overrides
        out.update(get_policy_overrides())
    except Exception:
        pass

    return out
