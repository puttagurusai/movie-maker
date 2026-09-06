"""
Face control tools — update shared FacePolicy used by runtime agents.

LLMs do not drive 30fps mesh directly; they set policy parameters.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import Tool, ToolRegistry, ToolResult


# Shared mutable policy (process-wide)
FACE_POLICY: Dict[str, Any] = {
    "emotion": "neutral",
    "intensity": 0.7,
    "mouth_gain": 1.0,          # multiplies lip opener strength (relative)
    "target_jaw_peak": 0.15,  # match wav2arkit default; LLM can raise via set_lips_params
    "face_time_lead_s": 0.04,
    "blink_scale": 1.0,
    "gaze_scale": 1.0,
    "brow_scale": 1.0,
    "notes": "",
}


def get_face_policy() -> Dict[str, Any]:
    return dict(FACE_POLICY)


def register_face_tools(reg: ToolRegistry) -> None:
    def get_policy() -> ToolResult:
        return ToolResult(ok=True, data=get_face_policy())

    def set_emotion(emotion: str, intensity: float = 0.7) -> ToolResult:
        emotion = (emotion or "neutral").lower().strip()
        intensity = max(0.0, min(1.0, float(intensity)))
        FACE_POLICY["emotion"] = emotion
        FACE_POLICY["intensity"] = intensity
        return ToolResult(ok=True, data=get_face_policy())

    def set_lips_params(
        mouth_gain: float = None,
        target_jaw_peak: float = None,
        face_time_lead_s: float = None,
    ) -> ToolResult:
        if mouth_gain is not None:
            FACE_POLICY["mouth_gain"] = max(0.2, min(3.0, float(mouth_gain)))
        if target_jaw_peak is not None:
            FACE_POLICY["target_jaw_peak"] = max(0.3, min(1.0, float(target_jaw_peak)))
        if face_time_lead_s is not None:
            FACE_POLICY["face_time_lead_s"] = max(0.0, min(0.2, float(face_time_lead_s)))
        return ToolResult(ok=True, data=get_face_policy())

    def set_eyes_params(blink_scale: float = None, gaze_scale: float = None) -> ToolResult:
        if blink_scale is not None:
            FACE_POLICY["blink_scale"] = max(0.2, min(2.0, float(blink_scale)))
        if gaze_scale is not None:
            FACE_POLICY["gaze_scale"] = max(0.0, min(2.0, float(gaze_scale)))
        return ToolResult(ok=True, data=get_face_policy())

    def set_brows_params(brow_scale: float = None) -> ToolResult:
        if brow_scale is not None:
            FACE_POLICY["brow_scale"] = max(0.2, min(2.0, float(brow_scale)))
        return ToolResult(ok=True, data=get_face_policy())

    def set_notes(notes: str) -> ToolResult:
        FACE_POLICY["notes"] = str(notes)[:500]
        return ToolResult(ok=True, data={"notes": FACE_POLICY["notes"]})

    reg.register(Tool(
        name="get_face_policy",
        description="Read current face control policy (emotion, mouth, eyes, brows).",
        parameters={"type": "object", "properties": {}},
        handler=get_policy,
        allowed_agents=["director", "lips", "eyes", "brows"],
    ))
    reg.register(Tool(
        name="set_emotion",
        description="Set speaking emotion and intensity 0-1.",
        parameters={
            "type": "object",
            "properties": {
                "emotion": {"type": "string"},
                "intensity": {"type": "number"},
            },
            "required": ["emotion"],
        },
        handler=set_emotion,
        allowed_agents=["director", "brows", "eyes"],
    ))
    reg.register(Tool(
        name="set_lips_params",
        description="Tune lips: mouth_gain, target_jaw_peak, face_time_lead_s.",
        parameters={
            "type": "object",
            "properties": {
                "mouth_gain": {"type": "number"},
                "target_jaw_peak": {"type": "number"},
                "face_time_lead_s": {"type": "number"},
            },
        },
        handler=set_lips_params,
        allowed_agents=["director", "lips"],
    ))
    reg.register(Tool(
        name="set_eyes_params",
        description="Tune eyes: blink_scale, gaze_scale.",
        parameters={
            "type": "object",
            "properties": {
                "blink_scale": {"type": "number"},
                "gaze_scale": {"type": "number"},
            },
        },
        handler=set_eyes_params,
        allowed_agents=["director", "eyes"],
    ))
    reg.register(Tool(
        name="set_brows_params",
        description="Tune brows: brow_scale.",
        parameters={
            "type": "object",
            "properties": {"brow_scale": {"type": "number"}},
        },
        handler=set_brows_params,
        allowed_agents=["director", "brows"],
    ))
    reg.register(Tool(
        name="set_notes",
        description="Store a short note about face direction for this session.",
        parameters={
            "type": "object",
            "properties": {"notes": {"type": "string"}},
            "required": ["notes"],
        },
        handler=set_notes,
        allowed_agents=["director", "lips", "eyes", "brows"],
    ))
