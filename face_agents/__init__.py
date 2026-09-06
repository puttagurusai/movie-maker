"""
face_agents — multi-agent runtime for realistic talking-face expression.

  lips_agent   → wav2arkit mouth / jaw
  eyes_agent   → blink, gaze, Brain eye micro-expression
  brows_agent  → emotion_map + Brain encoder + speech energy
  cheeks_agent → smile / frown / nose + Brain
  head_agent   → pitch / yaw / roll
  brain_track  → bake Brain upper-face track once per sentence

Entry: python orchestrator_agents.py
"""

from .base import FaceContext, BaseFaceAgent
from .lips_agent import LipsAgent
from .eyes_agent import EyesAgent
from .brows_agent import BrowsAgent
from .cheeks_agent import CheeksAgent
from .head_agent import HeadAgent
from .micro_expression_agent import MicroExpressionAgent
from .expression_tokens import (
    ExpressionEvent,
    TextSegment,
    parse_tokens,
    evaluate_tokens,
    strip_tokens,
    mix_token_sounds,
    split_token_segments,
    resolve_sound_path,
)
from .feedback_logger import FeedbackLogger


def __getattr__(name: str):
    # Lazy load coordinator (bytecode-backed after recovery)
    if name == "FaceCoordinator":
        from .coordinator import FaceCoordinator
        return FaceCoordinator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FaceContext",
    "BaseFaceAgent",
    "FaceCoordinator",
    "LipsAgent",
    "EyesAgent",
    "BrowsAgent",
    "CheeksAgent",
    "HeadAgent",
    "MicroExpressionAgent",
    "ExpressionEvent",
    "TextSegment",
    "parse_tokens",
    "evaluate_tokens",
    "strip_tokens",
    "mix_token_sounds",
    "split_token_segments",
    "resolve_sound_path",
    "FeedbackLogger",
]
