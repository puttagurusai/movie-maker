"""
movie_timeline.py — shared clock for audio / face / body / camera.

One beat (sentence) becomes a MovieBeat with optional camera plan.
Full MovieTimeline can hold multiple beats for multi-line clips.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .camera_agent import CameraPlan, plan_camera_for_beat


@dataclass
class MovieBeat:
    id: str
    t0: float
    t1: float
    text: str
    emotion: str = "neutral"
    intensity: float = 0.7
    state: str = "standing"
    actions: List[str] = field(default_factory=list)
    body_mode: str = "catalog"
    humanml_prompt: str = ""
    audio_path: str = ""
    camera: Optional[Dict[str, Any]] = None

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.t1) - float(self.t0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "t0": self.t0,
            "t1": self.t1,
            "duration_s": self.duration_s,
            "text": self.text,
            "emotion": self.emotion,
            "intensity": self.intensity,
            "state": self.state,
            "actions": list(self.actions),
            "body_mode": self.body_mode,
            "humanml_prompt": self.humanml_prompt,
            "audio_path": self.audio_path,
            "camera": self.camera,
        }


@dataclass
class MovieTimeline:
    fps: float = 30.0
    beats: List[MovieBeat] = field(default_factory=list)
    title: str = ""

    @property
    def duration_s(self) -> float:
        if not self.beats:
            return 0.0
        return max(b.t1 for b in self.beats)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "beats": [b.to_dict() for b in self.beats],
        }

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p


def build_beat_with_camera(
    *,
    beat_id: str,
    t0: float,
    duration_s: float,
    text: str,
    emotion: str = "neutral",
    intensity: float = 0.7,
    state: str = "standing",
    actions: Optional[List[str]] = None,
    body_mode: str = "catalog",
    humanml_prompt: str = "",
    audio_path: str = "",
    fps: float = 30.0,
    plan_camera: bool = True,
) -> MovieBeat:
    """Create one MovieBeat and attach a camera plan."""
    acts = list(actions or [])
    t1 = float(t0) + max(0.1, float(duration_s))
    cam_dict = None
    if plan_camera:
        plan: CameraPlan = plan_camera_for_beat(
            duration_s=duration_s,
            emotion=emotion,
            intensity=intensity,
            state=state,
            actions=acts,
            fps=fps,
            humanml_prompt=humanml_prompt or "",
        )
        cam_dict = plan.to_dict()
    return MovieBeat(
        id=beat_id,
        t0=float(t0),
        t1=t1,
        text=text,
        emotion=emotion,
        intensity=float(intensity),
        state=state,
        actions=acts,
        body_mode=body_mode,
        humanml_prompt=humanml_prompt or "",
        audio_path=audio_path or "",
        camera=cam_dict,
    )
