"""
Shot / beat schema for the movie production package.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ..look_schema import LookPlan


@dataclass
class ShotPlan:
    """Stage 1 high-level shot (before detail expansion)."""

    shot_id: str
    emotional_goal: str = "neutral"
    location: str = "default"
    rough_duration_s: float = 8.0
    characters: List[str] = field(default_factory=lambda: ["hero"])
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShotDetail:
    """Stage 2 — fully specified shot (director + workers fill fields)."""

    shot_id: str
    text: str  # original combined line (stage + speech)
    spoken: str = ""  # dialogue only → TTS / lips
    stage: str = ""  # physical directions → body router
    emotion: str = "neutral"
    intensity: float = 0.75
    state: str = "standing"
    actions: List[str] = field(default_factory=list)
    body_mode: str = "auto"
    humanml_prompt: str = ""
    allow_sync_gen: bool = False
    motion_reason: str = ""
    motion_engines: List[str] = field(default_factory=list)
    camera_shot: str = ""  # MS/CU/…
    camera_move: str = ""
    camera_role: str = "A_cam"  # A_cam | B_cam | C_cam | env_cam
    transition_in: str = "cut"  # cut | crossfade
    location: str = "default"
    summary: str = ""
    # Director-owned timing (not fixed globally)
    target_duration_s: float = 0.0
    hold_before_s: float = 0.0
    hold_after_s: float = 0.0
    pace: str = "medium"  # slow | medium | fast
    clip_speed: float = 1.0  # body playback rate; <1 = slow-mo, >1 = faster
    clip_policy: str = "hold_end"  # hold_end | momask_match | loop | stretch(forbidden)
    director_notes: str = ""
    look: LookPlan = field(default_factory=LookPlan)
    # filled at resolve/bake
    t0: float = 0.0
    t1: float = 0.0
    duration_s: float = 0.0
    speech_duration_s: float = 0.0
    audio_path: str = ""
    face_timeline_path: str = ""
    body_action: str = ""
    body_library: str = ""
    camera: Optional[Dict[str, Any]] = None
    world_in: Optional[Dict[str, Any]] = None
    world_out: Optional[Dict[str, Any]] = None
    bake_ok: bool = False
    bake_errors: List[str] = field(default_factory=list)
    bake_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_director_sentence(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "spoken": self.spoken or self.text,
            "stage": self.stage,
            "emotion": self.emotion,
            "intensity": self.intensity,
            "state": self.state,
            "actions": list(self.actions),
            "body_mode": self.body_mode,
            "humanml_prompt": self.humanml_prompt,
            "allow_sync_gen": self.allow_sync_gen,
            "motion_reason": self.motion_reason,
            "motion_engines": self.motion_engines,
            "camera_shot": self.camera_shot,
            "camera_move": self.camera_move,
            "camera_role": self.camera_role,
            "target_duration_s": self.target_duration_s,
            "hold_before_s": self.hold_before_s,
            "hold_after_s": self.hold_after_s,
            "pace": self.pace,
            "clip_speed": self.clip_speed,
            "clip_policy": self.clip_policy,
            "director_notes": self.director_notes,
            "look": self.look.to_dict() if self.look else LookPlan().to_dict(),
        }


@dataclass
class MoviePackage:
    """Full production package: plan + master timeline + validation."""

    title: str = ""
    story: str = ""
    fps: float = 20.0
    shots: List[ShotDetail] = field(default_factory=list)
    world_final: Optional[Dict[str, Any]] = None
    continuity: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    output_dir: str = ""
    director_source: str = "rules"

    @property
    def duration_s(self) -> float:
        if not self.shots:
            return 0.0
        return max(float(s.t1) for s in self.shots)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "story": self.story,
            "fps": self.fps,
            "duration_s": self.duration_s,
            "director_source": self.director_source,
            "shots": [s.to_dict() for s in self.shots],
            "world_final": self.world_final,
            "continuity": self.continuity,
            "validation": self.validation,
            "output_dir": self.output_dir,
        }
