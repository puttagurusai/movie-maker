"""
Continuity Memory Board — EXTERNAL memory for memory-less LLMs.

LLM has no chat memory across calls. Every call gets this board injected.
After each director/bake step we WRITE back. Prefer one Director plan call.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


CAMERA_SIZES = ("ECU", "CU", "MCU", "MS", "MLS", "WS")
CAMERA_MOVES = (
    "static", "dolly_in", "dolly_out",
    "truck", "truck_left", "truck_right",
    "crane_up", "crane_down",
    "pan_left", "pan_right",
    "orbit", "arc",
    "tilt_up", "tilt_down",
    "reveal", "follow", "handheld",
)
CAMERA_ROLES = ("A_cam", "B_cam", "C_cam", "env_cam")
CLIP_POLICIES = ("loop", "hold_end", "stretch", "momask_match")
PACES = ("slow", "medium", "fast")


@dataclass
class ShotMemory:
    shot_id: str
    summary: str = ""
    text: str = ""
    emotion: str = "neutral"
    intensity: float = 0.75
    state: str = "standing"
    body_mode: str = "catalog"
    actions: List[str] = field(default_factory=list)
    humanml_prompt: str = ""
    camera_shot: str = "MS"
    camera_move: str = "static"
    camera_role: str = "A_cam"
    transition_in: str = "cut"
    target_duration_s: float = 0.0
    hold_before_s: float = 0.0
    hold_after_s: float = 0.0
    pace: str = "medium"
    clip_policy: str = "hold_end"
    director_notes: str = ""
    actual_duration_s: float = 0.0
    t0: float = 0.0
    t1: float = 0.0
    bake_ok: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShotMemory":
        return cls(**{
            k: d.get(k, getattr(cls, k).default if hasattr(getattr(cls, k, None), "default") else
                     ("" if k in ("summary", "text", "emotion", "state", "body_mode", "humanml_prompt",
                                  "camera_shot", "camera_move", "camera_role", "transition_in",
                                  "pace", "clip_policy", "director_notes") else
                      (0.0 if k.endswith("_s") or k in ("t0", "t1", "intensity") else
                       ([] if k == "actions" else None if k == "bake_ok" else d.get(k)))))
            for k in (
                "shot_id", "summary", "text", "emotion", "intensity", "state", "body_mode",
                "actions", "humanml_prompt", "camera_shot", "camera_move", "camera_role",
                "transition_in", "target_duration_s", "hold_before_s", "hold_after_s",
                "pace", "clip_policy", "director_notes", "actual_duration_s", "t0", "t1", "bake_ok",
            )
        }) if False else cls(
            shot_id=str(d.get("shot_id") or ""),
            summary=str(d.get("summary") or ""),
            text=str(d.get("text") or ""),
            emotion=str(d.get("emotion") or "neutral"),
            intensity=float(d.get("intensity") or 0.75),
            state=str(d.get("state") or "standing"),
            body_mode=str(d.get("body_mode") or "catalog"),
            actions=list(d.get("actions") or []),
            humanml_prompt=str(d.get("humanml_prompt") or ""),
            camera_shot=str(d.get("camera_shot") or "MS"),
            camera_move=str(d.get("camera_move") or "static"),
            camera_role=str(d.get("camera_role") or "A_cam"),
            transition_in=str(d.get("transition_in") or "cut"),
            target_duration_s=float(d.get("target_duration_s") or 0.0),
            hold_before_s=float(d.get("hold_before_s") or 0.0),
            hold_after_s=float(d.get("hold_after_s") or 0.0),
            pace=str(d.get("pace") or "medium"),
            clip_policy=str(d.get("clip_policy") or "hold_end"),
            director_notes=str(d.get("director_notes") or ""),
            actual_duration_s=float(d.get("actual_duration_s") or 0.0),
            t0=float(d.get("t0") or 0.0),
            t1=float(d.get("t1") or 0.0),
            bake_ok=d.get("bake_ok"),
        )


@dataclass
class ContinuityBoard:
    story: str = ""
    title: str = ""
    style: str = "cinematic"
    target_total_s: float = 60.0
    location: str = "default"
    character_state: str = "standing"
    character_emotion: str = "neutral"
    character_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    last_camera_shot: str = "MS"
    last_camera_move: str = "static"
    last_camera_role: str = "A_cam"
    cameras_created: List[str] = field(default_factory=list)
    cameras_used: List[str] = field(default_factory=list)
    moves_used: List[str] = field(default_factory=list)
    roles_used: List[str] = field(default_factory=list)
    emotional_arc: List[str] = field(default_factory=list)
    shots: List[ShotMemory] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    director_source: str = "rules"
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "story": self.story,
            "title": self.title,
            "style": self.style,
            "target_total_s": self.target_total_s,
            "location": self.location,
            "character_state": self.character_state,
            "character_emotion": self.character_emotion,
            "character_position": list(self.character_position),
            "last_camera_shot": self.last_camera_shot,
            "last_camera_move": self.last_camera_move,
            "last_camera_role": self.last_camera_role,
            "cameras_created": list(self.cameras_created),
            "cameras_used": list(self.cameras_used),
            "moves_used": list(self.moves_used),
            "roles_used": list(self.roles_used),
            "emotional_arc": list(self.emotional_arc),
            "shots": [s.to_dict() for s in self.shots],
            "notes": list(self.notes),
            "director_source": self.director_source,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContinuityBoard":
        shots = [ShotMemory.from_dict(s) for s in (d.get("shots") or []) if isinstance(s, dict)]
        return cls(
            story=str(d.get("story") or ""),
            title=str(d.get("title") or ""),
            style=str(d.get("style") or "cinematic"),
            target_total_s=float(d.get("target_total_s") or 60.0),
            location=str(d.get("location") or "default"),
            character_state=str(d.get("character_state") or "standing"),
            character_emotion=str(d.get("character_emotion") or "neutral"),
            character_position=list(d.get("character_position") or [0, 0, 0]),
            last_camera_shot=str(d.get("last_camera_shot") or "MS"),
            last_camera_move=str(d.get("last_camera_move") or "static"),
            last_camera_role=str(d.get("last_camera_role") or "A_cam"),
            cameras_created=list(d.get("cameras_created") or []),
            cameras_used=list(d.get("cameras_used") or []),
            moves_used=list(d.get("moves_used") or []),
            roles_used=list(d.get("roles_used") or []),
            emotional_arc=list(d.get("emotional_arc") or []),
            shots=shots,
            notes=list(d.get("notes") or []),
            director_source=str(d.get("director_source") or "rules"),
            updated_at=float(d.get("updated_at") or time.time()),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = time.time()
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "ContinuityBoard":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def record_shot_plan(self, shot: ShotMemory) -> None:
        self.shots = [s for s in self.shots if s.shot_id != shot.shot_id]
        self.shots.append(shot)
        if shot.camera_shot:
            self.last_camera_shot = shot.camera_shot
            if shot.camera_shot not in self.cameras_used:
                self.cameras_used.append(shot.camera_shot)
        if shot.camera_move:
            self.last_camera_move = shot.camera_move
            if shot.camera_move not in self.moves_used:
                self.moves_used.append(shot.camera_move)
        if shot.camera_role:
            self.last_camera_role = shot.camera_role
            if shot.camera_role not in self.roles_used:
                self.roles_used.append(shot.camera_role)
            if shot.camera_role not in self.cameras_created:
                self.cameras_created.append(shot.camera_role)
        if shot.emotion:
            self.character_emotion = shot.emotion
            self.emotional_arc.append(shot.emotion)
        if shot.state:
            self.character_state = shot.state
        self.updated_at = time.time()

    def record_bake(
        self,
        shot_id: str,
        *,
        actual_duration_s: float,
        t0: float,
        t1: float,
        bake_ok: bool,
        state: Optional[str] = None,
    ) -> None:
        for s in self.shots:
            if s.shot_id == shot_id:
                s.actual_duration_s = float(actual_duration_s)
                s.t0 = float(t0)
                s.t1 = float(t1)
                s.bake_ok = bool(bake_ok)
                if state:
                    s.state = state
                    self.character_state = state
                break
        self.updated_at = time.time()

    def prompt_pack(self, *, max_prior_shots: int = 16) -> str:
        prior = self.shots[-max_prior_shots:]
        prior_lines = []
        for s in prior:
            prior_lines.append(
                f"- {s.shot_id}: emo={s.emotion} state={s.state} "
                f"role={s.camera_role} cam={s.camera_shot}/{s.camera_move} "
                f"target_dur={s.target_duration_s:.1f}s pace={s.pace} "
                f"body={s.body_mode} clip={s.clip_policy} | {(s.summary or s.text)[:70]}"
            )
        pack = {
            "MEMORY_NOTE": (
                "This model has NO conversation memory. "
                "CONTINUITY_BOARD is the only truth about prior decisions. "
                "Respect last_camera, character_state, emotional_arc. "
                "Prefer cinematic longer takes (6–14s), not 3s reels cuts."
            ),
            "title": self.title,
            "style": self.style,
            "target_total_s": self.target_total_s,
            "location": self.location,
            "character_state": self.character_state,
            "character_emotion": self.character_emotion,
            "character_position": self.character_position,
            "last_camera_shot": self.last_camera_shot,
            "last_camera_move": self.last_camera_move,
            "last_camera_role": self.last_camera_role,
            "cameras_created": self.cameras_created,
            "cameras_used": self.cameras_used,
            "moves_used": self.moves_used,
            "roles_used": self.roles_used,
            "emotional_arc": self.emotional_arc[-16:],
            "planned_shots_so_far": prior_lines,
            "director_notes": self.notes[-8:],
            "story": (self.story or "")[:2500],
            "camera_sizes_allowed": list(CAMERA_SIZES),
            "camera_moves_allowed": list(CAMERA_MOVES),
            "camera_roles_allowed": list(CAMERA_ROLES),
            "clip_policies_allowed": list(CLIP_POLICIES),
            "paces_allowed": list(PACES),
        }
        return json.dumps(pack, indent=2, ensure_ascii=False)
