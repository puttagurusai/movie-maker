"""
Stage 3 — World / continuity tracker (mavie.txt).

Each shot READs state at start and WRITEs updates at end so the sequence
does not teleport the character between clips.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CharacterState:
    character_id: str = "hero"
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_state: str = "standing"  # standing | sitting | walking | dancing
    holding: str = ""
    emotion: str = "neutral"
    intensity: float = 0.7
    last_action: str = ""
    body_mode: str = "catalog"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["position"] = list(self.position)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CharacterState":
        pos = d.get("position") or [0, 0, 0]
        return cls(
            character_id=str(d.get("character_id") or "hero"),
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            base_state=str(d.get("base_state") or "standing"),
            holding=str(d.get("holding") or ""),
            emotion=str(d.get("emotion") or "neutral"),
            intensity=float(d.get("intensity") or 0.7),
            last_action=str(d.get("last_action") or ""),
            body_mode=str(d.get("body_mode") or "catalog"),
        )


@dataclass
class WorldState:
    """Persistent across the whole movie sequence."""

    location: str = "default"
    time_of_day: str = "day"
    characters: Dict[str, CharacterState] = field(default_factory=dict)
    props: Dict[str, List[float]] = field(default_factory=dict)
    shot_index: int = 0
    notes: List[str] = field(default_factory=list)

    def ensure_character(self, cid: str = "hero") -> CharacterState:
        if cid not in self.characters:
            self.characters[cid] = CharacterState(character_id=cid)
        return self.characters[cid]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "time_of_day": self.time_of_day,
            "shot_index": self.shot_index,
            "characters": {k: v.to_dict() for k, v in self.characters.items()},
            "props": dict(self.props),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorldState":
        chars = {
            k: CharacterState.from_dict(v)
            for k, v in (d.get("characters") or {}).items()
        }
        return cls(
            location=str(d.get("location") or "default"),
            time_of_day=str(d.get("time_of_day") or "day"),
            characters=chars,
            props=dict(d.get("props") or {}),
            shot_index=int(d.get("shot_index") or 0),
            notes=list(d.get("notes") or []),
        )

    def apply_shot_end(
        self,
        *,
        character_id: str = "hero",
        base_state: Optional[str] = None,
        emotion: Optional[str] = None,
        intensity: Optional[float] = None,
        last_action: Optional[str] = None,
        body_mode: Optional[str] = None,
        position_delta: Optional[Tuple[float, float, float]] = None,
        note: str = "",
    ) -> None:
        """WRITE continuity after a shot finishes baking/playing."""
        ch = self.ensure_character(character_id)
        if base_state:
            ch.base_state = base_state
        if emotion:
            ch.emotion = emotion
        if intensity is not None:
            ch.intensity = float(intensity)
        if last_action:
            ch.last_action = last_action
        if body_mode:
            ch.body_mode = body_mode
        if position_delta:
            x, y, z = ch.position
            dx, dy, dz = position_delta
            ch.position = (x + dx, y + dy, z + dz)
        self.shot_index += 1
        if note:
            self.notes.append(note)
