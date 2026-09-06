"""
cast_schema.py — Hero + extras cast (typed menus, not freeform bpy).

Hero: ARKit face + Session body timeline.
Extras: body-only NPCs (no speech / no blendshape requirement in v1).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


FACE_MODES = frozenset({"arkit", "none", "bones"})
ACTOR_ROLES = frozenset({"hero", "extra"})
EXTRA_MOTIONS = frozenset({"idle", "walk_loop", "sit"})
EXTRAS_PRESETS = frozenset({"none", "sidewalk", "park_path", "plaza"})

# Outfit presets = Blender collections Wardrobe_<id> (hide/show).
WARDROBE_IDS = frozenset({
    "hero_default",
    "casual_01",
    "formal_01",
    "hoodie_01",
    "worker_01",
    "npc_variant_a",
    "npc_variant_b",
    "npc_variant_c",
})

WARDROBE_ALIASES = {
    "default": "hero_default",
    "hero": "hero_default",
    "base": "hero_default",
    "naked": "hero_default",
    "casual": "casual_01",
    "tee": "casual_01",
    "tshirt": "casual_01",
    "jeans": "casual_01",
    "streetwear": "casual_01",
    "formal": "formal_01",
    "suit": "formal_01",
    "business": "formal_01",
    "blazer": "formal_01",
    "hoodie": "hoodie_01",
    "hooded": "hoodie_01",
    "worker": "worker_01",
    "work": "worker_01",
    "uniform": "worker_01",
}


def normalize_wardrobe_id(raw: str) -> str:
    s = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not s:
        return "hero_default"
    if s in WARDROBE_IDS:
        return s
    if s in WARDROBE_ALIASES:
        return WARDROBE_ALIASES[s]
    for wid in WARDROBE_IDS:
        if wid in s or s in wid:
            return wid
    return "hero_default"


def normalize_extras_preset(raw: str) -> str:
    s = str(raw or "none").strip().lower().replace(" ", "_").replace("-", "_")
    if s in ("crowd", "people", "side_walk"):
        s = "sidewalk"
    if s in ("park", "path", "garden"):
        s = "park_path"
    if s in ("square", "plaza_crowd"):
        s = "plaza"
    if s not in EXTRAS_PRESETS:
        return "none"
    return s


def clamp_extras_count(n: Any) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = 0
    return max(0, min(8, v))


@dataclass
class CastActor:
    id: str = "hero"
    role: str = "hero"  # hero | extra
    body_object: str = ""
    face_mesh: str = ""
    face_mode: str = "arkit"  # arkit | none | bones
    outfit_id: str = "hero_default"
    spawn: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x,y,z,yaw
    motion: str = "idle"  # extras: idle | walk_loop | sit

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["spawn"] = list(self.spawn)
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "CastActor":
        raw = dict(d or {})
        role = str(raw.get("role") or "extra").strip().lower()
        if role not in ACTOR_ROLES:
            role = "extra"
        face_mode = str(raw.get("face_mode") or ("arkit" if role == "hero" else "none")).strip().lower()
        if face_mode not in FACE_MODES:
            face_mode = "arkit" if role == "hero" else "none"
        motion = str(raw.get("motion") or "idle").strip().lower()
        if motion not in EXTRA_MOTIONS:
            motion = "idle"
        sp = raw.get("spawn") or [0.0, 0.0, 0.0, 0.0]
        try:
            spawn = (
                float(sp[0]), float(sp[1]), float(sp[2]),
                float(sp[3]) if len(sp) > 3 else 0.0,
            )
        except Exception:
            spawn = (0.0, 0.0, 0.0, 0.0)
        aid = str(raw.get("id") or ("hero" if role == "hero" else "npc")).strip() or "npc"
        return cls(
            id=aid,
            role=role,
            body_object=str(raw.get("body_object") or ""),
            face_mesh=str(raw.get("face_mesh") or ""),
            face_mode=face_mode,
            outfit_id=normalize_wardrobe_id(str(raw.get("outfit_id") or "hero_default")),
            spawn=spawn,
            motion=motion,
        )

    @property
    def speech_ready(self) -> bool:
        return self.role == "hero" and self.face_mode == "arkit"


@dataclass
class CastState:
    """Session cast: one hero + zero or more extras."""

    hero: CastActor = field(default_factory=lambda: CastActor(
        id="hero", role="hero", face_mode="arkit", outfit_id="hero_default",
    ))
    extras: List[CastActor] = field(default_factory=list)
    extras_preset: str = "none"
    extras_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hero": self.hero.to_dict(),
            "extras": [e.to_dict() for e in self.extras],
            "extras_preset": normalize_extras_preset(self.extras_preset),
            "extras_count": clamp_extras_count(self.extras_count),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "CastState":
        raw = dict(d or {})
        hero_raw = raw.get("hero") if isinstance(raw.get("hero"), dict) else {}
        hero = CastActor.from_dict({**{"id": "hero", "role": "hero", "face_mode": "arkit"}, **hero_raw})
        hero.role = "hero"
        extras = [
            CastActor.from_dict(e)
            for e in (raw.get("extras") or [])
            if isinstance(e, dict)
        ]
        for i, e in enumerate(extras):
            e.role = "extra"
            if not e.id or e.id == "hero":
                e.id = f"npc_{i + 1}"
            if e.face_mode == "arkit":
                e.face_mode = "none"  # v1: extras never speech-ready
        return cls(
            hero=hero,
            extras=extras,
            extras_preset=normalize_extras_preset(str(raw.get("extras_preset") or "none")),
            extras_count=clamp_extras_count(raw.get("extras_count", len(extras))),
        )

    def clear_extras(self) -> None:
        self.extras.clear()
        self.extras_count = 0
        self.extras_preset = "none"

    def set_hero_outfit(self, outfit_id: str) -> str:
        wid = normalize_wardrobe_id(outfit_id)
        self.hero.outfit_id = wid
        return wid
