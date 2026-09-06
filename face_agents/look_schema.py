"""
look_schema.py — Director-owned scene / lighting plan.

Look is NEVER sent to MoMask. It drives Blender Look_Set (ground, HDRI, lights, props).
Any free-text location maps to a kit id; unknown → nearest kit (never hard-fail).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


# Canonical location kits (extend freely — Blender builds by set_style)
LOOK_LOCATIONS = frozenset({
    "studio",
    "interior_room",
    "corridor",
    "street",
    "park",
    "forest",
    "playground",
    "station",
    "beach",
    "stage",
    "default",
})
LOOK_TIMES = frozenset({
    "dawn", "day", "golden_hour", "dusk", "night", "overcast",
})
LOOK_MOODS = frozenset({
    "neutral", "soft", "hard", "warm", "cool", "noir", "high_key", "practical",
})
LOOK_GRADES = frozenset({
    "none", "neutral", "warm", "cool", "film_contrast", "bleach",
})
LOOK_SETS = frozenset({
    "empty_floor",
    "studio_cyc",
    "interior_simple",
    "exterior_ground",
    "exterior_park",
    "exterior_forest",
    "exterior_street",
    "exterior_playground",
    "exterior_station",
    "exterior_beach",
})
LOOK_EXTRAS_PRESETS = frozenset({"none", "sidewalk", "park_path", "plaza"})

LOOK_LOCATION_ALIAS = {
    "default": "studio",
    "": "studio",
    "outdoors": "park",
    "outdoor": "park",
    "garden": "park",
    "woods": "forest",
    "jungle": "forest",
    "city": "street",
    "alley": "street",
    "road": "street",
    "sidewalk": "street",
    "train": "station",
    "platform": "station",
    "subway": "station",
    "office": "interior_room",
    "kitchen": "interior_room",
    "bedroom": "interior_room",
    "room": "interior_room",
    "hallway": "corridor",
    "theatre": "stage",
    "theater": "stage",
    "sand": "beach",
    "seaside": "beach",
}
LOOK_TIME_ALIAS = {"sunset": "golden_hour", "sunrise": "dawn"}

# Free-text → nearest kit when not in LOOK_LOCATIONS
LOCATION_FALLBACK_KEYWORDS = (
    ("forest", ("forest", "woods", "jungle", "tree", "trees", "pine")),
    ("playground", ("playground", "swing", "sandbox", "play area")),
    ("station", ("station", "platform", "train", "subway", "metro", "rail")),
    ("beach", ("beach", "sand", "ocean", "seaside", "shore")),
    ("street", ("street", "road", "alley", "city", "sidewalk", "urban", "shop")),
    ("park", ("park", "garden", "lawn", "grass", "field", "outdoor", "outdoors")),
    ("interior_room", ("room", "interior", "indoor", "office", "kitchen", "bedroom", "living")),
    ("corridor", ("corridor", "hallway", "hall")),
    ("stage", ("stage", "theatre", "theater")),
    ("studio", ("studio",)),
)


@dataclass
class LookPlan:
    location: str = "studio"
    time_of_day: str = "day"
    light_mood: str = "soft"
    grade: str = "neutral"
    set_preset: str = "studio_cyc"
    hdri_name: str = ""  # empty → resolve in scene_presets
    wardrobe_id: str = "hero_default"
    extras_count: int = 0
    extras_preset: str = "none"  # none | sidewalk | park_path | plaza
    locked: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None, **flat: Any) -> "LookPlan":
        from .cast_schema import (
            clamp_extras_count,
            normalize_extras_preset,
            normalize_wardrobe_id,
        )

        raw = dict(d or {})
        for f in fields(cls):
            k = f"look_{f.name}"
            if k in flat and f.name not in raw:
                raw[f.name] = flat[k]
        loc = str(raw.get("location") or "studio").strip().lower()
        tod = str(raw.get("time_of_day") or "day").strip().lower()
        loc = LOOK_LOCATION_ALIAS.get(loc, loc)
        tod = LOOK_TIME_ALIAS.get(tod, tod)
        if loc not in LOOK_LOCATIONS:
            loc = nearest_location(loc) or "studio"
        if tod not in LOOK_TIMES:
            tod = "day"
        mood = str(raw.get("light_mood") or "soft")
        if mood not in LOOK_MOODS:
            mood = "soft"
        grade = str(raw.get("grade") or "neutral")
        if grade not in LOOK_GRADES:
            grade = "neutral"
        set_preset = str(raw.get("set_preset") or "").strip() or default_set_for_location(loc)
        if set_preset not in LOOK_SETS:
            set_preset = default_set_for_location(loc)
        extras_preset = normalize_extras_preset(str(raw.get("extras_preset") or "none"))
        if extras_preset not in LOOK_EXTRAS_PRESETS:
            extras_preset = "none"
        return cls(
            location=loc,
            time_of_day=tod,
            light_mood=mood,
            grade=grade,
            set_preset=set_preset,
            hdri_name=str(raw.get("hdri_name") or ""),
            wardrobe_id=normalize_wardrobe_id(str(raw.get("wardrobe_id") or "hero_default")),
            extras_count=clamp_extras_count(raw.get("extras_count", 0)),
            extras_preset=extras_preset,
            locked=bool(raw.get("locked", True)),
            notes=str(raw.get("notes") or "")[:200],
        )


def default_set_for_location(loc: str) -> str:
    return {
        "studio": "studio_cyc",
        "stage": "studio_cyc",
        "interior_room": "interior_simple",
        "corridor": "interior_simple",
        "park": "exterior_park",
        "forest": "exterior_forest",
        "playground": "exterior_playground",
        "street": "exterior_street",
        "station": "exterior_station",
        "beach": "exterior_beach",
        "default": "studio_cyc",
    }.get(loc, "exterior_ground")


def nearest_location(text: str) -> Optional[str]:
    """Map arbitrary words to a kit id (any-scene fallback)."""
    low = (text or "").lower().replace("-", " ").replace("_", " ")
    if low in LOOK_LOCATIONS:
        return low
    if low in LOOK_LOCATION_ALIAS:
        return LOOK_LOCATION_ALIAS[low]
    for kit, words in LOCATION_FALLBACK_KEYWORDS:
        for w in words:
            if w in low or low in w:
                return kit
    return None


def movie_look_enabled() -> bool:
    import os
    return os.environ.get("USE_MOVIE_LOOK", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )
