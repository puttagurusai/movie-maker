"""
scene_presets.py — Look recipes → HDRI + lights + set_style for ANY location kit.

Unknown location falls back to nearest recipe. Missing HDRI → three_point + procedural set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .look_schema import LookPlan, default_set_for_location, nearest_location

ROOT = Path(__file__).resolve().parents[1]
HDRI_DIR = ROOT / "assets" / "looks" / "hdri"


def _outdoor(
    hdri: str,
    *,
    strength: float = 0.95,
    ground: float = 24.0,
    set_style: str = "exterior_ground",
    sun: float = 2.5,
    key: float = 220.0,
    fill: float = 90.0,
    rim: float = 50.0,
    key_color=(1.0, 0.98, 0.92),
    fill_color=(0.7, 0.82, 1.0),
) -> Dict[str, Any]:
    return {
        "hdri_name": hdri,
        "world_strength": strength,
        "use_three_point": False,
        "key_energy": key,
        "fill_energy": fill,
        "rim_energy": rim,
        "sun_energy": sun,
        "ground_size": ground,
        "set_style": set_style,
        "key_color": key_color,
        "fill_color": fill_color,
        "rim_color": (1.0, 1.0, 1.0),
    }


_RECIPES: Dict[str, Dict[str, Any]] = {
    "studio|day": {
        "hdri_name": "studio_soft",
        "world_strength": 0.85,
        "use_three_point": False,
        "key_energy": 400.0,
        "fill_energy": 140.0,
        "rim_energy": 80.0,
        "sun_energy": 0.0,
        "ground_size": 12.0,
        "set_style": "studio_cyc",
        "key_color": (1.0, 0.98, 0.95),
        "fill_color": (0.75, 0.82, 1.0),
        "rim_color": (1.0, 1.0, 1.0),
    },
    "studio|night": {
        "hdri_name": "studio_soft",
        "world_strength": 0.25,
        "use_three_point": True,
        "key_energy": 500.0,
        "fill_energy": 40.0,
        "rim_energy": 120.0,
        "sun_energy": 0.0,
        "ground_size": 12.0,
        "set_style": "studio_cyc",
        "key_color": (1.0, 0.95, 0.9),
        "fill_color": (0.4, 0.5, 0.8),
        "rim_color": (0.9, 0.95, 1.0),
    },
    "park|day": _outdoor("park_day", set_style="exterior_park", sun=2.8),
    "park|golden_hour": _outdoor(
        "golden_hour",
        set_style="exterior_park",
        strength=1.0,
        sun=3.0,
        key_color=(1.0, 0.75, 0.45),
        fill_color=(0.9, 0.7, 0.5),
    ),
    "park|night": _outdoor(
        "night_urban",
        set_style="exterior_park",
        strength=0.4,
        sun=0.5,
        key=180,
        fill=50,
    ),
    "forest|day": _outdoor("forest_day", set_style="exterior_forest", ground=28.0, sun=2.0),
    "forest|golden_hour": _outdoor(
        "golden_hour", set_style="exterior_forest", ground=28.0, sun=2.5,
        key_color=(1.0, 0.8, 0.5),
    ),
    "forest|night": _outdoor("night_urban", set_style="exterior_forest", strength=0.35, sun=0.3),
    "playground|day": _outdoor("playground_day", set_style="exterior_playground", ground=22.0),
    "playground|night": _outdoor("night_urban", set_style="exterior_playground", strength=0.4, sun=0.4),
    "street|day": _outdoor("street_day", set_style="exterior_street", ground=22.0, sun=3.0),
    "street|night": {
        "hdri_name": "night_urban",
        "world_strength": 0.45,
        "use_three_point": False,
        "key_energy": 200.0,
        "fill_energy": 60.0,
        "rim_energy": 40.0,
        "sun_energy": 2.0,
        "ground_size": 22.0,
        "set_style": "exterior_street",
        "key_color": (1.0, 0.85, 0.6),
        "fill_color": (0.5, 0.65, 1.0),
        "rim_color": (0.7, 0.8, 1.0),
    },
    "street|golden_hour": _outdoor(
        "golden_hour", set_style="exterior_street", sun=3.0,
        key_color=(1.0, 0.75, 0.45),
    ),
    "station|day": _outdoor("street_day", set_style="exterior_station", ground=26.0, sun=2.2),
    "station|night": _outdoor("night_urban", set_style="exterior_station", strength=0.45, sun=0.8),
    "beach|day": _outdoor("golden_hour", set_style="exterior_beach", ground=30.0, sun=3.5),
    "beach|golden_hour": _outdoor("golden_hour", set_style="exterior_beach", ground=30.0, sun=3.5),
    "interior_room|day": {
        "hdri_name": "studio_soft",
        "world_strength": 0.55,
        "use_three_point": True,
        "key_energy": 300.0,
        "fill_energy": 80.0,
        "rim_energy": 50.0,
        "sun_energy": 0.0,
        "ground_size": 10.0,
        "set_style": "interior_simple",
        "key_color": (1.0, 0.95, 0.85),
        "fill_color": (0.8, 0.85, 1.0),
        "rim_color": (1.0, 1.0, 1.0),
    },
    "interior_room|warm": {
        "hdri_name": "golden_hour",
        "world_strength": 0.65,
        "use_three_point": True,
        "key_energy": 280.0,
        "fill_energy": 70.0,
        "rim_energy": 45.0,
        "sun_energy": 0.0,
        "ground_size": 10.0,
        "set_style": "interior_simple",
        "key_color": (1.0, 0.8, 0.55),
        "fill_color": (0.9, 0.7, 0.5),
        "rim_color": (1.0, 0.9, 0.7),
    },
    "corridor|day": {
        "hdri_name": "studio_soft",
        "world_strength": 0.45,
        "use_three_point": True,
        "key_energy": 260.0,
        "fill_energy": 70.0,
        "rim_energy": 40.0,
        "sun_energy": 0.0,
        "ground_size": 14.0,
        "set_style": "interior_simple",
        "key_color": (1.0, 0.97, 0.92),
        "fill_color": (0.75, 0.8, 1.0),
        "rim_color": (1.0, 1.0, 1.0),
    },
    "stage|day": {
        "hdri_name": "studio_soft",
        "world_strength": 0.5,
        "use_three_point": True,
        "key_energy": 450.0,
        "fill_energy": 100.0,
        "rim_energy": 150.0,
        "sun_energy": 0.0,
        "ground_size": 14.0,
        "set_style": "studio_cyc",
        "key_color": (1.0, 0.98, 0.95),
        "fill_color": (0.7, 0.75, 1.0),
        "rim_color": (1.0, 1.0, 1.0),
    },
}


def _hdri_path(name: str) -> Optional[Path]:
    if not name:
        return None
    for ext in (".exr", ".hdr"):
        p = HDRI_DIR / f"{name}{ext}"
        if p.is_file() and p.stat().st_size > 10_000:
            return p
    return None


def resolve_look(look: LookPlan) -> Dict[str, Any]:
    """
    Expand LookPlan into apply parameters for Blender (any location kit).
    """
    loc = look.location or "studio"
    if loc not in (
        "studio", "park", "forest", "playground", "street", "station",
        "beach", "interior_room", "corridor", "stage",
    ):
        loc = nearest_location(loc) or "studio"

    tod = look.time_of_day or "day"
    mood = look.light_mood or "soft"

    key = f"{loc}|{tod}"
    if key not in _RECIPES and mood == "warm" and loc == "interior_room":
        key = "interior_room|warm"
    if key not in _RECIPES and tod == "golden_hour":
        for cand in (f"{loc}|golden_hour", "park|golden_hour", "beach|golden_hour", f"{loc}|day"):
            if cand in _RECIPES:
                key = cand
                break
    if key not in _RECIPES and tod == "night":
        for cand in (f"{loc}|night", "street|night", "studio|night"):
            if cand in _RECIPES:
                key = cand
                break
    if key not in _RECIPES and tod in ("dawn", "dusk", "overcast"):
        key = f"{loc}|day" if f"{loc}|day" in _RECIPES else "studio|day"
    if key not in _RECIPES:
        key = f"{loc}|day" if f"{loc}|day" in _RECIPES else "studio|day"

    base = dict(_RECIPES[key])
    hdri_name = (look.hdri_name or base.get("hdri_name") or "").strip()
    hdri_path = _hdri_path(hdri_name)
    # Fallback chain for outdoor kits
    if hdri_path is None and loc in ("park", "forest", "playground", "street", "station", "beach"):
        for alt in ("park_day", "golden_hour", "street_day", "studio_soft"):
            hdri_path = _hdri_path(alt)
            if hdri_path:
                hdri_name = alt
                break

    fallback = False
    if hdri_path is None:
        fallback = True
        base["use_three_point"] = True
        base["world_strength"] = min(0.35, float(base.get("world_strength") or 0.35))

    if mood == "noir":
        base["world_strength"] = min(0.3, float(base["world_strength"]))
        base["fill_energy"] = float(base["fill_energy"]) * 0.35
        base["key_energy"] = float(base["key_energy"]) * 1.15
        base["use_three_point"] = True
    elif mood == "hard":
        base["fill_energy"] = float(base["fill_energy"]) * 0.5
        base["key_energy"] = float(base["key_energy"]) * 1.2
    elif mood == "soft":
        base["fill_energy"] = float(base["fill_energy"]) * 1.15
    elif mood == "cool":
        base["fill_color"] = (0.55, 0.7, 1.0)
        base["key_color"] = (0.85, 0.9, 1.0)
    elif mood == "warm":
        base["key_color"] = (1.0, 0.82, 0.55)
        base["fill_color"] = (1.0, 0.75, 0.5)

    set_style = look.set_preset if look.set_preset else base.get("set_style")
    if not set_style or set_style == "exterior_ground":
        set_style = default_set_for_location(loc)

    return {
        "recipe_name": key,
        "hdri_name": hdri_name,
        "hdri_path": str(hdri_path) if hdri_path else "",
        "use_three_point": bool(base.get("use_three_point")),
        "world_strength": float(base.get("world_strength") or 0.7),
        "key_energy": float(base.get("key_energy") or 300),
        "fill_energy": float(base.get("fill_energy") or 100),
        "rim_energy": float(base.get("rim_energy") or 50),
        "sun_energy": float(base.get("sun_energy") or 0),
        "ground_size": float(base.get("ground_size") or 12),
        "set_style": set_style,
        "key_color": tuple(base.get("key_color") or (1, 1, 1)),
        "fill_color": tuple(base.get("fill_color") or (0.8, 0.85, 1)),
        "rim_color": tuple(base.get("rim_color") or (1, 1, 1)),
        "fallback": fallback,
        "grade": look.grade,
        "location": loc,
        "time_of_day": tod,
        "light_mood": mood,
    }
