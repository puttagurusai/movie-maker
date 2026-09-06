"""
look_agent.py — Infer LookPlan from user text / previous look.

Never writes into humanml_prompt.
Maps ANY scene wording to a kit (park/forest/street/…) via nearest_location.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .cast_schema import (
    clamp_extras_count,
    normalize_extras_preset,
    normalize_wardrobe_id,
)
from .look_schema import (
    LookPlan,
    default_set_for_location,
    nearest_location,
)


def plan_look(
    user_text: str = "",
    *,
    prev_look: Optional[LookPlan] = None,
    explicit: Optional[Dict[str, Any]] = None,
) -> LookPlan:
    """
    Build a LookPlan for this beat / pre-session setup.

    Priority: explicit JSON/dict → keyword inference → locked previous → studio default.
    """
    if explicit:
        return LookPlan.from_dict(
            explicit if isinstance(explicit, dict) else None,
            **(explicit if isinstance(explicit, dict) else {}),
        )

    low = (user_text or "").lower()
    locked = prev_look if (prev_look and prev_look.locked) else None

    # New scene unlocks look
    if re.search(r"(?i)\b(new\s+scene|new\s+set|change\s+(the\s+)?(scene|set|location))\b", low):
        locked = None

    loc = None
    tod = None
    mood = None
    grade = None
    set_preset = None
    wardrobe_id = None
    extras_count = None
    extras_preset = None

    # Wardrobe (typed presets — LookAgent owns this, not BodyDirector)
    if re.search(r"(?i)\b(suit|formal|blazer|business\s*attire|wear\s+a\s+suit)\b", low):
        wardrobe_id = "formal_01"
    elif re.search(r"(?i)\b(casual|jeans|t-?shirt|tee|streetwear|wear\s+casual)\b", low):
        wardrobe_id = "casual_01"
    elif re.search(r"(?i)\b(default\s+clothes|wear\s+default|plain\s+clothes)\b", low):
        wardrobe_id = "hero_default"
    m_wear = re.search(r"(?i)\bwear\s+([a-z0-9_\-]+)", low)
    if m_wear and wardrobe_id is None:
        wardrobe_id = normalize_wardrobe_id(m_wear.group(1))

    # Extras / crowd (body-only NPCs)
    if re.search(r"(?i)\b(clear\s+people|no\s+people|empty\s+crowd|remove\s+extras)\b", low):
        extras_count = 0
        extras_preset = "none"
    else:
        m_add = re.search(
            r"(?i)\b(?:add|spawn)\s+(\d+)\s+(?:people|persons|extras|npcs|pedestrians)\b",
            low,
        )
        m_crowd = re.search(r"(?i)\b(crowd|busy\s+street|many\s+people|lots\s+of\s+people)\b", low)
        if m_add:
            extras_count = clamp_extras_count(m_add.group(1))
        elif m_crowd:
            extras_count = 5
        if extras_count:
            if re.search(r"(?i)\b(park|garden|path)\b", low):
                extras_preset = "park_path"
            elif re.search(r"(?i)\b(plaza|square)\b", low):
                extras_preset = "plaza"
            else:
                extras_preset = "sidewalk"

    # Specific kits first (order matters)
    if re.search(r"(?i)\b(forest|woods|jungle|pine\s+trees?)\b", low):
        loc = "forest"
    elif re.search(r"(?i)\b(playground|swing|sandbox)\b", low):
        loc = "playground"
    elif re.search(r"(?i)\b(station|train\s+platform|subway|metro|rail\s*way)\b", low):
        loc = "station"
    elif re.search(r"(?i)\b(beach|sand|seaside|shore|ocean)\b", low):
        loc = "beach"
    elif re.search(r"(?i)\b(street|road|alley|sidewalk|city|urban|shop)\b", low):
        loc = "street"
    elif re.search(r"(?i)\b(park|garden|lawn|grass|field|outdoor|outdoors)\b", low):
        loc = "park"
    elif re.search(r"(?i)\b(corridor|hallway)\b", low):
        loc = "corridor"
    elif re.search(r"(?i)\b(room|interior|indoor|office|kitchen|bedroom|living\s*room)\b", low):
        loc = "interior_room"
    elif re.search(r"(?i)\b(stage|theatre|theater)\b", low):
        loc = "stage"
    elif re.search(r"(?i)\b(studio)\b", low):
        loc = "studio"
    else:
        # Any other place words → nearest kit (never invent a random unknown id)
        guessed = nearest_location(low)
        if guessed and guessed != "studio":
            loc = guessed

    if loc:
        set_preset = default_set_for_location(loc)

    if re.search(r"(?i)\b(night|midnight|moonlight)\b", low):
        tod = "night"
    elif re.search(r"(?i)\b(sunset|golden\s*hour|dusk)\b", low):
        tod = "golden_hour"
        mood = mood or "warm"
        grade = grade or "warm"
    elif re.search(r"(?i)\b(dawn|sunrise|morning)\b", low):
        tod = "dawn"
    elif re.search(r"(?i)\b(overcast|cloudy|grey|gray\s+sky)\b", low):
        tod = "overcast"
    elif re.search(r"(?i)\b(day|daytime|afternoon|noon)\b", low):
        tod = "day"

    if re.search(r"(?i)\b(noir|moody|dramatic\s+light)\b", low):
        mood = "noir"
        grade = grade or "film_contrast"
    elif re.search(r"(?i)\b(warm(\s+light|\s+room)?|cozy|orange\s+light)\b", low):
        mood = "warm"
        grade = grade or "warm"
    elif re.search(r"(?i)\b(cool\s+light|blue\s+light|cold)\b", low):
        mood = "cool"
        grade = grade or "cool"
    elif re.search(r"(?i)\b(soft\s+light|soft\s+lighting)\b", low):
        mood = "soft"
    elif re.search(r"(?i)\b(hard\s+light|harsh\s+light)\b", low):
        mood = "hard"

    inferred = any(
        x is not None
        for x in (loc, tod, mood, grade, set_preset, wardrobe_id, extras_count, extras_preset)
    )
    if locked and not inferred:
        return LookPlan.from_dict(locked.to_dict())

    base = locked.to_dict() if locked else {}
    old_loc = str(base.get("location") or "studio")
    if loc:
        base["location"] = loc
        base["set_preset"] = set_preset or default_set_for_location(loc)
        if loc != old_loc and tod is None:
            base["time_of_day"] = "day"
    if tod:
        base["time_of_day"] = tod
    if mood:
        base["light_mood"] = mood
    if grade:
        base["grade"] = grade
    if set_preset and "set_preset" not in base:
        base["set_preset"] = set_preset
    if wardrobe_id:
        base["wardrobe_id"] = normalize_wardrobe_id(wardrobe_id)
    if extras_count is not None:
        base["extras_count"] = clamp_extras_count(extras_count)
    if extras_preset is not None:
        base["extras_preset"] = normalize_extras_preset(extras_preset)
        if base["extras_preset"] != "none" and not base.get("extras_count"):
            base["extras_count"] = 3
    if not base:
        base = {
            "location": "studio",
            "time_of_day": "day",
            "light_mood": "soft",
            "set_preset": "studio_cyc",
            "wardrobe_id": "hero_default",
            "extras_count": 0,
            "extras_preset": "none",
        }
    base["locked"] = True
    return LookPlan.from_dict(base)
