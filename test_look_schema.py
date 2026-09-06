"""LookPlan nested/flat round-trip; look never mutates HumanML."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from face_agents.look_schema import LookPlan
from face_agents.look_agent import plan_look
from face_agents.director_schema import parse_beat
from face_agents.momask_body_pipeline import to_humanml_caption


def test_nested_and_flat():
    nested = LookPlan.from_dict({
        "location": "street",
        "time_of_day": "sunset",
        "light_mood": "warm",
        "grade": "warm",
    })
    assert nested.location == "street"
    assert nested.time_of_day == "golden_hour"  # alias
    d = nested.to_dict()
    again = LookPlan.from_dict(d)
    assert again.to_dict() == d

    flat = LookPlan.from_dict(None, look_location="park", look_time_of_day="night", look_grade="cool")
    assert flat.location == "park"
    assert flat.time_of_day == "night"
    assert flat.grade == "cool"
    default = LookPlan.from_dict({"location": "default"})
    assert default.location == "studio"
    print("  OK nested+flat round-trip + aliases")


def test_parse_beat_keeps_look_out_of_humanml():
    beat = parse_beat({
        "text": "Walk forward and say we won.",
        "humanml_prompt": "a person walks forward",
        "look": {"location": "street", "time_of_day": "golden_hour", "grade": "warm"},
        "look_light_mood": "warm",
    }, apply_emotion_defaults=False)
    assert beat.look.location == "street"
    assert beat.look.time_of_day == "golden_hour"
    assert "golden" not in beat.humanml_prompt
    assert "street" not in beat.humanml_prompt
    face = beat.to_face_sentence()
    assert "look" not in face
    d = beat.to_dict()
    assert d["look"]["location"] == "street"
    print("  OK parse_beat nested look; to_face_sentence has no look")


def test_plan_look_does_not_touch_caption():
    hml = to_humanml_caption("walk forward")
    look = plan_look("walk forward at golden hour on the street")
    assert look.time_of_day == "golden_hour"
    assert look.location == "street"
    assert hml == to_humanml_caption("walk forward")
    assert "golden" not in hml
    print(f"  OK look inference separate from HumanML {hml!r}")


def main() -> int:
    print("=" * 50)
    print("LOOK SCHEMA")
    print("=" * 50)
    test_nested_and_flat()
    test_parse_beat_keeps_look_out_of_humanml()
    test_plan_look_does_not_touch_caption()
    print("=" * 50)
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
