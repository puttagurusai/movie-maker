"""
test_movie_camera.py — unit tests for movie camera planner (no Blender required).

  python test_movie_camera.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from face_agents.camera_agent import (
    plan_camera_for_beat,
    select_shot_and_move,
    movie_camera_enabled,
)
from face_agents.movie_timeline import build_beat_with_camera, MovieTimeline


def test_rules():
    cases = [
        ("standing", "neutral", 0.7, [], "MS"),
        ("walking", "neutral", 0.7, ["talk_open"], "WS"),
        ("standing", "sad", 0.8, ["talk_open"], "MCU"),
        ("standing", "angry", 0.9, ["talk_open"], "CU"),
        ("standing", "happy", 0.8, ["wave"], "MS"),
    ]
    for state, emo, inten, acts, expect_shot in cases:
        sh, mv, notes = select_shot_and_move(
            emotion=emo, intensity=inten, state=state, actions=acts, duration_s=4.0,
            session_clip_index=3,
        )
        assert sh == expect_shot, f"{state}/{emo}: got {sh} want {expect_shot} ({notes})"
        print(f"  OK rule {state}/{emo} → {sh} {mv}")

    # Any HumanML / open motion — not a named-action list — is framed full-body
    for caption in (
        "a person cartwheels across the room",
        "a person climbs onto a box then drops down",
        "a figure balances on one foot then spins",
    ):
        sh, mv, notes = select_shot_and_move(
            emotion="neutral",
            intensity=0.7,
            state="standing",
            actions=[],
            duration_s=5.0,
            humanml_prompt=caption,
            session_clip_index=1,
        )
        assert sh in ("WS", "MLS"), f"{caption!r} → {sh} {notes}"
        assert mv in ("follow", "truck"), f"{caption!r} move={mv}"
        print(f"  OK open motion {caption[:32]!r} → {sh} {mv}")

    sh, mv, notes = select_shot_and_move(
        emotion="neutral", intensity=0.7, state="locomotion",
        actions=[], duration_s=5.0,
        session_clip_index=1,
    )
    assert sh == "WS" and mv == "follow", f"locomotion class → {sh}/{mv} {notes}"
    print(f"  OK locomotion class → {sh} {mv}")


def test_plan_keyframes():
    plan = plan_camera_for_beat(
        duration_s=4.0,
        emotion="happy",
        intensity=0.8,
        state="standing",
        actions=["talk_open"],
        fps=30.0,
        session_clip_index=1,
    )
    assert plan.keyframes, "no keyframes"
    assert plan.keyframes[0].t == 0.0
    assert plan.keyframes[-1].t >= 3.9
    pkt = plan.udp_packet()
    assert pkt["type"] == "camera"
    assert pkt["op"] == "plan"
    assert len(pkt["keyframes"]) >= 2
    # locations are -Y (in front of character)
    y0 = pkt["keyframes"][0]["location"][1]
    assert y0 < 0, f"camera should be on -Y, got y={y0}"
    print(f"  OK plan shot={plan.shot} move={plan.move_type} keys={len(plan.keyframes)} y0={y0}")


def test_clip0_establish():
    sh, mv, notes = select_shot_and_move(
        emotion="happy", intensity=0.8, state="standing",
        actions=["talk_open"], duration_s=3.0, session_clip_index=0,
    )
    assert sh == "WS", f"clip0 hello must be WS, got {sh} {notes}"
    assert mv in ("static", "reveal"), mv
    plan = plan_camera_for_beat(
        duration_s=3.0, emotion="happy", intensity=0.8, state="standing",
        actions=["talk_open"], fps=20.0, session_clip_index=0, camera_role="",
    )
    assert plan.shot == "WS"
    assert plan.camera_role == "env_cam"
    assert plan.camera_name == "MovieCam_Env"
    # JSON hint wins
    plan2 = plan_camera_for_beat(
        duration_s=3.0, emotion="happy", intensity=0.8, state="standing",
        actions=["talk_open"], fps=20.0, session_clip_index=0,
        shot="MCU", move_type="static", camera_role="A_cam",
    )
    assert plan2.shot == "MCU" and plan2.camera_role == "A_cam"
    # location change later in the session also establishes
    sh2, mv2, n2 = select_shot_and_move(
        emotion="neutral", intensity=0.7, state="standing",
        actions=["talk_open"], duration_s=3.0, session_clip_index=4,
        location_changed=True,
    )
    assert sh2 == "WS", n2
    print(f"  OK clip0/new-loc WS env_cam; JSON hint wins ({plan2.shot}/{plan2.camera_role})")


def test_walk_truck():
    plan = plan_camera_for_beat(
        duration_s=5.0,
        emotion="neutral",
        intensity=0.7,
        state="walking",
        actions=["talk_open"],
        fps=30.0,
        session_clip_index=1,
    )
    assert plan.shot == "WS"
    # Locomotion: full-body subject follow (FilmAgent-style, not hip-only lock)
    assert plan.move_type in ("follow", "truck")
    assert plan.subject_relative is True
    assert plan.subject_anchor == "full_body"
    xs = [k.location[0] for k in plan.keyframes]
    assert max(xs) - min(xs) > 0.15, f"follow/truck should move in X, got {xs}"
    print(
        f"  OK walk {plan.move_type} anchor={plan.subject_anchor} "
        f"X range {min(xs):.2f}..{max(xs):.2f}"
    )


def test_timeline_json(tmp: Path):
    beat = build_beat_with_camera(
        beat_id="b1",
        t0=0.0,
        duration_s=3.5,
        text="Hello everyone.",
        emotion="happy",
        intensity=0.8,
        state="standing",
        actions=["wave", "talk_open"],
        fps=30.0,
    )
    tl = MovieTimeline(fps=30.0, title="test", beats=[beat])
    out = tmp / "movie_timeline_test.json"
    tl.save(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["beats"][0]["camera"]["keyframes"]
    print(f"  OK timeline {out.name} duration={data['duration_s']}")


def main() -> int:
    print("=" * 50)
    print("MOVIE CAMERA UNIT TESTS")
    print("=" * 50)
    assert movie_camera_enabled(), "USE_MOVIE_CAMERA should default on"
    test_rules()
    test_clip0_establish()
    test_plan_keyframes()
    test_walk_truck()
    test_timeline_json(ROOT / "temp")
    print("=" * 50)
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
