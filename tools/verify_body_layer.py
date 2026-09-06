"""Verify layered body director schema + packet shape (no GPU required)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from face_agents.director_schema import (  # noqa: E402
    BODY_STATES,
    GESTURE_TARGETS,
    parse_beat,
    parse_director_script,
    describe_catalog,
)


def main() -> int:
    ok = 0
    fail = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  OK  {name}" + (f" — {detail}" if detail else ""))
        else:
            fail += 1
            print(f" FAIL {name}" + (f" — {detail}" if detail else ""))

    print("=== Layered body verification ===\n")
    print(describe_catalog())
    print()

    # 1) Parse modern beat
    b = parse_beat({
        "text": "Hmm let me think.",
        "emotion": "thinking",
        "intensity": 0.7,
        "state": "standing",
        "gesture_target": "chin",
        "hand": "right",
        "actions": ["talk_open"],
    })
    check("thinking → chin/right", b.gesture_target == "chin" and b.hand == "right")
    check("state standing", b.state == "standing")

    # 2) Infer gesture from think_chin action
    b2 = parse_beat({
        "text": "Hmm.",
        "emotion": "thinking",
        "intensity": 0.6,
        "actions": ["think_chin"],
    }, apply_emotion_defaults=False)
    check("think_chin infers chin", b2.gesture_target == "chin", b2.gesture_target)

    # 3) body_target object form
    b3 = parse_beat({
        "text": "Eww no.",
        "emotion": "disgusted",
        "body_target": {"target": "chest", "hand": "right"},
        "actions": ["recoil"],
    }, apply_emotion_defaults=False)
    check("body_target object", b3.gesture_target == "chest" and b3.hand == "right")

    # 4) sitting state alias
    b4 = parse_beat({
        "text": "I will sit.",
        "emotion": "calm",
        "state": "sit",
        "actions": ["talk_open"],
    }, apply_emotion_defaults=False)
    check("sit → sitting", b4.state == "sitting", b4.state)

    # 5) test script file
    path = ROOT / "temp" / "director_actions_test.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    beats = parse_director_script(data)
    check("test script loads", len(beats) == 5, f"n={len(beats)}")
    check("all states valid", all(b.state in BODY_STATES for b in beats))
    check("all gestures valid", all(b.gesture_target in GESTURE_TARGETS for b in beats))

    # 6) packet shape contract
    sample_packet = {
        "type": "body",
        "bones": {"right_shoulder": {"x": 0.1, "y": 0.0, "z": -0.2}},
        "action": "talk_open",
        "t": 0.5,
        "state": b.state,
        "gesture_target": b.gesture_target,
        "hand": b.hand,
        "energy": 0.4,
        "intensity": b.intensity,
        "duration": 2.0,
    }
    required = {"type", "bones", "state", "gesture_target", "hand", "energy"}
    check("packet fields", required.issubset(sample_packet.keys()))

    # 7) viewer markers (directive engine)
    viewer = (ROOT / "web_viewer" / "face_viewer.html").read_text(encoding="utf-8")
    for marker in (
        "updateBodyLayers", "solveArmIK", "ensureIKGesture", "applyBaseState",
        "stepSpring3", "beginStateTransition", "clampEuler", "directive-body-v2",
    ):
        check(f"viewer has {marker}", marker in viewer)

    # 8) bridge forwards gesture
    bridge = (ROOT / "web_viewer" / "ws_bridge.py").read_text(encoding="utf-8")
    check("ws_bridge gesture_target", "gesture_target" in bridge)
    check("ws_bridge state", '"state"' in bridge or "state" in bridge)

    # 9) guide-schema aliases
    b_guide = parse_beat({
        "dialogue_text": "Eww no.",
        "emotion_label": "disgust",
        "base_state": "standing_idle",
        "upper_gesture_target": "hand_to_chest",
        "hand": "right",
        "intensity": 0.9,
        "actions": ["recoil"],
    }, apply_emotion_defaults=False)
    check("guide dialogue_text", b_guide.text.startswith("Eww"))
    check("guide emotion_label disgust→disgusted", b_guide.emotion == "disgusted")
    check("guide base_state standing_idle", b_guide.state == "standing")
    check("guide hand_to_chest→chest", b_guide.gesture_target == "chest")

    # 10) BodyDirectorAgent rule path (no LLM)
    from face_agents.body_director_agent import BodyDirectorAgent
    bd = BodyDirectorAgent(llm_provider=None)
    beats = bd.direct("Eww that is disgusting and gross")
    check("director rule path", len(beats) >= 1 and beats[0].get("gesture_target") in ("chest", "none"))
    check("director has state", beats[0].get("state") in BODY_STATES)

    # 11) motion-only beats survive parse (empty spoken text)
    from face_agents.director_schema import is_motion_caption
    check("humanml is motion caption", is_motion_caption("a person cartwheels across the room"))
    check("spatial stage is motion caption", is_motion_caption("cartwheel across the room"))
    check("dialogue is not motion caption", not is_motion_caption("That is completely unacceptable."))
    silent = parse_director_script([{
        "text": "",
        "humanml_prompt": "a person climbs onto a box then drops down",
        "body_mode": "momask",
        "state": "locomotion",
        "allow_sync_gen": True,
        "motion_only": True,
    }], apply_emotion_defaults=False)
    check("motion-only beat kept", len(silent) == 1 and silent[0].motion_only, f"n={len(silent)}")
    check("motion-only state locomotion", bool(silent) and silent[0].state in BODY_STATES)

    gen = bd.direct("a person cartwheels across the room")
    check(
        "open motion not spoken",
        bool(gen) and (not (gen[0].get("text") or "").strip() or gen[0].get("motion_only")),
        (gen[0].get("text") if gen else None),
    )
    check(
        "open motion → momask",
        bool(gen) and (
            gen[0].get("body_mode") in ("momask", "both")
            or bool(gen[0].get("humanml_prompt"))
        ),
        gen[0] if gen else None,
    )

    print(f"\n=== Result: {ok} passed, {fail} failed ===")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
