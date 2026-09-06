"""
Per-action body test: drive BodyDirector + BodyAgent with fixed inputs,
sample poses over time, score quality, print a clear report.

  python tools/test_each_action.py
  python tools/test_each_action.py --action wave
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from face_agents.body_agent import ACTION_CLIPS, BASE_STATE_POSES, BodyAgent  # noqa: E402
from face_agents.body_director_agent import BodyDirectorAgent  # noqa: E402
from face_agents.director_schema import ACTION_CATALOG, EMOTION_DEFAULT_ACTIONS  # noqa: E402

# Fixed natural-language inputs that SHOULD map to each action (or pair)
ACTION_TEST_INPUTS: Dict[str, str] = {
    "wave": "wave and say welcome",
    "shrug": "shrug and say I don't know",
    "talk_open": "just say hello there friend",
    "talk_emphasize": "I am so angry this is completely unacceptable",
    "recoil": "oh my god what is that thing coming toward us",
    "celebrate": "ha that is hilarious and funny",
    "slump": "I feel really sad and unhappy about this",
    "tense": "I am furious and mad about this mess",
    "think_chin": "hmm let me think about that carefully",
    "hands_reject": "eww that is disgusting and gross",
    "point_forward": "point at that and say look over there",
    "nod_yes": "nod and say yes I agree completely",
    "shake_no": "shake your head and say no way",
    "look_camera": "look at the camera and say hi",
    "look_left": "glance left and say over there",
    "look_right": "look right and say on that side",
    "weight_shift": "shift your weight and say okay",
    "idle": "stay still and calm",
}

# Expected pose signatures (bones that MUST move with meaningful magnitude)
ACTION_EXPECT: Dict[str, Dict[str, Any]] = {
    "wave": {
        "must_bones": ["right_shoulder", "right_elbow"],
        "nice_bones": ["spine3", "left_shoulder", "pelvis", "right_wrist"],
        "min_peak": 1.2,
        "min_frames_active": 3,
    },
    "shrug": {
        "must_bones": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
        "nice_bones": ["left_collar", "right_collar", "neck"],
        "min_peak": 2.0,
        "symmetric_shoulders": True,
    },
    "talk_open": {
        "must_bones": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
        "min_peak": 0.8,
        "loop": True,
    },
    "talk_emphasize": {
        "must_bones": ["right_shoulder", "right_elbow"],
        "nice_bones": ["spine3"],
        "min_peak": 0.9,
    },
    "recoil": {
        "must_bones": ["spine2", "spine3", "left_shoulder", "right_shoulder"],
        "nice_bones": ["neck", "pelvis"],
        "min_peak": 2.0,
    },
    "celebrate": {
        "must_bones": ["left_shoulder", "right_shoulder"],
        "min_peak": 1.5,
        "symmetric_shoulders": True,
    },
    "slump": {
        "must_bones": ["spine1", "spine2", "spine3", "neck"],
        "min_peak": 0.8,
    },
    "tense": {
        "must_bones": ["spine1", "spine3", "left_shoulder", "right_shoulder"],
        "min_peak": 0.6,
    },
    "think_chin": {
        "must_bones": ["right_shoulder", "right_elbow"],
        "nice_bones": ["neck", "spine3"],
        "min_peak": 1.5,
    },
    "hands_reject": {
        "must_bones": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
        "min_peak": 1.5,
        "symmetric_shoulders": True,
    },
    "point_forward": {
        "must_bones": ["right_shoulder", "right_elbow"],
        "nice_bones": ["spine3", "neck"],
        "min_peak": 1.0,
    },
    "nod_yes": {
        "must_bones": ["neck"],
        "nice_bones": ["spine3", "head"],
        "min_peak": 0.25,
        "oscillating": "neck",  # x should change sign or multi-peak
    },
    "shake_no": {
        "must_bones": ["neck"],
        "nice_bones": ["spine3", "head"],
        "min_peak": 0.3,
        "oscillating": "neck",
    },
    "look_camera": {
        "must_bones": ["neck", "head"],
        "min_peak": 0.12,
        # neck/head may be smaller than talk arms — check absolute bone mag not only peak frame top
        "must_bone_min": 0.04,
    },
    "look_left": {
        "must_bones": ["neck"],
        "min_peak": 0.3,
    },
    "look_right": {
        "must_bones": ["neck"],
        "min_peak": 0.3,
    },
    "weight_shift": {
        "must_bones": ["pelvis", "spine1"],
        "min_peak": 0.2,
    },
    "idle": {
        "must_bones": [],
        "min_peak": 0.0,
        "allow_zero": True,
    },
}

# Sit / walk / dance state tests
STATE_INPUTS = {
    "sitting": "say hello while sitting",
    "walking": "walk over and think with hand on chin",
    "standing": "just say hello friend",
    "dancing": "dance and say party time",
}


class Ctx:
    def __init__(
        self,
        actions: List[str],
        state: str = "standing",
        gesture: str = "none",
        emotion: str = "neutral",
        intensity: float = 0.85,
        duration: float = 2.5,
        hand: str = "right",
    ):
        self.duration = duration
        self.emotion = emotion
        self.intensity = intensity
        self.is_speaking = True
        self.t = 0.0
        self.extras = {
            "body_actions": list(actions),
            "action_timing": "start",
            "gesture_target": gesture,
            "hand": hand,
            "body_state": state,
            "state": state,
        }

    def speech_energy(self) -> float:
        # fake speech envelope
        return 0.25 + 0.35 * abs(math.sin(self.t * 3.0))


def mag(e: Dict[str, float]) -> float:
    return math.sqrt(
        float(e.get("x", 0) or 0) ** 2
        + float(e.get("y", 0) or 0) ** 2
        + float(e.get("z", 0) or 0) ** 2
    )


def sample_timeline(agent: BodyAgent, ctx: Ctx, n: int = 12) -> List[Dict[str, Any]]:
    frames = []
    for i in range(n):
        ctx.t = (i / max(n - 1, 1)) * ctx.duration
        bones = agent.get_body(ctx)
        mags = {k: mag(v) for k, v in bones.items()}
        total = sum(mags.values())
        frames.append({
            "t": round(ctx.t, 3),
            "total": round(total, 4),
            "bones": {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in bones.items()},
            "mags": {k: round(m, 4) for k, m in mags.items()},
            "top": sorted(mags.items(), key=lambda x: -x[1])[:6],
        })
    return frames


def score_action(name: str, frames: List[Dict[str, Any]], expect: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (critical, warnings)."""
    crit: List[str] = []
    warn: List[str] = []
    if expect.get("allow_zero"):
        return crit, warn

    peaks = [f["total"] for f in frames]
    peak = max(peaks) if peaks else 0.0
    active = sum(1 for p in peaks if p >= expect.get("min_peak", 0.2) * 0.35)

    if peak < expect.get("min_peak", 0.2):
        crit.append(f"peak total mag {peak:.3f} < min {expect['min_peak']}")

    if active < expect.get("min_frames_active", 2):
        warn.append(f"only {active} frames with solid motion (clip may be too short/late)")

    # bone presence at peak frame
    peak_f = max(frames, key=lambda f: f["total"])
    present = set(peak_f["mags"].keys())
    bone_min = float(expect.get("must_bone_min", 0.05))
    # max mag of each must-bone across timeline (not only global peak frame)
    max_per_bone: Dict[str, float] = {}
    for f in frames:
        for b, m in f["mags"].items():
            max_per_bone[b] = max(max_per_bone.get(b, 0.0), m)
    for b in expect.get("must_bones", []):
        m = max_per_bone.get(b, 0.0)
        if m < bone_min:
            crit.append(f"must-bone '{b}' missing/weak across clip (max mag={m:.3f})")

    for b in expect.get("nice_bones", []):
        m = peak_f["mags"].get(b, 0.0)
        if m < 0.03:
            warn.append(f"nice-bone '{b}' weak/absent (mag={m:.3f})")

    if expect.get("symmetric_shoulders"):
        ls = peak_f["mags"].get("left_shoulder", 0)
        rs = peak_f["mags"].get("right_shoulder", 0)
        if ls + rs > 0.1:
            ratio = min(ls, rs) / max(ls, rs)
            if ratio < 0.55:
                warn.append(f"shoulders asymmetric L={ls:.2f} R={rs:.2f} ratio={ratio:.2f}")

    # oscillation check for nod/shake
    axis_bone = expect.get("oscillating")
    if axis_bone:
        # neck x for nod, neck y for shake
        xs = []
        ys = []
        for f in frames:
            b = f["bones"].get(axis_bone) or {}
            xs.append(float(b.get("x", 0) or 0))
            ys.append(float(b.get("y", 0) or 0))
        if name == "nod_yes":
            if max(xs) - min(xs) < 0.08:
                crit.append(f"nod neck.x range too small ({min(xs):.3f}..{max(xs):.3f})")
            # should go both positive-ish over time (down then up)
            if max(xs) < 0.05:
                warn.append("nod never dips head down much")
        if name == "shake_no":
            if max(ys) - min(ys) < 0.1:
                crit.append(f"shake neck.y range too small ({min(ys):.3f}..{max(ys):.3f})")
            if max(ys) < 0.05 or min(ys) > -0.05:
                warn.append("shake may not go both left and right")

    # flatline: all frames same total
    if len(set(round(p, 3) for p in peaks)) == 1 and peak > 0.05 and name not in ("tense", "slump", "idle"):
        # looping holds can be flat-ish; only warn for one-shots that never change
        clip = ACTION_CLIPS.get(name) or {}
        if not clip.get("loop"):
            warn.append("motion flat across timeline (no attack/release shape)")

    return crit, warn


def test_one_action(name: str, use_director: bool = True) -> Dict[str, Any]:
    expect = ACTION_EXPECT.get(name, {"must_bones": [], "min_peak": 0.3})
    phrase = ACTION_TEST_INPUTS.get(name, f"do a {name}")
    director_beat = None
    actions = [name]
    state = "standing"
    gesture = "none"
    emotion = "neutral"
    intensity = 0.85
    hand = "right"

    if use_director:
        bd = BodyDirectorAgent(None)
        director_beat = bd.direct(phrase)[0]
        # Prefer explicit action under test, but keep director state/emotion
        state = str(director_beat.get("state") or "standing")
        gesture = str(director_beat.get("gesture_target") or "none")
        emotion = str(director_beat.get("emotion") or "neutral")
        intensity = float(director_beat.get("intensity") or 0.85)
        hand = str(director_beat.get("hand") or "right")
        # Force the action under test into queue (plus director extras)
        d_acts = list(director_beat.get("actions") or [])
        if name not in d_acts and name != "idle":
            actions = [name] + [a for a in d_acts if a != name]
        else:
            actions = d_acts if d_acts else [name]

    agent = BodyAgent()
    ctx = Ctx(
        actions=actions,
        state=state,
        gesture=gesture,
        emotion=emotion,
        intensity=intensity,
        duration=2.4,
        hand=hand,
    )
    agent.prepare(ctx)
    frames = sample_timeline(agent, ctx, n=12)
    crit, warn = score_action(name, frames, expect)

    # Director mapping issues
    if use_director and director_beat is not None:
        d_acts = director_beat.get("actions") or []
        if name not in ("idle", "talk_open", "look_camera") and name not in d_acts:
            # only warn if phrase was meant to trigger it
            if name in phrase or name.replace("_", " ") in phrase:
                warn.append(f"director did not select action '{name}' (got {d_acts})")

    peak_f = max(frames, key=lambda f: f["total"])
    return {
        "action": name,
        "input": phrase,
        "director": director_beat,
        "forced_actions": actions,
        "state": state,
        "emotion": emotion,
        "gesture": gesture,
        "peak_total": peak_f["total"],
        "peak_t": peak_f["t"],
        "peak_top": peak_f["top"],
        "timeline_totals": [f["total"] for f in frames],
        "critical": crit,
        "warnings": warn,
        "pass": len(crit) == 0,
        "frames_sample": [
            {"t": frames[0]["t"], "total": frames[0]["total"], "top": frames[0]["top"]},
            {"t": peak_f["t"], "total": peak_f["total"], "top": peak_f["top"]},
            {"t": frames[-1]["t"], "total": frames[-1]["total"], "top": frames[-1]["top"]},
        ],
    }


def test_states() -> List[Dict[str, Any]]:
    out = []
    bd = BodyDirectorAgent(None)
    for state, phrase in STATE_INPUTS.items():
        beat = bd.direct(phrase)[0]
        agent = BodyAgent()
        ctx = Ctx(
            actions=list(beat.get("actions") or ["talk_open"]),
            state=str(beat.get("state") or state),
            gesture=str(beat.get("gesture_target") or "none"),
            emotion=str(beat.get("emotion") or "neutral"),
            intensity=float(beat.get("intensity") or 0.8),
            duration=2.0,
        )
        agent.prepare(ctx)
        frames = sample_timeline(agent, ctx, n=10)
        # check base bones for sit/walk
        peak = max(frames, key=lambda f: f["total"])
        base_keys = {"pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "spine1"}
        base_mag = sum(peak["mags"].get(k, 0) for k in base_keys)
        crit, warn = [], []
        if beat.get("state") != state and state in ("sitting", "walking"):
            # dancing may not parse from "dance and say"
            if state != "dancing":
                crit.append(f"director state={beat.get('state')} expected {state}")
        if state == "sitting":
            if peak["mags"].get("left_hip", 0) < 0.8:
                crit.append(f"sit left_hip mag={peak['mags'].get('left_hip',0):.2f}")
            if base_mag < 2.0:
                crit.append(f"sit base_mag={base_mag:.2f} too low")
        if state == "walking":
            totals = [f["total"] for f in frames]
            if max(totals) - min(totals) < 0.15:
                warn.append("walk cycle barely varies over time")
            if base_mag < 0.4:
                crit.append(f"walk base_mag={base_mag:.2f} too low")
        out.append({
            "state": state,
            "input": phrase,
            "director_state": beat.get("state"),
            "director_text": beat.get("text"),
            "base_mag": round(base_mag, 3),
            "peak_total": peak["total"],
            "critical": crit,
            "warnings": warn,
            "pass": len(crit) == 0,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="", help="Test only this action")
    ap.add_argument("--out", type=Path, default=ROOT / "temp" / "action_test_report.json")
    ap.add_argument("--no-director", action="store_true", help="Force action only, skip director NL")
    args = ap.parse_args()

    actions = [args.action] if args.action else sorted(ACTION_CLIPS.keys())
    print("=" * 70)
    print("PER-ACTION BODY TEST")
    print("=" * 70)

    results = []
    n_pass = n_fail = 0
    for name in actions:
        if name not in ACTION_CLIPS:
            print(f"  SKIP unknown action {name}")
            continue
        r = test_one_action(name, use_director=not args.no_director)
        results.append(r)
        status = "PASS" if r["pass"] else "FAIL"
        if r["pass"]:
            n_pass += 1
        else:
            n_fail += 1
        print(f"\n[{status}] {name}")
        print(f"  input:    {r['input']!r}")
        if r.get("director"):
            d = r["director"]
            print(
                f"  director: emo={d.get('emotion')} state={d.get('state')} "
                f"gest={d.get('gesture_target')}/{d.get('hand')} acts={d.get('actions')}"
            )
            print(f"  says:     {d.get('text')!r}")
        print(f"  peak:     {r['peak_total']:.3f} @ t={r['peak_t']:.2f}")
        print(f"  top:      {r['peak_top']}")
        print(f"  timeline: {r['timeline_totals']}")
        for c in r["critical"]:
            print(f"  !! CRIT  {c}")
        for w in r["warnings"]:
            print(f"  !! WARN  {w}")

    print("\n" + "=" * 70)
    print("STATE TESTS")
    print("=" * 70)
    state_results = test_states()
    for r in state_results:
        status = "PASS" if r["pass"] else "FAIL"
        if r["pass"]:
            n_pass += 1
        else:
            n_fail += 1
        print(f"\n[{status}] state={r['state']}")
        print(f"  input: {r['input']!r}")
        print(f"  director_state={r['director_state']} text={r['director_text']!r}")
        print(f"  base_mag={r['base_mag']} peak_total={r['peak_total']}")
        for c in r["critical"]:
            print(f"  !! CRIT  {c}")
        for w in r["warnings"]:
            print(f"  !! WARN  {w}")

    report = {
        "actions": results,
        "states": state_results,
        "summary": {"pass": n_pass, "fail": n_fail},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_pass} pass, {n_fail} fail")
    print(f"Report:  {args.out}")
    print("=" * 70)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
