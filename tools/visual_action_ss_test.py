"""
Real workflow visual test:
  1) Open face_viewer.html (Playwright)
  2) For each action: drive BodyAgent poses + emotion via UDP → ws_bridge
  3) Capture screenshots at peak pose
  4) Write JSON report with expected pose stats

Prereqs (already typical for this project):
  - HTTP server on :8000 serving web_viewer/
  - python web_viewer/ws_bridge.py on :8765 / UDP :9001

Run:
  python tools/visual_action_ss_test.py
  python tools/visual_action_ss_test.py --only wave,shrug,sitting
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import emotion_map
from face_agents.body_agent import ACTION_CLIPS, BodyAgent
from face_agents.body_director_agent import BodyDirectorAgent

OUT_DIR = ROOT / "temp" / "action_ss"
UDP_ADDR = ("127.0.0.1", 9001)
VIEW_URL = "http://localhost:8000/face_viewer.html"

# Action → NL input used in real director path
ACTION_INPUTS: Dict[str, Dict[str, Any]] = {
    "wave": {"text": "wave and say welcome", "force_action": "wave"},
    "shrug": {"text": "shrug and say I don't know", "force_action": "shrug"},
    "recoil": {"text": "oh my god what is that coming toward us", "force_action": "recoil"},
    "celebrate": {"text": "ha that is hilarious and funny", "force_action": "celebrate"},
    "hands_reject": {"text": "eww that is disgusting and gross", "force_action": "hands_reject"},
    "think_chin": {"text": "hmm let me think about that carefully", "force_action": "think_chin"},
    "point_forward": {"text": "point at that and say look over there", "force_action": "point_forward"},
    "tense": {"text": "I am furious and mad about this", "force_action": "tense"},
    "talk_emphasize": {"text": "I am so angry this is unacceptable", "force_action": "talk_emphasize"},
    "slump": {"text": "I feel really sad and unhappy", "force_action": "slump"},
    "talk_open": {"text": "hello there my friend how are you", "force_action": "talk_open"},
    "nod_yes": {"text": "nod and say yes I agree completely", "force_action": "nod_yes"},
    "shake_no": {"text": "shake your head and say no way", "force_action": "shake_no"},
    "look_camera": {"text": "look at the camera and say hi everyone", "force_action": "look_camera"},
    "sitting": {"text": "say hello while sitting", "force_action": "wave", "expect_state": "sitting"},
    "walking": {"text": "walk over and think with hand on chin", "force_action": "talk_open", "expect_state": "walking"},
}


class Ctx:
    def __init__(self, actions, state, gesture, emotion, intensity, duration, hand="right"):
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
        return 0.35 + 0.3 * abs(math.sin(self.t * 3.2))


def udp_send(sock: socket.socket, packet: dict) -> None:
    data = json.dumps(packet).encode("utf-8")
    sock.sendto(data, UDP_ADDR)


def mag(e: dict) -> float:
    return math.sqrt(
        float(e.get("x", 0) or 0) ** 2
        + float(e.get("y", 0) or 0) ** 2
        + float(e.get("z", 0) or 0) ** 2
    )


def build_case(key: str) -> Dict[str, Any]:
    meta = ACTION_INPUTS[key]
    bd = BodyDirectorAgent(None)
    beat = bd.direct(meta["text"])[0]
    force = meta.get("force_action")
    acts = list(beat.get("actions") or [])
    if force and force not in acts:
        acts = [force] + acts
    state = meta.get("expect_state") or beat.get("state") or "standing"
    if meta.get("expect_state"):
        state = meta["expect_state"]
    return {
        "key": key,
        "input": meta["text"],
        "beat": beat,
        "actions": acts,
        "state": state,
        "gesture": beat.get("gesture_target") or "none",
        "hand": beat.get("hand") or "right",
        "emotion": beat.get("emotion") or "neutral",
        "intensity": float(beat.get("intensity") or 0.85),
        "say": beat.get("text"),
    }


def stream_pose(
    sock: socket.socket,
    case: Dict[str, Any],
    duration: float = 1.35,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """Stream body + face packets; return peak pose stats."""
    agent = BodyAgent()
    ctx = Ctx(
        actions=case["actions"],
        state=case["state"],
        gesture=case["gesture"],
        emotion=case["emotion"],
        intensity=case["intensity"],
        duration=max(duration, 1.2),
        hand=case["hand"],
    )
    agent.prepare(ctx)

    # Face emotion
    face = emotion_map.get_blendshapes(case["emotion"], min(1.0, case["intensity"] * 1.2))
    udp_send(sock, {"type": "emotion", "emotion": case["emotion"], "blendshapes": face})
    udp_send(sock, {"type": "viseme", "blendshapes": {
        "jawOpen": 0.12 + 0.08 * (case["emotion"] in ("surprised", "happy")),
        "mouthSmileLeft": face.get("mouthSmileLeft", 0),
        "mouthSmileRight": face.get("mouthSmileRight", 0),
    }})

    n = int(duration * fps)
    peak_total = 0.0
    peak_bones: Dict[str, float] = {}
    peak_t = 0.0
    dt = 1.0 / fps
    for i in range(n):
        ctx.t = i * dt
        bones = agent.get_body(ctx)
        mags = {k: mag(v) for k, v in bones.items()}
        total = sum(mags.values())
        if total > peak_total:
            peak_total = total
            peak_bones = mags
            peak_t = ctx.t
        udp_send(sock, {
            "type": "body",
            "bones": bones,
            "action": agent.current_action_label(ctx),
            "t": ctx.t,
            "state": case["state"],
            "gesture_target": case["gesture"],
            "hand": case["hand"],
            "energy": ctx.speech_energy(),
            "intensity": case["intensity"],
            "duration": ctx.duration,
        })
        # small speech mouth flutter
        if i % 4 == 0:
            jaw = 0.08 + 0.12 * abs(math.sin(ctx.t * 9.0))
            udp_send(sock, {"type": "viseme", "blendshapes": {"jawOpen": jaw}})
        time.sleep(dt)

    # Hold peak a moment for screenshot stability: re-send last strong frame
    ctx.t = peak_t
    bones = agent.get_body(ctx)
    for _ in range(8):
        udp_send(sock, {
            "type": "body",
            "bones": bones,
            "action": case["key"],
            "t": peak_t,
            "state": case["state"],
            "gesture_target": case["gesture"],
            "hand": case["hand"],
            "energy": 0.5,
            "intensity": case["intensity"],
            "duration": ctx.duration,
        })
        time.sleep(0.03)

    top = sorted(peak_bones.items(), key=lambda x: -x[1])[:8]
    return {
        "peak_total": round(peak_total, 4),
        "peak_t": round(peak_t, 3),
        "top_bones": [{ "name": n, "mag": round(m, 4) } for n, m in top],
        "has_hips": any(k in peak_bones for k in ("left_hip", "right_hip", "pelvis")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of keys")
    ap.add_argument("--url", default=VIEW_URL)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    keys = list(ACTION_INPUTS.keys())
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]

    args.out.mkdir(parents=True, exist_ok=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("=" * 70)
    print("VISUAL ACTION SS TEST (real viewer + UDP body stream)")
    print(f"URL: {args.url}")
    print(f"OUT: {args.out}")
    print("=" * 70)

    from playwright.sync_api import sync_playwright

    report: Dict[str, Any] = {"cases": [], "url": args.url}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        print("Loading viewer (textures/glb may take a bit)...")
        page.goto(args.url, wait_until="domcontentloaded", timeout=120000)
        # Wait for avatar load message
        try:
            page.wait_for_function(
                """() => {
                  const el = document.getElementById('info');
                  return el && /Avatar ready|skinned body|morphs/i.test(el.textContent || '');
                }""",
                timeout=120000,
            )
        except Exception:
            print("WARN: load banner not seen — waiting extra 12s")
            page.wait_for_timeout(12000)
        page.wait_for_timeout(2500)

        # Frame upper body / face for readable SS
        page.keyboard.press("2")
        page.wait_for_timeout(400)

        rest_ss = args.out / "00_rest.png"
        page.screenshot(path=str(rest_ss), full_page=False)
        print(f"  wrote {rest_ss.name}")

        for i, key in enumerate(keys, 1):
            case = build_case(key)
            print(f"\n[{i}/{len(keys)}] {key}")
            print(f"  input: {case['input']!r}")
            print(
                f"  director: emo={case['emotion']} state={case['state']} "
                f"gest={case['gesture']}/{case['hand']} acts={case['actions']}"
            )
            print(f"  says: {case['say']!r}")

            # Camera: full body for sit/walk, upper for gestures
            if case["state"] in ("sitting", "walking", "dancing"):
                page.keyboard.press("1")
            else:
                page.keyboard.press("2")
            page.wait_for_timeout(300)

            stats = stream_pose(sock, case, duration=1.4, fps=28)
            page.wait_for_timeout(250)
            ss_path = args.out / f"{i:02d}_{key}.png"
            page.screenshot(path=str(ss_path), full_page=False)
            print(f"  peak={stats['peak_total']} @t={stats['peak_t']} top={stats['top_bones'][:4]}")
            print(f"  ss → {ss_path.name}")

            # brief return toward rest between actions
            udp_send(sock, {
                "type": "body",
                "bones": {},
                "action": "idle",
                "t": 0,
                "state": "standing",
                "gesture_target": "none",
                "hand": "right",
                "energy": 0,
                "intensity": 0.5,
                "duration": 0.5,
            })
            page.wait_for_timeout(350)

            report["cases"].append({
                **case,
                "screenshot": str(ss_path.relative_to(ROOT)),
                "pose_stats": stats,
            })

        browser.close()

    rep_path = args.out / "visual_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"Done. {len(report['cases'])} screenshots in {args.out}")
    print(f"JSON: {rep_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
