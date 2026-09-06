"""Screenshot pure axis offsets on key bones to learn SMPL-X local axes."""
from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "temp" / "axis_probe"
UDP = ("127.0.0.1", 9001)
URL = "http://localhost:8000/face_viewer.html"

PROBES = [
    ("rest", {}),
    ("Rshoulder_x+0.6", {"right_shoulder": {"x": 0.6, "y": 0, "z": 0}}),
    ("Rshoulder_y+0.6", {"right_shoulder": {"x": 0, "y": 0.6, "z": 0}}),
    ("Rshoulder_z+0.6", {"right_shoulder": {"x": 0, "y": 0, "z": 0.6}}),
    ("Rshoulder_x-0.8", {"right_shoulder": {"x": -0.8, "y": 0, "z": 0}}),
    ("Rshoulder_z-0.8", {"right_shoulder": {"x": 0, "y": 0, "z": -0.8}}),
    ("Relbow_x+1.0", {"right_elbow": {"x": 1.0, "y": 0, "z": 0}}),
    ("Relbow_z-1.0", {"right_elbow": {"x": 0, "y": 0, "z": -1.0}}),
    ("Lhip_x-1.2", {"left_hip": {"x": -1.2, "y": 0, "z": 0}, "left_knee": {"x": 1.3, "y": 0, "z": 0},
                    "right_hip": {"x": -1.2, "y": 0, "z": 0}, "right_knee": {"x": 1.3, "y": 0, "z": 0},
                    "pelvis": {"x": 0.3, "y": 0, "z": 0}}),
    # candidate realistic wave (guess from probes — will re-tune after)
    ("wave_try_A", {
        "right_shoulder": {"x": 0.2, "y": -0.9, "z": -0.3},
        "right_elbow": {"x": 0.3, "y": 0.0, "z": -1.2},
        "right_wrist": {"x": 0.0, "y": 0.2, "z": -0.2},
        "spine3": {"x": 0.0, "y": 0.05, "z": -0.05},
    }),
    ("shrug_try_A", {
        "left_shoulder": {"x": -0.35, "y": 0.15, "z": 0.25},
        "right_shoulder": {"x": -0.35, "y": -0.15, "z": -0.25},
        "left_elbow": {"x": 0.9, "y": 0.2, "z": 0.3},
        "right_elbow": {"x": 0.9, "y": -0.2, "z": -0.3},
        "left_collar": {"x": 0.1, "y": 0, "z": 0.2},
        "right_collar": {"x": 0.1, "y": 0, "z": -0.2},
    }),
]


def send(sock, bones, state="standing"):
    pkt = {
        "type": "body",
        "bones": bones,
        "action": "probe",
        "t": 0.5,
        "state": state,
        "gesture_target": "none",
        "hand": "right",
        "energy": 0.2,
        "intensity": 0.8,
        "duration": 1.0,
    }
    sock.sendto(json.dumps(pkt).encode(), UDP)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 800})
        page.goto(URL, wait_until="domcontentloaded", timeout=120000)
        try:
            page.wait_for_function(
                "() => (document.getElementById('info')||{}).textContent?.includes('Avatar ready')",
                timeout=90000,
            )
        except Exception:
            page.wait_for_timeout(12000)
        page.keyboard.press("2")
        page.wait_for_timeout(500)

        for name, bones in PROBES:
            # if sit probe
            state = "sitting" if "hip" in name else "standing"
            for _ in range(10):
                send(sock, bones, state=state)
                time.sleep(0.03)
            page.wait_for_timeout(200)
            path = OUT / f"{name}.png"
            page.screenshot(path=str(path))
            print("wrote", path.name)

        browser.close()
    print("done", OUT)


if __name__ == "__main__":
    main()
