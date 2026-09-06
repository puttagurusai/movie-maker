#!/usr/bin/env python3
"""
Export a FEATURE comparison MP4 from the CONNECTED Blender scene.

Unlike export_movie_preview_mp4.py (opens a blank .blend in background),
this script expects blender_receiver already loaded with:
  - Look_Set (street/park/…)
  - Wardrobe clothes
  - Cast_Extras NPCs
  - Session body + SessionCam from a baked movie_package.json

Usage (from host Python, Blender stream running):

  python tools/export_live_feature_demo.py ^
    --package temp/movies/street_hello_30s_*/movie_package.json ^
    --look street --wardrobe casual_01 --extras 3

Or call apply_* helpers from Blender MCP then bpy.ops.session.export_preview().
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _udp(packet: dict, port: int = 9001) -> None:
    data = json.dumps(packet, ensure_ascii=False).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Chunk if large
    if len(data) < 60000:
        sock.sendto(data, ("127.0.0.1", port))
    else:
        # face receiver may expect single datagram — keep small
        sock.sendto(data[:60000], ("127.0.0.1", port))
    sock.close()


def apply_feature_look(
    *,
    location: str = "street",
    wardrobe_id: str = "casual_01",
    extras_count: int = 3,
    extras_preset: str = "sidewalk",
) -> None:
    _udp({
        "type": "look",
        "op": "apply",
        "look": {
            "location": location,
            "time_of_day": "day",
            "set_preset": f"exterior_{location}" if location in (
                "street", "park", "forest", "beach", "playground", "station",
            ) else "studio_cyc",
            "wardrobe_id": wardrobe_id,
            "extras_count": int(extras_count),
            "extras_preset": extras_preset,
        },
    })
    time.sleep(0.8)
    print(f"[demo] look={location} wardrobe={wardrobe_id} extras={extras_count}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, help="movie_package.json (glob ok)")
    ap.add_argument("--look", default="street")
    ap.add_argument("--wardrobe", default="casual_01")
    ap.add_argument("--extras", type=int, default=3)
    ap.add_argument("--extras-preset", default="sidewalk")
    ap.add_argument("--skip-look", action="store_true")
    args = ap.parse_args()

    pkg = Path(args.package)
    if "*" in args.package:
        matches = sorted(ROOT.glob(args.package), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            print("No package match")
            return 1
        pkg = matches[0]
    if not pkg.is_file():
        print(f"Missing {pkg}")
        return 1

    if not args.skip_look:
        apply_feature_look(
            location=args.look,
            wardrobe_id=args.wardrobe,
            extras_count=args.extras,
            extras_preset=args.extras_preset,
        )

    # Re-stitch audio correctly for this package
    from export_movie_preview_mp4 import _load_clips, stitch_audio_wav

    clips, fps, duration = _load_clips(pkg)
    audio = pkg.parent / "preview_audio_stitched.wav"
    stitch_audio_wav(clips, audio)
    print(f"[demo] package={pkg}")
    print(f"[demo] duration≈{duration:.1f}s clips={len(clips)} audio={audio}")
    print(
        "[demo] NEXT: in Blender run session.export_preview() "
        "OR tools/blender live OpenGL export after join-timeline"
    )
    print(
        "  Ensure join-timeline already applied body+cam, then:\n"
        "    bpy.ops.session.export_preview()\n"
        "  Output under temp/movies/session_* / session_preview.mp4"
    )
    # Write a marker the Blender side can pick up
    marker = pkg.parent / "FEATURE_DEMO_READY.json"
    marker.write_text(
        json.dumps({
            "package": str(pkg),
            "audio": str(audio),
            "look": args.look,
            "wardrobe": args.wardrobe,
            "extras": args.extras,
            "duration_s": duration,
        }, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
