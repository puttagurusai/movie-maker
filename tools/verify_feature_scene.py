#!/usr/bin/env python3
"""
Verify what may enter a comparison MP4.

Clothes rule (product):
  - Full Quaternius characters are NOT wearable clothes on our SMPL-X hero.
  - Only pass hero clothes if _verify_wardrobe_worn says garment fit OK.
  - Otherwise hero stays BodyMesh + ARKit face; NPCs may still use those characters.

Prints a JSON report. Exit 0 if scene+NPCs(+optional clothes) are acceptable for export.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    # This script is meant to be executed INSIDE Blender (bpy available).
    try:
        import bpy
    except ImportError:
        print("Run inside Blender or via blender --python")
        return 2

    sys.path.insert(0, str(ROOT))
    import importlib.util

    spec = importlib.util.spec_from_file_location("br", ROOT / "blender_receiver.py")
    br = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(br)

    # Try casual clothes, then verify
    tried = br._apply_wardrobe("casual_01", actor="hero", require_fit=True)
    ver = br._verify_wardrobe_worn(tried if tried != "hero_default" else "casual_01")
    if tried == "hero_default":
        ver = {
            "ok": True,
            "outfit_id": "hero_default",
            "kind": "none",
            "reason": "rejected non-garment asset; hero body+face only",
        }

    look_n = len([o for o in bpy.data.objects if o.name.startswith("Look_")])
    npc_n = len([o for o in bpy.data.objects if o.name.startswith("NPC_") and not o.hide_get()])
    body = bpy.data.objects.get("BodyMesh")
    face = None
    for o in bpy.data.objects:
        if o.type == "MESH" and o.data and o.data.shape_keys and "head" in o.name.lower():
            face = o
            break
    body_ok = body is not None and (tried == "hero_default" or not body.hide_get() or ver.get("kind") == "garment")
    # After rejected clothes, body must be visible
    if tried == "hero_default" and body is not None:
        try:
            body.hide_set(False)
            body.hide_render = False
            body.hide_viewport = False
        except Exception:
            pass
        body_ok = not body.hide_get()

    face_ok = face is not None and face.data.shape_keys is not None
    scene_ok = look_n >= 3
    npc_ok = npc_n >= 2

    report = {
        "clothes": ver,
        "applied_outfit": tried,
        "include_clothes_in_mp4": bool(ver.get("ok") and ver.get("kind") == "garment"),
        "scene": {"look_objects": look_n, "ok": scene_ok},
        "npcs": {"visible": npc_n, "ok": npc_ok},
        "hero": {
            "body": body.name if body else None,
            "body_visible": (not body.hide_get()) if body else False,
            "face": face.name if face else None,
            "face_ok": face_ok,
            "ok": bool(body_ok and face_ok),
        },
        "export_ok": bool(scene_ok and npc_ok and face_ok and body_ok),
    }
    print(json.dumps(report, indent=2))
    out = ROOT / "temp" / "movies" / "feature_verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["export_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
