"""
Ensure body Actions / scene never drive mouth/jaw.

Face pipeline (blender_receiver viseme/emotion) owns mouth shape keys.
Run once on a production blend after retarget if inner mouth moves during body playback.

  blender.exe whole_body_retargeted.blend --background --python tools/strip_face_from_body_actions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import bpy

MOUTH_PREFIXES = ("jaw", "mouth", "tongue")
MOUTH_EXACT = {
    "jawOpen", "jawForward", "jawLeft", "jawRight", "mouthClose",
    "mouthFunnel", "mouthPucker", "tongueOut",
}


def main() -> None:
    cleared = []
    for mesh in bpy.data.meshes:
        sk = mesh.shape_keys
        if not sk:
            continue
        if sk.animation_data:
            name = sk.animation_data.action.name if sk.animation_data.action else None
            sk.animation_data_clear()
            cleared.append((mesh.name, name))
        for kb in sk.key_blocks:
            if kb.name == "Basis":
                continue
            ln = kb.name.lower()
            if kb.name in MOUTH_EXACT or ln.startswith(MOUTH_PREFIXES):
                kb.value = 0.0

    for obj_name in (
        "teeth_ORIGINAL",
        "head_lod0_ORIGINAL",
        "eyeLeft_ORIGINAL",
        "eyeRight_ORIGINAL",
    ):
        o = bpy.data.objects.get(obj_name)
        if o and o.animation_data:
            o.animation_data_clear()

    arm = bpy.data.objects.get("SMPL-X_Armature")
    if arm and "jaw" in arm.pose.bones:
        j = arm.pose.bones["jaw"]
        j.rotation_mode = "QUATERNION"
        j.rotation_quaternion = (1, 0, 0, 0)
        j.location = (0, 0, 0)

    scene = bpy.context.scene
    scene.render.fps = 30
    try:
        scene.sync_mode = "FRAME_DROP"
    except Exception:
        pass

    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        for m in o.modifiers:
            if m.type in {"SUBSURF", "DATA_TRANSFER"}:
                m.show_viewport = False

    out = Path(bpy.data.filepath) if bpy.data.filepath else Path("cleaned.blend")
    bpy.ops.wm.save_mainfile()
    print(f"[strip_face] cleared shape-key anim on {cleared}; saved {out}")


if __name__ == "__main__":
    main()
