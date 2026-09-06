#!/usr/bin/env python
"""
Post-fix: shoulders sitting slightly *back* on an existing SMPL-X Action.

Does NOT re-run t2m. Opens a blend, multiplies collar/shoulder local quats
forward each frame, saves.

Usage:
  blender.exe path/to/momask_xxx.blend --background --python tools/fix_smplx_shoulder_back.py -- ^
    --action momask_official_xxx --shoulder-fwd-deg 8 --collar-fwd-deg 4 --out same.blend
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Euler

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[shoulder_fix] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--action", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--shoulder-fwd-deg", type=float, default=8.0)
    p.add_argument("--collar-fwd-deg", type=float, default=4.0)
    return p.parse_args(argv)


def assign_slot(obj) -> None:
    if obj.animation_data and obj.animation_data.action:
        try:
            slots = obj.animation_data.action_suitable_slots
            if slots:
                obj.animation_data.action_slot = slots[0]
        except Exception:
            pass


def main():
    args = parse_args()
    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        raise SystemExit("SMPL-X_Armature missing")
    act = bpy.data.actions.get(args.action)
    if not act:
        # try first action
        acts = [a for a in bpy.data.actions if a.name.startswith("momask")]
        raise SystemExit(f"action {args.action!r} not found; have={[a.name for a in bpy.data.actions][:20]}")

    if not tgt.animation_data:
        tgt.animation_data_create()
    tgt.animation_data.action = act
    assign_slot(tgt)

    fr = act.frame_range
    f0, f1 = int(fr[0]), int(fr[1])
    sf = math.radians(args.shoulder_fwd_deg)
    cf = math.radians(args.collar_fwd_deg)
    fixes = {
        "left_shoulder": Euler((sf, 0.0, -sf * 0.15), "XYZ").to_quaternion(),
        "right_shoulder": Euler((sf, 0.0, sf * 0.15), "XYZ").to_quaternion(),
        "left_collar": Euler((cf * 0.5, 0.0, -cf), "XYZ").to_quaternion(),
        "right_collar": Euler((cf * 0.5, 0.0, cf), "XYZ").to_quaternion(),
    }
    names = [n for n in fixes if n in tgt.pose.bones]
    log(f"action={args.action} frames={f0}-{f1} bones={names} sh={args.shoulder_fwd_deg}° col={args.collar_fwd_deg}°")

    for f in range(f0, f1 + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        for bname in names:
            pb = tgt.pose.bones[bname]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = fixes[bname] @ pb.rotation_quaternion
            pb.keyframe_insert("rotation_quaternion", frame=f)

    bpy.context.scene.frame_set(f0)
    out = args.out.strip()
    if out:
        out_p = Path(out)
        if not out_p.is_absolute():
            out_p = ROOT / out_p
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Blender 5: avoid "Cannot change old file (file saved with @)"
        fp = str(out_p.resolve()).replace("\\", "/")
        try:
            bpy.ops.wm.save_as_mainfile(filepath=fp, copy=True, compress=True)
        except Exception:
            bpy.ops.wm.save_as_mainfile(filepath=fp, check_existing=False)
        log(f"saved {out_p}")
    else:
        try:
            bpy.ops.wm.save_mainfile(compress=True)
        except Exception:
            bpy.ops.wm.save_mainfile()
        log("saved current blend")
    log("DONE")


if __name__ == "__main__":
    main()
