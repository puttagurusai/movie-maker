"""
Make MoMask→SMPL-X hands look less stiff.

MoMask/HML22 has NO finger joints, so fingers stay open T-pose unless we pose them.
This post-process on an existing Action:
  1) Applies a natural relaxed finger curl (from catalog walk hand pose)
  2) Softens wrists with mild follow-through from forearm direction
  3) Optional slight idle variation so hands don't look locked

Usage:
  blender.exe body_motion/_momask_ours.blend --python tools/naturalize_smplx_hands.py -- ^
    --action momask_from_calib --out body_motion/_momask_ours.blend
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]

# Relaxed open-hand curl sampled from catalog walk (SMPL-X locals) + mild extra curl
# Format: bone -> quaternion (w,x,y,z)
RELAXED_HAND = {
    # Left
    "left_thumb1": (0.994, 0.05, 0.03, 0.10),
    "left_thumb2": (0.990, 0.08, 0.02, 0.05),
    "left_thumb3": (0.995, 0.06, 0.01, 0.02),
    "left_index1": (0.970, 0.00, 0.00, -0.24),
    "left_index2": (0.980, 0.00, 0.00, -0.20),
    "left_index3": (0.990, 0.00, 0.00, -0.12),
    "left_middle1": (0.960, 0.00, 0.02, -0.28),
    "left_middle2": (0.975, 0.00, 0.00, -0.22),
    "left_middle3": (0.990, 0.00, 0.00, -0.12),
    "left_ring1": (0.955, 0.00, 0.03, -0.29),
    "left_ring2": (0.975, 0.00, 0.00, -0.22),
    "left_ring3": (0.990, 0.00, 0.00, -0.12),
    "left_pinky1": (0.950, 0.00, 0.04, -0.31),
    "left_pinky2": (0.975, 0.00, 0.00, -0.20),
    "left_pinky3": (0.990, 0.00, 0.00, -0.10),
    # Right (mirror Z local roughly — x/z signs flipped for many SMPL-X fingers)
    "right_thumb1": (0.994, 0.05, -0.03, -0.10),
    "right_thumb2": (0.990, 0.08, -0.02, -0.05),
    "right_thumb3": (0.995, 0.06, -0.01, -0.02),
    "right_index1": (0.970, 0.00, 0.00, 0.24),
    "right_index2": (0.980, 0.00, 0.00, 0.20),
    "right_index3": (0.990, 0.00, 0.00, 0.12),
    "right_middle1": (0.960, 0.00, -0.02, 0.28),
    "right_middle2": (0.975, 0.00, 0.00, 0.22),
    "right_middle3": (0.990, 0.00, 0.00, 0.12),
    "right_ring1": (0.955, 0.00, -0.03, 0.29),
    "right_ring2": (0.975, 0.00, 0.00, 0.22),
    "right_ring3": (0.990, 0.00, 0.00, 0.12),
    "right_pinky1": (0.950, 0.00, -0.04, 0.31),
    "right_pinky2": (0.975, 0.00, 0.00, 0.20),
    "right_pinky3": (0.990, 0.00, 0.00, 0.10),
}


def log(msg: str) -> None:
    print(f"[hands] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--action", default="momask_from_calib")
    p.add_argument("--out", default="")
    p.add_argument("--curl", type=float, default=1.0, help="0=open, 1=full relaxed curl")
    p.add_argument("--wrist_soft", type=float, default=0.35, help="extra wrist softness 0-1")
    return p.parse_args(argv)


def assign_slot(obj) -> None:
    if obj.animation_data and obj.animation_data.action:
        try:
            slots = obj.animation_data.action_suitable_slots
            if slots:
                obj.animation_data.action_slot = slots[0]
        except Exception:
            pass


def slerp_q(a: Quaternion, b: Quaternion, t: float) -> Quaternion:
    return a.slerp(b, t)


def main():
    args = parse_args()
    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        raise SystemExit("SMPL-X_Armature missing")

    act = bpy.data.actions.get(args.action)
    if not act:
        raise SystemExit(f"action not found: {args.action}")
    if not tgt.animation_data:
        tgt.animation_data_create()
    tgt.animation_data.action = act
    assign_slot(tgt)

    f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])
    scene = bpy.context.scene
    curl = max(0.0, min(1.0, args.curl))
    wrist_soft = max(0.0, min(1.0, args.wrist_soft))

    # Precompute relaxed quats
    relaxed = {k: Quaternion(v) for k, v in RELAXED_HAND.items()}
    identity = Quaternion((1, 0, 0, 0))

    # Sample existing wrist keys for soft secondary
    log(f"naturalize hands on action={act.name} frames={f0}-{f1} curl={curl}")

    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()

        # Subtle phase so hands aren't statue-locked (breath-like)
        phase = 0.5 + 0.5 * math.sin(2.0 * math.pi * (f - f0) / max(1, (f1 - f0) / 2.5))
        curl_f = curl * (0.85 + 0.15 * phase)

        # Fingers: blend rest → relaxed curl
        for bn, rq in relaxed.items():
            if bn not in tgt.pose.bones:
                continue
            pb = tgt.pose.bones[bn]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = slerp_q(identity, rq, curl_f)
            pb.keyframe_insert("rotation_quaternion", frame=f)

        # Wrists: keep existing aim, add slight flex toward relaxed hang + micro motion
        for side in ("left", "right"):
            wn = f"{side}_wrist"
            en = f"{side}_elbow"
            if wn not in tgt.pose.bones:
                continue
            pb = tgt.pose.bones[wn]
            pb.rotation_mode = "QUATERNION"
            q0 = pb.rotation_quaternion.copy()
            # mild flex: small Euler offset in local space (natural hanging wrist)
            sign = 1.0 if side == "left" else -1.0
            flex = Euler(
                (
                    0.08 * wrist_soft * (0.7 + 0.3 * phase),
                    0.05 * wrist_soft * sign * math.sin(2 * math.pi * (f - f0) / 40.0),
                    -0.12 * wrist_soft * sign,
                ),
                "XYZ",
            ).to_quaternion()
            pb.rotation_quaternion = (q0 @ flex).normalized()
            pb.keyframe_insert("rotation_quaternion", frame=f)

        if f == f0 or f % 40 == 0 or f == f1:
            li = tgt.pose.bones.get("left_index1")
            lw = tgt.pose.bones.get("left_wrist")
            log(
                f"f{f}: L_index1={[round(x,3) for x in li.rotation_quaternion] if li else None} "
                f"L_wrist={[round(x,3) for x in lw.rotation_quaternion] if lw else None}"
            )

    scene.frame_set(f0)
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        try:
            bpy.ops.wm.save_as_mainfile(filepath=str(out))
        except RuntimeError:
            alt = out.with_name(out.stem + "_hands" + out.suffix)
            bpy.ops.wm.save_as_mainfile(filepath=str(alt))
            out = alt
        log(f"saved {out}")
    else:
        try:
            bpy.ops.wm.save_mainfile()
        except Exception:
            out = ROOT / "body_motion" / "_momask_ours_hands.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(out))
            log(f"saved {out}")
    log("DONE natural hands")


if __name__ == "__main__":
    main()
