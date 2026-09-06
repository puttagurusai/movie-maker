"""
Export SMPL-X_Armature REST pose as a full BVH for make_momask_template.py.

Fixes the manual issues you hit:
  - no action (clears pose)
  - clear object transforms → origin
  - correct facing / Y-up MoMask axes
  - root centered (XZ=0), hip height kept

Output goes to:
  third_party/momask-codes/visualization/data/our_smplx_rest.bvh

Then:
  python tools/build_our_momask_template.py
  → our_momask_template.bvh (used by Joint2BVH / agents)

Usage:
  blender.exe whole_body_retargeted.blend --background --python tools/export_smplx_rest_for_momask_template.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    ROOT
    / "third_party"
    / "momask-codes"
    / "visualization"
    / "data"
    / "our_smplx_rest.bvh"
)

# MoMask 22 joints we care about (export full tree; template script strips)
KEEP = {
    "pelvis",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "spine1",
    "spine2",
    "spine3",
    "neck",
    "head",
    "left_collar",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_collar",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
}


def log(msg: str) -> None:
    print(f"[export_rest_bvh] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--armature", default="SMPL-X_Armature")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p.parse_args(argv)


def clear_pose(arm):
    if arm.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.transforms_clear()
    bpy.ops.object.mode_set(mode="OBJECT")


def clear_object_transform(arm):
    """Put armature object at origin, no rotation/scale — rest is pure bone data."""
    arm.location = (0.0, 0.0, 0.0)
    arm.rotation_euler = (0.0, 0.0, 0.0)
    arm.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    arm.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()


def rest_head_world(arm, name: str) -> Vector:
    b = arm.data.bones[name]
    return arm.matrix_world @ b.head_local


def blender_zup_to_yup(p: Vector) -> np.ndarray:
    """(x,y,z)_blender Z-up → (x,z,-y)_Y-up MoMask/HML convention."""
    return np.array([float(p.x), float(p.z), float(-p.y)], dtype=np.float64)


def write_rest_bvh(arm, out: Path) -> None:
    """
    Write a full-hierarchy-ish rest BVH with only the 22 body joints
    (enough for make_momask_template). Offsets from rest heads in Y-up.
    Root centered: XZ=0, Y=hip height.
    """
    missing = [n for n in KEEP if n not in arm.data.bones]
    if missing:
        raise SystemExit(f"missing bones: {missing}")

    # Collect Y-up rest heads
    heads = {n: blender_zup_to_yup(rest_head_world(arm, n)) for n in KEEP}

    # Center: subtract root XZ, keep heights relative
    root = heads["pelvis"].copy()
    # Shift all so pelvis XZ=0; keep absolute Y as hip height
    shift = np.array([root[0], 0.0, root[2]])
    for n in heads:
        heads[n] = heads[n] - shift
    # Zero tiny numerical XZ on pelvis
    heads["pelvis"][0] = 0.0
    heads["pelvis"][2] = 0.0

    # Hierarchy for offsets (parent → children)
    hierarchy = {
        "pelvis": ["left_hip", "right_hip", "spine1"],
        "left_hip": ["left_knee"],
        "left_knee": ["left_ankle"],
        "left_ankle": ["left_foot"],
        "right_hip": ["right_knee"],
        "right_knee": ["right_ankle"],
        "right_ankle": ["right_foot"],
        "spine1": ["spine2"],
        "spine2": ["spine3"],
        "spine3": ["neck", "left_collar", "right_collar"],
        "neck": ["head"],
        "left_collar": ["left_shoulder"],
        "left_shoulder": ["left_elbow"],
        "left_elbow": ["left_wrist"],
        "right_collar": ["right_shoulder"],
        "right_shoulder": ["right_elbow"],
        "right_elbow": ["right_wrist"],
        "left_foot": [],
        "right_foot": [],
        "head": [],
        "left_wrist": [],
        "right_wrist": [],
    }

    def offset(child, parent):
        if parent is None:
            return heads[child]
        return heads[child] - heads[parent]

    lines = ["HIERARCHY"]

    def write_joint(name: str, parent: str | None, depth: int, is_root: bool):
        ind = "\t" * depth
        tag = "ROOT" if is_root else "JOINT"
        off = offset(name, parent)
        lines.append(f"{ind}{tag} {name}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind}\tOFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
        if is_root:
            lines.append(
                f"{ind}\tCHANNELS 6 Xposition Yposition Zposition "
                f"Zrotation Yrotation Xrotation"
            )
        else:
            # MoMask official channel order
            lines.append(f"{ind}\tCHANNELS 3 Zrotation Yrotation Xrotation")
        kids = hierarchy.get(name, [])
        if not kids:
            lines.append(f"{ind}\tEnd Site")
            lines.append(f"{ind}\t{{")
            lines.append(f"{ind}\t\tOFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{ind}\t}}")
        else:
            for c in kids:
                write_joint(c, name, depth + 1, False)
        lines.append(f"{ind}}}")

    write_joint("pelvis", None, 0, True)
    lines.append("MOTION")
    lines.append("Frames: 1")
    lines.append("Frame Time: 0.050000")
    # root pos + rot + 21 * 3 rots = 6 + 63 = 69
    vals = [0.0] * (6 + 21 * 3)
    # root position uses offset? Motion zeros — rest pose in offsets only
    lines.append(" ".join(f"{v:.6f}" for v in vals))
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    # Diagnostics
    log(f"pelvis Y-up={heads['pelvis']}")
    log(f"L foot={heads['left_foot']} R foot={heads['right_foot']}")
    log(f"head={heads['head']}")
    leg = heads["left_knee"] - heads["left_hip"]
    spine = heads["spine2"] - heads["spine1"]
    log(f"L upperleg dir={leg} (expect mostly -Y)")
    log(f"spine1→2 dir={spine} (expect mostly +Y)")
    log(f"wrote {out}")


def main():
    args = parse_args()
    arm = bpy.data.objects.get(args.armature)
    if arm is None or arm.type != "ARMATURE":
        raise SystemExit(f"need armature {args.armature}")

    # Clear animation influence for export
    if arm.animation_data:
        arm.animation_data.action = None

    clear_object_transform(arm)
    clear_pose(arm)
    bpy.context.view_layer.update()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    write_rest_bvh(arm, out)
    log("DONE — next: make_momask_template.py on this file")


if __name__ == "__main__":
    main()
