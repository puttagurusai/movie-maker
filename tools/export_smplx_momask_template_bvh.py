"""
Export SMPL-X_Armature rest pose as a MoMask-compatible template.bvh
(same hierarchy/names as visualization/data/template.bvh, OUR bone lengths).

Usage:
  blender.exe whole_body_retargeted.blend --background --python tools/export_smplx_momask_template_bvh.py -- ^
    --out body_motion/smplx_momask_template.bvh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# MoMask template bone order (matches Joint2BVHConvertor.template.names)
BVH_NAMES = [
    "Hips",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
    "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
]

# parents for that order (from joints2bvh.py)
PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]

# Map BVH name → our SMPL-X bone (head of bone = joint)
BVH_TO_SMPLX = {
    "Hips": "pelvis",
    "LeftUpLeg": "left_hip",
    "LeftLeg": "left_knee",
    "LeftFoot": "left_ankle",
    "LeftToe": "left_foot",
    "RightUpLeg": "right_hip",
    "RightLeg": "right_knee",
    "RightFoot": "right_ankle",
    "RightToe": "right_foot",
    "Spine": "spine1",
    "Spine1": "spine2",
    "Spine2": "spine3",
    "Neck": "neck",
    "Head": "head",
    "LeftShoulder": "left_collar",
    "LeftArm": "left_shoulder",
    "LeftForeArm": "left_elbow",
    "LeftHand": "left_wrist",
    "RightShoulder": "right_collar",
    "RightArm": "right_shoulder",
    "RightForeArm": "right_elbow",
    "RightHand": "right_wrist",
}


def log(msg: str) -> None:
    print(f"[export_tpl] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--armature", default="SMPL-X_Armature")
    p.add_argument(
        "--out",
        default=str(ROOT / "body_motion" / "smplx_momask_template.bvh"),
    )
    return p.parse_args(argv)


def blender_zup_to_yup(p) -> np.ndarray:
    """(x,y,z)_blender → (x,z,-y)_yup  inverse of (x,y,z)_yup→(x,-z,y)"""
    return np.array([float(p[0]), float(p[2]), float(-p[1])], dtype=np.float64)


def rest_head_world(arm, name: str):
    b = arm.data.bones[name]
    return arm.matrix_world @ b.head_local


def main():
    args = parse_args()
    arm = bpy.data.objects.get(args.armature)
    if arm is None or arm.type != "ARMATURE":
        raise SystemExit(f"need armature {args.armature}")

    # Collect rest heads in Y-up
    heads = []
    for bn in BVH_NAMES:
        sn = BVH_TO_SMPLX[bn]
        if sn not in arm.data.bones:
            raise SystemExit(f"missing bone {sn} for BVH {bn}")
        heads.append(blender_zup_to_yup(rest_head_world(arm, sn)))
    heads = np.stack(heads, 0)

    offsets = np.zeros((22, 3), dtype=np.float64)
    offsets[0] = heads[0]
    for i in range(1, 22):
        p = PARENTS[i]
        offsets[i] = heads[i] - heads[p]

    # Report collar vs shoulder height (Y-up height = Y)
    li, la = BVH_NAMES.index("LeftShoulder"), BVH_NAMES.index("LeftArm")
    ri, ra = BVH_NAMES.index("RightShoulder"), BVH_NAMES.index("RightArm")
    log(
        f"REST L collar_y={heads[li,1]:.4f} L shoulder_y={heads[la,1]:.4f} "
        f"dy={heads[la,1]-heads[li,1]:.4f} (SMPL-X should be non-zero)"
    )
    log(
        f"REST R collar_y={heads[ri,1]:.4f} R shoulder_y={heads[ra,1]:.4f} "
        f"dy={heads[ra,1]-heads[ri,1]:.4f}"
    )
    log(f"L collar-arm offset len={np.linalg.norm(offsets[la]):.4f}")

    # Write BVH (static rest + 1 dummy frame so BVH.load works)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    def indent(n):
        return "\t" * n

    lines = ["HIERARCHY"]

    def write_joint(idx: int, depth: int, is_root: bool = False):
        name = BVH_NAMES[idx]
        off = offsets[idx]
        kids = [i for i, p in enumerate(PARENTS) if p == idx]
        tag = "ROOT" if is_root else "JOINT"
        lines.append(f"{indent(depth)}{tag} {name}")
        lines.append(f"{indent(depth)}{{")
        lines.append(f"{indent(depth+1)}OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
        if is_root:
            lines.append(
                f"{indent(depth+1)}CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation"
            )
        else:
            lines.append(f"{indent(depth+1)}CHANNELS 3 Zrotation Yrotation Xrotation")
        if not kids:
            lines.append(f"{indent(depth+1)}End Site")
            lines.append(f"{indent(depth+1)}{{")
            lines.append(f"{indent(depth+2)}OFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{indent(depth+1)}}}")
        else:
            for c in kids:
                write_joint(c, depth + 1, False)
        lines.append(f"{indent(depth)}}}")

    write_joint(0, 0, True)
    lines.append("MOTION")
    lines.append("Frames: 1")
    lines.append("Frame Time: 0.050000")
    # one rest frame: root pos + zeros rotations
    chans = [offsets[0, 0], offsets[0, 1], offsets[0, 2]] + [0.0] * (3 + 21 * 3)
    # root has 6 channels, each other joint 3 → 6 + 21*3 = 69
    nchan = 6 + 21 * 3
    chans = [offsets[0, 0], offsets[0, 1], offsets[0, 2]] + [0.0] * (nchan - 3)
    lines.append(" ".join(f"{c:.6f}" for c in chans))

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote {out}")
    # also save npy offsets for debug
    np.savez(
        out.with_suffix(".npz"),
        names=np.array(BVH_NAMES),
        parents=np.array(PARENTS),
        offsets=offsets,
        heads_yup=heads,
        map=np.array([BVH_TO_SMPLX[n] for n in BVH_NAMES]),
    )
    log("DONE")


if __name__ == "__main__":
    main()
