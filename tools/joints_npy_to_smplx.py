"""
MoMask joints (T,22,3) → SMPL-X_Armature Action  (OUR skeleton lengths + axes)

This is the right place to match *our* avatar — NOT rewriting official
joints2bvh template.bvh (that template is Mixamo-family; changing names alone
does not fix rest axes / bone lengths).

Flow (same root + foot path as MoMask Joint2BVHConvertor):
  official gen_t2m → joints/*.npy
    → prefer sibling *_ik.npy  (MoMask remove_fs foot IK + FK recon)
       or run remove_fs + BasicIK via --run-momask-ik
    → absolute root: pelvis tracks full hips trajectory every frame
    → scale root height to SMPL-X hips→head
    → Y-up → Blender Z-up
    → swing-only aim on each SMPL-X bone (source direction × OUR length)
    → bake Action (pelvis location keyframes = walk travel)

Usage:
  blender.exe whole_body_retargeted.blend --background --python tools/joints_npy_to_smplx.py -- ^
    --joints third_party/momask-codes/generation/.../sample0_repeat0_len80.npy ^
    --action momask_joints_direct ^
    --out body_motion/momask_cache/official_t2m_scale/walk_joints_direct.blend
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
MOMASK = ROOT / "third_party" / "momask-codes"

# HumanML3D / MoMask 22 joint index → SMPL-X bone (head of bone ≈ joint)
# See body_motion/HML22_TO_SMPLX_MAP.txt
HML_IDX_TO_NAME = {
    0: "Hips",
    1: "LeftUpLeg",
    2: "RightUpLeg",
    3: "Spine",
    4: "LeftLeg",
    5: "RightLeg",
    6: "Spine1",
    7: "LeftFoot",
    8: "RightFoot",
    9: "Spine2",
    10: "LeftToe",
    11: "RightToe",
    12: "Neck",
    13: "LeftShoulder",
    14: "RightShoulder",
    15: "Head",
    16: "LeftArm",
    17: "RightArm",
    18: "LeftForeArm",
    19: "RightForeArm",
    20: "LeftHand",
    21: "RightHand",
}

# Swing edges: (parent_smplx, child_smplx, src_joint_a, src_joint_b)
# Aim parent so child head follows source direction with OUR bone length.
AIM_EDGES = [
    ("pelvis", "spine1", "Hips", "Spine"),
    ("spine1", "spine2", "Spine", "Spine1"),
    ("spine2", "spine3", "Spine1", "Spine2"),
    ("spine3", "neck", "Spine2", "Neck"),
    ("neck", "head", "Neck", "Head"),
    ("left_hip", "left_knee", "LeftUpLeg", "LeftLeg"),
    ("left_knee", "left_ankle", "LeftLeg", "LeftFoot"),
    ("left_ankle", "left_foot", "LeftFoot", "LeftToe"),
    ("right_hip", "right_knee", "RightUpLeg", "RightLeg"),
    ("right_knee", "right_ankle", "RightLeg", "RightFoot"),
    ("right_ankle", "right_foot", "RightFoot", "RightToe"),
    # Arms: collar from spine3→shoulder joint; shoulder from arm chain
    ("left_collar", "left_shoulder", "LeftShoulder", "LeftArm"),
    ("left_shoulder", "left_elbow", "LeftArm", "LeftForeArm"),
    ("left_elbow", "left_wrist", "LeftForeArm", "LeftHand"),
    ("right_collar", "right_shoulder", "RightShoulder", "RightArm"),
    ("right_shoulder", "right_elbow", "RightArm", "RightForeArm"),
    ("right_elbow", "right_wrist", "RightForeArm", "RightHand"),
]

# Hip bones aimed from pelvis using hip→knee direction
HIP_EDGES = [
    ("left_hip", "left_knee", "LeftUpLeg", "LeftLeg"),
    ("right_hip", "right_knee", "RightUpLeg", "RightLeg"),
]


def log(msg: str) -> None:
    print(f"[joints_smplx] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--joints", required=True, help="(T,22,3) .npy from official MoMask")
    p.add_argument("--action", default="momask_joints_direct")
    p.add_argument("--out", default="")
    p.add_argument("--armature", default="SMPL-X_Armature")
    # absolute = full hips XY/Z travel (MoMask BVH root). Default so walks move.
    p.add_argument(
        "--root",
        default="absolute",
        choices=("relative", "absolute", "inplace"),
        help="absolute=full root travel (MoMask default); relative=start near rest; inplace=freeze XY",
    )
    p.add_argument(
        "--prefer-ik",
        action="store_true",
        default=True,
        help="If joints path has no _ik suffix, load sibling *_ik.npy when present (default on)",
    )
    p.add_argument(
        "--no-prefer-ik",
        action="store_true",
        help="Do not auto-switch to sibling *_ik.npy",
    )
    p.add_argument(
        "--run-momask-ik",
        action="store_true",
        help="Run MoMask Joint2BVHConvertor foot_ik (remove_fs + bone IK) even without *_ik.npy",
    )
    p.add_argument(
        "--lock-collars",
        action="store_true",
        default=False,
        help="Keep collars at rest (less chest twist; arms only from shoulder)",
    )
    return p.parse_args(argv)


def resolve_joints_path(jpath: Path, prefer_ik: bool) -> Path:
    """Prefer MoMask foot-IK joints (*_ik.npy) — same glb as their walking BVH path."""
    if not prefer_ik:
        return jpath
    stem = jpath.stem
    if stem.endswith("_ik"):
        return jpath
    ik = jpath.with_name(f"{stem}_ik{jpath.suffix}")
    if ik.is_file():
        log(f"using MoMask foot-IK joints: {ik.name}")
        return ik
    return jpath


def apply_momask_foot_ik(joints: np.ndarray) -> np.ndarray:
    """
    Same as visualization.joints2bvh.Joint2BVHConvertor.convert(..., foot_ik=True):
    reorder → remove_fs → BasicInverseKinematics → global joints in HML order.
    """
    if not MOMASK.is_dir():
        raise SystemExit(f"--run-momask-ik needs {MOMASK}")
    prev = os.getcwd()
    try:
        os.chdir(MOMASK)
        if str(MOMASK) not in sys.path:
            sys.path.insert(0, str(MOMASK))
        from visualization.joints2bvh import Joint2BVHConvertor

        conv = Joint2BVHConvertor()
        _anim, glb = conv.convert(
            joints.astype(np.float32),
            filename=None,
            iterations=100,
            foot_ik=True,
        )
        out = np.asarray(glb, dtype=np.float64)
        log(f"MoMask foot_ik applied  glb={out.shape}")
        return out
    finally:
        os.chdir(prev)


def assign_slot(obj) -> None:
    if obj.animation_data and obj.animation_data.action:
        try:
            slots = obj.animation_data.action_suitable_slots
            if slots:
                obj.animation_data.action_slot = slots[0]
        except Exception:
            pass


def y_up_to_z_up(p: np.ndarray) -> np.ndarray:
    """(T,J,3) Y-up → Blender Z-up: (x,y,z)_yup → (x, -z, y)_zup"""
    out = np.empty_like(p)
    out[..., 0] = p[..., 0]
    out[..., 1] = -p[..., 2]
    out[..., 2] = p[..., 1]
    return out


def rest_head_world(arm, name: str) -> Vector:
    return (arm.matrix_world @ arm.data.bones[name].head_local).copy()


def world_head(arm, name: str) -> Vector:
    return (arm.matrix_world @ arm.pose.bones[name].head).copy()


def rest_length(arm, parent: str, child: str) -> float:
    return (arm.data.bones[child].head_local - arm.data.bones[parent].head_local).length


def clear_pose(arm) -> None:
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
        pb.location = Vector((0, 0, 0))
    bpy.context.view_layer.update()


def set_pelvis_location(arm, desired_w: Vector) -> None:
    """Place root pelvis head at desired_w (absolute walk travel).

    For a parentless root bone, Blender uses:
        matrix = matrix_local @ T(pose.location) @ R(pose.rotation)
    so the bone origin (head) in armature space is:
        head_arm = R_rest @ location + t_rest
    independent of pose rotation. Solve location in closed form, then refine.
    """
    pb = arm.pose.bones["pelvis"]
    bone = arm.data.bones["pelvis"]
    rest = bone.matrix_local
    rest_r = rest.to_3x3()
    rest_t = rest.to_translation()
    rest_r_inv = rest_r.inverted()
    arm_inv = arm.matrix_world.inverted()

    desired_arm = arm_inv @ desired_w
    # closed-form for root origin
    pb.location = rest_r_inv @ (desired_arm - rest_t)
    bpy.context.view_layer.update()

    # refine if head != origin (custom rest) or armature non-uniform scale
    for _ in range(6):
        cur = arm.matrix_world @ pb.head
        err_w = desired_w - cur
        if err_w.length < 1e-5:
            break
        err_arm = arm_inv.to_3x3() @ err_w
        pb.location = pb.location + rest_r_inv @ err_arm
        bpy.context.view_layer.update()




def swing_aim_bone(arm, bone_name: str, child_name: str, desired_child_w: Vector) -> None:
    """SWING-ONLY: rest local → minimal rotation so rest limb aims at desired child.

    CRITICAL: never wipe pelvis location — that is the walk root travel
    (same role as new_anim.positions[:,0] in MoMask joints2bvh).
    """
    pb = arm.pose.bones[bone_name]
    if child_name not in arm.pose.bones:
        return
    saved_loc = pb.location.copy()
    is_root = bone_name == "pelvis" or pb.parent is None

    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
    if is_root:
        pb.location = saved_loc  # keep walk root
    else:
        pb.location = Vector((0, 0, 0))
    bpy.context.view_layer.update()

    head_w = arm.matrix_world @ pb.head
    child_rest_w = arm.matrix_world @ arm.pose.bones[child_name].head
    v_rest = child_rest_w - head_w
    v_tgt = desired_child_w - head_w
    if v_rest.length < 1e-8 or v_tgt.length < 1e-8:
        if is_root:
            pb.location = saved_loc
        return
    v_rest.normalize()
    v_tgt.normalize()
    if v_rest.dot(v_tgt) > 0.999999:
        if is_root:
            pb.location = saved_loc
        return

    q_swing = v_rest.rotation_difference(v_tgt)
    M = arm.matrix_world @ pb.matrix
    head = M.to_translation()
    M_new = Matrix.Translation(head) @ q_swing.to_matrix().to_4x4() @ Matrix.Translation(-head) @ M
    mat_arm = arm.matrix_world.inverted() @ M_new
    bone = arm.data.bones[bone_name]
    if pb.parent:
        pre = pb.parent.matrix @ pb.parent.bone.matrix_local.inverted() @ bone.matrix_local
        basis = pre.inverted() @ mat_arm
    else:
        basis = bone.matrix_local.inverted() @ mat_arm
    _loc, rot, _sc = basis.decompose()
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = rot
    if is_root:
        pb.location = saved_loc
    else:
        pb.location = Vector((0, 0, 0))
    bpy.context.view_layer.update()


def frame_src_pos(joints_t: np.ndarray) -> dict[str, Vector]:
    """joints_t: (22,3) already Z-up."""
    pos = {}
    for i, name in HML_IDX_TO_NAME.items():
        pos[name] = Vector((float(joints_t[i, 0]), float(joints_t[i, 1]), float(joints_t[i, 2])))
    return pos


def apply_frame(arm, src_pos: dict[str, Vector], lock_collars: bool) -> None:
    clear_pose(arm)
    # 1) root travel first (MoMask: new_anim.positions[:,0] = positions[:,0])
    set_pelvis_location(arm, src_pos["Hips"])

    def aim(parent: str, child: str, sa: str, sb: str):
        if parent not in arm.pose.bones or child not in arm.pose.bones:
            return
        if sa not in src_pos or sb not in src_pos:
            return
        bpy.context.view_layer.update()
        p = world_head(arm, parent)
        d = src_pos[sb] - src_pos[sa]
        if d.length < 1e-8:
            d = rest_head_world(arm, child) - rest_head_world(arm, parent)
        d = d.normalized()
        L = rest_length(arm, parent, child)
        if L < 1e-8:
            return
        swing_aim_bone(arm, parent, child, p + d * L)

    # Torso (pelvis aim must PRESERVE location — fixed in swing_aim_bone)
    for parent, child, sa, sb in AIM_EDGES[:5]:
        aim(parent, child, sa, sb)

    # Legs
    for parent, child, sa, sb in AIM_EDGES[5:11]:
        aim(parent, child, sa, sb)

    # Arms
    for parent, child, sa, sb in AIM_EDGES[11:]:
        if lock_collars and child in ("left_collar", "right_collar"):
            continue
        if lock_collars and parent in ("left_collar", "right_collar"):
            aim(parent, child, sa, sb)
            continue
        aim(parent, child, sa, sb)

    # Re-apply pelvis location after torso aim in case anything drifted
    set_pelvis_location(arm, src_pos["Hips"])


def main():
    args = parse_args()
    jpath = Path(args.joints)
    if not jpath.is_absolute():
        jpath = ROOT / jpath
    if not jpath.is_file():
        raise SystemExit(f"missing joints {jpath}")

    prefer_ik = bool(args.prefer_ik) and not bool(args.no_prefer_ik)
    jpath = resolve_joints_path(jpath, prefer_ik=prefer_ik)

    joints = np.load(str(jpath)).astype(np.float64)
    if joints.ndim != 3 or joints.shape[1] != 22 or joints.shape[2] != 3:
        raise SystemExit(f"need (T,22,3), got {joints.shape}")

    used_ik = jpath.stem.endswith("_ik")
    if args.run_momask_ik and not used_ik:
        joints = apply_momask_foot_ik(joints)
        used_ik = True

    arm = bpy.data.objects.get(args.armature)
    if arm is None or arm.type != "ARMATURE":
        raise SystemExit(f"need armature {args.armature}")

    # Y-up → Z-up
    joints_z = y_up_to_z_up(joints)
    T = joints_z.shape[0]

    # Scale so mean hips→head matches our rest
    h_src = np.linalg.norm(joints_z[:, 15] - joints_z[:, 0], axis=-1).mean()
    h_tgt = (rest_head_world(arm, "head") - rest_head_world(arm, "pelvis")).length
    scale = float(h_tgt / h_src) if h_src > 1e-6 else 1.0
    joints_z = joints_z * scale

    root_mode = args.root
    # Anchor: put first-frame hips on our rest pelvis (still keep later-frame deltas).
    p0 = joints_z[0, 0].copy()
    pelvis_rest = rest_head_world(arm, "pelvis")
    if root_mode == "absolute":
        # MoMask BVH style: full trajectory, only shift so f0 sits on rest pelvis
        offset0 = np.array(
            [pelvis_rest.x - p0[0], pelvis_rest.y - p0[1], pelvis_rest.z - p0[2]]
        )
        joints_z = joints_z + offset0
    elif root_mode == "relative":
        # same as absolute for joints path (already clip-local); keep for API parity
        offset0 = np.array(
            [pelvis_rest.x - p0[0], pelvis_rest.y - p0[1], pelvis_rest.z - p0[2]]
        )
        joints_z = joints_z + offset0
    else:  # inplace — zero horizontal travel after centering
        offset0 = np.array(
            [pelvis_rest.x - p0[0], pelvis_rest.y - p0[1], pelvis_rest.z - p0[2]]
        )
        joints_z = joints_z + offset0

    hip_travel = joints_z[:, 0] - joints_z[0, 0]
    hip_travel_max = float(np.linalg.norm(hip_travel, axis=-1).max())
    log(
        f"joints {joints.shape} file={jpath.name} foot_ik={used_ik} root={root_mode} "
        f"scale×{scale:.4f} h_src={h_src:.3f} h_tgt={h_tgt:.3f} frames={T} "
        f"hip_travel_max={hip_travel_max:.4f}m end_delta={hip_travel[-1].tolist()}"
    )

    if not arm.animation_data:
        arm.animation_data_create()
    old = bpy.data.actions.get(args.action)
    if old:
        if arm.animation_data.action == old:
            arm.animation_data.action = None
        bpy.data.actions.remove(old)
    act = bpy.data.actions.new(args.action)
    act.use_fake_user = True
    arm.animation_data.action = act
    assign_slot(arm)

    hips0 = Vector(joints_z[0, 0])

    key_bones = [
        "pelvis", "spine1", "spine2", "spine3", "neck", "head",
        "left_hip", "left_knee", "left_ankle", "left_foot",
        "right_hip", "right_knee", "right_ankle", "right_foot",
        "left_collar", "left_shoulder", "left_elbow", "left_wrist",
        "right_collar", "right_shoulder", "right_elbow", "right_wrist",
    ]

    pelvis_w_samples = []

    for t in range(T):
        f = t + 1
        src_pos = frame_src_pos(joints_z[t])
        if root_mode == "inplace":
            d = src_pos["Hips"] - hips0
            src_pos["Hips"] = Vector((hips0.x, hips0.y, src_pos["Hips"].z))
            for k in list(src_pos.keys()):
                if k == "Hips":
                    continue
                src_pos[k] = Vector((
                    src_pos[k].x - d.x,
                    src_pos[k].y - d.y,
                    src_pos[k].z,
                ))
        apply_frame(arm, src_pos, lock_collars=bool(args.lock_collars))

        for bn in key_bones:
            if bn not in arm.pose.bones:
                continue
            pb = arm.pose.bones[bn]
            pb.rotation_mode = "QUATERNION"
            pb.keyframe_insert("rotation_quaternion", frame=f)
            if bn == "pelvis":
                pb.keyframe_insert("location", frame=f)

        if t == 0 or t % 20 == 0 or t == T - 1:
            pw = world_head(arm, "pelvis")
            pelvis_w_samples.append((f, pw.x, pw.y, pw.z))
            sh = world_head(arm, "left_shoulder")
            el = world_head(arm, "left_elbow")
            ploc = arm.pose.bones["pelvis"].location
            log(
                f"  f{f}: pelvis_w=({pw.x:.3f},{pw.y:.3f},{pw.z:.3f}) "
                f"pelvis_loc=({ploc.x:.4f},{ploc.y:.4f},{ploc.z:.4f}) "
                f"L_arm={(el-sh).length:.3f}"
            )

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = T
    bpy.context.scene.frame_set(1)

    if len(pelvis_w_samples) >= 2:
        a, b = pelvis_w_samples[0], pelvis_w_samples[-1]
        log(
            f"pelvis world travel f{a[0]}→f{b[0]}: "
            f"d=({b[1]-a[1]:.4f},{b[2]-a[2]:.4f},{b[3]-a[3]:.4f}) "
            f"|d|={((b[1]-a[1])**2+(b[2]-a[2])**2+(b[3]-a[3])**2)**0.5:.4f}"
        )

    out = args.out.strip()
    if out:
        out_p = Path(out)
        if not out_p.is_absolute():
            out_p = ROOT / out_p
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fp = str(out_p.resolve()).replace("\\", "/")
        try:
            bpy.ops.wm.save_as_mainfile(filepath=fp, compress=True)
        except Exception:
            bpy.ops.wm.save_as_mainfile(filepath=fp, check_existing=False)
        log(f"saved {out_p}")
    meta = {
        "action": args.action,
        "joints": str(jpath),
        "scale": scale,
        "frames": T,
        "root_mode": root_mode,
        "foot_ik": used_ik,
        "hip_travel_max_m": hip_travel_max,
        "method": "joints_npy_to_smplx_swing_ik",
        "note": "Absolute root + MoMask *_ik foot lock; source directions × OUR bone lengths",
    }
    if out:
        meta_path = Path(out).resolve().parent / f"{args.action}_joints_direct.json"
    else:
        meta_path = ROOT / "body_motion" / "momask_cache" / f"{args.action}_joints_direct.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"meta {meta_path}")
    log("DONE")


if __name__ == "__main__":
    main()

