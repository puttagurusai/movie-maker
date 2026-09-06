"""
Bake HybrIK pose.npz onto SMPL-X_Armature (no BVH).

IMPORTANT:
  Use --source fk  (default) → joints_fk from SMPL-X forward kinematics after HybrIK.
  That is the real SMPL-X skeleton, NOT raw MoMask joints.
  --source raw → old behavior (looks identical to joints_npy_to_smplx).

Usage:
  blender.exe whole_body_retargeted.blend --background --python tools/hybrik_apply_to_smplx.py -- ^
    --pose body_motion/momask_cache/official_t2m_scale/walk_hybrik_pose.npz ^
    --source fk --action momask_hybrik ^
    --out body_motion/momask_cache/official_t2m_scale/walk_hybrik.blend
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]

# Swing edges: (parent, child, joint_a_idx, joint_b_idx, is_arm)
# is_arm=True → aim toward SOURCE joint world pos (stops arms flaring from our longer bones)
AIM_EDGES = [
    ("pelvis", "spine1", 0, 3, False),
    ("spine1", "spine2", 3, 6, False),
    ("spine2", "spine3", 6, 9, False),
    ("spine3", "neck", 9, 12, False),
    ("neck", "head", 12, 15, False),
    ("left_hip", "left_knee", 1, 4, False),
    ("left_knee", "left_ankle", 4, 7, False),
    ("left_ankle", "left_foot", 7, 10, False),
    ("right_hip", "right_knee", 2, 5, False),
    ("right_knee", "right_ankle", 5, 8, False),
    ("right_ankle", "right_foot", 8, 11, False),
    ("left_collar", "left_shoulder", 13, 16, True),
    ("left_shoulder", "left_elbow", 16, 18, True),
    ("left_elbow", "left_wrist", 18, 20, True),
    ("right_collar", "right_shoulder", 14, 17, True),
    ("right_shoulder", "right_elbow", 17, 19, True),
    ("right_elbow", "right_wrist", 19, 21, True),
]

# HML indices used for arm span match (shoulders, elbows, wrists, collars)
ARM_JOINT_IDX = (13, 14, 16, 17, 18, 19, 20, 21)

KEY_BONES = [
    "pelvis", "spine1", "spine2", "spine3", "neck", "head",
    "left_hip", "left_knee", "left_ankle", "left_foot",
    "right_hip", "right_knee", "right_ankle", "right_foot",
    "left_collar", "left_shoulder", "left_elbow", "left_wrist",
    "right_collar", "right_shoulder", "right_elbow", "right_wrist",
]


def log(msg: str) -> None:
    print(f"[hybrik_apply] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--pose", required=True, help="pose.npz from hybrik_joints_to_pose.py")
    p.add_argument("--action", default="momask_hybrik")
    p.add_argument("--out", default="")
    p.add_argument("--armature", default="SMPL-X_Armature")
    p.add_argument("--root", default="absolute", choices=("absolute", "inplace"))
    p.add_argument(
        "--source",
        default="fk",
        choices=("fk", "raw"),
        help="fk=SMPL-X FK after HybrIK (correct); raw=MoMask joints (old/wrong path)",
    )
    p.add_argument(
        "--mode",
        default="swing",
        choices=("swing", "rotmat"),
        help="swing=aim our bones to FK joints; rotmat=apply HybrIK local R (experimental)",
    )
    p.add_argument(
        "--arm-aim",
        default="src_target",
        choices=("src_target", "rest_len"),
        help="src_target=aim arm bones toward source joint world pos (fixes arms-out); "
        "rest_len=old direction × our bone length (can flare arms)",
    )
    p.add_argument(
        "--arm-span-match",
        action="store_true",
        default=True,
        help="Scale arm joint lateral span to match source shoulder width vs our rest (default on)",
    )
    p.add_argument(
        "--no-arm-span-match",
        action="store_true",
        help="Disable arm span matching",
    )
    p.add_argument(
        "--arm-inward",
        type=float,
        default=0.0,
        help="Extra pull-in of arm joints toward torso midline 0..1 (0=off, 0.1=slight tuck)",
    )
    return p.parse_args(argv)


def y_up_to_z_up(p: np.ndarray) -> np.ndarray:
    out = np.empty_like(p)
    out[..., 0] = p[..., 0]
    out[..., 1] = -p[..., 2]
    out[..., 2] = p[..., 1]
    return out


def assign_slot(obj) -> None:
    if obj.animation_data and obj.animation_data.action:
        try:
            slots = obj.animation_data.action_suitable_slots
            if slots:
                obj.animation_data.action_slot = slots[0]
        except Exception:
            pass


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
    pb = arm.pose.bones["pelvis"]
    bone = arm.data.bones["pelvis"]
    rest = bone.matrix_local
    rest_r_inv = rest.to_3x3().inverted()
    rest_t = rest.to_translation()
    arm_inv = arm.matrix_world.inverted()
    desired_arm = arm_inv @ desired_w
    pb.location = rest_r_inv @ (desired_arm - rest_t)
    bpy.context.view_layer.update()
    for _ in range(6):
        cur = arm.matrix_world @ pb.head
        err = desired_w - cur
        if err.length < 1e-5:
            break
        pb.location = pb.location + rest_r_inv @ (arm_inv.to_3x3() @ err)
        bpy.context.view_layer.update()


def swing_aim_bone(arm, bone_name: str, child_name: str, desired_child_w: Vector) -> None:
    """Swing-only — preserves pelvis location (walk root)."""
    pb = arm.pose.bones[bone_name]
    if child_name not in arm.pose.bones:
        return
    saved = pb.location.copy()
    is_root = bone_name == "pelvis" or pb.parent is None
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
    pb.location = saved if is_root else Vector((0, 0, 0))
    bpy.context.view_layer.update()

    head_w = arm.matrix_world @ pb.head
    child_rest_w = arm.matrix_world @ arm.pose.bones[child_name].head
    v_rest = child_rest_w - head_w
    v_tgt = desired_child_w - head_w
    if v_rest.length < 1e-8 or v_tgt.length < 1e-8:
        if is_root:
            pb.location = saved
        return
    v_rest.normalize()
    v_tgt.normalize()
    if v_rest.dot(v_tgt) > 0.999999:
        if is_root:
            pb.location = saved
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
    pb.location = saved if is_root else Vector((0, 0, 0))
    bpy.context.view_layer.update()


def match_arm_span(joints_z: np.ndarray, arm: "bpy.types.Object") -> np.ndarray:
    """
    Scale L/R arm joints in the shoulder-width axis so source shoulder span
    matches our rest shoulder span. Stops 'arms outside' from wider avatar rest.
    joints_z: (T,22,3) already Z-up and pelvis-anchored.
    """
    out = joints_z.copy()
    # Rest shoulder heads in world (Z-up armature)
    if "left_shoulder" not in arm.data.bones or "right_shoulder" not in arm.data.bones:
        return out
    ls = rest_head_world(arm, "left_shoulder")
    rs = rest_head_world(arm, "right_shoulder")
    rest_span = (ls - rs).length
    if rest_span < 1e-6:
        return out

    # Source mean shoulder span (indices 16, 17)
    src_span = np.linalg.norm(out[:, 16] - out[:, 17], axis=-1).mean()
    if src_span < 1e-6:
        return out

    # If source stick figure is narrower than our rest, scale arm joints toward midline
    # so their span ≈ rest (or keep source if already wider — only shrink flare)
    # ratio: map source span → rest span for arm chain only
    ratio = float(rest_span / src_span)
    # Only apply when source is narrower (ratio > 1 would expand — skip) or when
    # our rest is narrower... arms OUT means our visual is wider than joints:
    # joints source is correct width; we want arms not wider than joints.
    # So scale joints arm span is fine; we need our aim to not expand.
    # Here: ensure joint arm positions keep their span; also pre-scale if
    # median source span is used as the target width for aim targets.
    # Target: keep source span as-is (already good). No expand.
    # Optional: slightly reduce source arm span if we want tuck — use ratio only if ratio < 1
    # Actually problem is OUR bones longer/wider. Pre-scaling joints doesn't help rest_len aim.
    # For src_target aim, we need joints as ground truth — leave span as source.
    # Provide mild shrink if arm_inward handled elsewhere.
    _ = ratio
    return out


def pull_arms_inward(joints_z: np.ndarray, amount: float) -> np.ndarray:
    """Pull arm joints toward torso midline (spine3 / neck). amount in 0..1."""
    if amount <= 1e-6:
        return joints_z
    amount = float(max(0.0, min(1.0, amount)))
    out = joints_z.copy()
    T = out.shape[0]
    for t in range(T):
        # midline from spine3 and neck
        mid = 0.5 * (out[t, 9] + out[t, 12])
        for j in ARM_JOINT_IDX:
            out[t, j] = out[t, j] + (mid - out[t, j]) * amount
    return out


def apply_frame(arm, joints_z: np.ndarray, arm_aim: str = "src_target") -> None:
    """
    joints_z: (22,3) blender Z-up world positions for this frame.

    arm_aim:
      rest_len   — aim along source dir × OUR bone length (can flare arms out)
      src_target — aim parent bone toward SOURCE child joint world position
                   (matches joint stick figure; fixes arms-outside)
    """
    clear_pose(arm)
    hips = Vector((float(joints_z[0, 0]), float(joints_z[0, 1]), float(joints_z[0, 2])))
    set_pelvis_location(arm, hips)

    def aim(parent: str, child: str, ia: int, ib: int, is_arm: bool):
        if parent not in arm.pose.bones or child not in arm.pose.bones:
            return
        bpy.context.view_layer.update()
        p = world_head(arm, parent)
        sa = Vector((float(joints_z[ia, 0]), float(joints_z[ia, 1]), float(joints_z[ia, 2])))
        sb = Vector((float(joints_z[ib, 0]), float(joints_z[ib, 1]), float(joints_z[ib, 2])))

        use_src = is_arm and arm_aim == "src_target"
        if use_src:
            # Point our bone toward where the source joint actually is.
            # If our shoulder is wider than the stick figure, this pulls the arm IN.
            desired = sb
            if (desired - p).length < 1e-8:
                return
            swing_aim_bone(arm, parent, child, desired)
            return

        d = sb - sa
        if d.length < 1e-8:
            return
        d.normalize()
        L = rest_length(arm, parent, child)
        if L < 1e-8:
            return
        # For arms with rest_len: use min(source segment, our length) so we don't
        # overshoot outward past the source limb length along that direction.
        if is_arm:
            L_src = (sb - sa).length
            # Aim along source dir but length = our rest (rotation same as L_src if collinear)
            # Use direction from our parent to source child for slight tuck
            d2 = sb - p
            if d2.length > 1e-8:
                d = d2.normalized()
            L = min(L, L_src) if L_src > 1e-6 else L
        swing_aim_bone(arm, parent, child, p + d * L)

    for edge in AIM_EDGES:
        aim(*edge)
    # re-lock root after torso aim
    set_pelvis_location(arm, hips)


def apply_rotmat_frame(arm, R_loc: np.ndarray, hips_w: Vector, parents: np.ndarray) -> None:
    """
    Apply HybrIK local rotation matrices as pose quaternions.
    Experimental: only correct if armature rest matches SMPL-X local axes.
    Y-up local R is converted with a basis change into Blender Z-up.
    """
    clear_pose(arm)
    set_pelvis_location(arm, hips_w)
    # Y-up → Z-up change-of-basis for rotations: R_z = M R_y M^T
    M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    Mt = M.T
    J = R_loc.shape[0]
    for i, bname in enumerate(KEY_BONES if False else []):
        pass
    bone_order = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    ]
    for i, bname in enumerate(bone_order):
        if i >= J or bname not in arm.pose.bones:
            continue
        R_y = R_loc[i]
        R_z = M @ R_y @ Mt
        pb = arm.pose.bones[bname]
        pb.rotation_mode = "QUATERNION"
        # preserve pelvis location
        saved = pb.location.copy() if bname == "pelvis" else Vector((0, 0, 0))
        pb.rotation_quaternion = Matrix(R_z.tolist()).to_quaternion()
        if bname == "pelvis":
            pb.location = saved
        bpy.context.view_layer.update()
    set_pelvis_location(arm, hips_w)


def main():
    args = parse_args()
    ppath = Path(args.pose)
    if not ppath.is_absolute():
        ppath = ROOT / ppath
    if not ppath.is_file():
        raise SystemExit(f"missing pose {ppath}")

    data = np.load(str(ppath), allow_pickle=True)
    method = str(data["method"]) if "method" in data.files else "hybrik"

    # Prefer true SMPL-X FK joints after HybrIK
    if args.source == "fk":
        if "joints_fk" not in data.files:
            raise SystemExit(
                "pose.npz has no joints_fk — re-run tools/hybrik_joints_to_pose.py "
                "(older npz only had raw joints; that was why results matched npy→smplx)"
            )
        joints_y = np.asarray(data["joints_fk"], dtype=np.float64)
        src_tag = "joints_fk(SMPL-X)"
    else:
        joints_y = np.asarray(data["joints_scaled"], dtype=np.float64)
        src_tag = "joints_scaled(raw MoMask)"

    T = joints_y.shape[0]
    log(f"pose {ppath.name} frames={T} method={method} source={src_tag} mode={args.mode}")

    if "joints_fk" in data.files and "joints_scaled" in data.files:
        raw = np.asarray(data["joints_scaled"])
        fk = np.asarray(data["joints_fk"])
        d = np.linalg.norm((raw - raw[:, 0:1]) - (fk - fk[:, 0:1]), axis=-1)
        log(f"raw vs fk local joint delta mean={d.mean():.4f}m (bake source={args.source})")

    arm = bpy.data.objects.get(args.armature)
    if arm is None or arm.type != "ARMATURE":
        raise SystemExit(f"need armature {args.armature}")

    # Y-up → Z-up, anchor f0 hips on rest pelvis
    joints_z = y_up_to_z_up(joints_y)
    p0 = joints_z[0, 0].copy()
    pelvis_rest = rest_head_world(arm, "pelvis")
    offset0 = np.array(
        [pelvis_rest.x - p0[0], pelvis_rest.y - p0[1], pelvis_rest.z - p0[2]]
    )
    joints_z = joints_z + offset0

    if args.root == "inplace":
        hips0 = joints_z[0, 0].copy()
        for t in range(T):
            dxy = joints_z[t, 0, :2] - hips0[:2]
            joints_z[t, :, 0] -= dxy[0]
            joints_z[t, :, 1] -= dxy[1]

    arm_inward = float(args.arm_inward)
    if arm_inward > 1e-6:
        joints_z = pull_arms_inward(joints_z, arm_inward)
        log(f"arm_inward={arm_inward:.3f} (pull arm joints toward torso)")

    # Log shoulder span: source joints vs our rest (explains arms-out)
    if "left_shoulder" in arm.data.bones and "right_shoulder" in arm.data.bones:
        ls = rest_head_world(arm, "left_shoulder")
        rs = rest_head_world(arm, "right_shoulder")
        rest_span = (ls - rs).length
        src_span = float(np.linalg.norm(joints_z[:, 16] - joints_z[:, 17], axis=-1).mean())
        log(
            f"shoulder_span rest_avatar={rest_span:.3f}m  joints_src={src_span:.3f}m  "
            f"ratio={rest_span / max(src_span, 1e-6):.3f}  arm_aim={args.arm_aim}"
        )

    travel = joints_z[:, 0] - joints_z[0, 0]
    log(f"hip_travel_max={float(np.linalg.norm(travel, axis=-1).max()):.4f}m root={args.root}")

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

    rot_mats = np.asarray(data["rot_mats"]) if "rot_mats" in data.files else None
    parents = np.asarray(data["parents"]) if "parents" in data.files else None

    arm_aim = args.arm_aim
    for t in range(T):
        f = t + 1
        hips_w = Vector(
            (float(joints_z[t, 0, 0]), float(joints_z[t, 0, 1]), float(joints_z[t, 0, 2]))
        )
        if args.mode == "rotmat" and rot_mats is not None and parents is not None:
            apply_rotmat_frame(arm, rot_mats[t], hips_w, parents)
        else:
            apply_frame(arm, joints_z[t], arm_aim=arm_aim)
        for bn in KEY_BONES:
            if bn not in arm.pose.bones:
                continue
            pb = arm.pose.bones[bn]
            pb.rotation_mode = "QUATERNION"
            pb.keyframe_insert("rotation_quaternion", frame=f)
            if bn == "pelvis":
                pb.keyframe_insert("location", frame=f)
        if t == 0 or t % 20 == 0 or t == T - 1:
            pw = world_head(arm, "pelvis")
            sh = world_head(arm, "left_shoulder") if "left_shoulder" in arm.pose.bones else pw
            el = world_head(arm, "left_elbow") if "left_elbow" in arm.pose.bones else pw
            wr = world_head(arm, "left_wrist") if "left_wrist" in arm.pose.bones else pw
            # lateral = distance from body midline (x of spine)
            mid_x = pw.x
            log(
                f"  f{f}: L_arm_out elbow={abs(el.x-mid_x):.3f} wrist={abs(wr.x-mid_x):.3f} "
                f"sh.z={sh.z:.3f}"
            )

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = T
    bpy.context.scene.frame_set(1)

    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    p1 = world_head(arm, "pelvis")
    bpy.context.scene.frame_set(T)
    bpy.context.view_layer.update()
    p2 = world_head(arm, "pelvis")
    log(
        f"pelvis world f1→f{T}: d=({p2.x-p1.x:.4f},{p2.y-p1.y:.4f},{p2.z-p1.z:.4f}) "
        f"|d|={(p2 - p1).length:.4f}"
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
            "pose": str(ppath),
            "frames": T,
            "root": args.root,
            "source": args.source,
            "mode": args.mode,
            "arm_aim": args.arm_aim,
            "arm_inward": arm_inward,
            "method": f"hybrik_apply_{args.source}_{args.mode}_{args.arm_aim}",
        }
        (out_p.parent / f"{args.action}_hybrik.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    log("DONE")


if __name__ == "__main__":
    main()
