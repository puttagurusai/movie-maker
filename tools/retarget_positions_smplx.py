"""
Position-based retarget: MoMask/BVH → SMPL-X with TARGET bone lengths, SWING-ONLY (no twist).

Why twists remained before:
  1. SMPL-X rest bone +Y often does NOT point along the limb (dot can be ~0 or -1).
  2. Setting pose via full matrix / unconstrained rotation keeps a free ROLL around the limb
     → candy-wrapper twist on stomach, collars, upper arms.
  3. Fix: each bone starts at local REST, then apply only the minimal SWING that aims
     rest-limb → desired-limb. No extra roll.

Usage:
  blender.exe scene.blend --python tools/retarget_positions_smplx.py -- ^
    --source momask_walk --action momask_walk_fixed_lengths ^
    --out body_motion/_momask_fixed_lengths.blend
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]

SRC_TO_TGT = {
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

# Parent→child edges and the source pair that supplies the direction
EDGES: list[tuple[str, str, str, str]] = [
    # (tgt_parent, tgt_child, src_a, src_b)
    ("pelvis", "spine1", "Hips", "Spine"),
    ("spine1", "spine2", "Spine", "Spine1"),
    ("spine2", "spine3", "Spine1", "Spine2"),
    ("spine3", "neck", "Spine2", "Neck"),
    ("neck", "head", "Neck", "Head"),
    # legs (do NOT re-aim pelvis at hips — pelvis aims spine only)
    ("left_hip", "left_knee", "LeftUpLeg", "LeftLeg"),
    ("left_knee", "left_ankle", "LeftLeg", "LeftFoot"),
    ("left_ankle", "left_foot", "LeftFoot", "LeftToe"),
    ("right_hip", "right_knee", "RightUpLeg", "RightLeg"),
    ("right_knee", "right_ankle", "RightLeg", "RightFoot"),
    ("right_ankle", "right_foot", "RightFoot", "RightToe"),
    # hip bones themselves: aim from pelvis rest offset using source hip dir
    # handled as special: left_hip aims left_knee after we set left_hip from pelvis→hip edge
    ("pelvis", "left_hip", "Hips", "LeftUpLeg"),
    ("pelvis", "right_hip", "Hips", "RightUpLeg"),
    # arms — collars optional (see --lock_collars)
    ("spine3", "left_collar", "Spine2", "LeftShoulder"),
    ("left_collar", "left_shoulder", "LeftShoulder", "LeftArm"),
    ("left_shoulder", "left_elbow", "LeftArm", "LeftForeArm"),
    ("left_elbow", "left_wrist", "LeftForeArm", "LeftHand"),
    ("spine3", "right_collar", "Spine2", "RightShoulder"),
    ("right_collar", "right_shoulder", "RightShoulder", "RightArm"),
    ("right_shoulder", "right_elbow", "RightArm", "RightForeArm"),
    ("right_elbow", "right_wrist", "RightForeArm", "RightHand"),
]

# Apply order: each bone once, root to leaf. Multi-child: spine before legs on pelvis.
APPLY_ORDER: list[tuple[str, str, str, str]] = [
    # pelvis aims torso (spine) first
    ("pelvis", "spine1", "Hips", "Spine"),
    ("spine1", "spine2", "Spine", "Spine1"),
    ("spine2", "spine3", "Spine1", "Spine2"),
    ("spine3", "neck", "Spine2", "Neck"),
    ("neck", "head", "Neck", "Head"),
    # legs: aim hip from pelvis without changing pelvis (use hip bone only)
    ("left_hip", "left_knee", "LeftUpLeg", "LeftLeg"),
    ("left_knee", "left_ankle", "LeftLeg", "LeftFoot"),
    ("left_ankle", "left_foot", "LeftFoot", "LeftToe"),
    ("right_hip", "right_knee", "RightUpLeg", "RightLeg"),
    ("right_knee", "right_ankle", "RightLeg", "RightFoot"),
    ("right_ankle", "right_foot", "RightFoot", "RightToe"),
    # arms
    ("left_collar", "left_shoulder", "LeftShoulder", "LeftArm"),
    ("left_shoulder", "left_elbow", "LeftArm", "LeftForeArm"),
    ("left_elbow", "left_wrist", "LeftForeArm", "LeftHand"),
    ("right_collar", "right_shoulder", "RightShoulder", "RightArm"),
    ("right_shoulder", "right_elbow", "RightArm", "RightForeArm"),
    ("right_elbow", "right_wrist", "RightForeArm", "RightHand"),
]

KEY_BONES = [
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
]


def log(msg: str) -> None:
    print(f"[pos_retarget] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="momask_walk")
    p.add_argument("--armature", default="SMPL-X_Armature")
    p.add_argument("--action", default="momask_walk_fixed_lengths")
    p.add_argument("--out", default="")
    p.add_argument("--root", default="relative", choices=("relative", "absolute", "inplace"))
    p.add_argument(
        "--lock_collars",
        action="store_true",
        default=True,
        help="Keep collars at REST (recommended — kills chest twist)",
    )
    p.add_argument("--no_lock_collars", action="store_true", help="Allow collar swing")
    p.add_argument(
        "--lock_spine",
        action="store_true",
        default=False,
        help="Also lock spine1/2/3 at rest (stiff torso, zero stomach twist)",
    )
    return p.parse_args(argv)


def assign_slot(obj) -> None:
    if not obj.animation_data or not obj.animation_data.action:
        return
    try:
        slots = obj.animation_data.action_suitable_slots
        if slots:
            obj.animation_data.action_slot = slots[0]
    except Exception:
        pass


def world_head(arm, name: str) -> Vector:
    return (arm.matrix_world @ arm.pose.bones[name].head).copy()


def rest_head_world(arm, name: str) -> Vector:
    return (arm.matrix_world @ arm.data.bones[name].head_local).copy()


def rest_length(arm, parent: str, child: str) -> float:
    return (arm.data.bones[child].head_local - arm.data.bones[parent].head_local).length


def clear_pose(arm) -> None:
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
        pb.location = Vector((0, 0, 0))
        pb.scale = Vector((1, 1, 1))
    bpy.context.view_layer.update()


def build_desired(
    src_pos: dict[str, Vector],
    tgt,
    root_offset: Vector,
    lock_collars: bool,
    lock_spine: bool,
) -> dict[str, Vector]:
    """SOURCE directions × TARGET lengths."""
    desired: dict[str, Vector] = {}
    desired["pelvis"] = src_pos["Hips"] + root_offset

    def add_edge(tp: str, tc: str, sa: str, sb: str):
        if tp not in desired:
            return
        if sa not in src_pos or sb not in src_pos:
            return
        d = src_pos[sb] - src_pos[sa]
        if d.length < 1e-8:
            d = rest_head_world(tgt, tc) - rest_head_world(tgt, tp)
        d = d.normalized()
        L = rest_length(tgt, tp, tc)
        if L < 1e-8:
            return
        desired[tc] = desired[tp] + d * L

    # Build full tree for positions (even if we lock some bones later)
    for tp, tc, sa, sb in [
        ("pelvis", "spine1", "Hips", "Spine"),
        ("spine1", "spine2", "Spine", "Spine1"),
        ("spine2", "spine3", "Spine1", "Spine2"),
        ("spine3", "neck", "Spine2", "Neck"),
        ("neck", "head", "Neck", "Head"),
        ("pelvis", "left_hip", "Hips", "LeftUpLeg"),
        ("left_hip", "left_knee", "LeftUpLeg", "LeftLeg"),
        ("left_knee", "left_ankle", "LeftLeg", "LeftFoot"),
        ("left_ankle", "left_foot", "LeftFoot", "LeftToe"),
        ("pelvis", "right_hip", "Hips", "RightUpLeg"),
        ("right_hip", "right_knee", "RightUpLeg", "RightLeg"),
        ("right_knee", "right_ankle", "RightLeg", "RightFoot"),
        ("right_ankle", "right_foot", "RightFoot", "RightToe"),
        ("spine3", "left_collar", "Spine2", "LeftShoulder"),
        ("left_collar", "left_shoulder", "LeftShoulder", "LeftArm"),
        ("left_shoulder", "left_elbow", "LeftArm", "LeftForeArm"),
        ("left_elbow", "left_wrist", "LeftForeArm", "LeftHand"),
        ("spine3", "right_collar", "Spine2", "RightShoulder"),
        ("right_collar", "right_shoulder", "RightShoulder", "RightArm"),
        ("right_shoulder", "right_elbow", "RightArm", "RightForeArm"),
        ("right_elbow", "right_wrist", "RightForeArm", "RightHand"),
    ]:
        if lock_spine and tc in ("spine1", "spine2", "spine3"):
            # place at rest direction from parent (no source bend)
            if tp not in desired:
                continue
            d = rest_head_world(tgt, tc) - rest_head_world(tgt, tp)
            if d.length < 1e-8:
                continue
            desired[tc] = desired[tp] + d.normalized() * rest_length(tgt, tp, tc)
            continue
        if lock_collars and tc in ("left_collar", "right_collar"):
            if tp not in desired:
                continue
            d = rest_head_world(tgt, tc) - rest_head_world(tgt, tp)
            if d.length < 1e-8:
                continue
            desired[tc] = desired[tp] + d.normalized() * rest_length(tgt, tp, tc)
            continue
        if lock_collars and tc in ("left_shoulder", "right_shoulder"):
            # Direction from source upper-arm, length = our collar-end to shoulder? 
            # Shoulder joint: from locked collar using source arm direction
            if tp not in desired:
                continue
            # tp is left_collar; use LeftShoulder→LeftArm direction for aim of collar→shoulder
            sa, sb = ("LeftShoulder", "LeftArm") if "left" in tc else ("RightShoulder", "RightArm")
            if sa in src_pos and sb in src_pos:
                d = (src_pos[sb] - src_pos[sa]).normalized()
            else:
                d = (rest_head_world(tgt, tc) - rest_head_world(tgt, tp)).normalized()
            desired[tc] = desired[tp] + d * rest_length(tgt, tp, tc)
            continue
        add_edge(tp, tc, sa, sb)

    return desired


def set_pelvis_location(arm, desired_pelvis_w: Vector) -> None:
    pb = arm.pose.bones["pelvis"]
    bpy.context.view_layer.update()
    cur = arm.matrix_world @ pb.head
    delta_w = desired_pelvis_w - cur
    delta_a = arm.matrix_world.inverted().to_3x3() @ delta_w
    rest = arm.data.bones["pelvis"].matrix_local
    local_delta = rest.to_3x3().inverted() @ delta_a
    pb.location = pb.location + local_delta
    bpy.context.view_layer.update()


def swing_aim_bone(arm, bone_name: str, child_name: str, desired_child_w: Vector) -> None:
    """
    SWING-ONLY aim:
      - force this bone to local REST (identity basis) first
      - measure rest limb under current parent pose
      - apply minimal rotation that maps rest_limb → desired_limb
      - write clean local quaternion (no residual matrix roll)
    """
    pb = arm.pose.bones[bone_name]
    if child_name not in arm.pose.bones:
        return

    # Reset ONLY this bone to rest relative to parent
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
    pb.location = Vector((0, 0, 0))
    bpy.context.view_layer.update()

    head_w = arm.matrix_world @ pb.head
    child_rest_w = arm.matrix_world @ arm.pose.bones[child_name].head
    v_rest = child_rest_w - head_w
    v_tgt = desired_child_w - head_w
    if v_rest.length < 1e-8 or v_tgt.length < 1e-8:
        return
    v_rest.normalize()
    v_tgt.normalize()

    # already aligned
    if v_rest.dot(v_tgt) > 0.999999:
        return

    q_swing = v_rest.rotation_difference(v_tgt)  # pure swing in world

    # Current pose matrix (armature space) at rest-local
    M = arm.matrix_world @ pb.matrix
    head = M.to_translation()
    M_new = Matrix.Translation(head) @ q_swing.to_matrix().to_4x4() @ Matrix.Translation(-head) @ M

    # Convert desired armature-space pose matrix → local matrix_basis → quaternion only
    mat_arm = arm.matrix_world.inverted() @ M_new
    # pb.matrix setter can introduce shear; instead compute matrix_basis
    bone = arm.data.bones[bone_name]
    if pb.parent:
        # pose_matrix = parent.matrix @ inv(parent.bone.matrix_local) @ bone.matrix_local @ basis
        parent_mat = pb.parent.matrix
        pre = parent_mat @ pb.parent.bone.matrix_local.inverted() @ bone.matrix_local
        basis = pre.inverted() @ mat_arm
    else:
        basis = bone.matrix_local.inverted() @ mat_arm

    loc, rot, sca = basis.decompose()
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = rot
    # keep location 0 for non-root (connected chain)
    if bone_name != "pelvis":
        pb.location = Vector((0, 0, 0))
    bpy.context.view_layer.update()


def aim_hip_from_pelvis(arm, hip_name: str, knee_name: str, desired_hip_w: Vector, desired_knee_w: Vector) -> None:
    """
    Place hip bone so its HEAD moves toward desired_hip and limb aims at knee.
    Hip is a child of pelvis; with location locked, only rotation of pelvis would move hip head.
    We only rotate hip bone (not pelvis) so hip head stays at rest offset from pelvis —
    desired hip position is already built from pelvis+dir*len matching rest length, so
    rest offset direction may differ from desired direction — need to rotate PELVIS? No.

    Actually: with fixed bone lengths from pelvis, desired left_hip is at
    pelvis + dir * L. Rest left_hip is at pelvis_rest_offset. After pelvis is posed,
    rest left_hip head is already somewhere. Swing left_hip to aim knee; hip HEAD
    position is fixed by pelvis pose + rest bind. That rest offset direction may not
    equal desired hip direction from source.

    For accurate hip placement we'd rotate pelvis (conflicts with spine) or use hip location.
    Practical walk fix: only swing-aim hip→knee, hip→ankle chain; accept hip root at rest offset.
    """
    swing_aim_bone(arm, hip_name, knee_name, desired_knee_w)


def _src_dir(src_pos: dict[str, Vector], sa: str, sb: str, arm, parent: str, child: str) -> Vector:
    if sa in src_pos and sb in src_pos:
        d = src_pos[sb] - src_pos[sa]
        if d.length > 1e-8:
            return d.normalized()
    d = rest_head_world(arm, child) - rest_head_world(arm, parent)
    if d.length < 1e-8:
        return Vector((0, 0, 1))
    return d.normalized()


def apply_desired(
    arm,
    src_pos: dict[str, Vector],
    root_offset: Vector,
    lock_collars: bool,
    lock_spine: bool,
) -> None:
    """
    Incremental swing-only IK:
      for each edge, desired_child = actual_parent_head + source_dir * OUR_length
      then swing parent bone only (reset to rest, pure swing).
    This keeps OUR bone lengths and avoids roll accumulation.
    """
    clear_pose(arm)
    # place pelvis
    desired_pelvis = src_pos["Hips"] + root_offset
    set_pelvis_location(arm, desired_pelvis)

    def aim(parent: str, child: str, sa: str, sb: str):
        if parent not in arm.pose.bones or child not in arm.pose.bones:
            return
        bpy.context.view_layer.update()
        p = world_head(arm, parent)
        d = _src_dir(src_pos, sa, sb, arm, parent, child)
        L = rest_length(arm, parent, child)
        target = p + d * L
        swing_aim_bone(arm, parent, child, target)

    # 1) Torso
    if not lock_spine:
        aim("pelvis", "spine1", "Hips", "Spine")
        aim("spine1", "spine2", "Spine", "Spine1")
        aim("spine2", "spine3", "Spine1", "Spine2")
        aim("spine3", "neck", "Spine2", "Neck")
        aim("neck", "head", "Neck", "Head")
    else:
        # upright rest spine; only neck/head from source
        aim("neck", "head", "Neck", "Head")

    # 2) Legs (never re-aim pelvis)
    aim("left_hip", "left_knee", "LeftUpLeg", "LeftLeg")
    aim("left_knee", "left_ankle", "LeftLeg", "LeftFoot")
    aim("left_ankle", "left_foot", "LeftFoot", "LeftToe")
    aim("right_hip", "right_knee", "RightUpLeg", "RightLeg")
    aim("right_knee", "right_ankle", "RightLeg", "RightFoot")
    aim("right_ankle", "right_foot", "RightFoot", "RightToe")

    # 3) Arms
    if lock_collars:
        # collars = rest (identity). Aim shoulder→elbow→wrist only.
        aim("left_shoulder", "left_elbow", "LeftArm", "LeftForeArm")
        aim("left_elbow", "left_wrist", "LeftForeArm", "LeftHand")
        aim("right_shoulder", "right_elbow", "RightArm", "RightForeArm")
        aim("right_elbow", "right_wrist", "RightForeArm", "RightHand")
    else:
        aim("left_collar", "left_shoulder", "LeftShoulder", "LeftArm")
        aim("left_shoulder", "left_elbow", "LeftArm", "LeftForeArm")
        aim("left_elbow", "left_wrist", "LeftForeArm", "LeftHand")
        aim("right_collar", "right_shoulder", "RightShoulder", "RightArm")
        aim("right_shoulder", "right_elbow", "RightArm", "RightForeArm")
        aim("right_elbow", "right_wrist", "RightForeArm", "RightHand")


def keyframe_pose(arm, frame: int, bones: list[str]) -> None:
    for name in bones:
        pb = arm.pose.bones.get(name)
        if not pb:
            continue
        pb.rotation_mode = "QUATERNION"
        pb.keyframe_insert("rotation_quaternion", frame=frame)
        if name == "pelvis":
            pb.keyframe_insert("location", frame=frame)


def main():
    args = parse_args()
    lock_collars = not args.no_lock_collars
    lock_spine = args.lock_spine

    src = bpy.data.objects.get(args.source)
    tgt = bpy.data.objects.get(args.armature)
    if not src or src.type != "ARMATURE":
        raise SystemExit(f"source armature not found: {args.source}")
    if not tgt or tgt.type != "ARMATURE":
        raise SystemExit(f"target not found: {args.armature}")
    if not src.animation_data or not src.animation_data.action:
        raise SystemExit("source has no action")

    assign_slot(src)
    act_src = src.animation_data.action
    f0, f1 = int(act_src.frame_range[0]), int(act_src.frame_range[1])
    log(f"source={src.name} frames={f0}-{f1} lock_collars={lock_collars} lock_spine={lock_spine}")

    scene = bpy.context.scene
    scene.frame_set(f0)
    bpy.context.view_layer.update()
    src_hips0 = world_head(src, "Hips")
    tgt_pelvis_rest = rest_head_world(tgt, "pelvis")
    root_offset = tgt_pelvis_rest - src_hips0
    log(f"root_offset={tuple(round(c, 4) for c in root_offset)}")

    if not tgt.animation_data:
        tgt.animation_data_create()
    old = bpy.data.actions.get(args.action)
    if old:
        if tgt.animation_data.action == old:
            tgt.animation_data.action = None
        bpy.data.actions.remove(old)
    act = bpy.data.actions.new(args.action)
    act.use_fake_user = True
    tgt.animation_data.action = act
    assign_slot(tgt)

    pelvis0_loc = None

    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()

        src_pos = {sb: world_head(src, sb) for sb in SRC_TO_TGT if sb in src.pose.bones}

        if args.root == "inplace":
            h = src_pos["Hips"]
            dxy = Vector((src_hips0.x - h.x, src_hips0.y - h.y, 0.0))
            src_pos = {k: v + dxy for k, v in src_pos.items()}

        apply_desired(tgt, src_pos, root_offset, lock_collars, lock_spine)

        if args.root == "relative":
            pb = tgt.pose.bones["pelvis"]
            if pelvis0_loc is None:
                pelvis0_loc = pb.location.copy()
            pb.location = pb.location - pelvis0_loc

        keyframe_pose(tgt, f, KEY_BONES)

        if f == f0 or f % 30 == 0 or f == f1:
            hip = world_head(tgt, "left_hip")
            knee = world_head(tgt, "left_knee")
            ank = world_head(tgt, "left_ankle")
            col_q = list(tgt.pose.bones["left_collar"].rotation_quaternion)
            sp2_q = list(tgt.pose.bones["spine2"].rotation_quaternion)
            log(
                f"f{f}: legs_down={knee.z < hip.z and ank.z < knee.z} "
                f"collar_q=({col_q[0]:.3f},{col_q[1]:.3f},{col_q[2]:.3f},{col_q[3]:.3f}) "
                f"spine2_q=({sp2_q[0]:.3f},{sp2_q[1]:.3f},{sp2_q[2]:.3f},{sp2_q[3]:.3f})"
            )

    scene.frame_set(f0)
    bpy.context.view_layer.update()
    for label, a, b in (
        ("thigh", "left_hip", "left_knee"),
        ("collar", "left_collar", "left_shoulder"),
        ("spine", "spine1", "spine2"),
    ):
        Lw = (world_head(tgt, b) - world_head(tgt, a)).length
        Lr = rest_length(tgt, a, b)
        log(f"length {label}: pose={Lw:.4f} rest={Lr:.4f}")

    scene.frame_start = f0
    scene.frame_end = f1
    scene.frame_set(f0)
    log(f"action ready: {act.name}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
        log(f"saved {out}")
    log("DONE")


if __name__ == "__main__":
    main()
