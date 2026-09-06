"""
Direct NPZ → SMPL-X_Armature retarget (no BVH intermediate).

Takes a HumanML3D-style joints NPZ (T,22,3) Y-up, metres and
bakes it onto the SMPL-X_Armature inside the source blend file.
Saves the result to --out (original file is NEVER modified).

HumanML3D joint indices:
  0=pelvis  1=left_hip  2=right_hip  3=spine1  4=left_knee
  5=right_knee  6=spine2  7=left_ankle  8=right_ankle  9=spine3
  10=left_foot  11=right_foot  12=neck  13=left_collar  14=right_collar
  15=head  16=left_shoulder  17=right_shoulder  18=left_elbow
  19=right_elbow  20=left_wrist  21=right_wrist

Usage (run from project root, DO NOT alter the source .blend):
  blender.exe --background "whole body.blend" ^
    --python tools/apply_npz_to_blend.py -- ^
    --npz "lmm train/moonwalk.npz" ^
    --action moonwalk ^
    --out "body_motion/_moonwalk_applied.blend"
"""
from __future__ import annotations
import argparse, sys, math
from pathlib import Path

import bpy, numpy as np
from mathutils import Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]

# HumanML3D joint index → bone name in SMPL-X_Armature
HML_TO_BONE: dict[int, str] = {
    0:  "pelvis",
    1:  "left_hip",
    2:  "right_hip",
    3:  "spine1",
    4:  "left_knee",
    5:  "right_knee",
    6:  "spine2",
    7:  "left_ankle",
    8:  "right_ankle",
    9:  "spine3",
    10: "left_foot",
    11: "right_foot",
    12: "neck",
    13: "left_collar",
    14: "right_collar",
    15: "head",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_wrist",
    21: "right_wrist",
}

# HumanML3D BVH-style name (for retarget logic) → joint index
BVH_NAME_TO_IDX: dict[str, int] = {
    "Hips": 0,
    "LeftUpLeg": 1,    "RightUpLeg": 2,
    "Spine": 3,
    "LeftLeg": 4,      "RightLeg": 5,
    "Spine1": 6,
    "LeftFoot": 7,     "RightFoot": 8,
    "Spine2": 9,
    "LeftToe": 10,     "RightToe": 11,
    "Neck": 12,
    "LeftShoulder": 13, "RightShoulder": 14,
    "Head": 15,
    "LeftArm": 16,     "RightArm": 17,
    "LeftForeArm": 18, "RightForeArm": 19,
    "LeftHand": 20,    "RightHand": 21,
}

# Swing-only aim order: (tgt_parent, tgt_child, src_a, src_b)
APPLY_ORDER = [
    ("pelvis", "spine1",       "Hips",        "Spine"),
    ("spine1", "spine2",       "Spine",       "Spine1"),
    ("spine2", "spine3",       "Spine1",      "Spine2"),
    ("spine3", "neck",         "Spine2",      "Neck"),
    ("neck",   "head",         "Neck",        "Head"),
    ("left_hip",   "left_knee",   "LeftUpLeg",   "LeftLeg"),
    ("left_knee",  "left_ankle",  "LeftLeg",     "LeftFoot"),
    ("left_ankle", "left_foot",   "LeftFoot",    "LeftToe"),
    ("right_hip",  "right_knee",  "RightUpLeg",  "RightLeg"),
    ("right_knee", "right_ankle", "RightLeg",    "RightFoot"),
    ("right_ankle","right_foot",  "RightFoot",   "RightToe"),
    ("left_shoulder",  "left_elbow",  "LeftArm",    "LeftForeArm"),
    ("left_elbow",     "left_wrist",  "LeftForeArm","LeftHand"),
    ("right_shoulder", "right_elbow", "RightArm",   "RightForeArm"),
    ("right_elbow",    "right_wrist", "RightForeArm","RightHand"),
]

KEY_BONES = list({v for _, v in HML_TO_BONE.items()})


def log(msg: str) -> None:
    print(f"[npz_retarget] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--npz",       required=True)
    p.add_argument("--armature",  default="SMPL-X_Armature")
    p.add_argument("--action",    default="moonwalk")
    p.add_argument("--out",       required=True)
    p.add_argument("--root",      default="absolute",
                   choices=("relative", "absolute", "inplace"))
    p.add_argument("--lock_collars", action="store_true", default=True)
    p.add_argument("--no_lock_collars", action="store_true")
    return p.parse_args(argv)


def hml_to_blender(pt: np.ndarray) -> Vector:
    """Y-up HumanML3D → Blender Z-up: (x, y, z)_hml → (x, -z, y)_blender"""
    return Vector((float(pt[0]), float(-pt[2]), float(pt[1])))


def world_head(arm, bone_name: str) -> Vector:
    dg = bpy.context.evaluated_depsgraph_get()
    ae = arm.evaluated_get(dg)
    return arm.matrix_world @ ae.pose.bones[bone_name].head


def rest_head_world(arm, bone_name: str) -> Vector:
    return arm.matrix_world @ arm.data.bones[bone_name].head_local


def rest_length(arm, parent: str, child: str) -> float:
    return (arm.data.bones[child].head_local - arm.data.bones[parent].head_local).length


def clear_pose(arm) -> None:
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
        pb.location = Vector((0, 0, 0))
        pb.scale = Vector((1, 1, 1))
    bpy.context.view_layer.update()


def set_pelvis_location(arm, desired_world: Vector) -> None:
    """Move root pelvis to desired_world via iterative refinement using evaluated DG.
    Converges in ~3 steps regardless of bone local-frame orientation."""
    pb = arm.pose.bones["pelvis"]
    pb.location = Vector((0, 0, 0))
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
    bpy.context.view_layer.update()
    arm_inv_rot = arm.matrix_world.inverted().to_3x3()
    for _ in range(10):
        cur_world = world_head(arm, "pelvis")   # evaluated DG — correct position
        err_world = desired_world - cur_world
        if err_world.length < 1e-5:
            break
        pb.location = pb.location + arm_inv_rot @ err_world
    bpy.context.view_layer.update()


def swing_aim_bone(arm, bone_name: str, child_name: str,
                   desired_child_w: Vector) -> None:
    """Swing-only: reset bone to rest then apply minimal rotation to aim child."""
    pb = arm.pose.bones[bone_name]
    if child_name not in arm.pose.bones:
        return
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
    pb.location = Vector((0, 0, 0))
    bpy.context.view_layer.update()

    head_w = arm.matrix_world @ pb.head
    child_rest_w = arm.matrix_world @ arm.pose.bones[child_name].head
    v_rest = (child_rest_w - head_w).normalized()
    v_tgt  = (desired_child_w - head_w)
    if v_tgt.length < 1e-8:
        return
    v_tgt = v_tgt.normalized()
    if v_rest.dot(v_tgt) > 0.999999:
        return

    q_swing = v_rest.rotation_difference(v_tgt)
    from mathutils import Matrix
    M = arm.matrix_world @ pb.matrix
    head = M.to_translation()
    M_new = (Matrix.Translation(head) @
             q_swing.to_matrix().to_4x4() @
             Matrix.Translation(-head) @ M)

    mat_arm = arm.matrix_world.inverted() @ M_new
    bone = arm.data.bones[bone_name]
    if pb.parent:
        parent_mat = pb.parent.matrix
        pre = parent_mat @ pb.parent.bone.matrix_local.inverted() @ bone.matrix_local
        basis = pre.inverted() @ mat_arm
    else:
        basis = bone.matrix_local.inverted() @ mat_arm

    _, rot, _ = basis.decompose()
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = rot
    if bone_name != "pelvis":
        pb.location = Vector((0, 0, 0))
    bpy.context.view_layer.update()


def apply_frame(arm, src: dict[str, Vector], lock_collars: bool) -> None:
    clear_pose(arm)
    set_pelvis_location(arm, src["Hips"])

    def aim(parent: str, child: str, sa: str, sb: str):
        if parent not in arm.pose.bones or child not in arm.pose.bones:
            return
        if sa not in src or sb not in src:
            return
        bpy.context.view_layer.update()
        p = world_head(arm, parent)
        d = (src[sb] - src[sa])
        if d.length < 1e-8:
            d = rest_head_world(arm, child) - rest_head_world(arm, parent)
        d = d.normalized()
        L = rest_length(arm, parent, child)
        target = p + d * L
        swing_aim_bone(arm, parent, child, target)

    # Torso
    aim("pelvis", "spine1", "Hips", "Spine")
    aim("spine1", "spine2", "Spine", "Spine1")
    aim("spine2", "spine3", "Spine1", "Spine2")
    aim("spine3", "neck",  "Spine2", "Neck")
    aim("neck",   "head",  "Neck",   "Head")

    # Legs
    aim("left_hip",    "left_knee",   "LeftUpLeg", "LeftLeg")
    aim("left_knee",   "left_ankle",  "LeftLeg",   "LeftFoot")
    aim("left_ankle",  "left_foot",   "LeftFoot",  "LeftToe")
    aim("right_hip",   "right_knee",  "RightUpLeg","RightLeg")
    aim("right_knee",  "right_ankle", "RightLeg",  "RightFoot")
    aim("right_ankle", "right_foot",  "RightFoot", "RightToe")

    # Arms
    if lock_collars:
        aim("left_shoulder",  "left_elbow",  "LeftArm",    "LeftForeArm")
        aim("left_elbow",     "left_wrist",  "LeftForeArm","LeftHand")
        aim("right_shoulder", "right_elbow", "RightArm",   "RightForeArm")
        aim("right_elbow",    "right_wrist", "RightForeArm","RightHand")
    else:
        aim("left_collar",    "left_shoulder",  "LeftShoulder","LeftArm")
        aim("left_shoulder",  "left_elbow",     "LeftArm",    "LeftForeArm")
        aim("left_elbow",     "left_wrist",     "LeftForeArm","LeftHand")
        aim("right_collar",   "right_shoulder", "RightShoulder","RightArm")
        aim("right_shoulder", "right_elbow",    "RightArm",   "RightForeArm")
        aim("right_elbow",    "right_wrist",    "RightForeArm","RightHand")


def keyframe_pose(arm, frame: int) -> None:
    for name in KEY_BONES:
        pb = arm.pose.bones.get(name)
        if not pb:
            continue
        pb.rotation_mode = "QUATERNION"
        pb.keyframe_insert("rotation_quaternion", frame=frame)
        if name == "pelvis":
            pb.keyframe_insert("location", frame=frame)


def assign_slot(obj) -> None:
    if not obj.animation_data or not obj.animation_data.action:
        return
    try:
        slots = obj.animation_data.action_suitable_slots
        if slots:
            obj.animation_data.action_slot = slots[0]
    except Exception:
        pass


def main():
    args = parse_args()
    lock_collars = not args.no_lock_collars

    npz_path = Path(args.npz)
    if not npz_path.is_absolute():
        npz_path = (ROOT / npz_path).resolve()
    if not npz_path.exists():
        raise SystemExit(f"NPZ not found: {npz_path}")

    d = np.load(str(npz_path), allow_pickle=True)
    joints = d["joints"]     # (T, 22, 3)  Y-up, metres
    fps    = int(d.get("fps", 30))
    T      = joints.shape[0]
    prompt = str(d.get("meta_prompt", "unknown"))
    log(f"loaded NPZ: T={T} fps={fps} prompt='{prompt}'")

    arm = bpy.data.objects.get(args.armature)
    if not arm or arm.type != "ARMATURE":
        raise SystemExit(f"Armature '{args.armature}' not found in scene")
    log(f"target armature: {arm.name}")

    bpy.context.scene.render.fps = fps

    # Scale factor: match our rest hip height to NPZ hip height range
    hml_hip_y = float(joints[:, 0, 1].mean())   # mean Y in HML space
    our_pelvis_rest_z = float(
        (arm.matrix_world @ arm.data.bones["pelvis"].head_local).z)
    log(f"HML mean pelvis Y = {hml_hip_y:.3f}m  SMPLX rest pelvis Z = {our_pelvis_rest_z:.3f}m")

    # ── Prepare animation data: detach existing action, mute all NLA tracks ──
    if not arm.animation_data:
        arm.animation_data_create()
    # Detach current action so existing clips don't override manual poses
    arm.animation_data.action = None
    # Mute every NLA track (pre-existing walk/wave/idle clips)
    n_muted = 0
    for track in arm.animation_data.nla_tracks:
        track.mute = True
        n_muted += 1
    arm.animation_data.use_nla = False
    log(f"NLA tracks muted: {n_muted}, use_nla=False")
    bpy.context.view_layer.update()

    # ── Create new empty action for baking ────────────────────────────────
    old = bpy.data.actions.get(args.action)
    if old:
        bpy.data.actions.remove(old)
    act = bpy.data.actions.new(args.action)
    act.use_fake_user = True
    arm.animation_data.action = act
    assign_slot(arm)

    # ── Diagnostic: verify pb.location is accepted (not overridden) ───────
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    _pb_test = arm.pose.bones["pelvis"]
    _pb_test.location = Vector((0, 0, 5.0))
    bpy.context.view_layer.update()
    _read_loc = world_head(arm, "pelvis")
    log(f"DIAG: set pelvis loc=(0,0,5) → world_head.z={_read_loc.z:.3f}  (expect ~5 if working, ~0.99 if overridden)")
    _pb_test.location = Vector((0, 0, 0))
    bpy.context.view_layer.update()

    # ── Verify & log target bone lengths (from OUR body, NOT from MoMask) ──
    # The script uses rest_length(arm, ...) = SMPLX armature rest lengths.
    # Direction comes from MoMask; LENGTH always stays our body's own length.
    for label, pa, ch in [
        ("thigh (L)",  "left_hip",       "left_knee"),
        ("shin  (L)",  "left_knee",      "left_ankle"),
        ("upper arm",  "left_shoulder",  "left_elbow"),
        ("forearm",    "left_elbow",     "left_wrist"),
        ("spine",      "pelvis",         "spine1"),
    ]:
        try:
            L = rest_length(arm, pa, ch)
            log(f"bone_length {label:12s} = {L:.4f}m  [from OUR body — preserved every frame]")
        except Exception:
            pass

    pelvis0_loc = None
    f_start = 1

    # Debug: print NPZ and converted position for first frame
    j0 = joints[0, 0]
    log(f"NPZ frame0 pelvis HML: x={j0[0]:.4f} y={j0[1]:.4f} z={j0[2]:.4f}")
    j0b = hml_to_blender(j0)
    log(f"NPZ frame0 pelvis Blender: x={j0b.x:.4f} y={j0b.y:.4f} z={j0b.z:.4f}")
    j60 = joints[min(60, T-1), 0]
    j60b = hml_to_blender(j60)
    log(f"NPZ frame60 pelvis Blender: x={j60b.x:.4f} y={j60b.y:.4f} z={j60b.z:.4f}  (Y should differ from frame0)")
    rest_pelv = arm.matrix_world @ arm.data.bones["pelvis"].head_local
    log(f"SMPLX rest pelvis world: x={rest_pelv.x:.4f} y={rest_pelv.y:.4f} z={rest_pelv.z:.4f}")
    log(f"arm.matrix_world scale: {arm.matrix_world.to_scale()}")

    for t in range(T):
        frame = f_start + t
        bpy.context.scene.frame_set(frame)

        # Convert HML (Y-up) → Blender (Z-up) for this frame
        src: dict[str, Vector] = {}
        for bvh_name, idx in BVH_NAME_TO_IDX.items():
            src[bvh_name] = hml_to_blender(joints[t, idx])

        apply_frame(arm, src, lock_collars)

        # Relative root (clip starts near rest position)
        if args.root == "relative":
            pb = arm.pose.bones["pelvis"]
            if pelvis0_loc is None:
                pelvis0_loc = pb.location.copy()
            pb.location = pb.location - pelvis0_loc

        keyframe_pose(arm, frame)

        if t == 0 or t % 30 == 0 or t == T - 1:
            hip = world_head(arm, "left_hip")
            knee = world_head(arm, "left_knee")
            ank = world_head(arm, "left_ankle")
            pelv = world_head(arm, "pelvis")
            ok = knee.z < hip.z and ank.z < knee.z
            log(f"f{frame:3d}: legs_down={ok}  "
                f"pelvis=({pelv.x:.3f},{pelv.y:.3f},{pelv.z:.3f})  "
                f"hip.z={hip.z:.3f} knee.z={knee.z:.3f} ank.z={ank.z:.3f}")

    bpy.context.scene.frame_start = f_start
    bpy.context.scene.frame_end   = f_start + T - 1
    bpy.context.scene.frame_set(f_start)
    log(f"action '{act.name}' baked: {f_start}-{f_start+T-1}")

    out = Path(args.out)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
        log(f"saved → {out}")
    except RuntimeError as e:
        alt = out.with_name(out.stem + "_v2" + out.suffix)
        log(f"save failed ({e}); retrying → {alt}")
        bpy.ops.wm.save_as_mainfile(filepath=str(alt))
        log(f"saved → {alt}")
    log("DONE")


if __name__ == "__main__":
    main()
