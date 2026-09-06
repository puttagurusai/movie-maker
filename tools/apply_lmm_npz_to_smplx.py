"""
Retarget MoMask/HumanML3D → SMPL-X_Armature.

Path:
  joints NPZ or BVH (MoMask official IK)
    → import BVH (Y-up)
    → optional orientation fix
    → Rokoko-style helper bones (match SMPL-X rest axes)
    → COPY_ROTATION from helpers + visual bake

Why helpers (not raw WORLD copy of BVH bones):
  MoMask/BVH bones point along the limb (bone +Y ≈ limb dir).
  SMPL-X_Armature display axes often point +Z while the limb goes −Z.
  WORLD COPY_ROTATION of BVH bones therefore folds legs upside-down.
  Helpers parented to BVH bones with SMPL-X rest head/tail/roll fix that.

Bone map: body_motion/HML22_TO_SMPLX_MAP.txt

Usage:
  blender.exe whole_body_retargeted.blend --python tools/apply_lmm_npz_to_smplx.py -- ^
    --bvh \"lmm train/momask_walk.bvh\" --action momask_walk ^
    --out body_motion/_momask_applied.blend
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
HELPER_SUFFIX = "_HML_H"

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
    print(f"[smplx_retarget] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", default="")
    p.add_argument("--npz", default="")
    p.add_argument(
        "--momask_python",
        default=str(ROOT / "third_party" / "momask_venv" / "Scripts" / "python.exe"),
    )
    p.add_argument("--armature", default="SMPL-X_Armature")
    p.add_argument("--action", default="momask_motion")
    p.add_argument("--out", default="")
    p.add_argument("--ik", action="store_true")
    p.add_argument("--keep_bvh", action="store_true")
    p.add_argument(
        "--flip",
        type=str,
        default="auto",
        help="auto|none|x|y|z — rotate BVH armature 180° if legs inverted",
    )
    return p.parse_args(argv)


def ensure_bvh(args) -> Path:
    if args.bvh:
        bvh = Path(args.bvh)
        if not bvh.is_absolute():
            bvh = (ROOT / bvh).resolve()
        if not bvh.is_file():
            raise SystemExit(f"BVH not found: {bvh}")
        return bvh
    if not args.npz:
        raise SystemExit("Need --bvh or --npz")
    npz = Path(args.npz)
    if not npz.is_absolute():
        npz = (ROOT / npz).resolve()
    bvh = npz.with_suffix(".bvh")
    py = Path(args.momask_python)
    script = ROOT / "tools" / "momask_joints_to_bvh.py"
    cmd = [str(py), str(script), "--npz", str(npz), "--out", str(bvh), "--iterations", "30"]
    if args.ik:
        cmd.append("--ik")
    log("NPZ→BVH via MoMask IK...")
    subprocess.check_call(cmd, cwd=str(ROOT))
    return bvh


def import_bvh(bvh: Path):
    before = set(bpy.data.objects.keys())
    # MoMask BVH is Y-up (same as HumanML3D joints)
    bpy.ops.import_anim.bvh(
        filepath=str(bvh),
        axis_forward="-Z",
        axis_up="Y",
        target="ARMATURE",
        global_scale=1.0,
        frame_start=1,
        use_fps_scale=True,
        update_scene_fps=True,
        update_scene_duration=True,
        use_cyclic=False,
        rotate_mode="NATIVE",
    )
    new = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]
    arms = [o for o in new if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError("BVH import produced no armature")
    src = arms[-1]
    log(f"imported BVH armature: {src.name}")
    if src.animation_data and src.animation_data.action:
        assign_slot(src)
        fr = src.animation_data.action.frame_range
        log(f"source action={src.animation_data.action.name} frames={int(fr[0])}–{int(fr[1])}")
    else:
        log("WARN: BVH armature has no action after import")
    return src


def world_head(arm, bone_name: str) -> Vector:
    pb = arm.pose.bones[bone_name]
    return arm.matrix_world @ pb.head


def legs_point_up(src) -> bool:
    """True if knee is above hip in world Z (Blender Z-up) — inverted legs."""
    bpy.context.view_layer.update()
    try:
        hip = world_head(src, "LeftUpLeg")
        knee = world_head(src, "LeftLeg")
        ankle = world_head(src, "LeftFoot")
    except KeyError:
        return False
    up = knee.z > hip.z + 0.05 or ankle.z > hip.z + 0.05
    log(f"leg check hip.z={hip.z:.3f} knee.z={knee.z:.3f} ankle.z={ankle.z:.3f} inverted={up}")
    return up


def apply_180_fix(src, axis: str):
    """Rotate source armature 180° about world axis and apply."""
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    bpy.context.view_layer.objects.active = src
    e = list(src.rotation_euler)
    if axis == "x":
        src.rotation_euler = Euler((e[0] + math.pi, e[1], e[2]), "XYZ")
    elif axis == "y":
        src.rotation_euler = Euler((e[0], e[1] + math.pi, e[2]), "XYZ")
    elif axis == "z":
        src.rotation_euler = Euler((e[0], e[1], e[2] + math.pi), "XYZ")
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    bpy.context.view_layer.update()
    log(f"applied 180° rotation about {axis.upper()}")


def fix_orientation(src, flip_mode: str):
    """Ensure feet are below hips after import."""
    scene = bpy.context.scene
    f0 = 1
    if src.animation_data and src.animation_data.action:
        f0 = int(src.animation_data.action.frame_range[0])
    scene.frame_set(f0)

    if flip_mode == "none":
        return
    if flip_mode in ("x", "y", "z"):
        apply_180_fix(src, flip_mode)
        return

    if not legs_point_up(src):
        log("legs orientation OK (knees below hips)")
        return
    for axis in ("x", "z", "y"):
        apply_180_fix(src, axis)
        if not legs_point_up(src):
            log(f"auto-fix fixed with 180° on {axis.upper()}")
            return
    log("WARN: could not auto-fix inverted legs; try --flip x|y|z")


def bone_height(arm, a: str, b: str):
    try:
        return (arm.data.bones[b].head_local - arm.data.bones[a].head_local).length
    except KeyError:
        return None


def clear_constraints(pb) -> None:
    while pb.constraints:
        pb.constraints.remove(pb.constraints[0])


def assign_slot(obj) -> None:
    if not obj.animation_data or not obj.animation_data.action:
        return
    try:
        slots = obj.animation_data.action_suitable_slots
        if slots:
            obj.animation_data.action_slot = slots[0]
    except Exception:
        pass


def build_helper_bones(src, tgt, pairs: list[tuple[str, str]]) -> None:
    """
    Rokoko-style: add helper bones on SOURCE with TARGET rest head/tail/roll,
    parented to the animated BVH bone. COPY_ROTATION then reads helpers.
    """
    mw_src_inv = src.matrix_world.inverted()

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="EDIT")

    bone_transforms = {}
    for eb in tgt.data.edit_bones:
        head = mw_src_inv @ (tgt.matrix_world @ eb.head)
        tail = mw_src_inv @ (tgt.matrix_world @ eb.tail)
        bone_transforms[eb.name] = (head.copy(), tail.copy(), eb.roll)
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    bpy.context.view_layer.objects.active = src
    bpy.ops.object.mode_set(mode="EDIT")

    n_helpers = 0
    for sb, sm in pairs:
        parent = src.data.edit_bones.get(sb)
        if parent is None or sm not in bone_transforms:
            continue
        hname = sm + HELPER_SUFFIX
        if hname in src.data.edit_bones:
            src.data.edit_bones.remove(src.data.edit_bones[hname])
        head, tail, roll = bone_transforms[sm]
        nb = src.data.edit_bones.new(hname)
        nb.head = head
        nb.tail = tail
        if (nb.tail - nb.head).length < 1e-5:
            nb.tail = nb.head + Vector((0, 0.05, 0))
        nb.roll = roll
        nb.parent = parent
        nb.use_connect = False
        n_helpers += 1

    bpy.ops.object.mode_set(mode="OBJECT")
    log(f"helper bones created: {n_helpers}")


def retarget_and_bake(src, tgt, action_name: str):
    hs = bone_height(src, "Hips", "Head")
    ht = bone_height(tgt, "pelvis", "head")
    # Keep scale on object — do not apply (same lesson as Mixamo retarget)
    if hs and ht and hs > 1e-6:
        s = ht / hs
        src.scale = (s * src.scale[0], s * src.scale[1], s * src.scale[2])
        bpy.context.view_layer.update()
        log(f"scale source height × {s:.4f} (object scale, not applied)")

    scene = bpy.context.scene
    if not src.animation_data or not src.animation_data.action:
        raise RuntimeError("BVH armature has no action")
    assign_slot(src)
    f0 = int(src.animation_data.action.frame_range[0])
    f1 = int(src.animation_data.action.frame_range[1])
    scene.frame_set(f0)
    bpy.context.view_layer.update()

    # Prove source animation evaluates (Blender 5 needs action_slot)
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        se = src.evaluated_get(dg)
        q_a = se.pose.bones["LeftUpLeg"].matrix.to_quaternion().copy()
        scene.frame_set(min(f0 + 30, f1))
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        se = src.evaluated_get(dg)
        q_b = se.pose.bones["LeftUpLeg"].matrix.to_quaternion().copy()
        d = sum(abs(a - b) for a, b in zip(q_a, q_b))
        log(f"source LeftUpLeg quat delta = {d:.4f}")
        if d < 1e-4:
            log("WARN: source appears static — animation may not be evaluating")
        scene.frame_set(f0)
        bpy.context.view_layer.update()
    except Exception as e:
        log(f"source motion check skipped: {e}")

    try:
        src_hip = src.matrix_world @ src.pose.bones["Hips"].head
        tgt_hip = tgt.matrix_world @ tgt.pose.bones["pelvis"].head
        src.location += tgt_hip - src_hip
        bpy.context.view_layer.update()
        log("aligned Hips → pelvis")
    except Exception as e:
        log(f"align skipped: {e}")

    if legs_point_up(src):
        log("WARN: source legs inverted before helper build — trying X flip")
        apply_180_fix(src, "x")
        scene.frame_set(f0)
        bpy.context.view_layer.update()

    pairs: list[tuple[str, str]] = []
    for bvh_name, smplx_name in BVH_TO_SMPLX.items():
        if bvh_name not in src.pose.bones or smplx_name not in tgt.pose.bones:
            log(f"  skip {bvh_name} → {smplx_name}")
            continue
        pairs.append((bvh_name, smplx_name))
    if len(pairs) < 10:
        raise SystemExit(f"too few bone pairs: {len(pairs)}")
    log(f"mapped pairs: {len(pairs)}")

    build_helper_bones(src, tgt, pairs)

    # Constraints: target bones follow helpers (SMPL-X rest axes, driven by BVH motion)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="POSE")

    for pb in tgt.pose.bones:
        clear_constraints(pb)
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.select = False

    mapped = 0
    for bvh_name, smplx_name in pairs:
        hname = smplx_name + HELPER_SUFFIX
        if hname not in src.pose.bones:
            log(f"  no helper for {smplx_name}")
            continue
        pb = tgt.pose.bones[smplx_name]
        c = pb.constraints.new("COPY_ROTATION")
        c.name = "MAP_COPY_ROT"
        c.target = src
        c.subtarget = hname
        c.mix_mode = "REPLACE"
        # Default spaces (LOCAL) work with helpers; WORLD also OK after helpers
        c.target_space = "WORLD"
        c.owner_space = "WORLD"
        if smplx_name == "pelvis":
            # Root location from real BVH Hips (not helper — helpers sit at rest offset)
            cl = pb.constraints.new("COPY_LOCATION")
            cl.name = "MAP_COPY_LOC"
            cl.target = src
            cl.subtarget = bvh_name
            cl.target_space = "WORLD"
            cl.owner_space = "WORLD"
        pb.select = True
        mapped += 1

    log(f"constraints: {mapped} bones  bake {f0}–{f1}")

    # Manual visual bake (Blender 5-friendly, same as retarget_final_one)
    if not tgt.animation_data:
        tgt.animation_data_create()
    old = bpy.data.actions.get(action_name)
    if old:
        if tgt.animation_data.action == old:
            tgt.animation_data.action = None
        bpy.data.actions.remove(old)

    act = bpy.data.actions.new(action_name)
    act.use_fake_user = True
    tgt.animation_data.action = act
    assign_slot(tgt)

    bones = [tgt.pose.bones[sm] for _, sm in pairs if sm in tgt.pose.bones]
    pelvis0 = None
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        arm_e = tgt.evaluated_get(dg)
        for pb in bones:
            pb_e = arm_e.pose.bones[pb.name]
            mat_local = tgt.convert_space(
                pose_bone=pb, matrix=pb_e.matrix, from_space="POSE", to_space="LOCAL"
            )
            loc, rot, _sca = mat_local.decompose()
            if pb.name == "pelvis":
                if pelvis0 is None:
                    pelvis0 = loc.copy()
                # relative root so clip starts near origin
                pb.location = loc - pelvis0
                pb.keyframe_insert("location", frame=f)
            pb.rotation_quaternion = rot
            pb.keyframe_insert("rotation_quaternion", frame=f)

    for pb in tgt.pose.bones:
        clear_constraints(pb)
        pb.select = False

    tgt.animation_data.action = act
    assign_slot(tgt)
    log(f"action ready: {act.name}")

    # Verify target legs after bake
    scene.frame_set(f0)
    bpy.context.view_layer.update()
    try:
        hip = tgt.matrix_world @ tgt.pose.bones["left_hip"].head
        knee = tgt.matrix_world @ tgt.pose.bones["left_knee"].head
        ankle = tgt.matrix_world @ tgt.pose.bones["left_ankle"].head
        ok = knee.z < hip.z and ankle.z < knee.z
        log(
            f"TARGET leg check hip.z={hip.z:.3f} knee.z={knee.z:.3f} "
            f"ankle.z={ankle.z:.3f} OK_downward={ok}"
        )
        thigh = (hip - knee).length
        shin = (knee - ankle).length
        log(f"TARGET bone lengths thigh={thigh:.3f} shin={shin:.3f} (constant on this rig)")
        if not ok:
            log("ERROR: target legs still inverted after helper retarget")
    except Exception as e:
        log(f"target check skipped: {e}")

    # Mid-frame motion check
    mid = (f0 + f1) // 2
    try:
        scene.frame_set(f0)
        bpy.context.view_layer.update()
        q0 = tgt.pose.bones["left_hip"].rotation_quaternion.copy()
        scene.frame_set(mid)
        bpy.context.view_layer.update()
        q1 = tgt.pose.bones["left_hip"].rotation_quaternion.copy()
        delta = sum(abs(a - b) for a, b in zip(q0, q1))
        log(f"left_hip quat delta f{f0}→f{mid} = {delta:.4f}")
    except Exception:
        pass

    scene.frame_start = f0
    scene.frame_end = f1
    scene.frame_set(f0)
    return f0, f1


def main():
    args = parse_args()
    bvh = ensure_bvh(args)

    tgt = bpy.data.objects.get(args.armature)
    if not tgt or tgt.type != "ARMATURE":
        raise SystemExit(f"Missing {args.armature}")

    src = import_bvh(bvh)
    fix_orientation(src, args.flip)

    f0, f1 = retarget_and_bake(src, tgt, args.action)

    if not args.keep_bvh:
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.data.objects.remove(src, do_unlink=True)
        log("removed temp BVH armature")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            bpy.ops.wm.save_as_mainfile(filepath=str(out))
            log(f"saved {out}")
        except RuntimeError as e:
            alt = out.with_name(out.stem + "_v2" + out.suffix)
            log(f"save failed ({e}); trying {alt}")
            bpy.ops.wm.save_as_mainfile(filepath=str(alt))
            log(f"saved {alt}")

    log(f"DONE {f0}-{f1} action={args.action}")


if __name__ == "__main__":
    main()
