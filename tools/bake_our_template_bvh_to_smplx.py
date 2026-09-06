"""
Bake MoMask-named BVH (built on OUR template) onto SMPL-X_Armature.

Uses direct bone map BVH→SMPL-X (no Mixamo KeeMap middle).
Verifies collar vs shoulder head heights after bake.

Usage:
  blender.exe whole_body_retargeted.blend --background --python tools/bake_our_template_bvh_to_smplx.py -- ^
    --bvh body_motion/momask_cache/t2m_walk_hybrik/walk_our_template_ik.bvh ^
    --action walk_our_tpl_ik ^
    --out body_motion/momask_cache/t2m_walk_hybrik/walk_our_template_ik.blend
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]

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

KEY_BONES = list(BVH_TO_SMPLX.values())


def log(msg: str) -> None:
    print(f"[bake_our_tpl] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", required=True)
    p.add_argument("--action", default="walk_our_tpl_ik")
    p.add_argument("--out", required=True)
    p.add_argument("--armature", default="SMPL-X_Armature")
    return p.parse_args(argv)


def assign_slot(obj) -> None:
    if obj.animation_data and obj.animation_data.action:
        try:
            slots = obj.animation_data.action_suitable_slots
            if slots:
                obj.animation_data.action_slot = slots[0]
        except Exception:
            pass


def world_head(arm, name: str) -> Vector:
    return (arm.matrix_world @ arm.pose.bones[name].head).copy()


def import_bvh(path: Path):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_anim.bvh(
        filepath=str(path).replace("\\", "/"),
        axis_forward="-Z",
        axis_up="Y",
        global_scale=1.0,
        use_fps_scale=True,
    )
    after = set(bpy.data.objects.keys())
    new = [bpy.data.objects[n] for n in after - before]
    arms = [o for o in new if o.type == "ARMATURE"]
    if not arms:
        raise SystemExit("BVH import produced no armature")
    return arms[0]


def clear_pose(arm):
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
        pb.location = Vector((0, 0, 0))


def set_pelvis(arm, desired: Vector) -> None:
    pb = arm.pose.bones["pelvis"]
    bone = arm.data.bones["pelvis"]
    rest = bone.matrix_local
    r_inv = rest.to_3x3().inverted()
    t = rest.to_translation()
    arm_inv = arm.matrix_world.inverted()
    desired_a = arm_inv @ desired
    pb.location = r_inv @ (desired_a - t)
    bpy.context.view_layer.update()
    for _ in range(10):
        err = desired - (arm.matrix_world @ pb.head)
        if err.length < 1e-5:
            break
        pb.location += r_inv @ (arm_inv.to_3x3() @ err)
        bpy.context.view_layer.update()


def main():
    args = parse_args()
    bvh = Path(args.bvh)
    if not bvh.is_absolute():
        bvh = ROOT / bvh
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    tgt = bpy.data.objects.get(args.armature)
    if tgt is None or tgt.type != "ARMATURE":
        raise SystemExit(f"need {args.armature}")

    src = import_bvh(bvh)
    log(f"imported BVH armature {src.name}")
    # read frame range from the imported BVH action, not scene
    if src.animation_data and src.animation_data.action:
        fr = src.animation_data.action.frame_range
        f0, f1 = max(1, int(fr[0])), int(fr[1])
        log(f"frames {f0}..{f1} (from BVH action)")
    else:
        f0, f1 = 1, 80
        log(f"frames {f0}..{f1} (fallback)")

    # Map source bone names (may be exact BVH names)
    def find_src(name: str):
        if name in src.pose.bones:
            return name
        for b in src.pose.bones:
            if b.name.endswith(name) or name in b.name:
                return b.name
        return None

    pairs = []
    for bvh_n, sm in BVH_TO_SMPLX.items():
        sn = find_src(bvh_n)
        if sn and sm in tgt.pose.bones:
            pairs.append((sn, sm))
        else:
            log(f"  skip map {bvh_n}→{sm} (src={sn})")
    log(f"mapped {len(pairs)} bones")

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

    # Rest-relative rotation transfer: use matrix copy in world, convert to local pose
    # Precompute rest world rotations
    def rest_world_rot(arm, bone_name: str) -> Matrix:
        b = arm.data.bones[bone_name]
        return (arm.matrix_world @ b.matrix_local).to_3x3()

    # For each frame: copy world delta from rest
    samples_L = []
    for f in range(f0, f1 + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        clear_pose(tgt)
        bpy.context.view_layer.update()

        # root location: Hips — use set_pelvis for correct world-to-local conversion
        hips_src = find_src("Hips")
        if hips_src and "pelvis" in tgt.pose.bones:
            sh = src.matrix_world @ src.pose.bones[hips_src].head
            set_pelvis(tgt, sh)

        for sn, sm in pairs:
            pb = tgt.pose.bones[sm]
            bone = pb.bone

            # Relative rotation: delta from rest pose (works even if rest poses differ)
            src_rest_w = src.matrix_world @ src.data.bones[sn].matrix_local
            src_pose_w = src.matrix_world @ src.pose.bones[sn].matrix
            delta_w = src_rest_w.inverted() @ src_pose_w

            # Apply delta on top of tgt rest world orientation
            tgt_rest_w = tgt.matrix_world @ bone.matrix_local
            desired_w = tgt_rest_w @ delta_w

            # Decompose to tgt pose-bone local rotation
            mat_arm = tgt.matrix_world.inverted() @ desired_w
            if pb.parent:
                pre = (
                    pb.parent.matrix
                    @ pb.parent.bone.matrix_local.inverted()
                    @ bone.matrix_local
                )
                basis = pre.inverted() @ mat_arm
            else:
                basis = bone.matrix_local.inverted() @ mat_arm
            _loc, rot, _sc = basis.decompose()
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = rot
            if sm != "pelvis":
                pb.location = Vector((0, 0, 0))
            bpy.context.view_layer.update()

        for sm in KEY_BONES:
            if sm not in tgt.pose.bones:
                continue
            pb = tgt.pose.bones[sm]
            pb.rotation_mode = "QUATERNION"
            pb.keyframe_insert("rotation_quaternion", frame=f)
            if sm == "pelvis":
                pb.keyframe_insert("location", frame=f)

        # sample collar/shoulder
        if "left_collar" in tgt.pose.bones and "left_shoulder" in tgt.pose.bones:
            lc = world_head(tgt, "left_collar")
            ls = world_head(tgt, "left_shoulder")
            samples_L.append((f, lc.z, ls.z, ls.z - lc.z))

        if f == f0 or f % 20 == 0 or f == f1:
            pel = world_head(tgt, "pelvis") if "pelvis" in tgt.pose.bones else None
            if samples_L:
                _, cz, sz, dz = samples_L[-1]
                pel_str = f" pelvis=({pel.x:.2f},{pel.y:.2f},{pel.z:.2f})" if pel else ""
                log(f"  f{f}: L collar_z={cz:.3f} shoulder_z={sz:.3f} Δz={dz:.4f}{pel_str}")

    # rest reference
    bpy.context.scene.frame_set(f0)
    clear_pose(tgt)
    bpy.context.view_layer.update()
    rest_dz = (world_head(tgt, "left_shoulder") - world_head(tgt, "left_collar")).z

    if samples_L:
        dzs = [s[3] for s in samples_L]
        mean_dz = sum(dzs) / len(dzs)
        log(f"VERIFY L shoulder-collar Δz: mean={mean_dz:.4f} rest_pose_Δz={rest_dz:.4f}")
        ok = abs(mean_dz - rest_dz) < 0.05 and mean_dz > 0.02
        log(f"VERIFY collar/shoulder levels: {'PASS' if ok else 'CHECK'} (expect non-zero Δ like rest)")
    else:
        ok = False
        mean_dz = 0.0

    # remove imported BVH armature
    bpy.data.objects.remove(src, do_unlink=True)

    bpy.context.scene.frame_start = f0
    bpy.context.scene.frame_end = f1
    out.parent.mkdir(parents=True, exist_ok=True)
    fp = str(out.resolve()).replace("\\", "/")
    bpy.ops.wm.save_as_mainfile(filepath=fp, compress=True)
    log(f"saved {out}")

    report = {
        "bvh": str(bvh),
        "action": args.action,
        "out": str(out),
        "L_shoulder_collar_dz_mean": mean_dz,
        "L_shoulder_collar_dz_rest": rest_dz,
        "collar_shoulder_ok": bool(ok),
        "method": "momask_ik_our_smplx_template",
    }
    (out.parent / f"{args.action}_verify.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    log(f"report {report}")
    log("DONE")


if __name__ == "__main__":
    main()
