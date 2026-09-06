"""
Official MoMask Visualization path (README):

  Import BVH + Mixamo T-pose FBX
  KeeMap + assets/mapping.json
  Transfer → export Mixamo FBX

Uses official leg corrections from their mapping.json (feet/legs stay correct).
Recomputes corrections only for upper body so arms animate on our Mixamo FBX
(README optional: adjust rotations for your character).

Usage:
  blender.exe --background --python tools/momask_to_mixamo_official.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
KEEMAP_PARENT = ROOT / "third_party" / "keemap"
MAP_JSON = ROOT / "third_party" / "momask-codes" / "assets" / "mapping.json"


def log(msg: str) -> None:
    print(f"[momask_mixamo] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", default=str(ROOT / "lmm train" / "momask_walk.bvh"))
    # Prefer Idle T-pose character — walk.fbx rest is fine too, but export FBX
    # strips bones; keep the .blend as source of truth for SMPL-X hop.
    p.add_argument("--mixamo", default=str(ROOT / "body_motion" / "source_fbx" / "Idle.fbx"))
    p.add_argument(
        "--out",
        default=str(ROOT / "body_motion" / "source_fbx" / "momask_mixamo_official.fbx"),
    )
    p.add_argument(
        "--save_blend",
        default=str(ROOT / "body_motion" / "_momask_mixamo_only.blend"),
    )
    p.add_argument("--map", default=str(MAP_JSON))
    return p.parse_args(argv)


def enable_keemap() -> None:
    if str(KEEMAP_PARENT) not in sys.path:
        sys.path.insert(0, str(KEEMAP_PARENT))
    import KeeMapAnimRetarget

    try:
        KeeMapAnimRetarget.unregister()
    except Exception:
        pass
    KeeMapAnimRetarget.register()
    log("KeeMap enabled")


def assign_slot(obj) -> None:
    if not obj.animation_data or not obj.animation_data.action:
        return
    try:
        slots = obj.animation_data.action_suitable_slots
        if slots:
            obj.animation_data.action_slot = slots[0]
    except Exception:
        pass


def clear_to_rest(arm) -> None:
    if arm.animation_data:
        arm.animation_data.action = None
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.transforms_clear()
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()


def world_head(arm, name: str) -> Vector:
    return arm.matrix_world @ arm.pose.bones[name].head


def rest_head_world(arm, name: str) -> Vector:
    return arm.matrix_world @ arm.data.bones[name].head_local


def rest_seg_len(arm, a: str, b: str) -> float:
    return (rest_head_world(arm, b) - rest_head_world(arm, a)).length


# BVH bone → Mixamo bone (arm chain only; KeeMap mapping has no hands)
ARM_EDGES = [
    # (bvh_parent, bvh_child, mix_parent, mix_child) — rotate mix_parent toward child
    ("LeftShoulder", "LeftArm", "mixamorig:LeftShoulder", "mixamorig:LeftArm"),
    ("LeftArm", "LeftForeArm", "mixamorig:LeftArm", "mixamorig:LeftForeArm"),
    ("LeftForeArm", "LeftHand", "mixamorig:LeftForeArm", "mixamorig:LeftHand"),
    ("RightShoulder", "RightArm", "mixamorig:RightShoulder", "mixamorig:RightArm"),
    ("RightArm", "RightForeArm", "mixamorig:RightArm", "mixamorig:RightForeArm"),
    ("RightForeArm", "RightHand", "mixamorig:RightForeArm", "mixamorig:RightHand"),
]

ARM_MIX_BONES = [
    "mixamorig:LeftShoulder",
    "mixamorig:LeftArm",
    "mixamorig:LeftForeArm",
    "mixamorig:LeftHand",
    "mixamorig:RightShoulder",
    "mixamorig:RightArm",
    "mixamorig:RightForeArm",
    "mixamorig:RightHand",
]


def swing_aim_bone(arm, bone_name: str, child_name: str, desired_child_w: Vector) -> None:
    """Swing-only: reset bone to rest, rotate so limb aims at desired_child_w."""
    if bone_name not in arm.pose.bones or child_name not in arm.pose.bones:
        return
    pb = arm.pose.bones[bone_name]
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
    pb.location = Vector((0, 0, 0))
    bpy.context.view_layer.update()

    head_w = arm.matrix_world @ pb.head
    child_w = arm.matrix_world @ arm.pose.bones[child_name].head
    v_rest = child_w - head_w
    v_tgt = desired_child_w - head_w
    if v_rest.length < 1e-8 or v_tgt.length < 1e-8:
        return
    v_rest.normalize()
    v_tgt.normalize()
    if v_rest.dot(v_tgt) > 0.999999:
        return

    q = v_rest.rotation_difference(v_tgt)
    M = arm.matrix_world @ pb.matrix
    head = M.to_translation()
    M_new = Matrix.Translation(head) @ q.to_matrix().to_4x4() @ Matrix.Translation(-head) @ M
    mat_arm = arm.matrix_world.inverted() @ M_new
    bone = arm.data.bones[bone_name]
    if pb.parent:
        pre = pb.parent.matrix @ pb.parent.bone.matrix_local.inverted() @ bone.matrix_local
        basis = pre.inverted() @ mat_arm
    else:
        basis = bone.matrix_local.inverted() @ mat_arm
    _loc, rot, _sca = basis.decompose()
    pb.rotation_quaternion = rot
    pb.location = Vector((0, 0, 0))
    bpy.context.view_layer.update()


def fix_mixamo_arms_from_bvh(src, dst, f0: int, f1: int) -> None:
    """
    KeeMap arm corrections often leave Mixamo arms in bad/T-pose orientations.
    Rebuild arms each frame: BVH limb DIRECTION × Mixamo bone LENGTH (swing-only).
    Hands: MoMask has no fingers; wrist follows forearm→hand direction from BVH.
    """
    log(f"3b) fix arms/hands from BVH directions (frames {f0}-{f1})…")
    scene = bpy.context.scene
    assign_slot(src)
    assign_slot(dst)

    # Ensure Mixamo action exists
    if not dst.animation_data:
        dst.animation_data_create()
    if not dst.animation_data.action:
        act = bpy.data.actions.new("Mixamo_MoMask")
        act.use_fake_user = True
        dst.animation_data.action = act
    assign_slot(dst)

    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()

        # Reset arm bones to rest relative to (already KeeMap-posed) spine
        for bn in ARM_MIX_BONES:
            if bn not in dst.pose.bones:
                continue
            pb = dst.pose.bones[bn]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
            pb.location = Vector((0, 0, 0))
        bpy.context.view_layer.update()

        for sa, sb, ma, mb in ARM_EDGES:
            if sa not in src.pose.bones or sb not in src.pose.bones:
                continue
            if ma not in dst.pose.bones or mb not in dst.pose.bones:
                continue
            # direction from BVH
            d = world_head(src, sb) - world_head(src, sa)
            if d.length < 1e-8:
                continue
            d.normalize()
            L = rest_seg_len(dst, ma, mb)
            if L < 1e-8:
                # fallback: pose distance at rest
                L = (dst.data.bones[mb].head_local - dst.data.bones[ma].head_local).length
            p = world_head(dst, ma)
            target = p + d * L
            swing_aim_bone(dst, ma, mb, target)

        # Keyframe arm bones
        for bn in ARM_MIX_BONES:
            if bn not in dst.pose.bones:
                continue
            pb = dst.pose.bones[bn]
            pb.rotation_mode = "QUATERNION"
            pb.keyframe_insert("rotation_quaternion", frame=f)

        if f == f0 or f % 30 == 0 or f == f1:
            arm = world_head(dst, "mixamorig:LeftArm")
            hand = world_head(dst, "mixamorig:LeftHand")
            log(
                f"  armfix f{f}: hand_drop_z={arm.z - hand.z:.3f} "
                f"ab_x={abs(hand.x - arm.x):.3f}"
            )
    log("3b) arms/hands fixed")


def match_source_scale_to_dest(src, dst) -> float:
    """
    Uniform-scale BVH (source) so hips→head length matches Mixamo (dest) in WORLD space.
    Does NOT apply scale (keeps animation keys valid). Prevents height mismatch;
    KeeMap only copies rotations — bone lengths always stay Mixamo's.
    """
    src_h = "Hips"
    src_hd = "Head"
    dst_h = "mixamorig:Hips" if "mixamorig:Hips" in dst.data.bones else "Hips"
    dst_hd = "mixamorig:Head" if "mixamorig:Head" in dst.data.bones else "Head"
    if src_h not in src.data.bones or src_hd not in src.data.bones:
        log("scale skip: BVH missing Hips/Head")
        return 1.0
    if dst_h not in dst.data.bones or dst_hd not in dst.data.bones:
        log("scale skip: Mixamo missing Hips/Head")
        return 1.0
    ls = rest_seg_len(src, src_h, src_hd)
    ld = rest_seg_len(dst, dst_h, dst_hd)
    if ls < 1e-6 or ld < 1e-6:
        return 1.0
    s = ld / ls
    src.scale = (src.scale[0] * s, src.scale[1] * s, src.scale[2] * s)
    bpy.context.view_layer.update()
    # Align hips in world (XY + height) so both skeletons share same root height
    sh = rest_head_world(src, src_h)
    dh = rest_head_world(dst, dst_h)
    src.location += dh - sh
    bpy.context.view_layer.update()
    log(f"scale BVH × {s:.4f} (hips→head BVH={ls:.3f} → Mixamo={ld:.3f}); hips aligned")
    return s


def main() -> None:
    args = parse_args()
    bvh = Path(args.bvh) if Path(args.bvh).is_absolute() else ROOT / args.bvh
    mixamo = Path(args.mixamo) if Path(args.mixamo).is_absolute() else ROOT / args.mixamo
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    map_path = Path(args.map) if Path(args.map).is_absolute() else ROOT / args.map
    save_blend = (
        Path(args.save_blend)
        if Path(args.save_blend).is_absolute()
        else ROOT / args.save_blend
    )

    if not bvh.is_file():
        raise SystemExit(f"BVH missing: {bvh}")
    if not mixamo.is_file():
        raise SystemExit(f"Mixamo FBX missing: {mixamo}")
    if not map_path.is_file():
        raise SystemExit(f"map missing: {map_path}")

    bpy.ops.wm.read_homefile(use_empty=True)
    enable_keemap()

    # 1) BVH
    before = set(bpy.data.objects.keys())
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
        rotate_mode="NATIVE",
    )
    src = next(
        bpy.data.objects[k]
        for k in bpy.data.objects.keys()
        if k not in before and bpy.data.objects[k].type == "ARMATURE"
    )
    assign_slot(src)
    fr = src.animation_data.action.frame_range
    f0, f1 = int(fr[0]), int(fr[1])
    n_frames = f1 - f0 + 1
    log(f"1) BVH source={src.name} frames={f0}-{f1}")

    # 2) Mixamo T-pose character
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(
        filepath=str(mixamo),
        ignore_leaf_bones=True,
        automatic_bone_orientation=False,
    )
    dst = next(
        bpy.data.objects[k]
        for k in bpy.data.objects.keys()
        if k not in before and bpy.data.objects[k].type == "ARMATURE"
    )
    clear_to_rest(dst)
    log(f"2) Mixamo dest={dst.name} file={mixamo.name} (rest/T-pose)")

    # 2b) Scale + align BVH to Mixamo height (uniform object scale only)
    scale_factor = match_source_scale_to_dest(src, dst)

    # 3) KeeMap mapping.json
    scene = bpy.context.scene
    KeeMap = scene.keemap_settings
    KeeMap.bone_mapping_file = str(map_path)
    bpy.ops.wm.keemap_read_file()
    KeeMap.source_rig_name = src.name
    KeeMap.destination_rig_name = dst.name
    KeeMap.start_frame_to_apply = f0
    KeeMap.number_of_frames_to_apply = n_frames
    KeeMap.keyframe_every_n_frames = 1

    bone_list = scene.keemap_bone_mapping_list
    # KeeMap arms often break (T-pose / twisted). Disable arm bones in KeeMap;
    # body/legs use mapping.json; arms fixed in step 3b from BVH directions.
    ARM_SKIP = {
        "LeftShoulder",
        "RightShoulder",
        "LeftArm",
        "RightArm",
        "LeftForeArm",
        "RightForeArm",
        "LeftHand",
        "RightHand",
    }

    valid = 0
    for item in bone_list:
        ok = (
            item.SourceBoneName in src.pose.bones
            and item.DestinationBoneName in dst.pose.bones
        )
        if not ok:
            item.set_bone_rotation = False
            item.set_bone_position = False
            item.keyframe_this_bone = False
            continue
        if item.SourceBoneName in ARM_SKIP:
            item.set_bone_rotation = False
            item.keyframe_this_bone = False
            log(f"  KeeMap skip arm (fix later): {item.SourceBoneName}")
            continue
        valid += 1
    log(f"3) map={map_path.name} body/leg pairs={valid} (arms skipped in KeeMap)")

    # Use mapping.json factors as-is (spine/legs). No calc_all — avoids crushed torso.
    KeeMap.bone_rotation_mode = "EULER"
    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    dst.select_set(True)
    bpy.context.view_layer.objects.active = dst
    bpy.ops.object.mode_set(mode="POSE")
    log("3) Transfer body/legs (KeeMap + your mapping.json)…")
    bpy.ops.wm.perform_animation_transfer()
    bpy.ops.object.mode_set(mode="OBJECT")
    assign_slot(dst)

    # 3b) Arms + hands from BVH limb directions × Mixamo lengths
    fix_mixamo_arms_from_bvh(src, dst, f0, f1)

    # Verify
    def snap(frame: int) -> dict:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        hip = world_head(dst, "mixamorig:LeftUpLeg")
        knee = world_head(dst, "mixamorig:LeftLeg")
        foot = world_head(dst, "mixamorig:LeftFoot")
        arm = world_head(dst, "mixamorig:LeftArm")
        hand = world_head(dst, "mixamorig:LeftHand")
        # spine should go UP: hips < spine < spine2 < head
        hz = world_head(dst, "mixamorig:Hips").z
        s1z = world_head(dst, "mixamorig:Spine").z
        s2z = world_head(dst, "mixamorig:Spine1").z
        s3z = world_head(dst, "mixamorig:Spine2").z
        hdz = world_head(dst, "mixamorig:Head").z
        spine_ok = hz < s1z + 0.02 and s1z < s3z + 0.05 and s3z < hdz + 0.02
        q = list(dst.pose.bones["mixamorig:LeftArm"].rotation_quaternion)
        return {
            "frame": frame,
            "legs_down": bool(knee.z < hip.z + 0.05 and foot.z < knee.z + 0.05),
            "spine_up": bool(spine_ok),
            "spine_z": [round(hz, 3), round(s1z, 3), round(s2z, 3), round(s3z, 3), round(hdz, 3)],
            "hip_z": round(hip.z, 3),
            "knee_z": round(knee.z, 3),
            "foot_z": round(foot.z, 3),
            "hand_drop": round(arm.z - hand.z, 3),
            "LeftArm_q": [round(float(x), 3) for x in q],
        }

    c1, c2 = snap(f0), snap(min(f0 + 30, f1))
    scene.frame_set(f0)
    bpy.context.view_layer.update()
    q0 = dst.pose.bones["mixamorig:LeftArm"].rotation_quaternion.copy()
    scene.frame_set(min(f0 + 30, f1))
    bpy.context.view_layer.update()
    q1 = dst.pose.bones["mixamorig:LeftArm"].rotation_quaternion.copy()
    arm_delta = sum(abs(a - b) for a, b in zip(q0, q1))
    log(f"verify f{f0}: legs={c1['legs_down']} spine_up={c1['spine_up']} spine_z={c1['spine_z']}")
    log(f"verify f+30: legs={c2['legs_down']} spine_up={c2['spine_up']} spine_z={c2['spine_z']}")
    log(f"LeftArm motion delta={arm_delta:.4f} scale_factor={scale_factor:.4f}")

    # 4) Export FBX
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    dst.select_set(True)
    for o in bpy.data.objects:
        if o.parent == dst:
            o.select_set(True)
    bpy.context.view_layer.objects.active = dst
    bpy.ops.export_scene.fbx(
        filepath=str(out),
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_space_transform=False,
        path_mode="COPY",
    )
    log(f"4) FBX exported: {out} ({out.stat().st_size} bytes)")

    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(save_blend))
        log(f"blend: {save_blend}")
    except RuntimeError:
        alt = save_blend.with_name(save_blend.stem + "_v2" + save_blend.suffix)
        bpy.ops.wm.save_as_mainfile(filepath=str(alt))
        log(f"blend: {alt}")

    ok = bool(
        out.is_file()
        and c1["legs_down"]
        and c2["legs_down"]
        and c1.get("spine_up", True)
        and c2.get("spine_up", True)
    )
    rep = {
        "ok": ok,
        "fbx": str(out),
        "blend": str(save_blend),
        "bvh": str(bvh),
        "mixamo": str(mixamo),
        "map": str(map_path),
        "frames": [f0, f1],
        "scale_bvh_to_mixamo": round(scale_factor, 4),
        "arm_delta": round(arm_delta, 4),
        "checks": [c1, c2],
        "how_to_view": "Open blend or FBX, select Mixamo Armature, play timeline",
        "note": "KeeMap does not scale Mixamo bones; BVH is scaled to hips-head match. Spine uses mapping.json (not crushed by calc).",
    }
    rep_path = ROOT / "body_motion" / "momask_mixamo_official_report.json"
    rep_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    log(f"report: {rep_path}")
    log(f"DONE ok={ok}")
    if not ok:
        raise SystemExit("Checks failed — open blend to inspect")


if __name__ == "__main__":
    main()
