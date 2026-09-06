"""
One Blender process: MoMask BVH → KeeMap Mixamo → SMPL-X Action.

Faster than two separate Blender launches (keemap_exact + retarget_rsl).

Usage:
  blender.exe whole_body_retargeted.blend --background \\
    --python tools/momask_bvh_to_smplx_fast.py -- \\
    --bvh path/to/sample_ik.bvh \\
    --action momask_xxx \\
    --out body_motion/momask_cache/momask_xxx.blend \\
    --rokoko_map "final hml3dto smpl.json"
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
KEEMAP_PARENT = ROOT / "third_party" / "keemap"
MAP_KEEMAP = ROOT / "third_party" / "momask-codes" / "assets" / "mapping.json"
FINAL_MAP_PATH = ROOT / "body_motion" / "FINAL_BONE_MAP.json"
HELPER_SUFFIX = "_RSL_H"

# After Mixamo→SMPL-X, arms often sit slightly *back* (posterior). These local
# euler deltas (radians applied as pre-multiply on local quat) pull them forward.
# Axis tuned for SMPL-X rest T-pose in our production blends (Z-up, face -Y).
# Override with env: SHOULDER_FWD_DEG, COLLAR_FWD_DEG
def _shoulder_fix_quats(shoulder_fwd_deg: float, collar_fwd_deg: float) -> dict:
    """Local correction quats: q_fixed = q_delta @ q_baked."""
    sf = math.radians(float(shoulder_fwd_deg))
    cf = math.radians(float(collar_fwd_deg))
    # X: bring arm forward (counter "shoulders back"); small Z: open collar slightly
    return {
        "left_shoulder": Euler((sf, 0.0, -sf * 0.15), "XYZ").to_quaternion(),
        "right_shoulder": Euler((sf, 0.0, sf * 0.15), "XYZ").to_quaternion(),
        "left_collar": Euler((cf * 0.5, 0.0, -cf), "XYZ").to_quaternion(),
        "right_collar": Euler((cf * 0.5, 0.0, cf), "XYZ").to_quaternion(),
    }


def log(msg: str) -> None:
    print(f"[momask_fast] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = {}
    def get(flag, default=""):
        if flag in argv:
            i = argv.index(flag)
            return argv[i + 1] if i + 1 < len(argv) else default
        return default
    def getf(flag, default):
        try:
            return float(get(flag, str(default)))
        except ValueError:
            return float(default)

    return {
        "bvh": get("--bvh"),
        "action": get("--action", "momask_motion"),
        "out": get("--out", str(ROOT / "body_motion" / "momask_cache" / "momask_out.blend")),
        "rokoko_map": get("--rokoko_map", str(ROOT / "final hml3dto smpl.json")),
        "mixamo": get("--mixamo", str(ROOT / "body_motion" / "source_fbx" / "Idle.fbx")),
        "root": get("--root", "location"),
        # Degrees to pull arms forward (fix shoulders-back look). 0 = off.
        "shoulder_fwd_deg": getf(
            "--shoulder-fwd-deg",
            os.environ.get("SHOULDER_FWD_DEG", "8"),
        ),
        "collar_fwd_deg": getf(
            "--collar-fwd-deg",
            os.environ.get("COLLAR_FWD_DEG", "4"),
        ),
    }


def assign_slot(obj) -> None:
    if not obj.animation_data or not obj.animation_data.action:
        return
    try:
        slots = obj.animation_data.action_suitable_slots
        if slots:
            obj.animation_data.action_slot = slots[0]
    except Exception:
        pass


def set_active(obj) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    try:
        obj.hide_set(False)
        obj.hide_viewport = False
    except Exception:
        pass
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass


def enable_keemap() -> None:
    if str(KEEMAP_PARENT) not in sys.path:
        sys.path.insert(0, str(KEEMAP_PARENT))
    import KeeMapAnimRetarget
    try:
        KeeMapAnimRetarget.unregister()
    except Exception:
        pass
    KeeMapAnimRetarget.register()
    log("KeeMap registered")


def find_src_bone(arm, mixamo_full: str) -> str | None:
    if mixamo_full in arm.pose.bones:
        return mixamo_full
    short = mixamo_full.split(":")[-1]
    for b in arm.pose.bones:
        if b.name.split(":")[-1] == short or b.name == short:
            return b.name
    return None


def height_hips_head(arm, is_mixamo: bool) -> float | None:
    if is_mixamo:
        h = find_src_bone(arm, "mixamorig:Hips")
        hd = find_src_bone(arm, "mixamorig:Head")
        if not h or not hd:
            return None
        a = arm.matrix_world @ arm.data.bones[h].head_local
        b = arm.matrix_world @ arm.data.bones[hd].head_local
    else:
        if "pelvis" not in arm.data.bones or "head" not in arm.data.bones:
            return None
        a = arm.matrix_world @ arm.data.bones["pelvis"].head_local
        b = arm.matrix_world @ arm.data.bones["head"].head_local
    return (b - a).length


def clear_constraints(pb) -> None:
    while pb.constraints:
        pb.constraints.remove(pb.constraints[0])


def keemap_bvh_to_mixamo(bvh: Path, mixamo_fbx: Path, map_path: Path, *, keep_bvh: bool = True):
    enable_keemap()
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
    log(f"BVH {src.name} frames={f0}-{f1}")
    bvh_arm = src  # keep reference for aim fix

    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(
        filepath=str(mixamo_fbx),
        ignore_leaf_bones=False,
        automatic_bone_orientation=False,
    )
    dst = next(
        bpy.data.objects[k]
        for k in bpy.data.objects.keys()
        if k not in before and bpy.data.objects[k].type == "ARMATURE"
    )
    if dst.animation_data and dst.animation_data.action:
        dst.animation_data.action = None
    log(f"Mixamo {dst.name} bones={len(dst.data.bones)}")

    set_active(dst)
    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    dst.select_set(True)
    bpy.context.view_layer.objects.active = dst
    try:
        bpy.ops.object.mode_set(mode="POSE")
    except Exception:
        pass

    scene = bpy.context.scene
    KeeMap = scene.keemap_settings
    KeeMap.bone_mapping_file = str(map_path)
    bpy.ops.wm.keemap_read_file()
    KeeMap.source_rig_name = src.name
    KeeMap.destination_rig_name = dst.name
    KeeMap.start_frame_to_apply = f0
    KeeMap.number_of_frames_to_apply = n_frames
    KeeMap.keyframe_every_n_frames = 1
    if hasattr(KeeMap, "bone_rotation_mode"):
        KeeMap.bone_rotation_mode = "EULER"

    valid = 0
    for item in scene.keemap_bone_mapping_list:
        if item.SourceBoneName in src.pose.bones and item.DestinationBoneName in dst.pose.bones:
            valid += 1
        else:
            item.set_bone_rotation = False
            item.keyframe_this_bone = False
    log(f"KeeMap valid pairs={valid}")
    bpy.ops.wm.perform_animation_transfer()
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    assign_slot(dst)
    # Keep BVH for data-driven shoulder aim (official source). Remove only if not keep_bvh.
    if not keep_bvh:
        try:
            bpy.data.objects.remove(src, do_unlink=True)
        except Exception:
            pass
        bvh_arm = None
    return dst, f0, f1, bvh_arm


def _apply_shoulder_forward_fix(
    tgt,
    action_name: str,
    f0: int,
    f1: int,
    *,
    shoulder_fwd_deg: float = 8.0,
    collar_fwd_deg: float = 4.0,
) -> None:
    """Legacy fixed-euler fallback (used only if aim-fix unavailable)."""
    if abs(float(shoulder_fwd_deg)) < 0.05 and abs(float(collar_fwd_deg)) < 0.05:
        return
    act = bpy.data.actions.get(action_name)
    if act is None or not tgt.animation_data:
        return
    tgt.animation_data.action = act
    assign_slot(tgt)
    fixes = _shoulder_fix_quats(shoulder_fwd_deg, collar_fwd_deg)
    names = [n for n in fixes if n in tgt.pose.bones]
    if not names:
        return
    log(
        f"shoulder fix (euler fallback): fwd_shoulder={shoulder_fwd_deg:.1f}° "
        f"fwd_collar={collar_fwd_deg:.1f}° bones={names}"
    )
    for f in range(int(f0), int(f1) + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        for bname in names:
            pb = tgt.pose.bones[bname]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = fixes[bname] @ pb.rotation_quaternion
            pb.keyframe_insert("rotation_quaternion", frame=f)
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()


def _swing_align(from_dir: Vector, to_dir: Vector) -> Quaternion:
    """Shortest rotation taking from_dir → to_dir (unit vectors)."""
    a = from_dir.normalized() if from_dir.length > 1e-8 else Vector((0, 0, 1))
    b = to_dir.normalized() if to_dir.length > 1e-8 else Vector((0, 0, 1))
    c = max(-1.0, min(1.0, a.dot(b)))
    if c > 0.999999:
        return Quaternion((1, 0, 0, 0))
    if c < -0.999999:
        # 180° — pick any orthogonal axis
        axis = a.cross(Vector((1, 0, 0)))
        if axis.length < 1e-4:
            axis = a.cross(Vector((0, 1, 0)))
        return Quaternion(axis.normalized(), math.pi)
    axis = a.cross(b).normalized()
    return Quaternion(axis, math.acos(c))


def _apply_shoulder_aim_from_source(
    src_ref,
    tgt,
    action_name: str,
    f0: int,
    f1: int,
    *,
    ref_kind: str = "bvh",
) -> dict:
    """
    Data-driven: swing left/right_shoulder so SMPL-X shoulder→elbow matches
    the *official source* upper-arm direction each frame.

    Prefer official MoMask BVH (HML LeftArm→LeftForeArm) after height scale —
    not Mixamo intermediate (KeeMap can already bias shoulders back).
    """
    act = bpy.data.actions.get(action_name)
    if act is None or src_ref is None:
        return {"ok": False, "error": "no action or ref"}
    if not tgt.animation_data:
        tgt.animation_data_create()
    tgt.animation_data.action = act
    assign_slot(tgt)

    # Resolve bone names on reference
    pairs = []
    if ref_kind == "bvh":
        for side, arm_n, fore_n, sm_sh, sm_el in (
            ("L", "LeftArm", "LeftForeArm", "left_shoulder", "left_elbow"),
            ("R", "RightArm", "RightForeArm", "right_shoulder", "right_elbow"),
        ):
            if arm_n not in src_ref.pose.bones or fore_n not in src_ref.pose.bones:
                continue
            if sm_sh not in tgt.pose.bones or sm_el not in tgt.pose.bones:
                continue
            pairs.append((side, arm_n, fore_n, sm_sh, sm_el))
    else:
        for side, mix_arm, mix_fore, sm_sh, sm_el in (
            ("L", "mixamorig:LeftArm", "mixamorig:LeftForeArm", "left_shoulder", "left_elbow"),
            ("R", "mixamorig:RightArm", "mixamorig:RightForeArm", "right_shoulder", "right_elbow"),
        ):
            sa = find_src_bone(src_ref, mix_arm)
            sf = find_src_bone(src_ref, mix_fore)
            if not sa or not sf:
                continue
            if sm_sh not in tgt.pose.bones or sm_el not in tgt.pose.bones:
                continue
            pairs.append((side, sa, sf, sm_sh, sm_el))

    if not pairs:
        return {"ok": False, "error": "no shoulder pairs"}

    ang_before = []
    ang_after = []
    for f in range(int(f0), int(f1) + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        src_e = src_ref.evaluated_get(dg)

        for side, sa, sf, sm_sh, sm_el in pairs:
            sh_m = src_ref.matrix_world @ src_e.pose.bones[sa].head
            el_m = src_ref.matrix_world @ src_e.pose.bones[sf].head
            ref_dir = el_m - sh_m
            if ref_dir.length < 1e-6:
                continue

            # Re-eval target after previous side update
            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            tgt_e = tgt.evaluated_get(dg)
            sh_s = tgt.matrix_world @ tgt_e.pose.bones[sm_sh].head
            el_s = tgt.matrix_world @ tgt_e.pose.bones[sm_el].head
            cur_dir = el_s - sh_s
            if cur_dir.length < 1e-6:
                continue

            c0 = max(-1.0, min(1.0, cur_dir.normalized().dot(ref_dir.normalized())))
            ang_before.append(math.degrees(math.acos(c0)))

            R_swing = _swing_align(cur_dir, ref_dir)

            pb = tgt.pose.bones[sm_sh]
            mat_w = tgt.convert_space(
                pose_bone=pb, matrix=pb.matrix, from_space="POSE", to_space="WORLD"
            ).copy()
            head = mat_w.translation.copy()
            rot_w = R_swing.to_matrix() @ mat_w.to_3x3()
            mat_w_new = rot_w.to_4x4()
            mat_w_new.translation = head

            mat_pose = tgt.matrix_world.inverted() @ mat_w_new
            bone = pb.bone
            if pb.parent:
                pre = (
                    pb.parent.matrix
                    @ pb.parent.bone.matrix_local.inverted()
                    @ bone.matrix_local
                )
                basis = pre.inverted() @ mat_pose
            else:
                basis = bone.matrix_local.inverted() @ mat_pose
            _loc, rot, _sc = basis.decompose()
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = rot
            pb.keyframe_insert("rotation_quaternion", frame=f)

            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            tgt_e2 = tgt.evaluated_get(dg)
            sh2 = tgt.matrix_world @ tgt_e2.pose.bones[sm_sh].head
            el2 = tgt.matrix_world @ tgt_e2.pose.bones[sm_el].head
            d2 = el2 - sh2
            if d2.length > 1e-6:
                c1 = max(-1.0, min(1.0, d2.normalized().dot(ref_dir.normalized())))
                ang_after.append(math.degrees(math.acos(c1)))

    mean_b = sum(ang_before) / max(1, len(ang_before))
    mean_a = sum(ang_after) / max(1, len(ang_after))
    log(
        f"shoulder AIM vs {ref_kind}: frames={f0}-{f1} "
        f"err_before≈{mean_b:.1f}° → after≈{mean_a:.1f}°"
    )
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    return {
        "ok": True,
        "mean_err_before_deg": round(mean_b, 2),
        "mean_err_after_deg": round(mean_a, 2),
        "ref_kind": ref_kind,
        "n": len(ang_before),
    }


def bake_mixamo_to_smplx(
    src,
    tgt,
    action_name: str,
    rokoko_map: Path,
    root_mode: str,
    f0: int,
    f1: int,
    *,
    shoulder_fwd_deg: float = 8.0,
    collar_fwd_deg: float = 4.0,
):
    # Scale/align
    bpy.context.view_layer.update()
    hs, ht = height_hips_head(src, True), height_hips_head(tgt, False)
    if hs and ht and hs > 1e-6:
        s = ht / hs
        src.scale = (s * src.scale[0], s * src.scale[1], s * src.scale[2])
        bpy.context.view_layer.update()
        log(f"scale {s:.4f}")

    hips = find_src_bone(src, "mixamorig:Hips")
    if hips and "pelvis" in tgt.data.bones:
        sh = src.matrix_world @ src.data.bones[hips].head_local
        th = tgt.matrix_world @ tgt.data.bones["pelvis"].head_local
        src.location += th - sh
        bpy.context.view_layer.update()

    # Bone map: FINAL + Rokoko custom overrides
    bone_map = dict(json.loads(FINAL_MAP_PATH.read_text(encoding="utf-8"))["map"])
    if rokoko_map.is_file() and rokoko_map.suffix == ".json":
        data = json.loads(rokoko_map.read_text(encoding="utf-8"))
        # map Rokoko parts → mixamo (minimal critical overrides)
        part_to_mix = {
            "leftUpperArm": "mixamorig:LeftArm",
            "rightUpperArm": "mixamorig:RightArm",
            "leftToe": "mixamorig:LeftToeBase",
            "rightToe": "mixamorig:RightToeBase",
            "leftThumbProximal": "mixamorig:LeftHandThumb1",
            "leftThumbMedial": "mixamorig:LeftHandThumb2",
            "leftThumbDistal": "mixamorig:LeftHandThumb3",
            "leftIndexProximal": "mixamorig:LeftHandIndex1",
            "leftIndexMedial": "mixamorig:LeftHandIndex2",
            "leftIndexDistal": "mixamorig:LeftHandIndex3",
            "leftMiddleDistal": "mixamorig:LeftHandMiddle3",
            "leftRingProximal": "mixamorig:LeftHandRing1",
            "leftRingMedial": "mixamorig:LeftHandRing2",
            "leftRingDistal": "mixamorig:LeftHandRing3",
            "leftLittleProximal": "mixamorig:LeftHandPinky1",
            "leftLittleMedial": "mixamorig:LeftHandPinky2",
            "leftLittleDistal": "mixamorig:LeftHandPinky3",
            "rightThumbProximal": "mixamorig:RightHandThumb1",
            "rightThumbMedial": "mixamorig:RightHandThumb2",
            "rightThumbDistal": "mixamorig:RightHandThumb3",
            "rightIndexProximal": "mixamorig:RightHandIndex1",
            "rightIndexMedial": "mixamorig:RightHandIndex2",
            "rightIndexDistal": "mixamorig:RightHandIndex3",
            "rightMiddleDistal": "mixamorig:RightHandMiddle3",
            "rightRingProximal": "mixamorig:RightHandRing1",
            "rightRingMedial": "mixamorig:RightHandRing2",
            "rightRingDistal": "mixamorig:RightHandRing3",
            "rightLittleProximal": "mixamorig:RightHandPinky1",
            "rightLittleMedial": "mixamorig:RightHandPinky2",
            "rightLittleDistal": "mixamorig:RightHandPinky3",
        }
        for part, targets in (data.get("bones") or {}).items():
            if not targets:
                continue
            mix = part_to_mix.get(part)
            if mix:
                bone_map[mix] = targets[0]
    bone_map["mixamorig:LeftArm"] = "left_shoulder"
    bone_map["mixamorig:RightArm"] = "right_shoulder"

    pairs = []
    seen = set()
    for mix_name, sm in bone_map.items():
        if sm in seen or sm not in tgt.pose.bones:
            continue
        sb = find_src_bone(src, mix_name)
        if not sb:
            continue
        seen.add(sm)
        pairs.append((sb, sm))
    log(f"bake pairs={len(pairs)} frames={f0}-{f1}")

    # helpers
    mw_inv = src.matrix_world.inverted()
    set_active(tgt)
    bpy.ops.object.mode_set(mode="EDIT")
    transforms = {}
    for eb in tgt.data.edit_bones:
        transforms[eb.name] = (
            (mw_inv @ (tgt.matrix_world @ eb.head)).copy(),
            (mw_inv @ (tgt.matrix_world @ eb.tail)).copy(),
            eb.roll,
        )
    bpy.ops.object.mode_set(mode="OBJECT")

    set_active(src)
    bpy.ops.object.mode_set(mode="EDIT")
    for sb, sm in pairs:
        parent = src.data.edit_bones.get(sb)
        if parent is None or sm not in transforms:
            continue
        hname = sm + HELPER_SUFFIX
        if hname in src.data.edit_bones:
            src.data.edit_bones.remove(src.data.edit_bones[hname])
        head, tail, roll = transforms[sm]
        nb = src.data.edit_bones.new(hname)
        nb.head, nb.tail, nb.roll = head, tail, roll
        if (nb.tail - nb.head).length < 1e-5:
            nb.tail = nb.head + Vector((0, 0.05, 0))
        nb.parent = parent
        nb.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    set_active(tgt)
    bpy.ops.object.mode_set(mode="POSE")
    for sb, sm in pairs:
        pb = tgt.pose.bones[sm]
        clear_constraints(pb)
        c = pb.constraints.new("COPY_ROTATION")
        c.target = src
        c.subtarget = sm + HELPER_SUFFIX
        c.mix_mode = "REPLACE"
        pb.rotation_mode = "QUATERNION"
        if sm == "pelvis" and root_mode == "location":
            cl = pb.constraints.new("COPY_LOCATION")
            cl.target = src
            cl.subtarget = sb

    old = bpy.data.actions.get(action_name)
    if old:
        if tgt.animation_data and tgt.animation_data.action == old:
            tgt.animation_data.action = None
        bpy.data.actions.remove(old)
    if not tgt.animation_data:
        tgt.animation_data_create()
    act = bpy.data.actions.new(action_name)
    act.use_fake_user = True
    tgt.animation_data.action = act
    assign_slot(tgt)

    bones = [tgt.pose.bones[sm] for _, sm in pairs]
    pelvis0 = None
    for f in range(f0, f1 + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        arm_e = tgt.evaluated_get(dg)
        for pb in bones:
            pb_e = arm_e.pose.bones[pb.name]
            mat_local = tgt.convert_space(
                pose_bone=pb, matrix=pb_e.matrix, from_space="POSE", to_space="LOCAL"
            )
            loc, rot, _ = mat_local.decompose()
            if pb.name == "pelvis":
                if root_mode == "location":
                    if pelvis0 is None:
                        pelvis0 = loc.copy()
                    pb.location = loc - pelvis0
                    pb.keyframe_insert("location", frame=f)
                else:
                    pb.location = (0, 0, 0)
                    pb.keyframe_insert("location", frame=f)
            pb.rotation_quaternion = rot
            pb.keyframe_insert("rotation_quaternion", frame=f)

    for pb in tgt.pose.bones:
        clear_constraints(pb)
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    # Aim fix is applied later when we still have official BVH in the scene
    log(f"action {action_name} constraint-baked {f0}-{f1} (aim next)")
    return f0, f1


def main():
    args = parse_args()
    bvh = Path(args["bvh"])
    if not bvh.is_absolute():
        bvh = ROOT / bvh
    out = Path(args["out"])
    if not out.is_absolute():
        out = ROOT / out
    mixamo = Path(args["mixamo"])
    if not mixamo.is_absolute():
        mixamo = ROOT / mixamo
    rokoko = Path(args["rokoko_map"])
    if not rokoko.is_absolute():
        rokoko = ROOT / rokoko
    action = args["action"]
    root_mode = args["root"]
    sh_fwd = float(args.get("shoulder_fwd_deg", 8.0))
    col_fwd = float(args.get("collar_fwd_deg", 4.0))

    if not bvh.is_file():
        raise SystemExit(f"missing BVH {bvh}")
    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        raise SystemExit("Need SMPL-X_Armature — open whole_body_retargeted.blend")

    # clear other armatures first
    for o in list(bpy.data.objects):
        if o.type == "ARMATURE" and o.name != "SMPL-X_Armature":
            bpy.data.objects.remove(o, do_unlink=True)

    set_active(tgt)
    mix_arm, f0, f1, bvh_arm = keemap_bvh_to_mixamo(bvh, mixamo, MAP_KEEMAP, keep_bvh=True)

    # Scale official BVH to same hips→head as SMPL-X (reference for arm aim)
    if bvh_arm is not None:
        try:
            hs = height_hips_head(bvh_arm, False)
            # BVH uses Hips/Head names
            h_b = None
            if "Hips" in bvh_arm.data.bones and "Head" in bvh_arm.data.bones:
                a = bvh_arm.matrix_world @ bvh_arm.data.bones["Hips"].head_local
                b = bvh_arm.matrix_world @ bvh_arm.data.bones["Head"].head_local
                h_b = (b - a).length
            ht = height_hips_head(tgt, False)
            if h_b and ht and h_b > 1e-6:
                s = ht / h_b
                bvh_arm.scale = (s, s, s)
                bpy.context.view_layer.update()
                sh = bvh_arm.matrix_world @ bvh_arm.data.bones["Hips"].head_local
                th = tgt.matrix_world @ tgt.data.bones["pelvis"].head_local
                bvh_arm.location += th - sh
                bpy.context.view_layer.update()
                log(f"BVH ref scale×{s:.4f} for shoulder aim")
        except Exception as e:
            log(f"BVH scale for aim warn: {e}")

    bake_mixamo_to_smplx(
        mix_arm,
        tgt,
        action,
        rokoko,
        root_mode,
        f0,
        f1,
        shoulder_fwd_deg=sh_fwd,
        collar_fwd_deg=col_fwd,
    )

    # Prefer official BVH upper-arm aim (source of truth)
    aim = {"ok": False}
    if bvh_arm is not None:
        aim = _apply_shoulder_aim_from_source(
            bvh_arm, tgt, action, f0, f1, ref_kind="bvh"
        )
    if not aim.get("ok") and mix_arm is not None:
        aim = _apply_shoulder_aim_from_source(
            mix_arm, tgt, action, f0, f1, ref_kind="mixamo"
        )
    if not aim.get("ok"):
        log(f"aim fix skipped ({aim.get('error')}) — euler fallback")
        _apply_shoulder_forward_fix(
            tgt, action, f0, f1, shoulder_fwd_deg=sh_fwd, collar_fwd_deg=col_fwd
        )

    for o in (mix_arm, bvh_arm):
        if o is None:
            continue
        try:
            bpy.data.objects.remove(o, do_unlink=True)
        except Exception:
            pass

    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.frame_start = f0
    bpy.context.scene.frame_end = f1
    fp = str(out.resolve()).replace("\\", "/")
    try:
        bpy.ops.wm.save_as_mainfile(filepath=fp, compress=True)
    except Exception:
        bpy.ops.wm.save_as_mainfile(filepath=fp, check_existing=False)
    log(f"saved {out}")
    log("DONE ok=True")


if __name__ == "__main__":
    main()
