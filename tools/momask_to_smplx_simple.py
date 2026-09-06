"""
Simple path = MoMask demos + our working Mixamo→SMPL-X.

  whole_body_retargeted.blend (has SMPL-X)
    1. Import MoMask BVH
    2. Import Mixamo FBX, clear to T-pose rest
    3. KeeMap (mapping + auto corrections for this character)
    4. SAME SCENE: helper-bone retarget Mixamo → SMPL-X  (NO fbx re-export)
    5. Save action on SMPL-X_Armature

Why no FBX export between 3 and 4:
  Export/import rewrites Mixamo rest axes → arms stick in T-pose on SMPL-X.
  Catalog walk.fbx works because rest is clean Mixamo T-pose.

Usage:
  blender.exe --background --python tools/momask_to_smplx_simple.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
KEEMAP_PARENT = ROOT / "third_party" / "keemap"
MAP_FILE = ROOT / "body_motion" / "mapping_momask_mixamo_hands.json"
MAP_FALLBACK = ROOT / "third_party" / "momask-codes" / "assets" / "mapping.json"
BVH = ROOT / "lmm train" / "momask_walk.bvh"
MIXAMO_FBX = ROOT / "body_motion" / "source_fbx" / "Idle.fbx"
# Prefer walk.fbx skeleton (same rest as catalog retarget that already works)
MIXAMO_ALT = ROOT / "body_motion" / "source_fbx" / "walk.fbx"
SMPLX_BLEND = ROOT / "whole_body_retargeted.blend"
OUT_BLEND = ROOT / "body_motion" / "_momask_via_mixamo.blend"
ACTION = "momask_via_mixamo"
FINAL_MAP = ROOT / "body_motion" / "FINAL_BONE_MAP.json"
HELPER_SUFFIX = "_RSL_H"
REPORT = ROOT / "body_motion" / "momask_simple_report.json"


def log(msg: str) -> None:
    print(f"[simple] {msg}", flush=True)


def enable_keemap() -> None:
    if str(KEEMAP_PARENT) not in sys.path:
        sys.path.insert(0, str(KEEMAP_PARENT))
    import KeeMapAnimRetarget

    try:
        KeeMapAnimRetarget.unregister()
    except Exception:
        pass
    KeeMapAnimRetarget.register()
    log("KeeMap ON")


def assign_slot(obj) -> None:
    if not obj.animation_data or not obj.animation_data.action:
        return
    try:
        slots = obj.animation_data.action_suitable_slots
        if slots:
            obj.animation_data.action_slot = slots[0]
    except Exception:
        pass


def reset_pose(arm) -> None:
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


def clear_constraints(pb) -> None:
    while pb.constraints:
        pb.constraints.remove(pb.constraints[0])


def world_head(arm, name: str) -> Vector:
    return arm.matrix_world @ arm.pose.bones[name].head


def strip_prefix(name: str) -> str:
    return name.split(":")[-1].replace("mixamorig", "")


def find_src_bone(arm, mixamo_full: str) -> str | None:
    if mixamo_full in arm.pose.bones:
        return mixamo_full
    short = strip_prefix(mixamo_full)
    for b in arm.pose.bones:
        if strip_prefix(b.name) == short or b.name == short:
            return b.name
    alt = mixamo_full.replace("mixamorig:", "")
    if alt in arm.pose.bones:
        return alt
    return None


def height_hips_head(arm, mixamo: bool) -> float | None:
    if mixamo:
        h = find_src_bone(arm, "mixamorig:Hips") or find_src_bone(arm, "Hips")
        hd = find_src_bone(arm, "mixamorig:Head") or find_src_bone(arm, "Head")
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


def check_arms(arm, mixamo: bool, frame: int) -> dict:
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    if mixamo:
        sh_n, el_n, wr_n = "mixamorig:LeftArm", "mixamorig:LeftForeArm", "mixamorig:LeftHand"
    else:
        sh_n, el_n, wr_n = "left_shoulder", "left_elbow", "left_wrist"
    sh, el, wr = world_head(arm, sh_n), world_head(arm, el_n), world_head(arm, wr_n)
    ab = abs(wr.x - sh.x)
    drop = sh.z - wr.z
    q = arm.pose.bones[sh_n].rotation_quaternion
    # good walk-like: wrists not far out (ab small), hands below shoulders
    ok = ab < 0.35 and drop > 0.2
    d = {
        "frame": frame,
        "ab_x": round(ab, 3),
        "drop_z": round(drop, 3),
        "sh_q": [round(float(x), 3) for x in q],
        "ok": bool(ok),
    }
    log(f"  arms f{frame}: ab_x={d['ab_x']} drop_z={d['drop_z']} ok={ok}")
    return d


def retarget_mixamo_to_smplx_inplace(src, tgt, action_name: str) -> tuple[int, int]:
    """Same method as retarget_final_one — proven for Mixamo→SMPL-X."""
    bone_map = json.loads(FINAL_MAP.read_text(encoding="utf-8"))["map"]

    # scale source to target height (do NOT apply scale)
    bpy.context.view_layer.update()
    hs, ht = height_hips_head(src, True), height_hips_head(tgt, False)
    if hs and ht and hs > 1e-6:
        s = ht / hs
        src.scale = (s * src.scale[0], s * src.scale[1], s * src.scale[2])
        bpy.context.view_layer.update()
        log(f"scale Mixamo × {s:.4f}")

    hips = find_src_bone(src, "mixamorig:Hips")
    if hips and "pelvis" in tgt.data.bones:
        sh = src.matrix_world @ src.data.bones[hips].head_local
        th = tgt.matrix_world @ tgt.data.bones["pelvis"].head_local
        src.location += th - sh
        bpy.context.view_layer.update()

    pairs: list[tuple[str, str]] = []
    for mix_name, sm_name in bone_map.items():
        if sm_name not in tgt.pose.bones:
            continue
        sb = find_src_bone(src, mix_name)
        if not sb:
            continue
        pairs.append((sb, sm_name))
    log(f"Mixamo→SMPL-X pairs={len(pairs)}")
    if len(pairs) < 10:
        raise SystemExit("too few pairs")

    # helpers = SMPL-X rest axes parented to Mixamo bones
    mw_src_inv = src.matrix_world.inverted()
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
    bpy.ops.object.mode_set(mode="OBJECT")

    act_src = src.animation_data.action
    assign_slot(src)
    f0, f1 = int(act_src.frame_range[0]), int(act_src.frame_range[1])
    log(f"bake frames {f0}-{f1}")

    # root travel
    def hips_w(frame: int):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        se = src.evaluated_get(dg)
        return (src.matrix_world @ se.pose.bones[hips].matrix).to_translation().copy()

    root_mode = "inplace"
    if hips:
        p_a, p_b, p_c = hips_w(f0), hips_w((f0 + f1) // 2), hips_w(f1)
        travel = max((p_b - p_a).length, (p_c - p_a).length, (p_c - p_b).length)
        if travel > 0.15:
            root_mode = "location"
        log(f"root_mode={root_mode} travel≈{travel:.3f}")

    bpy.ops.object.select_all(action="DESELECT")
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="POSE")
    for pb in tgt.pose.bones:
        clear_constraints(pb)
        pb.select = False

    root_names = {"pelvis"} if root_mode == "location" else set()
    for sb, sm in pairs:
        pb = tgt.pose.bones[sm]
        c = pb.constraints.new("COPY_ROTATION")
        c.name = "Copy Rot" + HELPER_SUFFIX
        c.target = src
        c.subtarget = sm + HELPER_SUFFIX
        c.mix_mode = "REPLACE"
        pb.rotation_mode = "QUATERNION"
        pb.select = True
        if sm in root_names:
            cl = pb.constraints.new("COPY_LOCATION")
            cl.name = "Copy Loc" + HELPER_SUFFIX
            cl.target = src
            cl.subtarget = sb

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
                else:
                    pb.location = (0.0, 0.0, 0.0)
                pb.keyframe_insert("location", frame=f)
            pb.rotation_quaternion = rot
            pb.keyframe_insert("rotation_quaternion", frame=f)

    for pb in tgt.pose.bones:
        clear_constraints(pb)

    bpy.ops.object.mode_set(mode="OBJECT")
    # remove temp Mixamo + BVH
    for o in list(bpy.data.objects):
        if o != tgt and o.type == "ARMATURE":
            bpy.data.objects.remove(o, do_unlink=True)

    tgt.animation_data.action = act
    assign_slot(tgt)
    bpy.context.scene.frame_start = f0
    bpy.context.scene.frame_end = f1
    bpy.context.scene.frame_set(f0)
    log(f"action={act.name}")
    return f0, f1


def main() -> None:
    report: dict = {"ok": False, "steps": []}

    if not SMPLX_BLEND.is_file():
        raise SystemExit(f"missing {SMPLX_BLEND}")
    bpy.ops.wm.open_mainfile(filepath=str(SMPLX_BLEND))
    enable_keemap()

    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        raise SystemExit("no SMPL-X_Armature")

    # Remove leftover armatures from prior runs
    for o in list(bpy.data.objects):
        if o.type == "ARMATURE" and o.name != "SMPL-X_Armature":
            bpy.data.objects.remove(o, do_unlink=True)

    # --- 1 BVH ---
    if not BVH.is_file():
        raise SystemExit(f"missing {BVH}")
    before = set(bpy.data.objects.keys())
    bpy.ops.import_anim.bvh(
        filepath=str(BVH),
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
    src_bvh = next(
        bpy.data.objects[k]
        for k in bpy.data.objects.keys()
        if k not in before and bpy.data.objects[k].type == "ARMATURE"
    )
    assign_slot(src_bvh)
    fr = src_bvh.animation_data.action.frame_range
    f0, f1 = int(fr[0]), int(fr[1])
    n_frames = f1 - f0 + 1
    log(f"STEP1 BVH={src_bvh.name} frames={f0}-{f1}")
    report["steps"].append({"step": 1, "ok": True})

    # --- 2 Mixamo (prefer walk.fbx rest — matches catalog) ---
    mixamo_path = MIXAMO_ALT if MIXAMO_ALT.is_file() else MIXAMO_FBX
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(
        filepath=str(mixamo_path),
        ignore_leaf_bones=True,
        automatic_bone_orientation=False,
    )
    mix = next(
        bpy.data.objects[k]
        for k in bpy.data.objects.keys()
        if k not in before and bpy.data.objects[k].type == "ARMATURE"
    )
    # drop meshes clutter
    for o in list(bpy.data.objects):
        if o.type == "MESH" and o.parent == mix:
            bpy.data.objects.remove(o, do_unlink=True)
    reset_pose(mix)
    log(f"STEP2 Mixamo={mix.name} from {mixamo_path.name} (rest/T-pose)")
    report["steps"].append({"step": 2, "mixamo_fbx": mixamo_path.name})

    # --- 3 KeeMap ---
    map_path = MAP_FILE if MAP_FILE.is_file() else MAP_FALLBACK
    scene = bpy.context.scene
    KeeMap = scene.keemap_settings
    KeeMap.bone_mapping_file = str(map_path)
    bpy.ops.wm.keemap_read_file()
    KeeMap.source_rig_name = src_bvh.name
    KeeMap.destination_rig_name = mix.name
    KeeMap.start_frame_to_apply = f0
    KeeMap.number_of_frames_to_apply = n_frames
    KeeMap.keyframe_every_n_frames = 1
    KeeMap.bone_rotation_mode = "QUATERNION"

    bone_list = scene.keemap_bone_mapping_list
    valid = 0
    for item in bone_list:
        ok = (
            item.SourceBoneName in src_bvh.pose.bones
            and item.DestinationBoneName in mix.pose.bones
        )
        if not ok:
            item.set_bone_rotation = False
            item.set_bone_position = False
            item.keyframe_this_bone = False
        else:
            valid += 1
            # identity first, then auto-calc for THIS Mixamo (official json is for their char)
            item.CorrectionFactor = (0.0, 0.0, 0.0)
            item.QuatCorrectionFactor = (1.0, 0.0, 0.0, 0.0)
    log(f"STEP3 KeeMap pairs={valid} map={map_path.name}")

    bvh_act = src_bvh.animation_data.action
    if src_bvh.animation_data:
        src_bvh.animation_data.action = None
    reset_pose(mix)
    bpy.ops.object.select_all(action="DESELECT")
    src_bvh.select_set(True)
    mix.select_set(True)
    bpy.context.view_layer.objects.active = mix
    bpy.ops.object.mode_set(mode="POSE")
    log("STEP3 calc corrections for our Mixamo…")
    bpy.ops.wm.calc_correct_all_bones()
    if src_bvh.animation_data:
        src_bvh.animation_data.action = bvh_act
        assign_slot(src_bvh)

    KeeMap.bone_rotation_mode = "QUATERNION"
    log("STEP3 transfer…")
    bpy.ops.wm.perform_animation_transfer()
    bpy.ops.object.mode_set(mode="OBJECT")
    assign_slot(mix)

    m1 = check_arms(mix, True, f0)
    m2 = check_arms(mix, True, min(f0 + 30, f1))
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    q0 = mix.pose.bones["mixamorig:LeftArm"].rotation_quaternion.copy()
    bpy.context.scene.frame_set(min(f0 + 30, f1))
    bpy.context.view_layer.update()
    q1 = mix.pose.bones["mixamorig:LeftArm"].rotation_quaternion.copy()
    d_arm = sum(abs(a - b) for a, b in zip(q0, q1))
    log(f"STEP3 Mixamo LeftArm delta={d_arm:.4f}")
    report["steps"].append(
        {"step": 3, "ok": d_arm > 0.05, "arm_delta": round(d_arm, 4), "checks": [m1, m2]}
    )
    if d_arm < 0.05:
        raise SystemExit("STEP3 FAIL: Mixamo arms frozen")

    # remove BVH (done)
    bpy.data.objects.remove(src_bvh, do_unlink=True)

    # --- 4 Mixamo → SMPL-X in place ---
    log("STEP4 Mixamo→SMPL-X (same scene, no FBX)…")
    f0b, f1b = retarget_mixamo_to_smplx_inplace(mix, tgt, ACTION)

    # If legs fold upward (common after MoMask→Mixamo axis mismatch), flip hip chain 180° local X
    bpy.context.scene.frame_set(f0b)
    bpy.context.view_layer.update()
    hip = world_head(tgt, "left_hip")
    knee = world_head(tgt, "left_knee")
    if knee.z > hip.z + 0.05:
        log("STEP4 legs inverted on SMPL-X — apply 180° X on leg bones…")
        from mathutils import Quaternion

        flip = Quaternion((0.0, 1.0, 0.0, 0.0))  # 180° around X
        leg_bones = [
            "left_hip",
            "left_knee",
            "left_ankle",
            "left_foot",
            "right_hip",
            "right_knee",
            "right_ankle",
            "right_foot",
        ]
        act = tgt.animation_data.action
        assign_slot(tgt)
        for f in range(f0b, f1b + 1):
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()
            for bn in leg_bones:
                pb = tgt.pose.bones.get(bn)
                if not pb:
                    continue
                pb.rotation_mode = "QUATERNION"
                pb.rotation_quaternion = (flip @ pb.rotation_quaternion).normalized()
                pb.keyframe_insert("rotation_quaternion", frame=f)
        bpy.context.view_layer.update()
        hip = world_head(tgt, "left_hip")
        knee = world_head(tgt, "left_knee")
        log(f"STEP4 after leg flip hip.z={hip.z:.3f} knee.z={knee.z:.3f}")

    s1 = check_arms(tgt, False, f0b)
    s2 = check_arms(tgt, False, min(f0b + 30, f1b))
    bpy.context.scene.frame_set(f0b)
    bpy.context.view_layer.update()
    hip = world_head(tgt, "left_hip")
    knee = world_head(tgt, "left_knee")
    ank = world_head(tgt, "left_ankle")
    legs = knee.z < hip.z and ank.z < knee.z + 0.05
    bpy.context.scene.frame_set(f0b)
    bpy.context.view_layer.update()
    q0 = tgt.pose.bones["left_shoulder"].rotation_quaternion.copy()
    bpy.context.scene.frame_set(min(f0b + 30, f1b))
    bpy.context.view_layer.update()
    q1 = tgt.pose.bones["left_shoulder"].rotation_quaternion.copy()
    d_sh = sum(abs(a - b) for a, b in zip(q0, q1))
    log(f"STEP4 left_shoulder delta={d_sh:.4f} legs_down={legs}")

    smplx_ok = d_sh > 0.05 and legs and (s1["ok"] or s2["ok"] or s2["ab_x"] < 0.4)
    report["steps"].append(
        {
            "step": 4,
            "ok": smplx_ok,
            "shoulder_delta": round(d_sh, 4),
            "legs_down": bool(legs),
            "checks": [s1, s2],
        }
    )

    # save
    out = OUT_BLEND
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
    except RuntimeError:
        out = OUT_BLEND.with_name(OUT_BLEND.stem + "_v3" + OUT_BLEND.suffix)
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
    log(f"saved {out}")

    report["ok"] = all(s.get("ok") for s in report["steps"] if "ok" in s)
    report["open"] = {"blend": str(out), "action": ACTION}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"REPORT {REPORT}")
    log(f"DONE ok={report['ok']}")
    if not report["ok"]:
        raise SystemExit("PIPELINE checks failed — see report")


if __name__ == "__main__":
    main()
