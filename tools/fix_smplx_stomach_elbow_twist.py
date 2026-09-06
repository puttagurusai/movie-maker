"""
Fix stomach + elbow mesh twists on MoMask→SMPL-X bake.

Rest-relative BVH matrices over-rotate spine/elbows → mesh twist. Full lock of
spine2/3 kills twist but makes the backbone stiff and drops shoulders.

Strategy (skin-safe, natural flex):
  - Scale source BVH to our height
  - Full root travel (pelvis follows Hips)
  - Swing-only aim for legs/arms (keeps SMPL-X rest ROLL)
  - Full spine chain with damp + max-angle caps (not hard lock to rest)
  - Undamped collars/shoulders so arm height matches source
  - Strip residual local-Y twist on elbows/hips only (not shoulders)

Usage:
  blender.exe whole_body_retargeted.blend --python tools/fix_smplx_stomach_elbow_twist.py -- ^
    --bvh "lmm train/momask_walk.bvh" --action momask_from_calib --out body_motion/_momask_ours.blend
"""
from __future__ import annotations

import argparse
import json
import math
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

KEY = [
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
    print(f"[fix_twist] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", default=str(ROOT / "lmm train" / "momask_walk.bvh"))
    p.add_argument("--calib", default=str(ROOT / "body_motion" / "source_to_smplx_calib.json"))
    p.add_argument("--action", default="momask_from_calib")
    p.add_argument("--out", default=str(ROOT / "body_motion" / "_momask_ours.blend"))
    return p.parse_args(argv)


def assign_slot(obj) -> None:
    if obj.animation_data and obj.animation_data.action:
        try:
            slots = obj.animation_data.action_suitable_slots
            if slots:
                obj.animation_data.action_slot = slots[0]
        except Exception:
            pass


def rest_len(arm, a: str, b: str) -> float:
    return (
        arm.matrix_world @ arm.data.bones[b].head_local
        - arm.matrix_world @ arm.data.bones[a].head_local
    ).length


def wh(arm, n: str) -> Vector:
    return arm.matrix_world @ arm.pose.bones[n].head


def remove_twist_y(q: Quaternion) -> Quaternion:
    """Remove twist around local Y (bone axis) — keeps swing only."""
    if q.w < 0:
        q = -q
    tw = Quaternion((q.w, 0.0, q.y, 0.0))
    if tw.magnitude < 1e-8:
        return q
    tw.normalize()
    return (q @ tw.inverted()).normalized()


def clamp_quat_angle(q: Quaternion, max_deg: float) -> Quaternion:
    """Limit rotation magnitude; keeps axis, caps angle (anti-twist / anti-overflex)."""
    q = q.normalized()
    if q.w < 0:
        q = -q
    angle = 2.0 * math.acos(min(1.0, max(-1.0, q.w)))
    max_rad = math.radians(max_deg)
    if angle <= max_rad or angle < 1e-8:
        return q
    axis = Vector((q.x, q.y, q.z))
    if axis.length < 1e-8:
        return Quaternion((1, 0, 0, 0))
    axis.normalize()
    return Quaternion(axis, max_rad).normalized()


def swing_aim(arm, bone_name: str, child_name: str, desired_child_w: Vector) -> None:
    pb = arm.pose.bones[bone_name]
    saved = pb.location.copy() if bone_name == "pelvis" else Vector((0, 0, 0))
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
    pb.location = saved if bone_name == "pelvis" else Vector((0, 0, 0))
    bpy.context.view_layer.update()

    head_w = arm.matrix_world @ pb.head
    child_w = arm.matrix_world @ arm.pose.bones[child_name].head
    v0 = child_w - head_w
    v1 = desired_child_w - head_w
    if v0.length < 1e-8 or v1.length < 1e-8:
        return
    v0.normalize()
    v1.normalize()
    if v0.dot(v1) > 0.999999:
        return
    q = v0.rotation_difference(v1)
    M = arm.matrix_world @ pb.matrix
    head = M.to_translation()
    M_new = Matrix.Translation(head) @ q.to_matrix().to_4x4() @ Matrix.Translation(-head) @ M
    mat_pose = arm.matrix_world.inverted() @ M_new
    bone = pb.bone
    if pb.parent:
        pre = pb.parent.matrix @ pb.parent.bone.matrix_local.inverted() @ bone.matrix_local
        basis = pre.inverted() @ mat_pose
    else:
        basis = bone.matrix_local.inverted() @ mat_pose
    _loc, rot, _ = basis.decompose()
    pb.rotation_quaternion = rot
    pb.location = saved if bone_name == "pelvis" else Vector((0, 0, 0))
    bpy.context.view_layer.update()


def main():
    args = parse_args()
    bvh = Path(args.bvh)
    if not bvh.is_absolute():
        bvh = ROOT / bvh
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    calib_path = Path(args.calib)
    if not calib_path.is_absolute():
        calib_path = ROOT / calib_path

    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        raise SystemExit("SMPL-X_Armature missing — open whole_body_retargeted.blend")

    for o in list(bpy.data.objects):
        if o.type == "ARMATURE" and o.name != "SMPL-X_Armature":
            bpy.data.objects.remove(o, do_unlink=True)

    # Ensure object mode with active target (BVH import needs valid context)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt

    scale = 1.1223
    if calib_path.is_file():
        scale = float(
            json.loads(calib_path.read_text(encoding="utf-8"))
            .get("height", {})
            .get("uniform_scale_on_source", scale)
        )

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
    src.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    scene = bpy.context.scene
    fr = src.animation_data.action.frame_range
    f0, f1 = int(fr[0]), int(fr[1])
    scene.frame_set(f0)
    bpy.context.view_layer.update()
    sh0 = src.matrix_world @ src.pose.bones["Hips"].head
    th0 = tgt.matrix_world @ tgt.data.bones["pelvis"].head_local
    src.location += th0 - sh0
    bpy.context.view_layer.update()
    log(f"scale={scale:.4f} frames={f0}-{f1}")

    tlen = {
        ("pelvis", "spine1"): rest_len(tgt, "pelvis", "spine1"),
        ("spine1", "spine2"): rest_len(tgt, "spine1", "spine2"),
        ("spine2", "spine3"): rest_len(tgt, "spine2", "spine3"),
        ("spine3", "neck"): rest_len(tgt, "spine3", "neck"),
        ("neck", "head"): rest_len(tgt, "neck", "head"),
        ("left_hip", "left_knee"): rest_len(tgt, "left_hip", "left_knee"),
        ("left_knee", "left_ankle"): rest_len(tgt, "left_knee", "left_ankle"),
        ("left_ankle", "left_foot"): rest_len(tgt, "left_ankle", "left_foot"),
        ("right_hip", "right_knee"): rest_len(tgt, "right_hip", "right_knee"),
        ("right_knee", "right_ankle"): rest_len(tgt, "right_knee", "right_ankle"),
        ("right_ankle", "right_foot"): rest_len(tgt, "right_ankle", "right_foot"),
        ("left_collar", "left_shoulder"): rest_len(tgt, "left_collar", "left_shoulder"),
        ("left_shoulder", "left_elbow"): rest_len(tgt, "left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"): rest_len(tgt, "left_elbow", "left_wrist"),
        ("right_collar", "right_shoulder"): rest_len(tgt, "right_collar", "right_shoulder"),
        ("right_shoulder", "right_elbow"): rest_len(tgt, "right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"): rest_len(tgt, "right_elbow", "right_wrist"),
    }

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

    def aim(spos, sa, sb, ta, tb, damp=1.0, max_deg=None):
        if sa not in spos or sb not in spos:
            return
        d = spos[sb] - spos[sa]
        if d.length < 1e-8:
            return
        d.normalize()
        if damp < 0.999:
            rd = (
                tgt.matrix_world @ tgt.data.bones[tb].head_local
                - tgt.matrix_world @ tgt.data.bones[ta].head_local
            )
            if rd.length > 1e-8:
                rd.normalize()
                blend = d * damp + rd * (1.0 - damp)
                if blend.length > 1e-8:
                    d = blend.normalized()
        L = tlen.get((ta, tb), rest_len(tgt, ta, tb))
        p = wh(tgt, ta)
        swing_aim(tgt, ta, tb, p + d * L)
        if max_deg is not None:
            pb = tgt.pose.bones[ta]
            pb.rotation_quaternion = clamp_quat_angle(
                pb.rotation_quaternion.copy(), max_deg
            )
            bpy.context.view_layer.update()

    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        se = src.evaluated_get(dg)
        spos = {
            bn: src.matrix_world @ se.pose.bones[bn].head
            for bn in BVH_TO_SMPLX
            if bn in se.pose.bones
        }

        for pb in tgt.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
            pb.location = Vector((0, 0, 0))
        bpy.context.view_layer.update()

        # Spine chain: natural flex with damp + angle caps (no hard lock → not stiff)
        # Caps keep stomach mesh from twisting while allowing bend/sway.
        aim(spos, "Hips", "Spine", "pelvis", "spine1", damp=0.80, max_deg=28.0)
        aim(spos, "Spine", "Spine1", "spine1", "spine2", damp=0.65, max_deg=22.0)
        aim(spos, "Spine1", "Spine2", "spine2", "spine3", damp=0.55, max_deg=18.0)
        aim(spos, "Spine2", "Neck", "spine3", "neck", damp=0.70, max_deg=25.0)
        aim(spos, "Neck", "Head", "neck", "head", damp=0.75, max_deg=30.0)

        # Strip residual bone-axis twist, then re-cap angles (twist strip can grow swing a bit)
        spine_caps = {"spine1": 28.0, "spine2": 22.0, "spine3": 18.0}
        for bn, cap in spine_caps.items():
            pb = tgt.pose.bones[bn]
            q = remove_twist_y(pb.rotation_quaternion.copy())
            pb.rotation_quaternion = clamp_quat_angle(q, cap)
        bpy.context.view_layer.update()

        # Legs
        aim(spos, "LeftUpLeg", "LeftLeg", "left_hip", "left_knee")
        aim(spos, "LeftLeg", "LeftFoot", "left_knee", "left_ankle")
        aim(spos, "LeftFoot", "LeftToe", "left_ankle", "left_foot")
        aim(spos, "RightUpLeg", "RightLeg", "right_hip", "right_knee")
        aim(spos, "RightLeg", "RightFoot", "right_knee", "right_ankle")
        aim(spos, "RightFoot", "RightToe", "right_ankle", "right_foot")

        # Arms: full collar/shoulder drive so shoulders don't sink
        aim(spos, "LeftShoulder", "LeftArm", "left_collar", "left_shoulder")
        aim(spos, "LeftArm", "LeftForeArm", "left_shoulder", "left_elbow")
        aim(spos, "LeftForeArm", "LeftHand", "left_elbow", "left_wrist")
        aim(spos, "RightShoulder", "RightArm", "right_collar", "right_shoulder")
        aim(spos, "RightArm", "RightForeArm", "right_shoulder", "right_elbow")
        aim(spos, "RightForeArm", "RightHand", "right_elbow", "right_wrist")

        # Root travel
        hips_w = spos["Hips"]
        pb = tgt.pose.bones["pelvis"]
        mat_w = tgt.convert_space(
            pose_bone=pb, matrix=pb.matrix, from_space="POSE", to_space="WORLD"
        ).copy()
        mat_w.translation = hips_w
        mat_pose = tgt.matrix_world.inverted() @ mat_w
        basis = pb.bone.matrix_local.inverted() @ mat_pose
        loc, rot, _ = basis.decompose()
        pb.rotation_quaternion = rot
        pb.location = loc
        bpy.context.view_layer.update()

        # Elbow/hip twist only — leave shoulders/collars full (lift restored)
        for bn in (
            "left_elbow",
            "right_elbow",
            "left_hip",
            "right_hip",
        ):
            pb = tgt.pose.bones[bn]
            pb.rotation_quaternion = remove_twist_y(pb.rotation_quaternion.copy())

        for bn in KEY:
            pb = tgt.pose.bones[bn]
            pb.rotation_mode = "QUATERNION"
            pb.keyframe_insert("rotation_quaternion", frame=f)
            if bn == "pelvis":
                pb.keyframe_insert("location", frame=f)

        if f == f0 or f % 30 == 0 or f == f1:
            pel = wh(tgt, "pelvis")
            log(
                f"f{f}: pelvis=({pel.x:.2f},{pel.y:.2f},{pel.z:.2f}) "
                f"spine1={[round(x,3) for x in tgt.pose.bones['spine1'].rotation_quaternion]} "
                f"spine2={[round(x,3) for x in tgt.pose.bones['spine2'].rotation_quaternion]} "
                f"spine3={[round(x,3) for x in tgt.pose.bones['spine3'].rotation_quaternion]} "
                f"L_sh_z={wh(tgt,'left_shoulder').z:.3f} "
                f"L_elb={[round(x,3) for x in tgt.pose.bones['left_elbow'].rotation_quaternion]} "
                f"drop={wh(tgt,'left_shoulder').z - wh(tgt,'left_wrist').z:.3f}"
            )

    bpy.data.objects.remove(src, do_unlink=True)

    # Final compare numbers
    checks = []
    for f in (f0, min(f0 + 30, f1), min(f0 + 60, f1)):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        s1 = list(tgt.pose.bones["spine1"].rotation_quaternion)
        s2 = list(tgt.pose.bones["spine2"].rotation_quaternion)
        s3 = list(tgt.pose.bones["spine3"].rotation_quaternion)
        checks.append(
            {
                "f": f,
                "spine1": [round(x, 3) for x in s1],
                "spine2": [round(x, 3) for x in s2],
                "spine3": [round(x, 3) for x in s3],
                "spine1_deg": round(
                    math.degrees(2 * math.acos(min(1.0, abs(s1[0])))), 2
                ),
                "spine2_deg": round(
                    math.degrees(2 * math.acos(min(1.0, abs(s2[0])))), 2
                ),
                "spine3_deg": round(
                    math.degrees(2 * math.acos(min(1.0, abs(s3[0])))), 2
                ),
                "L_elb": [round(x, 3) for x in tgt.pose.bones["left_elbow"].rotation_quaternion],
                "R_elb": [round(x, 3) for x in tgt.pose.bones["right_elbow"].rotation_quaternion],
                "L_sh_z": round(wh(tgt, "left_shoulder").z, 3),
                "L_collar_z": round(wh(tgt, "left_collar").z, 3),
                "legs_down": wh(tgt, "left_knee").z < wh(tgt, "left_hip").z,
                "hand_drop": round(
                    wh(tgt, "left_shoulder").z - wh(tgt, "left_wrist").z, 3
                ),
                "pelvis": [round(x, 3) for x in wh(tgt, "pelvis")],
            }
        )
        log(f"verify {checks[-1]}")

    scene.frame_set(f0)
    bpy.context.view_layer.update()
    p0 = wh(tgt, "pelvis").copy()
    scene.frame_set(f1)
    bpy.context.view_layer.update()
    p1 = wh(tgt, "pelvis").copy()
    travel = (p1 - p0).length

    # Spine should flex (not all locked ~I) but stay under caps
    spine_flex = any(c["spine1_deg"] > 2.0 or c["spine2_deg"] > 1.0 for c in checks)
    spine_capped = all(
        c["spine1_deg"] <= 30.0 and c["spine2_deg"] <= 24.0 and c["spine3_deg"] <= 22.0
        for c in checks
    )
    spine_ok = spine_flex and spine_capped
    # elbows should be mild like walk (w > 0.9 typically)
    elb_ok = all(c["L_elb"][0] > 0.85 and c["R_elb"][0] > 0.85 for c in checks)

    report = {
        "ok": travel > 0.5 and spine_ok and elb_ok and all(c["legs_down"] for c in checks),
        "travel": round(travel, 3),
        "spine_ok": spine_ok,
        "spine_flex": spine_flex,
        "spine_capped": spine_capped,
        "elbow_ok": elb_ok,
        "checks": checks,
        "action": args.action,
        "saved": str(out),
        "policy": {
            "spine1": "damp=0.80 max=28° + y-twist strip",
            "spine2": "damp=0.65 max=22° + y-twist strip",
            "spine3": "damp=0.55 max=18° + y-twist strip",
            "shoulders": "undamped full aim (no twist strip)",
            "elbows": "swing aim + y-twist strip",
        },
    }
    rep = ROOT / "body_motion" / "fix_twist_report.json"
    rep.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
    except RuntimeError:
        alt = out.with_name(out.stem + "_v2" + out.suffix)
        bpy.ops.wm.save_as_mainfile(filepath=str(alt))
        report["saved"] = str(alt)

    log(f"report {rep}")
    log(f"DONE ok={report['ok']} travel={travel:.3f} spine_ok={spine_ok} elbow_ok={elb_ok}")
    if not report["ok"]:
        raise SystemExit("fix incomplete")


if __name__ == "__main__":
    main()
