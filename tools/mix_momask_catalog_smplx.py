"""
Hybrid MoMask → SMPL-X bake: mix joints + correct-motion rotations.

Why pure swing-aim / pure rest-relative look wrong:
  - Swing-aim from BVH joints loses roll and "animation language"
  - Rest-relative BVH rotations twist SMPL-X skin (spine/elbow candy-wrap)
  - Catalog Mixamo→SMPL-X actions (walk, idle, …) already look correct

Hybrid (this script):
  1) Sample catalog Action locals each frame (correct body rotations + fingers)
  2) Soft-pull limb/spine bones toward MoMask joint directions (silhouette)
  3) Root travel from MoMask Hips (absolute path)
  4) Optional rest-relative MoMask for pelvis yaw / facing blend

Result = correct-looking SMPL-X mesh + MoMask motion intent.

Usage:
  blender.exe whole_body_retargeted.blend --python tools/mix_momask_catalog_smplx.py -- ^
    --bvh "lmm train/momask_walk.bvh" ^
    --ref_action walk ^
    --action momask_mixed ^
    --out body_motion/_momask_ours.blend
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

# (bvh_parent, bvh_child, smplx_bone, smplx_child, pull_strength)
# pull_strength: 0 = keep catalog rotation only, 1 = full MoMask joint aim
# Spine/neck/head stay pure catalog (walk locals ~3–9° — any MoMask pull
# candy-wraps the mesh). Legs follow MoMask gait; arms blend.
AIM_CHAIN = [
    # legs — MoMask joint silhouette / gait
    ("LeftUpLeg", "LeftLeg", "left_hip", "left_knee", 0.90),
    ("LeftLeg", "LeftFoot", "left_knee", "left_ankle", 0.92),
    ("LeftFoot", "LeftToe", "left_ankle", "left_foot", 0.80),
    ("RightUpLeg", "RightLeg", "right_hip", "right_knee", 0.90),
    ("RightLeg", "RightFoot", "right_knee", "right_ankle", 0.92),
    ("RightFoot", "RightToe", "right_ankle", "right_foot", 0.80),
    # arms — catalog shoulder language + MoMask swing
    ("LeftShoulder", "LeftArm", "left_collar", "left_shoulder", 0.40),
    ("LeftArm", "LeftForeArm", "left_shoulder", "left_elbow", 0.70),
    ("LeftForeArm", "LeftHand", "left_elbow", "left_wrist", 0.65),
    ("RightShoulder", "RightArm", "right_collar", "right_shoulder", 0.40),
    ("RightArm", "RightForeArm", "right_shoulder", "right_elbow", 0.70),
    ("RightForeArm", "RightHand", "right_elbow", "right_wrist", 0.65),
]

# Bones whose final local rotation is forced back to catalog after limb pulls
# (keeps backbone/shoulders from collapsing when legs move)
CATALOG_LOCK = (
    "spine1",
    "spine2",
    "spine3",
    "neck",
    "head",
)

BODY_KEY = [
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

FINGER_PREFIXES = (
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
)


def log(msg: str) -> None:
    print(f"[mix] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", default=str(ROOT / "lmm train" / "momask_walk.bvh"))
    p.add_argument("--calib", default=str(ROOT / "body_motion" / "source_to_smplx_calib.json"))
    p.add_argument("--ref_action", default="walk", help="catalog Action with correct rotations")
    p.add_argument("--action", default="momask_mixed")
    p.add_argument("--out", default=str(ROOT / "body_motion" / "_momask_ours.blend"))
    p.add_argument(
        "--pull",
        type=float,
        default=1.0,
        help="global scale on all MoMask joint pulls (0=catalog only, 1=default mix)",
    )
    p.add_argument(
        "--pelvis_yaw",
        type=float,
        default=0.65,
        help="how much MoMask hips facing blends into pelvis rotation",
    )
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


def finger_bones(arm) -> list[str]:
    out = []
    for pb in arm.pose.bones:
        if any(pb.name.startswith(p) for p in FINGER_PREFIXES):
            out.append(pb.name)
    return out


def sample_catalog_locals(arm, action, frame: int, names: list[str]) -> dict[str, Quaternion]:
    """Evaluate catalog action at frame; return local quats for given bones."""
    if not arm.animation_data:
        arm.animation_data_create()
    prev = arm.animation_data.action
    arm.animation_data.action = action
    assign_slot(arm)
    scene = bpy.context.scene
    scene.frame_set(int(frame))
    bpy.context.view_layer.update()
    out = {}
    for n in names:
        if n in arm.pose.bones:
            pb = arm.pose.bones[n]
            pb.rotation_mode = "QUATERNION"
            out[n] = pb.rotation_quaternion.copy()
    arm.animation_data.action = prev
    assign_slot(arm)
    return out


def apply_world_rotation_keep_head(arm, bone_name: str, R_world_4x4: Matrix) -> None:
    pb = arm.pose.bones[bone_name]
    bpy.context.view_layer.update()
    head = arm.matrix_world @ pb.head
    mat_w = R_world_4x4.copy()
    mat_w.translation = head
    mat_pose = arm.matrix_world.inverted() @ mat_w
    bone = pb.bone
    if pb.parent:
        pre = pb.parent.matrix @ pb.parent.bone.matrix_local.inverted() @ bone.matrix_local
        basis = pre.inverted() @ mat_pose
    else:
        basis = bone.matrix_local.inverted() @ mat_pose
    _loc, rot, _ = basis.decompose()
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = rot
    if bone_name != "pelvis":
        pb.location = Vector((0, 0, 0))
    bpy.context.view_layer.update()


def soft_swing_aim(
    arm, bone_name: str, child_name: str, desired_child_w: Vector, strength: float
) -> None:
    """Partially rotate bone so child aims toward desired world point (keeps most current roll)."""
    if strength <= 1e-6:
        return
    strength = max(0.0, min(1.0, strength))
    pb = arm.pose.bones[bone_name]
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
    q_full = v0.rotation_difference(v1)
    q = Quaternion((1, 0, 0, 0)).slerp(q_full, strength)
    M = arm.matrix_world @ pb.matrix
    head = M.to_translation()
    M_new = Matrix.Translation(head) @ q.to_matrix().to_4x4() @ Matrix.Translation(-head) @ M
    mat_pose = arm.matrix_world.inverted() @ M_new
    bone = pb.bone
    saved_loc = pb.location.copy() if bone_name == "pelvis" else Vector((0, 0, 0))
    if pb.parent:
        pre = pb.parent.matrix @ pb.parent.bone.matrix_local.inverted() @ bone.matrix_local
        basis = pre.inverted() @ mat_pose
    else:
        basis = bone.matrix_local.inverted() @ mat_pose
    _loc, rot, _ = basis.decompose()
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = rot
    pb.location = saved_loc
    bpy.context.view_layer.update()


def remove_twist_y(q: Quaternion) -> Quaternion:
    if q.w < 0:
        q = -q
    tw = Quaternion((q.w, 0.0, q.y, 0.0))
    if tw.magnitude < 1e-8:
        return q
    tw.normalize()
    return (q @ tw.inverted()).normalized()


def extract_twist_y(q: Quaternion) -> Quaternion:
    if q.w < 0:
        q = -q
    tw = Quaternion((q.w, 0.0, q.y, 0.0))
    if tw.magnitude < 1e-8:
        return Quaternion((1, 0, 0, 0))
    return tw.normalized()


def set_pelvis_world_loc(arm, hips_w: Vector) -> None:
    pb = arm.pose.bones["pelvis"]
    mat_w = arm.convert_space(
        pose_bone=pb, matrix=pb.matrix, from_space="POSE", to_space="WORLD"
    ).copy()
    mat_w.translation = hips_w
    mat_pose = arm.matrix_world.inverted() @ mat_w
    basis = pb.bone.matrix_local.inverted() @ mat_pose
    loc, rot, _ = basis.decompose()
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = rot
    pb.location = loc
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

    ref_act = bpy.data.actions.get(args.ref_action)
    if not ref_act:
        raise SystemExit(
            f"ref_action '{args.ref_action}' not in blend. "
            f"Available: {[a.name for a in bpy.data.actions if not a.name.startswith('Armature|')]}"
        )

    # Clear foreign armatures
    for o in list(bpy.data.objects):
        if o.type == "ARMATURE" and o.name != "SMPL-X_Armature":
            bpy.data.objects.remove(o, do_unlink=True)

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

    ref_fr = ref_act.frame_range
    rf0, rf1 = int(ref_fr[0]), int(ref_fr[1])
    ref_len = max(1, rf1 - rf0)
    fingers = finger_bones(tgt)
    sample_names = BODY_KEY + fingers

    log(
        f"scale={scale:.4f} momask={f0}-{f1} ref={args.ref_action} {rf0}-{rf1} "
        f"pull={args.pull:.2f} fingers={len(fingers)}"
    )

    # Rest lengths for aiming (keep SMPL-X proportions)
    tlen = {}
    for _sa, _sb, ta, tb, _ in AIM_CHAIN:
        tlen[(ta, tb)] = rest_len(tgt, ta, tb)

    if not tgt.animation_data:
        tgt.animation_data_create()
    old = bpy.data.actions.get(args.action)
    if old:
        if tgt.animation_data.action == old:
            tgt.animation_data.action = None
        bpy.data.actions.remove(old)
    act = bpy.data.actions.new(args.action)
    act.use_fake_user = True

    # Offset matrices for optional pelvis rest-relative facing
    calib = {}
    if calib_path.is_file():
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
    rest_src = calib.get("rest_src_world_quat", {})
    rest_tgt = calib.get("rest_tgt_world_quat", {})
    offset_pelvis = None
    if "Hips" in rest_src and "pelvis" in rest_tgt:
        R_s = Quaternion(rest_src["Hips"]).to_matrix().to_4x4()
        R_t = Quaternion(rest_tgt["pelvis"]).to_matrix().to_4x4()
        offset_pelvis = R_s.inverted() @ R_t

    for f in range(f0, f1 + 1):
        # Phase-wrap catalog walk to MoMask timeline
        phase = rf0 + ((f - f0) % ref_len)
        catalog = sample_catalog_locals(tgt, ref_act, phase, sample_names)

        # MoMask joints this frame
        scene.frame_set(f)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        se = src.evaluated_get(dg)
        spos = {
            bn: src.matrix_world @ se.pose.bones[bn].head
            for bn in BVH_TO_SMPLX
            if bn in se.pose.bones
        }

        # Start from catalog pose (correct rotations + fingers)
        tgt.animation_data.action = act
        assign_slot(tgt)
        for pb in tgt.pose.bones:
            pb.rotation_mode = "QUATERNION"
            if pb.name in catalog:
                pb.rotation_quaternion = catalog[pb.name].copy()
            else:
                pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
            if pb.name != "pelvis":
                pb.location = Vector((0, 0, 0))
        bpy.context.view_layer.update()

        # Soft-pull limbs toward MoMask joint directions (catalog roll preserved)
        for sa, sb, ta, tb, strength in AIM_CHAIN:
            if sa not in spos or sb not in spos:
                continue
            d = spos[sb] - spos[sa]
            if d.length < 1e-8:
                continue
            d.normalize()
            L = tlen.get((ta, tb), rest_len(tgt, ta, tb))
            p = wh(tgt, ta)
            desired = p + d * L
            soft_swing_aim(tgt, ta, tb, desired, strength * args.pull)

        # Re-lock spine/neck/head to catalog (limb pulls must not warp backbone)
        for bn in CATALOG_LOCK:
            if bn in catalog:
                tgt.pose.bones[bn].rotation_mode = "QUATERNION"
                tgt.pose.bones[bn].rotation_quaternion = catalog[bn].copy()
        bpy.context.view_layer.update()

        # Pelvis: location from MoMask; mild facing blend with MoMask hips
        hips_w = spos["Hips"]
        if offset_pelvis is not None and args.pelvis_yaw > 1e-6 and "Hips" in se.pose.bones:
            R_src = (src.matrix_world @ se.pose.bones["Hips"].matrix).to_3x3().to_4x4()
            R_tgt_w = R_src @ offset_pelvis
            pb = tgt.pose.bones["pelvis"]
            cur_w = (tgt.matrix_world @ pb.matrix).to_3x3().to_4x4()
            q_cur = cur_w.to_quaternion()
            q_mom = R_tgt_w.to_quaternion()
            q_mix = q_cur.slerp(q_mom, args.pelvis_yaw)
            apply_world_rotation_keep_head(tgt, "pelvis", q_mix.to_matrix().to_4x4())

        set_pelvis_world_loc(tgt, hips_w)

        # Elbow: MoMask swing + catalog Y-twist (skin-safe roll)
        for bn in ("left_elbow", "right_elbow"):
            if bn not in catalog:
                continue
            pb = tgt.pose.bones[bn]
            q = pb.rotation_quaternion.copy()
            swing = remove_twist_y(q)
            tw = extract_twist_y(catalog[bn])
            tw_mix = Quaternion((1, 0, 0, 0)).slerp(tw, 0.85)
            pb.rotation_quaternion = (swing @ tw_mix).normalized()

        # Fingers pure catalog
        for bn in fingers:
            if bn in catalog:
                tgt.pose.bones[bn].rotation_mode = "QUATERNION"
                tgt.pose.bones[bn].rotation_quaternion = catalog[bn].copy()

        bpy.context.view_layer.update()

        # Keyframe
        for bn in BODY_KEY + fingers:
            if bn not in tgt.pose.bones:
                continue
            pb = tgt.pose.bones[bn]
            pb.rotation_mode = "QUATERNION"
            pb.keyframe_insert("rotation_quaternion", frame=f)
            if bn == "pelvis":
                pb.keyframe_insert("location", frame=f)

        if f == f0 or f % 30 == 0 or f == f1:
            pel = wh(tgt, "pelvis")
            log(
                f"f{f} phase={phase}: pelvis=({pel.x:.2f},{pel.y:.2f},{pel.z:.2f}) "
                f"spine2={[round(x, 3) for x in tgt.pose.bones['spine2'].rotation_quaternion]} "
                f"L_sh_z={wh(tgt, 'left_shoulder').z:.3f} "
                f"drop={wh(tgt, 'left_shoulder').z - wh(tgt, 'left_wrist').z:.3f} "
                f"legs_down={wh(tgt, 'left_knee').z < wh(tgt, 'left_hip').z}"
            )

    bpy.data.objects.remove(src, do_unlink=True)
    tgt.animation_data.action = act
    assign_slot(tgt)

    # Verify
    checks = []
    for f in (f0, min(f0 + 30, f1), min(f0 + 60, f1)):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        s2 = list(tgt.pose.bones["spine2"].rotation_quaternion)
        checks.append(
            {
                "f": f,
                "spine2": [round(x, 3) for x in s2],
                "spine2_deg": round(
                    math.degrees(2 * math.acos(min(1.0, abs(s2[0])))), 2
                ),
                "L_elb_w": round(tgt.pose.bones["left_elbow"].rotation_quaternion.w, 3),
                "L_sh_z": round(wh(tgt, "left_shoulder").z, 3),
                "legs_down": wh(tgt, "left_knee").z < wh(tgt, "left_hip").z,
                "hand_drop": round(
                    wh(tgt, "left_shoulder").z - wh(tgt, "left_wrist").z, 3
                ),
                "pelvis": [round(x, 3) for x in wh(tgt, "pelvis")],
                "has_fingers": bool(
                    fingers
                    and abs(tgt.pose.bones[fingers[0]].rotation_quaternion.w - 1.0)
                    > 0.01
                ),
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

    ok = (
        travel > 0.5
        and all(c["legs_down"] for c in checks)
        and all(c["L_elb_w"] > 0.7 for c in checks)
    )
    report = {
        "ok": ok,
        "method": "mix_catalog_rots_momask_joints",
        "ref_action": args.ref_action,
        "action": args.action,
        "travel": round(travel, 3),
        "pull": args.pull,
        "pelvis_yaw": args.pelvis_yaw,
        "checks": checks,
        "saved": str(out),
        "policy": {
            "rotations": f"catalog '{args.ref_action}' phase-wrapped (spine/neck/head/fingers locked)",
            "joints": "MoMask BVH soft swing-aim on legs+arms only",
            "root": "MoMask Hips absolute + mild facing blend",
            "fingers": f"pure catalog '{args.ref_action}'",
            "elbows": "MoMask swing + 85% catalog Y-twist",
        },
    }
    rep = ROOT / "body_motion" / "mix_momask_catalog_report.json"
    rep.write_text(json.dumps(report, indent=2), encoding="utf-8")

    scene.frame_start = f0
    scene.frame_end = f1
    scene.frame_set(f0)
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
    except RuntimeError:
        alt = out.with_name(out.stem + "_v2" + out.suffix)
        bpy.ops.wm.save_as_mainfile(filepath=str(alt))
        report["saved"] = str(alt)
        rep.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log(f"report {rep}")
    log(f"DONE ok={ok} travel={travel:.3f} action={args.action} saved={report['saved']}")
    if not ok:
        raise SystemExit("mix bake failed checks")


if __name__ == "__main__":
    main()
