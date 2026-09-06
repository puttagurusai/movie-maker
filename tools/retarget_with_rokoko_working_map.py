"""
ARCHIVED / CLI-only — NOT the live MoMask product path.

Live path is: face_agents/momask_body_pipeline.retarget_bvh_to_smplx_action
→ tools/bvh_rokoko_direct_to_smplx.py (helper bake + mesh sole + foot lock).

This script: Mixamo (KeeMap output) → SMPL-X using Rokoko mapping overrides.
Kept for offline experiments only; do not wire into orchestrator.

Usage:
  blender.exe whole_body_retargeted.blend --background \\
    --python tools/retarget_with_rokoko_working_map.py -- \\
    --src_blend body_motion/_momask_keemap_exact.blend \\
    --rokoko_map "final hml3dto smpl.json" \\
    --action momask_via_mixamo \\
    --out body_motion/_momask_ours.blend \\
    --root location
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
FINAL_MAP_PATH = ROOT / "body_motion" / "FINAL_BONE_MAP.json"
HELPER_SUFFIX = "_RSL_H"
RETARGET_ID = "_RSL_WORKING"

# Rokoko body-part key → Mixamo bone name (mixamorig:*)
ROKOKO_PART_TO_MIXAMO = {
    "hips": "mixamorig:Hips",
    "spine": "mixamorig:Spine",
    "chest": "mixamorig:Spine1",
    "upperChest": "mixamorig:Spine2",
    "neck": "mixamorig:Neck",
    "head": "mixamorig:Head",
    "leftShoulder": "mixamorig:LeftShoulder",
    "rightShoulder": "mixamorig:RightShoulder",
    "leftUpperArm": "mixamorig:LeftArm",
    "rightUpperArm": "mixamorig:RightArm",
    "leftLowerArm": "mixamorig:LeftForeArm",
    "rightLowerArm": "mixamorig:RightForeArm",
    "leftHand": "mixamorig:LeftHand",
    "rightHand": "mixamorig:RightHand",
    "leftUpperLeg": "mixamorig:LeftUpLeg",
    "rightUpperLeg": "mixamorig:RightUpLeg",
    "leftLowerLeg": "mixamorig:LeftLeg",
    "rightLowerLeg": "mixamorig:RightLeg",
    "leftFoot": "mixamorig:LeftFoot",
    "rightFoot": "mixamorig:RightFoot",
    "leftToe": "mixamorig:LeftToeBase",
    "rightToe": "mixamorig:RightToeBase",
    # fingers
    "leftThumbProximal": "mixamorig:LeftHandThumb1",
    "leftThumbMedial": "mixamorig:LeftHandThumb2",
    "leftThumbDistal": "mixamorig:LeftHandThumb3",
    "leftIndexProximal": "mixamorig:LeftHandIndex1",
    "leftIndexMedial": "mixamorig:LeftHandIndex2",
    "leftIndexDistal": "mixamorig:LeftHandIndex3",
    "leftMiddleProximal": "mixamorig:LeftHandMiddle1",
    "leftMiddleMedial": "mixamorig:LeftHandMiddle2",
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
    "rightMiddleProximal": "mixamorig:RightHandMiddle1",
    "rightMiddleMedial": "mixamorig:RightHandMiddle2",
    "rightMiddleDistal": "mixamorig:RightHandMiddle3",
    "rightRingProximal": "mixamorig:RightHandRing1",
    "rightRingMedial": "mixamorig:RightHandRing2",
    "rightRingDistal": "mixamorig:RightHandRing3",
    "rightLittleProximal": "mixamorig:RightHandPinky1",
    "rightLittleMedial": "mixamorig:RightHandPinky2",
    "rightLittleDistal": "mixamorig:RightHandPinky3",
}


def log(msg: str) -> None:
    print(f"[rokoko_map] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src_blend",
        default=str(ROOT / "body_motion" / "_momask_keemap_exact.blend"),
        help="KeeMap Mixamo result blend",
    )
    p.add_argument(
        "--rokoko_map",
        default=str(ROOT / "final hml3dto smpl.json"),
        help="Canonical: final hml3dto smpl.json",
    )
    p.add_argument("--action", default="momask_via_mixamo")
    p.add_argument("--out", default=str(ROOT / "body_motion" / "_momask_ours.blend"))
    p.add_argument("--root", default="location", choices=("location", "inplace", "auto"))
    return p.parse_args(argv)


def build_bone_map(rokoko_path: Path) -> dict[str, str]:
    """FINAL body map + Rokoko working custom overrides."""
    final = json.loads(FINAL_MAP_PATH.read_text(encoding="utf-8"))["map"]
    bone_map = dict(final)
    overrides = []

    data = json.loads(rokoko_path.read_text(encoding="utf-8"))
    if not data.get("rokoko_custom_names"):
        # already mixamo→smplx flat map?
        if "map" in data:
            bone_map.update(data["map"])
            return bone_map
        # bones as mixamo→smplx direct?
        for k, v in data.get("bones", {}).items():
            if isinstance(v, list) and v:
                bone_map[k] = v[0]
            elif isinstance(v, str):
                bone_map[k] = v
        return bone_map

    for part, targets in data.get("bones", {}).items():
        if not targets:
            continue
        smplx = targets[0]
        mix = ROKOKO_PART_TO_MIXAMO.get(part)
        if not mix:
            log(f"  warn: unknown Rokoko part '{part}' → skip")
            continue
        old = bone_map.get(mix)
        bone_map[mix] = smplx
        if old != smplx:
            overrides.append(f"{part}: {mix} → {smplx} (was {old})")
        else:
            overrides.append(f"{part}: {mix} → {smplx} (same as FINAL)")

    log(f"working Rokoko overrides applied ({len(overrides)}):")
    for line in overrides:
        log(f"  {line}")
    log(f"total map pairs: {len(bone_map)}")
    return bone_map


def assign_slot(obj) -> None:
    if not obj.animation_data or not obj.animation_data.action:
        return
    try:
        slots = obj.animation_data.action_suitable_slots
        if slots:
            obj.animation_data.action_slot = slots[0]
    except Exception:
        pass


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


def clear_constraints(pb) -> None:
    while pb.constraints:
        pb.constraints.remove(pb.constraints[0])


def height_hips_head(arm, is_mixamo: bool) -> float | None:
    if is_mixamo:
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


def append_mixamo(src_blend: Path):
    before = set(bpy.data.objects.keys())
    before_acts = set(bpy.data.actions.keys())
    with bpy.data.libraries.load(str(src_blend), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
        data_to.actions = list(data_from.actions)
        data_to.armatures = list(data_from.armatures)

    for obj in data_to.objects:
        if obj is None:
            continue
        if obj.name not in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.link(obj)

    new_arms = [
        bpy.data.objects[k]
        for k in bpy.data.objects.keys()
        if k not in before and bpy.data.objects[k].type == "ARMATURE"
    ]
    src = None
    for a in new_arms:
        if any("Hips" in b.name for b in a.data.bones) and any(
            "Arm" in b.name for b in a.data.bones
        ):
            src = a
            break
    if src is None and new_arms:
        src = new_arms[0]
    if src is None:
        raise SystemExit(f"no Mixamo armature in {src_blend}")

    # Drop newly linked meshes only
    for o in list(bpy.data.objects):
        if o.name not in before and o.type == "MESH":
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass

    if not src.animation_data or not src.animation_data.action:
        new_acts = [
            bpy.data.actions[k] for k in bpy.data.actions.keys() if k not in before_acts
        ]
        if new_acts:
            if not src.animation_data:
                src.animation_data_create()
            src.animation_data.action = new_acts[0]
    assign_slot(src)
    log(
        f"src={src.name} bones={len(src.data.bones)} "
        f"action={src.animation_data.action.name if src.animation_data and src.animation_data.action else None} "
        f"Head={'mixamorig:Head' in src.data.bones}"
    )
    return src


def main():
    args = parse_args()
    src_blend = Path(args.src_blend)
    if not src_blend.is_absolute():
        src_blend = ROOT / src_blend
    rokoko_path = Path(args.rokoko_map)
    if not rokoko_path.is_absolute():
        rokoko_path = ROOT / rokoko_path
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    if not src_blend.is_file():
        raise SystemExit(f"missing KeeMap Mixamo blend: {src_blend}")
    if not rokoko_path.is_file():
        raise SystemExit(f"missing Rokoko map: {rokoko_path}")

    bone_map = build_bone_map(rokoko_path)
    # persist resolved map for inspection
    resolved_path = ROOT / "body_motion" / "HML3D_TO_SMPLX_resolved.json"
    resolved_path.write_text(
        json.dumps(
            {
                "source": "FINAL_BONE_MAP + final hml3dto smpl.json (Rokoko custom)",
                "rokoko_file": str(rokoko_path),
                "map": bone_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"wrote resolved map {resolved_path}")

    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        raise SystemExit("open whole_body_retargeted.blend (need SMPL-X_Armature)")

    for o in list(bpy.data.objects):
        if o.type == "ARMATURE" and o.name != "SMPL-X_Armature":
            bpy.data.objects.remove(o, do_unlink=True)

    action_name = args.action
    old = bpy.data.actions.get(action_name)
    if old:
        if tgt.animation_data and tgt.animation_data.action == old:
            tgt.animation_data.action = None
        bpy.data.actions.remove(old)

    src = append_mixamo(src_blend)
    if not src.animation_data or not src.animation_data.action:
        raise SystemExit("Mixamo source has no action")

    # Scale / align like catalog retarget
    bpy.context.view_layer.update()
    hs, ht = height_hips_head(src, True), height_hips_head(tgt, False)
    if hs and ht and hs > 1e-6:
        s = ht / hs
        src.scale = (s * src.scale[0], s * src.scale[1], s * src.scale[2])
        bpy.context.view_layer.update()
        log(f"scale factor {s:.4f}")

    hips = find_src_bone(src, "mixamorig:Hips")
    if hips and "pelvis" in tgt.data.bones:
        sh = src.matrix_world @ src.data.bones[hips].head_local
        th = tgt.matrix_world @ tgt.data.bones["pelvis"].head_local
        src.location += th - sh
        bpy.context.view_layer.update()

    pairs: list[tuple[str, str]] = []
    for mix_name, sm_name in bone_map.items():
        if sm_name not in tgt.pose.bones:
            log(f"skip missing target {sm_name}")
            continue
        sb = find_src_bone(src, mix_name)
        if not sb:
            log(f"skip missing source {mix_name}")
            continue
        pairs.append((sb, sm_name))
    log(f"mapped pairs: {len(pairs)}")
    if len(pairs) < 10:
        raise SystemExit("too few pairs")

    # Helper bones (Rokoko-style / catalog method)
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
        if parent is None:
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

    bpy.ops.object.select_all(action="DESELECT")
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="POSE")

    act_src = src.animation_data.action
    fr = act_src.frame_range
    f0, f1 = int(fr[0]), int(fr[1])
    log(f"frames {f0}..{f1}")

    hips_bn = find_src_bone(src, "mixamorig:Hips")
    root_mode = args.root
    if root_mode == "auto" and hips_bn:
        def _hips_w(frame: int):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            se = src.evaluated_get(dg)
            return (src.matrix_world @ se.pose.bones[hips_bn].matrix).to_translation().copy()

        pa, pb, pc = _hips_w(f0), _hips_w((f0 + f1) // 2), _hips_w(f1)
        travel = max((pb - pa).length, (pc - pa).length)
        root_mode = "location" if travel > 0.15 else "inplace"
        log(f"auto root_mode={root_mode} (hips travel≈{travel:.3f})")

    root_names = {"pelvis"} if root_mode == "location" else set()
    for sb, sm in pairs:
        pb = tgt.pose.bones[sm]
        clear_constraints(pb)
        c = pb.constraints.new("COPY_ROTATION")
        c.name = "Copy Rot" + RETARGET_ID
        c.target = src
        c.subtarget = sm + HELPER_SUFFIX
        c.mix_mode = "REPLACE"
        pb.rotation_mode = "QUATERNION"
        if sm in root_names:
            cl = pb.constraints.new("COPY_LOCATION")
            cl.name = "Copy Loc" + RETARGET_ID
            cl.target = src
            cl.subtarget = sb

    # Source moves?
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    check = find_src_bone(src, "mixamorig:RightArm") or pairs[0][0]
    dg = bpy.context.evaluated_depsgraph_get()
    q1 = src.evaluated_get(dg).pose.bones[check].matrix.to_quaternion().copy()
    mid = (f0 + f1) // 2
    bpy.context.scene.frame_set(mid)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    q2 = src.evaluated_get(dg).pose.bones[check].matrix.to_quaternion().copy()
    if q1 == q2:
        raise SystemExit("SOURCE NOT ANIMATING")
    log(f"source moves OK ({check})")

    if not tgt.animation_data:
        tgt.animation_data_create()
    act = bpy.data.actions.new(action_name)
    act.use_fake_user = True
    tgt.animation_data.action = act
    assign_slot(tgt)

    bones = [tgt.pose.bones[sm] for _, sm in pairs]
    max_rs = 0.0
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
            if pb.name == "right_shoulder":
                max_rs = max(
                    max_rs, abs(rot.x) + abs(rot.y) + abs(rot.z) + abs(1.0 - rot.w)
                )
            if pb.name == "pelvis":
                if root_mode == "location":
                    if pelvis0 is None:
                        pelvis0 = loc.copy()
                    pb.location = loc - pelvis0
                    pb.keyframe_insert("location", frame=f)
                else:
                    pb.location = (0.0, 0.0, 0.0)
                    pb.keyframe_insert("location", frame=f)
            pb.rotation_quaternion = rot
            pb.keyframe_insert("rotation_quaternion", frame=f)

    for pb in tgt.pose.bones:
        clear_constraints(pb)

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(src, do_unlink=True)

    tgt.animation_data.action = act
    assign_slot(tgt)
    bpy.context.scene.frame_start = f0
    bpy.context.scene.frame_end = f1
    bpy.context.scene.frame_set(f0)

    def wh(n):
        return tgt.matrix_world @ tgt.pose.bones[n].head

    checks = []
    for f in (f0, min(f0 + 30, f1), min(f0 + 60, f1)):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        hip, knee, ank = wh("left_hip"), wh("left_knee"), wh("left_ankle")
        sh, wr = wh("left_shoulder"), wh("left_wrist")
        pel, head = wh("pelvis"), wh("head")
        checks.append(
            {
                "f": f,
                "legs_down": bool(knee.z < hip.z and ank.z < knee.z + 0.05),
                "spine_up": bool(head.z > pel.z + 0.3),
                "hand_drop": round(sh.z - wr.z, 3),
                "L_sh_z": round(sh.z, 3),
                "spine2_w": round(tgt.pose.bones["spine2"].rotation_quaternion.w, 3),
                "collar_w": round(tgt.pose.bones["left_collar"].rotation_quaternion.w, 3),
                "L_elb_w": round(tgt.pose.bones["left_elbow"].rotation_quaternion.w, 3),
                "pelvis": [round(x, 3) for x in pel],
            }
        )
        log(f"verify {checks[-1]}")

    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    p0 = wh("pelvis").copy()
    bpy.context.scene.frame_set(f1)
    bpy.context.view_layer.update()
    p1 = wh("pelvis").copy()
    travel = (p1 - p0).length
    ok = travel > 0.5 and all(c["legs_down"] and c["spine_up"] for c in checks)

    report = {
        "ok": ok,
        "method": "KeeMap Mixamo → SMPL-X with working Rokoko map + helper bake",
        "src_blend": str(src_blend),
        "rokoko_map": str(rokoko_path),
        "resolved_map": str(resolved_path),
        "action": action_name,
        "pairs": len(pairs),
        "frames": [f0, f1],
        "root_mode": root_mode,
        "travel": round(travel, 3),
        "right_shoulder_delta": max_rs,
        "checks": checks,
        "saved": str(out),
    }
    rep = ROOT / "body_motion" / "_proof_momask_rokoko_working_report.json"
    rep.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
    except RuntimeError:
        alt = out.with_name(out.stem + "_v2" + out.suffix)
        bpy.ops.wm.save_as_mainfile(filepath=str(alt))
        report["saved"] = str(alt)
        rep.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log(f"report {rep}")
    log(
        f"DONE ok={ok} travel={travel:.3f} pairs={len(pairs)} "
        f"action={action_name} saved={report['saved']}"
    )
    if not ok:
        raise SystemExit("retarget checks failed")


if __name__ == "__main__":
    main()
