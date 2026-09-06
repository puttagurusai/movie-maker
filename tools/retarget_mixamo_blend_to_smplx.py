"""
Mixamo armature (already KeeMap'd) → SMPL-X using FINAL_BONE_MAP.

Same method as catalog clips (retarget_final_one helper-bone COPY_ROTATION),
but sources a .blend armature instead of FBX (FBX export was stripping Head
and collapsing arm quats to ~I).

Usage:
  blender.exe whole_body_retargeted.blend --background \\
    --python tools/retarget_mixamo_blend_to_smplx.py -- \\
    --src_blend body_motion/_momask_mixamo_only.blend \\
    --action momask_via_mixamo \\
    --out body_motion/_momask_ours.blend \\
    --root location
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "body_motion" / "FINAL_BONE_MAP.json"
RETARGET_ID = "_RSL_FINAL"
HELPER_SUFFIX = "_RSL_H"


def log(msg: str) -> None:
    print(f"[mixamo→smplx] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = {}
    # simple parse
    def get(flag, default=None):
        if flag in argv:
            i = argv.index(flag)
            return argv[i + 1] if i + 1 < len(argv) else default
        return default

    return {
        "src_blend": get("--src_blend", str(ROOT / "body_motion" / "_momask_mixamo_only.blend")),
        "action": get("--action", "momask_via_mixamo"),
        "out": get("--out", str(ROOT / "body_motion" / "_momask_ours.blend")),
        "root": get("--root", "auto"),  # auto | location | inplace
    }


def load_map() -> dict[str, str]:
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    log(f"FINAL map {len(data['map'])} pairs status={data.get('status')}")
    return data["map"]


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


def assign_slot(obj) -> None:
    if not obj.animation_data or not obj.animation_data.action:
        return
    try:
        slots = obj.animation_data.action_suitable_slots
        if slots:
            obj.animation_data.action_slot = slots[0]
    except Exception:
        pass


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


def append_mixamo_armature(src_blend: Path):
    """Append first Mixamo-like armature + its action from blend."""
    before = set(bpy.data.objects.keys())
    before_acts = set(bpy.data.actions.keys())
    with bpy.data.libraries.load(str(src_blend), link=False) as (data_from, data_to):
        # Prefer armatures
        arms = [n for n in data_from.objects if True]
        data_to.objects = list(data_from.objects)
        data_to.actions = list(data_from.actions)
        data_to.armatures = list(data_from.armatures)

    # Link objects into scene
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
        names = [b.name for b in a.data.bones]
        if any("Hips" in n for n in names) and any("Arm" in n for n in names):
            src = a
            break
    if src is None and new_arms:
        src = new_arms[0]
    if src is None:
        raise SystemExit(f"no armature appended from {src_blend}")

    # Drop meshes from source character (keep armature only)
    for o in list(bpy.data.objects):
        if o.type == "MESH" and (o.parent == src or o.name not in before):
            # only remove newly appended meshes
            if o.name not in before or o.parent == src:
                try:
                    if o.name not in before:
                        bpy.data.objects.remove(o, do_unlink=True)
                except Exception:
                    pass

    # Ensure action
    if not src.animation_data or not src.animation_data.action:
        # pick newest action with mixamo-ish fcurves
        new_acts = [bpy.data.actions[k] for k in bpy.data.actions.keys() if k not in before_acts]
        if new_acts:
            if not src.animation_data:
                src.animation_data_create()
            src.animation_data.action = new_acts[0]
    assign_slot(src)
    log(
        f"appended src={src.name} bones={len(src.data.bones)} "
        f"action={src.animation_data.action.name if src.animation_data and src.animation_data.action else None} "
        f"Head={'mixamorig:Head' in src.data.bones}"
    )
    return src


def main():
    args = parse_args()
    src_blend = Path(args["src_blend"])
    if not src_blend.is_absolute():
        src_blend = ROOT / src_blend
    out = Path(args["out"])
    if not out.is_absolute():
        out = ROOT / out
    action_name = args["action"]
    root_mode_arg = args["root"]

    if not src_blend.is_file():
        raise SystemExit(f"missing src blend: {src_blend}")

    bone_map = load_map()
    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        raise SystemExit("no SMPL-X_Armature — open whole_body_retargeted.blend")

    # Clear foreign armatures
    for o in list(bpy.data.objects):
        if o.type == "ARMATURE" and o.name != "SMPL-X_Armature":
            bpy.data.objects.remove(o, do_unlink=True)

    old = bpy.data.actions.get(action_name)
    if old:
        if tgt.animation_data and tgt.animation_data.action == old:
            tgt.animation_data.action = None
        bpy.data.actions.remove(old)

    src = append_mixamo_armature(src_blend)
    if not src.animation_data or not src.animation_data.action:
        raise SystemExit("source has no action")

    # Scale source to target height (keep object scale)
    bpy.context.view_layer.update()
    hs, ht = height_hips_head(src, True), height_hips_head(tgt, False)
    if hs and ht and hs > 1e-6:
        s = ht / hs
        src.scale = (s * src.scale[0], s * src.scale[1], s * src.scale[2])
        bpy.context.view_layer.update()
        log(f"scale factor {s:.4f} → obj_scale={tuple(round(x, 5) for x in src.scale)}")

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
        raise SystemExit("too few bone pairs")

    # Helper bones (Rokoko-style) — same as retarget_final_one
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
    if f1 <= f0:
        f0, f1 = 1, 60
    log(f"frames {f0}..{f1}")

    hips_bn = find_src_bone(src, "mixamorig:Hips")
    root_mode = "inplace"
    if root_mode_arg in ("location", "inplace"):
        root_mode = root_mode_arg
    elif hips_bn:
        def _hips_world(frame: int):
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            se = src.evaluated_get(dg)
            return (src.matrix_world @ se.pose.bones[hips_bn].matrix).to_translation().copy()

        p_a = _hips_world(f0)
        p_b = _hips_world((f0 + f1) // 2)
        p_c = _hips_world(f1)
        hip_travel = max((p_b - p_a).length, (p_c - p_a).length, (p_c - p_b).length)
        if hip_travel > 0.15:
            root_mode = "location"
        log(f"source hips travel≈{hip_travel:.4f} → root_mode={root_mode}")

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

    # Source animates?
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
        raise SystemExit("SOURCE NOT ANIMATING — abort")
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
            loc, rot, sca = mat_local.decompose()
            if pb.name == "right_shoulder":
                max_rs = max(max_rs, abs(rot.x) + abs(rot.y) + abs(rot.z) + abs(1.0 - rot.w))
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

    changing = 0
    try:
        strip = act.layers[0].strips[0]
        cb = strip.channelbags[0]
        fcurves = list(cb.fcurves)
    except Exception:
        fcurves = list(getattr(act, "fcurves", []) or [])
    for fc in fcurves:
        try:
            if abs(fc.evaluate(f0) - fc.evaluate(mid)) > 0.02:
                changing += 1
        except Exception:
            pass
    log(f"channels_changing={changing} right_shoulder_delta={max_rs:.4f}")
    if changing < 5 and max_rs < 0.05:
        raise SystemExit("BAKE LOOKS REST — abort")

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(src, do_unlink=True)

    tgt.animation_data.action = act
    assign_slot(tgt)
    bpy.context.scene.frame_start = f0
    bpy.context.scene.frame_end = f1
    bpy.context.scene.frame_set(f0)

    # Verify
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
                "pelvis": [round(x, 3) for x in pel],
                "L_elb_w": round(tgt.pose.bones["left_elbow"].rotation_quaternion.w, 3),
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
    # hand_drop should be clearly positive for walk (arms hanging)
    hands_ok = all(c["hand_drop"] > 0.2 for c in checks)

    report = {
        "ok": ok and hands_ok,
        "ok_core": ok,
        "hands_ok": hands_ok,
        "method": "mixamo_blend_FINAL_BONE_MAP_helpers",
        "action": action_name,
        "src_blend": str(src_blend),
        "pairs": len(pairs),
        "frames": [f0, f1],
        "root_mode": root_mode,
        "travel": round(travel, 3),
        "channels_changing": changing,
        "right_shoulder_delta": max_rs,
        "checks": checks,
        "saved": str(out),
    }
    rep = ROOT / "body_motion" / "_proof_momask_via_mixamo_final_report.json"
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
        f"DONE ok={report['ok']} travel={travel:.3f} hands_ok={hands_ok} "
        f"action={action_name} pairs={len(pairs)} saved={report['saved']}"
    )
    if not report["ok"]:
        raise SystemExit("retarget checks failed")


if __name__ == "__main__":
    main()
