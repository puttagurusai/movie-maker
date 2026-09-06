"""
KeeMap Mixamo → SMPL-X using the *working Rokoko mapping* + catalog bake.

What catalog walk/wave used:
  - Rokoko mapping: final hml3dto smpl.json (canonical HML3D/Mixamo → SMPL-X)
  - Helper-bone COPY_ROTATION bake (same math as rsl.retarget_animation)

Blender 5's rsl.retarget_animation only keeps ~25-frame chunks (broken merge).
So we:
  1) import final HML3D→SMPL Rokoko scheme
  2) build_bone_list (Rokoko auto-detect + your overrides)
  3) merge FINAL body/finger pairs so full body is covered
  4) full-frame helper bake (catalog method) for all MoMask frames

Usage:
  blender.exe whole_body_retargeted.blend --background \\
    --python tools/retarget_keemap_mixamo_rsl.py -- \\
    --src_blend body_motion/_momask_keemap_exact.blend \\
    --rokoko_map "final hml3dto smpl.json" \\
    --action momask_via_mixamo \\
    --out body_motion/_momask_ours.blend
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
FINAL_MAP_PATH = ROOT / "body_motion" / "FINAL_BONE_MAP.json"
HELPER_SUFFIX = "_RSL_H"
RETARGET_ID = "_RSL_WORKING"


def log(msg: str) -> None:
    print(f"[rsl_retarget] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src_blend",
        default=str(ROOT / "body_motion" / "_momask_keemap_exact.blend"),
    )
    p.add_argument(
        "--rokoko_map",
        default=str(ROOT / "final hml3dto smpl.json"),
        help="Rokoko custom-names JSON (canonical: final hml3dto smpl.json)",
    )
    p.add_argument("--action", default="momask_via_mixamo")
    p.add_argument("--out", default=str(ROOT / "body_motion" / "_momask_ours.blend"))
    p.add_argument("--root", default="location", choices=("location", "inplace"))
    return p.parse_args(argv)


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


def ensure_object_mode(obj=None) -> None:
    """Background-safe: need an active object before mode_set."""
    if obj is None:
        obj = bpy.context.view_layer.objects.active
    if obj is None:
        # pick any armature in the scene
        for o in bpy.data.objects:
            if o.type == "ARMATURE":
                obj = o
                break
    if obj is None:
        return
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.hide_viewport = False
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass


def set_active(obj) -> None:
    ensure_object_mode(obj)
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def strip_prefix(name: str) -> str:
    return name.split(":")[-1]


def find_src_bone(arm, mixamo_full: str) -> str | None:
    if mixamo_full in arm.pose.bones:
        return mixamo_full
    short = strip_prefix(mixamo_full)
    for b in arm.pose.bones:
        if strip_prefix(b.name) == short or b.name == short:
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


def get_custom_schemes_manager():
    for name, mod in sys.modules.items():
        if name.endswith("custom_schemes_manager"):
            return mod
    import importlib.util

    csm_path = Path(
        r"C:\Users\putta.gurusaireddy\AppData\Roaming\Blender Foundation\Blender\5.1\scripts\addons\rokoko-studio-live-blender-master\core\custom_schemes_manager.py"
    )
    if not csm_path.is_file():
        raise SystemExit(f"missing {csm_path}")
    spec = importlib.util.spec_from_file_location("rsl_csm", csm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
        if any("mixamorig" in b.name or b.name.endswith("Hips") for b in a.data.bones):
            src = a
            break
    if src is None and new_arms:
        src = new_arms[0]
    if src is None:
        raise SystemExit(f"no Mixamo armature in {src_blend}")
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
        f"Mixamo src={src.name} bones={len(src.data.bones)} "
        f"action={src.animation_data.action.name if src.animation_data and src.animation_data.action else None}"
    )
    return src


def main():
    args = parse_args()
    src_blend = Path(args.src_blend)
    if not src_blend.is_absolute():
        src_blend = ROOT / src_blend
    rokoko_map = Path(args.rokoko_map)
    if not rokoko_map.is_absolute():
        rokoko_map = ROOT / rokoko_map
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        raise SystemExit("Need SMPL-X_Armature")

    for o in list(bpy.data.objects):
        if o.type == "ARMATURE" and o.name != "SMPL-X_Armature":
            bpy.data.objects.remove(o, do_unlink=True)

    old = bpy.data.actions.get(args.action)
    if old:
        if tgt.animation_data and tgt.animation_data.action == old:
            tgt.animation_data.action = None
        bpy.data.actions.remove(old)

    src = append_mixamo(src_blend)
    if not src.animation_data or not src.animation_data.action:
        raise SystemExit("Mixamo has no action")

    scene = bpy.context.scene
    scene.rsl_retargeting_armature_source = src
    scene.rsl_retargeting_armature_target = tgt
    scene.rsl_retargeting_auto_scaling = True
    scene.rsl_retargeting_use_pose = "REST"

    # --- Rokoko working map (same file as catalog) ---
    csm = get_custom_schemes_manager()
    log(f"import Rokoko scheme {rokoko_map.name}")
    csm.import_custom_list(str(rokoko_map.parent) + os.sep, rokoko_map.name)
    csm.save_to_file_and_update()

    # Background Blender: must have active object before any mode_set / RSL ops
    set_active(tgt)
    ensure_object_mode(tgt)
    try:
        bpy.ops.rsl.build_bone_list()
    except Exception as e:
        log(f"rsl.build_bone_list failed ({e}) — using FINAL map only")
    rsl_pairs: dict[str, str] = {}
    for item in scene.rsl_retargeting_bone_list:
        if item.bone_name_source and item.bone_name_target:
            rsl_pairs[item.bone_name_source] = item.bone_name_target
    log(f"Rokoko bone list mapped={len(rsl_pairs)}")
    for s, t in list(rsl_pairs.items())[:15]:
        log(f"  RSL {s} → {t}")

    # Merge FINAL map so fingers/feet/spine fill gaps (Rokoko list is partial)
    final_map = json.loads(FINAL_MAP_PATH.read_text(encoding="utf-8"))["map"]
    bone_map: dict[str, str] = dict(final_map)
    # Rokoko list wins (includes working custom overrides applied at detect time)
    for mix_bone, sm in rsl_pairs.items():
        # normalize key to mixamorig form if needed
        key = mix_bone if mix_bone.startswith("mixamorig:") else f"mixamorig:{mix_bone}"
        if mix_bone in src.pose.bones:
            bone_map[mix_bone] = sm
        elif key in src.pose.bones:
            bone_map[key] = sm
        else:
            bone_map[mix_bone] = sm

    # Explicit working-json arm override (leftUpperArm → left_shoulder)
    bone_map["mixamorig:LeftArm"] = "left_shoulder"
    bone_map["mixamorig:RightArm"] = "right_shoulder"
    bone_map["mixamorig:LeftToeBase"] = "left_foot"
    bone_map["mixamorig:RightToeBase"] = "right_foot"

    resolved = ROOT / "body_motion" / "HML3D_TO_SMPLX_resolved.json"
    resolved.write_text(
        json.dumps(
            {
                "source": "Rokoko build_bone_list + final hml3dto smpl.json + FINAL fill",
                "rokoko_map": str(rokoko_map),
                "rsl_pairs": rsl_pairs,
                "map": bone_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"resolved map → {resolved} ({len(bone_map)} entries)")

    # Scale / align like catalog
    bpy.context.view_layer.update()
    hs, ht = height_hips_head(src, True), height_hips_head(tgt, False)
    if hs and ht and hs > 1e-6:
        s = ht / hs
        src.scale = (s * src.scale[0], s * src.scale[1], s * src.scale[2])
        bpy.context.view_layer.update()
        log(f"scale factor {s:.4f}")

    hips = find_src_bone(src, "mixamorig:Hips")
    if hips:
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
    # de-dupe by target
    seen_tgt = set()
    uniq = []
    for sb, sm in pairs:
        if sm in seen_tgt:
            continue
        seen_tgt.add(sm)
        uniq.append((sb, sm))
    pairs = uniq
    log(f"bake pairs={len(pairs)}")
    if len(pairs) < 10:
        raise SystemExit("too few pairs")

    # Helper bones + bake (catalog / Rokoko method, full frame range)
    mw_src_inv = src.matrix_world.inverted()
    set_active(tgt)
    bpy.ops.object.mode_set(mode="EDIT")
    bone_transforms = {}
    for eb in tgt.data.edit_bones:
        head = mw_src_inv @ (tgt.matrix_world @ eb.head)
        tail = mw_src_inv @ (tgt.matrix_world @ eb.tail)
        bone_transforms[eb.name] = (head.copy(), tail.copy(), eb.roll)
    bpy.ops.object.mode_set(mode="OBJECT")

    set_active(src)
    bpy.ops.object.mode_set(mode="EDIT")
    for sb, sm in pairs:
        parent = src.data.edit_bones.get(sb)
        if parent is None:
            continue
        hname = sm + HELPER_SUFFIX
        if hname in src.data.edit_bones:
            src.data.edit_bones.remove(src.data.edit_bones[hname])
        if sm not in bone_transforms:
            continue
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

    set_active(tgt)
    bpy.ops.object.mode_set(mode="POSE")

    fr = src.animation_data.action.frame_range
    f0, f1 = int(fr[0]), int(fr[1])
    log(f"frames {f0}..{f1} root={args.root}")

    root_names = {"pelvis"} if args.root == "location" else set()
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

    if not tgt.animation_data:
        tgt.animation_data_create()
    act = bpy.data.actions.new(args.action)
    act.use_fake_user = True
    tgt.animation_data.action = act
    assign_slot(tgt)
    # Blender 5 layered Actions default to a ~25-frame strip — expand first
    try:
        layers = getattr(act, "layers", None)
        if layers:
            for layer in layers:
                for strip in getattr(layer, "strips", []) or []:
                    try:
                        strip.frame_start = float(f0)
                        strip.frame_end = float(f1 + 1)
                    except Exception:
                        pass
            log(f"layered strip expanded {f0}..{f1}")
    except Exception:
        pass

    bones = [tgt.pose.bones[sm] for _, sm in pairs]
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
            loc, rot, _ = mat_local.decompose()
            if pb.name == "pelvis":
                if args.root == "location":
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

    ensure_object_mode(tgt)
    try:
        bpy.data.objects.remove(src, do_unlink=True)
    except Exception:
        pass

    tgt.animation_data.action = act
    assign_slot(tgt)
    scene.frame_start, scene.frame_end = f0, f1
    scene.frame_set(f0)

    def wh(n):
        return tgt.matrix_world @ tgt.pose.bones[n].head

    checks = []
    for f in (f0, min(f0 + 30, f1), min(f0 + 60, f1)):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        hip, knee = wh("left_hip"), wh("left_knee")
        sh, wr = wh("left_shoulder"), wh("left_wrist")
        pel, head = wh("pelvis"), wh("head")
        checks.append(
            {
                "f": f,
                "legs_down": bool(knee.z < hip.z),
                "spine_up": bool(head.z > pel.z + 0.3),
                "hand_drop": round(sh.z - wr.z, 3),
                "spine2_w": round(tgt.pose.bones["spine2"].rotation_quaternion.w, 3),
                "collar_w": round(tgt.pose.bones["left_collar"].rotation_quaternion.w, 3),
                "L_elb_w": round(tgt.pose.bones["left_elbow"].rotation_quaternion.w, 3),
                "pelvis": [round(x, 3) for x in pel],
            }
        )
        log(f"verify {checks[-1]}")

    scene.frame_set(f0)
    bpy.context.view_layer.update()
    p0 = wh("pelvis").copy()
    scene.frame_set(f1)
    bpy.context.view_layer.update()
    p1 = wh("pelvis").copy()
    travel = (p1 - p0).length
    ok = travel > 0.5 and all(c["legs_down"] and c["spine_up"] for c in checks)

    report = {
        "ok": ok,
        "method": "Rokoko working map + full-frame helper bake (catalog math)",
        "src_blend": str(src_blend),
        "rokoko_map": str(rokoko_map),
        "rsl_pairs": len(rsl_pairs),
        "bake_pairs": len(pairs),
        "action": args.action,
        "frames": [f0, f1],
        "travel": round(travel, 3),
        "checks": checks,
        "saved": str(out),
        "note": "final hml3dto smpl.json is the canonical Rokoko custom map "
        "(leftUpperArm→left_shoulder, toes, fingers). Full body filled from FINAL_BONE_MAP.",
    }
    rep = ROOT / "body_motion" / "_proof_momask_rsl_working_report.json"
    rep.write_text(json.dumps(report, indent=2), encoding="utf-8")

    scene.frame_set(f0)
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
    except RuntimeError:
        alt = out.with_name(out.stem + "_v2" + out.suffix)
        bpy.ops.wm.save_as_mainfile(filepath=str(alt))
        report["saved"] = str(alt)
        rep.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log(f"report {rep}")
    log(
        f"DONE ok={ok} travel={travel:.3f} frames={f0}-{f1} pairs={len(pairs)} "
        f"saved={report['saved']}"
    )
    if not ok:
        raise SystemExit("retarget checks failed")


if __name__ == "__main__":
    main()
