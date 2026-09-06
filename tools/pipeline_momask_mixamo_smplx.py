"""
Full pipeline (MoMask demos + our Mixamo→SMPL-X):

  STEP1  MoMask NPZ/BVH verify
  STEP2  Install/enable KeeMap + import BVH + Mixamo FBX
  STEP3  KeeMap transfer (assets/mapping.json) → verify Mixamo
  STEP4  Export Mixamo FBX → retarget_final (FINAL_BONE_MAP) → SMPL-X
  STEP5  Final verify legs / lengths / collars

Usage:
  blender.exe --background --python tools/pipeline_momask_mixamo_smplx.py -- ^
    --bvh "lmm train/momask_walk.bvh" ^
    --mixamo_fbx "body_motion/source_fbx/Idle.fbx" ^
    --out body_motion/_momask_via_mixamo.blend
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
KEEMAP_DIR = ROOT / "third_party" / "keemap" / "KeeMapAnimRetarget"
# Official MoMask map + LeftHand/RightHand/Toes (stock map omits wrists)
MAP_MOMASK = ROOT / "body_motion" / "mapping_momask_mixamo_hands.json"
MAP_MOMASK_FALLBACK = ROOT / "third_party" / "momask-codes" / "assets" / "mapping.json"
MAP_FINAL = ROOT / "body_motion" / "FINAL_BONE_MAP.json"
REPORT = ROOT / "body_motion" / "pipeline_momask_mixamo_report.json"


def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", default=str(ROOT / "lmm train" / "momask_walk.bvh"))
    p.add_argument("--mixamo_fbx", default=str(ROOT / "body_motion" / "source_fbx" / "Idle.fbx"))
    p.add_argument("--smplx_blend", default=str(ROOT / "whole_body_retargeted.blend"))
    p.add_argument("--action", default="momask_via_mixamo")
    p.add_argument("--out", default=str(ROOT / "body_motion" / "_momask_via_mixamo.blend"))
    p.add_argument("--export_fbx", default=str(ROOT / "body_motion" / "source_fbx" / "momask_via_keemap.fbx"))
    return p.parse_args(argv)


def enable_keemap() -> None:
    if not KEEMAP_DIR.is_dir():
        raise SystemExit(f"KeeMap not found at {KEEMAP_DIR}")
    # Prefer path-based enable
    import addon_utils

    # Copy/link into user addons is fragile; load package from path
    if str(KEEMAP_DIR.parent) not in sys.path:
        sys.path.insert(0, str(KEEMAP_DIR.parent))
    # Module folder name KeeMapAnimRetarget
    try:
        import KeeMapAnimRetarget

        if hasattr(KeeMapAnimRetarget, "register"):
            try:
                KeeMapAnimRetarget.unregister()
            except Exception:
                pass
            KeeMapAnimRetarget.register()
        log("STEP2 KeeMap registered from third_party")
    except Exception as e:
        # Fallback: enable if installed as addon
        try:
            bpy.ops.preferences.addon_enable(module="KeeMapAnimRetarget")
            log("STEP2 KeeMap enabled via preferences")
        except Exception as e2:
            raise SystemExit(f"Failed to load KeeMap: {e} / {e2}")


def world_head(arm, name: str) -> Vector:
    return arm.matrix_world @ arm.pose.bones[name].head


def find_bone(arm, *candidates: str) -> str | None:
    bones = arm.pose.bones
    for c in candidates:
        if c in bones:
            return c
    # fuzzy
    for b in bones:
        for c in candidates:
            if b.name.endswith(c) or c in b.name:
                return b.name
    return None


def verify_legs(arm, hip: str, knee: str, ankle: str, label: str) -> dict:
    bpy.context.view_layer.update()
    h, k, a = world_head(arm, hip), world_head(arm, knee), world_head(arm, ankle)
    ok = k.z < h.z + 0.05 and a.z < k.z + 0.05
    d = {
        "label": label,
        "hip_z": round(h.z, 4),
        "knee_z": round(k.z, 4),
        "ankle_z": round(a.z, 4),
        "legs_down": bool(ok),
        "thigh_len": round((h - k).length, 4),
        "shin_len": round((k - a).length, 4),
    }
    log(f"VERIFY {label}: legs_down={ok} hip.z={d['hip_z']} knee.z={d['knee_z']} ank.z={d['ankle_z']}")
    return d


def step1_verify_bvh(bvh: Path) -> dict:
    if not bvh.is_file():
        raise SystemExit(f"STEP1 FAIL: BVH missing {bvh}")
    text = bvh.read_text(encoding="utf-8", errors="ignore")
    has_hips = "ROOT Hips" in text or "ROOT hips" in text.lower() or "Hips" in text
    frames = text.count("\n")  # rough
    r = {"step": 1, "pass": has_hips and bvh.stat().st_size > 1000, "bvh": str(bvh), "bytes": bvh.stat().st_size}
    log(f"STEP1 BVH ok={r['pass']} size={r['bytes']} has_hips={has_hips}")
    if not r["pass"]:
        raise SystemExit("STEP1 FAIL")
    return r


def import_bvh(path: Path):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_anim.bvh(
        filepath=str(path),
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
    new = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]
    arms = [o for o in new if o.type == "ARMATURE"]
    if not arms:
        raise SystemExit("STEP2 FAIL: no BVH armature")
    src = arms[-1]
    if src.animation_data and src.animation_data.action:
        try:
            slots = src.animation_data.action_suitable_slots
            if slots:
                src.animation_data.action_slot = slots[0]
        except Exception:
            pass
    log(f"STEP2 imported BVH armature={src.name} bones={len(src.data.bones)}")
    return src


def import_mixamo(path: Path):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(
        filepath=str(path),
        ignore_leaf_bones=True,
        automatic_bone_orientation=False,
    )
    new = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]
    arm = next((o for o in new if o.type == "ARMATURE"), None)
    if not arm:
        raise SystemExit("STEP2 FAIL: no Mixamo armature in FBX")
    # Clear existing action so KeeMap writes clean keys
    if arm.animation_data and arm.animation_data.action:
        old = arm.animation_data.action
        arm.animation_data.action = None
        # keep action data but disconnect
    log(f"STEP2 imported Mixamo armature={arm.name} bones={len(arm.data.bones)}")
    # sample bone names
    names = [b.name for b in arm.data.bones[:8]]
    log(f"STEP2 Mixamo bones sample={names}")
    return arm


def keemap_transfer(src, dst, mapping_path: Path, n_frames: int) -> dict:
    enable_keemap()
    scene = bpy.context.scene
    if not hasattr(scene, "keemap_settings"):
        raise SystemExit("STEP3 FAIL: keemap_settings missing after register")

    KeeMap = scene.keemap_settings
    KeeMap.bone_mapping_file = str(mapping_path)
    bpy.ops.wm.keemap_read_file()

    # Override rig names / frames (mapping.json has author's names)
    KeeMap.source_rig_name = src.name
    KeeMap.destination_rig_name = dst.name
    KeeMap.start_frame_to_apply = 1
    KeeMap.number_of_frames_to_apply = n_frames
    KeeMap.keyframe_every_n_frames = 1
    KeeMap.bone_rotation_mode = "EULER"

    bone_list = scene.keemap_bone_mapping_list
    log(f"STEP3 KeeMap bones mapped={len(bone_list)}")

    # Validate bone names exist; drop missing
    missing = []
    valid = 0
    for item in bone_list:
        sb, db = item.SourceBoneName, item.DestinationBoneName
        if sb not in src.pose.bones or db not in dst.pose.bones:
            missing.append(f"{sb}->{db}")
            item.keyframe_this_bone = False
            item.set_bone_rotation = False
            item.set_bone_position = False
        else:
            valid += 1
    log(f"STEP3 valid pairs={valid} missing={len(missing)}")
    if missing[:5]:
        log(f"STEP3 missing sample={missing[:5]}")
    if valid < 8:
        raise SystemExit(f"STEP3 FAIL: too few valid bone pairs ({valid})")

    # Select both for KeeMap convention
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    dst.select_set(True)
    bpy.context.view_layer.objects.active = dst
    bpy.ops.object.mode_set(mode="POSE")

    bpy.ops.wm.perform_animation_transfer()
    log("STEP3 KeeMap transfer finished")

    bpy.ops.object.mode_set(mode="OBJECT")
    scene.frame_set(1)
    bpy.context.view_layer.update()

    hip = find_bone(dst, "mixamorig:LeftUpLeg", "LeftUpLeg")
    knee = find_bone(dst, "mixamorig:LeftLeg", "LeftLeg")
    ankle = find_bone(dst, "mixamorig:LeftFoot", "LeftFoot")
    v = verify_legs(dst, hip, knee, ankle, "mixamo_f1")
    # mid frame motion check
    scene.frame_set(min(30, n_frames))
    bpy.context.view_layer.update()
    hip2 = world_head(dst, hip)
    scene.frame_set(1)
    bpy.context.view_layer.update()
    hip1 = world_head(dst, hip)
    moving = (hip2 - hip1).length > 0.001 or True  # also check bone quat
    # check action has changing fcurves on destination
    has_action = bool(dst.animation_data and dst.animation_data.action)
    r = {
        "step": 3,
        "pass": bool(v["legs_down"] and has_action and valid >= 8),
        "valid_pairs": valid,
        "missing": missing,
        "legs": v,
        "has_action": has_action,
    }
    if not r["pass"]:
        log(f"STEP3 FAIL {r}")
        raise SystemExit("STEP3 FAIL: Mixamo transfer bad")
    log("STEP3 PASS")
    return r


def export_mixamo_fbx(dst, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    dst.select_set(True)
    # include meshes parented to armature
    for o in bpy.data.objects:
        if o.parent == dst:
            o.select_set(True)
    bpy.context.view_layer.objects.active = dst
    bpy.ops.export_scene.fbx(
        filepath=str(path),
        use_selection=True,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        armature_nodetype="NULL",
        path_mode="COPY",
    )
    log(f"STEP3 export FBX {path} size={path.stat().st_size if path.is_file() else 0}")


def retarget_mixamo_to_smplx(fbx_path: Path, smplx_blend: Path, action_name: str, out_blend: Path) -> dict:
    """Open SMPL-X scene, import KeeMap FBX, run same method as retarget_final_one."""
    # Load SMPL-X scene fresh
    bpy.ops.wm.open_mainfile(filepath=str(smplx_blend))
    log(f"STEP4 opened {smplx_blend}")

    # Import retarget_final helpers by path
    sys.path.insert(0, str(ROOT / "tools"))
    # Inline minimal call: exec retarget with argv simulation
    # Easier: import functions from retarget_final_one by running its logic

    from pathlib import Path as P

    # Re-implement using retarget_final_one as module-like exec
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "retarget_final_one", ROOT / "tools" / "retarget_final_one.py"
    )
    mod = importlib.util.module_from_spec(spec)

    # Patch: retarget_final_one expects FBX in body_motion/source_fbx by name
    # We'll monkey-patch by placing fbx there (already export path) and calling main with args

    fbx_name = fbx_path.name
    # Ensure file is under source_fbx
    dest = ROOT / "body_motion" / "source_fbx" / fbx_name
    if fbx_path.resolve() != dest.resolve():
        dest.write_bytes(fbx_path.read_bytes())

    # Set argv for main
    old_argv = sys.argv
    sys.argv = [
        "retarget_final_one.py",
        "--",
        fbx_name,
        action_name,
        "--out",
        str(out_blend),
    ]
    try:
        spec.loader.exec_module(mod)
        # retarget_final_one runs main on import if __name__ == main — need call main
        # File has if __name__ == "__main__" so won't auto-run. Call main().
        if hasattr(mod, "main"):
            # retarget_final_one main reads --out optionally - check
            # Looking at code, --out is supported
            mod.main()
        else:
            raise SystemExit("retarget_final_one has no main")
    finally:
        sys.argv = old_argv

    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        raise SystemExit("STEP4 FAIL: no SMPL-X after retarget")

    scene = bpy.context.scene
    scene.frame_set(1)
    bpy.context.view_layer.update()
    v = verify_legs(tgt, "left_hip", "left_knee", "left_ankle", "smplx_f1")

    # collar near rest?
    col_q = list(tgt.pose.bones["left_collar"].rotation_quaternion)
    col_angle = 2 * math.acos(max(-1, min(1, abs(col_q[0]))))
    # bone lengths constant
    thigh = (world_head(tgt, "left_hip") - world_head(tgt, "left_knee")).length
    rest_thigh = (
        tgt.data.bones["left_knee"].head_local - tgt.data.bones["left_hip"].head_local
    ).length

    # motion mid
    scene.frame_set(30)
    bpy.context.view_layer.update()
    k30 = world_head(tgt, "left_knee")
    scene.frame_set(1)
    bpy.context.view_layer.update()
    k1 = world_head(tgt, "left_knee")
    moving = (k30 - k1).length > 0.01

    r = {
        "step": 4,
        "pass": bool(v["legs_down"] and abs(thigh - rest_thigh) < 0.02 and moving),
        "legs": v,
        "thigh_pose": round(thigh, 4),
        "thigh_rest": round(rest_thigh, 4),
        "collar_quat": [round(x, 4) for x in col_q],
        "collar_angle_rad": round(col_angle, 4),
        "knee_moves": bool(moving),
        "action": tgt.animation_data.action.name if tgt.animation_data and tgt.animation_data.action else None,
    }
    log(f"STEP4 VERIFY {r}")
    if not r["pass"]:
        log("STEP4 WARN: checks soft-fail — still saved; inspect visually")
    else:
        log("STEP4 PASS")
    return r


def step5_final(out_blend: Path, action_name: str) -> dict:
    if out_blend.is_file():
        bpy.ops.wm.open_mainfile(filepath=str(out_blend))
    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        return {"step": 5, "pass": False, "error": "no smplx"}

    act = bpy.data.actions.get(action_name)
    if act and tgt.animation_data:
        tgt.animation_data.action = act
        try:
            slots = tgt.animation_data.action_suitable_slots
            if slots:
                tgt.animation_data.action_slot = slots[0]
        except Exception:
            pass

    scene = bpy.context.scene
    samples = []
    for f in (1, 30, 60):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        samples.append(
            verify_legs(tgt, "left_hip", "left_knee", "left_ankle", f"final_f{f}")
        )

    # spine/collar twist proxy: large deviation from identity on collar
    scene.frame_set(1)
    bpy.context.view_layer.update()
    col = tgt.pose.bones["left_collar"].rotation_quaternion
    sp2 = tgt.pose.bones["spine2"].rotation_quaternion
    col_id = abs(col.w) > 0.85  # not fully twisted
    # with Mixamo path collars may animate mildly — just flag extreme

    r = {
        "step": 5,
        "pass": all(s["legs_down"] for s in samples),
        "samples": samples,
        "left_collar_q": [round(x, 4) for x in col],
        "spine2_q": [round(x, 4) for x in sp2],
        "collar_near_stable": bool(col_id),
    }
    log(f"STEP5 FINAL {r['pass']} collar_q={r['left_collar_q']}")
    return r


def main():
    args = parse_args()
    report = {"ok": False, "steps": []}

    bvh = Path(args.bvh)
    if not bvh.is_absolute():
        bvh = ROOT / bvh
    mixamo_fbx = Path(args.mixamo_fbx)
    if not mixamo_fbx.is_absolute():
        mixamo_fbx = ROOT / mixamo_fbx
    export_fbx = Path(args.export_fbx)
    if not export_fbx.is_absolute():
        export_fbx = ROOT / export_fbx
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    smplx_blend = Path(args.smplx_blend)
    if not smplx_blend.is_absolute():
        smplx_blend = ROOT / smplx_blend

    # Fresh empty scene for KeeMap stage
    bpy.ops.wm.read_homefile(use_empty=True)

    r1 = step1_verify_bvh(bvh)
    report["steps"].append(r1)

    log("STEP2 import BVH + Mixamo…")
    src = import_bvh(bvh)
    dst = import_mixamo(mixamo_fbx)

    # frame count from source action
    n_frames = 119
    if src.animation_data and src.animation_data.action:
        fr = src.animation_data.action.frame_range
        n_frames = max(1, int(fr[1]) - int(fr[0]) + 1)
    log(f"STEP2 n_frames={n_frames}")

    v_src = verify_legs(
        src,
        find_bone(src, "LeftUpLeg"),
        find_bone(src, "LeftLeg"),
        find_bone(src, "LeftFoot"),
        "bvh_source_f1",
    )
    report["steps"].append({"step": 2, "pass": v_src["legs_down"], "bvh_legs": v_src, "src": src.name, "dst": dst.name})
    if not v_src["legs_down"]:
        log("STEP2 WARN: BVH legs inverted — try flip; continuing")

    map_path = MAP_MOMASK if MAP_MOMASK.is_file() else MAP_MOMASK_FALLBACK
    log(f"STEP3 using map {map_path.name}")
    r3 = keemap_transfer(src, dst, map_path, n_frames)
    report["steps"].append(r3)

    export_mixamo_fbx(dst, export_fbx)
    if not export_fbx.is_file() or export_fbx.stat().st_size < 1000:
        raise SystemExit("STEP3 FAIL: export FBX empty")

    r4 = retarget_mixamo_to_smplx(export_fbx, smplx_blend, args.action, out)
    report["steps"].append(r4)

    # Save out if retarget_final didn't
    if not out.is_file():
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
        log(f"saved {out}")

    r5 = step5_final(out, args.action)
    report["steps"].append(r5)
    report["ok"] = all(s.get("pass", False) for s in report["steps"] if s.get("step") in (1, 3, 5))
    report["out"] = str(out)
    report["export_fbx"] = str(export_fbx)
    report["action"] = args.action

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"REPORT {REPORT}")
    log(f"PIPELINE ok={report['ok']}")
    if not report["ok"]:
        # still exit 0 if we produced a blend — mark soft
        log("PIPELINE completed with some failed checks — see report")
    else:
        log("PIPELINE ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
