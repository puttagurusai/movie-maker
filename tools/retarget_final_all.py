"""
Batch retarget all catalog Mixamo FBXs → SMPL-X Actions inside the open blend.

Uses FINAL_BONE_MAP.json + inplace root fix (no Apply Scale).

Usage:
  blender.exe "whole body.blend" --background --python tools/retarget_final_all.py --
  blender.exe "whole body.blend" --background --python tools/retarget_final_all.py -- --out "whole body.blend"
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "body_motion" / "FINAL_BONE_MAP.json"
CATALOG_PATH = ROOT / "body_motion" / "catalog.json"
SOURCE_DIR = ROOT / "body_motion" / "source_fbx"
RETARGET_ID = "_RSL_FINAL"
HELPER_SUFFIX = "_RSL_H"

# Clips that must keep root location even if auto-detect is borderline
FORCE_LOCATION = {"walk", "walk_back", "run"}
# Clips that must stay planted
FORCE_INPLACE = {
    "idle", "wave", "shrug", "point", "talk_open", "talk_emphasize",
    "nod_yes", "shake_no", "agree", "celebrate", "hands_reject",
    "tense", "slump", "bow", "look_around", "think", "recoil",
    "sit_idle", "turn_left", "turn_right",
}


def log(msg: str) -> None:
    print(f"[final_all] {msg}", flush=True)


def load_map() -> dict[str, str]:
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
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


def remove_action_if_exists(name: str, tgt) -> None:
    old = bpy.data.actions.get(name)
    if not old:
        return
    if tgt.animation_data and tgt.animation_data.action == old:
        tgt.animation_data.action = None
    try:
        bpy.data.actions.remove(old)
    except Exception:
        pass


def cleanup_mixamo_orphans(keep_names: set[str]) -> None:
    for a in list(bpy.data.actions):
        if a.name in keep_names:
            continue
        n = a.name.lower()
        if "mixamo" in n or "armature|" in n or a.name.startswith("mixamorig"):
            try:
                if a.users == 0:
                    bpy.data.actions.remove(a)
            except Exception:
                pass


def retarget_one(
    tgt,
    bone_map: dict[str, str],
    fbx_path: Path,
    action_name: str,
    force_root: str | None = None,
) -> dict:
    t0 = time.time()
    remove_action_if_exists(action_name, tgt)

    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(
        filepath=str(fbx_path),
        ignore_leaf_bones=True,
        automatic_bone_orientation=False,
    )
    new_objs = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]
    src = next((o for o in new_objs if o.type == "ARMATURE"), None)
    if not src:
        raise RuntimeError("no armature in FBX")
    for o in list(new_objs):
        if o.type == "MESH":
            bpy.data.objects.remove(o, do_unlink=True)

    if not src.animation_data or not src.animation_data.action:
        raise RuntimeError("source has no action")
    assign_slot(src)

    # Match height — keep object scale (do NOT apply)
    bpy.context.view_layer.update()
    hs, ht = height_hips_head(src, True), height_hips_head(tgt, False)
    if hs and ht and hs > 1e-6:
        s = ht / hs
        src.scale = (s * src.scale[0], s * src.scale[1], s * src.scale[2])
        bpy.context.view_layer.update()

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
    if len(pairs) < 10:
        raise RuntimeError(f"too few pairs: {len(pairs)}")

    # Helper bones (target rest orientation, parented to source)
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
    for pb in tgt.pose.bones:
        clear_constraints(pb)
        pb.select = False

    act_src = src.animation_data.action
    fr = act_src.frame_range
    f0, f1 = int(fr[0]), int(fr[1])
    if f1 <= f0:
        f0, f1 = 1, 60

    hips_bn = find_src_bone(src, "mixamorig:Hips")
    root_mode = force_root or "inplace"
    hip_travel = 0.0
    if force_root is None and hips_bn:
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
        else:
            root_mode = "inplace"

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
        pb.select = True
        if sm in root_names:
            cl = pb.constraints.new("COPY_LOCATION")
            cl.name = "Copy Loc" + RETARGET_ID
            cl.target = src
            cl.subtarget = sb

    # Source must move
    check = find_src_bone(src, "mixamorig:RightArm") or pairs[0][0]
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    q1 = src.evaluated_get(dg).pose.bones[check].matrix.to_quaternion().copy()
    mid = (f0 + f1) // 2
    bpy.context.scene.frame_set(mid)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    q2 = src.evaluated_get(dg).pose.bones[check].matrix.to_quaternion().copy()
    if q1 == q2:
        raise RuntimeError("SOURCE NOT ANIMATING")

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
            loc, rot, _sca = mat_local.decompose()
            if pb.name == "right_shoulder":
                max_rs = max(
                    max_rs,
                    abs(rot.x) + abs(rot.y) + abs(rot.z) + abs(1.0 - rot.w),
                )
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
        pb.select = False

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

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(src, do_unlink=True)

    ok = changing >= 5 or max_rs >= 0.05
    elapsed = time.time() - t0
    return {
        "action": action_name,
        "ok": ok,
        "pairs": len(pairs),
        "frames": [f0, f1],
        "root_mode": root_mode,
        "hip_travel": round(hip_travel, 4),
        "channels_changing": changing,
        "right_shoulder_delta": round(max_rs, 4),
        "seconds": round(elapsed, 1),
    }


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    out = ROOT / "whole body.blend"
    if "--out" in args:
        out = Path(args[args.index("--out") + 1])

    bone_map = load_map()
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    clips = catalog["clips"]

    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        raise SystemExit("no SMPL-X_Armature")

    # Playback / scene FPS (Mixamo is typically 30)
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.render.fps_base = 1.0
    if hasattr(scene, "sync_mode"):
        try:
            scene.sync_mode = "FRAME_DROP"  # prefer realtime over perfect frames
        except Exception:
            pass

    results = []
    keep = set()
    for clip_id, meta in clips.items():
        fbx_name = meta["fbx"]
        action_name = meta.get("action") or clip_id
        fbx_path = SOURCE_DIR / fbx_name
        if not fbx_path.is_file():
            log(f"SKIP {clip_id}: missing {fbx_name}")
            results.append({"action": action_name, "ok": False, "error": "missing fbx"})
            continue

        force = None
        if clip_id in FORCE_LOCATION:
            force = "location"
        elif clip_id in FORCE_INPLACE:
            force = "inplace"

        log(f"=== {clip_id} ← {fbx_name} → action={action_name} force_root={force}")
        try:
            r = retarget_one(tgt, bone_map, fbx_path, action_name, force_root=force)
            keep.add(action_name)
            cleanup_mixamo_orphans(keep)
            status = "OK" if r["ok"] else "WEAK"
            log(
                f"{status} {action_name} root={r['root_mode']} "
                f"frames={r['frames']} ch={r['channels_changing']} t={r['seconds']}s"
            )
            results.append(r)
        except Exception as e:
            log(f"FAIL {clip_id}: {e}")
            traceback.print_exc()
            results.append({"action": action_name, "ok": False, "error": str(e)})
            # best-effort cleanup of leftover import
            for o in list(bpy.data.objects):
                if o.type == "ARMATURE" and o != tgt and "mixamo" in o.name.lower():
                    try:
                        bpy.data.objects.remove(o, do_unlink=True)
                    except Exception:
                        pass

    # Leave idle (or wave) assigned for open
    preferred = ["idle", "wave", "walk"]
    for name in preferred:
        if bpy.data.actions.get(name):
            if not tgt.animation_data:
                tgt.animation_data_create()
            tgt.animation_data.action = bpy.data.actions[name]
            assign_slot(tgt)
            try:
                fr = bpy.data.actions[name].frame_range
                scene.frame_start = int(fr[0])
                scene.frame_end = int(fr[1])
            except Exception:
                pass
            break

    report = {
        "out": str(out),
        "map": str(MAP_PATH),
        "fps": scene.render.fps,
        "results": results,
        "ok_count": sum(1 for r in results if r.get("ok")),
        "total": len(results),
    }
    rep = ROOT / "body_motion" / "retarget_all_report.json"
    rep.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"report {rep}  ok={report['ok_count']}/{report['total']}")

    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
        log(f"SAVED {out}")
    except RuntimeError as e:
        alt = out.with_name(out.stem.replace(" ", "_") + "_retargeted.blend")
        if out.name == "whole body.blend":
            alt = ROOT / "whole_body_retargeted.blend"
        log(f"save failed ({e}); saving {alt}")
        bpy.ops.wm.save_as_mainfile(filepath=str(alt))
        log(f"SAVED {alt}")
        report["out"] = str(alt)
        rep.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
