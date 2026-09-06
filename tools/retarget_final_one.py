"""
Retarget ONE Mixamo FBX → SMPL-X using FINAL_BONE_MAP.json.

Rokoko-style: helper bones on source match target rest orientation,
COPY_ROTATION (and COPY_LOCATION on pelvis), then visual-key bake.

Usage (Blender 5.1):
  blender.exe "body_motion/whole body.blend" --background --python tools/retarget_final_one.py -- wave.fbx wave
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "body_motion" / "FINAL_BONE_MAP.json"
RETARGET_ID = "_RSL_FINAL"
HELPER_SUFFIX = "_RSL_H"


def log(msg: str) -> None:
    print(f"[final_one] {msg}", flush=True)


def load_map() -> dict[str, str]:
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    m = data["map"]
    log(f"loaded FINAL map ({len(m)} pairs) status={data.get('status')}")
    return m


def strip_prefix(name: str) -> str:
    return name.split(":")[-1].replace("mixamorig", "")


def find_src_bone(arm, mixamo_full: str) -> str | None:
    """Resolve mixamorig:Bone or Bone on source armature."""
    if mixamo_full in arm.pose.bones:
        return mixamo_full
    short = strip_prefix(mixamo_full)
    for b in arm.pose.bones:
        if strip_prefix(b.name) == short or b.name == short:
            return b.name
    # try without mixamorig:
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


def mat3_to_vec_roll(mat):
    """Same idea as Rokoko utils: extract roll from 3x3."""
    vec = mat.col[1].normalized()
    # upsample to roll angle
    matq = mat.to_quaternion()
    # Blender edit_bones use roll around Y; approximate via matrix
    # Prefer built-in if available
    try:
        from mathutils import Matrix as M

        # use vector and align
        x = mat.col[0].normalized()
        y = mat.col[1].normalized()
        z = mat.col[2].normalized()
        # roll from X projected
        return y, math.atan2(x.z, z.z) if abs(y.y) < 0.99 else 0.0
    except Exception:
        return vec, 0.0


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


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    fbx_name = args[0] if args else "wave.fbx"
    action_name = args[1] if len(args) > 1 else Path(fbx_name).stem.lower().replace(" ", "_").replace("@", "_")
    # clean catalog-style names
    for ch in "()[]":
        action_name = action_name.replace(ch, "")
    action_name = action_name.replace("__", "_")

    fbx = ROOT / "body_motion" / "source_fbx" / fbx_name
    if not fbx.is_file():
        raise SystemExit(f"missing FBX: {fbx}")

    bone_map = load_map()
    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        raise SystemExit("no SMPL-X_Armature in file")

    # Clear old action with same name
    old = bpy.data.actions.get(action_name)
    if old:
        if tgt.animation_data and tgt.animation_data.action == old:
            tgt.animation_data.action = None
        bpy.data.actions.remove(old)

    # Import Mixamo FBX
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(filepath=str(fbx), ignore_leaf_bones=True, automatic_bone_orientation=False)
    new_objs = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]
    src = next((o for o in new_objs if o.type == "ARMATURE"), None)
    if not src:
        raise SystemExit("no armature in FBX")
    for o in list(new_objs):
        if o.type == "MESH":
            bpy.data.objects.remove(o, do_unlink=True)

    if not src.animation_data or not src.animation_data.action:
        raise SystemExit("source FBX has no action")
    assign_slot(src)
    log(f"source={src.name} action={src.animation_data.action.name}")

    # Scale source to match target height.
    # IMPORTANT: keep scale on the object — do NOT apply scale.
    # Mixamo FBX imports at ~0.01 scale with hip location keys in cm-like units.
    # Applying scale turns tiny world motion into ~2m fake root slide on wave/idle.
    bpy.context.view_layer.update()
    hs, ht = height_hips_head(src, True), height_hips_head(tgt, False)
    if hs and ht and hs > 1e-6:
        s = ht / hs
        src.scale = (s * src.scale[0], s * src.scale[1], s * src.scale[2])
        bpy.context.view_layer.update()
        log(f"scale factor {s:.4f} (object scale kept, not applied) → {tuple(round(x, 5) for x in src.scale)}")

    # Align hips horizontally / vertically
    hips = find_src_bone(src, "mixamorig:Hips")
    if hips and "pelvis" in tgt.data.bones:
        sh = src.matrix_world @ src.data.bones[hips].head_local
        th = tgt.matrix_world @ tgt.data.bones["pelvis"].head_local
        src.location += th - sh
        bpy.context.view_layer.update()

    # Build resolved pairs (src bone name → tgt bone name)
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

    # --- Rokoko-style helper bones on source (target rest in source space) ---
    mw_src_inv = src.matrix_world.inverted()
    # capture target bone rest head/tail/roll in source local
    bpy.ops.object.select_all(action="DESELECT")
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="EDIT")
    bone_transforms = {}
    for eb in tgt.data.edit_bones:
        head = mw_src_inv @ (tgt.matrix_world @ eb.head)
        tail = mw_src_inv @ (tgt.matrix_world @ eb.tail)
        # roll from world bone matrix
        m3 = (mw_src_inv.to_3x3() @ (tgt.matrix_world.to_3x3() @ eb.matrix.to_3x3()))
        # EditBone.roll is already set; recompute roughly
        bone_transforms[eb.name] = (head.copy(), tail.copy(), eb.roll)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Add helpers parented to source bones
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

    # Constraints on target
    bpy.ops.object.select_all(action="DESELECT")
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="POSE")

    for pb in tgt.pose.bones:
        clear_constraints(pb)
        pb.select = False

    # Frame range from source action (needed before root-mode decision)
    act_src = src.animation_data.action
    fr = act_src.frame_range
    f0, f1 = int(fr[0]), int(fr[1])
    if f1 <= f0:
        f0, f1 = 1, 60
    log(f"frames {f0}..{f1}")

    # Measure source hip world travel — wave/idle stay planted; walk/run translate.
    # Baking pelvis location when source is planted caused ~2m whole-body slide.
    hips_bn = find_src_bone(src, "mixamorig:Hips")
    root_mode = "inplace"  # default: rotation only on pelvis (no root location keys)
    if "--root" in args:
        root_mode = args[args.index("--root") + 1].strip().lower()
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
        # threshold in Blender units after source scale (~meters). Locomotion >> 0.15
        if hip_travel > 0.15:
            root_mode = "location"
        log(f"source hips travel≈{hip_travel:.4f} → root_mode={root_mode}")
    else:
        log(f"root_mode={root_mode} (no hips bone)")

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

    # Verify source animates
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

    # Bake visual pose into new action via keyframe_insert
    if not tgt.animation_data:
        tgt.animation_data_create()
    act = bpy.data.actions.new(action_name)
    act.use_fake_user = True
    tgt.animation_data.action = act
    assign_slot(tgt)

    bones = [tgt.pose.bones[sm] for _, sm in pairs]
    max_rs = 0.0
    pelvis0 = None  # first-frame local loc for relative root (locomotion)
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
                    # relative to start so clip begins at origin (no spawn offset)
                    pb.location = loc - pelvis0
                    pb.keyframe_insert("location", frame=f)
                else:
                    # inplace: keep feet/root planted (source wave/idle don't translate)
                    pb.location = (0.0, 0.0, 0.0)
                    pb.keyframe_insert("location", frame=f)
            pb.rotation_quaternion = rot
            pb.keyframe_insert("rotation_quaternion", frame=f)

    # Remove constraints
    for pb in tgt.pose.bones:
        clear_constraints(pb)
        pb.select = False

    # Verify non-rest keys
    changing = 0
    try:
        # Blender 5 layered
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
        raise SystemExit("BAKE LOOKS REST — abort save")

    # Cleanup source + helpers
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(src, do_unlink=True)
    for a in list(bpy.data.actions):
        if a == act:
            continue
        if "mixamo" in a.name.lower() or "Armature|" in a.name or a.name.startswith("mixamorig"):
            try:
                if a.users == 0:
                    bpy.data.actions.remove(a)
            except Exception:
                pass

    # Assign action on target for open-in-UI
    tgt.animation_data.action = act
    assign_slot(tgt)
    bpy.context.scene.frame_start = f0
    bpy.context.scene.frame_end = f1
    bpy.context.scene.frame_set(f0)

    out = ROOT / "body_motion" / f"_proof_{action_name}_final.blend"
    if "--out" in args:
        i = args.index("--out")
        out = Path(args[i + 1])
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
    except RuntimeError as e:
        # File locked / "saved with @" — write alternate path
        alt = out.with_name(out.stem + "_v2" + out.suffix)
        log(f"save failed ({e}); trying {alt}")
        bpy.ops.wm.save_as_mainfile(filepath=str(alt))
        out = alt
    log(f"OK action={action_name} pairs={len(pairs)} root_mode={root_mode} saved={out}")

    # Write small report
    report = {
        "action": action_name,
        "fbx": fbx_name,
        "pairs": len(pairs),
        "frames": [f0, f1],
        "channels_changing": changing,
        "right_shoulder_delta": max_rs,
        "blend": str(out),
        "map": str(MAP_PATH),
    }
    rep_path = ROOT / "body_motion" / f"_proof_{action_name}_final_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"report {rep_path}")


if __name__ == "__main__":
    main()
