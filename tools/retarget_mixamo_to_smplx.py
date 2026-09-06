"""
Retarget ALL Mixamo FBXs in body_motion/source_fbx onto SMPL-X_Armature.

Uses the same method as retarget_one_clip.py (proven MOVING):
  - assign Blender 5 action slots on Mixamo source
  - COPY_ROTATION WORLD
  - bake via evaluated depsgraph → local keys
  - abort if keys stay rest-pose

Run (saves NEW file — does not lock open blends):
  blender --background "body_motion/whole_body_motions_face_attached.blend" ^
    --python tools/retarget_mixamo_to_smplx.py ^
    -- --out "body_motion/avatar_with_clips.blend"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
FBX_DIR = ROOT / "body_motion" / "source_fbx"
CATALOG = ROOT / "body_motion" / "catalog.json"
DEFAULT_OUT = ROOT / "body_motion" / "avatar_with_clips.blend"
ARM_NAME = "SMPL-X_Armature"

BONE_MAP = {
    "Hips": "pelvis",
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
    "LeftUpLeg": "left_hip",
    "LeftLeg": "left_knee",
    "LeftFoot": "left_ankle",
    "LeftToeBase": "left_foot",
    "RightUpLeg": "right_hip",
    "RightLeg": "right_knee",
    "RightFoot": "right_ankle",
    "RightToeBase": "right_foot",
}


def log(msg: str) -> None:
    print(f"[retarget] {msg}", flush=True)


def find_bn(arm, logical: str) -> str | None:
    for b in arm.data.bones:
        n = b.name.split(":")[-1].replace("mixamorig", "")
        if n == logical:
            return b.name
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


def get_target():
    arm = bpy.data.objects.get(ARM_NAME)
    if arm:
        return arm
    for o in bpy.data.objects:
        if o.type == "ARMATURE":
            return o
    raise RuntimeError("No armature")


def clear_constraints(arm) -> None:
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    for pb in arm.pose.bones:
        while pb.constraints:
            pb.constraints.remove(pb.constraints[0])
    bpy.ops.object.mode_set(mode="OBJECT")


def retarget_one(tgt, fbx_path: Path, action_name: str) -> str:
    """Returns 'ok' or error string."""
    if not fbx_path.is_file():
        return "missing_fbx"

    # remove previous action same name
    old = bpy.data.actions.get(action_name)
    if old:
        try:
            bpy.data.actions.remove(old)
        except Exception:
            pass

    clear_constraints(tgt)
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(filepath=str(fbx_path), ignore_leaf_bones=True)
    new = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]
    arms = [o for o in new if o.type == "ARMATURE"]
    if not arms:
        return "no_armature"
    src = arms[0]
    for o in new:
        if o.type == "MESH":
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass

    # bind mixamo action + slot (critical on Blender 5)
    if not src.animation_data or not src.animation_data.action:
        for act in bpy.data.actions:
            if "mixamo" in act.name.lower() or "|" in act.name:
                if not src.animation_data:
                    src.animation_data_create()
                src.animation_data.action = act
                break
    assign_slot(src)

    # scale to target height
    bpy.context.view_layer.update()

    def height(arm, mix: bool):
        if mix:
            h, hd = find_bn(arm, "Hips"), find_bn(arm, "Head")
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

    hs, ht = height(src, True), height(tgt, False)
    if hs and ht and hs > 1e-6:
        s = ht / hs
        src.scale = (s, s, s)
        bpy.context.view_layer.update()

    h = find_bn(src, "Hips")
    if h and "pelvis" in tgt.data.bones:
        sh = src.matrix_world @ src.data.bones[h].head_local
        th = tgt.matrix_world @ tgt.data.bones["pelvis"].head_local
        src.location += th - sh
        bpy.context.view_layer.update()

    # verify source animates
    bn = find_bn(src, "RightArm") or find_bn(src, "LeftArm")
    if not bn or not src.animation_data or not src.animation_data.action:
        return "source_no_anim"
    fr = src.animation_data.action.frame_range
    f0, f1 = int(fr[0]), int(max(fr[1], fr[0] + 1))
    if f1 - f0 > 300:
        f1 = f0 + 300
    mid = (f0 + f1) // 2

    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    q1 = src.evaluated_get(dg).pose.bones[bn].matrix.to_quaternion().copy()
    bpy.context.scene.frame_set(mid)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    q2 = src.evaluated_get(dg).pose.bones[bn].matrix.to_quaternion().copy()
    if q1 == q2:
        return "source_not_moving"

    # constraints
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="POSE")
    mapped = []
    for mix, sm in BONE_MAP.items():
        if sm not in tgt.pose.bones:
            continue
        sbn = find_bn(src, mix)
        if not sbn:
            continue
        pb = tgt.pose.bones[sm]
        while pb.constraints:
            pb.constraints.remove(pb.constraints[0])
        c = pb.constraints.new("COPY_ROTATION")
        c.target = src
        c.subtarget = sbn
        c.target_space = "WORLD"
        c.owner_space = "WORLD"
        c.mix_mode = "REPLACE"
        pb.rotation_mode = "QUATERNION"
        mapped.append(sm)

    if not tgt.animation_data:
        tgt.animation_data_create()
    act = bpy.data.actions.new(name=action_name)
    act.use_fake_user = True
    tgt.animation_data.action = act
    assign_slot(tgt)

    bones = [tgt.pose.bones[n] for n in mapped]
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
            pb.location = loc
            pb.rotation_quaternion = rot
            pb.scale = sca
            pb.keyframe_insert("location", frame=f)
            pb.keyframe_insert("rotation_quaternion", frame=f)
            pb.keyframe_insert("scale", frame=f)

    for pb in tgt.pose.bones:
        while pb.constraints:
            pb.constraints.remove(pb.constraints[0])

    # verify keys not rest
    try:
        strip = act.layers[0].strips[0]
        cb = strip.channelbags[0]
        diffs = 0
        for fc in cb.fcurves:
            if abs(fc.evaluate(f0) - fc.evaluate(mid)) > 0.02:
                diffs += 1
    except Exception:
        diffs = 0

    bpy.ops.object.mode_set(mode="OBJECT")
    if tgt.animation_data:
        tgt.animation_data.action = None

    # remove source armature
    try:
        bpy.data.objects.remove(src, do_unlink=True)
    except Exception:
        pass
    for a in list(bpy.data.actions):
        if "mixamo" in a.name.lower() or "|" in a.name:
            try:
                bpy.data.actions.remove(a)
            except Exception:
                pass

    if diffs < 5:
        # remove bad action
        try:
            bpy.data.actions.remove(act)
        except Exception:
            pass
        return f"rest_keys(diffs={diffs})"

    log(f"  OK {action_name} frames {f0}-{f1} changing_channels={diffs}")
    return "ok"


def main():
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    out = DEFAULT_OUT
    if "--out" in args:
        out = Path(args[args.index("--out") + 1])

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    clips = catalog["clips"]
    tgt = get_target()
    log(f"target {tgt.name} bones={len(tgt.data.bones)}")
    log(f"out → {out}")

    report = {}
    for clip_id, meta in clips.items():
        fbx = FBX_DIR / meta["fbx"]
        action = meta.get("action") or clip_id
        log(f"=== {clip_id} ← {fbx.name}")
        try:
            status = retarget_one(tgt, fbx, action)
        except Exception as e:
            status = f"error:{e}"
            log(f"  FAIL {e}")
        report[clip_id] = status
        if status != "ok":
            log(f"  FAIL {status}")

    # leave idle active if ok
    if not tgt.animation_data:
        tgt.animation_data_create()
    if bpy.data.actions.get("idle"):
        tgt.animation_data.action = bpy.data.actions["idle"]
        assign_slot(tgt)
    try:
        tgt.hide_set(False)
        tgt.hide_viewport = False
    except Exception:
        pass

    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    log(f"saved {out}")

    rep_path = ROOT / "body_motion" / "retarget_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ok = sum(1 for v in report.values() if v == "ok")
    log(f"DONE {ok}/{len(report)} → {rep_path}")

    # final verify wave
    if bpy.data.actions.get("wave"):
        tgt.animation_data.action = bpy.data.actions["wave"]
        assign_slot(tgt)
        bpy.context.scene.frame_set(1)
        bpy.context.view_layer.update()
        q1 = list(tgt.pose.bones["right_shoulder"].rotation_quaternion)
        bpy.context.scene.frame_set(70)
        bpy.context.view_layer.update()
        q2 = list(tgt.pose.bones["right_shoulder"].rotation_quaternion)
        log("VERIFY wave " + ("MOVING" if q1 != q2 else "NOT MOVING"))


if __name__ == "__main__":
    main()
