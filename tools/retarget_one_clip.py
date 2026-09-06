"""Retarget ONE Mixamo FBX onto SMPL-X with verified non-rest keys. Blender 5 safe."""
from __future__ import annotations

import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
# FINAL map (logical Mixamo name without mixamorig:). See body_motion/FINAL_BONE_MAP.json
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
    "LeftHandThumb1": "left_thumb1",
    "LeftHandThumb2": "left_thumb2",
    "LeftHandThumb3": "left_thumb3",
    "LeftHandIndex1": "left_index1",
    "LeftHandIndex2": "left_index2",
    "LeftHandIndex3": "left_index3",
    "LeftHandMiddle1": "left_middle1",
    "LeftHandMiddle2": "left_middle2",
    "LeftHandMiddle3": "left_middle3",
    "LeftHandRing1": "left_ring1",
    "LeftHandRing2": "left_ring2",
    "LeftHandRing3": "left_ring3",
    "LeftHandPinky1": "left_pinky1",
    "LeftHandPinky2": "left_pinky2",
    "LeftHandPinky3": "left_pinky3",
    "RightHandThumb1": "right_thumb1",
    "RightHandThumb2": "right_thumb2",
    "RightHandThumb3": "right_thumb3",
    "RightHandIndex1": "right_index1",
    "RightHandIndex2": "right_index2",
    "RightHandIndex3": "right_index3",
    "RightHandMiddle1": "right_middle1",
    "RightHandMiddle2": "right_middle2",
    "RightHandMiddle3": "right_middle3",
    "RightHandRing1": "right_ring1",
    "RightHandRing2": "right_ring2",
    "RightHandRing3": "right_ring3",
    "RightHandPinky1": "right_pinky1",
    "RightHandPinky2": "right_pinky2",
    "RightHandPinky3": "right_pinky3",
}


def log(m):
    print(f"[one] {m}", flush=True)


def find_bn(arm, logical):
    for b in arm.data.bones:
        n = b.name.split(":")[-1].replace("mixamorig", "")
        if n == logical:
            return b.name
    return None


def assign_slot(obj):
    if not obj.animation_data or not obj.animation_data.action:
        return
    try:
        slots = obj.animation_data.action_suitable_slots
        if slots:
            obj.animation_data.action_slot = slots[0]
            log(f"slot {obj.name} → {obj.animation_data.action_slot.identifier}")
    except Exception as e:
        log(f"slot fail {e}")


def main():
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    fbx_name = args[0] if args else "wave.fbx"
    action_name = args[1] if len(args) > 1 else Path(fbx_name).stem.lower().replace(" ", "_")
    fbx = ROOT / "body_motion" / "source_fbx" / fbx_name
    if not fbx.is_file():
        raise SystemExit(f"missing {fbx}")

    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        raise SystemExit("no SMPL-X_Armature")

    # clear old action same name
    old = bpy.data.actions.get(action_name)
    if old:
        bpy.data.actions.remove(old)

    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(filepath=str(fbx), ignore_leaf_bones=True)
    new = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]
    src = next(o for o in new if o.type == "ARMATURE")
    for o in new:
        if o.type == "MESH":
            bpy.data.objects.remove(o, do_unlink=True)

    assign_slot(src)
    # scale
    bpy.context.view_layer.update()

    def height(arm, mix):
        if mix:
            h, hd = find_bn(arm, "Hips"), find_bn(arm, "Head")
            if not h or not hd:
                return None
            a = arm.matrix_world @ arm.data.bones[h].head_local
            b = arm.matrix_world @ arm.data.bones[hd].head_local
        else:
            a = arm.matrix_world @ arm.data.bones["pelvis"].head_local
            b = arm.matrix_world @ arm.data.bones["head"].head_local
        return (b - a).length

    hs, ht = height(src, True), height(tgt, False)
    if hs and ht:
        s = ht / hs
        src.scale = (s, s, s)
        bpy.context.view_layer.update()
        log(f"scale {s:.3f}")

    # align hips
    h = find_bn(src, "Hips")
    if h:
        sh = src.matrix_world @ src.data.bones[h].head_local
        th = tgt.matrix_world @ tgt.data.bones["pelvis"].head_local
        src.location += th - sh
        bpy.context.view_layer.update()

    # constraints
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="POSE")
    mapped = []
    for mix, sm in BONE_MAP.items():
        if sm not in tgt.pose.bones:
            continue
        bn = find_bn(src, mix)
        if not bn:
            continue
        pb = tgt.pose.bones[sm]
        while pb.constraints:
            pb.constraints.remove(pb.constraints[0])
        c = pb.constraints.new("COPY_ROTATION")
        c.target = src
        c.subtarget = bn
        c.target_space = "WORLD"
        c.owner_space = "WORLD"
        c.mix_mode = "REPLACE"
        pb.rotation_mode = "QUATERNION"
        mapped.append(sm)
    log(f"mapped {len(mapped)} bones")

    # verify source moves
    assign_slot(src)
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    bn = find_bn(src, "RightArm")
    dg = bpy.context.evaluated_depsgraph_get()
    src_e = src.evaluated_get(dg)
    q1 = src_e.pose.bones[bn].matrix.to_quaternion().copy()
    fr = src.animation_data.action.frame_range
    f0, f1 = int(fr[0]), int(fr[1])
    mid = (f0 + f1) // 2
    bpy.context.scene.frame_set(mid)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    src_e = src.evaluated_get(dg)
    q2 = src_e.pose.bones[bn].matrix.to_quaternion().copy()
    log(f"source RightArm moved={q1 != q2} f0={f0} f1={f1}")
    if q1 == q2:
        raise SystemExit("SOURCE NOT ANIMATING — abort")

    # bake
    if not tgt.animation_data:
        tgt.animation_data_create()
    act = bpy.data.actions.new(action_name)
    act.use_fake_user = True
    tgt.animation_data.action = act
    assign_slot(tgt)

    bones = [tgt.pose.bones[n] for n in mapped]
    max_diff = 0.0
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
            # track non-rest
            if pb.name == "right_shoulder":
                max_diff = max(max_diff, abs(rot.x) + abs(rot.y) + abs(rot.z) + abs(1.0 - rot.w))
            pb.location = loc
            pb.rotation_quaternion = rot
            pb.scale = sca
            pb.keyframe_insert("location", frame=f)
            pb.keyframe_insert("rotation_quaternion", frame=f)
            pb.keyframe_insert("scale", frame=f)

    for pb in tgt.pose.bones:
        while pb.constraints:
            pb.constraints.remove(pb.constraints[0])

    # clear constraints done; verify keyed values
    strip = act.layers[0].strips[0]
    cb = strip.channelbags[0]
    diffs = 0
    for fc in cb.fcurves:
        if abs(fc.evaluate(f0) - fc.evaluate(mid)) > 0.02:
            diffs += 1
    log(f"keyed channels changing={diffs} right_shoulder_rest_delta={max_diff:.4f}")
    if diffs < 5:
        raise SystemExit("BAKE STILL REST — abort save")

    # remove source
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(src, do_unlink=True)
    # remove mixamo action orphans optional
    for a in list(bpy.data.actions):
        if "mixamo" in a.name.lower() or "|" in a.name:
            try:
                bpy.data.actions.remove(a)
            except Exception:
                pass

    out = ROOT / "body_motion" / f"_proof_{action_name}.blend"
    if "--out" in args:
        i = args.index("--out")
        out = Path(args[i + 1])
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    log(f"OK saved {out}")


if __name__ == "__main__":
    main()
