"""
MoMask BVH → Mixamo character retarget (official MoMask visualization path).

Uses the same helper-bone + COPY_ROTATION + visual-bake approach as
apply_lmm_npz_to_smplx.py but targets Mixamo bones instead of SMPL-X.
MoMask BVH and Mixamo share the same bone-name family, so no twist.

Usage:
  blender.exe --background --python tools/apply_bvh_to_mixamo.py -- ^
    --fbx "body_motion/source_fbx/Idle.fbx" ^
    --bvh "lmm train/momask_walk.bvh" ^
    --action moonwalk ^
    --out "body_motion/_moonwalk_mixamo.blend"
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Vector

ROOT        = Path(__file__).resolve().parents[1]
HELPER_SFXS = "_MIX_H"

# MoMask BVH bone name → Mixamo bone name (with or without prefix)
BVH_TO_MIXAMO: dict[str, str] = {
    "Hips":         "mixamorig:Hips",
    "LeftUpLeg":    "mixamorig:LeftUpLeg",
    "LeftLeg":      "mixamorig:LeftLeg",
    "LeftFoot":     "mixamorig:LeftFoot",
    "LeftToe":      "mixamorig:LeftToeBase",
    "RightUpLeg":   "mixamorig:RightUpLeg",
    "RightLeg":     "mixamorig:RightLeg",
    "RightFoot":    "mixamorig:RightFoot",
    "RightToe":     "mixamorig:RightToeBase",
    "Spine":        "mixamorig:Spine",
    "Spine1":       "mixamorig:Spine1",
    "Spine2":       "mixamorig:Spine2",
    "Neck":         "mixamorig:Neck",
    "Head":         "mixamorig:Head",
    "LeftShoulder": "mixamorig:LeftShoulder",
    "LeftArm":      "mixamorig:LeftArm",
    "LeftForeArm":  "mixamorig:LeftForeArm",
    "LeftHand":     "mixamorig:LeftHand",
    "RightShoulder":"mixamorig:RightShoulder",
    "RightArm":     "mixamorig:RightArm",
    "RightForeArm": "mixamorig:RightForeArm",
    "RightHand":    "mixamorig:RightHand",
}


def log(msg: str) -> None:
    print(f"[bvh_mixamo] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--fbx",    default="", help="Mixamo FBX (optional if blend already has Mixamo armature)")
    p.add_argument("--bvh",    required=True, help="MoMask BVH file")
    p.add_argument("--action", default="moonwalk")
    p.add_argument("--out",    required=True)
    p.add_argument("--root",   default="absolute", choices=("absolute", "inplace"))
    return p.parse_args(argv)


def resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (ROOT / pp).resolve()


def assign_slot(obj) -> None:
    if not obj.animation_data or not obj.animation_data.action:
        return
    try:
        slots = obj.animation_data.action_suitable_slots
        if slots:
            obj.animation_data.action_slot = slots[0]
    except Exception:
        pass


def find_mixamo_arm(exclude_name: str) -> bpy.types.Object | None:
    """Find the Mixamo armature (any armature with mixamorig: bones)."""
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE" or obj.name == exclude_name:
            continue
        if any("mixamorig" in b.name for b in obj.pose.bones):
            return obj
        if any(b.name in ("Hips", "LeftUpLeg", "Spine") for b in obj.pose.bones):
            return obj
    return None


def import_fbx(fbx_path: Path):
    before = {o.name for o in bpy.data.objects}
    bpy.ops.import_scene.fbx(
        filepath=str(fbx_path),
        use_anim=False,          # skeleton only (T-pose), no clip
        global_scale=1.0,
        axis_forward="-Z",
        axis_up="Y",
    )
    new_arms = [o for o in bpy.data.objects
                if o.name not in before and o.type == "ARMATURE"]
    if new_arms:
        arm = new_arms[0]
        log(f"imported Mixamo FBX: {arm.name}  bones={len(arm.pose.bones)}")
        return arm
    return find_mixamo_arm("")


def import_bvh(bvh_path: Path):
    before = {o.name for o in bpy.data.objects}
    bpy.ops.import_anim.bvh(
        filepath=str(bvh_path),
        axis_forward="-Z",
        axis_up="Y",
        target="ARMATURE",
        global_scale=1.0,
        frame_start=1,
        use_fps_scale=True,
        update_scene_fps=True,
        update_scene_duration=True,
        use_cyclic=False,
        rotate_mode="NATIVE",
    )
    new_arms = [o for o in bpy.data.objects
                if o.name not in before and o.type == "ARMATURE"]
    if not new_arms:
        raise RuntimeError("BVH import produced no armature")
    src = new_arms[0]
    assign_slot(src)
    if src.animation_data and src.animation_data.action:
        fr = src.animation_data.action.frame_range
        log(f"BVH: {src.name}  frames={int(fr[0])}–{int(fr[1])}")
    return src


def legs_point_up(src) -> bool:
    bpy.context.view_layer.update()
    try:
        hip   = src.matrix_world @ src.pose.bones["LeftUpLeg"].head
        knee  = src.matrix_world @ src.pose.bones["LeftLeg"].head
        ankle = src.matrix_world @ src.pose.bones["LeftFoot"].head
    except KeyError:
        return False
    up = knee.z > hip.z + 0.05 or ankle.z > hip.z + 0.05
    log(f"leg check hip.z={hip.z:.3f} knee.z={knee.z:.3f} ank.z={ankle.z:.3f} inverted={up}")
    return up


def fix_orientation(src):
    scene = bpy.context.scene
    if src.animation_data and src.animation_data.action:
        scene.frame_set(int(src.animation_data.action.frame_range[0]))
    if not legs_point_up(src):
        log("legs OK after BVH import")
        return
    for axis in ("x", "z", "y"):
        bpy.ops.object.select_all(action="DESELECT")
        src.select_set(True)
        bpy.context.view_layer.objects.active = src
        e = list(src.rotation_euler)
        if axis == "x":   src.rotation_euler = (e[0] + math.pi, e[1], e[2])
        elif axis == "z": src.rotation_euler = (e[0], e[1], e[2] + math.pi)
        else:             src.rotation_euler = (e[0], e[1] + math.pi, e[2])
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        bpy.context.view_layer.update()
        if not legs_point_up(src):
            log(f"orientation fixed with 180° {axis.upper()}")
            return
    log("WARN: could not auto-fix inverted legs")


def resolve_mixamo_bone(tgt, bvh_name: str) -> str | None:
    """Find the Mixamo bone name that corresponds to bvh_name."""
    # Try direct mapping first
    candidate = BVH_TO_MIXAMO.get(bvh_name)
    if candidate and candidate in tgt.pose.bones:
        return candidate
    # Try without prefix
    if candidate:
        short = candidate.replace("mixamorig:", "")
        if short in tgt.pose.bones:
            return short
    # Try bvh_name directly
    if bvh_name in tgt.pose.bones:
        return bvh_name
    return None


def build_helpers(src, tgt, pairs: list[tuple[str, str]]) -> None:
    """Create helper bones on BVH armature with Mixamo's rest head/tail/roll."""
    mw_inv = src.matrix_world.inverted()

    bpy.ops.object.select_all(action="DESELECT")
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="EDIT")
    transforms: dict[str, tuple] = {}
    for eb in tgt.data.edit_bones:
        h = mw_inv @ (tgt.matrix_world @ eb.head)
        t = mw_inv @ (tgt.matrix_world @ eb.tail)
        transforms[eb.name] = (h.copy(), t.copy(), eb.roll)
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    bpy.context.view_layer.objects.active = src
    bpy.ops.object.mode_set(mode="EDIT")

    n = 0
    for sb, sm in pairs:
        par = src.data.edit_bones.get(sb)
        if not par or sm not in transforms:
            continue
        hname = sm + HELPER_SFXS
        if hname in src.data.edit_bones:
            src.data.edit_bones.remove(src.data.edit_bones[hname])
        h, t, roll = transforms[sm]
        nb = src.data.edit_bones.new(hname)
        nb.head = h; nb.tail = t
        if (nb.tail - nb.head).length < 1e-5:
            nb.tail = nb.head + Vector((0, 0.05, 0))
        nb.roll = roll
        nb.parent = par
        nb.use_connect = False
        n += 1
    bpy.ops.object.mode_set(mode="OBJECT")
    log(f"helper bones: {n}")


def clear_constraints(pb) -> None:
    while pb.constraints:
        pb.constraints.remove(pb.constraints[0])


def retarget_and_bake(src, tgt, action_name: str, root_mode: str) -> tuple[int, int]:
    scene = bpy.context.scene

    # Scale source to match target height in WORLD space (Mixamo FBX has ~0.01
    # object scale — local bone lengths are ~100× world and must not be used raw).
    try:
        bpy.context.view_layer.update()
        sh_n = "Hips" if "Hips" in src.data.bones else None
        sd_n = "Head" if "Head" in src.data.bones else None
        th_n = next(
            (n for n in ("mixamorig:Hips", "Hips") if n in tgt.data.bones), None
        )
        td_n = next(
            (n for n in ("mixamorig:Head", "Head") if n in tgt.data.bones), None
        )
        if sh_n and sd_n and th_n and td_n:
            hs = (
                src.matrix_world @ src.data.bones[sd_n].head_local
                - src.matrix_world @ src.data.bones[sh_n].head_local
            ).length
            ht = (
                tgt.matrix_world @ tgt.data.bones[td_n].head_local
                - tgt.matrix_world @ tgt.data.bones[th_n].head_local
            ).length
            if hs > 1e-6 and ht > 1e-6:
                s = ht / hs
                src.scale = (s * src.scale[0], s * src.scale[1], s * src.scale[2])
                bpy.context.view_layer.update()
                log(f"scale source × {s:.4f} (world hips→head src={hs:.3f} tgt={ht:.3f})")
    except Exception as e:
        log(f"scale skipped: {e}")

    if not src.animation_data or not src.animation_data.action:
        raise RuntimeError("BVH has no action")
    assign_slot(src)
    f0 = int(src.animation_data.action.frame_range[0])
    f1 = int(src.animation_data.action.frame_range[1])
    scene.frame_set(f0)
    bpy.context.view_layer.update()

    # Verify animation is live
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        se = src.evaluated_get(dg)
        q0 = se.pose.bones["LeftUpLeg"].matrix.to_quaternion()
        scene.frame_set(min(f0 + 30, f1))
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        se = src.evaluated_get(dg)
        q1 = se.pose.bones["LeftUpLeg"].matrix.to_quaternion()
        d = sum(abs(a - b) for a, b in zip(q0, q1))
        log(f"LeftUpLeg quat delta f0→f30 = {d:.4f}")
        scene.frame_set(f0)
        bpy.context.view_layer.update()
    except Exception as e:
        log(f"motion check: {e}")

    # Align source Hips to target Hips
    try:
        for hbvh in ("Hips",):
            for hmix in ("mixamorig:Hips", "Hips"):
                if hbvh in src.pose.bones and hmix in tgt.pose.bones:
                    scene.frame_set(f0)
                    bpy.context.view_layer.update()
                    sh = src.matrix_world @ src.pose.bones[hbvh].head
                    th = tgt.matrix_world @ tgt.pose.bones[hmix].head
                    src.location += th - sh
                    bpy.context.view_layer.update()
                    log("aligned source Hips → target Hips")
                    break
    except Exception as e:
        log(f"align skipped: {e}")

    fix_orientation(src)

    # Build pairs
    pairs: list[tuple[str, str]] = []
    for bvh_name in BVH_TO_MIXAMO:
        if bvh_name not in src.pose.bones:
            continue
        mix_name = resolve_mixamo_bone(tgt, bvh_name)
        if mix_name:
            pairs.append((bvh_name, mix_name))
        else:
            log(f"  skip {bvh_name} (no match in target)")
    log(f"bone pairs: {len(pairs)}")

    build_helpers(src, tgt, pairs)

    # Add constraints
    bpy.ops.object.select_all(action="DESELECT")
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="POSE")

    for pb in tgt.pose.bones:
        clear_constraints(pb)
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.select = False

    mapped = 0
    root_mix = None
    for bvh_name, mix_name in pairs:
        hname = mix_name + HELPER_SFXS
        if hname not in src.pose.bones:
            continue
        pb = tgt.pose.bones[mix_name]
        cr = pb.constraints.new("COPY_ROTATION")
        cr.name = "MIX_COPY_ROT"
        cr.target = src
        cr.subtarget = hname
        cr.mix_mode = "REPLACE"
        cr.target_space = "WORLD"
        cr.owner_space  = "WORLD"
        if bvh_name == "Hips":
            root_mix = mix_name
            cl = pb.constraints.new("COPY_LOCATION")
            cl.name = "MIX_COPY_LOC"
            cl.target = src
            cl.subtarget = "Hips"
            cl.target_space = "WORLD"
            cl.owner_space  = "WORLD"
        pb.select = True
        mapped += 1
    log(f"constraints: {mapped} bones  bake {f0}–{f1}")

    # Bake
    if not tgt.animation_data:
        tgt.animation_data_create()
    old = bpy.data.actions.get(action_name)
    if old:
        if tgt.animation_data.action == old:
            tgt.animation_data.action = None
        bpy.data.actions.remove(old)
    act = bpy.data.actions.new(action_name)
    act.use_fake_user = True
    tgt.animation_data.action = act
    assign_slot(tgt)

    bones_to_bake = [tgt.pose.bones[m] for _, m in pairs if m in tgt.pose.bones]
    pelvis0 = None

    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        arm_e = tgt.evaluated_get(dg)
        for pb in bones_to_bake:
            pb_e = arm_e.pose.bones[pb.name]
            mat_local = tgt.convert_space(
                pose_bone=pb,
                matrix=pb_e.matrix,
                from_space="POSE", to_space="LOCAL"
            )
            loc, rot, _ = mat_local.decompose()
            if pb.name == root_mix:
                if pelvis0 is None and root_mode == "absolute":
                    pass  # keep full translation
                elif pelvis0 is None:
                    pelvis0 = loc.copy()
                if root_mode != "absolute" and pelvis0 is not None:
                    loc = loc - pelvis0
                pb.location = loc
                pb.keyframe_insert("location", frame=f)
            pb.rotation_quaternion = rot
            pb.keyframe_insert("rotation_quaternion", frame=f)

    for pb in tgt.pose.bones:
        clear_constraints(pb)
        pb.select = False
    tgt.animation_data.action = act
    assign_slot(tgt)

    # Verify
    scene.frame_set(f0)
    bpy.context.view_layer.update()
    try:
        for hmix in ("mixamorig:LeftUpLeg", "LeftUpLeg"):
            if hmix in tgt.pose.bones:
                q0 = tgt.pose.bones[hmix].rotation_quaternion.copy()
                scene.frame_set((f0 + f1) // 2)
                bpy.context.view_layer.update()
                q1 = tgt.pose.bones[hmix].rotation_quaternion.copy()
                d = sum(abs(a - b) for a, b in zip(q0, q1))
                log(f"LeftUpLeg quat delta (baked) f{f0}→f{(f0+f1)//2} = {d:.4f}  (>0.1=motion OK)")
                break
    except Exception:
        pass

    scene.frame_start = f0
    scene.frame_end   = f1
    scene.frame_set(f0)
    log(f"action '{act.name}' baked: {f0}-{f1}")
    return f0, f1


def main():
    args = parse_args()
    bvh_path = resolve(args.bvh)
    out_path  = resolve(args.out)

    if not bvh_path.exists():
        raise SystemExit(f"BVH not found: {bvh_path}")

    # The blend file is already open (passed as --background <file>).
    # Find the Mixamo armature in the current scene.
    tgt = find_mixamo_arm("")
    if not tgt:
        # Try importing FBX if --fbx was given and no Mixamo arm found
        fbx_path_str = getattr(args, "fbx", None)
        if fbx_path_str:
            fbx_path = resolve(fbx_path_str)
            if fbx_path.exists():
                log(f"Importing FBX: {fbx_path.name}")
                tgt = import_fbx(fbx_path)
    if not tgt:
        raise SystemExit("No Mixamo armature found in scene")
    log(f"Mixamo armature: {tgt.name}  bones={len(tgt.pose.bones)}")

    log(f"BVH: {bvh_path.name}")
    src = import_bvh(bvh_path)

    retarget_and_bake(src, tgt, args.action, args.root)

    # Remove BVH armature (keep only Mixamo)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(src, do_unlink=True)
    log("removed BVH armature")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
        log(f"saved → {out_path}")
    except RuntimeError as e:
        alt = out_path.with_name(out_path.stem + "_v2" + out_path.suffix)
        bpy.ops.wm.save_as_mainfile(filepath=str(alt))
        log(f"saved (fallback) → {alt}")
    log("DONE")


if __name__ == "__main__":
    main()
