"""
MoMask Visualization → Mixamo retarget — EXACT steps from
https://github.com/EricGuo5513/momask-codes  (Visualization / Retargeting)

  1. KeeMap installed in Blender
  2. Import motion .bvh + Mixamo character .fbx (T-Pose with skeleton)
  3. Select source + destination skeletons
  4. Pose Mode → KeeMapRig
  5. Bone mapping = third_party/momask-codes/assets/mapping.json
     (or mapping6.json if mapping.json fails)
  6. Read In Bone Mapping File
  7. Set Number of Samples, Source Rig, Destination Rig Name
  8. Transfer Animation from Source Destination

NO custom arm fixes, NO swing-aim, NO SMPL-X hop.
Only official KeeMap + official mapping.json.

Usage:
  blender.exe --background --python tools/momask_keemap_exact.py -- ^
    --bvh "lmm train/momask_walk.bvh" ^
    --mixamo "body_motion/source_fbx/Idle.fbx" ^
    --map "third_party/momask-codes/assets/mapping.json" ^
    --out "body_motion/_momask_keemap_exact.blend"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
KEEMAP_PARENT = ROOT / "third_party" / "keemap"
DEFAULT_MAP = ROOT / "third_party" / "momask-codes" / "assets" / "mapping.json"
DEFAULT_MAP6 = ROOT / "third_party" / "momask-codes" / "assets" / "mapping6.json"


def log(msg: str) -> None:
    print(f"[momask_exact] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="MoMask README KeeMap→Mixamo (exact)")
    p.add_argument("--bvh", default=str(ROOT / "lmm train" / "momask_walk.bvh"))
    p.add_argument(
        "--mixamo",
        default=str(ROOT / "body_motion" / "source_fbx" / "Idle.fbx"),
        help="Mixamo character T-Pose FBX with skeleton (as README)",
    )
    p.add_argument("--map", default=str(DEFAULT_MAP))
    p.add_argument(
        "--out",
        default=str(ROOT / "body_motion" / "_momask_keemap_exact.blend"),
    )
    p.add_argument(
        "--export_fbx",
        default=str(ROOT / "body_motion" / "source_fbx" / "momask_keemap_exact.fbx"),
    )
    return p.parse_args(argv)


def enable_keemap() -> None:
    """Step 1: install/register KeeMap (from third_party like README)."""
    if str(KEEMAP_PARENT) not in sys.path:
        sys.path.insert(0, str(KEEMAP_PARENT))
    import KeeMapAnimRetarget

    try:
        KeeMapAnimRetarget.unregister()
    except Exception:
        pass
    KeeMapAnimRetarget.register()
    log("STEP1 KeeMap registered")


def assign_slot(obj) -> None:
    if not obj.animation_data or not obj.animation_data.action:
        return
    try:
        slots = obj.animation_data.action_suitable_slots
        if slots:
            obj.animation_data.action_slot = slots[0]
    except Exception:
        pass


def world_head(arm, name: str) -> Vector:
    return arm.matrix_world @ arm.pose.bones[name].head


def find_bone(arm, *cands: str) -> str | None:
    for c in cands:
        if c in arm.pose.bones:
            return c
    return None


def main():
    args = parse_args()
    bvh = Path(args.bvh) if Path(args.bvh).is_absolute() else ROOT / args.bvh
    mixamo = Path(args.mixamo) if Path(args.mixamo).is_absolute() else ROOT / args.mixamo
    map_path = Path(args.map) if Path(args.map).is_absolute() else ROOT / args.map
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    export_fbx = (
        Path(args.export_fbx)
        if Path(args.export_fbx).is_absolute()
        else ROOT / args.export_fbx
    )

    if not bvh.is_file():
        raise SystemExit(f"BVH missing: {bvh}")
    if not mixamo.is_file():
        raise SystemExit(f"Mixamo FBX missing: {mixamo}")
    if not map_path.is_file():
        raise SystemExit(f"mapping missing: {map_path}")

    # Fresh empty scene
    bpy.ops.wm.read_homefile(use_empty=True)
    enable_keemap()

    # --- STEP 2: Import motion (.bvh) and character (.fbx) ---
    log(f"STEP2 import BVH {bvh.name}")
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
    fr = src.animation_data.action.frame_range
    f0, f1 = int(fr[0]), int(fr[1])
    n_frames = f1 - f0 + 1
    log(f"  source={src.name} frames={f0}-{f1} (samples={n_frames})")

    log(f"STEP2 import Mixamo character {mixamo.name}")
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.fbx(
        filepath=str(mixamo),
        ignore_leaf_bones=False,  # keep Head / toes like full Mixamo skeleton
        automatic_bone_orientation=False,
    )
    dst = next(
        bpy.data.objects[k]
        for k in bpy.data.objects.keys()
        if k not in before and bpy.data.objects[k].type == "ARMATURE"
    )
    # Disconnect any imported clip action so KeeMap writes clean keys
    # (README: character is T-Pose with skeleton; do not need Rest Position for select)
    if dst.animation_data and dst.animation_data.action:
        dst.animation_data.action = None
    log(f"  dest={dst.name} bones={len(dst.data.bones)} Head={'mixamorig:Head' in dst.data.bones}")

    # --- STEP 3–4: Select both, Pose Mode (KeeMapRig panel) ---
    log("STEP3-4 select source+dest, Pose Mode")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    dst.select_set(True)
    bpy.context.view_layer.objects.active = dst
    bpy.ops.object.mode_set(mode="POSE")

    # --- STEP 5–7: mapping file, Read In, Samples / Source / Dest ---
    scene = bpy.context.scene
    if not hasattr(scene, "keemap_settings"):
        raise SystemExit("keemap_settings missing — KeeMap not registered")

    KeeMap = scene.keemap_settings
    KeeMap.bone_mapping_file = str(map_path)
    log(f"STEP5 bone mapping file = {map_path}")
    bpy.ops.wm.keemap_read_file()
    log("STEP5 Read In Bone Mapping File — done")

    KeeMap.source_rig_name = src.name
    KeeMap.destination_rig_name = dst.name
    KeeMap.start_frame_to_apply = f0
    KeeMap.number_of_frames_to_apply = n_frames
    KeeMap.keyframe_every_n_frames = 1
    # mapping.json uses EULER corrections from MoMask authors
    if hasattr(KeeMap, "bone_rotation_mode"):
        KeeMap.bone_rotation_mode = "EULER"

    bone_list = scene.keemap_bone_mapping_list
    valid = 0
    missing = []
    for item in bone_list:
        sb, db = item.SourceBoneName, item.DestinationBoneName
        ok = sb in src.pose.bones and db in dst.pose.bones
        if ok:
            valid += 1
        else:
            missing.append(f"{sb}->{db}")
    log(
        f"STEP7 Source Rig={src.name} Dest Rig={dst.name} "
        f"Samples={n_frames} mapped_pairs={valid}/{len(bone_list)}"
    )
    if missing:
        log(f"  missing pairs (skipped by KeeMap): {missing[:8]}")
    if valid < 8:
        # README: try mapping6.json if mapping.json doesn't work
        if map_path.resolve() != DEFAULT_MAP6.resolve() and DEFAULT_MAP6.is_file():
            log(f"too few pairs ({valid}) — retry with mapping6.json")
            KeeMap.bone_mapping_file = str(DEFAULT_MAP6)
            bpy.ops.wm.keemap_read_file()
            KeeMap.source_rig_name = src.name
            KeeMap.destination_rig_name = dst.name
            KeeMap.start_frame_to_apply = f0
            KeeMap.number_of_frames_to_apply = n_frames
            bone_list = scene.keemap_bone_mapping_list
            valid = sum(
                1
                for item in bone_list
                if item.SourceBoneName in src.pose.bones
                and item.DestinationBoneName in dst.pose.bones
            )
            log(f"  mapping6 valid pairs={valid}/{len(bone_list)}")
            map_path = DEFAULT_MAP6
        if valid < 8:
            raise SystemExit(f"too few valid bone pairs ({valid})")

    # --- STEP 8: Transfer Animation from Source Destination ---
    log("STEP8 Transfer Animation from Source Destination …")
    bpy.ops.wm.perform_animation_transfer()
    log("STEP8 transfer finished")

    bpy.ops.object.mode_set(mode="OBJECT")
    assign_slot(dst)

    # Basic verify (README quality: feet/legs)
    def snap(frame: int) -> dict:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        hip_n = find_bone(dst, "mixamorig:LeftUpLeg", "LeftUpLeg")
        knee_n = find_bone(dst, "mixamorig:LeftLeg", "LeftLeg")
        foot_n = find_bone(dst, "mixamorig:LeftFoot", "LeftFoot")
        arm_n = find_bone(dst, "mixamorig:LeftArm", "LeftArm")
        hand_n = find_bone(dst, "mixamorig:LeftHand", "LeftHand")
        hips_n = find_bone(dst, "mixamorig:Hips", "Hips")
        hip, knee, foot = world_head(dst, hip_n), world_head(dst, knee_n), world_head(dst, foot_n)
        arm, hand = world_head(dst, arm_n), world_head(dst, hand_n)
        hips = world_head(dst, hips_n)
        return {
            "frame": frame,
            "legs_down": bool(knee.z < hip.z + 0.05 and foot.z < knee.z + 0.05),
            "hand_drop": round(arm.z - hand.z, 3),
            "hips": [round(hips.x, 3), round(hips.y, 3), round(hips.z, 3)],
            "LeftArm_q": [
                round(float(x), 3)
                for x in dst.pose.bones[arm_n].rotation_quaternion
            ],
        }

    c1 = snap(f0)
    c2 = snap(min(f0 + 30, f1))
    # KeeMap keys rotation_euler (mapping.json bone_rotation_mode=EULER) — compare matrices
    arm_n = find_bone(dst, "mixamorig:LeftArm", "LeftArm")
    scene.frame_set(f0)
    bpy.context.view_layer.update()
    m0 = dst.pose.bones[arm_n].matrix.copy()
    scene.frame_set(min(f0 + 30, f1))
    bpy.context.view_layer.update()
    m1 = dst.pose.bones[arm_n].matrix.copy()
    arm_delta = sum(abs(a - b) for a, b in zip(m0.to_quaternion(), m1.to_quaternion()))
    e0 = list(dst.pose.bones[arm_n].rotation_euler)
    scene.frame_set(f0)
    bpy.context.view_layer.update()
    e_f0 = list(dst.pose.bones[arm_n].rotation_euler)
    scene.frame_set(min(f0 + 30, f1))
    bpy.context.view_layer.update()
    e_f30 = list(dst.pose.bones[arm_n].rotation_euler)
    c1["LeftArm_euler"] = [round(float(x), 3) for x in e_f0]
    c2["LeftArm_euler"] = [round(float(x), 3) for x in e_f30]
    log(f"verify f{f0}: {c1}")
    log(f"verify f+30: {c2}")
    log(f"LeftArm matrix-quat delta={arm_delta:.4f}")

    scene.frame_set(f0)
    bpy.context.view_layer.update()
    p0 = world_head(dst, find_bone(dst, "mixamorig:Hips", "Hips")).copy()
    scene.frame_set(f1)
    bpy.context.view_layer.update()
    p1 = world_head(dst, find_bone(dst, "mixamorig:Hips", "Hips")).copy()
    travel = (p1 - p0).length

    # Save blend (view Mixamo character + animation)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.frame_start = f0
    scene.frame_end = f1
    scene.frame_set(f0)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    log(f"saved blend {out}")

    # Optional FBX of Mixamo only (for later catalog hop if you want)
    bpy.ops.object.select_all(action="DESELECT")
    dst.select_set(True)
    for o in bpy.data.objects:
        if o.parent == dst:
            o.select_set(True)
    bpy.context.view_layer.objects.active = dst
    export_fbx.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(export_fbx),
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_space_transform=False,
        path_mode="COPY",
    )
    log(f"exported FBX {export_fbx}")

    report = {
        "ok": bool(c1["legs_down"] and c2["legs_down"] and arm_delta > 1e-4),
        "method": "MoMask README KeeMap exact (no custom fixes)",
        "readme": "https://github.com/EricGuo5513/momask-codes#retargeting",
        "bvh": str(bvh),
        "mixamo": str(mixamo),
        "map": str(map_path),
        "source_rig": src.name,
        "destination_rig": dst.name,
        "frames": [f0, f1],
        "samples": n_frames,
        "valid_pairs": valid,
        "travel": round(travel, 3),
        "arm_delta": round(arm_delta, 4),
        "checks": [c1, c2],
        "blend": str(out),
        "fbx": str(export_fbx),
        "how_to_view": "Open blend, select Mixamo Armature, play timeline",
    }
    rep = ROOT / "body_motion" / "momask_keemap_exact_report.json"
    rep.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"report {rep}")
    log(f"DONE ok={report['ok']} travel={travel:.3f} legs={c1['legs_down']}/{c2['legs_down']}")
    if not report["ok"]:
        raise SystemExit("KeeMap transfer checks failed")


if __name__ == "__main__":
    main()
