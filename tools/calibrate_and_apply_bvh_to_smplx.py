"""
MoMask/HML22 BVH → SMPL-X_Armature with T-pose calibration.

Anti-twist bake (IMPORTANT):
  For each bone, map SOURCE local rotation into TARGET local space using
  rest-pose alignment (same idea as proper retargeters):

    R_tgt_pose_world = R_src_pose_world * R_src_rest_world^{-1} * R_tgt_rest_world

  Then convert to target local quaternion and keyframe.
  This keeps SMPL-X bone roll correct for skinning (no candy-wrap).

Also:
  - uniform scale source so hips→head matches SMPL-X (from T-pose calib)
  - full root travel (pelvis location from Hips)
  - fingers left at rest

Usage:
  blender.exe whole_body_retargeted.blend --python tools/calibrate_and_apply_bvh_to_smplx.py -- ^
    --tpose_bvh "lmm train/momask_tpose.bvh" ^
    --motion_bvh "lmm train/momask_walk.bvh" ^
    --calib body_motion/source_to_smplx_calib.json ^
    --action momask_from_calib ^
    --out body_motion/_momask_ours.blend
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]

BVH_TO_SMPLX = {
    "Hips": "pelvis",
    "LeftUpLeg": "left_hip",
    "LeftLeg": "left_knee",
    "LeftFoot": "left_ankle",
    "LeftToe": "left_foot",
    "RightUpLeg": "right_hip",
    "RightLeg": "right_knee",
    "RightFoot": "right_ankle",
    "RightToe": "right_foot",
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
}

# Hierarchy order on TARGET (parent before child) for local conversion
SMPLX_ORDER = [
    "pelvis",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "spine1",
    "spine2",
    "spine3",
    "neck",
    "head",
    "left_collar",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_collar",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]

EDGES_REPORT = [
    ("Hips", "Head", "pelvis", "head"),
    ("LeftUpLeg", "LeftLeg", "left_hip", "left_knee"),
    ("LeftArm", "LeftForeArm", "left_shoulder", "left_elbow"),
]


def log(msg: str) -> None:
    print(f"[calib_smplx] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--tpose_bvh", default=str(ROOT / "lmm train" / "momask_tpose.bvh"))
    p.add_argument("--motion_bvh", default=str(ROOT / "lmm train" / "momask_walk.bvh"))
    p.add_argument("--calib", default=str(ROOT / "body_motion" / "source_to_smplx_calib.json"))
    p.add_argument("--action", default="momask_from_calib")
    p.add_argument("--out", default=str(ROOT / "body_motion" / "_momask_ours.blend"))
    p.add_argument("--calib_only", action="store_true")
    p.add_argument(
        "--root",
        default="absolute",
        choices=("relative", "absolute", "inplace"),
    )
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
    arm = next(
        bpy.data.objects[k]
        for k in bpy.data.objects.keys()
        if k not in before and bpy.data.objects[k].type == "ARMATURE"
    )
    assign_slot(arm)
    return arm


def clear_pose(arm) -> None:
    if arm.animation_data:
        arm.animation_data.action = None
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.transforms_clear()
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()


def world_head_pose(arm, name: str) -> Vector:
    return (arm.matrix_world @ arm.pose.bones[name].head).copy()


def world_head_rest(arm, name: str) -> Vector:
    return (arm.matrix_world @ arm.data.bones[name].head_local).copy()


def seg_len_rest(arm, a: str, b: str) -> float:
    return (world_head_rest(arm, b) - world_head_rest(arm, a)).length


def height_hips_head(arm, is_bvh: bool) -> float:
    if is_bvh:
        return seg_len_rest(arm, "Hips", "Head")
    return seg_len_rest(arm, "pelvis", "head")


def bone_rest_world_matrix(arm, name: str) -> Matrix:
    """Rest orientation+position of bone in world space."""
    return arm.matrix_world @ arm.data.bones[name].matrix_local


def bone_pose_world_matrix(arm, name: str) -> Matrix:
    return arm.matrix_world @ arm.pose.bones[name].matrix


def calibrate(tpose_src, tgt) -> dict:
    clear_pose(tpose_src)
    clear_pose(tgt)
    bpy.context.view_layer.update()

    h_src0 = height_hips_head(tpose_src, True)
    h_tgt = height_hips_head(tgt, False)
    scale = h_tgt / h_src0
    tpose_src.scale = (
        tpose_src.scale[0] * scale,
        tpose_src.scale[1] * scale,
        tpose_src.scale[2] * scale,
    )
    bpy.context.view_layer.update()
    sh = world_head_rest(tpose_src, "Hips")
    th = world_head_rest(tgt, "pelvis")
    tpose_src.location += th - sh
    bpy.context.view_layer.update()
    log(f"calibrate: H_src0={h_src0:.4f} H_tgt={h_tgt:.4f} scale={scale:.4f}")

    # Rest world rotations for retarget offset (at T-pose after scale/align)
    rest_src = {}
    rest_tgt = {}
    for bvh_n, sm in BVH_TO_SMPLX.items():
        if bvh_n not in tpose_src.data.bones or sm not in tgt.data.bones:
            continue
        Rs = bone_rest_world_matrix(tpose_src, bvh_n).to_3x3().to_4x4()
        Rt = bone_rest_world_matrix(tgt, sm).to_3x3().to_4x4()
        # Store as quaternion lists for JSON
        rest_src[bvh_n] = list(Rs.to_quaternion())
        rest_tgt[sm] = list(Rt.to_quaternion())

    segments = []
    for sa, sb, ta, tb in EDGES_REPORT:
        if sa in tpose_src.data.bones and ta in tgt.data.bones:
            segments.append(
                {
                    "bvh": [sa, sb],
                    "smplx": [ta, tb],
                    "src_len_scaled": round(seg_len_rest(tpose_src, sa, sb), 5),
                    "tgt_len": round(seg_len_rest(tgt, ta, tb), 5),
                }
            )

    calib = {
        "version": 2,
        "description": "BVH→SMPL-X rest-relative retarget (anti-twist)",
        "import_bvh": {"axis_forward": "-Z", "axis_up": "Y", "global_scale": 1.0},
        "bone_map": BVH_TO_SMPLX,
        "height": {
            "measure": "hips_to_head",
            "src_before_scale": round(h_src0, 5),
            "tgt": round(h_tgt, 5),
            "uniform_scale_on_source": round(scale, 6),
        },
        "rest_src_world_quat": rest_src,
        "rest_tgt_world_quat": rest_tgt,
        "segments_tpose": segments,
        "method": "R_tgt = R_src * R_src_rest^{-1} * R_tgt_rest  (world), then to local",
    }
    return calib, scale


def apply_motion(src, tgt, calib: dict, action_name: str, root_mode: str) -> tuple[int, int]:
    scale = float(calib["height"]["uniform_scale_on_source"])
    src.scale = (1.0, 1.0, 1.0)
    src.location = Vector((0, 0, 0))
    src.rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()
    src.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    scene = bpy.context.scene
    assign_slot(src)
    fr = src.animation_data.action.frame_range
    f0, f1 = int(fr[0]), int(fr[1])

    scene.frame_set(f0)
    bpy.context.view_layer.update()
    sh0 = world_head_pose(src, "Hips")
    th0 = world_head_rest(tgt, "pelvis")
    src.location += th0 - sh0
    bpy.context.view_layer.update()

    # Rest world rotations from bind pose (data.bones) — do NOT clear source action
    rest_src_mat: dict[str, Matrix] = {}
    for bvh_n in BVH_TO_SMPLX:
        if bvh_n in src.data.bones:
            rest_src_mat[bvh_n] = bone_rest_world_matrix(src, bvh_n).to_3x3().to_4x4()
    rest_tgt_mat: dict[str, Matrix] = {}
    for sm in BVH_TO_SMPLX.values():
        if sm in tgt.data.bones:
            rest_tgt_mat[sm] = bone_rest_world_matrix(tgt, sm).to_3x3().to_4x4()

    # R_tgt_world = R_src_world * R_src_rest^{-1} * R_tgt_rest
    offset_mat: dict[str, Matrix] = {}
    for bvh_n, sm in BVH_TO_SMPLX.items():
        if bvh_n not in rest_src_mat or sm not in rest_tgt_mat:
            continue
        offset_mat[sm] = rest_src_mat[bvh_n].inverted() @ rest_tgt_mat[sm]

    scene.frame_set(f1)
    bpy.context.view_layer.update()
    travel = (world_head_pose(src, "Hips") - sh0).length
    use_root = root_mode != "inplace"
    log(
        f"apply REST-RELATIVE bake: scale={scale:.4f} frames={f0}-{f1} "
        f"travel≈{travel:.3f} root={root_mode}"
    )

    # Prove source animates
    scene.frame_set(f0)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    q0 = src.evaluated_get(dg).pose.bones["LeftArm"].matrix.to_quaternion().copy()
    scene.frame_set(min(f0 + 30, f1))
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    q1 = src.evaluated_get(dg).pose.bones["LeftArm"].matrix.to_quaternion().copy()
    dsrc = sum(abs(a - b) for a, b in zip(q0, q1))
    log(f"source LeftArm quat delta = {dsrc:.4f}")
    if dsrc < 1e-4:
        log("WARN: source looks static — check action slot")

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

    hips0 = sh0.copy()
    pelvis0_loc = None
    sm_to_bvh = {s: b for b, s in BVH_TO_SMPLX.items() if s in offset_mat and b in src.pose.bones}

    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        src_e = src.evaluated_get(dg)

        hips_w = src.matrix_world @ src_e.pose.bones["Hips"].head
        if root_mode == "inplace":
            hips_w = Vector((hips0.x, hips0.y, hips_w.z))
        elif root_mode == "relative":
            hips_w = hips_w + (th0 - hips0)

        # Reset target
        for pb in tgt.pose.bones:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
            pb.location = Vector((0, 0, 0))
        bpy.context.view_layer.update()

        # Hierarchy order: rest-relative world rotation → matrix_basis (anti-twist)
        for sm in SMPLX_ORDER:
            bvh_n = sm_to_bvh.get(sm)
            if not bvh_n:
                continue
            R_src = (src.matrix_world @ src_e.pose.bones[bvh_n].matrix).to_3x3().to_4x4()
            R_tgt_world = R_src @ offset_mat[sm]

            pb = tgt.pose.bones[sm]
            bpy.context.view_layer.update()
            cur_head = tgt.matrix_world @ pb.head
            mat_w = R_tgt_world.copy()
            mat_w.translation = cur_head

            # Armature-space pose matrix, then matrix_basis vs parent
            mat_pose = tgt.matrix_world.inverted() @ mat_w
            bone = pb.bone
            if pb.parent:
                pre = (
                    pb.parent.matrix
                    @ pb.parent.bone.matrix_local.inverted()
                    @ bone.matrix_local
                )
                basis = pre.inverted() @ mat_pose
            else:
                basis = bone.matrix_local.inverted() @ mat_pose
            _loc, rot, _ = basis.decompose()
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = rot
            pb.location = Vector((0, 0, 0))
            bpy.context.view_layer.update()

        # Root travel after orientations (keep rot, set world head = source hips)
        if use_root:
            pb = tgt.pose.bones["pelvis"]
            mat_w = tgt.convert_space(
                pose_bone=pb, matrix=pb.matrix, from_space="POSE", to_space="WORLD"
            ).copy()
            mat_w.translation = hips_w
            mat_pose = tgt.matrix_world.inverted() @ mat_w
            bone = pb.bone
            basis = bone.matrix_local.inverted() @ mat_pose
            loc, rot, _ = basis.decompose()
            pb.rotation_quaternion = rot
            if root_mode == "relative":
                if pelvis0_loc is None:
                    pelvis0_loc = loc.copy()
                pb.location = loc - pelvis0_loc
            else:
                pb.location = loc
            bpy.context.view_layer.update()

        for sm in SMPLX_ORDER:
            if sm not in tgt.pose.bones:
                continue
            pb = tgt.pose.bones[sm]
            pb.rotation_mode = "QUATERNION"
            pb.keyframe_insert("rotation_quaternion", frame=f)
            if sm == "pelvis" and use_root:
                pb.keyframe_insert("location", frame=f)

        if f == f0 or f % 30 == 0 or f == f1:
            pel = world_head_pose(tgt, "pelvis")
            hip = world_head_pose(tgt, "left_hip")
            knee = world_head_pose(tgt, "left_knee")
            sh = world_head_pose(tgt, "left_shoulder")
            wr = world_head_pose(tgt, "left_wrist")
            log(
                f"  f{f}: pelvis=({pel.x:.2f},{pel.y:.2f},{pel.z:.2f}) "
                f"legs_down={knee.z < hip.z} hand_drop={sh.z - wr.z:.3f}"
            )

    scene.frame_set(f0)
    bpy.context.view_layer.update()
    p0 = world_head_pose(tgt, "pelvis").copy()
    scene.frame_set(f1)
    bpy.context.view_layer.update()
    p1 = world_head_pose(tgt, "pelvis").copy()
    log(f"pelvis travel = {(p1 - p0).length:.3f}")

    scene.frame_start = f0
    scene.frame_end = f1
    scene.frame_set(f0)
    return f0, f1


def verify(tgt, f0: int, f1: int) -> dict:
    scene = bpy.context.scene
    out = {"frames": []}
    for f in (f0, min(f0 + 30, f1), min(f0 + 60, f1)):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        hip = world_head_pose(tgt, "left_hip")
        knee = world_head_pose(tgt, "left_knee")
        ank = world_head_pose(tgt, "left_ankle")
        sh = world_head_pose(tgt, "left_shoulder")
        wr = world_head_pose(tgt, "left_wrist")
        pel = world_head_pose(tgt, "pelvis")
        head = world_head_pose(tgt, "head")
        out["frames"].append(
            {
                "f": f,
                "legs_down": bool(knee.z < hip.z and ank.z < knee.z),
                "spine_up": bool(head.z > pel.z + 0.3),
                "hand_drop": round(sh.z - wr.z, 3),
                "pelvis": [round(pel.x, 3), round(pel.y, 3), round(pel.z, 3)],
            }
        )
        log(f"verify f{f}: {out['frames'][-1]}")
    scene.frame_set(f0)
    bpy.context.view_layer.update()
    p0 = world_head_pose(tgt, "pelvis")
    scene.frame_set(f1)
    bpy.context.view_layer.update()
    p1 = world_head_pose(tgt, "pelvis")
    out["pelvis_travel"] = round((p1 - p0).length, 3)
    out["ok"] = all(fr["legs_down"] and fr["spine_up"] for fr in out["frames"]) and out[
        "pelvis_travel"
    ] > 0.5
    return out


def main():
    args = parse_args()
    tpose_path = Path(args.tpose_bvh)
    if not tpose_path.is_absolute():
        tpose_path = ROOT / tpose_path
    motion_path = Path(args.motion_bvh)
    if not motion_path.is_absolute():
        motion_path = ROOT / motion_path
    calib_path = Path(args.calib)
    if not calib_path.is_absolute():
        calib_path = ROOT / calib_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    if not tpose_path.is_file():
        raise SystemExit(f"missing T-pose BVH: {tpose_path}")

    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        raise SystemExit("Need SMPL-X_Armature in the open blend")

    for o in list(bpy.data.objects):
        if o.type == "ARMATURE" and o.name != "SMPL-X_Armature":
            bpy.data.objects.remove(o, do_unlink=True)

    log("=== STEP 1: T-pose calibration ===")
    tpose = import_bvh(tpose_path)
    calib, scale = calibrate(tpose, tgt)
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    calib_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    log(f"wrote {calib_path}")
    bpy.data.objects.remove(tpose, do_unlink=True)

    if args.calib_only:
        return

    if not motion_path.is_file():
        raise SystemExit(f"missing motion BVH: {motion_path}")

    log("=== STEP 2: rest-relative bake (anti-twist) ===")
    src = import_bvh(motion_path)
    f0, f1 = apply_motion(src, tgt, calib, args.action, args.root)
    bpy.data.objects.remove(src, do_unlink=True)

    log("=== STEP 3: verify ===")
    ver = verify(tgt, f0, f1)
    report = {
        "ok": ver.get("ok", False),
        "method": "rest_relative_world",
        "calib": str(calib_path),
        "action": args.action,
        "verify": ver,
        "scale": scale,
    }
    rep = ROOT / "body_motion" / "source_to_smplx_apply_report.json"
    rep.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"report {rep}")

    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
    except RuntimeError:
        alt = out_path.with_name(out_path.stem + "_v2" + out_path.suffix)
        bpy.ops.wm.save_as_mainfile(filepath=str(alt))
        out_path = alt
    log(f"saved {out_path}")
    log(f"DONE ok={report['ok']} action={args.action}")


if __name__ == "__main__":
    main()
