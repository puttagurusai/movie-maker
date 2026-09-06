"""
Direct MoMask BVH → SMPL-X_Armature (manual Rokoko parity).

Default path = what works by hand (NO KeeMap / NO Mixamo / NO auto-scale):
  import BVH (global_scale=1)
  → bone map + helper bones (Rokoko style)
  → COPY_ROTATION (+ pelvis COPY_LOCATION)
  → bake Action
  → save Action-only slim .blend

Experimental extras (floor plant, knock-knee, rest pads, auto-scale) are OFF
by default. See docs/RETARGET_EXTRAS_ARCHIVE.md to re-enable for research.

Usage:
  blender.exe whole_body_retargeted.blend --background \\
    --python tools/bvh_rokoko_direct_to_smplx.py -- \\
    --bvh path/to/sample_ik.bvh \\
    --action momask_xxx \\
    --out body_motion/momask_cache/momask_xxx.blend \\
    --rokoko_map "our modified bhv mapping to smplx.json"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
FINAL_MAP_PATH = ROOT / "body_motion" / "FINAL_BONE_MAP.json"
DEFAULT_ROKOKO = ROOT / "our modified bhv mapping to smplx.json"
PIPELINE_REST_JSON = ROOT / "body_motion" / "pipeline_rest_pose.json"
HELPER_SUFFIX = "_RSL_H"
RETARGET_ID = "_BVH_RSL"

# Rokoko custom-name part → Mixamo/BVH short bone (same hierarchy as MoMask template)
ROKOKO_PART_TO_BVH = {
    "hips": "Hips",
    "spine": "Spine",
    "chest": "Spine1",
    "upperChest": "Spine2",
    "neck": "Neck",
    "head": "Head",
    "leftShoulder": "LeftShoulder",
    "rightShoulder": "RightShoulder",
    "leftUpperArm": "LeftArm",
    "rightUpperArm": "RightArm",
    "leftLowerArm": "LeftForeArm",
    "rightLowerArm": "RightForeArm",
    "leftHand": "LeftHand",
    "rightHand": "RightHand",
    "leftUpperLeg": "LeftUpLeg",
    "rightUpperLeg": "RightUpLeg",
    "leftLowerLeg": "LeftLeg",
    "rightLowerLeg": "RightLeg",
    "leftFoot": "LeftFoot",
    "rightFoot": "RightFoot",
    "leftToe": "LeftToe",
    "rightToe": "RightToe",
    "leftThumbProximal": "LeftHandThumb1",
    "leftThumbMedial": "LeftHandThumb2",
    "leftThumbDistal": "LeftHandThumb3",
    "leftIndexProximal": "LeftHandIndex1",
    "leftIndexMedial": "LeftHandIndex2",
    "leftIndexDistal": "LeftHandIndex3",
    "leftMiddleProximal": "LeftHandMiddle1",
    "leftMiddleMedial": "LeftHandMiddle2",
    "leftMiddleDistal": "LeftHandMiddle3",
    "leftRingProximal": "LeftHandRing1",
    "leftRingMedial": "LeftHandRing2",
    "leftRingDistal": "LeftHandRing3",
    "leftLittleProximal": "LeftHandPinky1",
    "leftLittleMedial": "LeftHandPinky2",
    "leftLittleDistal": "LeftHandPinky3",
    "rightThumbProximal": "RightHandThumb1",
    "rightThumbMedial": "RightHandThumb2",
    "rightThumbDistal": "RightHandThumb3",
    "rightIndexProximal": "RightHandIndex1",
    "rightIndexMedial": "RightHandIndex2",
    "rightIndexDistal": "RightHandIndex3",
    "rightMiddleProximal": "RightHandMiddle1",
    "rightMiddleMedial": "RightHandMiddle2",
    "rightMiddleDistal": "RightHandMiddle3",
    "rightRingProximal": "RightHandRing1",
    "rightRingMedial": "RightHandRing2",
    "rightRingDistal": "RightHandRing3",
    "rightLittleProximal": "RightHandPinky1",
    "rightLittleMedial": "RightHandPinky2",
    "rightLittleDistal": "RightHandPinky3",
}


def log(msg: str) -> None:
    print(f"[bvh_rokoko] {msg}", flush=True)


def count_bvh_file_frames(path: Path) -> int:
    """Authoritative frame count from the BVH header (not Blender's Action.frame_range)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Frames:"):
            try:
                return max(0, int(s.split(":", 1)[1].strip()))
            except Exception:
                return 0
    return 0


def action_key_range(act) -> tuple[int, int] | None:
    """First/last keyframe actually stored on an Action (classic or layered)."""
    if act is None:
        return None
    lo, hi = None, None

    def _scan(fcus):
        nonlocal lo, hi
        for fcu in fcus or []:
            try:
                kps = fcu.keyframe_points
                if not kps:
                    continue
                a = float(kps[0].co[0])
                b = float(kps[-1].co[0])
                lo = a if lo is None else min(lo, a)
                hi = b if hi is None else max(hi, b)
            except Exception:
                continue

    try:
        _scan(getattr(act, "fcurves", None))
    except Exception:
        pass
    if lo is None:
        try:
            for layer in getattr(act, "layers", None) or []:
                for strip in getattr(layer, "strips", []) or []:
                    bags = getattr(strip, "channelbags", None) or []
                    if not bags:
                        try:
                            slots = getattr(act, "slots", None)
                            if slots and hasattr(strip, "channelbag"):
                                bag = strip.channelbag(slots[0])
                                if bag is not None:
                                    bags = [bag]
                        except Exception:
                            bags = []
                    for bag in bags:
                        _scan(getattr(bag, "fcurves", None))
        except Exception:
            pass
    if lo is None:
        return None
    return int(round(lo)), int(round(hi))


def ensure_layered_action(act, target_obj=None) -> None:
    """
    Blender 5.1 Actions are slotted/layered and start with 0 slots and 0 layers.
    keyframe_insert without a KEYFRAME strip is silently dropped (frozen bake).
    5.1 strips have infinite bounds — no frame_start/frame_end.
    """
    if act is None:
        return
    try:
        if hasattr(act, "slots") and len(act.slots) == 0:
            name = target_obj.name if target_obj is not None else "Slot"
            try:
                act.slots.new("OBJECT", name)
            except Exception:
                act.slots.new(id_type="OBJECT", name=name)
        if hasattr(act, "layers") and len(act.layers) == 0:
            layer = act.layers.new("Layer")
        else:
            layer = act.layers[0] if getattr(act, "layers", None) else None
        if layer is not None and len(getattr(layer, "strips", []) or []) == 0:
            try:
                layer.strips.new(type="KEYFRAME")
            except TypeError:
                layer.strips.new("KEYFRAME")
    except Exception as e:
        log(f"layered action setup skip: {e}")


def expand_layered_action_range(act, f0: int, f1: int) -> None:
    """
    Ensure the Action can hold keys for [f0, f1].

    Blender 5.0: strips expose frame_start/frame_end (default ~25).
    Blender 5.1: KEYFRAME strip is infinite — just create slot+layer+strip.
    """
    if act is None:
        return
    f0, f1 = int(f0), int(max(f0 + 1, f1))
    ensure_layered_action(act)
    try:
        layers = getattr(act, "layers", None)
        if not layers:
            return
        expanded = False
        for layer in layers:
            for strip in getattr(layer, "strips", []) or []:
                if not hasattr(strip, "frame_start"):
                    continue
                try:
                    cur0 = int(getattr(strip, "frame_start", f0) or f0)
                    cur1 = int(getattr(strip, "frame_end", f1) or f1)
                    strip.frame_start = float(min(cur0, f0))
                    strip.frame_end = float(max(cur1, f1 + 1))
                    expanded = True
                except Exception:
                    try:
                        strip.frame_end = float(f1 + 1)
                        expanded = True
                    except Exception:
                        pass
        if expanded:
            log(f"layered Action strip expanded to {f0}..{f1} (avoid 25-frame drop)")
    except Exception as e:
        log(f"strip expand skip: {e}")


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", required=True)
    p.add_argument("--action", default="momask_motion")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--rokoko_map",
        default=str(DEFAULT_ROKOKO),
        help="Rokoko custom-names JSON (our modified bhv mapping to smplx.json)",
    )
    p.add_argument("--root", default="location", choices=("location", "inplace", "auto"))
    p.add_argument("--armature", default="SMPL-X_Armature")
    # Manual Rokoko default: NO auto-scale, NO floor plant. Opt-in only.
    p.add_argument(
        "--auto-scale",
        action="store_true",
        help="Scale BVH armature hips→head to match SMPL-X (OFF = manual parity)",
    )
    p.add_argument(
        "--floor-plant",
        action="store_true",
        help="Opt-in penetration lift (OFF = manual Rokoko)",
    )
    p.add_argument(
        "--no_floor_plant",
        action="store_true",
        help="Keep floor plant off (default)",
    )
    p.add_argument(
        "--foot-lock",
        action="store_true",
        help="SMPL-X contact IK anti-slide after bake (product path passes this)",
    )
    p.add_argument(
        "--no-align",
        action="store_true",
        help="Do not canonicalize travel to SMPL-X forward or apply constant floor",
    )
    p.add_argument(
        "--no_foot_lock",
        action="store_true",
        help="Disable MoMask-style plant detect + IK on SMPL-X after retarget",
    )
    p.add_argument(
        "--shoulder-fwd-deg",
        type=float,
        default=float(os.environ.get("SHOULDER_FWD_DEG", "0")),
        help="Optional shoulder bias degrees (0=off, manual default)",
    )
    p.add_argument(
        "--collar-fwd-deg",
        type=float,
        default=float(os.environ.get("COLLAR_FWD_DEG", "0")),
        help="Optional collar bias degrees (0=off)",
    )
    p.add_argument(
        "--full-scene",
        action="store_true",
        help="Save entire production .blend (heavy). Default: Action-only slim cache.",
    )
    p.add_argument(
        "--no-slim",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return p.parse_args(argv)


def save_action_only_blend(action_name: str, out: Path) -> dict:
    """
    Write a minimal .blend that contains only the named Action (fake user).

    Receiver loads via bpy.data.libraries.load(..., actions=[name]) onto the
    live SMPL-X_Armature in the open scene — meshes/cameras stay in the
    production file, not in the cache.
    """
    act = bpy.data.actions.get(action_name)
    if act is None:
        # fallback: first momask_* action
        for a in bpy.data.actions:
            if a.name == action_name or a.name.startswith("momask_"):
                act = a
                action_name = a.name
                break
    if act is None:
        raise RuntimeError(f"no Action {action_name!r} to slim-save")

    act.use_fake_user = True
    kr = action_key_range(act)
    if kr:
        expand_layered_action_range(act, kr[0], kr[1])
    # Detach so objects can be deleted without losing the Action
    for obj in list(bpy.data.objects):
        try:
            if obj.animation_data and obj.animation_data.action == act:
                obj.animation_data.action = None
        except Exception:
            pass

    # Drop all scene objects (mesh, armature, lights, cameras, …)
    for obj in list(bpy.data.objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass

    # Drop every Action except the one we keep
    for a in list(bpy.data.actions):
        if a.name != act.name:
            try:
                bpy.data.actions.remove(a)
            except Exception:
                pass

    # Strip heavy datablocks (redundant for cache — live scene already has them)
    for coll_name in (
        "meshes",
        "armatures",
        "materials",
        "images",
        "textures",
        "lights",
        "cameras",
        "curves",
        "fonts",
        "lattices",
        "metaballs",
        "volumes",
        "grease_pencils",
        "node_groups",
        "worlds",
        "particles",
    ):
        coll = getattr(bpy.data, coll_name, None)
        if coll is None:
            continue
        for block in list(coll):
            try:
                coll.remove(block)
            except Exception:
                pass

    # Collections cleanup (empty leftovers)
    for c in list(bpy.data.collections):
        try:
            bpy.data.collections.remove(c)
        except Exception:
            pass

    try:
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True, do_linked_ids=True, do_recursive=True
        )
    except Exception:
        pass

    # Re-assert action after purge
    act = bpy.data.actions.get(action_name)
    if act is None and bpy.data.actions:
        act = bpy.data.actions[0]
        action_name = act.name
    if act is None:
        raise RuntimeError("Action lost during slim purge")
    act.use_fake_user = True

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fp = str(out.resolve()).replace("\\", "/")
    try:
        bpy.ops.wm.save_as_mainfile(filepath=fp, compress=True, copy=True)
    except TypeError:
        try:
            bpy.ops.wm.save_as_mainfile(filepath=fp, compress=True)
        except Exception:
            bpy.ops.wm.save_as_mainfile(filepath=fp, check_existing=False)
    except Exception:
        bpy.ops.wm.save_as_mainfile(filepath=fp, check_existing=False)

    size = out.stat().st_size if out.is_file() else 0
    n_fc = 0
    try:
        # classic + layered actions
        if hasattr(act, "fcurves") and act.fcurves:
            n_fc = len(act.fcurves)
        else:
            for sl in getattr(act, "slots", []) or []:
                pass
            # Blender 4.4+ layered: count channels via layers if present
            for layer in getattr(act, "layers", []) or []:
                for strip in getattr(layer, "strips", []) or []:
                    ch = getattr(strip, "channelbag", None) or getattr(strip, "channels", None)
                    if ch is not None and hasattr(ch, "fcurves"):
                        n_fc += len(ch.fcurves)
    except Exception:
        pass

    info = {
        "cache_format": "action_only",
        "action": action_name,
        "path": str(out),
        "size_bytes": size,
        "size_mb": round(size / (1024 * 1024), 3),
        "fcurve_count": n_fc,
        "actions_in_file": [a.name for a in bpy.data.actions],
    }
    log(
        f"slim Action-only cache: {action_name} → {out.name} "
        f"({info['size_mb']} MB, fcurves≈{n_fc})"
    )
    return info


def load_pipeline_rest_quats() -> dict[str, Quaternion]:
    """
    Authoritative custom rest (SMPL-X local wxyz) from pipeline_rest_pose.json.
    No A-pose / T-pose fallback.
    """
    path = PIPELINE_REST_JSON
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"pipeline rest read failed: {e}")
        return {}
    bones = data.get("bones_wxyz") or data.get("pose") or {}
    out: dict[str, Quaternion] = {}
    for name, q in bones.items():
        if isinstance(q, dict):
            q = q.get("quat") or q.get("wxyz") or None
        if not isinstance(q, (list, tuple)) or len(q) < 4:
            continue
        qq = Quaternion((float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        if qq.magnitude < 1e-8:
            continue
        qq.normalize()
        out[str(name)] = qq
    return out


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def apply_custom_rest_pads(
    tgt,
    action,
    f0: int,
    f1: int,
    n_in: int = 8,
    n_out: int = 12,
) -> dict:
    """
    Force start/end of baked Action to pipeline custom rest (SMPL-X space).

    BVH ease pads cannot correctly carry SMPL-X capture quats; re-apply here
    so idle/rest matches the user's captured pose (hands/legs/arms), not A/T.
    """
    rest = load_pipeline_rest_quats()
    if not rest:
        log("custom rest pads SKIP — missing body_motion/pipeline_rest_pose.json")
        return {"ok": False, "reason": "no_pipeline_rest"}
    if not action or "pelvis" not in tgt.pose.bones:
        return {"ok": False}
    if not tgt.animation_data:
        tgt.animation_data_create()
    tgt.animation_data.action = action
    assign_slot(tgt)
    try:
        bpy.ops.object.mode_set(mode="POSE")
    except Exception:
        pass

    bone_names = [n for n in rest if n in tgt.pose.bones]
    if len(bone_names) < 4:
        log(f"custom rest pads SKIP — only {len(bone_names)} matching bones")
        return {"ok": False, "reason": "few_bones"}

    span = max(1, int(f1) - int(f0))
    n_in = max(1, min(int(n_in), span // 3))
    n_out = max(1, min(int(n_out), span // 3))

    def _pose_at(frame: int) -> dict[str, Quaternion]:
        bpy.context.scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        d = {}
        for n in bone_names:
            pb = tgt.pose.bones[n]
            pb.rotation_mode = "QUATERNION"
            d[n] = pb.rotation_quaternion.copy()
        return d

    # Motion samples just inside the pad regions (after/before pure rest holds)
    motion_in = _pose_at(f0 + n_in)
    motion_out = _pose_at(max(f0, f1 - n_out))

    n_keys = 0
    # Intro: full rest → motion
    for i in range(n_in + 1):
        f = f0 + i
        t = _smoothstep(i / max(1, n_in))  # 0=rest, 1=motion
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        for n in bone_names:
            pb = tgt.pose.bones[n]
            pb.rotation_mode = "QUATERNION"
            rq = rest[n]
            mq = motion_in.get(n) or pb.rotation_quaternion
            pb.rotation_quaternion = rq.slerp(mq, t)
            pb.keyframe_insert("rotation_quaternion", frame=f)
            n_keys += 1
    # Outro: motion → full rest
    for i in range(n_out + 1):
        f = f1 - n_out + i
        if f < f0:
            continue
        t = _smoothstep(i / max(1, n_out))  # 0=motion, 1=rest
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        for n in bone_names:
            pb = tgt.pose.bones[n]
            pb.rotation_mode = "QUATERNION"
            rq = rest[n]
            mq = motion_out.get(n) or pb.rotation_quaternion
            pb.rotation_quaternion = mq.slerp(rq, t)
            pb.keyframe_insert("rotation_quaternion", frame=f)
            n_keys += 1

    bpy.context.scene.frame_set(f0)
    log(
        f"custom rest pads OK: bones={len(bone_names)} n_in={n_in} n_out={n_out} "
        f"keys≈{n_keys} from {PIPELINE_REST_JSON.name}"
    )
    return {
        "ok": True,
        "bones": len(bone_names),
        "n_in": n_in,
        "n_out": n_out,
        "path": str(PIPELINE_REST_JSON),
    }


def fix_knock_knees_action(tgt, action, f0: int, f1: int, max_deg: float = 6.0) -> dict:
    """
    Mild SMPL-X hip correction when knee sits medial to hip (inward legs).
    World X relative to pelvis: left should stay +X, right −X vs mid.
    """
    if not action or "left_hip" not in tgt.pose.bones or "right_hip" not in tgt.pose.bones:
        return {"ok": False}
    if not tgt.animation_data:
        tgt.animation_data_create()
    tgt.animation_data.action = action
    assign_slot(tgt)
    max_rad = math.radians(float(max_deg))
    n_fix = 0
    for f in range(int(f0), int(f1) + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        try:
            lh = world_head_pose(tgt, "left_hip")
            rh = world_head_pose(tgt, "right_hip")
            lk = world_head_pose(tgt, "left_knee")
            rk = world_head_pose(tgt, "right_knee")
            mid_x = 0.5 * (lh.x + rh.x)
        except Exception:
            continue
        # Left: positive lateral is +X from mid; right is −X
        l_hip_lat = lh.x - mid_x
        l_kn_lat = lk.x - mid_x
        r_hip_lat = mid_x - rh.x  # positive = right side extent
        r_kn_lat = mid_x - rk.x
        # If knee less lateral than 90% hip → push hip slightly outward (Y axis local)
        for side, hip_lat, kn_lat, bname, sign in (
            ("L", l_hip_lat, l_kn_lat, "left_hip", 1.0),
            ("R", r_hip_lat, r_kn_lat, "right_hip", -1.0),
        ):
            if hip_lat <= 1e-4:
                continue
            if kn_lat >= hip_lat * 0.92:
                continue
            # amount of medial error (meters) → small hip yaw
            err = (hip_lat * 0.95) - kn_lat
            err = max(0.0, min(0.05, float(err)))
            ang = min(max_rad, err * 8.0)  # soft gain
            if ang < 1e-4:
                continue
            pb = tgt.pose.bones[bname]
            pb.rotation_mode = "QUATERNION"
            # Local Y rotation spreads legs on SMPL-X (outward)
            delta = Euler((0.0, sign * ang, 0.0), "XYZ").to_quaternion()
            pb.rotation_quaternion = (pb.rotation_quaternion @ delta).normalized()
            pb.keyframe_insert("rotation_quaternion", frame=f)
            n_fix += 1
    log(f"knock-knee fix: frames_touched≈{n_fix} max_deg={max_deg}")
    return {"ok": True, "n_fix": n_fix}


def apply_shoulder_forward_fix(tgt, action, f0: int, f1: int, shoulder_fwd_deg: float, collar_fwd_deg: float) -> None:
    """
    Pre-multiply local shoulder/collar quats so arms aren't stuck posterior.
    Same axis map as tools/fix_smplx_shoulder_back.py / momask_bvh_to_smplx_fast.
    SMPL-X rest faces −Y in Blender; +X euler on shoulder pulls arm forward.
    """
    if abs(float(shoulder_fwd_deg)) < 0.05 and abs(float(collar_fwd_deg)) < 0.05:
        return
    sf = math.radians(float(shoulder_fwd_deg))
    cf = math.radians(float(collar_fwd_deg))
    fixes = {
        "left_shoulder": Euler((sf, 0.0, -sf * 0.15), "XYZ").to_quaternion(),
        "right_shoulder": Euler((sf, 0.0, sf * 0.15), "XYZ").to_quaternion(),
        "left_collar": Euler((cf * 0.5, 0.0, -cf), "XYZ").to_quaternion(),
        "right_collar": Euler((cf * 0.5, 0.0, cf), "XYZ").to_quaternion(),
    }
    names = [n for n in fixes if n in tgt.pose.bones]
    if not names:
        return
    if not tgt.animation_data:
        tgt.animation_data_create()
    tgt.animation_data.action = action
    assign_slot(tgt)
    try:
        bpy.ops.object.mode_set(mode="POSE")
    except Exception:
        pass
    log(
        f"shoulder forward fix: sh={shoulder_fwd_deg:.1f}° col={collar_fwd_deg:.1f}° "
        f"bones={names} frames={f0}-{f1}"
    )
    for f in range(f0, f1 + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        for bname in names:
            pb = tgt.pose.bones[bname]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = fixes[bname] @ pb.rotation_quaternion
            pb.keyframe_insert("rotation_quaternion", frame=f)
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass


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


def short_name(name: str) -> str:
    return name.split(":")[-1]


def find_src_bone(arm, wanted: str) -> str | None:
    """Find bone on BVH armature: Hips, mixamorig:Hips, LeftToe/LeftToeBase, etc."""
    if wanted in arm.pose.bones:
        return wanted
    w = short_name(wanted)
    aliases = {
        "LeftToeBase": ["LeftToe", "LeftToeBase"],
        "RightToeBase": ["RightToe", "RightToeBase"],
        "LeftToe": ["LeftToe", "LeftToeBase"],
        "RightToe": ["RightToe", "RightToeBase"],
    }
    cands = aliases.get(w, [w])
    for b in arm.pose.bones:
        bn = short_name(b.name)
        if bn == w or bn in cands or b.name in cands:
            return b.name
    return None


def build_bone_map(rokoko_path: Path) -> dict[str, str]:
    """
    FINAL Mixamo→SMPL-X body map + Rokoko custom overrides.
    Keys stored as short BVH names (Hips, LeftArm, …) for direct BVH match.
    """
    final = json.loads(FINAL_MAP_PATH.read_text(encoding="utf-8"))["map"]
    # normalize keys to short names
    bone_map: dict[str, str] = {}
    for k, v in final.items():
        bone_map[short_name(k)] = v
    # LeftToeBase alias
    if "LeftToeBase" in bone_map:
        bone_map.setdefault("LeftToe", bone_map["LeftToeBase"])
    if "RightToeBase" in bone_map:
        bone_map.setdefault("RightToe", bone_map["RightToeBase"])

    if not rokoko_path.is_file():
        log(f"warn: missing rokoko map {rokoko_path} — FINAL only")
        return bone_map

    data = json.loads(rokoko_path.read_text(encoding="utf-8"))
    if data.get("rokoko_custom_names"):
        n = 0
        for part, targets in (data.get("bones") or {}).items():
            if not targets:
                continue
            smplx = targets[0] if isinstance(targets, list) else targets
            bvh = ROKOKO_PART_TO_BVH.get(part)
            if not bvh:
                continue
            bone_map[bvh] = smplx
            # toe aliases
            if bvh == "LeftToe":
                bone_map["LeftToeBase"] = smplx
            if bvh == "RightToe":
                bone_map["RightToeBase"] = smplx
            n += 1
        log(f"Rokoko overrides applied: {n} from {rokoko_path.name}")
    elif "map" in data:
        for k, v in data["map"].items():
            bone_map[short_name(k)] = v
        log(f"flat map merged from {rokoko_path.name}")
    else:
        for k, v in (data.get("bones") or {}).items():
            if isinstance(v, list) and v:
                bone_map[short_name(k)] = v[0]
            elif isinstance(v, str):
                bone_map[short_name(k)] = v

    # Critical arm mapping (Mixamo Arm = SMPL-X shoulder)
    bone_map["LeftArm"] = "left_shoulder"
    bone_map["RightArm"] = "right_shoulder"
    bone_map["LeftShoulder"] = "left_collar"
    bone_map["RightShoulder"] = "right_collar"
    return bone_map


def height_hips_head(arm) -> float | None:
    h = find_src_bone(arm, "Hips")
    hd = find_src_bone(arm, "Head")
    if not h or not hd:
        return None
    a = arm.matrix_world @ arm.data.bones[h].head_local
    b = arm.matrix_world @ arm.data.bones[hd].head_local
    return (b - a).length


def world_head_pose(arm, bone: str) -> Vector:
    return (arm.matrix_world @ arm.pose.bones[bone].head).copy()


def rest_head_world(arm, bone: str) -> Vector:
    return (arm.matrix_world @ arm.data.bones[bone].head_local).copy()


def set_pelvis_world_delta(arm, delta: Vector) -> None:
    """Add world-space translation to pelvis pose location."""
    if delta.length < 1e-8:
        return
    pb = arm.pose.bones["pelvis"]
    bone = arm.data.bones["pelvis"]
    r_inv = bone.matrix_local.to_3x3().inverted()
    arm_inv = arm.matrix_world.inverted()
    pb.location += r_inv @ (arm_inv.to_3x3() @ delta)
    bpy.context.view_layer.update()


_FOOT_BONES = ("left_foot", "right_foot", "left_ankle", "right_ankle", "left_toe", "right_toe")
_EXTRA_SUPPORT = (
    "left_knee", "right_knee",
    "left_wrist", "right_wrist", "left_hand", "right_hand",
)


def _bone_z(tgt, name: str):
    if name not in tgt.pose.bones:
        return None
    try:
        return float(world_head_pose(tgt, name).z)
    except Exception:
        return None


def _support_z(tgt) -> float:
    """Lowest point that is actually on the ground this frame.

    Feet/toes always count. Knees/hands count only when they sit next to
    the lowest foot (supporting the body). A swing foot, tucked knee, or
    raised hand is ignored — so this is the same rule for every clip.
    """
    foot_zs = [z for z in (_bone_z(tgt, n) for n in _FOOT_BONES) if z is not None]
    lo_foot = min(foot_zs) if foot_zs else None
    extras = []
    if lo_foot is not None:
        for n in _EXTRA_SUPPORT:
            z = _bone_z(tgt, n)
            if z is not None and z <= lo_foot + 0.08:
                extras.append(z)
    bag = foot_zs + extras
    return min(bag) if bag else 0.0


def _lowest_contact_z(tgt) -> float:
    return _support_z(tgt)


def _rest_floor_z(tgt) -> float:
    zs = []
    for n in _FOOT_BONES:
        if n not in tgt.data.bones:
            continue
        try:
            zs.append(float(rest_head_world(tgt, n).z))
        except Exception:
            continue
    return min(zs) if zs else 0.0


_SOLE_VGROUPS = (
    "left_foot", "right_foot", "left_ankle", "right_ankle",
    "left_toe", "right_toe",
)
_HAND_VGROUPS = (
    "left_wrist", "right_wrist",
    "left_index1", "right_index1",
    "left_middle1", "right_middle1",
    "left_index3", "right_index3",
    "left_middle3", "right_middle3",
    "left_pinky3", "right_pinky3",
)


def _body_mesh():
    import bpy
    return bpy.data.objects.get("BodyMesh")


def _rest_sole_pad(tgt) -> float:
    """How far the visible sole sits below the foot *joint* in rest.

    Foot bones are inside the mesh. Planting the joint on z=0 drives the
    skin through the floor by this pad (~1.4cm on this character).
    """
    mesh = _body_mesh()
    joint = _rest_floor_z(tgt)
    if mesh is None:
        return max(0.008, float(joint) - 0.0)
    prev = tgt.data.pose_position
    try:
        tgt.data.pose_position = "REST"
        if tgt.animation_data:
            tgt.animation_data.action = None
        import bpy
        bpy.context.view_layer.update()
        zmin = _contact_mesh_zmin(tgt, mesh, hands=False)
    except Exception:
        zmin = 0.0
    finally:
        try:
            tgt.data.pose_position = prev
        except Exception:
            pass
    if zmin is None:
        return max(0.008, float(joint))
    return max(0.006, float(joint) - float(zmin))


def _contact_mesh_zmin(tgt, mesh, *, hands: bool) -> float | None:
    """Lowest posed *skin* on contact regions (soles, optionally palms).

    Ignores belly/head/chest verts so a crawl torso does not hoist the body.
    """
    import bpy
    if mesh is None or mesh.type != "MESH":
        return None
    names = list(_SOLE_VGROUPS)
    if hands:
        names.extend(_HAND_VGROUPS)
    gmap = {g.name: g.index for g in mesh.vertex_groups}
    idxs = [gmap[n] for n in names if n in gmap]
    if not idxs:
        return None
    idx_set = set(idxs)
    raw = mesh.data
    pick = []
    for i, v in enumerate(raw.vertices):
        wmax = 0.0
        for g in v.groups:
            if g.group in idx_set and g.weight > wmax:
                wmax = g.weight
        if wmax >= 0.35:
            pick.append(i)
    if len(pick) < 8:
        return None
    deps = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(deps)
    me = ev.to_mesh()
    try:
        zs = []
        n = len(me.vertices)
        for i in pick:
            if i < n:
                zs.append(float((ev.matrix_world @ me.vertices[i].co).z))
    finally:
        ev.to_mesh_clear()
    return min(zs) if zs else None


def apply_mesh_sole_clearance(tgt, action_name: str, f0: int, f1: int) -> dict:
    """Lift only so contact *mesh* sits on the plane — never plant joints.

    Source BVH / HumanML put joints on z=0. Those joints live inside the
    SMPL-X mesh, so matching them to the plane puts the sole through it.

    Two passes: first allows a larger per-frame lift to clear deep sinks;
    second soft-passes residual error (cap 0.08) so skin stays ≥ plane.
    """
    act = bpy.data.actions.get(action_name)
    if not act or "pelvis" not in tgt.pose.bones:
        return {"ok": False}
    if not tgt.animation_data:
        tgt.animation_data_create()
    tgt.animation_data.action = act
    assign_slot(tgt)

    plane_z = 0.0
    mesh = _body_mesh()
    pad = _rest_sole_pad(tgt)
    n_lift = 0
    after = []

    def _skin_at_frame():
        wr = _bone_z(tgt, "left_wrist")
        wr2 = _bone_z(tgt, "right_wrist")
        hands_down = any(z is not None and z < 0.22 for z in (wr, wr2))
        skin = _contact_mesh_zmin(tgt, mesh, hands=hands_down) if mesh else None
        if skin is None:
            lo_j = _support_z(tgt)
            skin = float(lo_j) - float(pad)
        return float(skin)

    for pass_i, cap in ((1, 0.20), (2, 0.08)):
        after = []
        for f in range(int(f0), int(f1) + 1):
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()
            skin = _skin_at_frame()
            err = plane_z - float(skin)
            if err > 0.002:
                lift = min(float(err), float(cap))
                set_pelvis_world_delta(tgt, Vector((0.0, 0.0, lift)))
                tgt.pose.bones["pelvis"].keyframe_insert("location", frame=f)
                n_lift += 1
                skin = float(skin) + lift
            after.append(float(skin))
        # Early exit if already clear after first pass
        if after and min(after) >= plane_z - 0.002:
            break

    bpy.context.scene.frame_set(int(f0))
    bpy.context.view_layer.update()
    log(
        f"mesh sole clearance pad={pad:.4f} lifts={n_lift} "
        f"contact_skin=[{min(after):.4f}..{max(after):.4f}] plane={plane_z:.3f}"
    )
    return {
        "ok": True,
        "mode": "mesh_sole",
        "sole_pad": round(float(pad), 4),
        "lifts": n_lift,
        "min_skin": float(min(after)) if after else None,
        "max_skin": float(max(after)) if after else None,
        "floor_z": float(plane_z),
    }


def floor_plant_action(tgt, action_name: str, f0: int, f1: int) -> dict:
    """
    Manual-Rokoko parity: do NOT translate the whole character.

    Height is source Hips (stamp). This pass only lifts a frame when a
    support is *through* the floor (a few cm). Never pulls the body down —
    that would bury walks/stands.
    """
    act = bpy.data.actions.get(action_name)
    if not act or "pelvis" not in tgt.pose.bones:
        return {"ok": False}
    if not tgt.animation_data:
        tgt.animation_data_create()
    tgt.animation_data.action = act
    assign_slot(tgt)

    rest_floor = _rest_floor_z(tgt)
    target_floor = rest_floor - 0.004
    supports = []
    for f in range(int(f0), int(f1) + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        supports.append(_support_z(tgt))
    if not supports:
        return {"ok": False, "reason": "no_contacts"}

    ordered = sorted(supports)
    lo_ref = ordered[max(0, int(0.12 * (len(ordered) - 1)))]
    # Manual Rokoko does NOT translate the whole character onto the floor.
    # Height comes from source Hips (stamp). A constant drop would shove
    # walks/stands through the floor. Only lift true penetration.
    n_const = 0
    dz = 0.0

    # Penetration only — lift, never magnet down.
    # If any real support is already on/above the floor, do not lift the
    # whole body (that floats planted hands). IK fixes the through limb.
    lifts = []
    for f in range(int(f0), int(f1) + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        lo = _support_z(tgt)
        err = target_floor - lo
        if err > 0.003:
            feet = [z for z in (_bone_z(tgt, n) for n in _FOOT_BONES) if z is not None]
            extras = []
            for n in _EXTRA_SUPPORT:
                z = _bone_z(tgt, n)
                if z is not None and z <= (min(feet) if feet else 9.0) + 0.12:
                    extras.append(z)
            planted = [z for z in (feet + extras) if z < target_floor + 0.10]
            on_floor = any(z >= target_floor - 0.005 for z in planted) if planted else False
            if on_floor:
                lifts.append(0.0)
            else:
                lifts.append(min(float(err), 0.08))
        else:
            lifts.append(0.0)
    if len(lifts) >= 3:
        sm = lifts[:]
        for i in range(1, len(lifts) - 1):
            if lifts[i] == 0.0 and lifts[i - 1] == 0.0 and lifts[i + 1] == 0.0:
                sm[i] = 0.0
            else:
                sm[i] = max(0.0, 0.2 * lifts[i - 1] + 0.6 * lifts[i] + 0.2 * lifts[i + 1])
        lifts = sm

    n_lift = 0
    for i, f in enumerate(range(int(f0), int(f1) + 1)):
        if lifts[i] < 1e-5:
            continue
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        set_pelvis_world_delta(tgt, Vector((0.0, 0.0, lifts[i])))
        tgt.pose.bones["pelvis"].keyframe_insert("location", frame=f)
        n_lift += 1

    after = []
    for f in range(int(f0), int(f1) + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        after.append(_support_z(tgt))
    bpy.context.scene.frame_set(int(f0))
    bpy.context.view_layer.update()
    log(
        f"floor plant: constant_dz={dz:.4f} rest_floor={rest_floor:.4f} "
        f"lo_ref={lo_ref:.4f} penetrate_lifts={n_lift} "
        f"support=[{min(after):.4f}..{max(after):.4f}]"
    )
    return {
        "ok": True,
        "mode": "constant_plus_penetration",
        "floor_z": rest_floor,
        "dz": round(float(dz), 4),
        "lo_ref": round(float(lo_ref), 4),
        "min_contact": float(min(after)),
        "max_contact": float(max(after)),
        "fixed_frames": n_const,
        "penetrate_lifts": n_lift,
    }


def _eval_world_bone(tgt, name: str):
    pb = tgt.pose.bones.get(name)
    if pb is None:
        return None
    return (tgt.matrix_world @ pb.matrix).copy()


def align_heading_and_floor(tgt, action, f0: int, f1: int) -> dict:
    """
    Heading only (our side — not MoMask). HumanML3D emits a random facing;
    rotate the baked pelvis path + yaw so travel matches SMPL-X forward (−Y).

    Floor is not done here — floor_plant_action runs once for every clip.
    Does not rewrite limb keys.
    """
    from mathutils import Matrix

    act = action or (tgt.animation_data.action if tgt.animation_data else None)
    if tgt is None or act is None or "pelvis" not in tgt.pose.bones:
        return {"ok": False, "reason": "no_target"}
    if not tgt.animation_data:
        tgt.animation_data_create()
    tgt.animation_data.action = act
    assign_slot(tgt)

    f0, f1 = int(f0), int(f1)
    samples = []
    for f in (f0, f0 + max(1, (f1 - f0) // 5), f0 + max(1, 4 * (f1 - f0) // 5), f1):
        f = max(f0, min(f1, int(f)))
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        w = _eval_world_bone(tgt, "pelvis")
        if w is None:
            continue
        samples.append(w.to_translation().copy())
    if len(samples) < 2:
        return {"ok": False, "reason": "no_samples"}

    travel = samples[-1] - samples[0]
    travel.z = 0.0
    yaw = 0.0
    method = "none"
    # SMPL-X rest faces −Y (camera). Align FACE, not travel —
    # travel-to-−Y flipped walks/crawls that already faced opposite their path.
    target_fwd = Vector((0.0, -1.0, 0.0))

    def _face_fwd():
        lh = world_head_pose(tgt, "left_hip") if "left_hip" in tgt.pose.bones else None
        rh = world_head_pose(tgt, "right_hip") if "right_hip" in tgt.pose.bones else None
        if lh is None or rh is None:
            return None
        across = lh - rh
        across.z = 0.0
        if across.length < 1e-4:
            return None
        across.normalize()
        # left−right × world +Z = body forward (rest −Y). Old (−y, x) was +Y = 180°.
        fwd = Vector((across.y, -across.x, 0.0))
        if fwd.length < 1e-4:
            return None
        fwd.normalize()
        return fwd

    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    face = _face_fwd()
    if face is not None:
        a = math.atan2(face.x, face.y)
        b = math.atan2(target_fwd.x, target_fwd.y)
        yaw = b - a
        method = "face"
    elif travel.length >= 0.18:
        a = math.atan2(travel.x, travel.y)
        b = math.atan2(target_fwd.x, target_fwd.y)
        yaw = b - a
        method = "travel"

    # wrap to [-pi, pi]
    while yaw > math.pi:
        yaw -= 2.0 * math.pi
    while yaw < -math.pi:
        yaw += 2.0 * math.pi
    if abs(yaw) < math.radians(3.0):
        yaw = 0.0

    origin = Vector((samples[0].x, samples[0].y, 0.0))
    R = Matrix.Rotation(yaw, 4, "Z")
    T = Matrix.Translation(origin)
    M = T @ R @ T.inverted() if abs(yaw) > 1e-6 else None

    if M is not None:
        pb = tgt.pose.bones["pelvis"]
        pb.rotation_mode = "QUATERNION"
        for f in range(f0, f1 + 1):
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()
            world_mat = tgt.matrix_world @ pb.matrix
            new_world = M @ world_mat
            pose_mat = tgt.matrix_world.inverted() @ new_world
            local = tgt.convert_space(
                pose_bone=pb, matrix=pose_mat, from_space="POSE", to_space="LOCAL"
            )
            loc, rot, _ = local.decompose()
            pb.location = loc
            pb.rotation_quaternion = rot
            pb.keyframe_insert("location", frame=f)
            pb.keyframe_insert("rotation_quaternion", frame=f)
        log(f"heading align yaw={math.degrees(yaw):.1f}° method={method} travel={travel.length:.3f}m → −Y")

    # Floor is handled once in floor_plant_action (constant + penetration).
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    return {
        "ok": True,
        "yaw_deg": round(math.degrees(yaw), 2),
        "method": method,
        "travel": round(float(travel.length), 3),
        "floor_dz": 0.0,
    }


def main():
    args = parse_args()
    bvh = Path(args.bvh)
    if not bvh.is_absolute():
        bvh = ROOT / bvh
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    rokoko = Path(args.rokoko_map)
    if not rokoko.is_absolute():
        rokoko = ROOT / rokoko

    if not bvh.is_file():
        raise SystemExit(f"missing BVH {bvh}")

    tgt = bpy.data.objects.get(args.armature)
    if not tgt or tgt.type != "ARMATURE":
        raise SystemExit(f"need armature {args.armature} — open whole_body_retargeted.blend")

    # Remove other armatures
    for o in list(bpy.data.objects):
        if o.type == "ARMATURE" and o.name != tgt.name:
            bpy.data.objects.remove(o, do_unlink=True)

    action_name = args.action
    old = bpy.data.actions.get(action_name)
    if old:
        if tgt.animation_data and tgt.animation_data.action == old:
            tgt.animation_data.action = None
        bpy.data.actions.remove(old)

    bone_map = build_bone_map(rokoko)
    log(f"map pairs={len(bone_map)} rokoko={rokoko.name}")

    # --- Import BVH only (no Mixamo) ---
    before = set(bpy.data.objects.keys())
    bpy.ops.import_anim.bvh(
        filepath=str(bvh.resolve()).replace("\\", "/"),
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
    if not src.animation_data or not src.animation_data.action:
        raise SystemExit("BVH has no action")
    fr = src.animation_data.action.frame_range
    f0, f1 = int(fr[0]), int(fr[1])
    file_n = count_bvh_file_frames(bvh)
    key_rng = action_key_range(src.animation_data.action)
    if file_n >= 2:
        # Authoritative: bake every BVH frame. Blender 5 frame_range is often ~25.
        want_f1 = max(f0 + file_n - 1, f1)
        if f1 - f0 + 1 < file_n:
            log(
                f"WARN blender frame_range={f0}-{f1} but BVH header Frames={file_n} "
                f"— baking full {f0}-{want_f1}"
            )
            f1 = want_f1
    if key_rng is not None:
        f0 = min(f0, key_rng[0])
        f1 = max(f1, key_rng[1])
    expand_layered_action_range(src.animation_data.action, f0, f1)
    log(
        f"BVH {src.name} frames={f0}-{f1} file_frames={file_n} "
        f"key_range={key_rng} bones={list(src.data.bones.keys())[:6]}…"
    )

    # Manual Rokoko: no auto-scale unless --auto-scale
    bpy.context.view_layer.update()
    scale_used = 1.0
    if args.auto_scale:
        hs = height_hips_head(src)
        ht = None
        if "pelvis" in tgt.data.bones and "head" in tgt.data.bones:
            a = tgt.matrix_world @ tgt.data.bones["pelvis"].head_local
            b = tgt.matrix_world @ tgt.data.bones["head"].head_local
            ht = (b - a).length
        if hs and ht and hs > 1e-6:
            scale_used = ht / hs
            src.scale = (scale_used, scale_used, scale_used)
            bpy.context.view_layer.update()
            log(f"auto-scale×{scale_used:.4f} (hips→head)")
    else:
        log("auto-scale OFF (manual Rokoko parity, global_scale=1)")

    # Align Hips rest head to pelvis rest head (constraint setup only, not scale)
    hips = find_src_bone(src, "Hips")
    if hips and "pelvis" in tgt.data.bones:
        sh = src.matrix_world @ src.data.bones[hips].head_local
        th = tgt.matrix_world @ tgt.data.bones["pelvis"].head_local
        src.location += th - sh
        bpy.context.view_layer.update()
        log(f"aligned Hips→pelvis Δ={(th - sh).length:.4f}")

    # Pairs
    pairs: list[tuple[str, str]] = []
    seen_tgt = set()
    for src_key, sm in bone_map.items():
        if sm in seen_tgt or sm not in tgt.pose.bones:
            continue
        sb = find_src_bone(src, src_key)
        if not sb:
            continue
        seen_tgt.add(sm)
        pairs.append((sb, sm))
    log(f"mapped pairs={len(pairs)}")
    if len(pairs) < 10:
        raise SystemExit(f"too few pairs ({len(pairs)}) — check BVH names vs map")

    # Helper bones on source (Rokoko / catalog method)
    mw_inv = src.matrix_world.inverted()
    bpy.ops.object.select_all(action="DESELECT")
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="EDIT")
    transforms = {}
    for eb in tgt.data.edit_bones:
        transforms[eb.name] = (
            (mw_inv @ (tgt.matrix_world @ eb.head)).copy(),
            (mw_inv @ (tgt.matrix_world @ eb.tail)).copy(),
            eb.roll,
        )
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    bpy.context.view_layer.objects.active = src
    bpy.ops.object.mode_set(mode="EDIT")
    for sb, sm in pairs:
        parent = src.data.edit_bones.get(sb)
        if parent is None or sm not in transforms:
            continue
        hname = sm + HELPER_SUFFIX
        if hname in src.data.edit_bones:
            src.data.edit_bones.remove(src.data.edit_bones[hname])
        head, tail, roll = transforms[sm]
        nb = src.data.edit_bones.new(hname)
        nb.head, nb.tail, nb.roll = head, tail, roll
        if (nb.tail - nb.head).length < 1e-5:
            nb.tail = nb.head + Vector((0, 0.05, 0))
        nb.parent = parent
        nb.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    def hips_world(frame: int):
        bpy.context.scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        se = src.evaluated_get(dg)
        hb = hips or find_src_bone(src, "Hips")
        if not hb or hb not in se.pose.bones:
            return None
        return (src.matrix_world @ se.pose.bones[hb].matrix).to_translation().copy()

    # Root mode — crawl/sit drop Z even when XY travel is small. Never
    # skip COPY_LOCATION just because the character barely moved forward.
    root_mode = args.root
    src_z_vals = []
    if hips:
        for f in (f0, (f0 + f1) // 2, f1):
            hw = hips_world(f)
            if hw is not None:
                src_z_vals.append(float(hw.z))
    src_z_span = (max(src_z_vals) - min(src_z_vals)) if src_z_vals else 0.0
    if root_mode == "auto" and hips:
        pa, pb, pc = hips_world(f0), hips_world((f0 + f1) // 2), hips_world(f1)
        pts = [p for p in (pa, pb, pc) if p is not None]
        travel = 0.0
        if len(pts) >= 2:
            travel = max((pts[i] - pts[0]).length for i in range(1, len(pts)))
        root_mode = "location" if (travel > 0.05 or src_z_span > 0.08) else "inplace"
        log(f"auto root={root_mode} travel≈{travel:.3f} hips_z_span={src_z_span:.3f}")
    # Crawl/sit/lie: always copy hips translation (otherwise hip stays stand-height)
    if src_z_span > 0.12:
        root_mode = "location"
        log(f"force root=location (source hips drop {src_z_span:.3f}m — crawl/sit)")

    # Constraints
    bpy.ops.object.select_all(action="DESELECT")
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="POSE")
    for sb, sm in pairs:
        pb = tgt.pose.bones[sm]
        clear_constraints(pb)
        c = pb.constraints.new("COPY_ROTATION")
        c.name = "Copy Rot" + RETARGET_ID
        c.target = src
        c.subtarget = sm + HELPER_SUFFIX
        c.mix_mode = "REPLACE"
        pb.rotation_mode = "QUATERNION"
        # Always copy helper location onto pelvis (height + travel).
        if sm == "pelvis":
            # Manual Rokoko: COPY_LOCATION from the helper, not raw Hips.
            # Helper sits at SMPL-X pelvis rest, parented to Hips — so height
            # is SMPL-X rest + source motion. Copying Hips world dumps HumanML
            # hips Z onto our pelvis and puts feet through the floor.
            helper = sm + HELPER_SUFFIX
            cl = pb.constraints.new("COPY_LOCATION")
            cl.name = "Copy Loc" + RETARGET_ID
            cl.target = src
            cl.subtarget = helper if src.pose.bones.get(helper) else sb

    if not tgt.animation_data:
        tgt.animation_data_create()

    def _action_in_use(act) -> bool:
        if act is None:
            return False
        for obj in bpy.data.objects:
            try:
                if obj.animation_data and obj.animation_data.action == act:
                    return True
            except Exception:
                continue
        return False

    def _new_target_action() -> object:
        # NEVER delete the imported BVH Action — it is named after the .bvh
        # stem (same as --action), and removing it freezes the source pose.
        name = action_name
        existing = bpy.data.actions.get(name)
        if existing is not None and _action_in_use(existing):
            name = action_name + "_smplx"
            leftover = bpy.data.actions.get(name)
            if leftover is not None and not _action_in_use(leftover):
                try:
                    bpy.data.actions.remove(leftover)
                except Exception:
                    pass
        elif existing is not None:
            try:
                if tgt.animation_data and tgt.animation_data.action == existing:
                    tgt.animation_data.action = None
                bpy.data.actions.remove(existing)
            except Exception:
                pass
        created = bpy.data.actions.new(name)
        created.use_fake_user = True
        ensure_layered_action(created, tgt)
        tgt.animation_data.action = created
        assign_slot(tgt)
        expand_layered_action_range(created, f0, f1)
        log(f"target Action {created.name!r} (source BVH Action kept)")
        return created

    def _bake_keys(created) -> None:
        bones = [tgt.pose.bones[sm] for _, sm in pairs]
        for f in range(int(f0), int(f1) + 1):
            bpy.context.scene.frame_set(f)
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
                    # Bake the helper COPY_LOCATION as-is (manual Rokoko).
                    pb.location = loc
                    pb.keyframe_insert("location", frame=f)
                pb.rotation_quaternion = rot
                pb.keyframe_insert("rotation_quaternion", frame=f)

    def _world_pelvis_z(frame: int) -> float | None:
        bpy.context.scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        if "pelvis" not in tgt.pose.bones:
            return None
        return float((tgt.matrix_world @ tgt.pose.bones["pelvis"].head).z)

    def _stamp_pelvis_from_source_hips() -> int:
        """Copy source Hips WORLD path onto pelvis: relative XY + relative Z.

        SMPL-X pelvis local axes are not world axes (armature origin is near
        the chest). Stamping in world space keeps travel.

        Height is NOT absolute source Hips Z. MoMask/HumanML rest hips sit
        ~0.89–0.94 m; SMPL-X rest pelvis is ~0.99 m. Absolute Z puts walks
        through the floor. Manual Rokoko never stamps — helper COPY_LOCATION
        keeps SMPL-X rest height + source delta. Match that:
            pelvis_z = smplx_rest_z + (src_hips_z - src_rest_z)
        Crawl/sit drop because source hips drop vs *source rest*, not because
        we shove the whole skeleton.
        """
        hb = hips or find_src_bone(src, "Hips")
        if not hb:
            return 0
        pb = tgt.pose.bones.get("pelvis")
        if pb is None:
            return 0
        tgt_rest_z = float(rest_head_world(tgt, "pelvis").z)
        src_rest_z = tgt_rest_z
        try:
            if hb in src.data.bones:
                src_rest_z = float((src.matrix_world @ src.data.bones[hb].head_local).z)
        except Exception:
            pass
        log(
            f"stamp height relative smplx_rest_z={tgt_rest_z:.4f} "
            f"src_rest_hips_z={src_rest_z:.4f} (not absolute source Z)"
        )
        n = 0
        origin = None
        for f in range(int(f0), int(f1) + 1):
            hw = hips_world(f)
            if hw is None:
                continue
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()
            wmat = tgt.matrix_world @ pb.matrix
            _wloc, wrot, wscl = wmat.decompose()
            if origin is None:
                origin = hw.copy()
            height_z = tgt_rest_z + (float(hw.z) - src_rest_z)
            if root_mode == "location":
                wloc = Vector((hw.x - origin.x, hw.y - origin.y, height_z))
            else:
                wloc = Vector((0.0, 0.0, height_z))
            new_w = Matrix.LocRotScale(wloc, wrot, wscl)
            pose_mat = tgt.matrix_world.inverted() @ new_w
            local = tgt.convert_space(
                pose_bone=pb, matrix=pose_mat, from_space="POSE", to_space="LOCAL"
            )
            loc, _rot, _ = local.decompose()
            pb.location = loc
            pb.keyframe_insert("location", frame=f)
            n += 1
        return n

    act = _new_target_action()
    _bake_keys(act)

    # Verify hips height survived retarget. Compare WORLD Z of source Hips vs
    # baked pelvis (span-only check missed crawls that start already low).
    tgt.animation_data.action = act
    assign_slot(tgt)
    pel_zs = []
    src_zs_chk = []
    for f in (f0, (f0 + f1) // 2, f1):
        z = _world_pelvis_z(f)
        if z is not None:
            pel_zs.append(z)
        hw = hips_world(f)
        if hw is not None:
            src_zs_chk.append(float(hw.z))
    pel_z_span = (max(pel_zs) - min(pel_zs)) if pel_zs else 0.0
    mid_src = src_zs_chk[len(src_zs_chk) // 2] if src_zs_chk else None
    mid_pel = pel_zs[len(pel_zs) // 2] if pel_zs else None
    height_err = (
        abs(float(mid_src) - float(mid_pel))
        if (mid_src is not None and mid_pel is not None)
        else 0.0
    )
    log(
        f"height check source_hips_z_span={src_z_span:.3f} "
        f"baked_pelvis_z_span={pel_z_span:.3f} "
        f"mid_src_z={mid_src} mid_pelvis_z={mid_pel} err={height_err:.3f}"
    )
    # Do not stamp over the helper bake. Manual Rokoko keeps COPY_LOCATION.
    # Stamping source Hips world Z is what put walks through the floor and
    # crawls at the wrong contact height.
    n_stamp = 0
    pel_zs = []
    for f in (f0, (f0 + f1) // 2, f1):
        z = _world_pelvis_z(f)
        if z is not None:
            pel_zs.append(z)
    pel_z_span = (max(pel_zs) - min(pel_zs)) if pel_zs else 0.0
    log(
        f"height after helper bake n_stamp={n_stamp} baked_pelvis_z_span={pel_z_span:.3f} "
        f"mid={ [round(z, 3) for z in pel_zs] }"
    )

    # Source-based crawl/sit flag — must run before we delete the BVH armature.
    lowered = False
    try:
        rest_src_z = None
        if hips and hips in src.data.bones:
            rest_src_z = float((src.matrix_world @ src.data.bones[hips].head_local).z)
        mid_src_z = src_zs_chk[len(src_zs_chk) // 2] if src_zs_chk else None
        if rest_src_z is not None and mid_src_z is not None:
            lowered = (rest_src_z - mid_src_z) > 0.12
        if lowered:
            log(f"lowered locomotion src_mid_z={mid_src_z:.3f} src_rest_z={rest_src_z:.3f}")
    except Exception:
        lowered = False

    baked_try = action_key_range(act)
    baked_n_try = (baked_try[1] - baked_try[0] + 1) if baked_try else 0
    if baked_n_try < max(8, int((f1 - f0 + 1) * 0.8)):
        log(
            f"first bake short keys={baked_try}; recreate Action + expand + rebake"
        )
        act = _new_target_action()
        expand_layered_action_range(act, f0, f1 + 8)
        _bake_keys(act)

    for pb in tgt.pose.bones:
        clear_constraints(pb)
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    # Remove source BVH armature + its Action, then take the requested name.
    src_act = None
    try:
        src_act = src.animation_data.action if src.animation_data else None
    except Exception:
        src_act = None
    try:
        bpy.data.objects.remove(src, do_unlink=True)
    except Exception:
        pass
    if src_act is not None:
        try:
            bpy.data.actions.remove(src_act)
        except Exception:
            pass
    if act.name != action_name:
        clash = bpy.data.actions.get(action_name)
        if clash is not None and clash != act:
            try:
                bpy.data.actions.remove(clash)
            except Exception:
                pass
        try:
            act.name = action_name
            log(f"renamed target Action → {action_name!r}")
        except Exception as e:
            log(f"rename target Action skip: {e}")

    tgt.animation_data.action = act
    assign_slot(tgt)

    baked = action_key_range(act)
    expect_n = int(f1) - int(f0) + 1
    baked_n = (baked[1] - baked[0] + 1) if baked else 0
    if baked is None or baked_n < max(8, int(expect_n * 0.8)):
        log(
            f"ERROR bake incomplete: keys={baked} expected={f0}-{f1} ({expect_n} frames). "
            "Layered strip likely clipped keys — expanding and this run is invalid."
        )
        expand_layered_action_range(act, f0, f1)
    else:
        log(f"bake keys={baked[0]}-{baked[1]} ({baked_n} frames, expected {expect_n})")

    # Heading + constant floor (MoMask heading is random; floor Z was zeroed before)
    align = {"ok": False, "skipped": True}
    if not bool(args.no_align):
        try:
            align = align_heading_and_floor(tgt, act, f0, f1)
        except Exception as e:
            log(f"heading/floor align failed: {e}")
            align = {"ok": False, "error": str(e)}

    # Fallback lowered detect from baked pelvis (source already removed).
    if not lowered:
        try:
            bpy.context.scene.frame_set((int(f0) + int(f1)) // 2)
            bpy.context.view_layer.update()
            mid_z = float((tgt.matrix_world @ tgt.pose.bones["pelvis"].head).z)
            rest_z = rest_head_world(tgt, "pelvis").z
            lowered = (rest_z - mid_z) > 0.12
            if lowered:
                log(f"lowered locomotion mid_pelvis_z={mid_z:.3f} rest={rest_z:.3f}")
        except Exception:
            pass

    # Hands same rule as feet: palm *skin* on the plane, palm facing down.
    # Run BEFORE sole clearance so the lift uses oriented palm verts.
    hands = {"ok": False, "skipped": True}
    try:
        from smplx_contact_ik import apply_hand_floor_orient
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from smplx_contact_ik import apply_hand_floor_orient
    try:
        expand_layered_action_range(act, f0, f1)
        tgt.animation_data.action = act
        assign_slot(tgt)
        hands = apply_hand_floor_orient(tgt, act, f0, f1, 0.0)
    except Exception as e:
        log(f"hand floor orient failed: {e}")
        hands = {"ok": False, "error": str(e)}

    # Joint-on-plane plant is wrong (bones sit inside the mesh). Always
    # clear the *sole/palm skin* to z=0. Optional old joint plant stays opt-in.
    plant = apply_mesh_sole_clearance(tgt, action_name, f0, f1)
    if isinstance(plant, dict):
        plant["hands"] = hands
    if bool(getattr(args, "floor_plant", False)) and not bool(args.no_floor_plant):
        plant = {**plant, "joint_plant": floor_plant_action(tgt, action_name, f0, f1)}

    # Re-solve planted feet/hands on SMPL-X (BVH lock does not survive retarget).
    foot_lock = {"ok": False, "skipped": True}
    if bool(getattr(args, "foot_lock", False)) and not bool(
        getattr(args, "no_foot_lock", False)
    ):
        try:
            from smplx_contact_ik import apply_contact_ik
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from smplx_contact_ik import apply_contact_ik
        try:
            expand_layered_action_range(act, f0, f1)
            tgt.animation_data.action = act
            assign_slot(tgt)
            floor_z = float(plant.get("floor_z", _rest_floor_z(tgt))) if isinstance(plant, dict) else _rest_floor_z(tgt)
            foot_lock = apply_contact_ik(tgt, act, f0, f1, floor_z)
        except Exception as e:
            log(f"contact IK failed: {e}")
            foot_lock = {"ok": False, "error": str(e)}

    knee_fix = {"ok": False, "skipped": True}
    rest_pads = {"ok": False, "skipped": True}
    if os.environ.get("MOMASK_KNOCK_KNEE", "0").strip().lower() in ("1", "true", "yes"):
        knee_fix = fix_knock_knees_action(tgt, act, f0, f1, max_deg=6.0)
    if abs(float(args.shoulder_fwd_deg)) > 0.05 or abs(float(args.collar_fwd_deg)) > 0.05:
        apply_shoulder_forward_fix(
            tgt, act, f0, f1,
            shoulder_fwd_deg=float(args.shoulder_fwd_deg),
            collar_fwd_deg=float(args.collar_fwd_deg),
        )
    # Rest pads overwrite the first/last frames with a standing pose.
    # Skip whenever the authored height already varies (any clip — not a
    # named action list): hip drop, takeoff/land, or large support span.
    skip_pads = bool(lowered)
    pad_reason = "lowered_pose" if lowered else ""
    if not skip_pads:
        try:
            pz, sz = [], []
            for f in (int(f0), (int(f0) + int(f1)) // 2, int(f1)):
                bpy.context.scene.frame_set(f)
                bpy.context.view_layer.update()
                pz.append(float((tgt.matrix_world @ tgt.pose.bones["pelvis"].head).z))
                sz.append(_support_z(tgt))
            if pz and (max(pz) - min(pz)) > 0.16:
                skip_pads, pad_reason = True, "pelvis_span"
            elif sz and (max(sz) - min(sz)) > 0.16:
                skip_pads, pad_reason = True, "support_span"
        except Exception:
            pass
    if skip_pads:
        rest_pads = {"ok": False, "skipped": True, "reason": pad_reason}
        log(f"skip custom rest pads ({pad_reason})")
    elif os.environ.get("MOMASK_REST_PADS", "1").strip().lower() not in (
        "0", "false", "no", "off",
    ):
        rest_pads = apply_custom_rest_pads(tgt, act, f0, f1, n_in=8, n_out=6)

    # Verify samples
    def wh(n):
        return tgt.matrix_world @ tgt.pose.bones[n].head

    for f in (f0, (f0 + f1) // 2, f1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        pel, lf, rf = wh("pelvis"), wh("left_foot"), wh("right_foot")
        lk = wh("left_knee") if "left_knee" in tgt.pose.bones else lf
        lw = wh("left_wrist") if "left_wrist" in tgt.pose.bones else pel
        log(
            f"  f{f}: pelvis=({pel.x:.3f},{pel.y:.3f},{pel.z:.3f}) "
            f"feet_z=({lf.z:.3f},{rf.z:.3f}) knee_z={lk.z:.3f} wrist_z={lw.z:.3f} "
            f"min_contact={min(lf.z, rf.z, lk.z, lw.z):.3f}"
        )

    bpy.context.scene.frame_start = f0
    bpy.context.scene.frame_end = f1
    bpy.context.scene.frame_set(f0)
    out.parent.mkdir(parents=True, exist_ok=True)

    want_full = bool(args.full_scene or args.no_slim)
    slim_info: dict = {}
    if want_full:
        fp = str(out.resolve()).replace("\\", "/")
        try:
            bpy.ops.wm.save_as_mainfile(filepath=fp, compress=True)
        except Exception:
            bpy.ops.wm.save_as_mainfile(filepath=fp, check_existing=False)
        log(f"saved FULL scene {out} (heavy cache)")
        slim_info = {
            "cache_format": "full_scene",
            "size_bytes": out.stat().st_size if out.is_file() else 0,
        }
    else:
        # Default: Action-only slim file for momask_cache (not mesh/camera/scene)
        slim_info = save_action_only_blend(action_name, out)

    log(
        json.dumps(
            {
                "ok": True,
                "method": "direct_bvh_rokoko_no_keemap",
                "action": action_name,
                "pairs": len(pairs),
                "root": root_mode,
                "rokoko_map": str(rokoko),
                "auto_scale": bool(args.auto_scale),
                "scale_used": float(scale_used),
                "heading_floor_align": align,
                "floor_plant": plant,
                "foot_lock": foot_lock,
                "custom_rest_pads": rest_pads,
                "knock_knee_fix": knee_fix,
                "shoulder_fwd_deg": float(args.shoulder_fwd_deg),
                "collar_fwd_deg": float(args.collar_fwd_deg),
                "method_note": "manual_rokoko_parity_no_auto_scale",
                "cache_format": slim_info.get("cache_format", "action_only"),
                "size_bytes": slim_info.get("size_bytes", 0),
                "size_mb": slim_info.get("size_mb")
                or round(float(slim_info.get("size_bytes", 0)) / (1024 * 1024), 3),
                # Camera is session/UDP — not embedded in body cache
                "camera": "live_udp_not_in_cache",
            }
        )
    )
    log("DONE")


if __name__ == "__main__":
    main()
