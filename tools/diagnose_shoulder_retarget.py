"""
Compare arm/shoulder directions: official MoMask BVH (scaled) vs SMPL-X Action.

Outputs body_motion/shoulder_diagnose_report.json with measured error and
recommended local shoulder/collar fix (degrees).

Usage:
  blender.exe whole_body_retargeted.blend --background --python tools/diagnose_shoulder_retarget.py -- ^
    --bvh body_motion/momask_cache/official_t2m_scale/momask_official_1786113934_official_ik.bvh ^
    --smplx-blend body_motion/momask_cache/official_t2m_scale/momask_official_1786113934.blend ^
    --action momask_official_1786113934 ^
    --catalog-walk-blend whole_body_retargeted.blend
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[shoulder_diag] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    d = {
        "bvh": str(ROOT / "body_motion" / "momask_cache" / "official_t2m_scale" / "momask_official_1786113934_official_ik.bvh"),
        "smplx_blend": str(ROOT / "body_motion" / "momask_cache" / "official_t2m_scale" / "momask_official_1786113934.blend"),
        "action": "momask_official_1786113934",
        "catalog_action": "walk",
        "out": str(ROOT / "body_motion" / "shoulder_diagnose_report.json"),
        "samples": "12",
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and i + 1 < len(argv):
            key = a[2:].replace("-", "_")
            d[key] = argv[i + 1]
            i += 2
        else:
            i += 1
    return d


def assign_slot(obj) -> None:
    if obj and obj.animation_data and obj.animation_data.action:
        try:
            slots = obj.animation_data.action_suitable_slots
            if slots:
                obj.animation_data.action_slot = slots[0]
        except Exception:
            pass


def world_head(arm, bone: str) -> Vector:
    return (arm.matrix_world @ arm.pose.bones[bone].head).copy()


def world_tail(arm, bone: str) -> Vector:
    return (arm.matrix_world @ arm.pose.bones[bone].tail).copy()


def height_hips_head_bvh(arm) -> float:
    a = arm.matrix_world @ arm.data.bones["Hips"].head_local
    b = arm.matrix_world @ arm.data.bones["Head"].head_local
    return (b - a).length


def height_hips_head_smplx(arm) -> float:
    a = arm.matrix_world @ arm.data.bones["pelvis"].head_local
    b = arm.matrix_world @ arm.data.bones["head"].head_local
    return (b - a).length


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
        update_scene_fps=False,
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


def arm_chain_world(arm, side: str, is_bvh: bool):
    """Return dict of world positions for shoulder chain + pelvis."""
    if is_bvh:
        # HML/MoMask BVH names
        names = {
            "pelvis": "Hips",
            "collar": "LeftShoulder" if side == "L" else "RightShoulder",
            "shoulder": "LeftArm" if side == "L" else "RightArm",
            "elbow": "LeftForeArm" if side == "L" else "RightForeArm",
            "wrist": "LeftHand" if side == "L" else "RightHand",
        }
    else:
        names = {
            "pelvis": "pelvis",
            "collar": "left_collar" if side == "L" else "right_collar",
            "shoulder": "left_shoulder" if side == "L" else "right_shoulder",
            "elbow": "left_elbow" if side == "L" else "right_elbow",
            "wrist": "left_wrist" if side == "L" else "right_wrist",
        }
    out = {}
    for k, bn in names.items():
        if bn not in arm.pose.bones:
            return None
        out[k] = world_head(arm, bn)
    return out


def pelvis_local(chain: dict) -> dict:
    """Express chain points in a pelvis-centered frame (no rotation align)."""
    p = chain["pelvis"]
    return {k: (v - p) for k, v in chain.items()}


def unit(v: Vector) -> Vector:
    if v.length < 1e-8:
        return Vector((0, 0, 0))
    return v.normalized()


def angle_deg(a: Vector, b: Vector) -> float:
    a, b = unit(a), unit(b)
    if a.length < 1e-8 or b.length < 1e-8:
        return 0.0
    c = max(-1.0, min(1.0, a.dot(b)))
    return math.degrees(math.acos(c))


def sample_arm_metrics(arm, side: str, is_bvh: bool) -> dict | None:
    ch = arm_chain_world(arm, side, is_bvh)
    if not ch:
        return None
    loc = pelvis_local(ch)
    # Upper-arm direction: shoulder → elbow
    upper = loc["elbow"] - loc["shoulder"]
    # Collar direction: collar → shoulder
    collar = loc["shoulder"] - loc["collar"]
    # Full arm: shoulder → wrist
    full = loc["wrist"] - loc["shoulder"]
    # "Forward" in MoMask Y-up BVH is often +Z or -Z; in SMPL-X Z-up face -Y
    # Use relative: Y component in pelvis space (Blender world after import)
    return {
        "upper": [upper.x, upper.y, upper.z],
        "collar": [collar.x, collar.y, collar.z],
        "full": [full.x, full.y, full.z],
        "shoulder_pos": [loc["shoulder"].x, loc["shoulder"].y, loc["shoulder"].z],
        "elbow_pos": [loc["elbow"].x, loc["elbow"].y, loc["elbow"].z],
        "wrist_pos": [loc["wrist"].x, loc["wrist"].y, loc["wrist"].z],
        # How far back: more positive Y if face -Y? Character faces -Y so back is +Y
        "shoulder_y": loc["shoulder"].y,
        "elbow_y": loc["elbow"].y,
        "wrist_y": loc["wrist"].y,
        "upper_len": upper.length,
        "full_len": full.length,
    }


def mean(xs):
    return sum(xs) / max(1, len(xs))


def main():
    args = parse_args()
    bvh_path = Path(args["bvh"])
    if not bvh_path.is_absolute():
        bvh_path = ROOT / bvh_path
    out_path = Path(args["out"])
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    n_samples = max(4, int(args["samples"]))

    # --- Scene: keep SMPL-X, import BVH ---
    tgt = bpy.data.objects.get("SMPL-X_Armature")
    if not tgt:
        # try load action from smplx blend by linking
        raise SystemExit("Open whole_body_retargeted.blend or a blend with SMPL-X_Armature")

    # Load retargeted action from cache blend if needed
    action_name = args["action"]
    act = bpy.data.actions.get(action_name)
    if act is None:
        smplx_blend = Path(args["smplx_blend"])
        if not smplx_blend.is_absolute():
            smplx_blend = ROOT / smplx_blend
        if smplx_blend.is_file():
            log(f"link action from {smplx_blend.name}")
            with bpy.data.libraries.load(str(smplx_blend), link=False) as (data_from, data_to):
                if action_name in (data_from.actions or []):
                    data_to.actions = [action_name]
                else:
                    # take any momask action
                    moms = [a for a in (data_from.actions or []) if "momask" in a.lower() or a == action_name]
                    data_to.actions = moms[:1]
                    if moms:
                        action_name = moms[0]
            act = bpy.data.actions.get(action_name)
    if act is None:
        raise SystemExit(f"Action {args['action']!r} not found")

    if not tgt.animation_data:
        tgt.animation_data_create()
    tgt.animation_data.action = act
    assign_slot(tgt)
    fr = act.frame_range
    f0, f1 = int(fr[0]), int(fr[1])

    # Catalog walk reference on same skeleton if present
    cat = bpy.data.actions.get(args["catalog_action"])
    has_catalog = cat is not None

    # Import official BVH
    if not bvh_path.is_file():
        raise SystemExit(f"missing BVH {bvh_path}")
    src = import_bvh(bvh_path)
    log(f"BVH {src.name} bones={len(src.data.bones)}")

    # Scale BVH to SMPL-X height
    hs, ht = height_hips_head_bvh(src), height_hips_head_smplx(tgt)
    scale = ht / hs if hs > 1e-6 else 1.0
    src.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    # Align hips
    sh = src.matrix_world @ src.data.bones["Hips"].head_local
    th = tgt.matrix_world @ tgt.data.bones["pelvis"].head_local
    src.location += th - sh
    bpy.context.view_layer.update()
    log(f"scale BVH×{scale:.4f} (hips-head src={hs:.3f} tgt={ht:.3f})")

    sfr = src.animation_data.action.frame_range
    sf0, sf1 = int(sfr[0]), int(sfr[1])
    # Align sample windows
    n = min(f1 - f0, sf1 - sf0)
    frames_s = [sf0 + int(i * n / (n_samples - 1)) for i in range(n_samples)]
    frames_t = [f0 + int(i * (f1 - f0) / (n_samples - 1)) for i in range(n_samples)]

    def collect(arm, frames, is_bvh, set_action=None):
        rows = {"L": [], "R": []}
        if set_action is not None:
            if not arm.animation_data:
                arm.animation_data_create()
            arm.animation_data.action = set_action
            assign_slot(arm)
        for f in frames:
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()
            for side in ("L", "R"):
                m = sample_arm_metrics(arm, side, is_bvh)
                if m:
                    rows[side].append(m)
        return rows

    bvh_m = collect(src, frames_s, True)
    ret_m = collect(tgt, frames_t, False, set_action=act)
    cat_m = collect(tgt, frames_t, False, set_action=cat) if has_catalog else None

    def compare(ref_rows, test_rows, label: str):
        """Compare upper-arm direction angle and elbow_y (backness)."""
        report = {"label": label, "L": {}, "R": {}}
        for side in ("L", "R"):
            ref, tes = ref_rows[side], test_rows[side]
            n = min(len(ref), len(tes))
            if n == 0:
                continue
            ang_upper = []
            ang_full = []
            dy_elbow = []  # test - ref for Y (back if character faces -Y, larger +Y = more back)
            dy_wrist = []
            for i in range(n):
                ru = Vector(ref[i]["upper"])
                tu = Vector(tes[i]["upper"])
                rf = Vector(ref[i]["full"])
                tf = Vector(tes[i]["full"])
                ang_upper.append(angle_deg(ru, tu))
                ang_full.append(angle_deg(rf, tf))
                dy_elbow.append(tes[i]["elbow_y"] - ref[i]["elbow_y"])
                dy_wrist.append(tes[i]["wrist_y"] - ref[i]["wrist_y"])
            report[side] = {
                "n": n,
                "upper_arm_angle_mean_deg": round(mean(ang_upper), 2),
                "upper_arm_angle_max_deg": round(max(ang_upper), 2),
                "full_arm_angle_mean_deg": round(mean(ang_full), 2),
                "elbow_y_delta_mean": round(mean(dy_elbow), 4),
                "wrist_y_delta_mean": round(mean(dy_wrist), 4),
                "ref_elbow_y_mean": round(mean([r["elbow_y"] for r in ref[:n]]), 4),
                "test_elbow_y_mean": round(mean([t["elbow_y"] for t in tes[:n]]), 4),
            }
        return report

    # BVH vs retarget: need comparable orientation. After scale+align both Z-up in Blender.
    # BVH import is Y-up converted by Blender importer to scene.
    comp_bvh = compare(bvh_m, ret_m, "official_bvh_scaled_vs_smplx_retarget")
    comp_cat = compare(cat_m, ret_m, "catalog_walk_vs_smplx_retarget") if cat_m else None

    # Recommend fix from mean elbow/wrist Y delta vs catalog (best "our skeleton" ref)
    # If retarget elbow_y > catalog elbow_y (more +Y = more back when facing -Y), need forward fix.
    rec = {"shoulder_fwd_deg": 0.0, "collar_fwd_deg": 0.0, "basis": "none"}
    if comp_cat:
        # Average L/R wrist/elbow delta
        de = mean([
            comp_cat["L"].get("elbow_y_delta_mean", 0),
            comp_cat["R"].get("elbow_y_delta_mean", 0),
        ])
        dw = mean([
            comp_cat["L"].get("wrist_y_delta_mean", 0),
            comp_cat["R"].get("wrist_y_delta_mean", 0),
        ])
        # Empirical: ~0.02m back ≈ 3–4°; scale deg ≈ 180 * delta / (pi * arm_len)
        arm_len = 0.28  # approx upper arm
        # If test more positive Y than catalog (back), de > 0 → need negative? 
        # facing -Y: forward is -Y, so more back = larger Y. Forward fix reduces Y.
        # Our fix uses Euler X positive as "forward" on SMPL-X — keep sign calibrated.
        mean_back = mean([de, dw])  # meters
        # Convert back offset to degrees at shoulder (small angle)
        deg = max(-20.0, min(20.0, math.degrees(math.atan2(mean_back, arm_len))))
        # If mean_back > 0 (elbow more +Y / back), we want forward → positive shoulder_fwd in our fix
        rec = {
            "shoulder_fwd_deg": round(deg * 1.2, 2),  # slight gain on upper arm
            "collar_fwd_deg": round(deg * 0.5, 2),
            "basis": "catalog_walk_vs_retarget_Y",
            "mean_elbow_wrist_y_delta_m": round(mean_back, 4),
            "note": "positive shoulder_fwd_deg pulls arms forward if retarget is more +Y (back)",
        }
    elif comp_bvh:
        # Fallback: compare to BVH Y (may need axis remap) — use angle only
        ang = mean([
            comp_bvh["L"].get("upper_arm_angle_mean_deg", 0),
            comp_bvh["R"].get("upper_arm_angle_mean_deg", 0),
        ])
        rec = {
            "shoulder_fwd_deg": round(min(15.0, max(0.0, ang * 0.35)), 2),
            "collar_fwd_deg": round(min(8.0, max(0.0, ang * 0.15)), 2),
            "basis": "bvh_upper_arm_angle",
            "mean_upper_arm_angle_deg": round(ang, 2),
        }

    report = {
        "bvh": str(bvh_path),
        "action": action_name,
        "scale": round(scale, 5),
        "smplx_frames": [f0, f1],
        "bvh_frames": [sf0, sf1],
        "samples": n_samples,
        "compare_bvh_vs_retarget": comp_bvh,
        "compare_catalog_walk_vs_retarget": comp_cat,
        "recommended_fix": rec,
        "interpretation": {
            "elbow_y_delta": "test - ref; larger +Y usually means arms more 'back' if character faces -Y",
            "upper_arm_angle_deg": "direction mismatch shoulder→elbow between ref and retarget",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"wrote {out_path}")
    log(f"recommended_fix={json.dumps(rec)}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
