"""
Bake MoMask IK glb (our template) → SMPL-X_Armature.

Motion already includes walk root + colinear spine from momask_joints_ik_our_template.py.
Bake only aims bones at those targets (+ light elbow inward).

Usage:
  blender.exe whole_body_retargeted.blend --background --python tools/bake_our_template_glb_to_smplx.py -- ^
    --npz .../walk_our_template_ik.npz --action walk_our_tpl_ik --out .../walk_our_template_ik.blend
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]

HML_TO_BONE = {
    0: "pelvis",
    1: "left_hip",
    2: "right_hip",
    3: "spine1",
    4: "left_knee",
    5: "right_knee",
    6: "spine2",
    7: "left_ankle",
    8: "right_ankle",
    9: "spine3",
    10: "left_foot",
    11: "right_foot",
    12: "neck",
    13: "left_collar",
    14: "right_collar",
    15: "head",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_wrist",
    21: "right_wrist",
}

EDGES = [
    ("pelvis", "spine1", 0, 3),
    ("spine1", "spine2", 3, 6),
    ("spine2", "spine3", 6, 9),
    ("spine3", "neck", 9, 12),
    ("neck", "head", 12, 15),
    ("left_hip", "left_knee", 1, 4),
    ("left_knee", "left_ankle", 4, 7),
    ("left_ankle", "left_foot", 7, 10),
    ("right_hip", "right_knee", 2, 5),
    ("right_knee", "right_ankle", 5, 8),
    ("right_ankle", "right_foot", 8, 11),
    ("left_collar", "left_shoulder", 13, 16),
    ("left_shoulder", "left_elbow", 16, 18),
    ("left_elbow", "left_wrist", 18, 20),
    ("right_collar", "right_shoulder", 14, 17),
    ("right_shoulder", "right_elbow", 17, 19),
    ("right_elbow", "right_wrist", 19, 21),
]

KEY = list(HML_TO_BONE.values())

RELAXED_HAND = {
    "left_thumb1": (0.994, 0.05, 0.03, 0.10),
    "left_thumb2": (0.990, 0.08, 0.02, 0.05),
    "left_thumb3": (0.995, 0.06, 0.01, 0.02),
    "left_index1": (0.970, 0.00, 0.00, -0.24),
    "left_index2": (0.980, 0.00, 0.00, -0.20),
    "left_index3": (0.990, 0.00, 0.00, -0.12),
    "left_middle1": (0.960, 0.00, 0.02, -0.28),
    "left_middle2": (0.975, 0.00, 0.00, -0.22),
    "left_middle3": (0.990, 0.00, 0.00, -0.12),
    "left_ring1": (0.955, 0.00, 0.03, -0.29),
    "left_ring2": (0.975, 0.00, 0.00, -0.22),
    "left_ring3": (0.990, 0.00, 0.00, -0.12),
    "left_pinky1": (0.950, 0.00, 0.04, -0.31),
    "left_pinky2": (0.975, 0.00, 0.00, -0.20),
    "left_pinky3": (0.990, 0.00, 0.00, -0.10),
    "right_thumb1": (0.994, 0.05, -0.03, -0.10),
    "right_thumb2": (0.990, 0.08, -0.02, -0.05),
    "right_thumb3": (0.995, 0.06, -0.01, -0.02),
    "right_index1": (0.970, 0.00, 0.00, 0.24),
    "right_index2": (0.980, 0.00, 0.00, 0.20),
    "right_index3": (0.990, 0.00, 0.00, 0.12),
    "right_middle1": (0.960, 0.00, -0.02, 0.28),
    "right_middle2": (0.975, 0.00, 0.00, 0.22),
    "right_middle3": (0.990, 0.00, 0.00, 0.12),
    "right_ring1": (0.955, 0.00, -0.03, 0.29),
    "right_ring2": (0.975, 0.00, 0.00, 0.22),
    "right_ring3": (0.990, 0.00, 0.00, 0.12),
    "right_pinky1": (0.950, 0.00, -0.04, 0.31),
    "right_pinky2": (0.975, 0.00, 0.00, 0.20),
    "right_pinky3": (0.990, 0.00, 0.00, 0.10),
}


def log(msg: str) -> None:
    print(f"[bake_glb] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--action", default="walk_our_tpl_ik")
    p.add_argument("--out", required=True)
    p.add_argument("--armature", default="SMPL-X_Armature")
    p.add_argument("--elbow_inward", type=float, default=0.18)
    p.add_argument("--fix_hands", action="store_true", default=True)
    p.add_argument("--no_fix_hands", action="store_true")
    p.add_argument("--hand_curl", type=float, default=0.85)
    p.add_argument("--spine_colinear", type=float, default=1.0,
                   help="Force spine1/2/3 onto pelvis->neck line (0=off, 1=full)")
    p.add_argument("--walk_root", action="store_true", default=True,
                   help="Rebuild horizontal root from planted feet for forward motion")
    p.add_argument("--no_walk_root", action="store_true",
                   help="Disable walk root rebuild (stays in-place)")
    p.add_argument("--ik_iters", type=int, default=4,
                   help="Swing-aim passes per frame; more=closer to target (default 4)")
    return p.parse_args(argv)


def yup_to_zup(p: np.ndarray) -> np.ndarray:
    out = np.empty_like(p)
    out[..., 0] = p[..., 0]
    out[..., 1] = -p[..., 2]
    out[..., 2] = p[..., 1]
    return out


def rest_head_w(arm, n):
    return (arm.matrix_world @ arm.data.bones[n].head_local).copy()


def world_head(arm, n):
    return (arm.matrix_world @ arm.pose.bones[n].head).copy()


def clear_pose(arm):
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
        pb.location = Vector((0, 0, 0))


def set_pelvis(arm, desired: Vector):
    pb = arm.pose.bones["pelvis"]
    bone = arm.data.bones["pelvis"]
    rest = bone.matrix_local
    r_inv = rest.to_3x3().inverted()
    t = rest.to_translation()
    arm_inv = arm.matrix_world.inverted()
    desired_a = arm_inv @ desired
    pb.location = r_inv @ (desired_a - t)
    bpy.context.view_layer.update()
    for _ in range(10):
        err = desired - (arm.matrix_world @ pb.head)
        if err.length < 1e-5:
            break
        pb.location += r_inv @ (arm_inv.to_3x3() @ err)
        bpy.context.view_layer.update()


def swing_aim(arm, bone_name, child_name, desired_child_w: Vector):
    pb = arm.pose.bones[bone_name]
    if child_name not in arm.pose.bones:
        return
    saved = pb.location.copy()
    is_root = bone_name == "pelvis"
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
    pb.location = saved if is_root else Vector((0, 0, 0))
    bpy.context.view_layer.update()
    head = arm.matrix_world @ pb.head
    child_rest = arm.matrix_world @ arm.pose.bones[child_name].head
    v_rest = child_rest - head
    v_tgt = desired_child_w - head
    if v_rest.length < 1e-8 or v_tgt.length < 1e-8:
        return
    v_rest.normalize()
    v_tgt.normalize()
    if v_rest.dot(v_tgt) > 0.999999:
        return
    q = v_rest.rotation_difference(v_tgt)
    M = arm.matrix_world @ pb.matrix
    h = M.to_translation()
    M_new = Matrix.Translation(h) @ q.to_matrix().to_4x4() @ Matrix.Translation(-h) @ M
    mat_arm = arm.matrix_world.inverted() @ M_new
    bone = arm.data.bones[bone_name]
    if pb.parent:
        pre = pb.parent.matrix @ pb.parent.bone.matrix_local.inverted() @ bone.matrix_local
        basis = pre.inverted() @ mat_arm
    else:
        basis = bone.matrix_local.inverted() @ mat_arm
    _l, rot, _s = basis.decompose()
    pb.rotation_quaternion = rot
    pb.location = saved if is_root else Vector((0, 0, 0))
    bpy.context.view_layer.update()


def colinear_spine_zup(joints_z: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Force pelvis-spine1-spine2-spine3-neck onto the pelvis->neck line (Z-up, HML order)."""
    if amount <= 1e-6:
        return joints_z
    out = joints_z.copy()
    a = float(np.clip(amount, 0.0, 1.0))
    for t in range(out.shape[0]):
        p = out[t, 0]
        neck = out[t, 12]
        axis = neck - p
        L = float(np.linalg.norm(axis))
        if L < 1e-6:
            continue
        axis = axis / L
        for ji in (3, 6, 9):
            s = float(np.dot(out[t, ji] - p, axis))
            s = float(np.clip(s, 0.02 * L, 0.95 * L))
            on_line = p + s * axis
            out[t, ji] = (1.0 - a) * out[t, ji] + a * on_line
    return out


def walk_root_zup(joints_z: np.ndarray) -> tuple:
    """
    Rebuild horizontal root from planted feet so the body walks forward.
    Works on Z-up HML joint order: 0=pelvis, 7=left_ankle, 10=left_foot,
    8=right_ankle, 11=right_foot.  Horizontal plane = XY.
    """
    pos = np.asarray(joints_z, dtype=np.float64)
    T = pos.shape[0]
    rel = pos - pos[:, 0:1, :]

    def contact_mask(ankle_i: int, toe_i: int) -> np.ndarray:
        f = pos[:, ankle_i]
        toe = pos[:, toe_i]
        spd = np.zeros(T)
        spd[1:] = np.linalg.norm(np.diff(f[:, :2], axis=0), axis=1)
        spd[0] = spd[1] if T > 1 else 0.0
        h = np.minimum(f[:, 2], toe[:, 2])
        floor = float(np.percentile(h, 20))
        return (spd < 0.012) & (h < floor + 0.04)

    LF, RF = 7, 8
    c_l = contact_mask(LF, 10)
    c_r = contact_mask(RF, 11)
    for t in range(T):
        if not c_l[t] and not c_r[t]:
            if pos[t, LF, 2] <= pos[t, RF, 2]:
                c_l[t] = True
            else:
                c_r[t] = True

    pelvis = np.zeros((T, 3), dtype=np.float64)
    pelvis[0] = pos[0, 0]
    lock = {
        LF: (pelvis[0, :2] + rel[0, LF, :2]).copy(),
        RF: (pelvis[0, :2] + rel[0, RF, :2]).copy(),
    }
    was = {LF: bool(c_l[0]), RF: bool(c_r[0])}

    for t in range(1, T):
        cands = []
        for fi, c in ((LF, c_l[t]), (RF, c_r[t])):
            if c:
                if not was[fi]:
                    lock[fi] = pelvis[t - 1, :2] + rel[t, fi, :2]
                cands.append(lock[fi] - rel[t, fi, :2])
                was[fi] = True
            else:
                was[fi] = False
                lock[fi] = pelvis[t - 1, :2] + rel[t, fi, :2]
        if cands:
            pelvis[t, :2] = np.mean(np.stack(cands, 0), axis=0)
        else:
            pelvis[t, :2] = pelvis[t - 1, :2] + (
                pos[t, 0, :2] - pos[t - 1, 0, :2]
            )
        pelvis[t, 2] = pos[t, 0, 2]
        for fi, c in ((LF, c_l[t]), (RF, c_r[t])):
            if not c:
                lock[fi] = pelvis[t, :2] + rel[t, fi, :2]

    out = rel + pelvis[:, None, :]
    meta = {"contact_L": int(c_l.sum()), "contact_R": int(c_r.sum())}
    return out, meta


def slight_elbow_inward(j: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 1e-6:
        return j
    out = j.copy()
    a = max(0.0, min(0.5, float(amount)))
    for t in range(out.shape[0]):
        spine_x = float(out[t, 9, 0])
        for el_i, wr_i, sh_i in ((18, 20, 16), (19, 21, 17)):
            sh_x = float(out[t, sh_i, 0])
            inward_x = 0.75 * sh_x + 0.25 * spine_x
            out[t, el_i, 0] = (1.0 - a) * out[t, el_i, 0] + a * inward_x
            out[t, wr_i, 0] = (1.0 - 0.5 * a) * out[t, wr_i, 0] + (0.5 * a) * out[t, el_i, 0]
    return out


def apply_hands(arm, frame: int, curl: float, phase: float):
    identity = Quaternion((1, 0, 0, 0))
    curl_f = max(0.0, min(1.0, curl)) * (0.88 + 0.12 * phase)
    for side in ("left", "right"):
        wn = f"{side}_wrist"
        if wn not in arm.pose.bones:
            continue
        pb = arm.pose.bones[wn]
        pb.rotation_mode = "QUATERNION"
        sign = 1.0 if side == "left" else -1.0
        hang = Euler((0.05, 0.02 * sign * math.sin(phase * math.pi), -0.08 * sign), "XYZ").to_quaternion()
        pb.rotation_quaternion = pb.rotation_quaternion.slerp(hang, 0.55).normalized()
        pb.keyframe_insert("rotation_quaternion", frame=frame)
    for bn, qwxyz in RELAXED_HAND.items():
        if bn not in arm.pose.bones:
            continue
        pb = arm.pose.bones[bn]
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = identity.slerp(Quaternion(qwxyz), curl_f).normalized()
        pb.keyframe_insert("rotation_quaternion", frame=frame)


def assign_action_slot(arm, act):
    if not arm.animation_data:
        arm.animation_data_create()
    arm.animation_data.action = act
    try:
        slots = arm.animation_data.action_suitable_slots
        if slots:
            arm.animation_data.action_slot = slots[0]
    except Exception:
        pass


def main():
    args = parse_args()
    npz = Path(args.npz)
    if not npz.is_absolute():
        npz = ROOT / npz
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    fix_hands = bool(args.fix_hands) and not args.no_fix_hands
    elbow_in = max(0.0, min(0.5, float(args.elbow_inward)))
    sc = float(args.spine_colinear)
    do_walk_root = bool(args.walk_root) and not args.no_walk_root
    ik_iters = max(1, int(args.ik_iters))

    data = np.load(str(npz))
    glb = np.asarray(data["glb_hml_order"], dtype=np.float64)
    T = glb.shape[0]
    method = str(data["method"]) if "method" in data.files else "?"
    log(f"glb {glb.shape} method={method}")

    arm = bpy.data.objects.get(args.armature)
    if arm is None:
        raise SystemExit("need SMPL-X_Armature")

    joints_z = yup_to_zup(glb)

    if do_walk_root:
        before_travel = float(np.linalg.norm(joints_z[-1, 0, :2] - joints_z[0, 0, :2]))
        joints_z, wr_meta = walk_root_zup(joints_z)
        after_travel = float(np.linalg.norm(joints_z[-1, 0, :2] - joints_z[0, 0, :2]))
        log(f"walk_root_zup: travel {before_travel:.3f}->{after_travel:.3f}m "
            f"contacts L={wr_meta['contact_L']} R={wr_meta['contact_R']}")
    else:
        log("walk root disabled (in-place)")

    p0 = joints_z[0, 0].copy()
    pr = rest_head_w(arm, "pelvis")
    joints_z = joints_z + np.array([pr.x - p0[0], pr.y - p0[1], pr.z - p0[2]])

    if sc > 0:
        joints_z = colinear_spine_zup(joints_z, sc)
        log(f"colinear_spine_zup amount={sc}")

    joints_z = slight_elbow_inward(joints_z, elbow_in)

    travel = float(np.linalg.norm(joints_z[-1, 0, :2] - joints_z[0, 0, :2]))
    path = float(np.linalg.norm(np.diff(joints_z[:, 0, :2], axis=0), axis=1).sum())
    log(f"pelvis travel={travel:.3f}m path={path:.3f}m elbow_in={elbow_in} spine_colinear={sc}")

    foot_idxs = [7, 8, 10, 11]
    min_foot_z = float(np.min(joints_z[:, foot_idxs, 2]))
    if min_foot_z < 0.0:
        joints_z[:, :, 2] -= min_foot_z
        log(f"floor lift: raised all joints by {-min_foot_z:.4f}m so min foot Z=0")
    floor_z_target = float(np.percentile(joints_z[:, foot_idxs, 2], 2))
    log(f"floor_z_target={floor_z_target:.4f}m  ik_iters={ik_iters}")

    if not arm.animation_data:
        arm.animation_data_create()
    old = bpy.data.actions.get(args.action)
    if old:
        if arm.animation_data.action == old:
            arm.animation_data.action = None
        bpy.data.actions.remove(old)
    act = bpy.data.actions.new(args.action)
    act.use_fake_user = True
    assign_action_slot(arm, act)

    foot_bone_names = [bn for bn in ("left_ankle", "left_foot", "right_ankle", "right_foot")]

    samples = []
    spine_lat = []
    for t in range(T):
        f = t + 1
        clear_pose(arm)
        hips = Vector(joints_z[t, 0].tolist())
        set_pelvis(arm, hips)
        for _pass in range(ik_iters):
            for parent, child, ia, ib in EDGES:
                if parent in arm.pose.bones and child in arm.pose.bones:
                    swing_aim(arm, parent, child, Vector(joints_z[t, ib].tolist()))
            set_pelvis(arm, hips)

        active_feet = [bn for bn in foot_bone_names if bn in arm.pose.bones]
        if active_feet:
            min_fz = min(world_head(arm, bn).z for bn in active_feet)
            if min_fz < floor_z_target - 1e-4:
                dz = floor_z_target - min_fz
                set_pelvis(arm, Vector((hips.x, hips.y, hips.z + dz)))

        if fix_hands:
            phase = 0.5 + 0.5 * math.sin(2.0 * math.pi * t / max(1, T / 2.5))
            apply_hands(arm, f, float(args.hand_curl), phase)

        for bn in KEY:
            if bn not in arm.pose.bones:
                continue
            pb = arm.pose.bones[bn]
            pb.rotation_mode = "QUATERNION"
            pb.keyframe_insert("rotation_quaternion", frame=f)
            if bn == "pelvis":
                pb.keyframe_insert("location", frame=f)

        # spine colinearity check on posed bones
        names = ["pelvis", "spine1", "spine2", "spine3", "neck"]
        pts = [world_head(arm, n) for n in names if n in arm.pose.bones]
        if len(pts) >= 3:
            p0v, p1v = pts[0], pts[-1]
            axis = p1v - p0v
            if axis.length > 1e-6:
                axis.normalize()
                dsum = 0.0
                for q in pts[1:-1]:
                    w = q - p0v
                    dsum += (w - axis * w.dot(axis)).length
                spine_lat.append(dsum / max(1, len(pts) - 2))

        lc = world_head(arm, "left_collar")
        ls = world_head(arm, "left_shoulder")
        samples.append(ls.z - lc.z)
        if t == 0 or t % 20 == 0 or t == T - 1:
            pel = world_head(arm, "pelvis")
            log(f"  f{f}: pelvis=({pel.x:.2f},{pel.y:.2f},{pel.z:.2f}) Δcollar={ls.z-lc.z:.3f}")

    clear_pose(arm)
    bpy.context.view_layer.update()
    rest_dz = (world_head(arm, "left_shoulder") - world_head(arm, "left_collar")).z
    mean_dz = float(np.mean(samples))
    ok = mean_dz > 0.02 and abs(mean_dz - rest_dz) < 0.08
    mean_lat = float(np.mean(spine_lat)) if spine_lat else 0.0
    walk_ok = travel > 0.25 or path > 0.4

    log(f"VERIFY collar dz={mean_dz:.4f} rest={rest_dz:.4f} -> {'PASS' if ok else 'CHECK'}")
    log(f"VERIFY walk travel={travel:.3f} path={path:.3f} -> {'PASS' if walk_ok else 'CHECK'}")
    log(f"VERIFY spine lateral mean={mean_lat:.4f}m (lower=more colinear)")

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = T
    assign_action_slot(arm, act)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out.resolve()).replace("\\", "/"), compress=True)
    log(f"saved {out}")
    rep = {
        "action": args.action,
        "out": str(out),
        "collar_shoulder_ok": bool(ok),
        "L_dz_mean": mean_dz,
        "pelvis_travel_m": travel,
        "pelvis_path_m": path,
        "walk_ok": bool(walk_ok),
        "spine_lateral_mean_m": mean_lat,
        "elbow_inward": elbow_in,
        "spine_colinear": sc,
        "walk_root": do_walk_root,
        "ik_method": method,
        "method": "bake_momask_ik_walk_spine",
    }
    (out.parent / f"{args.action}_verify.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    log(json.dumps(rep, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
