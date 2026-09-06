"""
MoMask Joint2BVHConvertor.convert() with OUR SMPL-X template.

Extra (required for usable SMPL-X bake):
  1) After remove_fs foot locks → rebuild root from planted feet so the
     clip actually walks (HML joints are often in-place; MoMask foot IK
     freezes feet but does not move the pelvis — root must follow locks).
  2) Colinear pelvis → spine → neck targets before BasicIK.

Usage:
  python tools/momask_joints_ik_our_template.py \\
    --joints .../walk_t2m_joints.npy \\
    --template third_party/momask-codes/visualization/data/our_momask_template.bvh \\
    --out .../walk_our_template_ik.bvh
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOMASK = ROOT / "third_party" / "momask-codes"

RE_ORDER = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]
RE_ORDER_INV = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 18, 13, 15, 19, 16, 20, 17, 21]
PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]

# BVH-order spine chain: Hips, Spine, Spine1, Spine2, Neck, Head
SPINE_BVH = [0, 9, 10, 11, 12, 13]


def log(msg: str) -> None:
    print(f"[momask_ik] {msg}", flush=True)


def colinear_spine_bvh(pos: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Force Hips–Spine–Spine1–Spine2–Neck on the Hips→Head line (Y-up BVH)."""
    if amount <= 1e-6:
        return pos
    out = pos.copy()
    a = float(np.clip(amount, 0.0, 1.0))
    for t in range(out.shape[0]):
        p = out[t, 0]
        head = out[t, 13]
        axis = head - p
        L = float(np.linalg.norm(axis))
        if L < 1e-6:
            continue
        axis = axis / L
        # Place chain joints by original arc-length parameter along axis
        for ji in (9, 10, 11, 12):  # Spine, Spine1, Spine2, Neck
            s = float(np.dot(out[t, ji] - p, axis))
            s = float(np.clip(s, 0.02 * L, 0.95 * L))
            on_line = p + s * axis
            out[t, ji] = (1.0 - a) * out[t, ji] + a * on_line
    return out


def root_from_momask_foot_locks(pos: np.ndarray, fid_l=(3, 4), fid_r=(7, 8)) -> np.ndarray:
    """
    MoMask remove_fs freezes contact feet but leaves pelvis in-place.
    Rebuild horizontal root so planted feet stay fixed and the body walks.

    pos: (T,22,3) BVH order, Y-up, AFTER remove_fs.
    """
    pos = np.asarray(pos, dtype=np.float64)
    T = pos.shape[0]
    # Relative pose to original pelvis (keeps MoMask leg/arm cycle)
    rel = pos - pos[:, 0:1, :]

    def contact_mask(fids):
        # Stationary + near floor after remove_fs (Y-up: floor ~ 0)
        f = pos[:, fids[0]]  # ankle
        toe = pos[:, fids[1]]
        spd = np.zeros(T)
        spd[1:] = np.linalg.norm(np.diff(f[:, [0, 2]], axis=0), axis=1)
        spd[0] = spd[1] if T > 1 else 0.0
        h = np.minimum(f[:, 1], toe[:, 1])
        floor = float(np.percentile(h, 20))
        return (spd < 0.012) & (h < floor + 0.04)

    c_l = contact_mask(fid_l)
    c_r = contact_mask(fid_r)
    for t in range(T):
        if not c_l[t] and not c_r[t]:
            if pos[t, fid_l[0], 1] <= pos[t, fid_r[0], 1]:
                c_l[t] = True
            else:
                c_r[t] = True

    LF, RF = fid_l[0], fid_r[0]  # use ankle for root solve
    pelvis = np.zeros((T, 3), dtype=np.float64)
    pelvis[0] = pos[0, 0]
    lock = {
        LF: (pelvis[0, [0, 2]] + rel[0, LF, [0, 2]]).copy(),
        RF: (pelvis[0, [0, 2]] + rel[0, RF, [0, 2]]).copy(),
    }
    was = {LF: bool(c_l[0]), RF: bool(c_r[0])}

    for t in range(1, T):
        cands = []
        for fi, c in ((LF, c_l[t]), (RF, c_r[t])):
            if c:
                if not was[fi]:
                    lock[fi] = pelvis[t - 1, [0, 2]] + rel[t, fi, [0, 2]]
                cands.append(lock[fi] - rel[t, fi, [0, 2]])
                was[fi] = True
            else:
                was[fi] = False
                lock[fi] = pelvis[t - 1, [0, 2]] + rel[t, fi, [0, 2]]
        if cands:
            pelvis[t, [0, 2]] = np.mean(np.stack(cands, 0), axis=0)
        else:
            pelvis[t, [0, 2]] = pelvis[t - 1, [0, 2]] + (
                pos[t, 0, [0, 2]] - pos[t - 1, 0, [0, 2]]
            )
        pelvis[t, 1] = pos[t, 0, 1]
        for fi, c in ((LF, c_l[t]), (RF, c_r[t])):
            if not c:
                lock[fi] = pelvis[t, [0, 2]] + rel[t, fi, [0, 2]]

    out = rel + pelvis[:, None, :]
    return out, {"contact_L": int(c_l.sum()), "contact_R": int(c_r.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", required=True)
    ap.add_argument(
        "--template",
        default=str(
            ROOT
            / "third_party"
            / "momask-codes"
            / "visualization"
            / "data"
            / "our_momask_template.bvh"
        ),
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--no-foot-ik", action="store_true")
    ap.add_argument("--spine_colinear", type=float, default=1.0)
    ap.add_argument(
        "--no-walk-root",
        action="store_true",
        help="Disable contact root (stay in-place like raw HML)",
    )
    args = ap.parse_args()

    jpath = Path(args.joints)
    if not jpath.is_absolute():
        jpath = ROOT / jpath
    tpath = Path(args.template)
    if not tpath.is_absolute():
        tpath = ROOT / tpath
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    if not tpath.is_file():
        raise SystemExit(f"missing template {tpath}")

    positions_hml = np.load(str(jpath)).astype(np.float64)
    if positions_hml.ndim != 3 or positions_hml.shape[1] != 22:
        raise SystemExit(f"need (T,22,3), got {positions_hml.shape}")
    log(f"joints {positions_hml.shape}")

    os.chdir(MOMASK)
    if str(MOMASK) not in sys.path:
        sys.path.insert(0, str(MOMASK))

    import visualization.Animation as Animation
    import visualization.BVH_mod as BVH
    from visualization.InverseKinematics import BasicInverseKinematics
    from visualization.Quaternions import Quaternions
    from visualization.remove_fs import remove_fs

    template = BVH.load(str(tpath), need_quater=True)
    log(f"template={tpath.name} bones={len(template.names)}")

    # Uniform scale to our template height
    rest = np.zeros((22, 3))
    rest[0] = template.offsets[0]
    for i in range(1, 22):
        rest[i] = rest[PARENTS[i]] + template.offsets[i]
    h_tpl = float(np.linalg.norm(rest[13] - rest[0]))
    h_src = float(np.linalg.norm(positions_hml[:, 15] - positions_hml[:, 0], axis=-1).mean())
    scale = (h_tpl / h_src) if h_src > 1e-6 else 1.0
    positions_hml = positions_hml * scale
    log(f"scale×{scale:.4f}")

    # --- MoMask convert ---
    positions = positions_hml[:, RE_ORDER].copy()
    new_anim = template.copy()
    new_anim.rotations = Quaternions.id(positions.shape[:-1])
    new_anim.positions = new_anim.positions[0:1].repeat(positions.shape[0], axis=0)
    new_anim.positions[:, 0] = positions[:, 0]

    foot_ik = not args.no_foot_ik
    if foot_ik:
        positions = remove_fs(
            positions, None, fid_l=(3, 4), fid_r=(7, 8), interp_length=5, force_on_floor=True
        )
        log("remove_fs foot_ik=True")

    # Root from planted feet (walk). MoMask locks feet; pelvis must follow.
    if not args.no_walk_root:
        before = float(np.linalg.norm(positions[-1, 0, [0, 2]] - positions[0, 0, [0, 2]]))
        positions, meta = root_from_momask_foot_locks(positions)
        after = float(np.linalg.norm(positions[-1, 0, [0, 2]] - positions[0, 0, [0, 2]]))
        path = float(
            np.linalg.norm(np.diff(positions[:, 0, [0, 2]], axis=0), axis=1).sum()
        )
        log(
            f"walk root from foot locks: travel {before:.3f}->{after:.3f}m path={path:.3f}m "
            f"contacts L={meta['contact_L']} R={meta['contact_R']}"
        )
    else:
        log("walk root disabled (in-place)")

    # Spine colinear targets → IK matches a straight column
    sc = float(args.spine_colinear)
    if sc > 0:
        positions = colinear_spine_bvh(positions, amount=sc)
        log(f"spine colinear amount={sc}")

    new_anim.positions[:, 0] = positions[:, 0]
    ik_solver = BasicInverseKinematics(
        new_anim, positions, iterations=int(args.iters), silent=True
    )
    new_anim = ik_solver()
    log(f"BasicInverseKinematics iters={args.iters}")

    glb_bvh = Animation.positions_global(new_anim)
    glb_hml = glb_bvh[:, RE_ORDER_INV]

    dy = glb_bvh[:, 15, 1] - glb_bvh[:, 14, 1]
    travel = float(np.linalg.norm(glb_hml[-1, 0, [0, 2]] - glb_hml[0, 0, [0, 2]]))
    path = float(np.linalg.norm(np.diff(glb_hml[:, 0, [0, 2]], axis=0), axis=1).sum())
    log(f"AFTER IK L collar-sh dY={dy.mean():.4f} rest={rest[15,1]-rest[14,1]:.4f}")
    log(f"AFTER IK root travel={travel:.3f}m path={path:.3f}m")

    # Spine line check (lateral deviation of spine joints from pelvis→neck)
    p = glb_hml[:, 0]
    n = glb_hml[:, 12]
    axis = n - p
    axis = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-9)
    lat = []
    for ji in (3, 6, 9):
        w = glb_hml[:, ji] - p
        proj = (w * axis).sum(-1, keepdims=True) * axis
        lat.append(np.linalg.norm(w - proj, axis=-1).mean())
    log(f"spine lateral dev mean s1/s2/s3={lat[0]:.4f}/{lat[1]:.4f}/{lat[2]:.4f}m")

    BVH.save(
        str(out),
        new_anim,
        names=new_anim.names,
        frametime=1 / 20,
        order="zyx",
        quater=True,
    )
    log(f"wrote {out}")

    np.savez(
        out.with_suffix(".npz"),
        glb_bvh_order=glb_bvh.astype(np.float32),
        glb_hml_order=glb_hml.astype(np.float32),
        rest_heads=rest.astype(np.float32),
        scale=np.float32(scale),
        L_shoulder_collar_dy=dy.astype(np.float32),
        method="momask_ik_walk_root_spine_colinear",
        iters=np.int32(args.iters),
        foot_ik=np.bool_(foot_ik),
        travel_m=np.float32(travel),
        path_m=np.float32(path),
    )
    log("DONE")


if __name__ == "__main__":
    main()
