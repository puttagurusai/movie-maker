"""
Test official MoMask Joint2BVHConvertor with OUR template:
  - walk root motion
  - bone-axis twist (should be small, like stock template)

Usage (from project root, momask_venv):
  python tools/test_momask_template_twist.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOMASK = ROOT / "third_party" / "momask-codes"
JOINTS = ROOT / "body_motion" / "momask_cache" / "t2m_walk_hybrik" / "walk_t2m_joints.npy"
OUT_DIR = ROOT / "body_motion" / "momask_cache" / "t2m_walk_hybrik"


def twist_angles(anim) -> np.ndarray:
    """
    Per-joint mean |twist| deg around rest bone axis (parent→child offset).
    Twist = rotation component about the bone's rest direction.
    """
    from visualization.Quaternions import Quaternions

    parents = anim.parents
    offsets = anim.offsets
    # rotations: Quaternions (T, J)
    qs = anim.rotations.qs  # (T,J,4) wxyz
    T, J, _ = qs.shape
    twists = np.zeros((T, J), dtype=np.float64)
    for j in range(1, J):
        axis = offsets[j].astype(np.float64)
        n = np.linalg.norm(axis)
        if n < 1e-8:
            continue
        axis = axis / n
        # quaternion (w,x,y,z) → twist about axis:
        # q = [cos(a/2), sin(a/2)*u]; twist = 2*atan2(dot(v,axis), w) after swing-twist
        w = qs[:, j, 0]
        v = qs[:, j, 1:4]
        # swing-twist decomp: twist_axis = axis
        # t = [dot(v,axis), w] related: twist_angle = 2 * atan2(dot(v, axis), w)
        d = v @ axis
        ang = 2.0 * np.arctan2(d, w)
        # wrap to [-pi,pi]
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        twists[:, j] = np.degrees(np.abs(ang))
    return twists


def run_convert(template_rel: str, joints: np.ndarray, out_bvh: Path, label: str):
    import visualization.BVH_mod as BVH
    import visualization.Animation as Animation
    from visualization.InverseKinematics import BasicInverseKinematics
    from visualization.Quaternions import Quaternions
    from visualization.remove_fs import remove_fs

    re_order = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]
    re_order_inv = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 18, 13, 15, 19, 16, 20, 17, 21]

    template = BVH.load(template_rel, need_quater=True)
    positions = joints[:, re_order].copy()
    new_anim = template.copy()
    new_anim.rotations = Quaternions.id(positions.shape[:-1])
    new_anim.positions = new_anim.positions[0:1].repeat(positions.shape[0], axis=0)
    new_anim.positions[:, 0] = positions[:, 0]
    positions = remove_fs(
        positions, None, fid_l=(3, 4), fid_r=(7, 8), interp_length=5, force_on_floor=True
    )
    new_anim.positions[:, 0] = positions[:, 0]
    ik = BasicInverseKinematics(new_anim, positions, iterations=10, silent=True)
    new_anim = ik()
    glb = Animation.positions_global(new_anim)[:, re_order_inv]

    twists = twist_angles(new_anim)
    mean_tw = twists[:, 1:].mean(axis=0)  # skip root
    names = list(new_anim.names)
    travel = float(np.linalg.norm(glb[-1, 0, [0, 2]] - glb[0, 0, [0, 2]]))
    path = float(np.linalg.norm(np.diff(glb[:, 0, [0, 2]], axis=0), axis=1).sum())

    # spine colinearity
    p, n = glb[:, 0], glb[:, 12]
    ax = n - p
    ax = ax / (np.linalg.norm(ax, axis=-1, keepdims=True) + 1e-9)
    lat = []
    for ji in (3, 6, 9):
        w = glb[:, ji] - p
        proj = (w * ax).sum(-1, keepdims=True) * ax
        lat.append(float(np.linalg.norm(w - proj, axis=-1).mean()))

    print(f"\n=== {label} ===")
    print(f"template bones: {names[:4]}... root_off={template.offsets[0]}")
    print(f"travel={travel:.3f}m path={path:.3f}m")
    print(f"spine lateral s1/s2/s3 m = {lat[0]:.4f}/{lat[1]:.4f}/{lat[2]:.4f}")
    print(f"mean |twist| deg over joints: {mean_tw.mean():.2f}  max_joint={mean_tw.max():.2f}")
    # top twisting joints
    order = np.argsort(-mean_tw)
    print("top twist joints:")
    for i in order[:8]:
        print(f"  {names[i+1]:20s} {mean_tw[i]:.2f}°")  # +1 skip root in mean_tw indexing

    # fix indexing: mean_tw[j] corresponds to joint j+1
    # recount properly
    mean_all = twists.mean(axis=0)
    order = np.argsort(-mean_all)
    print("top twist (all frames mean):")
    for i in order[:10]:
        if i == 0:
            continue
        print(f"  {names[i]:20s} {mean_all[i]:.2f}°")

    out_bvh.parent.mkdir(parents=True, exist_ok=True)
    BVH.save(str(out_bvh), new_anim, names=new_anim.names, frametime=1 / 20, order="zyx", quater=True)
    print(f"wrote {out_bvh}")

    return {
        "label": label,
        "travel": travel,
        "path": path,
        "twist_mean": float(mean_all[1:].mean()),
        "twist_max": float(mean_all[1:].max()),
        "spine_lat": lat,
        "names": names,
    }


def main():
    os.chdir(MOMASK)
    if str(MOMASK) not in sys.path:
        sys.path.insert(0, str(MOMASK))

    joints = np.load(JOINTS).astype(np.float64)
    print("joints", joints.shape)

    r_stock = run_convert(
        "./visualization/data/template.bvh",
        joints,
        OUT_DIR / "walk_stock_template_ik.bvh",
        "STOCK MoMask template.bvh",
    )
    r_ours = run_convert(
        "./visualization/data/our_momask_template.bvh",
        joints,
        OUT_DIR / "walk_our_official_ik.bvh",
        "OUR our_momask_template.bvh",
    )

    print("\n======== SUMMARY ========")
    for r in (r_stock, r_ours):
        print(
            f"{r['label']}: travel={r['travel']:.3f} path={r['path']:.3f} "
            f"twist_mean={r['twist_mean']:.2f}° twist_max={r['twist_max']:.2f}° "
            f"spine_lat={r['spine_lat']}"
        )
    # Pass if our twist is not much worse than stock
    ok = r_ours["twist_mean"] <= r_stock["twist_mean"] * 1.5 + 2.0
    print(f"\nTWIST CHECK vs stock: {'PASS' if ok else 'CHECK'} "
          f"(ours {r_ours['twist_mean']:.2f} vs stock {r_stock['twist_mean']:.2f})")
    print("DONE")


if __name__ == "__main__":
    main()
