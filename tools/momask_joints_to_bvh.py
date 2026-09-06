"""Convert pipeline NPZ (or raw joints npy) → BVH using MoMask's official IK."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOMASK = ROOT / "third_party" / "momask-codes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="")
    ap.add_argument("--joints_npy", type=str, default="", help="raw (T,22,3) npy")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--ik", action="store_true", help="foot IK")
    ap.add_argument("--iterations", type=int, default=30)
    args = ap.parse_args()

    if not MOMASK.is_dir():
        raise SystemExit(f"missing {MOMASK}")
    os.chdir(MOMASK)
    sys.path.insert(0, str(MOMASK))

    if args.npz:
        d = np.load(args.npz, allow_pickle=True)
        joints = np.asarray(d["joints"], dtype=np.float32)
    elif args.joints_npy:
        joints = np.load(args.joints_npy).astype(np.float32)
    else:
        raise SystemExit("need --npz or --joints_npy")

    if joints.shape[-2:] != (22, 3):
        raise SystemExit(f"need (T,22,3), got {joints.shape}")

    from visualization.joints2bvh import Joint2BVHConvertor

    conv = Joint2BVHConvertor()
    out = Path(args.out)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    anim, glb = conv.convert(
        joints,
        filename=str(out),
        iterations=args.iterations,
        foot_ik=args.ik,
    )
    print(f"[bvh] wrote {out}  frames={joints.shape[0]} bones={anim.names}")
    print(f"[bvh] glb recon shape {glb.shape}")


if __name__ == "__main__":
    main()
