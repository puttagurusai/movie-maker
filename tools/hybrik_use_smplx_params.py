"""
How to USE official HybrIK output (SMPL-X parameters) — general path, no custom avatar.

Input: official_smplx_params.npz from tools/hybrik_official_npy_to_mesh.py
  keys: global_orient (T,3), body_pose (T,63), transl (T,3), rot_mats, joints_fk, ...

This script:
  1) loads those params
  2) runs standard smplx forward (same family as HybrIK mesh)
  3) exports mesh .obj sequence and optional stick gif of FK joints

Usage:
  python tools/hybrik_use_smplx_params.py \\
    --params body_motion/momask_cache/t2m_walk_hybrik/hybrik_official_mesh/official_smplx_params.npz \\
    --out-dir body_motion/momask_cache/t2m_walk_hybrik/hybrik_use_demo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SMPLX_MODELS = (
    ROOT
    / "third_party"
    / "DART"
    / "data"
    / "smplx_lockedhead_20230207"
    / "models_lockedhead"
)


def log(msg: str) -> None:
    print(f"[hybrik_use] {msg}", flush=True)


def yup_to_blender_zup(v: np.ndarray) -> np.ndarray:
    out = np.empty_like(v)
    out[..., 0] = v[..., 0]
    out[..., 1] = -v[..., 2]
    out[..., 2] = v[..., 1]
    return out


def save_obj(path: Path, verts: np.ndarray, faces: np.ndarray, *, blender_zup: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    v = yup_to_blender_zup(verts) if blender_zup else np.asarray(verts, dtype=np.float64)
    v = v.copy()
    v[:, 2] -= v[:, 2].min()
    with path.open("w", encoding="utf-8") as f:
        f.write("# SMPL-X from HybrIK params (Blender Z-up)\n")
        for p in v:
            f.write(f"v {float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}\n")
        for tri in faces:
            f.write(f"f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")


def main():
    ap = argparse.ArgumentParser(description="Use HybrIK SMPL-X params the standard way")
    ap.add_argument("--params", required=True, help="official_smplx_params.npz")
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "body_motion" / "momask_cache" / "t2m_walk_hybrik" / "hybrik_use_demo"),
    )
    ap.add_argument("--every", type=int, default=10, help="export every Nth mesh frame")
    args = ap.parse_args()

    ppath = Path(args.params)
    if not ppath.is_absolute():
        ppath = ROOT / ppath
    d = np.load(str(ppath), allow_pickle=True)

    go = np.asarray(d["global_orient"], dtype=np.float32)   # (T,3)
    bp = np.asarray(d["body_pose"], dtype=np.float32)       # (T,63)
    tr = np.asarray(d["transl"], dtype=np.float32)          # (T,3)
    T = go.shape[0]
    log(f"loaded {ppath.name}")
    log(f"  method={d['method'] if 'method' in d.files else '?'}")
    log(f"  global_orient {go.shape}  body_pose {bp.shape}  transl {tr.shape}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import smplx

    model = smplx.create(
        str(SMPLX_MODELS),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=10,
        ext="npz",
        batch_size=1,
    )
    faces = model.faces.astype(np.int64)
    mesh_dir = out_dir / "mesh_from_params"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Standard SMPL-X usage of HybrIK params (one frame at a time for clarity)
    joints_out = []
    with torch.no_grad():
        for t in range(T):
            out = model(
                global_orient=torch.from_numpy(go[t : t + 1]),
                body_pose=torch.from_numpy(bp[t : t + 1]),
                transl=torch.from_numpy(tr[t : t + 1]),
                betas=torch.zeros(1, 10),
                expression=torch.zeros(1, model.num_expression_coeffs),
                jaw_pose=torch.zeros(1, 3),
                leye_pose=torch.zeros(1, 3),
                reye_pose=torch.zeros(1, 3),
                left_hand_pose=torch.zeros(1, 45),
                right_hand_pose=torch.zeros(1, 45),
                return_verts=True,
            )
            v = out.vertices[0].cpu().numpy()
            j = out.joints[0, :22].cpu().numpy()
            joints_out.append(j)
            if t % max(1, args.every) == 0 or t == T - 1:
                save_obj(mesh_dir / f"frame_{t:04d}.obj", v, faces)
                log(f"  mesh frame {t} → {mesh_dir / f'frame_{t:04d}.obj'}")

    joints_out = np.stack(joints_out, axis=0)
    np.savez_compressed(
        out_dir / "replay_from_params.npz",
        joints=joints_out.astype(np.float32),
        global_orient=go,
        body_pose=bp,
        transl=tr,
        note="Rebuilt with smplx.create(**hybrik_params) — standard usage",
    )

    # Save a tiny Python snippet for copy-paste
    snippet = '''# Minimal: use HybrIK SMPL-X params
import numpy as np, torch, smplx

d = np.load("official_smplx_params.npz")
go, bp, tr = d["global_orient"], d["body_pose"], d["transl"]  # (T,3), (T,63), (T,3)

model = smplx.create("path/to/smplx_models", model_type="smplx",
                     gender="neutral", use_pca=False, ext="npz")

t = 0  # frame
out = model(
    global_orient=torch.tensor(go[t:t+1]),
    body_pose=torch.tensor(bp[t:t+1]),
    transl=torch.tensor(tr[t:t+1]),
    return_verts=True,
)
verts = out.vertices[0].numpy()   # (V,3) mesh
joints = out.joints[0].numpy()    # joints including body
# faces = model.faces
'''
    (out_dir / "HOW_TO_USE.py.txt").write_text(snippet, encoding="utf-8")

    meta = {
        "params": str(ppath),
        "frames": T,
        "hip_travel_m": float(np.linalg.norm(tr[-1] - tr[0])),
        "mesh_dir": str(mesh_dir),
        "usage": [
            "1. Load global_orient, body_pose, transl from npz",
            "2. smplx.create(...).forward with those tensors each frame",
            "3. Use vertices + model.faces for mesh (render / export OBJ/FBX)",
            "4. Do NOT swing-aim onto a custom rig unless you write a proper retargeter",
        ],
    }
    (out_dir / "USAGE.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"done → {out_dir}")
    log("Open mesh_from_params/*.obj in Blender (File→Import→Wavefront OBJ) to see official mesh")


if __name__ == "__main__":
    main()
