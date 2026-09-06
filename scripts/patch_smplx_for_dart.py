"""
Pad incomplete SMPL-X .npz files so `smplx.build_layer` can load them.

Our project NPZs have mesh + 55-joint LBS but lack hand-PCA / landmark arrays
that the official `smplx` package expects. DART only needs body pose + mesh;
hands/face PCA can be zeros for Phase 1 (default head, body motion).

Usage:
  py -3.11 scripts/patch_smplx_for_dart.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DART_SMPLX = (
    ROOT
    / "third_party"
    / "DART"
    / "data"
    / "smplx_lockedhead_20230207"
    / "models_lockedhead"
    / "smplx"
)
BACKUP_DIR = DART_SMPLX / "_unpatched_backup"

# SMPL-X hand: 15 joints * 3 = 45
HAND_DIM = 45
NUM_PCA = 45  # full basis size in file; runtime may slice to 12


def patch_one(path: Path) -> None:
    data = dict(np.load(str(path), allow_pickle=True))
    n_verts = int(data["v_template"].shape[0])
    changed = []

    def ensure(key: str, arr: np.ndarray) -> None:
        nonlocal changed
        if key not in data:
            data[key] = arr
            changed.append(key)

    # Hand PCA (identity + zero mean) — valid unused subspace
    eye = np.eye(HAND_DIM, dtype=np.float64)
    zero = np.zeros((HAND_DIM,), dtype=np.float64)
    ensure("hands_componentsl", eye.copy())
    ensure("hands_componentsr", eye.copy())
    ensure("hands_meanl", zero.copy())
    ensure("hands_meanr", zero.copy())

    # Landmarks (dummy single vertex) — face contour not used for body demo
    ensure("lmk_faces_idx", np.zeros((1,), dtype=np.int64))
    ensure("lmk_bary_coords", np.array([[1.0, 0.0, 0.0]], dtype=np.float64))
    ensure("dynamic_lmk_faces_idx", np.zeros((1, 1), dtype=np.int64))
    ensure("dynamic_lmk_bary_coords", np.array([[[1.0, 0.0, 0.0]]], dtype=np.float64))

    if not changed:
        print(f"  already complete: {path.name}")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    bak = BACKUP_DIR / path.name
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
        print(f"  backup → {bak.name}")

    np.savez(str(path), **data)
    print(f"  patched {path.name}: added {changed}")


def main() -> None:
    if not DART_SMPLX.is_dir():
        raise SystemExit(f"Missing {DART_SMPLX}; run scripts/setup_dart.ps1 first")
    print(f"Patching models in {DART_SMPLX}")
    for name in ("SMPLX_MALE.npz", "SMPLX_FEMALE.npz", "SMPLX_NEUTRAL.npz"):
        p = DART_SMPLX / name
        if p.is_file():
            patch_one(p)
        else:
            print(f"  skip missing {name}")

    # Smoke test (build_layer expects rotation matrices, not axis-angle)
    try:
        import smplx
        import torch

        model_root = DART_SMPLX.parent  # models_lockedhead
        m = smplx.build_layer(
            str(model_root),
            model_type="smplx",
            gender="male",
            ext="npz",
            num_pca_comps=12,
        )
        b = 1
        eye = torch.eye(3).view(1, 1, 3, 3)
        body = eye.expand(b, 21, 3, 3).contiguous()
        hands = eye.expand(b, 15, 3, 3).contiguous()
        out = m(
            betas=torch.zeros(b, 10),
            expression=torch.zeros(b, 10),
            body_pose=body,
            global_orient=eye,
            transl=torch.zeros(b, 3),
            left_hand_pose=hands,
            right_hand_pose=hands,
            jaw_pose=eye,
            leye_pose=eye,
            reye_pose=eye,
        )
        print(
            f"SMOKE OK verts={tuple(out.vertices.shape)} joints={tuple(out.joints.shape)}"
        )
    except Exception as e:
        print(f"SMOKE FAIL: {type(e).__name__}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
