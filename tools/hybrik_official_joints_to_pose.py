"""
Official HybrIK joints → SMPL-X parameters (NOT a custom IK reimplementation).

Uses third_party/HybrIK SMPLXLayer.hybrik() — the same hybrid analytical IK
the authors use for joints → rot_mats / SMPL-X pose.

Flow:
  MoMask (T,22,3) joints
    → scale to SMPL-X rest height
    → expand to 71-joint HybrIK-X skeleton (body from MoMask; hands/face/leaves = rest offsets)
    → phis = identity twist (no image network; cos=1,sin=0)
    → SMPLXLayer.hybrik(...)   # OFFICIAL
    → pose.npz with rot_mats (T,55,3,3), axis_angle, transl, joints_fk (official FK)

Bake onto avatar:
  blender ... --python tools/hybrik_apply_to_smplx.py -- \\
    --pose .../walk_hybrik_official_pose.npz --source fk --arm-aim src_target ...

Usage:
  python tools/hybrik_official_joints_to_pose.py \\
    --joints third_party/momask-codes/generation/.../sample0_repeat0_len80.npy \\
    --out body_motion/momask_cache/official_t2m_scale/walk_hybrik_official_pose.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HYBRIK_ROOT = ROOT / "third_party" / "HybrIK"
SMPLX_NPZ = (
    ROOT
    / "third_party"
    / "DART"
    / "data"
    / "smplx_lockedhead_20230207"
    / "models_lockedhead"
    / "smplx"
    / "SMPLX_NEUTRAL.npz"
)

# Install pytorch3d shim BEFORE importing HybrIK
sys.path.insert(0, str(ROOT / "tools"))
from _hybrik_official_shim import install as _install_p3d  # noqa: E402

_install_p3d()
sys.path.insert(0, str(HYBRIK_ROOT))

from hybrik.models.layers.smplx.body_models import SMPLXLayer  # noqa: E402
from hybrik.models.layers.smplx.lbs import vertices2joints  # noqa: E402


def log(msg: str) -> None:
    print(f"[hybrik_official] {msg}", flush=True)


def resolve_joints_path(jpath: Path, prefer_ik: bool) -> Path:
    if not prefer_ik:
        return jpath
    if jpath.stem.endswith("_ik"):
        return jpath
    ik = jpath.with_name(f"{jpath.stem}_ik{jpath.suffix}")
    if ik.is_file():
        log(f"using foot-IK joints: {ik.name}")
        return ik
    return jpath


def rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as SciR

    T, J, _, _ = R.shape
    return (
        SciR.from_matrix(R.reshape(-1, 3, 3))
        .as_rotvec()
        .reshape(T, J, 3)
        .astype(np.float32)
    )


def build_rest71(layer: SMPLXLayer) -> torch.Tensor:
    """Official rest skeleton: 55 regressor joints + leaf vertices → (1,71,3)."""
    J55 = vertices2joints(layer.J_regressor, layer.v_template.unsqueeze(0))
    leaf_idx = torch.tensor(layer.LEAF_INDICES, dtype=torch.long)
    leaf = layer.v_template[leaf_idx].unsqueeze(0)
    return torch.cat([J55, leaf], dim=1)


def expand_to_71(
    joints22: np.ndarray, rest71: torch.Tensor, parents: torch.Tensor
) -> torch.Tensor:
    """
    joints22: (B,22,3) already root-aligned to rest pelvis.
    Unknown joints (hands/face/leaves) = parent + rest offset (official tree).
    """
    B = joints22.shape[0]
    pose = rest71.expand(B, -1, -1).clone()
    pose[:, :22] = torch.from_numpy(joints22.astype(np.float32))
    for i in range(22, 71):
        p = int(parents[i].item())
        if p < 0:
            continue
        off = rest71[0, i] - rest71[0, p]
        pose[:, i] = pose[:, p] + off.unsqueeze(0)
    return pose


def main():
    ap = argparse.ArgumentParser(description="Official HybrIK SMPLXLayer.hybrik joints→pose")
    ap.add_argument("--joints", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-prefer-ik", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--naive",
        action="store_true",
        default=True,
        help="Use naive IK path (same as HybrIK training forward)",
    )
    args = ap.parse_args()

    if not SMPLX_NPZ.is_file():
        raise SystemExit(f"missing SMPL-X model {SMPLX_NPZ}")
    if not HYBRIK_ROOT.is_dir():
        raise SystemExit(f"missing HybrIK repo at {HYBRIK_ROOT}")

    jpath = Path(args.joints)
    if not jpath.is_absolute():
        jpath = ROOT / jpath
    jpath = resolve_joints_path(jpath, prefer_ik=not args.no_prefer_ik)
    joints = np.load(str(jpath)).astype(np.float64)
    if joints.ndim != 3 or joints.shape[1] != 22 or joints.shape[2] != 3:
        raise SystemExit(f"need (T,22,3), got {joints.shape}")
    T = joints.shape[0]

    log(f"loading official SMPLXLayer from {SMPLX_NPZ.name}")
    layer = SMPLXLayer(model_path=str(SMPLX_NPZ), num_betas=10, use_pca=False)
    layer.eval()
    rest71 = build_rest71(layer)
    parents = layer.extended_parents
    log(f"rest71={tuple(rest71.shape)} extended_parents={tuple(parents.shape)}")

    # Scale MoMask so hips→head matches official rest
    h_src = np.linalg.norm(joints[:, 15] - joints[:, 0], axis=-1).mean()
    h_rest = float(torch.norm(rest71[0, 15] - rest71[0, 0]).item())
    scale = float(h_rest / h_src) if h_src > 1e-6 else 1.0
    joints_s = joints * scale

    hips = joints_s[:, 0].copy()
    # root-relative, then place on rest pelvis (same as HybrIK meter-space skeleton)
    joints_ik = joints_s - hips[:, None, :] + rest71[0, 0].numpy()

    # transl so first hip at rest pelvis; later frames keep travel
    transl = hips - hips[0] + rest71[0, 0].numpy()

    all_R = []
    all_j55 = []
    batch = max(1, int(args.batch))
    log(f"running OFFICIAL layer.hybrik  frames={T} batch={batch} naive={args.naive}")

    with torch.no_grad():
        for s in range(0, T, batch):
            e = min(T, s + batch)
            b = e - s
            pose71 = expand_to_71(joints_ik[s:e], rest71, parents)
            betas = torch.zeros(b, 10, dtype=torch.float32)
            # HybrIK-X: 54 twist slots (body+hands+jaw chain); identity twist
            phis = torch.zeros(b, 54, 2, dtype=torch.float32)
            phis[..., 0] = 1.0
            out = layer.hybrik(
                betas=betas,
                pose_skeleton=pose71,
                phis=phis,
                return_verts=False,
                root_align=False,
                naive=bool(args.naive),
            )
            R = out.rot_mats  # (B,55,3,3)
            j55 = out.joints_55  # (B,55,3) official FK
            all_R.append(R.cpu().numpy())
            all_j55.append(j55.cpu().numpy())
            log(f"  official hybrik frames {s}-{e - 1}")

    rot_mats = np.concatenate(all_R, axis=0).astype(np.float32)  # T,55,3,3
    joints_55 = np.concatenate(all_j55, axis=0).astype(np.float32)
    # Put absolute root travel back onto FK joints
    joints_55 = joints_55 - joints_55[:, 0:1, :] + transl[:, None, :].astype(np.float32)
    joints_fk22 = joints_55[:, :22, :].copy()

    aa = rotmat_to_aa(rot_mats)  # T,55,3
    # body SMPL-X params: global_orient = aa[:,0], body_pose = aa[:,1:22] (21*3)
    global_orient = aa[:, 0, :]
    body_pose = aa[:, 1:22, :].reshape(T, -1)

    # Prove we differ from raw joints (official reprojection)
    raw_l = joints_s - joints_s[:, 0:1]
    fk_l = joints_fk22 - joints_fk22[:, 0:1]
    diff = np.linalg.norm(raw_l - fk_l, axis=-1)
    log(
        f"official FK vs raw joints mean Δ={diff.mean():.4f}m max={diff.max():.4f}m "
        f"scale×{scale:.4f}"
    )
    for name, idx in [
        ("L_shoulder", 16),
        ("spine3", 9),
        ("neck", 12),
        ("head", 15),
        ("L_elbow", 18),
    ]:
        log(f"  Δ {name}: mean={diff[:, idx].mean():.4f}m")

    hip_travel = float(np.linalg.norm(hips - hips[0], axis=-1).max())
    log(f"hip_travel_max={hip_travel:.4f}m method=official_SMPLXLayer.hybrik")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(out),
        # Official HybrIK outputs
        rot_mats=rot_mats,
        axis_angle=aa,
        global_orient=global_orient.astype(np.float32),
        body_pose=body_pose.astype(np.float32),
        transl=transl.astype(np.float32),
        joints_fk=joints_fk22,  # 22 body joints after official hybrik FK
        joints_55=joints_55,
        joints_scaled=joints_s.astype(np.float32),
        rest71=rest71[0].cpu().numpy().astype(np.float32),
        scale=np.float32(scale),
        joints_source=str(jpath),
        method="official_hybrik_SMPLXLayer.hybrik",
        coord="y_up",
        fps=np.float32(20.0),
        bake_joints_key="joints_fk",
        note="phis=identity_twist (no image net); body from MoMask22 expanded to 71",
    )
    meta = {
        "out": str(out),
        "joints": str(jpath),
        "frames": int(T),
        "scale": scale,
        "hip_travel_max_m": hip_travel,
        "method": "official_hybrik_SMPLXLayer.hybrik",
        "model": str(SMPLX_NPZ),
        "hybrik_repo": str(HYBRIK_ROOT),
        "mean_joint_delta_raw_vs_official_fk_m": float(diff.mean()),
        "note": "Uses third_party/HybrIK SMPLXLayer.hybrik — not custom IK",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"saved {out}")
    log("DONE")


if __name__ == "__main__":
    main()
