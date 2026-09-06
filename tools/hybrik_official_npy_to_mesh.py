"""
Official HybrIK path from joints npy ONLY (as the repo IK is designed).

IMPORTANT (README clarity):
  - scripts/demo_video*.py need pretrained HRNet on *images/video*, not npy.
  - For already-known 3D joints, the official conversion is:
        SMPLXLayer.hybrik(pose_skeleton, phis, betas)  in hybrik/models/layers/smplx/

This script:
  1) loads MoMask (T,22,3) npy
  2) plots input stick figure
  3) runs official HybrIK-X SMPLXLayer.hybrik (not custom IK)
  4) plots official FK joints + saves SMPL-X mesh frames + params

Usage:
  python tools/hybrik_official_npy_to_mesh.py \\
    --joints body_motion/momask_cache/t2m_walk_hybrik/walk_t2m_joints.npy \\
    --out-dir body_motion/momask_cache/t2m_walk_hybrik/hybrik_official_mesh
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
HYBRIK_ROOT = ROOT / "third_party" / "HybrIK"
# Prefer official HybrIK repo model_files (README package); fallback to DART copy
_SMPLX_CANDIDATES = [
    HYBRIK_ROOT / "model_files" / "smplx" / "SMPLX_NEUTRAL.npz",
    ROOT
    / "third_party"
    / "DART"
    / "data"
    / "smplx_lockedhead_20230207"
    / "models_lockedhead"
    / "smplx"
    / "SMPLX_NEUTRAL.npz",
]
SMPLX_NPZ = next((p for p in _SMPLX_CANDIDATES if p.is_file()), _SMPLX_CANDIDATES[0])

# SMPL-X / HumanML3D body parents (joints 0..21)
BODY_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)

sys.path.insert(0, str(ROOT / "tools"))
from _hybrik_official_shim import install as _install_p3d  # noqa: E402

_install_p3d()
sys.path.insert(0, str(HYBRIK_ROOT))

from hybrik.models.layers.smplx.body_models import SMPLXLayer  # noqa: E402
from hybrik.models.layers.smplx.lbs import vertices2joints  # noqa: E402

# HumanML3D kinematic chain for stick plots
HML_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]


def log(msg: str) -> None:
    print(f"[hybrik_official_mesh] {msg}", flush=True)


def plot_joints_gif(joints: np.ndarray, out: Path, title: str, step: int = 2) -> None:
    """joints (T,22,3) Y-up → gif"""
    j = joints
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def xyz(f):
        p = j[f]
        return p[:, 0], p[:, 2], p[:, 1]  # X, Z, Y

    xs, ys, zs = xyz(0)
    sc = ax.scatter(xs, ys, zs, c="r", s=18)
    lines = [ax.plot([], [], [], "b-", lw=2)[0] for _ in HML_CHAIN]
    r = 1.0
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(0, 2)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")

    def update(f):
        xs, ys, zs = xyz(f)
        sc._offsets3d = (xs, ys, zs)
        for ln, ch in zip(lines, HML_CHAIN):
            ln.set_data(xs[ch], ys[ch])
            ln.set_3d_properties(zs[ch])
        ax.view_init(elev=15, azim=40 + f)
        return [sc] + lines

    frames = list(range(0, len(j), step))
    anim = FuncAnimation(fig, update, frames=frames, interval=50)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=10))
    plt.close(fig)
    log(f"plot {out} ({out.stat().st_size} bytes)")


def yup_to_blender_zup(v: np.ndarray) -> np.ndarray:
    """SMPL/HybrIK Y-up → Blender Z-up: (x,y,z) → (x, -z, y)."""
    out = np.empty_like(v)
    out[..., 0] = v[..., 0]
    out[..., 1] = -v[..., 2]
    out[..., 2] = v[..., 1]
    return out


def save_obj(path: Path, verts: np.ndarray, faces: np.ndarray, *, blender_zup: bool = False) -> None:
    """Write Wavefront OBJ. Default keeps SMPL/HybrIK Y-up (native facing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    v = yup_to_blender_zup(verts) if blender_zup else np.asarray(verts, dtype=np.float64).copy()
    # feet on floor: Y-up → min Y; Blender Z-up → min Z
    up = 2 if blender_zup else 1
    v[:, up] -= v[:, up].min()
    with path.open("w", encoding="utf-8") as f:
        f.write("# HybrIK SMPL-X mesh Y-up (native)\n" if not blender_zup else "# HybrIK mesh Blender Z-up\n")
        for p in v:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")


def retarget_directions_to_rest_lengths(
    joints22: np.ndarray, rest22: np.ndarray
) -> np.ndarray:
    """
    Critical for correct HybrIK mesh: feed a skeleton that uses
    SOURCE directions but REST bone lengths (what HybrIK's IK expects).

    MoMask limb lengths ≠ SMPL-X rest lengths. Dumping raw MoMask positions
    into hybrik() can leave joints looking OK as a stick figure while the
    LBS mesh (spine/neck/head) looks bent or twisted.
    """
    j = joints22 - joints22[0]
    r = rest22 - rest22[0]
    out = np.zeros_like(j)
    out[0] = 0.0
    for i in range(1, 22):
        p = int(BODY_PARENTS[i])
        src = j[i] - j[p]
        n = float(np.linalg.norm(src))
        rest_vec = r[i] - r[p]
        L = float(np.linalg.norm(rest_vec))
        if n < 1e-8 or L < 1e-8:
            out[i] = out[p] + rest_vec
        else:
            out[i] = out[p] + (src / n) * L
    return out + joints22[0]


def expand_to_71(joints22: np.ndarray, rest71: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """joints22 already rest-length retargeted; fill hands/face/leaves from rest offsets."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", required=True, help="(T,22,3) npy from MoMask t2m")
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "body_motion" / "momask_cache" / "t2m_walk_hybrik" / "hybrik_official_mesh"),
    )
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    jpath = Path(args.joints)
    if not jpath.is_absolute():
        jpath = ROOT / jpath
    joints = np.load(str(jpath)).astype(np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise SystemExit(f"need (T,22,3), got {joints.shape}")
    T = joints.shape[0]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"input npy {jpath} shape={joints.shape}")
    log(f"hip travel={float(np.linalg.norm(joints[-1,0]-joints[0,0])):.4f}m")

    # ---- 1) plot INPUT npy first ----
    plot_joints_gif(joints, out_dir / "01_input_joints.npy.gif", "1) INPUT t2m joints npy")
    # mid frame png
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    mid = T // 2
    p = joints[mid]
    ax.scatter(p[:, 0], p[:, 2], p[:, 1], c="r", s=20)
    for ch in HML_CHAIN:
        ax.plot(p[ch, 0], p[ch, 2], p[ch, 1], "b-", lw=2)
    ax.set_title(f"input npy frame {mid}")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)
    fig.savefig(out_dir / "01_input_joints_mid.png", dpi=120)
    plt.close(fig)

    # ---- 2) official HybrIK SMPLXLayer.hybrik ----
    log(f"loading official SMPLXLayer  model={SMPLX_NPZ.name}")
    layer = SMPLXLayer(model_path=str(SMPLX_NPZ), num_betas=10, use_pca=False)
    layer.eval()
    J55 = vertices2joints(layer.J_regressor, layer.v_template.unsqueeze(0))
    leaf_idx = torch.tensor(layer.LEAF_INDICES, dtype=torch.long)
    leaf = layer.v_template[leaf_idx].unsqueeze(0)
    rest71 = torch.cat([J55, leaf], dim=1)
    parents = layer.extended_parents
    faces = np.asarray(layer.faces, dtype=np.int64)

    rest22 = rest71[0, :22].cpu().numpy()
    h_src = np.linalg.norm(joints[:, 15] - joints[:, 0], axis=-1).mean()
    h_rest = float(np.linalg.norm(rest22[15] - rest22[0]))
    scale = float(h_rest / h_src) if h_src > 1e-6 else 1.0
    joints_s = joints * scale

    # Direction-preserving, rest-length skeleton (official-IK friendly)
    joints_rl = np.stack(
        [retarget_directions_to_rest_lengths(joints_s[t], rest22) for t in range(T)],
        axis=0,
    )
    hips = joints_rl[:, 0].copy()
    # place on rest pelvis origin for hybrik
    joints_ik = joints_rl - hips[:, None, :] + rest22[0]
    transl = hips - hips[0] + rest22[0]

    all_R, all_j55, all_verts = [], [], []
    batch = max(1, args.batch)
    log(f"OFFICIAL layer.hybrik  model={SMPLX_NPZ}  T={T} batch={batch}")
    log("pre-IK: MoMask directions × SMPL-X rest bone lengths (not raw MoMask lengths)")
    with torch.no_grad():
        for s in range(0, T, batch):
            e = min(T, s + batch)
            b = e - s
            pose71 = expand_to_71(joints_ik[s:e], rest71, parents)
            betas = torch.zeros(b, 10)
            phis = torch.zeros(b, 54, 2)
            phis[..., 0] = 1.0
            out = layer.hybrik(
                betas=betas,
                pose_skeleton=pose71,
                phis=phis,
                return_verts=True,
                root_align=False,
                naive=True,
            )
            all_R.append(out.rot_mats.cpu().numpy())
            all_j55.append(out.joints_55.cpu().numpy())
            all_verts.append(out.vertices.cpu().numpy())
            log(f"  frames {s}-{e-1}")

    rot_mats = np.concatenate(all_R, axis=0).astype(np.float32)
    j55 = np.concatenate(all_j55, axis=0).astype(np.float32)
    verts = np.concatenate(all_verts, axis=0).astype(np.float32)

    # Same root travel on joints and mesh (keep Y-up; do not flip axes for OBJ)
    pelvis0 = j55[:, 0:1, :].copy()
    j55 = j55 - pelvis0 + transl[:, None, :].astype(np.float32)
    verts = verts - pelvis0 + transl[:, None, :].astype(np.float32)
    joints_fk = j55[:, :22].copy()

    # ---- 3) plot OFFICIAL FK joints ----
    plot_joints_gif(
        joints_fk,
        out_dir / "02_official_hybrik_fk_joints.gif",
        "2) OFFICIAL HybrIK FK joints (SMPLXLayer.hybrik)",
    )

    # side-by-side mid frame
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": "3d"})
    for ax, data, title in [
        (axes[0], joints_s - joints_s[mid : mid + 1, 0:1, :], "INPUT npy (root-rel)"),
        (axes[1], joints_fk - joints_fk[mid : mid + 1, 0:1, :], "OFFICIAL HybrIK FK"),
    ]:
        p = data[mid]
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], c="r", s=16)
        for ch in HML_CHAIN:
            ax.plot(p[ch, 0], p[ch, 2], p[ch, 1], "b-", lw=2)
        ax.set_title(title)
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(-0.2, 1.6)
    fig.tight_layout()
    fig.savefig(out_dir / "03_compare_input_vs_official_fk.png", dpi=140)
    plt.close(fig)
    log(f"saved compare png")

    # ---- 4) mesh overlay diagnostic (mesh cloud + joints, same frame) ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={"projection": "3d"})
    for ax, fi in zip(axes, [0, mid, T - 1]):
        vv = verts[fi]
        idx = np.linspace(0, len(vv) - 1, 900).astype(int)
        ax.scatter(vv[idx, 0], vv[idx, 2], vv[idx, 1], s=1, c="0.6", alpha=0.35)
        p = joints_fk[fi]
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], c="r", s=18)
        for ch in HML_CHAIN:
            ax.plot(p[ch, 0], p[ch, 2], p[ch, 1], "b-", lw=2)
        ax.set_title(f"mesh+joints f{fi}")
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-0.9, 0.9)
        ax.set_zlim(-0.2, 1.8)
    fig.tight_layout()
    fig.savefig(out_dir / "04_mesh_vs_joints_overlay.png", dpi=140)
    plt.close(fig)
    log("saved 04_mesh_vs_joints_overlay.png")

    # ---- 5) save mesh samples Y-up (SMPL native; facing as HybrIK) ----
    mesh_yup = out_dir / "mesh_obj_yup"
    for fi in [0, mid, T - 1, T // 4, 3 * T // 4]:
        v = verts[fi].copy()
        v[:, 1] -= v[:, 1].min()
        save_obj(mesh_yup / f"frame_{fi:04d}.obj", v, faces, blender_zup=False)
        log(f"mesh Y-up {mesh_yup / f'frame_{fi:04d}.obj'}")

    # SMPL-X params
    from scipy.spatial.transform import Rotation as SciR

    aa = (
        SciR.from_matrix(rot_mats.reshape(-1, 3, 3))
        .as_rotvec()
        .reshape(T, 55, 3)
        .astype(np.float32)
    )
    np.savez_compressed(
        out_dir / "official_smplx_params.npz",
        method="official_hybrik_SMPLXLayer.hybrik",
        global_orient=aa[:, 0],
        body_pose=aa[:, 1:22].reshape(T, -1),
        transl=transl.astype(np.float32),
        rot_mats=rot_mats,
        axis_angle=aa,
        joints_fk=joints_fk,
        joints_input=joints.astype(np.float32),
        joints_scaled=joints_s.astype(np.float32),
        vertices=verts,  # centered for viz; full mesh from hybrik
        scale=np.float32(scale),
        joints_source=str(jpath),
        note="Official HybrIK IK only. README demos are video; this is joints-npy path via layer.hybrik()",
    )

    diff = np.linalg.norm(
        (joints_s - joints_s[:, 0:1]) - (joints_fk - joints_fk[:, 0:1]), axis=-1
    )
    meta = {
        "input_npy": str(jpath),
        "method": "official_hybrik_SMPLXLayer.hybrik",
        "frames": T,
        "scale": scale,
        "mean_fk_vs_input_m": float(diff.mean()),
        "outputs": {
            "input_gif": str(out_dir / "01_input_joints.npy.gif"),
            "official_fk_gif": str(out_dir / "02_official_hybrik_fk_joints.gif"),
            "compare_png": str(out_dir / "03_compare_input_vs_official_fk.png"),
            "params": str(out_dir / "official_smplx_params.npz"),
            "mesh_dir": str(mesh_yup),
            "overlay_png": str(out_dir / "04_mesh_vs_joints_overlay.png"),
        },
        "readme_note": (
            "HybrIK README pretrained models are for image/video demos. "
            "Joints→SMPL-X uses SMPLXLayer.hybrik() in the same repo."
        ),
    }
    (out_dir / "REPORT.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"REPORT {out_dir / 'REPORT.json'}")
    log("DONE — open GIFs in out-dir (do NOT judge via custom Blender swing bake)")


if __name__ == "__main__":
    main()
