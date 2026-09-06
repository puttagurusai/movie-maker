"""
Drive official SMPL-X body from MoMask joints via official HybrIK API only.

Uses:
  - joints: MoMask (T,22,3) npy
  - model:  models/smplx/SMPLX_NEUTRAL.npz  (project official SMPL-X)
  - IK:     third_party/HybrIK SMPLXLayer.hybrik()  only

No custom swing retarget. No project SMPL-X_Armature.

Outputs under --out-dir:
  official_smplx_params.npz
  mesh_obj/*.obj  (Y-up native)
  stick gifs
  then optional Blender file with shape-key animated SMPL-X mesh

Usage:
  python tools/hybrik_drive_official_smplx.py \\
    --joints body_motion/momask_cache/t2m_walk_hybrik/walk_t2m_joints.npy \\
    --out-dir body_motion/momask_cache/t2m_walk_hybrik/drive_official_smplx

  blender --background --python tools/hybrik_drive_official_smplx.py -- \\
    --joints ... --out-dir ... --make-blend
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HYBRIK_ROOT = ROOT / "third_party" / "HybrIK"
DEFAULT_SMPLX = ROOT / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
DEFAULT_JOINTS = (
    ROOT / "body_motion" / "momask_cache" / "t2m_walk_hybrik" / "walk_t2m_joints.npy"
)

BODY_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)
HML_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]


def log(msg: str) -> None:
    print(f"[hybrik_drive] {msg}", flush=True)


def setup_hybrik():
    sys.path.insert(0, str(ROOT / "tools"))
    from _hybrik_official_shim import install

    install()
    sys.path.insert(0, str(HYBRIK_ROOT))
    from hybrik.models.layers.smplx.body_models import SMPLXLayer
    from hybrik.models.layers.smplx.lbs import vertices2joints

    return SMPLXLayer, vertices2joints


def retarget_directions_to_rest_lengths(joints22: np.ndarray, rest22: np.ndarray) -> np.ndarray:
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


def plot_gif(joints: np.ndarray, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    j = joints
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def xyz(f):
        p = j[f]
        return p[:, 0], p[:, 2], p[:, 1]

    xs, ys, zs = xyz(0)
    sc = ax.scatter(xs, ys, zs, c="r", s=18)
    lines = [ax.plot([], [], [], "b-", lw=2)[0] for _ in HML_CHAIN]
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)
    ax.set_title(title)

    def update(f):
        xs, ys, zs = xyz(f)
        sc._offsets3d = (xs, ys, zs)
        for ln, ch in zip(lines, HML_CHAIN):
            ln.set_data(xs[ch], ys[ch])
            ln.set_3d_properties(zs[ch])
        ax.view_init(elev=15, azim=40 + f)
        return [sc] + lines

    anim = FuncAnimation(fig, update, frames=range(0, len(j), 2), interval=50)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=10))
    plt.close(fig)
    log(f"gif {out}")


def save_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(verts, dtype=np.float64).copy()
    v[:, 1] -= v[:, 1].min()
    with path.open("w", encoding="utf-8") as f:
        f.write("# Official SMPL-X from HybrIK (Y-up)\n")
        for p in v:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")


def run_hybrik(joints_path: Path, smplx_path: Path, out_dir: Path, batch: int = 8):
    import torch

    SMPLXLayer, vertices2joints = setup_hybrik()
    if not smplx_path.is_file():
        raise SystemExit(f"missing SMPL-X model: {smplx_path}")
    joints = np.load(str(joints_path)).astype(np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise SystemExit(f"need (T,22,3), got {joints.shape}")
    T = joints.shape[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"joints {joints_path} shape={joints.shape}")
    log(f"smplx  {smplx_path}")
    plot_gif(joints, out_dir / "01_input_joints.gif", "input MoMask joints")

    layer = SMPLXLayer(model_path=str(smplx_path), num_betas=10, use_pca=False)
    layer.eval()
    J55 = vertices2joints(layer.J_regressor, layer.v_template.unsqueeze(0))
    leaf_idx = torch.tensor(layer.LEAF_INDICES, dtype=torch.long)
    rest71 = torch.cat([J55, layer.v_template[leaf_idx].unsqueeze(0)], dim=1)
    parents = layer.extended_parents
    faces = np.asarray(layer.faces, dtype=np.int64)
    rest22 = rest71[0, :22].cpu().numpy()

    h_src = np.linalg.norm(joints[:, 15] - joints[:, 0], axis=-1).mean()
    h_rest = float(np.linalg.norm(rest22[15] - rest22[0]))
    scale = float(h_rest / h_src) if h_src > 1e-6 else 1.0
    joints_s = joints * scale
    joints_rl = np.stack(
        [retarget_directions_to_rest_lengths(joints_s[t], rest22) for t in range(T)], 0
    )
    hips = joints_rl[:, 0].copy()
    joints_ik = joints_rl - hips[:, None, :] + rest22[0]
    transl = hips - hips[0] + rest22[0]

    log(f"scale×{scale:.4f}  official hybrik T={T}")
    all_R, all_j, all_v = [], [], []
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
            all_j.append(out.joints_55.cpu().numpy())
            all_v.append(out.vertices.cpu().numpy())
            log(f"  hybrik {s}-{e-1}")

    rot_mats = np.concatenate(all_R, 0).astype(np.float32)
    j55 = np.concatenate(all_j, 0).astype(np.float32)
    verts = np.concatenate(all_v, 0).astype(np.float32)
    pelvis = j55[:, 0:1, :].copy()
    j55 = j55 - pelvis + transl[:, None, :].astype(np.float32)
    verts = verts - pelvis + transl[:, None, :].astype(np.float32)
    joints_fk = j55[:, :22].copy()

    plot_gif(joints_fk, out_dir / "02_hybrik_fk_joints.gif", "official HybrIK FK")

    # overlay
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mid = T // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={"projection": "3d"})
    for ax, fi in zip(axes, [0, mid, T - 1]):
        vv = verts[fi]
        idx = np.linspace(0, len(vv) - 1, 900).astype(int)
        ax.scatter(vv[idx, 0], vv[idx, 2], vv[idx, 1], s=1, c="0.6", alpha=0.35)
        p = joints_fk[fi]
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], c="r", s=16)
        for ch in HML_CHAIN:
            ax.plot(p[ch, 0], p[ch, 2], p[ch, 1], "b-", lw=2)
        ax.set_title(f"official SMPL-X f{fi}")
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-0.9, 0.9)
        ax.set_zlim(-0.2, 1.8)
    fig.tight_layout()
    fig.savefig(out_dir / "03_mesh_joints_overlay.png", dpi=140)
    plt.close(fig)

    mesh_dir = out_dir / "mesh_obj"
    for fi in range(T):
        save_obj(mesh_dir / f"frame_{fi:04d}.obj", verts[fi], faces)
    log(f"wrote {T} objs → {mesh_dir}")

    from scipy.spatial.transform import Rotation as SciR

    aa = (
        SciR.from_matrix(rot_mats.reshape(-1, 3, 3))
        .as_rotvec()
        .reshape(T, 55, 3)
        .astype(np.float32)
    )
    params_path = out_dir / "official_smplx_params.npz"
    np.savez_compressed(
        params_path,
        method="official_hybrik_SMPLXLayer.hybrik",
        smplx_model=str(smplx_path),
        joints_source=str(joints_path),
        global_orient=aa[:, 0],
        body_pose=aa[:, 1:22].reshape(T, -1),
        transl=transl.astype(np.float32),
        rot_mats=rot_mats,
        axis_angle=aa,
        joints_fk=joints_fk,
        joints_input=joints.astype(np.float32),
        vertices=verts.astype(np.float32),
        faces=faces.astype(np.int32),
        scale=np.float32(scale),
        fps=np.float32(20.0),
    )
    meta = {
        "params": str(params_path),
        "mesh_dir": str(mesh_dir),
        "frames": T,
        "smplx": str(smplx_path),
        "joints": str(joints_path),
        "method": "official_hybrik_SMPLXLayer.hybrik",
        "note": "Official SMPL-X body only — no project SMPL-X_Armature swing",
    }
    (out_dir / "REPORT.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"params {params_path}")
    return params_path, out_dir


def make_blend(params_path: Path, out_blend: Path) -> None:
    """Create Blender file: official SMPL-X mesh animated with shape keys (Y-up → Z-up)."""
    import bpy
    from mathutils import Vector

    d = np.load(str(params_path))
    verts = np.asarray(d["vertices"], dtype=np.float64)  # T,V,3 Y-up
    faces = np.asarray(d["faces"], dtype=np.int32)
    T, V, _ = verts.shape
    log(f"building blend T={T} V={V}")

    # Y-up → Blender Z-up for display
    def y2z(v):
        o = np.empty_like(v)
        o[..., 0] = v[..., 0]
        o[..., 1] = -v[..., 2]
        o[..., 2] = v[..., 1]
        return o

    verts_z = y2z(verts)
    z0 = float(verts_z[:, :, 2].min())
    verts_z[..., 2] -= z0

    mesh = bpy.data.meshes.new("SMPL_X_HybrIK")
    obj = bpy.data.objects.new("SMPL_X_HybrIK", mesh)
    bpy.context.scene.collection.objects.link(obj)

    mesh.from_pydata(
        [Vector(p) for p in verts_z[0]],
        [],
        [tuple(map(int, f)) for f in faces],
    )
    mesh.update()

    # Relative shape keys: co = absolute target; value blends basis → that pose
    obj.shape_key_add(name="Basis", from_mix=False)
    for t in range(1, T):
        if t % 20 == 0:
            log(f"  shape key {t}/{T}")
        sk = obj.shape_key_add(name=f"f{t:04d}", from_mix=False)
        for i in range(V):
            sk.data[i].co = Vector(verts_z[t, i])

    keys = obj.data.shape_keys.key_blocks
    for t in range(T):
        frame = t + 1
        for i, kb in enumerate(keys):
            if i == 0:
                kb.value = 1.0
            else:
                kb.value = 1.0 if i == t else 0.0
            kb.keyframe_insert("value", frame=frame)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = T
    bpy.context.scene.frame_set(1)
    bpy.ops.object.camera_add(location=(0, -3.5, 1.2), rotation=(1.4, 0, 0))
    bpy.context.scene.camera = bpy.context.active_object
    bpy.ops.object.light_add(type="SUN", location=(2, -2, 5))

    out_blend.parent.mkdir(parents=True, exist_ok=True)
    fp = str(out_blend.resolve()).replace("\\", "/")
    bpy.ops.wm.save_as_mainfile(filepath=fp, compress=True)
    log(f"blend saved {out_blend}  object=SMPL_X_HybrIK frames=1..{T}")


def main():
    # support blender: ... --python this.py -- args
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]

    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", default=str(DEFAULT_JOINTS))
    ap.add_argument("--smplx", default=str(DEFAULT_SMPLX))
    ap.add_argument(
        "--out-dir",
        default=str(
            ROOT / "body_motion" / "momask_cache" / "t2m_walk_hybrik" / "drive_official_smplx"
        ),
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--make-blend",
        action="store_true",
        help="Also write Blender file with animated official SMPL-X mesh",
    )
    ap.add_argument(
        "--blend-out",
        default="",
        help="Path for .blend (default: out-dir/walk_official_smplx.blend)",
    )
    ap.add_argument(
        "--params-only-blend",
        action="store_true",
        help="Skip hybrik; only build blend from existing official_smplx_params.npz in out-dir",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    joints_path = Path(args.joints)
    if not joints_path.is_absolute():
        joints_path = ROOT / joints_path
    smplx_path = Path(args.smplx)
    if not smplx_path.is_absolute():
        smplx_path = ROOT / smplx_path

    # When run inside Blender for --make-blend only with existing params
    in_blender = "bpy" in sys.modules or any("blender" in a.lower() for a in sys.argv[:1])

    params_path = out_dir / "official_smplx_params.npz"
    if not args.params_only_blend:
        # If we're in blender AND make-blend, still need params first unless exist
        if in_blender and args.make_blend and params_path.is_file():
            log("using existing params for blend")
        else:
            # run hybrik outside blender ideally
            if in_blender:
                log("WARNING: running HybrIK inside Blender python — prefer system python first")
            params_path, out_dir = run_hybrik(
                joints_path, smplx_path, out_dir, batch=args.batch
            )

    if args.make_blend:
        import bpy  # noqa: F401

        blend_out = Path(args.blend_out) if args.blend_out else out_dir / "walk_official_smplx.blend"
        if not blend_out.is_absolute():
            blend_out = ROOT / blend_out
        if not params_path.is_file():
            raise SystemExit(f"missing {params_path} — run without blender first")
        # fresh empty-ish: user asked official body; start from empty factory
        bpy.ops.wm.read_factory_settings(use_empty=True)
        make_blend(params_path, blend_out)

    log("DONE")


if __name__ == "__main__":
    main()
