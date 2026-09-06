"""
MDM / joints2smpl-style fit: MoMask HumanML3D joints → official SMPL-X body.

Same idea as wangsen1312/joints2smpl used by MDM:
  optimize SMPL-X pose so model joints match input 3D joints (+ simple priors).

Uses project body model:
  models/smplx/SMPLX_NEUTRAL.npz

No custom armature swing. Output = official SMPL-X mesh + params.

Usage:
  python tools/joints_smplify_to_smplx.py \\
    --joints body_motion/momask_cache/t2m_walk_hybrik/walk_t2m_joints.npy \\
    --out-dir body_motion/momask_cache/t2m_walk_hybrik/joints2smpl_smplx_out \\
    --iters 80
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMPLX = ROOT / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
DEFAULT_JOINTS = (
    ROOT / "body_motion" / "momask_cache" / "t2m_walk_hybrik" / "walk_t2m_joints.npy"
)


def log(msg: str) -> None:
    print(f"[j2s_smplx] {msg}", flush=True)


def gmof(x: torch.Tensor, sigma: float = 100.0) -> torch.Tensor:
    """Geman-McClure robust penalty (joints2smpl / SMPLify)."""
    x2 = x * x
    s2 = sigma * sigma
    return (s2 * x2) / (s2 + x2)


def angle_prior(body_pose: torch.Tensor) -> torch.Tensor:
    """Penalize unnatural knee/elbow bend (indices in 21*3 body_pose)."""
    # body_pose layout SMPL-X: same first 21 body joints as SMPL body
    # L_knee=3, R_knee=4, L_elbow=17, R_elbow=18  (0-based among body joints)
    # in flat body_pose: joint i → 3*i : 3*i+3, use Y rotation-ish component like smplify
    idx = [3 * 3 + 1, 4 * 3 + 1, 17 * 3 + 1, 18 * 3 + 1]
    signs = torch.tensor([1.0, -1.0, -1.0, -1.0], device=body_pose.device)
    return torch.exp(body_pose[:, idx] * signs).pow(2).sum(dim=-1)


def fit_frame(
    model,
    joints_np: np.ndarray,
    *,
    init_go: torch.Tensor | None,
    init_bp: torch.Tensor | None,
    init_tr: torch.Tensor | None,
    init_beta: torch.Tensor | None,
    device: torch.device,
    iters_cam: int = 80,
    iters_body: int = 200,
    fix_shape: bool = False,
) -> dict:
    """Fit one frame. joints_np: (22,3) float. Adam (stable) like a light SMPLify3D."""
    j3d = torch.tensor(joints_np[None], dtype=torch.float32, device=device)  # 1,22,3

    go = (
        init_go.detach().clone().to(device)
        if init_go is not None
        else torch.zeros(1, 3, device=device)
    )
    bp = (
        init_bp.detach().clone().to(device)
        if init_bp is not None
        else torch.zeros(1, 63, device=device)
    )
    tr = (
        init_tr.detach().clone().to(device)
        if init_tr is not None
        else torch.zeros(1, 3, device=device)
    )
    beta = (
        init_beta.detach().clone().to(device)
        if init_beta is not None
        else torch.zeros(1, 10, device=device)
    )

    zeros3 = torch.zeros(1, 3, device=device)
    zeros45 = torch.zeros(1, 45, device=device)
    zeros_expr = torch.zeros(1, model.num_expression_coeffs, device=device)

    def forward():
        return model(
            global_orient=go,
            body_pose=bp,
            transl=tr,
            betas=beta,
            left_hand_pose=zeros45,
            right_hand_pose=zeros45,
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            expression=zeros_expr,
            return_verts=True,
        )

    # conf: weaker collars (HML coplanar), strong feet/arms/torso
    conf = torch.ones(1, 22, 1, device=device)
    conf[:, [7, 8, 10, 11]] = 2.0
    conf[:, [16, 17, 18, 19, 20, 21]] = 2.0  # arms/wrists strong
    conf[:, [13, 14]] = 0.3  # collars weak
    conf[:, [15]] = 1.0

    # init transl from pelvis
    with torch.no_grad():
        out0 = forward()
        tr0 = j3d[0, 0] - out0.joints[0, 0]
        tr.data.copy_(tr0.view(1, 3))

    # stage 1: root
    go = go.detach().requires_grad_(True)
    tr = tr.detach().requires_grad_(True)
    bp = bp.detach().requires_grad_(False)
    beta = beta.detach().requires_grad_(False)
    opt1 = torch.optim.Adam([go, tr], lr=0.05)
    for _ in range(iters_cam):
        out = forward()
        mj = out.joints[:, :22]
        loss = (conf * (mj - j3d).pow(2)).sum()
        opt1.zero_grad()
        loss.backward()
        opt1.step()

    # stage 2: full body
    go = go.detach().requires_grad_(True)
    tr = tr.detach().requires_grad_(True)
    bp = bp.detach().requires_grad_(True)
    beta = beta.detach().requires_grad_(not fix_shape)
    params = [go, tr, bp] + ([] if fix_shape else [beta])
    opt2 = torch.optim.Adam(params, lr=0.02)
    bp0 = bp.detach().clone()
    for i in range(iters_body):
        out = forward()
        mj = out.joints[:, :22]
        jloss = 600.0 * (conf * gmof(mj - j3d, 100.0)).sum()
        pose_reg = 0.5 * bp.pow(2).sum() + 20.0 * (bp - bp0).pow(2).sum()
        shape_reg = 5.0 * beta.pow(2).sum()
        ang = 5.0 * angle_prior(bp).sum()
        loss = jloss + pose_reg + shape_reg + ang
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        if i == iters_body // 2:
            for g in opt2.param_groups:
                g["lr"] *= 0.3

    with torch.no_grad():
        out = forward()
        mj = out.joints[:, :22]
        per = (mj - j3d).pow(2).sum(dim=-1).sqrt().mean()  # mean joint L2
        verts = out.vertices[0].cpu().numpy()

    return {
        "global_orient": go.detach().cpu().numpy()[0].astype(np.float32),
        "body_pose": bp.detach().cpu().numpy()[0].astype(np.float32),
        "transl": tr.detach().cpu().numpy()[0].astype(np.float32),
        "betas": beta.detach().cpu().numpy()[0].astype(np.float32),
        "vertices": verts.astype(np.float32),
        "joints_fit": mj[0].cpu().numpy().astype(np.float32),
        "joint_err": float(per.item()),
    }


def save_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    v = verts.copy()
    v[:, 1] -= v[:, 1].min()
    with path.open("w", encoding="utf-8") as f:
        f.write("# joints2smpl-style fit → official SMPL-X\n")
        for x, y, z in v:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {int(a)+1} {int(b)+1} {int(c)+1}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", default=str(DEFAULT_JOINTS))
    ap.add_argument("--smplx", default=str(DEFAULT_SMPLX))
    ap.add_argument(
        "--out-dir",
        default=str(
            ROOT / "body_motion" / "momask_cache" / "t2m_walk_hybrik" / "joints2smpl_smplx_out"
        ),
    )
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    jpath = Path(args.joints)
    if not jpath.is_absolute():
        jpath = ROOT / jpath
    smplx_path = Path(args.smplx)
    if not smplx_path.is_absolute():
        smplx_path = ROOT / smplx_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    joints = np.load(str(jpath)).astype(np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (22, 3):
        raise SystemExit(f"need (T,22,3), got {joints.shape}")
    T = joints.shape[0]
    log(f"input {jpath.name} {joints.shape}")
    log(f"model {smplx_path}")

    import smplx

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    # smplx.create wants folder or file — use parent with file name pattern
    # For single npz file path, pass the file via model_path folder + gender
    # smplx.create(dir) expects dir/smplx/SMPLX_NEUTRAL.npz
    # our file is models/smplx/SMPLX_NEUTRAL.npz → root = models/
    if smplx_path.name.upper().startswith("SMPLX") and smplx_path.suffix == ".npz":
        model_folder = str(smplx_path.parent.parent)
    else:
        model_folder = str(smplx_path)
    model = smplx.create(
        model_folder,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=10,
        ext="npz",
        batch_size=1,
    ).to(device)
    faces = np.asarray(model.faces, dtype=np.int32)
    log(f"device={device} frames={T} iters~{args.iters}")

    # scale joints to model height (hips-head)
    with torch.no_grad():
        out0 = model()
        rest_j = out0.joints[0, :22].cpu().numpy()
    h_src = np.linalg.norm(joints[:, 15] - joints[:, 0], axis=-1).mean()
    h_rest = float(np.linalg.norm(rest_j[15] - rest_j[0]))
    scale = float(h_rest / h_src) if h_src > 1e-6 else 1.0
    joints_s = joints * scale
    log(f"scale×{scale:.4f}")

    go = bp = tr = beta = None
    results = []
    for t in range(T):
        r = fit_frame(
            model,
            joints_s[t],
            init_go=torch.tensor(go[None], device=device) if go is not None else None,
            init_bp=torch.tensor(bp[None], device=device) if bp is not None else None,
            init_tr=torch.tensor(tr[None], device=device) if tr is not None else None,
            init_beta=torch.tensor(beta[None], device=device) if beta is not None else None,
            device=device,
            iters_cam=100,
            iters_body=max(150, args.iters),
            fix_shape=(t > 0),
        )
        go, bp, tr, beta = r["global_orient"], r["body_pose"], r["transl"], r["betas"]
        results.append(r)
        if t % 10 == 0 or t == T - 1:
            log(f"  frame {t}/{T-1} joint_err={r['joint_err']:.4f}")

    # pack
    global_orient = np.stack([r["global_orient"] for r in results])
    body_pose = np.stack([r["body_pose"] for r in results])
    transl = np.stack([r["transl"] for r in results])
    betas = np.stack([r["betas"] for r in results])
    vertices = np.stack([r["vertices"] for r in results])
    joints_fit = np.stack([r["joints_fit"] for r in results])
    errs = np.array([r["joint_err"] for r in results], dtype=np.float32)

    npz_path = out_dir / "smplx_params_joints2smpl.npz"
    np.savez_compressed(
        npz_path,
        method="joints2smpl_style_SMPLify3D_SMPLX",
        global_orient=global_orient,
        body_pose=body_pose,
        transl=transl,
        betas=betas,
        vertices=vertices,
        faces=faces,
        joints_fit=joints_fit,
        joints_input=joints_s,
        joint_err=errs,
        scale=np.float32(scale),
        joints_source=str(jpath),
        smplx_model=str(smplx_path),
        fps=np.float32(20.0),
    )

    obj_dir = out_dir / "mesh_obj"
    for fi in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
        save_obj(obj_dir / f"frame_{fi:04d}.obj", vertices[fi], faces)

    # stick-ish diagnostic: mean joint error & head kink before/after
    def head_kink(J):
        def ang(a, b, c):
            u = b - a
            w = c - b
            u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-8)
            w = w / (np.linalg.norm(w, axis=-1, keepdims=True) + 1e-8)
            return np.degrees(np.arccos(np.clip((u * w).sum(-1), -1, 1)))

        return float(ang(J[:, 9], J[:, 12], J[:, 15]).mean())

    meta = {
        "method": "joints2smpl_style_SMPLify3D_SMPLX",
        "like": "MDM joints2smpl / SMPLify3D but SMPL-X body model",
        "frames": T,
        "mean_joint_err": float(errs.mean()),
        "head_kink_input_deg": head_kink(joints_s),
        "head_kink_fit_deg": head_kink(joints_fit),
        "params": str(npz_path),
        "obj_dir": str(obj_dir),
        "note": "Official SMPL-X from models/smplx — not project armature",
    }
    (out_dir / "REPORT.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"DONE mean_joint_err={meta['mean_joint_err']:.4f}")
    log(f"head_kink input={meta['head_kink_input_deg']:.1f}° fit={meta['head_kink_fit_deg']:.1f}°")
    log(f"params {npz_path}")
    log(f"OBJ  {obj_dir}")


if __name__ == "__main__":
    main()
