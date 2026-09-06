"""
HybrIK analytical IK: MoMask joints (T,22,3) → SMPL-X body pose (no BVH, no mesh twist).

What this does
--------------
  Input joints already contain walk / jump / actions.
  HybrIK converts joint *positions* → local bone *rotations* via twist-and-swing
  decomposition, with **twist forced to zero** (no candy-wrapper mesh twist from
  BVH Euler / KeeMap).

  Root translation is kept from the hip trajectory so the skeleton **walks**.

Pipeline
--------
  joints.npy (prefer *_ik.npy foot plant)
    → scale to SMPL-X rest height
    → HybrIK IK (phis = zero twist)
    → pose.npz  {rot_mats, transl, joints_scaled, axis_angle, ...}

  Then bake onto your armature (no BVH):
    blender.exe whole_body_retargeted.blend --background \\
      --python tools/hybrik_apply_to_smplx.py -- \\
      --pose body_motion/momask_cache/.../walk_hybrik_pose.npz \\
      --out body_motion/momask_cache/.../walk_hybrik.blend

Usage
-----
  python tools/hybrik_joints_to_pose.py \\
    --joints third_party/momask-codes/generation/.../sample0_repeat0_len80.npy \\
    --out body_motion/momask_cache/official_t2m_scale/walk_hybrik_pose.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HYBRIK = ROOT / "third_party" / "HybrIK"
SMPLX_MODELS = (
    ROOT
    / "third_party"
    / "DART"
    / "data"
    / "smplx_lockedhead_20230207"
    / "models_lockedhead"
)

# HumanML3D / MoMask 22 = SMPL-X body joints 0..21
HML_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)

# Bone name on SMPL-X_Armature for each joint index
HML_TO_BONE = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


def log(msg: str) -> None:
    print(f"[hybrik_pose] {msg}", flush=True)


def build_children(parents: np.ndarray) -> torch.Tensor:
    """HybrIK-style children map: first child index, -1 leaf, -3 multi-child."""
    n = len(parents)
    children = torch.ones(n, dtype=torch.long) * -1
    counts = {}
    for i in range(1, n):
        p = int(parents[i])
        counts[p] = counts.get(p, 0) + 1
        if children[p] == -1:
            children[p] = i
        elif children[p] >= 0:
            children[p] = -3  # multi
        else:
            children[p] = -3
    # Force pelvis primary child = spine1 (idx 3) like HybrIK
    children[0] = 3
    # spine3 primary = neck (12)
    children[9] = 12
    # leaves stay -1
    for i in range(n):
        if counts.get(i, 0) == 0:
            children[i] = -1
    children[0] = 3
    children[9] = 12
    return children


def vectors2rotmat(vec_rest: torch.Tensor, vec_final: torch.Tensor, dtype) -> torch.Tensor:
    """Rodrigues swing from rest vector → final vector. (B,3,1) → (B,3,3)."""
    batch = vec_rest.shape[0]
    device = vec_rest.device
    rest_n = torch.norm(vec_rest, dim=1, keepdim=True).clamp_min(1e-8)
    fin_n = torch.norm(vec_final, dim=1, keepdim=True).clamp_min(1e-8)
    a = vec_rest / rest_n
    b = vec_final / fin_n
    axis = torch.cross(a, b, dim=1)
    axis_n = torch.norm(axis, dim=1, keepdim=True).clamp_min(1e-8)
    cos = (a * b).sum(dim=1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    sin = axis_n
    # when nearly parallel
    axis = axis / axis_n
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch, 1, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(batch, 3, 3)
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    return ident + sin * K + (1 - cos) * torch.bmm(K, K)


def pelvis_orient_svd(
    rel_pose: torch.Tensor, rel_rest: torch.Tensor, parents: torch.Tensor, dtype
) -> torch.Tensor:
    """Global pelvis rotation from children (spine + hips) via SVD (HybrIK)."""
    kids = [i for i in range(1, parents.shape[0]) if int(parents[i]) == 0]
    rest_mat = torch.cat([rel_rest[:, c] for c in kids], dim=2)  # B,3,K
    pose_mat = torch.cat([rel_pose[:, c] for c in kids], dim=2)
    S = rest_mat.bmm(pose_mat.transpose(1, 2))
    U, _, V = torch.svd(S.cpu())
    U, V = U.to(S.device), V.to(S.device)
    det = torch.det(torch.bmm(V, U.transpose(1, 2)))
    fix = torch.eye(3, device=S.device).unsqueeze(0).expand(U.shape[0], -1, -1).clone()
    fix[:, 2, 2] = det
    return torch.bmm(torch.bmm(V, fix), U.transpose(1, 2))


def multi_children_orient_svd(
    final_locs: list, rest_locs: list, parent_global: torch.Tensor, dtype
) -> torch.Tensor:
    rest_mat = torch.cat(rest_locs, dim=2)
    # bring final into parent local
    target = []
    for f in final_locs:
        target.append(torch.matmul(parent_global.transpose(1, 2), f))
    pose_mat = torch.cat(target, dim=2)
    S = rest_mat.bmm(pose_mat.transpose(1, 2))
    U, _, V = torch.svd(S.cpu())
    U, V = U.to(S.device), V.to(S.device)
    det = torch.det(torch.bmm(V, U.transpose(1, 2)))
    fix = torch.eye(3, device=S.device).unsqueeze(0).expand(U.shape[0], -1, -1).clone()
    fix[:, 2, 2] = det
    return torch.bmm(torch.bmm(V, fix), U.transpose(1, 2))


def hybrik_ik_zero_twist(
    pose_joints: torch.Tensor,
    rest_joints: torch.Tensor,
    parents: torch.Tensor,
    children: torch.Tensor,
) -> torch.Tensor:
    """
    HybrIK sequential IK with twist φ = 0 (swing only).

    pose_joints / rest_joints: (B, J, 3)
    returns local rot_mats (B, J, 3, 3)
    """
    B, J, _ = pose_joints.shape
    device = pose_joints.device
    dtype = pose_joints.dtype

    # relative rest / pose bone vectors
    rel_rest = rest_joints.clone()
    rel_rest[:, 1:] = rest_joints[:, 1:] - rest_joints[:, parents[1:]]
    rel_rest = rel_rest.unsqueeze(-1)  # B,J,3,1

    rel_pose = pose_joints.clone()
    rel_pose[:, 1:] = pose_joints[:, 1:] - pose_joints[:, parents[1:]]
    rel_pose = rel_pose.unsqueeze(-1)

    # root-relative final targets (preserve rest root height frame)
    final = pose_joints.clone().unsqueeze(-1)
    final = final - final[:, 0:1] + rel_rest[:, 0:1]

    rotate_rest = torch.zeros_like(rel_rest)
    rotate_rest[:, 0] = rel_rest[:, 0]

    # zero-twist phis: [cos, sin] = [1, 0]
    phis = torch.zeros(B, J - 1, 2, dtype=dtype, device=device)
    phis[..., 0] = 1.0

    R_global = pelvis_orient_svd(rel_pose, rel_rest, parents, dtype)
    chain = [R_global]
    local = [R_global]

    for i in range(1, J):
        p = int(parents[i])
        # place this joint from parent chain
        rotate_rest[:, i] = rotate_rest[:, p] + torch.matmul(chain[p], rel_rest[:, i])

        ch = int(children[i])
        if ch == -1:
            # leaf: identity local (no twist / no free roll)
            R = (
                torch.eye(3, dtype=dtype, device=device)
                .unsqueeze(0)
                .expand(B, -1, -1)
                .contiguous()
            )
            chain.append(torch.matmul(chain[p], R))
            local.append(R)
            continue

        if ch == -3:
            # multi-child (spine3): SVD over children
            kids = [c for c in range(1, J) if int(parents[c]) == i]
            finals = [final[:, c] - rotate_rest[:, i] for c in kids]
            rests = [rel_rest[:, c].clone() for c in kids]
            R = multi_children_orient_svd(finals, rests, chain[p], dtype)
            chain.append(torch.matmul(chain[p], R))
            local.append(R)
            continue

        # single child: swing + zero twist (HybrIK)
        child_final = final[:, ch] - rotate_rest[:, i]
        # normalize to rest bone length
        template = rel_rest[:, ch]
        tnorm = torch.norm(template, dim=1, keepdim=True).clamp_min(1e-8)
        cfn = torch.norm(child_final, dim=1, keepdim=True).clamp_min(1e-8)
        child_final = child_final * tnorm / cfn

        # to parent-local
        child_final_loc = torch.matmul(chain[p].transpose(1, 2), child_final)
        child_rest_loc = rel_rest[:, ch]

        # swing
        rest_n = torch.norm(child_rest_loc, dim=1, keepdim=True).clamp_min(1e-8)
        fin_n = torch.norm(child_final_loc, dim=1, keepdim=True).clamp_min(1e-8)
        axis = torch.cross(child_rest_loc, child_final_loc, dim=1)
        axis_n = torch.norm(axis, dim=1, keepdim=True).clamp_min(1e-8)
        cos = (child_rest_loc * child_final_loc).sum(dim=1, keepdim=True) / (rest_n * fin_n)
        cos = cos.clamp(-1 + 1e-6, 1 - 1e-6)
        sin = axis_n / (rest_n * fin_n)
        axis = axis / axis_n
        rx, ry, rz = torch.split(axis, 1, dim=1)
        zeros = torch.zeros((B, 1, 1), dtype=dtype, device=device)
        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(B, 3, 3)
        ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
        R_swing = ident + sin * K + (1 - cos) * torch.bmm(K, K)

        # twist φ = 0 → R_spin = I  (no mesh twist)
        R = R_swing
        chain.append(torch.matmul(chain[p], R))
        local.append(R)

    return torch.stack(local, dim=1)


def rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """(T,J,3,3) → (T,J,3) via scipy (stable)."""
    from scipy.spatial.transform import Rotation as SciR

    T, J, _, _ = R.shape
    flat = R.reshape(-1, 3, 3)
    # orthonormalize lightly
    aa = SciR.from_matrix(flat).as_rotvec().reshape(T, J, 3)
    return aa.astype(np.float32)


def load_smplx_model():
    import smplx

    if not SMPLX_MODELS.is_dir():
        raise SystemExit(f"missing SMPL-X models at {SMPLX_MODELS}")
    return smplx.create(
        str(SMPLX_MODELS),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=10,
        ext="npz",
        batch_size=1,
    )


def load_rest_joints() -> np.ndarray:
    m = load_smplx_model()
    with torch.no_grad():
        out = m()
    return out.joints[0, :22].detach().cpu().numpy().astype(np.float64)


def smplx_fk_from_aa(
    aa: np.ndarray, transl: np.ndarray, batch: int = 32
) -> np.ndarray:
    """
    True SMPL-X forward kinematics from HybrIK axis-angle.
    aa: (T,22,3)  joint0=global_orient, 1..21=body
    transl: (T,3) Y-up
    returns joints_fk (T,22,3) Y-up — bone-length-valid SMPL-X skeleton
    """
    import smplx

    T = aa.shape[0]
    model = smplx.create(
        str(SMPLX_MODELS),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=10,
        ext="npz",
        batch_size=batch,
    )
    device = torch.device("cpu")
    model = model.to(device)
    outs = []
    with torch.no_grad():
        for s in range(0, T, batch):
            e = min(T, s + batch)
            b = e - s
            # recreate if last partial batch
            if b != batch:
                model = smplx.create(
                    str(SMPLX_MODELS),
                    model_type="smplx",
                    gender="neutral",
                    use_pca=False,
                    num_betas=10,
                    ext="npz",
                    batch_size=b,
                )
            go = torch.from_numpy(aa[s:e, 0].astype(np.float32))  # B,3
            bp = torch.from_numpy(aa[s:e, 1:22].reshape(b, -1).astype(np.float32))  # B,63
            tr = torch.from_numpy(transl[s:e].astype(np.float32))
            betas = torch.zeros(b, 10, dtype=torch.float32)
            expr = torch.zeros(b, model.num_expression_coeffs, dtype=torch.float32)
            jaw = torch.zeros(b, 3, dtype=torch.float32)
            leye = torch.zeros(b, 3, dtype=torch.float32)
            reye = torch.zeros(b, 3, dtype=torch.float32)
            lhand = torch.zeros(b, 45, dtype=torch.float32)
            rhand = torch.zeros(b, 45, dtype=torch.float32)
            out = model(
                betas=betas,
                global_orient=go,
                body_pose=bp,
                left_hand_pose=lhand,
                right_hand_pose=rhand,
                jaw_pose=jaw,
                leye_pose=leye,
                reye_pose=reye,
                expression=expr,
                transl=tr,
                return_verts=False,
            )
            outs.append(out.joints[:, :22].cpu().numpy())
            log(f"  SMPL-X FK frames {s}-{e - 1}")
    return np.concatenate(outs, axis=0).astype(np.float32)


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


def main():
    ap = argparse.ArgumentParser(description="HybrIK joints → SMPL-X pose npz (zero twist)")
    ap.add_argument("--joints", required=True, help="(T,22,3) MoMask joints npy")
    ap.add_argument("--out", required=True, help="output pose.npz")
    ap.add_argument("--no-prefer-ik", action="store_true")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    jpath = Path(args.joints)
    if not jpath.is_absolute():
        jpath = ROOT / jpath
    jpath = resolve_joints_path(jpath, prefer_ik=not args.no_prefer_ik)
    if not jpath.is_file():
        raise SystemExit(f"missing {jpath}")

    joints = np.load(str(jpath)).astype(np.float64)
    if joints.ndim != 3 or joints.shape[1] != 22 or joints.shape[2] != 3:
        raise SystemExit(f"need (T,22,3), got {joints.shape}")
    T = joints.shape[0]

    rest = load_rest_joints()
    # scale source so mean hips→head matches rest
    h_src = np.linalg.norm(joints[:, 15] - joints[:, 0], axis=-1).mean()
    h_rest = np.linalg.norm(rest[15] - rest[0])
    scale = float(h_rest / h_src) if h_src > 1e-6 else 1.0
    joints_s = joints * scale

    # root-relative for IK; keep absolute hips as transl
    hips = joints_s[:, 0].copy()  # (T,3) Y-up
    joints_rel = joints_s - hips[:, None, :]
    # match rest root origin for IK
    joints_ik = joints_rel + rest[0:1]

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    parents = torch.from_numpy(HML_PARENTS).to(device)
    children = build_children(HML_PARENTS).to(device)
    rest_t = torch.from_numpy(rest[None].astype(np.float32)).to(device)

    all_R = []
    for start in range(0, T, args.batch):
        end = min(T, start + args.batch)
        pose = torch.from_numpy(joints_ik[start:end].astype(np.float32)).to(device)
        rest_b = rest_t.expand(end - start, -1, -1).contiguous()
        R = hybrik_ik_zero_twist(pose, rest_b, parents, children)
        all_R.append(R.cpu())
        log(f"  HybrIK IK frames {start}-{end - 1}")

    rot_mats = torch.cat(all_R, dim=0).numpy()  # T,22,3,3
    aa = rotmat_to_axis_angle(rot_mats)

    # transl: hip path in Y-up so f0 sits on rest pelvis
    transl = hips - hips[0] + rest[0]

    # ---- REAL SMPL-X output: FK with HybrIK pose ----
    log("running SMPL-X forward kinematics (true SMPL-X joints)...")
    joints_fk = smplx_fk_from_aa(aa, transl, batch=min(args.batch, 32))

    # compare raw MoMask joints vs SMPL-X FK (should differ — proves real path)
    # align frames at pelvis for fair local pose compare
    raw_local = joints_s - joints_s[:, 0:1, :]
    fk_local = joints_fk - joints_fk[:, 0:1, :]
    # scale fk height already matches rest; raw was scaled too
    diff = np.linalg.norm(raw_local - fk_local, axis=-1)  # T,22
    log(
        f"raw MoMask vs SMPL-X FK joint delta: "
        f"mean={diff.mean():.4f}m max={diff.max():.4f}m "
        f"(if ~0, bake would look identical — should be >0)"
    )
    for name, idx in [("L_shoulder", 16), ("spine3", 9), ("neck", 12), ("head", 15), ("L_hip", 1)]:
        log(f"  Δ {name}: mean={diff[:, idx].mean():.4f}m max={diff[:, idx].max():.4f}m")

    hip_travel = float(np.linalg.norm(hips - hips[0], axis=-1).max())
    log(
        f"joints {joints.shape} scale×{scale:.4f} hip_travel_max={hip_travel:.4f}m "
        f"zero_twist=True device={device}"
    )

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(out),
        rot_mats=rot_mats.astype(np.float32),
        axis_angle=aa.astype(np.float32),
        transl=transl.astype(np.float32),
        joints_scaled=joints_s.astype(np.float32),  # raw MoMask scaled
        joints_fk=joints_fk.astype(np.float32),  # TRUE SMPL-X after HybrIK
        joints_ik=joints_ik.astype(np.float32),
        rest_joints=rest.astype(np.float32),
        scale=np.float32(scale),
        parents=HML_PARENTS.astype(np.int64),
        bone_names=np.array(HML_TO_BONE),
        joints_source=str(jpath),
        method="hybrik_ik_plus_smplx_fk",
        coord="y_up",
        fps=np.float32(20.0),
        bake_joints_key="joints_fk",
    )
    meta = {
        "out": str(out),
        "joints": str(jpath),
        "frames": T,
        "scale": scale,
        "hip_travel_max_m": hip_travel,
        "method": "hybrik_ik_plus_smplx_fk",
        "mean_joint_delta_raw_vs_fk_m": float(diff.mean()),
        "note": "Bake uses joints_fk (SMPL-X FK), NOT raw MoMask joints. "
        "Apply with hybrik_apply_to_smplx.py --source fk",
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"saved {out}")
    log(f"meta {meta_path}")
    log("DONE")


if __name__ == "__main__":
    main()
