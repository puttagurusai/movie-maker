"""
MDM-style joints → SMPL: MoMask (T,22,3) via joints2smpl / SMPLify3D.

Uses third_party/DART/visualize/joints2smpl (same family as MDM mesh path).
Body model: SMPL neutral (from HybrIK model_files basicModel, standard).

Does NOT use project SMPL-X_Armature swing.

Usage (from project root):
  python tools/run_joints2smpl_momask.py \\
    --joints body_motion/momask_cache/t2m_walk_hybrik/walk_t2m_joints.npy \\
    --out-dir body_motion/momask_cache/t2m_walk_hybrik/joints2smpl_out \\
    --iters 50
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DART = ROOT / "third_party" / "DART"
J2S = DART / "visualize" / "joints2smpl"
SMPL_SRC = ROOT / "third_party" / "HybrIK" / "model_files" / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"


def log(msg: str) -> None:
    print(f"[joints2smpl] {msg}", flush=True)


def setup_paths():
    body = DART / "body_models" / "smpl"
    body.mkdir(parents=True, exist_ok=True)
    dst = body / "SMPL_NEUTRAL.pkl"
    if not dst.is_file():
        if not SMPL_SRC.is_file():
            raise SystemExit(f"missing SMPL model {SMPL_SRC}")
        shutil.copy2(SMPL_SRC, dst)
        log(f"copied SMPL_NEUTRAL.pkl → {dst}")
    # also under joints2smpl smpl_models
    alt = J2S / "smpl_models" / "smpl"
    alt.mkdir(parents=True, exist_ok=True)
    if not (alt / "SMPL_NEUTRAL.pkl").is_file():
        shutil.copy2(dst, alt / "SMPL_NEUTRAL.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--joints",
        default=str(
            ROOT / "body_motion" / "momask_cache" / "t2m_walk_hybrik" / "walk_t2m_joints.npy"
        ),
    )
    ap.add_argument(
        "--out-dir",
        default=str(
            ROOT / "body_motion" / "momask_cache" / "t2m_walk_hybrik" / "joints2smpl_out"
        ),
    )
    ap.add_argument("--iters", type=int, default=50, help="SMPLify iters per frame")
    ap.add_argument("--fix-foot", action="store_true", default=True)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = ap.parse_args()

    setup_paths()
    jpath = Path(args.joints)
    if not jpath.is_absolute():
        jpath = ROOT / jpath
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    joints = np.load(str(jpath)).astype(np.float32)
    if joints.ndim != 3 or joints.shape[1] != 22 or joints.shape[2] != 3:
        raise SystemExit(f"need (T,22,3), got {joints.shape}")
    T = joints.shape[0]
    log(f"input {jpath} shape={joints.shape}")

    # Run from DART so visualize.joints2smpl imports work
    os.chdir(DART)
    if str(DART) not in sys.path:
        sys.path.insert(0, str(DART))
    if str(J2S / "src") not in sys.path:
        sys.path.insert(0, str(J2S / "src"))

    # Patch config absolute paths
    import config as j2s_config

    j2s_config.SMPL_MODEL_DIR = str(DART / "body_models")
    j2s_config.GMM_MODEL_DIR = str(J2S / "smpl_models")
    j2s_config.SMPL_MEAN_FILE = str(J2S / "smpl_models" / "neutral_smpl_mean_params.h5")
    j2s_config.Part_Seg_DIR = str(J2S / "smpl_models" / "smplx_parts_segm.pkl")

    import h5py
    import joblib
    import smplx
    import trimesh
    from smplify import SMPLify3D
    from tqdm import tqdm

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    log(f"device={device} iters={args.iters}")

    try:
        smplmodel = smplx.create(
            j2s_config.SMPL_MODEL_DIR,
            model_type="smpl",
            gender="neutral",
            ext="pkl",
            batch_size=1,
        ).to(device)
    except Exception as e:
        raise SystemExit(
            f"Failed to load SMPL from {j2s_config.SMPL_MODEL_DIR}: {e}\n"
            "Need SMPL_NEUTRAL.pkl (smplify basicModel)."
        ) from e

    with h5py.File(j2s_config.SMPL_MEAN_FILE, "r") as f:
        init_mean_pose = torch.from_numpy(f["pose"][:]).unsqueeze(0).float()
        init_mean_shape = torch.from_numpy(f["shape"][:]).unsqueeze(0).float()

    smplify = SMPLify3D(
        smplxmodel=smplmodel,
        batch_size=1,
        joints_category="AMASS",
        num_iters=int(args.iters),
        device=device,
    )

    pred_pose = torch.zeros(1, 72, device=device)
    pred_betas = torch.zeros(1, 10, device=device)
    pred_cam_t = torch.zeros(1, 3, device=device)
    keypoints_3d = torch.zeros(1, 22, 3, device=device)

    poses = []
    betas_list = []
    trans_list = []
    joint_losses = []

    conf = torch.ones(22, device=device)
    if args.fix_foot:
        conf[7] = conf[8] = conf[10] = conf[11] = 1.5

    ply_dir = out_dir / "ply"
    ply_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(T), desc="joints2smpl"):
        keypoints_3d[0] = torch.tensor(joints[idx], device=device, dtype=torch.float32)
        if idx == 0:
            pred_betas[0] = init_mean_shape.to(device)
            pred_pose[0] = init_mean_pose.to(device)
            pred_cam_t[0] = 0.0
        else:
            pred_betas[0] = torch.tensor(betas_list[-1], device=device)
            pred_pose[0] = torch.tensor(poses[-1], device=device)
            pred_cam_t[0] = torch.tensor(trans_list[-1], device=device)

        (
            _verts,
            _j,
            new_pose,
            new_betas,
            new_cam,
            jloss,
        ) = smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=conf,
            seq_ind=idx,
        )

        pose_np = new_pose.detach().cpu().numpy()[0]
        beta_np = new_betas.detach().cpu().numpy()[0]
        cam_np = new_cam.detach().cpu().numpy()[0]
        poses.append(pose_np)
        betas_list.append(beta_np)
        trans_list.append(cam_np)
        joint_losses.append(float(jloss.detach().cpu().numpy().reshape(-1)[0]))

        with torch.no_grad():
            outp = smplmodel(
                betas=new_betas,
                global_orient=new_pose[:, :3],
                body_pose=new_pose[:, 3:],
                transl=new_cam,
                return_verts=True,
            )
        mesh = trimesh.Trimesh(
            vertices=outp.vertices.detach().cpu().numpy().squeeze(),
            faces=smplmodel.faces,
            process=False,
        )
        mesh.export(str(ply_dir / f"{idx:04d}.ply"))

        if idx % 20 == 0:
            log(f"  frame {idx} joint_loss={joint_losses[-1]:.6f}")

    poses = np.stack(poses, 0).astype(np.float32)
    betas_arr = np.stack(betas_list, 0).astype(np.float32)
    trans = np.stack(trans_list, 0).astype(np.float32)

    # full sequence mesh verts
    log("exporting full sequence params + sample objs")
    all_verts = []
    with torch.no_grad():
        for idx in range(T):
            outp = smplmodel(
                betas=torch.tensor(betas_arr[idx : idx + 1], device=device),
                global_orient=torch.tensor(poses[idx : idx + 1, :3], device=device),
                body_pose=torch.tensor(poses[idx : idx + 1, 3:], device=device),
                transl=torch.tensor(trans[idx : idx + 1], device=device),
                return_verts=True,
            )
            all_verts.append(outp.vertices[0].cpu().numpy())
    all_verts = np.stack(all_verts, 0).astype(np.float32)

    npz_path = out_dir / "smpl_params_joints2smpl.npz"
    np.savez_compressed(
        npz_path,
        method="joints2smpl_SMPLify3D_AMASS",
        pose=poses,  # (T,72) global_orient + body_pose
        global_orient=poses[:, :3],
        body_pose=poses[:, 3:],
        betas=betas_arr,
        transl=trans,
        vertices=all_verts,
        faces=np.asarray(smplmodel.faces, dtype=np.int32),
        joint_loss=np.array(joint_losses, dtype=np.float32),
        joints_source=str(jpath),
        smpl_model=str(DART / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl"),
        fps=np.float32(20.0),
    )

    # OBJ mid frames for quick view
    obj_dir = out_dir / "mesh_obj"
    obj_dir.mkdir(exist_ok=True)
    faces = np.asarray(smplmodel.faces, dtype=np.int32)
    for fi in [0, T // 2, T - 1]:
        v = all_verts[fi].copy()
        v[:, 1] -= v[:, 1].min()
        p = obj_dir / f"frame_{fi:04d}.obj"
        with p.open("w", encoding="utf-8") as f:
            f.write("# joints2smpl / SMPLify3D (MDM-style)\n")
            for x, y, z in v:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a+1} {b+1} {c+1}\n")
        log(f"obj {p}")

    meta = {
        "method": "joints2smpl_SMPLify3D",
        "like": "MDM mesh path (wangsen1312/joints2smpl)",
        "frames": T,
        "mean_joint_loss": float(np.mean(joint_losses)),
        "params": str(npz_path),
        "ply_dir": str(ply_dir),
        "obj_dir": str(obj_dir),
        "note": "Official SMPL body from MoMask joints — not project armature",
    }
    (out_dir / "REPORT.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"DONE mean_joint_loss={meta['mean_joint_loss']:.6f}")
    log(f"params {npz_path}")
    log(f"import OBJs from {obj_dir} or PLYs from {ply_dir}")


if __name__ == "__main__":
    main()
