"""
MoMask text→motion → projface pipeline NPZ (humanml3d_joints22).

Requires:
  - third_party/momask-codes (cloned)
  - third_party/momask_venv (or any env with momask deps)
  - pretrained checkpoints under third_party/momask-codes/checkpoints/t2m/

Usage:
  third_party\\momask_venv\\Scripts\\python.exe tools\\momask_infer.py \\
      --prompt "a man is walking" --out "lmm train\\momask_walk.npz"

  # also bake onto SMPL-X (needs Blender on PATH or --blender)
  ... --apply_blender
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

ROOT = Path(__file__).resolve().parents[1]
MOMASK = ROOT / "third_party" / "momask-codes"


def _ensure_momask_path():
    if not MOMASK.is_dir():
        raise SystemExit(f"Missing {MOMASK}. Clone: git clone https://github.com/EricGuo5513/momask-codes.git third_party/momask-codes")
    sys.path.insert(0, str(MOMASK))


def resample_joints(joints: np.ndarray, fps_in: int = 20, fps_out: int = 30) -> np.ndarray:
    """(T,22,3) linear time resample."""
    T_in = joints.shape[0]
    if T_in < 2 or fps_in == fps_out:
        return joints.astype(np.float32)
    dur = (T_in - 1) / float(fps_in)
    T_out = max(2, int(round(dur * fps_out)) + 1)
    t_in = np.linspace(0.0, 1.0, T_in)
    t_out = np.linspace(0.0, 1.0, T_out)
    out = np.zeros((T_out, joints.shape[1], 3), dtype=np.float32)
    for j in range(joints.shape[1]):
        for ax in range(3):
            out[:, j, ax] = np.interp(t_out, t_in, joints[:, j, ax])
    return out


def save_pipeline_npz(
    joints_22: np.ndarray,
    prompt: str,
    out_path: Path,
    fps: int = 30,
    seed: int = 0,
    use_ik: bool = False,
):
    """Save LMM_IO_FORMAT humanml3d_joints22."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    j = np.asarray(joints_22, dtype=np.float32)
    assert j.ndim == 3 and j.shape[1:] == (22, 3), j.shape
    np.savez(
        str(out_path),
        schema_version=np.int32(1),
        ok=True,
        joint_format="humanml3d_joints22",
        fps=np.int32(fps),
        frames=np.int32(j.shape[0]),
        joints=j,
        meta_prompt=str(prompt),
        meta_seed=np.int32(seed),
        meta_model_name="momask_t2m",
        meta_text_encoder="clip_vit_b32",
        meta_D_e=np.int32(512),
        meta_up_axis="y",
        meta_units="m",
        meta_n_joints=np.int32(22),
        meta_source="EricGuo5513/momask-codes",
        meta_ik=bool(use_ik),
    )
    print(f"[momask_infer] saved {out_path}  joints={j.shape} fps={fps}")


def generate_joints(
    prompt: str,
    *,
    gpu_id: int = 0,
    motion_length: int = 0,
    seed: int = 0,
    repeat: int = 0,
    time_steps: int = 10,
    cond_scale: float = 4.0,
    use_ik: bool = True,
) -> np.ndarray:
    """
    Run MoMask and return joints (T, 22, 3) at native 20 fps.
    """
    _ensure_momask_path()
    os.chdir(MOMASK)

    from os.path import join as pjoin

    from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
    from models.vq.model import RVQVAE, LengthEstimator
    from options.eval_option import EvalT2MOptions
    from utils.fixseed import fixseed
    from utils.get_opt import get_opt
    from utils.motion_process import recover_from_ric
    # Joint2BVHConvertor optional (needs older numpy umath_tests); only if use_ik

    # Build opt like gen_t2m CLI defaults for HumanML3D
    parser = EvalT2MOptions()
    # parse with minimal argv so defaults load
    sys_argv_backup = sys.argv
    sys.argv = [
        "gen_t2m.py",
        "--gpu_id", str(gpu_id),
        "--ext", "projface",
        "--text_prompt", prompt,
        "--motion_length", str(motion_length),
        "--seed", str(seed),
        "--repeat_times", "1",
        "--time_steps", str(time_steps),
        "--cond_scale", str(cond_scale),
    ]
    opt = parser.parse()
    sys.argv = sys_argv_backup

    fixseed(opt.seed)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else f"cuda:{opt.gpu_id}")
    if opt.gpu_id >= 0 and not torch.cuda.is_available():
        print("[momask_infer] CUDA not available — using CPU")
        opt.device = torch.device("cpu")
        opt.gpu_id = -1

    dim_pose = 263
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_opt_path = pjoin(root_dir, "opt.txt")
    if not os.path.isfile(model_opt_path):
        raise SystemExit(
            f"Missing MoMask checkpoints at {root_dir}\n"
            f"Run: cd third_party/momask-codes && bash prepare/download_models.sh\n"
            f"Or download from README Google Drive into third_party/momask-codes/checkpoints/"
        )

    model_opt = get_opt(model_opt_path, device=opt.device)

    # VQ
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, "opt.txt")
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model = RVQVAE(
        vq_opt,
        vq_opt.dim_pose,
        vq_opt.nb_code,
        vq_opt.code_dim,
        vq_opt.output_emb_width,
        vq_opt.down_t,
        vq_opt.stride_t,
        vq_opt.width,
        vq_opt.depth,
        vq_opt.dilation_growth_rate,
        vq_opt.vq_act,
        vq_opt.vq_norm,
    )
    ckpt = torch.load(
        pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, "model", "net_best_fid.tar"),
        map_location="cpu",
    )
    model_key = "vq_model" if "vq_model" in ckpt else "net"
    vq_model.load_state_dict(ckpt[model_key])
    print(f"[momask_infer] VQ {vq_opt.name} loaded")

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    # Residual transformer
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, "opt.txt")
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    clip_version = "ViT-B/32"
    res_model = ResidualTransformer(
        code_dim=vq_opt.code_dim,
        cond_mode="text",
        latent_dim=res_opt.latent_dim,
        ff_size=res_opt.ff_size,
        num_layers=res_opt.n_layers,
        num_heads=res_opt.n_heads,
        dropout=res_opt.dropout,
        clip_dim=512,
        shared_codebook=vq_opt.shared_codebook,
        cond_drop_prob=res_opt.cond_drop_prob,
        share_weight=res_opt.share_weight,
        clip_version=clip_version,
        opt=res_opt,
    )
    rckpt = torch.load(
        pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, "model", "net_best_fid.tar"),
        map_location=opt.device,
    )
    res_model.load_state_dict(rckpt["res_transformer"], strict=False)
    print(f"[momask_infer] Residual {res_opt.name} loaded")

    # Mask transformer
    t2m_transformer = MaskTransformer(
        code_dim=model_opt.code_dim,
        cond_mode="text",
        latent_dim=model_opt.latent_dim,
        ff_size=model_opt.ff_size,
        num_layers=model_opt.n_layers,
        num_heads=model_opt.n_heads,
        dropout=model_opt.dropout,
        clip_dim=512,
        cond_drop_prob=model_opt.cond_drop_prob,
        clip_version=clip_version,
        opt=model_opt,
    )
    tckpt = torch.load(
        pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, "model", "latest.tar"),
        map_location="cpu",
    )
    mkey = "t2m_transformer" if "t2m_transformer" in tckpt else "trans"
    t2m_transformer.load_state_dict(tckpt[mkey], strict=False)
    print(f"[momask_infer] MaskTransformer {opt.name} loaded")

    length_estimator = LengthEstimator(512, 50)
    lckpt = torch.load(
        pjoin(opt.checkpoints_dir, opt.dataset_name, "length_estimator", "model", "finest.tar"),
        map_location=opt.device,
    )
    length_estimator.load_state_dict(lckpt["estimator"])
    print("[momask_infer] LengthEstimator loaded")

    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()
    length_estimator.eval()
    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)
    length_estimator.to(opt.device)

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, "meta", "mean.npy"))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, "meta", "std.npy"))

    def inv_transform(data):
        return data * std + mean

    captions = [prompt]
    if motion_length and motion_length > 0:
        token_lens = torch.LongTensor([motion_length // 4]).to(opt.device)
    else:
        with torch.no_grad():
            text_embedding = t2m_transformer.encode_text(captions)
            pred_dis = length_estimator(text_embedding)
            probs = F.softmax(pred_dis, dim=-1)
            token_lens = Categorical(probs).sample()
        print(f"[momask_infer] estimated token_lens={int(token_lens[0])} poses~{int(token_lens[0])*4}")

    m_length = int(token_lens[0].item() * 4)

    with torch.no_grad():
        mids = t2m_transformer.generate(
            captions,
            token_lens,
            timesteps=time_steps,
            cond_scale=cond_scale,
            temperature=1.0,
            topk_filter_thres=0.9,
            gsample=False,
        )
        mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
        pred_motions = vq_model.forward_decoder(mids)
        pred_motions = pred_motions.detach().cpu().numpy()
        data = inv_transform(pred_motions)

    joint_data = data[0][:m_length]
    joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

    if use_ik:
        try:
            from visualization.joints2bvh import Joint2BVHConvertor

            converter = Joint2BVHConvertor()
            tmp = MOMASK / "generation" / "_tmp_ik.bvh"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            _, ik_joint = converter.convert(joint, filename=str(tmp), iterations=100)
            joint = ik_joint
            print("[momask_infer] foot IK applied")
        except Exception as e:
            print(f"[momask_infer] foot IK skipped: {e}")

    return np.asarray(joint, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="MoMask → projface NPZ")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--out", type=str, default=str(ROOT / "lmm train" / "momask_out.npz"))
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--motion_length", type=int, default=0, help="poses @20fps; 0=auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_steps", type=int, default=10)
    ap.add_argument("--cond_scale", type=float, default=4.0)
    ap.add_argument("--no_ik", action="store_true")
    ap.add_argument("--fps_export", type=int, default=30)
    ap.add_argument("--apply_blender", action="store_true")
    ap.add_argument(
        "--blender",
        type=str,
        default=r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
    )
    ap.add_argument(
        "--blend",
        type=str,
        default=str(ROOT / "whole_body_retargeted.blend"),
    )
    ap.add_argument(
        "--blend_out",
        type=str,
        default=str(ROOT / "body_motion" / "_momask_applied.blend"),
    )
    ap.add_argument("--action", type=str, default="momask_motion")
    args = ap.parse_args()

    print(f"[momask_infer] prompt={args.prompt!r}")
    # Resolve output paths BEFORE generate_joints() chdirs into momask-codes
    out = Path(args.out)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    blend_out = Path(args.blend_out)
    if not blend_out.is_absolute():
        blend_out = (ROOT / blend_out).resolve()
    blend_in = Path(args.blend)
    if not blend_in.is_absolute():
        blend_in = (ROOT / blend_in).resolve()

    joints20 = generate_joints(
        args.prompt,
        gpu_id=args.gpu_id,
        motion_length=args.motion_length,
        seed=args.seed,
        time_steps=args.time_steps,
        cond_scale=args.cond_scale,
        use_ik=not args.no_ik,
    )
    print(f"[momask_infer] joints @20fps: {joints20.shape}")

    # bone length quick check
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    lens = []
    for i in range(1, 22):
        L = np.linalg.norm(joints20[:, i] - joints20[:, parents[i]], axis=1)
        lens.append(L.std() / (L.mean() + 1e-8))
    print(f"[momask_infer] mean bone-length CV={float(np.mean(lens)):.4f} (<<0.1 is good)")

    joints30 = resample_joints(joints20, 20, args.fps_export)
    save_pipeline_npz(
        joints30,
        args.prompt,
        out,
        fps=args.fps_export,
        seed=args.seed,
        use_ik=not args.no_ik,
    )

    if args.apply_blender:
        bl = Path(args.blender)
        if not bl.is_file():
            raise SystemExit(f"Blender not found: {bl}")
        apply = ROOT / "tools" / "apply_lmm_npz_to_smplx.py"
        cmd = [
            str(bl),
            str(blend_in),
            "--background",
            "--python",
            str(apply),
            "--",
            "--npz",
            str(out),
            "--action",
            args.action,
            "--out",
            str(blend_out),
        ]
        print("[momask_infer] running Blender adapter...")
        subprocess.check_call(cmd)
        print(f"[momask_infer] blend → {blend_out}")


if __name__ == "__main__":
    main()
