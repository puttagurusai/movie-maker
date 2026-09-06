#!/usr/bin/env python
"""
Official MoMask t2m only → scale to our SMPL-X → bone map bake.

Does NOT use pre-made / old BVH files as input.
  1) third_party/momask-codes/gen_t2m.py  (official) → *_ik.bvh
  2) tools/momask_bvh_to_smplx_fast.py:
       KeeMap BVH→Mixamo (official mapping.json)
       scale Mixamo hips→head to SMPL-X
       FINAL_BONE_MAP Mixamo→SMPL-X Action

Usage:
  python tools/generate_t2m_scale_map_smplx.py
  python tools/generate_t2m_scale_map_smplx.py --prompt "a person walks forward" --seconds 4
  python tools/generate_t2m_scale_map_smplx.py --prompt "a person waves with the right hand" --seed 7
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Official t2m BVH → scale+map to SMPL-X")
    ap.add_argument(
        "--prompt",
        type=str,
        default="a person walks forward naturally",
        help="HumanML3D-style caption for official gen_t2m",
    )
    ap.add_argument("--seconds", type=float, default=4.0, help="Approx motion length (seconds)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--action",
        type=str,
        default="",
        help="Action name on SMPL-X (default auto momask_official_*)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "body_motion" / "momask_cache" / "official_t2m_scale"),
        help="Output folder for BVH copy + blend + report",
    )
    args = ap.parse_args()

    from face_agents.momask_body_pipeline import (
        generate_official_bvh,
        retarget_bvh_to_smplx_action,
        _clean_caption,
    )

    prompt = _clean_caption(args.prompt) or args.prompt.strip()
    # MoMask ~20 fps pose rate
    motion_length = int(round(max(2.0, min(12.0, float(args.seconds))) * 20))
    motion_length = max(32, min(196, (motion_length // 4) * 4))

    stamp = int(time.time())
    ext = f"official_t2m_{stamp}"
    action = (args.action or f"momask_official_{stamp}").strip()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_blend = out_dir / f"{action}.blend"
    report_path = out_dir / f"{action}_report.json"
    map_doc = ROOT / "body_motion" / "bvh_official_t2m_to_smplx_map.json"

    print("=" * 64)
    print("OFFICIAL t2m → scale → SMPL-X map")
    print("=" * 64)
    print(f"  prompt : {prompt}")
    print(f"  frames : {motion_length} (~{motion_length/20:.1f}s @ 20fps MoMask)")
    print(f"  seed   : {args.seed}")
    print(f"  action : {action}")
    print(f"  map    : {map_doc}")
    print()

    # 1) Official gen only
    print("[1/2] Official gen_t2m.py …")
    t0 = time.time()
    try:
        bvh = generate_official_bvh(
            prompt,
            ext=ext,
            seed=int(args.seed),
            motion_length=motion_length,
        )
    except Exception as e:
        print(f"FAIL gen_t2m: {e}")
        report_path.write_text(
            json.dumps({"ok": False, "step": "gen_t2m", "error": str(e)}, indent=2),
            encoding="utf-8",
        )
        return 1
    gen_s = time.time() - t0
    print(f"  BVH: {bvh}")
    print(f"  gen wall: {gen_s:.1f}s")

    # Copy official BVH next to outputs for inspection
    bvh_copy = out_dir / f"{action}_official_ik.bvh"
    try:
        shutil.copy2(bvh, bvh_copy)
        print(f"  copied → {bvh_copy}")
    except Exception as e:
        print(f"  copy warn: {e}")
        bvh_copy = Path(bvh)

    # 2) Scale + map via KeeMap Mixamo intermediate → FINAL_BONE_MAP SMPL-X
    print("[2/2] Scale + map (KeeMap Mixamo → SMPL-X FINAL_BONE_MAP) …")
    t1 = time.time()
    try:
        retarget_bvh_to_smplx_action(Path(bvh), action, out_blend)
    except Exception as e:
        print(f"FAIL retarget: {e}")
        report_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "step": "retarget",
                    "error": str(e),
                    "bvh": str(bvh),
                    "prompt": prompt,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 1
    ret_s = time.time() - t1
    print(f"  blend: {out_blend}")
    print(f"  retarget wall: {ret_s:.1f}s")

    report = {
        "ok": True,
        "prompt": prompt,
        "seed": args.seed,
        "motion_length": motion_length,
        "action_name": action,
        "official_bvh": str(bvh),
        "bvh_copy": str(bvh_copy),
        "out_blend": str(out_blend),
        "map_document": str(map_doc),
        "final_bone_map": str(ROOT / "body_motion" / "FINAL_BONE_MAP.json"),
        "keemap_map": str(ROOT / "third_party" / "momask-codes" / "assets" / "mapping.json"),
        "pipeline": [
            "official gen_t2m.py → *_ik.bvh",
            "KeeMap + mapping.json → Mixamo",
            "uniform scale hips→head to SMPL-X",
            "FINAL_BONE_MAP Mixamo→SMPL-X Action",
        ],
        "gen_s": round(gen_s, 2),
        "retarget_s": round(ret_s, 2),
        "total_s": round(gen_s + ret_s, 2),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print("=" * 64)
    print("DONE")
    print(f"  action : {action}")
    print(f"  bvh    : {bvh_copy}")
    print(f"  blend  : {out_blend}")
    print(f"  report : {report_path}")
    print("  Open blend → SMPL-X_Armature → Action Editor → action name above")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
