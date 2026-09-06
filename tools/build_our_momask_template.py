"""
Rebuild OUR MoMask template from SMPL-X rest (no Mixamo).

  whole_body_retargeted.blend → our_smplx_rest.bvh → our_momask_template.bvh

Files kept in third_party/momask-codes/visualization/data/:
  template.bvh              original MoMask
  our_smplx_rest.bvh        rest from our armature
  our_momask_template.bvh   converted (used by gen_t2m / agents)
  make_momask_template.py

Usage:
  python tools/build_our_momask_template.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOMASK = ROOT / "third_party" / "momask-codes"
DATA = MOMASK / "visualization" / "data"
BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
BLEND = ROOT / "whole_body_retargeted.blend"


def run(cmd, cwd=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", default=str(BLEND))
    ap.add_argument("--blender", default=str(BLENDER))
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    rest = DATA / "our_smplx_rest.bvh"
    tpl = DATA / "our_momask_template.bvh"

    if not args.skip_export:
        run(
            [
                args.blender,
                str(Path(args.blend)),
                "--background",
                "--python",
                str(ROOT / "tools" / "export_smplx_rest_for_momask_template.py"),
                "--",
                "--out",
                str(rest),
            ]
        )

    py = sys.executable
    vpy = ROOT / "third_party" / "momask_venv" / "Scripts" / "python.exe"
    if vpy.is_file():
        py = str(vpy)

    make = DATA / "make_momask_template.py"
    run([py, str(make), "--source", str(rest), "--out", str(tpl)])

    # Point joints2bvh at our_momask_template.bvh
    j2b = MOMASK / "visualization" / "joints2bvh.py"
    text = j2b.read_text(encoding="utf-8")
    want = "our_momask_template.bvh"
    if f"our_momask_template.bvh" not in text or "our_momask_template_mixamo" in text:
        import re

        text2 = re.sub(
            r"template_path\s*=\s*['\"][^'\"]*our_momask[^'\"]*['\"]",
            f"template_path='./visualization/data/{want}'",
            text,
            count=1,
        )
        text2 = re.sub(
            r"BVH\.load\(\s*['\"][^'\"]*our_momask[^'\"]*['\"]",
            f"BVH.load('./visualization/data/{want}'",
            text2,
            count=1,
        )
        if text2 != text:
            j2b.write_text(text2, encoding="utf-8")
            print(f"Updated joints2bvh.py → {want}")
        else:
            # ensure default path string exists
            if want not in j2b.read_text(encoding="utf-8"):
                print("WARNING: could not auto-patch joints2bvh.py — set default manually")
    else:
        print(f"joints2bvh already references {want}")

    print("\nDONE — data BVHs:")
    for name in ("template.bvh", "our_smplx_rest.bvh", "our_momask_template.bvh"):
        p = DATA / name
        print(f"  {'OK' if p.is_file() else 'MISSING'} {p.name}  ({p.stat().st_size if p.is_file() else 0} bytes)")


if __name__ == "__main__":
    main()
