"""
1) Compare rest bone lengths: MoMask/HumanML3D BVH vs Mixamo FBX (world units).
2) Write a T-pose BVH from the BVH hierarchy (all joint rotations = 0).

Usage:
  python tools/make_tpose_bvh.py
  python tools/make_tpose_bvh.py --bvh "lmm train/momask_walk.bvh" --fbx "body_motion/source_fbx/walk.fbx"
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (label, bvh_parent, bvh_child, mixamo_parent, mixamo_child)
SEGMENTS = [
    ("hips-head", "Hips", "Head", "mixamorig:Hips", "mixamorig:Head"),
    ("hips-spine", "Hips", "Spine", "mixamorig:Hips", "mixamorig:Spine"),
    ("spine-spine1", "Spine", "Spine1", "mixamorig:Spine", "mixamorig:Spine1"),
    ("spine1-spine2", "Spine1", "Spine2", "mixamorig:Spine1", "mixamorig:Spine2"),
    ("spine2-neck", "Spine2", "Neck", "mixamorig:Spine2", "mixamorig:Neck"),
    ("neck-head", "Neck", "Head", "mixamorig:Neck", "mixamorig:Head"),
    ("hips-spine2", "Hips", "Spine2", "mixamorig:Hips", "mixamorig:Spine2"),
    ("L_hips-upleg", "Hips", "LeftUpLeg", "mixamorig:Hips", "mixamorig:LeftUpLeg"),
    ("L_thigh", "LeftUpLeg", "LeftLeg", "mixamorig:LeftUpLeg", "mixamorig:LeftLeg"),
    ("L_shin", "LeftLeg", "LeftFoot", "mixamorig:LeftLeg", "mixamorig:LeftFoot"),
    ("L_foot", "LeftFoot", "LeftToe", "mixamorig:LeftFoot", "mixamorig:LeftToeBase"),
    ("R_thigh", "RightUpLeg", "RightLeg", "mixamorig:RightUpLeg", "mixamorig:RightLeg"),
    ("R_shin", "RightLeg", "RightFoot", "mixamorig:RightLeg", "mixamorig:RightFoot"),
    ("R_foot", "RightFoot", "RightToe", "mixamorig:RightFoot", "mixamorig:RightToeBase"),
    ("L_collar", "Spine2", "LeftShoulder", "mixamorig:Spine2", "mixamorig:LeftShoulder"),
    ("L_clavicle", "LeftShoulder", "LeftArm", "mixamorig:LeftShoulder", "mixamorig:LeftArm"),
    ("L_upper_arm", "LeftArm", "LeftForeArm", "mixamorig:LeftArm", "mixamorig:LeftForeArm"),
    ("L_forearm", "LeftForeArm", "LeftHand", "mixamorig:LeftForeArm", "mixamorig:LeftHand"),
    ("R_collar", "Spine2", "RightShoulder", "mixamorig:Spine2", "mixamorig:RightShoulder"),
    ("R_clavicle", "RightShoulder", "RightArm", "mixamorig:RightShoulder", "mixamorig:RightArm"),
    ("R_upper_arm", "RightArm", "RightForeArm", "mixamorig:RightArm", "mixamorig:RightForeArm"),
    ("R_forearm", "RightForeArm", "RightHand", "mixamorig:RightForeArm", "mixamorig:RightHand"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bvh", default=str(ROOT / "lmm train" / "momask_walk.bvh"))
    p.add_argument("--fbx", default=str(ROOT / "body_motion" / "source_fbx" / "walk.fbx"))
    p.add_argument(
        "--out_tpose",
        default=str(ROOT / "lmm train" / "momask_tpose.bvh"),
    )
    p.add_argument(
        "--out_report",
        default=str(ROOT / "body_motion" / "bone_length_bvh_vs_fbx"),
    )
    p.add_argument("--frames", type=int, default=30, help="T-pose BVH frame count")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--hips_height", type=float, default=None, help="Hips Y in T-pose (default: from first motion frame or 0.94)")
    return p.parse_args()


def parse_bvh_offsets(text: str) -> dict[str, tuple[float, float, float]]:
    """Bone name -> OFFSET (x,y,z) relative to parent."""
    offsets: dict[str, tuple[float, float, float]] = {}
    # track current joint name
    name = None
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"(ROOT|JOINT)\s+(\S+)", line)
        if m:
            name = m.group(2)
            continue
        m = re.match(r"OFFSET\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", line)
        if m and name:
            offsets[name] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            name = None  # next OFFSET under End Site ignored via name None after joint offset... 
            # Problem: End Site also has OFFSET. We set name=None after first offset for joint - good.
            # But ROOT/JOINT then OFFSET - good. End Site { OFFSET - name is None so skip... 
            # Wait we set name=None after joint offset, then End Site OFFSET won't store. Good.
    return offsets


def parse_bvh_channel_count(text: str) -> int:
    total = 0
    for m in re.finditer(r"CHANNELS\s+(\d+)", text):
        total += int(m.group(1))
    return total


def hierarchy_only(text: str) -> str:
    idx = text.find("MOTION")
    if idx < 0:
        raise ValueError("no MOTION section in BVH")
    return text[:idx].rstrip() + "\n"


def first_frame_hips_pos(text: str) -> tuple[float, float, float] | None:
    """First 3 channels of first frame are usually X Y Z position of Hips."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Frame Time"):
            if i + 1 < len(lines):
                vals = [float(x) for x in lines[i + 1].split()]
                if len(vals) >= 3:
                    return vals[0], vals[1], vals[2]
    return None


def rest_chain_length(offsets: dict, chain: list[str]) -> float:
    """Sum of offset lengths along chain of bone names (child offsets)."""
    total = 0.0
    for name in chain:
        if name not in offsets:
            return float("nan")
        x, y, z = offsets[name]
        total += math.sqrt(x * x + y * y + z * z)
    return total


def bvh_segment_len_from_offsets(offsets: dict, child: str) -> float:
    """Length of bone = length of child's OFFSET from parent."""
    if child not in offsets:
        return float("nan")
    x, y, z = offsets[child]
    return math.sqrt(x * x + y * y + z * z)


def write_tpose_bvh(
    src_bvh: Path,
    out_bvh: Path,
    n_frames: int,
    fps: float,
    hips_height: float | None,
) -> dict:
    text = src_bvh.read_text(encoding="utf-8", errors="ignore")
    hier = hierarchy_only(text)
    nchan = parse_bvh_channel_count(text)
    if nchan < 6:
        raise SystemExit(f"unexpected channel count {nchan}")

    # Hips position: X=0, Y=height, Z=0; all rotations 0 → T-pose for this bind
    if hips_height is None:
        pos = first_frame_hips_pos(text)
        hips_height = pos[1] if pos else 0.94

    # Channel layout: Hips 6 (pos+rot) then 3 per joint
    # Zero everything except hips Y
    frame_vals = [0.0] * nchan
    frame_vals[0] = 0.0  # X
    frame_vals[1] = float(hips_height)  # Y up
    frame_vals[2] = 0.0  # Z

    frame_line = " ".join(f"{v:.6f}" for v in frame_vals)
    dt = 1.0 / fps if fps > 0 else 0.05

    motion = [
        "MOTION",
        f"Frames: {n_frames}",
        f"Frame Time: {dt:.6f}",
    ]
    for _ in range(n_frames):
        motion.append(frame_line)

    out_bvh.parent.mkdir(parents=True, exist_ok=True)
    out_bvh.write_text(hier + "\n".join(motion) + "\n", encoding="utf-8")

    return {
        "out": str(out_bvh),
        "frames": n_frames,
        "channels": nchan,
        "hips_y": hips_height,
        "fps": fps,
        "note": "All joint rotations zero; hierarchy OFFSETs define T-pose bind (MoMask/HumanML3D style Y-up).",
    }


def compare_lengths_blender(bvh: Path, fbx: Path, out_prefix: Path) -> dict:
    """Use Blender for accurate WORLD rest lengths (Mixamo scale applied)."""
    import subprocess

    script = r"""
import bpy, json, math, statistics
from pathlib import Path
from mathutils import Vector

bvh = Path(r""" + repr(str(bvh)) + r""")
fbx = Path(r""" + repr(str(fbx)) + r""")
out_prefix = Path(r""" + repr(str(out_prefix)) + r""")
SEGMENTS = """ + repr(SEGMENTS) + r"""

bpy.ops.wm.read_homefile(use_empty=True)
before = set(bpy.data.objects.keys())
bpy.ops.import_anim.bvh(filepath=str(bvh), axis_forward='-Z', axis_up='Y', target='ARMATURE',
    global_scale=1.0, frame_start=1, use_fps_scale=False, update_scene_fps=False, update_scene_duration=False)
src = next(bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before and bpy.data.objects[k].type=='ARMATURE')
if src.animation_data:
    src.animation_data.action = None
bpy.context.view_layer.update()

before = set(bpy.data.objects.keys())
bpy.ops.import_scene.fbx(filepath=str(fbx), ignore_leaf_bones=True, automatic_bone_orientation=False)
dst = next(bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before and bpy.data.objects[k].type=='ARMATURE')
if dst.animation_data:
    dst.animation_data.action = None
bpy.ops.object.select_all(action='DESELECT')
dst.select_set(True)
bpy.context.view_layer.objects.active = dst
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.select_all(action='SELECT')
bpy.ops.pose.transforms_clear()
bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.view_layer.update()

def rest_len(arm, a, b):
    if a not in arm.data.bones or b not in arm.data.bones:
        return None
    A = arm.matrix_world @ arm.data.bones[a].head_local
    B = arm.matrix_world @ arm.data.bones[b].head_local
    return (B - A).length

rows = []
lines = []
lines.append('Bone length comparison (WORLD units, REST pose)')
lines.append(f'BVH: {bvh.name}  object_scale={tuple(round(x,4) for x in src.scale)}')
lines.append(f'FBX: {fbx.name}  object_scale={tuple(round(x,5) for x in dst.scale)}  armature={dst.name}')
lines.append('')
hdr = f'{"segment":28s} {"BVH":>10s} {"Mixamo":>10s} {"diff":>10s} {"ratio":>10s} {"pct":>8s}'
lines.append(hdr)
lines.append('-' * 90)

for label, ba, bb, ma, mb in SEGMENTS:
    lb = rest_len(src, ba, bb)
    lm = rest_len(dst, ma, mb)
    if lb is None or lm is None:
        lines.append(f'{label:28s}  MISSING')
        continue
    diff = lb - lm
    ratio = lb / lm if lm > 1e-9 else float('nan')
    pct = 100.0 * (ratio - 1.0)
    rows.append({
        'segment': label,
        'bvh_a': ba, 'bvh_b': bb,
        'mixamo_a': ma, 'mixamo_b': mb,
        'bvh_len': round(lb, 5),
        'mixamo_len': round(lm, 5),
        'diff_bvh_minus_mixamo': round(diff, 5),
        'ratio_bvh_over_mixamo': round(ratio, 4),
        'pct_longer_bvh': round(pct, 2),
    })
    lines.append(f'{label:28s} {lb:10.4f} {lm:10.4f} {diff:10.4f} {ratio:10.4f} {pct:7.1f}%')

ratios = [r['ratio_bvh_over_mixamo'] for r in rows]
lines.append('')
lines.append(f'mean ratio BVH/Mixamo = {statistics.mean(ratios):.4f}')
lines.append(f'min={min(ratios):.4f}  max={max(ratios):.4f}')
lines.append('')
lines.append('Interpretation:')
lines.append('- pct > 0: BVH segment longer than Mixamo')
lines.append('- pct < 0: BVH shorter than Mixamo')
lines.append('- Uniform scale can match hips-head overall, but spine/collar/legs ratios still differ')
lines.append('- That is why retarget can look OK on legs but odd on torso without good spine rotations')

txt = '\n'.join(lines) + '\n'
out_prefix.parent.mkdir(parents=True, exist_ok=True)
out_prefix.with_suffix('.txt').write_text(txt, encoding='utf-8')
out_prefix.with_suffix('.json').write_text(json.dumps({
    'bvh': str(bvh),
    'fbx': str(fbx),
    'bvh_object_scale': list(src.scale),
    'mixamo_object_scale': list(dst.scale),
    'segments': rows,
    'mean_ratio_bvh_over_mixamo': round(statistics.mean(ratios), 4),
    'min_ratio': round(min(ratios), 4),
    'max_ratio': round(max(ratios), 4),
}, indent=2), encoding='utf-8')
print(txt)
print('WROTE', out_prefix.with_suffix('.txt'))
print('WROTE', out_prefix.with_suffix('.json'))
"""
    blender = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
    if not blender.is_file():
        raise SystemExit(f"Blender not found: {blender}")
    subprocess.check_call(
        [str(blender), "--background", "--python-expr", script],
        cwd=str(ROOT),
    )
    return json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))


def main():
    args = parse_args()
    bvh = Path(args.bvh)
    if not bvh.is_absolute():
        bvh = ROOT / bvh
    fbx = Path(args.fbx)
    if not fbx.is_absolute():
        fbx = ROOT / fbx
    out_tpose = Path(args.out_tpose)
    if not out_tpose.is_absolute():
        out_tpose = ROOT / out_tpose
    out_report = Path(args.out_report)
    if not out_report.is_absolute():
        out_report = ROOT / out_report

    print("=== Bone length BVH vs Mixamo FBX ===")
    if fbx.is_file() and bvh.is_file():
        compare_lengths_blender(bvh, fbx, out_report)
    else:
        print("skip compare (missing bvh or fbx)")

    print("\n=== Write T-pose BVH ===")
    info = write_tpose_bvh(bvh, out_tpose, args.frames, args.fps, args.hips_height)
    print(json.dumps(info, indent=2))

    # also dump offset lengths from hierarchy (no blender)
    text = bvh.read_text(encoding="utf-8", errors="ignore")
    offs = parse_bvh_offsets(text)
    print("\nBVH hierarchy OFFSET lengths (child offset = bone length):")
    for name, (x, y, z) in offs.items():
        L = math.sqrt(x * x + y * y + z * z)
        print(f"  {name:16s} offset=({x:8.4f},{y:8.4f},{z:8.4f}) len={L:.4f}")


if __name__ == "__main__":
    main()
