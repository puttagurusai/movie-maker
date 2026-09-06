"""
Find where the *first complete* action ends inside a MoMask joint sequence / BVH.

Problem: MoMask often continues after jump+hi with a *different* second phrase
in the same file (not external loop). Stretching alone cannot invent "hi" if
it was never generated, and a fixed short length may only contain jump.

Strategy:
  1) Generate long enough to likely contain full jump+hi
  2) Detect first cycle end (events + energy settle OR self-similarity restart)
  3) Trim BVH to [0, cut) and re-apply rest ease

HML22 indices (Y-up):
  0 pelvis  10/11 feet  15 head  16/17 shoulders  20/21 wrists
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOMASK = ROOT / "third_party" / "momask-codes"


def _smooth(x: np.ndarray, w: int = 5) -> np.ndarray:
    w = max(1, int(w))
    if w == 1 or len(x) < w:
        return x.astype(np.float64)
    k = np.ones(w, dtype=np.float64) / w
    return np.convolve(x.astype(np.float64), k, mode="same")


def _peaks(y: np.ndarray, min_prom: float, min_dist: int) -> list[int]:
    """Simple local-max peaks with prominence and min distance."""
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    if T < 3:
        return []
    cands = []
    for i in range(1, T - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] >= min_prom:
            cands.append(i)
    # greedily keep by height with min_dist
    cands.sort(key=lambda i: -y[i])
    kept = []
    for i in cands:
        if all(abs(i - j) >= min_dist for j in kept):
            kept.append(i)
    return sorted(kept)


def motion_energy(joints: np.ndarray) -> np.ndarray:
    """Per-frame mean joint speed (T,)."""
    d = np.diff(joints, axis=0)  # T-1,22,3
    sp = np.linalg.norm(d, axis=-1).mean(axis=-1)
    return np.concatenate([[0.0], sp])


def find_intelligent_cut_frame(
    joints: np.ndarray,
    *,
    fps: float = 20.0,
    compound: bool = True,
    min_frac: float = 0.35,
    max_frac: float = 0.92,
) -> Dict[str, Any]:
    """
    Return cut frame index (exclusive): keep joints[0:cut].

    Heuristics (combined):
      A) Jump = peak pelvis/foot height
      B) Hi/wave = peak wrist raise after mid/jump
      C) Settle = energy drops after last required event
      D) Cycle restart = pose similar to early window again → cut before restart
    """
    j = np.asarray(joints, dtype=np.float64)
    if j.ndim != 3 or j.shape[1] < 22:
        raise ValueError(f"need (T,22+,3), got {j.shape}")
    T = j.shape[0]
    if T < 16:
        return {"cut": T, "reason": "too_short", "events": {}}

    t0 = max(2, int(T * min_frac))
    t1 = min(T - 2, int(T * max_frac))

    pelvis_y = j[:, 0, 1]
    foot_y = np.maximum(j[:, 10, 1], j[:, 11, 1])
    jump_sig = _smooth(0.6 * pelvis_y + 0.4 * foot_y, 5)
    jump_sig = jump_sig - np.median(jump_sig[: max(5, T // 10)])

    l_wr, r_wr = j[:, 20, 1], j[:, 21, 1]
    l_sh, r_sh = j[:, 16, 1], j[:, 17, 1]
    wrist_raise = _smooth(np.maximum(l_wr - l_sh, r_wr - r_sh), 5)

    energy = _smooth(motion_energy(j), 5)
    e_thr = float(np.percentile(energy, 40))
    e_hi = float(np.percentile(energy, 70))

    # Peaks
    j_prom = float(np.percentile(jump_sig, 75))
    w_prom = float(max(0.02, np.percentile(wrist_raise, 70)))
    min_dist = max(4, int(0.25 * fps))
    jump_peaks = _peaks(jump_sig, min_prom=j_prom * 0.5 + 1e-6, min_dist=min_dist)
    wave_peaks = _peaks(wrist_raise, min_prom=w_prom * 0.5, min_dist=min_dist)

    events: Dict[str, Any] = {
        "jump_peaks": jump_peaks,
        "wave_peaks": wave_peaks,
        "jump_prom": j_prom,
        "wave_prom": w_prom,
    }

    # --- Primary: after all required events, find settle ---
    required_end = 0
    if compound:
        jp = jump_peaks[0] if jump_peaks else None
        # first wave peak at/after jump (or any strong wave)
        wp = None
        for w in wave_peaks:
            if jp is None or w >= jp - int(0.15 * fps):
                wp = w
                break
        if wp is None and wave_peaks:
            wp = wave_peaks[0]
        events["jump0"] = jp
        events["wave0"] = wp
        ends = [x for x in (jp, wp) if x is not None]
        if ends:
            required_end = max(ends)
        else:
            # energy peaks as proxy
            e_peaks = _peaks(energy, min_prom=e_hi * 0.6, min_dist=min_dist)
            events["energy_peaks"] = e_peaks
            if len(e_peaks) >= 2:
                required_end = e_peaks[1]
            elif e_peaks:
                required_end = e_peaks[0]
            else:
                required_end = t0
    else:
        e_peaks = _peaks(energy, min_prom=e_hi * 0.55, min_dist=min_dist)
        events["energy_peaks"] = e_peaks
        required_end = e_peaks[0] if e_peaks else t0

    # Settle: first stretch after required_end where energy stays low
    settle_need = max(3, int(0.2 * fps))
    cut_settle = None
    low = 0
    for i in range(required_end + 1, t1):
        if energy[i] <= e_thr * 1.15:
            low += 1
            if low >= settle_need:
                cut_settle = i + 1
                break
        else:
            low = 0
    if cut_settle is None:
        cut_settle = min(T, required_end + int(0.8 * fps))

    # --- Secondary: self-similarity restart (second cycle beginning) ---
    # Compare pose to mean of first 15% active frames
    flat = j.reshape(T, -1)
    # reference = early motion window after small lead-in
    r0, r1 = max(1, int(0.05 * T)), max(3, int(0.2 * T))
    ref = flat[r0:r1].mean(axis=0)
    ref_n = np.linalg.norm(ref) + 1e-8
    sim = []
    for i in range(T):
        v = flat[i]
        sim.append(float(np.dot(v, ref) / ((np.linalg.norm(v) + 1e-8) * ref_n)))
    sim = _smooth(np.array(sim), 7)
    # after required_end, if sim rises near early high similarity → cycle restart
    early_sim = float(np.median(sim[r0:r1]))
    cut_cycle = None
    search_from = max(required_end + int(0.3 * fps), int(0.4 * T))
    for i in range(search_from, t1):
        if sim[i] >= early_sim * 0.97 and energy[i] > e_thr:
            # stepped back a bit into the valley before restart
            cut_cycle = max(required_end + settle_need, i - int(0.15 * fps))
            break

    # Choose cut: prefer settle after events; if cycle restart earlier, use that
    cut = cut_settle
    reason = "settle_after_events"
    if cut_cycle is not None and cut_cycle < cut_settle:
        cut = cut_cycle
        reason = "before_second_cycle"
    # clamp
    cut = int(np.clip(cut, t0, t1))
    # ensure we kept something after last event
    if required_end and cut < required_end + 2:
        cut = min(t1, required_end + settle_need)

    return {
        "cut": cut,
        "reason": reason,
        "T": T,
        "required_end": int(required_end),
        "cut_settle": int(cut_settle) if cut_settle is not None else None,
        "cut_cycle": int(cut_cycle) if cut_cycle is not None else None,
        "events": events,
        "fps": fps,
    }


def trim_joints(joints: np.ndarray, cut: int) -> np.ndarray:
    cut = int(np.clip(cut, 2, len(joints)))
    return joints[:cut].copy()


def trim_bvh_file(
    bvh_in: Path,
    bvh_out: Path,
    cut: int,
    *,
    re_rest_ease: bool = True,
) -> Dict[str, Any]:
    """Slice BVH animation to [0, cut) and optionally re-apply rest ease."""
    import os
    import sys

    os.chdir(MOMASK)
    if str(MOMASK) not in sys.path:
        sys.path.insert(0, str(MOMASK))
    import visualization.BVH_mod as BVH
    from visualization.joints2bvh import ease_rest_in_out

    anim = BVH.load(str(bvh_in), need_quater=True)
    T = anim.rotations.qs.shape[0]
    cut = int(np.clip(cut, 2, T))
    # slice
    anim.rotations.qs = anim.rotations.qs[:cut]
    anim.positions = anim.positions[:cut]
    if re_rest_ease:
        # strip existing ease pads roughly: if starts near identity, drop first pads
        anim = ease_rest_in_out(anim, n_in=5, n_out=8)
    bvh_out = Path(bvh_out)
    bvh_out.parent.mkdir(parents=True, exist_ok=True)
    BVH.save(
        str(bvh_out),
        anim,
        names=anim.names,
        frametime=1 / 20,
        order="zyx",
        quater=True,
    )
    return {"cut": cut, "out_frames": anim.rotations.qs.shape[0], "out": str(bvh_out)}


def cut_from_npy_and_bvh(
    npy_path: Optional[Path],
    bvh_path: Path,
    *,
    compound: bool = True,
    out_bvh: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Prefer joints npy (HML) for detection; fall back to BVH global positions
    reordered if needed.
    """
    import os
    import sys

    joints = None
    if npy_path and Path(npy_path).is_file():
        joints = np.load(str(npy_path))
        if joints.ndim == 3 and joints.shape[1] >= 22:
            pass
        else:
            joints = None

    if joints is None:
        os.chdir(MOMASK)
        if str(MOMASK) not in sys.path:
            sys.path.insert(0, str(MOMASK))
        import visualization.BVH_mod as BVH
        import visualization.Animation as Animation

        anim = BVH.load(str(bvh_path), need_quater=True)
        glb = Animation.positions_global(anim)  # BVH order
        # map BVH order → rough HML for indices we use
        # BVH: 0 Hips, 1 LUp, 2 LLeg, 3 LFoot, 4 LToe, 5 RUp, 6 RLeg, 7 RFoot, 8 RToe,
        # 9 Spine, ... 13 Head, 14 LSh, 15 LArm, 16 LFore, 17 LHand, 18 RSh, 19 RArm, 20 RFore, 21 RHand
        # HML: 0 pel, 10 Lfoot, 11 Rfoot, 15 head, 16 Lsh, 17 Rsh, 20 Lwr, 21 Rwr
        T = glb.shape[0]
        hml = np.zeros((T, 22, 3), dtype=np.float64)
        hml[:, 0] = glb[:, 0]
        hml[:, 10] = glb[:, 4] if glb.shape[1] > 4 else glb[:, 3]
        hml[:, 11] = glb[:, 8] if glb.shape[1] > 8 else glb[:, 7]
        hml[:, 15] = glb[:, 13]
        hml[:, 16] = glb[:, 15]  # LeftArm ~ shoulder joint in MoMask map
        hml[:, 17] = glb[:, 19]
        hml[:, 20] = glb[:, 17]
        hml[:, 21] = glb[:, 21]
        joints = hml

    info = find_intelligent_cut_frame(joints, compound=compound)
    cut = int(info["cut"])
    # If cut is almost full length, still ok
    out = Path(out_bvh) if out_bvh else Path(bvh_path).with_name(Path(bvh_path).stem + "_cut.bvh")
    # Map cut from joints T to BVH T if lengths differ (rest pads)
    # When joints from npy (pre-ease) vs bvh (post-ease): prefer cutting BVH by ratio
    os.chdir(MOMASK)
    if str(MOMASK) not in sys.path:
        sys.path.insert(0, str(MOMASK))
    import visualization.BVH_mod as BVH

    anim = BVH.load(str(bvh_path), need_quater=True)
    Tb = anim.rotations.qs.shape[0]
    Tj = joints.shape[0]
    if Tb != Tj and Tj > 0:
        # scale cut into BVH timeline (if npy is pre-pad and bvh post-pad, ratio works poorly)
        # Prefer: if BVH longer (pads), cut_bvh = cut + half of extra pads at start heuristic
        ratio = Tb / float(Tj)
        cut_b = int(round(cut * ratio))
        cut_b = int(np.clip(cut_b, 8, Tb))
    else:
        cut_b = int(np.clip(cut, 8, Tb))

    trim = trim_bvh_file(bvh_path, out, cut_b, re_rest_ease=True)
    info["bvh_cut"] = cut_b
    info["trim"] = trim
    info["out_bvh"] = str(out)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh", required=True)
    ap.add_argument("--npy", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--compound", action="store_true", default=True)
    ap.add_argument("--simple", action="store_true")
    args = ap.parse_args()
    compound = not args.simple
    out = args.out or str(Path(args.bvh).with_name(Path(args.bvh).stem + "_cut.bvh"))
    info = cut_from_npy_and_bvh(
        Path(args.npy) if args.npy else None,
        Path(args.bvh),
        compound=compound,
        out_bvh=Path(out),
    )
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
