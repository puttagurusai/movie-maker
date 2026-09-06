"""
body_agent.py — realistic procedural body actions for SMPL-X.

Each action is a short multi-keyframe clip (anticipation → peak → settle),
sampled with smooth easing. Output = local XYZ euler offsets (radians)
relative to rest pose.

Packet shape (UDP type=body):
  { "type": "body", "bones": { "left_shoulder": {"x":..,"y":..,"z":..}, ... },
    "action": "wave", "t": 0.3 }
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .base import BaseFaceAgent, FaceContext
from .director_schema import ACTION_CATALOG, normalize_action

BoneEuler = Dict[str, float]
Pose = Dict[str, BoneEuler]
# (norm_time 0..1, pose)
Keyframe = Tuple[float, Pose]
# queue entry: action, t0, duration, weight_scale
QueueItem = Tuple[str, float, float, float]


def _e(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> BoneEuler:
    return {"x": float(x), "y": float(y), "z": float(z)}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def _smoothstep(t: float) -> float:
    t = _clamp(t)
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(t: float) -> float:
    t = _clamp(t)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_eul(a: BoneEuler, b: BoneEuler, t: float) -> BoneEuler:
    return _e(
        _lerp(a.get("x", 0), b.get("x", 0), t),
        _lerp(a.get("y", 0), b.get("y", 0), t),
        _lerp(a.get("z", 0), b.get("z", 0), t),
    )


def _scale_pose(pose: Pose, w: float) -> Pose:
    if w >= 0.999:
        return pose
    out: Pose = {}
    for k, e in pose.items():
        out[k] = _e(e.get("x", 0) * w, e.get("y", 0) * w, e.get("z", 0) * w)
    return out


def _add_pose(dst: Pose, src: Pose, w: float = 1.0) -> None:
    if w <= 1e-5:
        return
    for name, eul in src.items():
        cur = dst.get(name, _e())
        dst[name] = _e(
            cur["x"] + eul.get("x", 0) * w,
            cur["y"] + eul.get("y", 0) * w,
            cur["z"] + eul.get("z", 0) * w,
        )


def _sample_keyframes(keys: Sequence[Keyframe], u: float) -> Pose:
    """u in [0,1] along clip. Smoothstep between neighboring keys."""
    if not keys:
        return {}
    u = _clamp(u)
    if u <= keys[0][0]:
        return keys[0][1]
    if u >= keys[-1][0]:
        return keys[-1][1]
    for i in range(len(keys) - 1):
        t0, p0 = keys[i]
        t1, p1 = keys[i + 1]
        if t0 <= u <= t1:
            span = max(t1 - t0, 1e-4)
            s = _smootherstep((u - t0) / span)
            names = set(p0) | set(p1)
            out: Pose = {}
            for n in names:
                out[n] = _lerp_eul(p0.get(n, _e()), p1.get(n, _e()), s)
            return out
    return keys[-1][1]


# ── SMPL-X axis helpers (calibrated via temp/axis_probe screenshots) ────────
# Rest pose of body.glb is ~T-pose (arms horizontal).
# RIGHT shoulder:  z<0 raises, z>0 lowers;  x>0 also lifts slightly
# LEFT  shoulder:  z>0 raises, z<0 lowers  (mirrored)
# Elbow either side: x>0 flexes (bends) the forearm

def _r_sh(raise_up: float = 0.0, forward: float = 0.0, twist: float = 0.0) -> BoneEuler:
    """Right shoulder: raise_up>0 lifts arm from T-pose toward head."""
    return _e(0.25 * raise_up + 0.15 * forward, -0.35 * forward + twist, -0.95 * raise_up)


def _l_sh(raise_up: float = 0.0, forward: float = 0.0, twist: float = 0.0) -> BoneEuler:
    """Left shoulder: raise_up>0 lifts arm from T-pose toward head."""
    return _e(0.25 * raise_up + 0.15 * forward, 0.35 * forward + twist, 0.95 * raise_up)


def _r_elb(flex: float = 0.0, twist: float = 0.0) -> BoneEuler:
    return _e(flex, twist, -0.15 * flex)


def _l_elb(flex: float = 0.0, twist: float = 0.0) -> BoneEuler:
    return _e(flex, -twist, 0.15 * flex)


# Soft A-pose-ish hold (lower arms slightly from T-pose for natural talk)
_REST_SOFT: Pose = {
    "left_shoulder": _l_sh(raise_up=-0.15, forward=0.05),
    "right_shoulder": _r_sh(raise_up=-0.15, forward=0.05),
    "left_elbow": _l_elb(0.35),
    "right_elbow": _r_elb(0.35),
    "spine2": _e(0.02, 0.0, 0.0),
}

# Continuous emotion posture under actions (body language of the feeling)
# Scaled by intensity in get_body — keeps angry looking angry even mid-talk.
EMOTION_UNDERLAY: Dict[str, Pose] = {
    "happy": {
        "spine2": _e(-0.04, 0.0, 0.0),
        "spine3": _e(-0.05, 0.0, 0.0),
        "left_shoulder": _l_sh(raise_up=-0.05, forward=0.1),
        "right_shoulder": _r_sh(raise_up=-0.05, forward=0.1),
        "neck": _e(-0.03, 0.0, 0.0),
    },
    "sad": {
        "spine1": _e(0.1, 0.0, 0.0),
        "spine2": _e(0.12, 0.0, 0.0),
        "spine3": _e(0.1, 0.0, 0.0),
        "neck": _e(0.14, 0.0, 0.0),
        "head": _e(0.08, 0.0, 0.0),
        "left_shoulder": _l_sh(raise_up=-0.2, forward=0.0),
        "right_shoulder": _r_sh(raise_up=-0.2, forward=0.0),
    },
    "angry": {
        "spine1": _e(-0.06, 0.0, 0.0),
        "spine2": _e(-0.08, 0.0, 0.0),
        "spine3": _e(-0.1, 0.0, 0.0),
        "neck": _e(-0.04, 0.0, 0.0),
        "left_shoulder": _l_sh(raise_up=-0.05, forward=0.12),
        "right_shoulder": _r_sh(raise_up=-0.05, forward=0.15),
    },
    "surprised": {
        "spine2": _e(-0.03, 0.0, 0.0),
        "spine3": _e(-0.04, 0.0, 0.0),
        "neck": _e(-0.06, 0.0, 0.0),
        "left_shoulder": _l_sh(raise_up=0.1, forward=0.15),
        "right_shoulder": _r_sh(raise_up=0.1, forward=0.15),
        "left_elbow": _l_elb(0.4),
        "right_elbow": _r_elb(0.4),
    },
    "disgusted": {
        "spine2": _e(0.06, 0.0, 0.0),
        "spine3": _e(0.08, 0.0, 0.0),
        "neck": _e(0.1, 0.05, 0.0),
        "head": _e(0.06, 0.04, 0.0),
        "left_shoulder": _l_sh(raise_up=-0.05, forward=0.1),
        "right_shoulder": _r_sh(raise_up=-0.05, forward=0.1),
    },
    "fearful": {
        "spine1": _e(0.06, 0.0, 0.0),
        "spine2": _e(0.08, 0.0, 0.0),
        "spine3": _e(0.1, 0.0, 0.0),
        "neck": _e(0.08, 0.0, 0.0),
        "left_shoulder": _l_sh(raise_up=0.12, forward=0.2),
        "right_shoulder": _r_sh(raise_up=0.12, forward=0.2),
        "left_elbow": _l_elb(0.7),
        "right_elbow": _r_elb(0.7),
    },
    "thinking": {
        "spine3": _e(0.03, 0.06, 0.04),
        "neck": _e(0.06, 0.1, 0.05),
        "head": _e(0.04, 0.08, 0.03),
        "right_shoulder": _r_sh(raise_up=0.05, forward=0.1),
    },
    "sarcastic": {
        "spine3": _e(0.02, -0.04, 0.03),
        "neck": _e(0.03, -0.06, 0.0),
        "head": _e(0.02, -0.05, 0.02),
        "left_shoulder": _l_sh(raise_up=-0.1, forward=0.05),
        "right_shoulder": _r_sh(raise_up=-0.08, forward=0.08),
    },
    "apologetic": {
        "spine2": _e(0.08, 0.0, 0.0),
        "spine3": _e(0.1, 0.0, 0.0),
        "neck": _e(0.12, 0.0, 0.0),
        "left_shoulder": _l_sh(raise_up=-0.15, forward=0.0),
        "right_shoulder": _r_sh(raise_up=-0.15, forward=0.0),
    },
    "assertive": {
        "spine1": _e(-0.05, 0.0, 0.0),
        "spine3": _e(-0.08, 0.0, 0.0),
        "left_shoulder": _l_sh(raise_up=-0.05, forward=0.1),
        "right_shoulder": _r_sh(raise_up=-0.05, forward=0.1),
    },
    "encouraging": {
        "spine3": _e(-0.04, 0.0, 0.0),
        "left_shoulder": _l_sh(raise_up=-0.05, forward=0.12),
        "right_shoulder": _r_sh(raise_up=-0.05, forward=0.12),
        "neck": _e(-0.02, 0.0, 0.0),
    },
    "concerned": {
        "spine3": _e(0.04, 0.0, 0.0),
        "neck": _e(0.05, 0.03, 0.0),
        "left_shoulder": _l_sh(raise_up=-0.08, forward=0.05),
        "right_shoulder": _r_sh(raise_up=-0.08, forward=0.05),
    },
    "calm": {
        "spine2": _e(0.02, 0.0, 0.0),
        "left_shoulder": _l_sh(raise_up=-0.12, forward=0.0),
        "right_shoulder": _r_sh(raise_up=-0.12, forward=0.0),
    },
}

# ── Multi-keyframe action clips (more human than single peak * envelope) ────
# Axes are local XYZ euler offsets on SMPL-X rest. Tuned for visible but natural motion.

ACTION_CLIPS: Dict[str, Dict] = {
    "idle": {
        "duration": 0.5,
        "keys": [(0.0, {}), (1.0, {})],
    },
    # Open conversational hands — lower from T-pose, elbows bent (axis-calibrated)
    "talk_open": {
        "duration": 2.2,
        "loop": True,
        "keys": [
            (0.0, {
                "left_shoulder": _l_sh(raise_up=-0.12, forward=0.12),
                "right_shoulder": _r_sh(raise_up=-0.12, forward=0.12),
                "left_elbow": _l_elb(0.55),
                "right_elbow": _r_elb(0.55),
                "left_wrist": _e(0.0, 0.1, 0.08),
                "right_wrist": _e(0.0, -0.1, -0.08),
                "spine3": _e(0.0, 0.0, 0.02),
            }),
            (0.35, {
                "left_shoulder": _l_sh(raise_up=-0.08, forward=0.22),
                "right_shoulder": _r_sh(raise_up=-0.05, forward=0.28),
                "left_elbow": _l_elb(0.7),
                "right_elbow": _r_elb(0.85),
                "left_wrist": _e(0.05, 0.15, 0.1),
                "right_wrist": _e(0.08, -0.12, -0.12),
                "spine3": _e(-0.02, 0.03, -0.04),
                "spine2": _e(0.0, 0.02, -0.02),
            }),
            (0.7, {
                "left_shoulder": _l_sh(raise_up=-0.05, forward=0.3),
                "right_shoulder": _r_sh(raise_up=-0.1, forward=0.18),
                "left_elbow": _l_elb(0.8),
                "right_elbow": _r_elb(0.6),
                "spine3": _e(-0.02, -0.02, 0.04),
            }),
            (1.0, {
                "left_shoulder": _l_sh(raise_up=-0.12, forward=0.12),
                "right_shoulder": _r_sh(raise_up=-0.12, forward=0.12),
                "left_elbow": _l_elb(0.55),
                "right_elbow": _r_elb(0.55),
                "spine3": _e(0.0, 0.0, 0.02),
            }),
        ],
    },
    # Beat / emphasis — right hand forward chop
    "talk_emphasize": {
        "duration": 1.6,
        "loop": True,
        "keys": [
            (0.0, {
                "right_shoulder": _r_sh(raise_up=-0.05, forward=0.35),
                "right_elbow": _r_elb(0.95),
                "right_wrist": _e(0.1, -0.1, -0.15),
                "spine3": _e(0.0, 0.0, -0.06),
                "left_shoulder": _l_sh(raise_up=-0.15, forward=0.05),
                "left_elbow": _l_elb(0.4),
            }),
            (0.4, {
                "right_shoulder": _r_sh(raise_up=0.05, forward=0.55),
                "right_elbow": _r_elb(0.45),
                "right_wrist": _e(0.2, 0.0, -0.1),
                "spine3": _e(-0.04, 0.05, -0.1),
                "spine2": _e(-0.02, 0.03, -0.05),
                "left_shoulder": _l_sh(raise_up=-0.12, forward=0.08),
            }),
            (0.7, {
                "right_shoulder": _r_sh(raise_up=-0.05, forward=0.3),
                "right_elbow": _r_elb(0.75),
                "spine3": _e(0.02, 0.0, -0.04),
            }),
            (1.0, {
                "right_shoulder": _r_sh(raise_up=-0.08, forward=0.25),
                "right_elbow": _r_elb(0.6),
                "spine3": _e(0.0, 0.0, -0.03),
            }),
        ],
    },
    # Friendly wave — raise R arm (z-), bend elbow, wag wrist (axis-calibrated)
    "wave": {
        "duration": 2.0,
        "keys": [
            (0.0, _REST_SOFT),
            (0.15, {
                "pelvis": _e(0.0, 0.0, -0.03),
                "spine3": _e(0.0, 0.04, -0.04),
                "right_shoulder": _r_sh(raise_up=0.55, forward=0.1),
                "right_elbow": _r_elb(0.85),
                "left_shoulder": _l_sh(raise_up=-0.2, forward=0.05),
                "left_elbow": _l_elb(0.4),
            }),
            (0.32, {
                "pelvis": _e(0.0, 0.0, -0.04),
                "spine3": _e(-0.02, 0.06, -0.06),
                "right_collar": _e(0.05, 0.05, -0.08),
                "right_shoulder": _r_sh(raise_up=0.95, forward=0.15),
                "right_elbow": _r_elb(1.05),
                "right_wrist": _e(0.1, 0.2, -0.25),
                "left_shoulder": _l_sh(raise_up=-0.18, forward=0.05),
                "neck": _e(0.0, 0.05, 0.0),
            }),
            (0.48, {
                "spine3": _e(-0.02, 0.06, -0.06),
                "right_shoulder": _r_sh(raise_up=1.0, forward=0.2),
                "right_elbow": _r_elb(0.95, twist=0.15),
                "right_wrist": _e(0.1, 0.35, -0.3),
                "left_shoulder": _l_sh(raise_up=-0.18, forward=0.05),
            }),
            (0.62, {
                "spine3": _e(-0.02, 0.06, -0.06),
                "right_shoulder": _r_sh(raise_up=0.95, forward=0.1),
                "right_elbow": _r_elb(1.05, twist=-0.1),
                "right_wrist": _e(0.15, -0.2, -0.2),
                "left_shoulder": _l_sh(raise_up=-0.18, forward=0.05),
            }),
            (0.76, {
                "spine3": _e(-0.02, 0.06, -0.06),
                "right_shoulder": _r_sh(raise_up=1.0, forward=0.18),
                "right_elbow": _r_elb(0.98, twist=0.12),
                "right_wrist": _e(0.12, 0.3, -0.28),
                "left_shoulder": _l_sh(raise_up=-0.18, forward=0.05),
            }),
            (0.9, {
                "right_shoulder": _r_sh(raise_up=0.4, forward=0.1),
                "right_elbow": _r_elb(0.7),
                "spine3": _e(0.0, 0.03, -0.03),
                "left_shoulder": _l_sh(raise_up=-0.15, forward=0.05),
            }),
            (1.0, _REST_SOFT),
        ],
    },
    # Shrug: small shoulder raise + strong elbow flex (NOT arms overhead)
    "shrug": {
        "duration": 1.35,
        "keys": [
            (0.0, _REST_SOFT),
            (0.25, {
                "left_collar": _e(0.08, 0.0, 0.12),
                "right_collar": _e(0.08, 0.0, -0.12),
                "left_shoulder": _l_sh(raise_up=0.2, forward=0.15),
                "right_shoulder": _r_sh(raise_up=0.2, forward=0.15),
                "left_elbow": _l_elb(1.15, twist=0.2),
                "right_elbow": _r_elb(1.15, twist=0.2),
                "left_wrist": _e(0.15, 0.35, 0.2),
                "right_wrist": _e(0.15, -0.35, -0.2),
                "spine3": _e(0.04, 0.0, 0.0),
                "neck": _e(0.06, 0.0, 0.0),
            }),
            (0.5, {
                "left_collar": _e(0.12, 0.0, 0.18),
                "right_collar": _e(0.12, 0.0, -0.18),
                "left_shoulder": _l_sh(raise_up=0.28, forward=0.2),
                "right_shoulder": _r_sh(raise_up=0.28, forward=0.2),
                "left_elbow": _l_elb(1.35, twist=0.25),
                "right_elbow": _r_elb(1.35, twist=0.25),
                "left_wrist": _e(0.2, 0.5, 0.3),
                "right_wrist": _e(0.2, -0.5, -0.3),
                "spine3": _e(0.06, 0.0, 0.0),
                "neck": _e(0.1, 0.0, 0.0),
                "head": _e(0.05, 0.0, 0.0),
            }),
            (0.75, {
                "left_shoulder": _l_sh(raise_up=0.22, forward=0.18),
                "right_shoulder": _r_sh(raise_up=0.22, forward=0.18),
                "left_elbow": _l_elb(1.2, twist=0.2),
                "right_elbow": _r_elb(1.2, twist=0.2),
                "spine3": _e(0.05, 0.0, 0.0),
                "neck": _e(0.08, 0.0, 0.0),
            }),
            (1.0, _REST_SOFT),
        ],
    },
    "point_forward": {
        "duration": 1.25,
        "keys": [
            (0.0, _REST_SOFT),
            (0.3, {
                "right_shoulder": _r_sh(raise_up=0.15, forward=0.65),
                "right_elbow": _r_elb(0.35),
                "right_wrist": _e(0.1, 0.0, -0.1),
                "spine3": _e(-0.04, 0.06, -0.08),
                "spine2": _e(0.0, 0.03, -0.04),
                "neck": _e(0.0, 0.05, 0.0),
                "left_shoulder": _l_sh(raise_up=-0.15, forward=0.05),
            }),
            (0.55, {
                "right_shoulder": _r_sh(raise_up=0.2, forward=0.85),
                "right_elbow": _r_elb(0.15),
                "right_wrist": _e(0.12, 0.05, -0.08),
                "spine3": _e(-0.05, 0.08, -0.1),
                "neck": _e(-0.02, 0.06, 0.0),
                "left_shoulder": _l_sh(raise_up=-0.12, forward=0.05),
            }),
            (1.0, _REST_SOFT),
        ],
    },
    # Surprise/fear: lean back + hands guard (not arms overhead)
    "recoil": {
        "duration": 1.1,
        "keys": [
            (0.0, _REST_SOFT),
            (0.15, {
                "spine1": _e(0.1, 0.0, 0.0),
                "spine2": _e(0.12, 0.0, 0.0),
                "spine3": _e(0.14, 0.0, 0.0),
                "neck": _e(0.1, 0.0, 0.0),
                "head": _e(0.06, 0.0, 0.0),
                "left_shoulder": _l_sh(raise_up=0.15, forward=0.25),
                "right_shoulder": _r_sh(raise_up=0.15, forward=0.25),
                "left_elbow": _l_elb(1.0),
                "right_elbow": _r_elb(1.0),
                "pelvis": _e(0.04, 0.0, 0.0),
            }),
            (0.4, {
                "spine1": _e(0.14, 0.0, 0.02),
                "spine2": _e(0.16, 0.0, 0.02),
                "spine3": _e(0.18, 0.0, 0.0),
                "neck": _e(0.14, 0.04, 0.0),
                "head": _e(0.1, 0.0, 0.0),
                "left_shoulder": _l_sh(raise_up=0.2, forward=0.35),
                "right_shoulder": _r_sh(raise_up=0.2, forward=0.35),
                "left_elbow": _l_elb(1.2),
                "right_elbow": _r_elb(1.2),
                "left_wrist": _e(0.15, 0.15, 0.15),
                "right_wrist": _e(0.15, -0.15, -0.15),
                "pelvis": _e(0.05, 0.0, 0.0),
            }),
            (0.75, {
                "spine1": _e(0.08, 0.0, 0.0),
                "spine2": _e(0.1, 0.0, 0.0),
                "spine3": _e(0.1, 0.0, 0.0),
                "neck": _e(0.08, 0.0, 0.0),
                "left_shoulder": _l_sh(raise_up=0.1, forward=0.2),
                "right_shoulder": _r_sh(raise_up=0.1, forward=0.2),
                "left_elbow": _l_elb(0.9),
                "right_elbow": _r_elb(0.9),
            }),
            (1.0, _REST_SOFT),
        ],
    },
    # Soft celebration — arms raised moderately (not full T overhead)
    "celebrate": {
        "duration": 1.5,
        "keys": [
            (0.0, _REST_SOFT),
            (0.25, {
                "left_shoulder": _l_sh(raise_up=0.55, forward=0.15),
                "right_shoulder": _r_sh(raise_up=0.55, forward=0.15),
                "left_elbow": _l_elb(0.45),
                "right_elbow": _r_elb(0.45),
                "spine3": _e(-0.08, 0.0, 0.0),
                "spine2": _e(-0.04, 0.0, 0.0),
                "neck": _e(-0.04, 0.0, 0.0),
            }),
            (0.55, {
                "left_shoulder": _l_sh(raise_up=0.85, forward=0.2),
                "right_shoulder": _r_sh(raise_up=0.85, forward=0.2),
                "left_elbow": _l_elb(0.35),
                "right_elbow": _r_elb(0.35),
                "spine3": _e(-0.1, 0.0, 0.0),
                "spine2": _e(-0.05, 0.0, 0.0),
                "neck": _e(-0.06, 0.0, 0.0),
                "head": _e(-0.03, 0.0, 0.0),
            }),
            (0.8, {
                "left_shoulder": _l_sh(raise_up=0.6, forward=0.15),
                "right_shoulder": _r_sh(raise_up=0.6, forward=0.15),
                "left_elbow": _l_elb(0.4),
                "right_elbow": _r_elb(0.4),
                "spine3": _e(-0.06, 0.0, 0.0),
            }),
            (1.0, _REST_SOFT),
        ],
    },
    "slump": {
        "duration": 2.0,
        "loop": True,
        "keys": [
            (0.0, {
                "spine1": _e(0.12, 0.0, 0.0),
                "spine2": _e(0.16, 0.0, 0.02),
                "spine3": _e(0.14, 0.0, 0.0),
                "neck": _e(0.18, 0.0, 0.0),
                "head": _e(0.1, 0.0, 0.0),
                "left_shoulder": _l_sh(raise_up=-0.25, forward=0.05),
                "right_shoulder": _r_sh(raise_up=-0.25, forward=0.05),
                "left_elbow": _l_elb(0.45),
                "right_elbow": _r_elb(0.45),
                "pelvis": _e(0.04, 0.0, 0.0),
            }),
            (0.5, {
                "spine1": _e(0.14, 0.0, 0.02),
                "spine2": _e(0.18, 0.0, 0.0),
                "spine3": _e(0.16, 0.0, -0.02),
                "neck": _e(0.2, 0.03, 0.0),
                "head": _e(0.12, 0.0, 0.0),
                "left_shoulder": _l_sh(raise_up=-0.28, forward=0.05),
                "right_shoulder": _r_sh(raise_up=-0.28, forward=0.05),
            }),
            (1.0, {
                "spine1": _e(0.12, 0.0, 0.0),
                "spine2": _e(0.16, 0.0, 0.02),
                "spine3": _e(0.14, 0.0, 0.0),
                "neck": _e(0.18, 0.0, 0.0),
                "head": _e(0.1, 0.0, 0.0),
                "left_shoulder": _e(0.2, 0.05, 0.18),
                "right_shoulder": _e(0.2, -0.05, -0.18),
            }),
        ],
    },
    # Angry / assertive — upright, shoulders set, slight lean in
    "tense": {
        "duration": 2.0,
        "loop": True,
        "keys": [
            (0.0, {
                "spine1": _e(-0.06, 0.0, 0.0),
                "spine2": _e(-0.08, 0.0, 0.0),
                "spine3": _e(-0.1, 0.0, 0.0),
                "neck": _e(-0.04, 0.0, 0.0),
                "left_shoulder": _l_sh(raise_up=-0.05, forward=0.15),
                "right_shoulder": _r_sh(raise_up=-0.05, forward=0.2),
                "left_elbow": _l_elb(0.55),
                "right_elbow": _r_elb(0.7),
                "pelvis": _e(-0.02, 0.0, 0.0),
            }),
            (0.4, {
                "spine1": _e(-0.07, 0.0, 0.02),
                "spine2": _e(-0.1, 0.0, 0.02),
                "spine3": _e(-0.12, 0.03, -0.04),
                "neck": _e(-0.05, 0.02, 0.0),
                "right_shoulder": _r_sh(raise_up=0.05, forward=0.4),
                "right_elbow": _r_elb(0.85),
                "left_shoulder": _l_sh(raise_up=-0.08, forward=0.12),
            }),
            (0.7, {
                "spine3": _e(-0.11, -0.02, 0.03),
                "left_shoulder": _l_sh(raise_up=0.0, forward=0.25),
                "left_elbow": _l_elb(0.7),
                "right_shoulder": _r_sh(raise_up=-0.05, forward=0.22),
                "right_elbow": _r_elb(0.65),
            }),
            (1.0, {
                "spine1": _e(-0.06, 0.0, 0.0),
                "spine3": _e(-0.1, 0.0, 0.0),
                "left_shoulder": _l_sh(raise_up=-0.05, forward=0.15),
                "right_shoulder": _r_sh(raise_up=-0.05, forward=0.2),
                "left_elbow": _l_elb(0.55),
                "right_elbow": _r_elb(0.7),
            }),
        ],
    },
    # Hand near chin — raise R slightly + strong elbow flex (axis-calibrated)
    "think_chin": {
        "duration": 2.0,
        "keys": [
            (0.0, _REST_SOFT),
            (0.3, {
                "right_shoulder": _r_sh(raise_up=0.45, forward=0.35),
                "right_elbow": _r_elb(1.45, twist=0.25),
                "right_wrist": _e(0.25, 0.2, -0.35),
                "spine3": _e(0.04, 0.08, 0.05),
                "neck": _e(0.08, 0.1, 0.06),
                "head": _e(0.04, 0.08, 0.04),
                "left_shoulder": _l_sh(raise_up=-0.15, forward=0.05),
                "left_elbow": _l_elb(0.4),
            }),
            (0.6, {
                "right_shoulder": _r_sh(raise_up=0.5, forward=0.4),
                "right_elbow": _r_elb(1.55, twist=0.3),
                "right_wrist": _e(0.3, 0.25, -0.4),
                "spine3": _e(0.05, 0.1, 0.06),
                "neck": _e(0.1, 0.12, 0.08),
                "head": _e(0.05, 0.1, 0.05),
                "left_shoulder": _l_sh(raise_up=-0.15, forward=0.05),
            }),
            (1.0, _REST_SOFT),
        ],
    },
    "hands_reject": {
        "duration": 1.1,
        "keys": [
            (0.0, _REST_SOFT),
            (0.25, {
                "left_shoulder": _l_sh(raise_up=0.25, forward=0.45),
                "right_shoulder": _r_sh(raise_up=0.25, forward=0.45),
                "left_elbow": _l_elb(0.5),
                "right_elbow": _r_elb(0.5),
                "left_wrist": _e(0.15, 0.25, 0.2),
                "right_wrist": _e(0.15, -0.25, -0.2),
                "spine3": _e(0.08, 0.0, 0.0),
                "neck": _e(0.06, 0.0, 0.0),
            }),
            (0.5, {
                "left_shoulder": _l_sh(raise_up=0.35, forward=0.7),
                "right_shoulder": _r_sh(raise_up=0.35, forward=0.7),
                "left_elbow": _l_elb(0.35),
                "right_elbow": _r_elb(0.35),
                "left_wrist": _e(0.2, 0.35, 0.25),
                "right_wrist": _e(0.2, -0.35, -0.25),
                "spine3": _e(0.1, 0.0, 0.0),
                "spine2": _e(0.05, 0.0, 0.0),
                "neck": _e(0.08, 0.0, 0.0),
            }),
            (1.0, _REST_SOFT),
        ],
    },
    "nod_yes": {
        "duration": 1.15,
        "keys": [
            (0.0, {}),
            (0.15, {
                "neck": _e(0.28, 0.0, 0.0), "spine3": _e(0.08, 0.0, 0.0),
                "spine2": _e(0.04, 0.0, 0.0), "head": _e(0.12, 0.0, 0.0),
            }),
            (0.32, {
                "neck": _e(-0.06, 0.0, 0.0), "spine3": _e(-0.02, 0.0, 0.0),
                "head": _e(-0.04, 0.0, 0.0),
            }),
            (0.48, {
                "neck": _e(0.24, 0.0, 0.0), "spine3": _e(0.07, 0.0, 0.0),
                "head": _e(0.1, 0.0, 0.0),
            }),
            (0.65, {
                "neck": _e(-0.04, 0.0, 0.0), "head": _e(-0.03, 0.0, 0.0),
            }),
            (0.8, {
                "neck": _e(0.12, 0.0, 0.0), "spine3": _e(0.03, 0.0, 0.0),
                "head": _e(0.05, 0.0, 0.0),
            }),
            (1.0, {}),
        ],
    },
    "shake_no": {
        "duration": 1.15,
        "keys": [
            (0.0, {}),
            (0.14, {
                "neck": _e(0.0, 0.38, 0.0), "spine3": _e(0.0, 0.1, 0.0),
                "head": _e(0.0, 0.14, 0.0),
            }),
            (0.32, {
                "neck": _e(0.0, -0.38, 0.0), "spine3": _e(0.0, -0.1, 0.0),
                "head": _e(0.0, -0.14, 0.0),
            }),
            (0.5, {
                "neck": _e(0.0, 0.32, 0.0), "spine3": _e(0.0, 0.08, 0.0),
                "head": _e(0.0, 0.1, 0.0),
            }),
            (0.68, {
                "neck": _e(0.0, -0.28, 0.0), "spine3": _e(0.0, -0.06, 0.0),
            }),
            (0.85, {"neck": _e(0.0, 0.12, 0.0)}),
            (1.0, {}),
        ],
    },
    "look_camera": {
        "duration": 1.6,
        "loop": True,
        "keys": [
            (0.0, {
                "neck": _e(-0.06, 0.0, 0.0),
                "spine3": _e(-0.04, 0.0, 0.0),
                "head": _e(-0.07, 0.0, 0.0),
                "left_shoulder": _e(-0.05, 0.0, 0.08),
                "right_shoulder": _e(-0.05, 0.0, -0.08),
            }),
            (0.5, {
                "neck": _e(-0.08, 0.02, 0.0),
                "spine3": _e(-0.05, 0.0, 0.0),
                "head": _e(-0.09, 0.0, 0.0),
                "left_shoulder": _e(-0.06, 0.0, 0.1),
                "right_shoulder": _e(-0.06, 0.0, -0.1),
            }),
            (1.0, {
                "neck": _e(-0.06, -0.02, 0.0),
                "spine3": _e(-0.04, 0.0, 0.0),
                "head": _e(-0.07, 0.0, 0.0),
                "left_shoulder": _e(-0.05, 0.0, 0.08),
                "right_shoulder": _e(-0.05, 0.0, -0.08),
            }),
        ],
    },
    "look_left": {
        "duration": 0.9,
        "keys": [
            (0.0, {}),
            (0.5, {"neck": _e(0.0, 0.45, 0.0), "spine3": _e(0.0, 0.12, 0.0), "head": _e(0.0, 0.1, 0.0)}),
            (1.0, {"neck": _e(0.0, 0.4, 0.0), "spine3": _e(0.0, 0.1, 0.0)}),
        ],
    },
    "look_right": {
        "duration": 0.9,
        "keys": [
            (0.0, {}),
            (0.5, {"neck": _e(0.0, -0.45, 0.0), "spine3": _e(0.0, -0.12, 0.0), "head": _e(0.0, -0.1, 0.0)}),
            (1.0, {"neck": _e(0.0, -0.4, 0.0), "spine3": _e(0.0, -0.1, 0.0)}),
        ],
    },
    "weight_shift": {
        "duration": 1.4,
        "keys": [
            (0.0, {}),
            (0.5, {
                "pelvis": _e(0.0, 0.0, 0.18),
                "spine1": _e(0.0, 0.0, 0.12),
                "spine2": _e(0.0, 0.0, 0.06),
                "left_hip": _e(0.05, 0.0, 0.1),
                "right_hip": _e(-0.05, 0.0, -0.1),
            }),
            (1.0, {}),
        ],
    },
}

# Back-compat name used by older imports / checks
ACTION_POSES: Dict[str, Pose] = {
    name: (clip["keys"][len(clip["keys"]) // 2][1] if clip.get("keys") else {})
    for name, clip in ACTION_CLIPS.items()
}

# Base-layer postures (legs/hips) — mirrored in viewer; also sent in UDP so
# sitting/walking works even if AnimationMixer tracks fail to bind.
BASE_STATE_POSES: Dict[str, Pose] = {
    "standing": {},
    "sitting": {
        "pelvis": _e(0.42, 0.0, 0.0),
        "spine1": _e(0.14, 0.0, 0.0),
        "spine2": _e(0.06, 0.0, 0.0),
        "left_hip": _e(-1.45, 0.06, 0.1),
        "right_hip": _e(-1.45, -0.06, -0.1),
        "left_knee": _e(1.55, 0.0, 0.0),
        "right_knee": _e(1.55, 0.0, 0.0),
        "left_ankle": _e(0.25, 0.0, 0.0),
        "right_ankle": _e(0.25, 0.0, 0.0),
    },
    "walking": {
        # Mid-cycle snapshot; viewer also animates walk procedurally
        "pelvis": _e(0.03, 0.0, 0.02),
        "spine1": _e(0.05, 0.0, -0.02),
        "left_hip": _e(0.4, 0.0, 0.05),
        "right_hip": _e(-0.35, 0.0, -0.05),
        "left_knee": _e(0.15, 0.0, 0.0),
        "right_knee": _e(0.75, 0.0, 0.0),
    },
    "dancing": {
        "pelvis": _e(0.06, 0.0, 0.08),
        "spine1": _e(0.05, 0.0, 0.04),
        "left_hip": _e(0.12, 0.0, 0.06),
        "right_hip": _e(0.12, 0.0, -0.06),
    },
}


def _clip_duration(name: str, fallback: float = 1.0) -> float:
    clip = ACTION_CLIPS.get(name)
    if clip:
        return float(clip.get("duration", fallback))
    meta = ACTION_CATALOG.get(name, {})
    return float(meta.get("duration_s", fallback))


def _sample_action(name: str, age: float, duration: float, intensity: float = 1.0) -> Pose:
    clip = ACTION_CLIPS.get(name)
    if not clip:
        return {}
    keys: Sequence[Keyframe] = clip["keys"]
    dur = max(duration, 0.05)
    loop = bool(clip.get("loop"))
    if loop:
        # loop while held; map age into 0..1 of clip cycle
        cycle = float(clip.get("duration", dur))
        u = (age % cycle) / cycle if age >= 0 else 0.0
        # fade in/out edges of full sentence hold
        fade = 0.25
        edge = 1.0
        if age < fade:
            edge = _smoothstep(age / fade)
        # no hard end fade here — prepare() sets finite hold via duration
        pose = _sample_keyframes(keys, u)
        return _scale_pose(pose, intensity * edge)
    # one-shot: 0..1 over duration with ease
    if age < 0 or age > dur:
        return {}
    u = age / dur
    # slight ease on whole clip progress already in key smootherstep
    pose = _sample_keyframes(keys, u)
    return _scale_pose(pose, intensity)


class BodyAgent(BaseFaceAgent):
    name = "body"
    owned_keys = set()

    def __init__(self) -> None:
        self._phase = random.random() * math.tau
        self._queue: List[QueueItem] = []
        self._active_action: Optional[str] = None
        self._action_start = 0.0
        self._intensity = 0.75
        self._base_state = "standing"

    def prepare(self, ctx: FaceContext) -> None:
        self._phase = random.random() * math.tau
        self._queue.clear()
        self._active_action = None
        self._action_start = 0.0
        self._intensity = float(getattr(ctx, "intensity", 0.75) or 0.75)
        self._intensity = _clamp(self._intensity, 0.35, 1.0)
        raw_state = str(
            ctx.extras.get("body_state")
            or ctx.extras.get("state")
            or "standing"
        ).lower().strip()
        if raw_state in ("sit", "seated", "sitting_idle"):
            raw_state = "sitting"
        elif raw_state in ("walk", "walk_loop"):
            raw_state = "walking"
        elif raw_state in ("dance", "wave_dance"):
            raw_state = "dancing"
        elif raw_state in ("stand", "idle", "standing_idle"):
            raw_state = "standing"
        self._base_state = raw_state if raw_state in BASE_STATE_POSES else "standing"

        actions = list(ctx.extras.get("body_actions") or [])
        timing = str(ctx.extras.get("action_timing") or "during").lower()
        sent_dur = max(0.4, float(ctx.duration or 1.0))
        gesture = str(ctx.extras.get("gesture_target") or "none").lower()
        # NOTE: Do NOT drop think_chin / point_forward when gesture_target is set.
        # Viewer IK may fail or be weak; procedural poses must still ship so
        # arms move even without a perfect IK solve.

        for a in actions:
            key = normalize_action(str(a)) or str(a).lower()
            if key not in ACTION_CLIPS and key not in ACTION_CATALOG:
                continue
            clip = ACTION_CLIPS.get(key, {})
            base_dur = _clip_duration(key, 1.0)
            loop = bool(clip.get("loop"))
            # Hold looping talk/posture for most of the line
            if loop:
                dur = max(base_dur, sent_dur * 0.92)
            else:
                dur = min(base_dur * 1.05, max(0.7, sent_dur * 0.85))

            if timing == "start":
                t0 = 0.02
            elif timing == "end":
                t0 = max(0.0, sent_dur - dur)
            else:
                t0 = max(0.05, min(sent_dur * 0.2, sent_dur - dur * 0.45))

            # One-shots: slight randomize start so multi-actions don't stack robotically
            if not loop:
                t0 += random.uniform(0.0, 0.08)
            self._queue.append((key, t0, dur, 1.0))

        # Emotion posture layer if director didn't already put a talk/posture action
        emo = (ctx.emotion or "neutral").lower()
        names = {q[0] for q in self._queue}
        posture_keys = {
            "talk_open", "talk_emphasize", "slump", "tense", "think_chin",
            "recoil", "celebrate", "wave", "shrug",
        }
        if not (names & posture_keys):
            if emo in ("happy", "encouraging"):
                self._queue.insert(0, ("talk_open", 0.0, sent_dur * 0.9, 0.85))
            elif emo in ("sad", "apologetic"):
                self._queue.insert(0, ("slump", 0.0, sent_dur * 0.95, 0.9))
            elif emo in ("angry", "assertive"):
                self._queue.insert(0, ("tense", 0.0, sent_dur * 0.95, 0.95))
            elif emo in ("thinking",):
                self._queue.insert(0, ("think_chin", 0.08, min(1.8, sent_dur * 0.6), 0.9))
            elif emo in ("disgusted",):
                self._queue.insert(0, ("hands_reject", 0.05, 1.0, 1.0))
            elif emo in ("surprised", "fearful"):
                self._queue.insert(0, ("recoil", 0.03, 1.05, 1.0))
            elif emo in ("sarcastic",):
                self._queue.insert(0, ("shrug", 0.08, 1.2, 0.95))
            else:
                self._queue.insert(0, ("talk_open", 0.0, sent_dur * 0.85, 0.7))

        # Sort by start time
        self._queue.sort(key=lambda q: q[1])
        # Resolve Mixamo catalog clip for this beat (Blender Action name)
        try:
            from .motion_controller import MotionController

            if not hasattr(self, "_motion_ctrl") or self._motion_ctrl is None:
                self._motion_ctrl = MotionController()
            plan = self._motion_ctrl.resolve(
                actions=[q[0] for q in self._queue],
                state=self._base_state,
                gesture_target=str(ctx.extras.get("gesture_target") or "none"),
                hand=str(ctx.extras.get("hand") or "right"),
                emotion=str(ctx.emotion or "neutral"),
                intensity=self._intensity,
                text=str(getattr(ctx, "text", "") or ""),
            )
            ctx.extras["motion_plan"] = plan.to_dict()
            print(
                f"[body_agent] Ready state={self._base_state} intensity={self._intensity:.2f} "
                f"clip={plan.clip_id} speed={plan.speed:.2f} "
                f"actions={[(a[0], round(a[1], 2), round(a[2], 2)) for a in self._queue]}"
            )
        except Exception as e:
            print(
                f"[body_agent] Ready state={self._base_state} intensity={self._intensity:.2f} "
                f"actions={[(a[0], round(a[1], 2), round(a[2], 2)) for a in self._queue]} "
                f"(catalog: {e})"
            )

    def sample(self, ctx: FaceContext) -> Dict[str, float]:
        return {}

    def get_body(self, ctx: FaceContext) -> Dict[str, BoneEuler]:
        t = float(ctx.t)
        energy = ctx.speech_energy() if ctx.is_speaking else 0.0
        bones: Pose = {}
        inten = self._intensity

        # Base posture (sitting / walking …) — always present so viewer can sit
        # even if AnimationMixer clip tracks fail to bind.
        if self._base_state == "walking":
            ph = t * 5.5 + self._phase
            a = math.sin(ph)
            b = math.sin(ph + math.pi)
            base_pose = {
                "pelvis": _e(0.03, 0.0, a * 0.03),
                "spine1": _e(0.05, 0.0, -a * 0.02),
                "left_hip": _e(a * 0.5, 0.0, 0.05),
                "right_hip": _e(b * 0.5, 0.0, -0.05),
                "left_knee": _e(max(0.0, -a) * 0.95, 0.0, 0.0),
                "right_knee": _e(max(0.0, -b) * 0.95, 0.0, 0.0),
                "left_ankle": _e(a * 0.15, 0.0, 0.0),
                "right_ankle": _e(b * 0.15, 0.0, 0.0),
            }
        elif self._base_state == "dancing":
            ph = t * 4.0 + self._phase
            base_pose = {
                "pelvis": _e(math.sin(ph) * 0.07, 0.0, math.sin(ph * 0.5) * 0.09),
                "spine1": _e(math.sin(ph * 0.7) * 0.06, 0.0, math.cos(ph) * 0.05),
                "left_hip": _e(math.sin(ph) * 0.14, 0.0, 0.06),
                "right_hip": _e(math.sin(ph + 1.0) * 0.14, 0.0, -0.06),
            }
        else:
            base_pose = dict(BASE_STATE_POSES.get(self._base_state) or {})
        _add_pose(bones, base_pose, 1.0)

        # Emotion body language (underlay) — scales with intensity
        emo = (ctx.emotion or "neutral").lower()
        under = EMOTION_UNDERLAY.get(emo)
        if under:
            # strong emotions read more on body
            emo_w = 0.55 + 0.55 * inten
            if emo in ("angry", "fearful", "disgusted", "surprised"):
                emo_w *= 1.15
            _add_pose(bones, under, min(1.25, emo_w))

        # Subtle speech-coupled micro-motion (never big float)
        if ctx.is_speaking and energy > 0.12:
            pulse = 0.5 + 0.5 * math.sin(t * 2.6 + self._phase)
            micro = 0.04 + 0.06 * energy
            _add_pose(bones, {
                "spine3": _e(-0.015 * pulse * energy, 0.01 * math.sin(t * 1.7), 0.0),
                "spine2": _e(-0.01 * energy, 0.0, 0.0),
                "left_shoulder": _e(0.0, 0.0, micro * pulse * 0.5),
                "right_shoulder": _e(0.0, 0.0, -micro * pulse * 0.65),
            }, 1.0)

        for name, t0, dur, wscale in self._queue:
            age = t - t0
            if age < -0.05:
                continue
            clip = ACTION_CLIPS.get(name, {})
            loop = bool(clip.get("loop"))
            if not loop and age > dur:
                continue
            if loop and age > dur:
                continue

            pose = _sample_action(name, age, dur, intensity=inten * wscale)
            if not pose:
                continue

            # Speech energy pumps talk gestures a bit
            if name in ("talk_open", "talk_emphasize") and energy > 0.1:
                boost = 1.0 + 0.25 * energy
                pose = _scale_pose(pose, boost)

            # Wave / emphasize: extra natural oscillation while near peak
            if name == "wave" and 0.35 < (age / max(dur, 1e-3)) < 0.85:
                osc = math.sin(age * 11.0) * 0.22 * inten
                _add_pose(pose, {
                    "right_wrist": _e(0.0, osc, osc * 0.4),
                    "right_elbow": _e(0.0, osc * 0.25, 0.0),
                }, 1.0)

            if name == "talk_emphasize" and ctx.is_speaking and energy > 0.25:
                # snap emphasis on loud peaks
                beat = max(0.0, energy - 0.25) * 0.8
                _add_pose(pose, {
                    "right_shoulder": _e(-0.12 * beat, 0.0, -0.1 * beat),
                    "right_elbow": _e(-0.15 * beat, 0.0, 0.0),
                    "spine3": _e(0.0, 0.0, -0.04 * beat),
                }, 1.0)

            _add_pose(bones, pose, 1.0)

        return bones

    def current_action_label(self, ctx: FaceContext) -> str:
        """Prefer MoMask Action, else MotionController catalog (Mixamo retarget)."""
        # Open-text MoMask body (SMPL-X Action already baked)
        ma = str(ctx.extras.get("momask_action") or "").strip()
        if ma:
            plan = ctx.extras.get("motion_plan") or {}
            if not plan or plan.get("engine") != "momask":
                ctx.extras["motion_plan"] = {
                    "clip_id": ma,
                    "action_name": ma,
                    "engine": "momask",
                    "loop": False,
                    "library_blend": ctx.extras.get("momask_library") or "",
                    "speed": 0.85 + 0.4 * float(ctx.intensity or 0.7),
                    "amp": 1.0,
                }
            return ma

        try:
            from .motion_controller import MotionController

            ctrl = getattr(self, "_motion_ctrl", None)
            if ctrl is None:
                self._motion_ctrl = MotionController()
                ctrl = self._motion_ctrl
            plan = ctrl.resolve(
                actions=list(ctx.extras.get("body_actions") or []),
                state=str(ctx.extras.get("body_state") or self._base_state or "standing"),
                gesture_target=str(ctx.extras.get("gesture_target") or "none"),
                hand=str(ctx.extras.get("hand") or "right"),
                emotion=str(ctx.emotion or "neutral"),
                intensity=float(ctx.intensity or self._intensity or 0.75),
                text=str(getattr(ctx, "text", "") or ""),
            )
            # stash for coordinator extras
            ctx.extras["motion_plan"] = plan.to_dict()
            return plan.action_name or plan.clip_id or "idle"
        except Exception:
            pass

        t = float(ctx.t)
        best = "idle"
        best_score = 0.0
        for name, t0, dur, _w in self._queue:
            age = t - t0
            if age < 0 or age > dur:
                continue
            # prefer one-shots mid-clip, else looping talk
            score = 1.0 - abs((age / max(dur, 1e-3)) - 0.45)
            if not ACTION_CLIPS.get(name, {}).get("loop"):
                score += 0.35
            if score > best_score:
                best_score = score
                best = name
        return best
