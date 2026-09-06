"""
camera_agent.py — cinematic camera worker (geometry + keyframes).

Director decides size/move/role/duration. This module NEVER invents story —
it places cameras safely (no head penetration) and builds keyframe paths.

Multi-cam: A_cam, B_cam, C_cam, env_cam are created on demand (roles).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Safe distances: never closer than MIN on -Y (outside mesh head/hair).
# Old ECU at -0.55 went inside the skull.
# World-origin presets (legacy / establish). Prefer SUBJECT_SHOTS for live follow.
SHOT_PRESETS: Dict[str, Dict[str, Any]] = {
    "ECU": {"location": (0.0, -1.25, 1.62), "look_at": (0.0, 0.08, 1.60), "fov_deg": 34.0},
    "CU":  {"location": (0.0, -1.60, 1.58), "look_at": (0.0, 0.06, 1.58), "fov_deg": 38.0},
    "MCU": {"location": (0.0, -2.15, 1.52), "look_at": (0.0, 0.0, 1.52), "fov_deg": 42.0},
    "MS":  {"location": (0.0, -2.90, 1.45), "look_at": (0.0, 0.0, 1.42), "fov_deg": 45.0},
    "MLS": {"location": (0.0, -3.70, 1.32), "look_at": (0.0, 0.0, 1.25), "fov_deg": 50.0},
    "WS":  {"location": (0.0, -5.20, 1.35), "look_at": (0.0, 0.0, 1.10), "fov_deg": 55.0},
}

# FilmAgent-style subject framing: offsets relative to a body anchor, NOT world origin.
# anchor: head | chest | pelvis | full_body
# cam_offset / look_bias are added to the live subject world position each frame.
SUBJECT_SHOTS: Dict[str, Dict[str, Any]] = {
    # Close / face — anchor on HEAD (not hip)
    "ECU": {
        "anchor": "head",
        "cam_offset": (0.0, -1.20, 0.02),
        "look_bias": (0.0, 0.05, 0.0),
        "fov_deg": 34.0,
    },
    "CU": {
        "anchor": "head",
        "cam_offset": (0.0, -1.55, 0.0),
        "look_bias": (0.0, 0.04, -0.02),
        "fov_deg": 38.0,
    },
    # Upper body / talk — CHEST (FilmAgent dialogue coverage)
    "MCU": {
        "anchor": "chest",
        "cam_offset": (0.0, -2.10, 0.15),
        "look_bias": (0.0, 0.0, 0.35),
        "fov_deg": 42.0,
    },
    "MS": {
        "anchor": "chest",
        "cam_offset": (0.0, -2.85, 0.05),
        "look_bias": (0.0, 0.0, 0.15),
        "fov_deg": 45.0,
    },
    # Full body / locomotion — PELVIS + raised look so whole figure is in frame
    "MLS": {
        "anchor": "full_body",
        "cam_offset": (0.0, -3.80, 1.05),
        "look_bias": (0.0, 0.0, 0.95),
        "fov_deg": 50.0,
    },
    "WS": {
        "anchor": "full_body",
        "cam_offset": (0.0, -5.40, 1.25),
        "look_bias": (0.0, 0.0, 0.90),
        "fov_deg": 55.0,
    },
}

MIN_CAM_LOOK_DIST = float(os.environ.get("MOVIE_CAM_MIN_DIST", "1.08"))
# Most negative Y allowed toward character along -Y axis (magnitude)
MIN_CAM_DIST_Y = float(os.environ.get("MOVIE_CAM_MIN_Y", "1.15"))
# Soft follow lag (0=hard lock, 1=no move). Lower = snappier.
CAM_FOLLOW_LAG = float(os.environ.get("MOVIE_CAM_FOLLOW_LAG", "0.18"))

# Role → default lateral / height bias (multi-cam coverage) — FilmAgent-like A/B/C
ROLE_OFFSETS: Dict[str, Tuple[float, float, float]] = {
    "A_cam": (0.0, 0.0, 0.0),          # front
    "B_cam": (1.15, 0.35, 0.05),       # 3/4 side
    "C_cam": (-0.35, -0.15, 0.08),     # slightly opposite + higher detail
    "env_cam": (0.55, 0.8, 0.25),      # establish wider / higher
    "MovieCam": (0.0, 0.0, 0.0),       # legacy alias of A
}

ROLE_TO_NAME = {
    "A_cam": "MovieCam_A",
    "B_cam": "MovieCam_B",
    "C_cam": "MovieCam_C",
    "env_cam": "MovieCam_Env",
    "MovieCam": "MovieCam_A",
}


def movie_camera_enabled() -> bool:
    return os.environ.get("USE_MOVIE_CAMERA", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def camera_name_for_role(role: str) -> str:
    r = (role or "A_cam").strip()
    return ROLE_TO_NAME.get(r, ROLE_TO_NAME.get(r.lower(), f"MovieCam_{r}"))


def look_name_for_role(role: str) -> str:
    return camera_name_for_role(role) + "_LookAt"


@dataclass
class CameraKeyframe:
    t: float
    frame: int
    location: Tuple[float, float, float]
    look_at: Tuple[float, float, float]
    fov_deg: float = 45.0
    interpolation: str = "BEZIER"
    move_type: str = "static"
    shot: str = "MS"
    # When subject_relative: location/look_at are OFFSETS from subject anchor
    rel_cam: Optional[Tuple[float, float, float]] = None
    rel_look: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["location"] = list(self.location)
        d["look_at"] = list(self.look_at)
        if self.rel_cam is not None:
            d["rel_cam"] = list(self.rel_cam)
        if self.rel_look is not None:
            d["rel_look"] = list(self.rel_look)
        return d


@dataclass
class CameraPlan:
    duration_s: float
    fps: float
    shot: str
    move_type: str
    keyframes: List[CameraKeyframe] = field(default_factory=list)
    track_head: bool = True
    notes: List[str] = field(default_factory=list)
    camera_role: str = "A_cam"
    camera_name: str = "MovieCam_A"
    look_at_name: str = "MovieCam_A_LookAt"
    pace: str = "medium"
    # FilmAgent-style subject framing (not always hip)
    subject_relative: bool = True
    subject_anchor: str = "chest"  # head | chest | pelvis | full_body
    follow_lag: float = 0.38  # higher = less flicker between updates
    hold_after: bool = True  # keep framing on subject after take ends

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "fps": self.fps,
            "shot": self.shot,
            "move_type": self.move_type,
            "track_head": self.track_head,
            "notes": list(self.notes),
            "camera_role": self.camera_role,
            "camera_name": self.camera_name,
            "look_at_name": self.look_at_name,
            "pace": self.pace,
            "subject_relative": self.subject_relative,
            "subject_anchor": self.subject_anchor,
            "follow_lag": self.follow_lag,
            "hold_after": self.hold_after,
            "min_cam_dist": MIN_CAM_LOOK_DIST,
            "keyframes": [k.to_dict() for k in self.keyframes],
        }

    def udp_packet(self) -> Dict[str, Any]:
        return {
            "type": "camera",
            "op": "plan",
            "duration": float(self.duration_s),
            "fps": float(self.fps if self.fps and 8.0 <= float(self.fps) <= 22.0 else 20.0),
            "shot": self.shot,
            "move_type": self.move_type,
            "track_head": bool(self.track_head),
            "camera_name": self.camera_name,
            "look_at_name": self.look_at_name,
            "camera_role": self.camera_role,
            "set_scene_camera": True,
            # Bake onto the session timeline for Space/scrub replay.
            # Live play still detaches the Action so it does not fight wall-clock.
            "bake_keyframes": True,
            # Append camera keys on session timeline (do not wipe prior turns)
            "clear_previous": False,  # never wipe prior cam keys mid-session
            "min_cam_dist": MIN_CAM_LOOK_DIST,
            "subject_relative": bool(self.subject_relative),
            "subject_anchor": str(self.subject_anchor or "chest"),
            "follow_lag": float(max(0.45, self.follow_lag)),
            "hold_after": bool(self.hold_after),
            "force_camera_role": True,  # each role keeps its own SessionCam_* history
            "keyframes": [k.to_dict() for k in self.keyframes],
        }


def subject_preset(shot: str) -> Dict[str, Any]:
    return SUBJECT_SHOTS.get((shot or "MS").upper(), SUBJECT_SHOTS["MS"])


def anchor_for_shot(shot: str) -> str:
    """FilmAgent-like: close-ups track head; talk chest; loco full body — not always hip."""
    return str(subject_preset(shot).get("anchor") or "chest")


def _vec3(v: Sequence[float]) -> Tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _lerp3(a, b, u: float):
    u = max(0.0, min(1.0, float(u)))
    return (
        a[0] + (b[0] - a[0]) * u,
        a[1] + (b[1] - a[1]) * u,
        a[2] + (b[2] - a[2]) * u,
    )


def _dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _preset(shot: str) -> Dict[str, Any]:
    return SHOT_PRESETS.get((shot or "MS").upper(), SHOT_PRESETS["MS"])


def _apply_role_offset(
    loc: Tuple[float, float, float],
    look: Tuple[float, float, float],
    role: str,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    ox, oy, oz = ROLE_OFFSETS.get(role, ROLE_OFFSETS["A_cam"])
    # lateral X, push slightly back on Y when side cam, height Z
    loc2 = (loc[0] + ox, loc[1] - abs(oy) * 0.15, loc[2] + oz)
    look2 = (look[0] + ox * 0.15, look[1], look[2] + oz * 0.3)
    return loc2, look2


def clamp_camera_pair(
    loc: Tuple[float, float, float],
    look: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Push camera out along view vector if too close to look_at (head safety)."""
    loc = _vec3(loc)
    look = _vec3(look)
    d = _dist(loc, look)
    if d < 1e-6:
        # force back on -Y
        loc = (look[0], look[1] - MIN_CAM_DIST_Y, look[2])
        return loc, look
    if d < MIN_CAM_LOOK_DIST:
        # move camera away from look along (loc-look) direction
        scale = MIN_CAM_LOOK_DIST / d
        loc = (
            look[0] + (loc[0] - look[0]) * scale,
            look[1] + (loc[1] - look[1]) * scale,
            look[2] + (loc[2] - look[2]) * scale,
        )
    # Also enforce Y distance magnitude from character origin-ish look
    # if camera is in front on -Y axis mainly
    if loc[1] > -MIN_CAM_DIST_Y and abs(loc[0]) < 1.5:
        loc = (loc[0], -MIN_CAM_DIST_Y, loc[2])
    return _vec3(loc), _vec3(look)


def select_shot_and_move(
    *,
    emotion: str = "neutral",
    intensity: float = 0.7,
    state: str = "standing",
    actions: Optional[Sequence[str]] = None,
    duration_s: float = 3.0,
    humanml_prompt: str = "",
    session_clip_index: int = 0,
    location_changed: bool = False,
) -> Tuple[str, str, List[str]]:
    """
    FilmAgent-inspired DP rules (code menu, not always hip follow).

    Any generated / traveling / grounded body is framed full-figure.
    Dialogue and emotion use MS/MCU/CU. No per-action verb lists.
    """
    from .director_schema import FULL_BODY_STATES, is_full_body_state

    emo = (emotion or "neutral").lower().strip()
    st = (state or "standing").lower().strip()
    inten = float(intensity or 0.7)
    acts = {str(a).lower() for a in (actions or [])}
    prompt = (humanml_prompt or "").strip()
    notes: List[str] = []
    idx = max(0, int(session_clip_index))

    full_body = bool(prompt) or is_full_body_state(st) or st in FULL_BODY_STATES
    seated = st in ("sitting", "grounded")
    if idx == 0 or location_changed:
        notes.append("establish clip0/new-loc→WS env")
        return "WS", "static" if duration_s < 3.5 else "reveal", notes
    if full_body:
        notes.append("full-body motion→WS/MLS follow")
        if seated and len(prompt) < 8:
            return "MLS", "follow" if duration_s >= 3.5 else "truck", notes
        if duration_s >= 4:
            return "WS", "follow", notes
        return "MLS", "follow" if duration_s >= 3 else "truck", notes
    if seated:
        notes.append("seated/grounded→MLS keep figure in frame")
        return "MLS", "truck", notes

    if "wave" in acts and not prompt:
        notes.append("wave→MS static (chest)")
        return "MS", "static", notes

    if inten >= 0.85 and emo in ("angry", "surprised", "fearful"):
        notes.append("high intensity→CU head (not hip)")
        return "CU", "dolly_in" if duration_s < 7 else "orbit", notes

    if emo in ("sad", "apologetic", "concerned"):
        notes.append(f"{emo}→MCU dolly_out")
        return "MCU", "dolly_out", notes

    if emo in ("thinking", "sarcastic"):
        notes.append(f"{emo}→MCU orbit")
        return "MCU", "orbit", notes

    if emo in ("happy", "encouraging") and duration_s >= 2.5:
        # Alternate arc / dolly for variety across session turns
        if idx % 2 == 0:
            notes.append(f"{emo}→MS arc")
            return "MS", "arc" if duration_s >= 5 else "dolly_in", notes
        notes.append(f"{emo}→MS dolly_in")
        return "MS", "dolly_in", notes

    if duration_s >= 6:
        notes.append("long dialogue→MS slow dolly_in")
        return "MS", "dolly_in", notes

    # Default dialogue: alternate MS static vs slight orbit for multi-turn variety
    if idx % 3 == 1:
        notes.append("dialogue→MCU static (chest)")
        return "MCU", "static", notes
    if idx % 3 == 2:
        notes.append("dialogue→MS soft orbit")
        return "MS", "orbit", notes
    notes.append("default→MS static (chest)")
    return "MS", "static", notes


def _shot_pair_for_move(shot: str, move: str) -> Tuple[str, str]:
    order = ["ECU", "CU", "MCU", "MS", "MLS", "WS"]
    if shot not in order:
        shot = "MS"
    i = order.index(shot)
    if move in ("dolly_in", "reveal"):
        # reveal: start wider end on shot
        if move == "reveal":
            start = order[min(len(order) - 1, i + 1)]
            return start, shot
        end = order[max(0, i - 1)]
        # never end at ECU from aggressive push unless already ECU
        if end == "ECU" and shot != "ECU":
            end = "CU"
        return shot, end
    if move == "dolly_out":
        end = order[min(len(order) - 1, i + 1)]
        return shot, end
    return shot, shot


def build_keyframes(
    *,
    duration_s: float,
    fps: float,
    shot: str,
    move_type: str,
    camera_role: str = "A_cam",
    pace: str = "medium",
    subject_relative: bool = True,
) -> Tuple[List[CameraKeyframe], str]:
    """
    Build 2–4 keyframes for the take.

    When subject_relative=True (default), location/look are **offsets from the
    live subject anchor** (head/chest/full_body) — not world-origin locks.
    Blender adds the live subject each frame so walks/runs stay framed.
    """
    dur = max(0.35, float(duration_s))
    fps = max(1.0, float(fps))
    move = (move_type or "static").lower()
    role = camera_role or "A_cam"
    pace = (pace or "medium").lower()
    amp = {"slow": 0.65, "medium": 1.0, "fast": 1.25}.get(pace, 1.0)

    start_shot, end_shot = _shot_pair_for_move(shot, move)
    anchor = anchor_for_shot(shot)

    if subject_relative:
        sa = subject_preset(start_shot)
        sb = subject_preset(end_shot)
        loc_a = _vec3(sa["cam_offset"])
        look_a = _vec3(sa["look_bias"])
        loc_b = _vec3(sb["cam_offset"])
        look_b = _vec3(sb["look_bias"])
        fov_a = float(sa["fov_deg"])
        fov_b = float(sb["fov_deg"])
        # Role offsets on relative frame
        ox, oy, oz = ROLE_OFFSETS.get(role, ROLE_OFFSETS["A_cam"])
        loc_a = (loc_a[0] + ox, loc_a[1] - abs(oy) * 0.15, loc_a[2] + oz)
        loc_b = (loc_b[0] + ox, loc_b[1] - abs(oy) * 0.15, loc_b[2] + oz)
        look_a = (look_a[0] + ox * 0.1, look_a[1], look_a[2] + oz * 0.2)
        look_b = (look_b[0] + ox * 0.1, look_b[1], look_b[2] + oz * 0.2)
    else:
        a = _preset(start_shot)
        b = _preset(end_shot)
        loc_a, look_a = _apply_role_offset(_vec3(a["location"]), _vec3(a["look_at"]), role)
        loc_b, look_b = _apply_role_offset(_vec3(b["location"]), _vec3(b["look_at"]), role)
        fov_a = float(a["fov_deg"])
        fov_b = float(b["fov_deg"])

    if move == "static":
        loc_b, look_b, fov_b = loc_a, look_a, fov_a

    elif move in ("truck", "truck_right"):
        t = 0.55 * amp
        loc_a = (loc_a[0] - t, loc_a[1], loc_a[2])
        loc_b = (loc_b[0] + t, loc_b[1], loc_b[2])

    elif move == "truck_left":
        t = 0.55 * amp
        loc_a = (loc_a[0] + t, loc_a[1], loc_a[2])
        loc_b = (loc_b[0] - t, loc_b[1], loc_b[2])

    elif move == "crane_up":
        loc_b = (loc_b[0], loc_b[1], loc_b[2] + 0.42 * amp)
        look_b = (look_b[0], look_b[1], look_b[2] + 0.08)

    elif move == "crane_down":
        loc_b = (loc_b[0], loc_b[1], loc_b[2] - 0.35 * amp)
        look_b = (look_b[0], look_b[1], look_b[2] - 0.05)

    elif move == "pan_left":
        loc_a = (loc_a[0] + 0.35 * amp, loc_a[1], loc_a[2])
        loc_b = (loc_b[0] - 0.35 * amp, loc_b[1], loc_b[2])
        look_a = (look_a[0] + 0.12, look_a[1], look_a[2])
        look_b = (look_b[0] - 0.12, look_b[1], look_b[2])

    elif move == "pan_right":
        loc_a = (loc_a[0] - 0.35 * amp, loc_a[1], loc_a[2])
        loc_b = (loc_b[0] + 0.35 * amp, loc_b[1], loc_b[2])
        look_a = (look_a[0] - 0.12, look_a[1], look_a[2])
        look_b = (look_b[0] + 0.12, look_b[1], look_b[2])

    elif move in ("orbit", "arc"):
        sign = 1.0 if move == "orbit" else -1.0
        r = 0.75 * amp
        loc_a = (loc_a[0] - sign * r, loc_a[1], loc_a[2])
        loc_b = (loc_b[0] + sign * r, loc_b[1] - 0.12 * amp, loc_b[2] + 0.05)

    elif move == "tilt_up":
        look_b = (look_b[0], look_b[1], look_b[2] + 0.22 * amp)
        loc_b = (loc_b[0], loc_b[1], loc_b[2] + 0.08)

    elif move == "tilt_down":
        look_b = (look_b[0], look_b[1], look_b[2] - 0.18 * amp)
        loc_b = (loc_b[0], loc_b[1], loc_b[2] - 0.05)

    elif move == "follow":
        # Relative side-follow path (subject still tracked live in Blender)
        loc_a = (loc_a[0] - 0.35 * amp, loc_a[1], loc_a[2])
        loc_b = (loc_b[0] + 0.45 * amp, loc_b[1] + 0.15 * amp, loc_b[2])
        look_b = (look_b[0] + 0.05, look_b[1], look_b[2])

    elif move == "handheld":
        pass

    elif move == "dolly_in":
        if subject_relative:
            # Pull closer on -Y (less negative = closer)
            loc_b = (loc_b[0], min(loc_b[1] + 0.45 * amp, -MIN_CAM_DIST_Y), loc_b[2])
        else:
            loc_b = (loc_b[0], min(loc_b[1], -MIN_CAM_DIST_Y), loc_b[2])

    elif move == "dolly_out":
        if subject_relative:
            loc_b = (loc_b[0], loc_b[1] - 0.55 * amp, loc_b[2])

    # World-space only: clamp absolute pairs. Relative offsets clamp in Blender after +subject.
    if not subject_relative:
        loc_a, look_a = clamp_camera_pair(loc_a, look_a)
        loc_b, look_b = clamp_camera_pair(loc_b, look_b)

    def kf(t: float, loc, look, fov, mtype: str, sh: str) -> CameraKeyframe:
        fr = max(1, int(round(t * fps)) + 1)
        loc_t, look_t = _vec3(loc), _vec3(look)
        if not subject_relative:
            loc_t, look_t = clamp_camera_pair(loc_t, look_t)
        return CameraKeyframe(
            t=float(t),
            frame=fr,
            location=loc_t,
            look_at=look_t,
            fov_deg=float(fov),
            interpolation="BEZIER",
            move_type=mtype,
            shot=sh,
            rel_cam=loc_t if subject_relative else None,
            rel_look=look_t if subject_relative else None,
        )

    keys = [
        kf(0.0, loc_a, look_a, fov_a, move, start_shot),
        kf(dur, loc_b, look_b, fov_b, move, end_shot),
    ]

    # Mid accents for longer cinematic takes
    if move in ("dolly_in", "dolly_out", "truck", "truck_left", "truck_right",
                "follow", "orbit", "arc", "reveal", "crane_up", "crane_down") and dur >= 2.5:
        mid_t = dur * (0.45 if pace == "slow" else 0.55)
        u = mid_t / dur
        keys.insert(
            1,
            kf(
                mid_t,
                _lerp3(loc_a, loc_b, u),
                _lerp3(look_a, look_b, u),
                fov_a + (fov_b - fov_a) * u,
                move,
                start_shot if u < 0.5 else end_shot,
            ),
        )

    if move == "handheld" and dur >= 1.5:
        # 3 micro offsets
        for frac, jx, jz in ((0.25, 0.04, 0.02), (0.55, -0.03, -0.015), (0.8, 0.025, 0.01)):
            t = dur * frac
            u = frac
            base = _lerp3(loc_a, loc_b, u)
            look = _lerp3(look_a, look_b, u)
            loc = (base[0] + jx * amp, base[1], base[2] + jz * amp)
            keys.insert(-1, kf(t, loc, look, fov_a + (fov_b - fov_a) * u, move, shot))

    keys.sort(key=lambda k: k.t)
    return keys, anchor


def plan_camera_for_beat(
    *,
    duration_s: float,
    emotion: str = "neutral",
    intensity: float = 0.7,
    state: str = "standing",
    actions: Optional[Sequence[str]] = None,
    fps: float = 30.0,
    shot: Optional[str] = None,
    move_type: Optional[str] = None,
    track_head: bool = True,
    camera_role: str = "A_cam",
    pace: str = "medium",
    humanml_prompt: str = "",
    session_clip_index: int = 0,
    subject_relative: bool = True,
    subject_anchor: Optional[str] = None,
    last_shot: str = "",
    last_role: str = "",
    location_changed: bool = False,
) -> CameraPlan:
    notes: List[str] = []
    from .director_schema import is_full_body_state
    idx = max(0, int(session_clip_index))
    full_body = bool((humanml_prompt or "").strip()) or is_full_body_state(state)
    role = (camera_role or "").strip()
    if (idx == 0 or location_changed) and not role:
        role = "env_cam"
        notes.append("establish coverage→env_cam")
    elif not role:
        if full_body:
            role = ("env_cam", "A_cam", "B_cam")[idx % 3]
        else:
            role = ("A_cam", "B_cam", "C_cam")[idx % 3]
        notes.append(f"coverage→{role}")
    elif role == "A_cam":
        if idx > 0 and idx % 4 == 3:
            role = "B_cam"
            notes.append("session variety→B_cam")
        elif idx > 0 and idx % 5 == 4:
            role = "env_cam"
            notes.append("session variety→env_cam")

    if shot and move_type:
        sh, mv = shot.upper(), move_type.lower()
        notes.append("director shot/move")
    else:
        sh, mv, notes2 = select_shot_and_move(
            emotion=emotion,
            intensity=intensity,
            state=state,
            actions=actions,
            duration_s=duration_s,
            humanml_prompt=humanml_prompt,
            session_clip_index=session_clip_index,
            location_changed=location_changed,
        )
        notes.extend(notes2)
        if shot:
            sh = shot.upper()
        if move_type:
            mv = move_type.lower()

    # Avoid identical shot thrash across turns when possible
    if last_shot and sh == last_shot.upper() and not shot:
        alt = {"MS": "MCU", "MCU": "MS", "WS": "MLS", "MLS": "WS", "CU": "MCU"}.get(sh)
        if alt:
            notes.append(f"continuity variety {sh}→{alt}")
            sh = alt

    # Hard safety: demote ECU aggressive moves
    if sh == "ECU" and mv in ("dolly_in", "follow", "handheld"):
        notes.append("safety: ECU aggressive→CU static")
        sh, mv = "CU", "static"

    # Head soft-track only on wider shots; CU/ECU already anchor=head
    if sh in ("ECU", "CU"):
        track_head = False
    elif duration_s >= 8 and track_head:
        notes.append("long take soft subject follow")

    subj = bool(subject_relative)
    keys, auto_anchor = build_keyframes(
        duration_s=duration_s,
        fps=fps,
        shot=sh,
        move_type=mv,
        camera_role=role,
        pace=pace,
        subject_relative=subj,
    )
    anchor = (subject_anchor or auto_anchor or anchor_for_shot(sh)).lower()
    cam_name = camera_name_for_role(role)
    look_name = look_name_for_role(role)
    notes.append(f"role={role} cam={cam_name} anchor={anchor} subj_rel={subj}")

    # Locomotion: slightly more lag so camera doesn't whip
    lag = CAM_FOLLOW_LAG
    if mv == "follow" or sh in ("WS", "MLS"):
        lag = max(lag, 0.22)
    if sh in ("CU", "ECU"):
        lag = min(lag, 0.12)

    return CameraPlan(
        duration_s=float(duration_s),
        fps=float(fps),
        shot=sh,
        move_type=mv,
        keyframes=keys,
        track_head=track_head,
        notes=notes,
        camera_role=role,
        camera_name=cam_name,
        look_at_name=look_name,
        pace=pace,
        subject_relative=subj,
        subject_anchor=anchor,
        follow_lag=float(lag),
        # False: SessionCam history must evaluate on Play/export (not freeze last frame).
        hold_after=False,
    )


class CameraAgent:
    name = "camera"

    def plan(
        self,
        *,
        duration_s: float,
        emotion: str = "neutral",
        intensity: float = 0.7,
        state: str = "standing",
        actions: Optional[Sequence[str]] = None,
        fps: float = 30.0,
        **kwargs: Any,
    ) -> CameraPlan:
        return plan_camera_for_beat(
            duration_s=duration_s,
            emotion=emotion,
            intensity=intensity,
            state=state,
            actions=actions,
            fps=fps,
            **kwargs,
        )
