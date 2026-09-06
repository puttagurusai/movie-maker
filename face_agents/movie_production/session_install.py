"""
Install a baked MoviePackage into Blender Session SoT (same path as chat).

Body → Session_<id> append (place continuity) — multi-beat when shot has walk+wave+talk
Face → keyframes on head
Camera → SessionCam_* bake only (NO live follow)
Look → Look_Set + NPCs; wardrobe only if fit gate passes
End → session_bind so Space/Play shows body motion (not idle_hold Death pose)

This is the product pipeline — not a separate “cinema mode.”
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .shot_schema import MoviePackage, ShotDetail


def _resolve_shot_action(sh: ShotDetail) -> str:
    act = str(sh.body_action or "").strip()
    if act:
        return act
    acts = [str(a).lower() for a in (sh.actions or [])]
    for prefer in (
        "wave", "talk_open", "talk_emphasize", "celebrate", "shrug",
        "look_around", "run", "walk", "nod_yes", "shake_no", "idle",
    ):
        if prefer in acts:
            return prefer
    st = str(sh.state or "").lower()
    if any(x in st for x in ("walk", "loco", "run")):
        return "run"
    return "talk_open"


def _clip_fps_for(sh: ShotDetail, act: str) -> float:
    bm = str(sh.body_mode or "").lower()
    if bm in ("momask", "both") or act.startswith("momask_"):
        return 20.0
    return 30.0


def _body_beats_for_shot(sh: ShotDetail, total_dur: float) -> List[Dict[str, Any]]:
    """
    Expand one shot into sequential body Actions that fill the take.

    Story like "Walk…, wave hello, say goodbye" often bakes as one shot with
    state=walking + actions=[wave, talk_open] + body_action=walk. Installing
    only walk then hold_end freezes motion for lips/camera — multi-beat fixes that.
    """
    total = max(0.5, float(total_dur or 4.0))
    primary = str(sh.body_action or "").strip().lower()
    st = str(sh.state or "").lower()
    listed = [str(a).lower().strip() for a in (sh.actions or []) if str(a).strip()]

    order: List[str] = []
    loco = None
    if primary in ("walk", "run"):
        loco = primary
    elif any(x in st for x in ("walk", "loco")):
        loco = "walk"
    elif "run" in st:
        loco = "run"
    if loco:
        order.append(loco)

    for a in listed:
        if a and a not in order:
            order.append(a)

    # MoMask / single explicit action with nothing else → one beat
    if primary.startswith("momask_") or (
        primary and primary not in order and not listed and not loco
    ):
        order = [primary]
    elif primary and primary not in order and primary not in ("",):
        # gesture primary when state wasn't loco
        if not loco:
            order.insert(0, primary)
        elif primary not in ("walk", "run"):
            order.append(primary)

    if not order:
        order = [primary or "talk_open"]

    if len(order) == 1:
        return [{
            "action": order[0],
            "duration": total,
            "with_speech": True,
            "clip_label": sh.shot_id,
        }]

    weights = []
    for a in order:
        if a in ("walk", "run"):
            weights.append(1.35)
        elif a.startswith("talk"):
            weights.append(1.15)
        else:
            weights.append(1.0)
    tw = sum(weights) or 1.0
    beats: List[Dict[str, Any]] = []
    remaining = total
    for i, a in enumerate(order):
        if i == len(order) - 1:
            d = max(0.55, remaining)
        else:
            d = max(0.55, total * (weights[i] / tw))
            remaining -= d
        beats.append({
            "action": a,
            "duration": d,
            "with_speech": (i == len(order) - 1),
            "clip_label": f"{sh.shot_id}_{a}",
        })
    return beats


def _policy_for_action(sh: ShotDetail, act: str, eng: str) -> str:
    if eng == "momask" or act.startswith("momask_"):
        return "momask_match"
    if act in ("idle", "talk_open", "talk_emphasize", "wave", "shrug", "nod_yes", "shake_no"):
        return "loop"
    pol = str(sh.clip_policy or "").lower()
    if pol == "loop" and act in ("walk", "run"):
        # Loco: one cycle then hold (root wrap teleports). Multi-beat covers the rest.
        return "hold_end"
    return pol or "hold_end"


def install_package_into_session(
    coord: Any,
    pkg: MoviePackage,
    *,
    reset: bool = True,
    apply_look: bool = True,
    bind_for_play: bool = True,
) -> Dict[str, Any]:
    """
    Push package into the live Blender receiver as one Session film.
    Returns a small report for logging / QC.
    """
    fps = float(pkg.fps or 20.0)
    report: Dict[str, Any] = {
        "shots": 0,
        "body": [],
        "look": None,
        "ok": True,
        "errors": [],
        "bound": False,
    }

    if reset:
        coord.send_udp({
            "type": "body",
            "session_reset": True,
            "rest": True,
            "session_id": getattr(pkg, "title", "") or "movie",
        })
        time.sleep(0.4)

    # Look / NPCs / wardrobe (fit-gated inside receiver)
    look: Dict[str, Any] = {}
    if apply_look:
        for sh in pkg.shots:
            if not sh.look:
                continue
            try:
                cand = sh.look.to_dict() if hasattr(sh.look, "to_dict") else dict(sh.look or {})
            except Exception:
                cand = {}
            loc = str(cand.get("location") or "").lower()
            if loc and loc not in ("studio", "default", ""):
                look = cand
                break
            if not look and cand:
                look = cand
        if not look or str(look.get("location") or "").lower() in ("studio", "default", ""):
            look = {
                "location": "street",
                "time_of_day": "day",
                "set_preset": "exterior_street",
                "wardrobe_id": "hero_default",
                "extras_count": 3,
                "extras_preset": "sidewalk",
            }
        look.setdefault("wardrobe_id", "hero_default")
        if int(look.get("extras_count") or 0) <= 0 and str(look.get("location") or "") not in (
            "studio", "stage",
        ):
            look["extras_count"] = 3
            look["extras_preset"] = look.get("extras_preset") or "sidewalk"
        coord.send_udp({"type": "look", "op": "apply", "look": look})
        report["look"] = look
        time.sleep(0.6)

    first = True
    f_end = 1
    session_fps = float(fps) if fps >= 1.0 else 20.0

    for sh in pkg.shots:
        if not sh.bake_ok:
            report["errors"].append(f"{sh.shot_id}: bake_ok=False")
            continue

        dur = max(0.2, float(sh.duration_s or (sh.t1 - sh.t0) or 4.0))
        speech = float(sh.speech_duration_s or 0.0)
        beats = _body_beats_for_shot(sh, dur)
        print(
            f"  [session] {sh.shot_id} → {len(beats)} body beat(s): "
            + ", ".join(f"{b['action']}({b['duration']:.1f}s)" for b in beats)
        )

        for bi, beat in enumerate(beats):
            act = str(beat["action"])
            bdur = float(beat["duration"])
            clip_fps = _clip_fps_for(sh, act)
            eng = "momask" if (
                str(sh.body_mode or "").lower() in ("momask", "both")
                or act.startswith("momask_")
            ) else "catalog"
            policy = _policy_for_action(sh, act, eng)
            take_frames = max(8, int(round(bdur * session_fps)))
            with_speech = bool(beat.get("with_speech"))
            body_pkt: Dict[str, Any] = {
                "type": "body",
                "action": act,
                "duration": bdur,
                "speed": float(sh.clip_speed or 1.0),
                "loop": policy == "loop",
                "append_timeline": True,
                # Force a new Session clip even if the Action name repeats
                # (e.g. two MoMask shots sharing a cache hit) — otherwise
                # same_action_live skips append and wave/walk disappear.
                "restart": True,
                "force": True,
                # Append into Session SoT only — no wall-clock live (bind at end)
                "append_only": True,
                "live": False,
                "session_install": True,
                "action_frames": take_frames,
                "motion_length": take_frames,
                "clip_label": str(beat.get("clip_label") or sh.shot_id),
                "text": (sh.spoken or sh.text or "") if with_speech else "",
                "audio_path": str(sh.audio_path or "") if with_speech else "",
                "speech_duration_s": speech if with_speech else 0.0,
                "speech_delay_s": float(sh.hold_before_s or 0.15) if with_speech else 0.0,
                "engine": eng,
                "clip_policy": policy,
                "clip_fps": clip_fps,
                "library_blend": str(sh.body_library or ""),
            }
            if eng == "momask" and sh.body_library:
                body_pkt["library_blend"] = sh.body_library
            coord.send_udp(body_pkt)
            report["body"].append({
                "shot": sh.shot_id,
                "action": act,
                "dur": bdur,
                "beat": bi,
            })
            time.sleep(0.14)

        # FACE (once per shot, on speech window)
        if sh.face_timeline_path and Path(sh.face_timeline_path).is_file():
            f_off = max(1, int(round(float(sh.t0) * fps)) + 1)
            coord.send_udp({
                "type": "face_keyframes",
                "path": str(Path(sh.face_timeline_path).resolve()),
                "set_frame_range": True,
                "clear_previous": bool(first),
                "frame_offset": f_off - 1,
            })

        # CAMERA — bake SessionCam only (hold_after False, no live follow)
        if sh.camera and os.environ.get("USE_MOVIE_CAMERA", "1") not in ("0", "false", "no"):
            cam = dict(sh.camera or {})
            kfs = list(cam.get("keyframes") or [])
            f_off = max(1, int(round(float(sh.t0) * fps)) + 1)
            for kf in kfs:
                if kf.get("frame_abs") is None:
                    tt = float(kf.get("t") or 0.0)
                    kf["t_abs"] = float(sh.t0) + tt
                    kf["frame_abs"] = f_off + max(0, int(round(tt * fps)))
            coord.send_udp({
                "type": "camera",
                "op": "plan",
                "duration": dur,
                "fps": fps,
                "shot": cam.get("shot") or sh.camera_shot or "MS",
                "move_type": cam.get("move_type") or sh.camera_move or "static",
                "track_head": False,
                "camera_name": cam.get("camera_name") or "MovieCam_A",
                "look_at_name": cam.get("look_at_name") or "MovieCam_A_LookAt",
                "camera_role": cam.get("camera_role") or sh.camera_role or "A_cam",
                "set_scene_camera": True,
                "bake_keyframes": True,
                "clear_previous": bool(first),
                "hold_after": False,
                "subject_relative": bool(cam.get("subject_relative", True)),
                "session_frame_start": f_off,
                "live": False,
                "keyframes": kfs,
            })
            coord.send_udp({"type": "camera", "op": "rest", "hold_after": False})

        f_end = max(f_end, int(round(float(sh.t1) * fps)) + 1)
        first = False
        report["shots"] += 1
        print(f"  [session] {sh.shot_id} installed t=[{sh.t0:.1f},{sh.t1:.1f}]")

    time.sleep(0.45)
    report["frame_end"] = f_end

    # CRITICAL: bind Session_<id> on SMPL-X so Play/scrub shows body (not Death/idle)
    if bind_for_play:
        coord.send_udp({"type": "body", "op": "session_bind"})
        time.sleep(0.2)
        report["bound"] = True
        print("[session] session_bind sent — Space/Play should show body motion")

    print(f"[session] installed {report['shots']} shots → ~frames 1–{f_end}")
    return report
