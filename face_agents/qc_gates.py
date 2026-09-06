"""
CPU QC gates for movie-level clips (no bpy).

Blender-only gates (floor, heading, frustum) live in blender_receiver.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .look_schema import LOOK_GRADES, LOOK_LOCATIONS, LOOK_MOODS, LOOK_TIMES, LookPlan
from .speech_safe import range_overlaps_speech_interior, speech_blocks

MASTER_FPS = 20.0

# Look-only tokens that must not leak into HumanML (motion captions).
_LOOK_POLLUTION = re.compile(
    r"(?i)\b(golden\s*hour|film[_\s-]?contrast|bleach|hdri|studio_cyc|"
    r"three[-\s]?point|noir\s+light(?:ing)?)\b"
)
_HUMANML_LEAD = re.compile(
    r"^(a\s+person|someone|a\s+man|a\s+woman|a\s+figure|"
    r"the\s+person|the\s+man|the\s+woman)\b",
    re.I,
)

ARMATURE_NAME = "SMPL-X_Armature"
BODYMESH_NAME = "BodyMesh"


def adelay_ms(speech_frame_start: int, master_fps: float = MASTER_FPS) -> int:
    """Milliseconds for ffmpeg adelay from a 1-based speech_frame_start."""
    s0 = int(speech_frame_start or 0)
    if s0 <= 0:
        return 0
    fps = max(1.0, float(master_fps))
    return int(round((s0 - 1) / fps * 1000.0))


def _gate(gid: str, ok: bool, msg: str, errors: List[str], warnings: List[str], *, warn: bool = False) -> None:
    if ok:
        return
    line = f"{gid}: {msg}"
    (warnings if warn else errors).append(line)


def qc_hml(humanml_prompt: str, *, engine: str = "") -> Optional[str]:
    """Fail if momask caption is not a HumanML third-person line."""
    cap = (humanml_prompt or "").strip()
    if str(engine or "").lower() not in ("momask", "both") and not cap:
        return None
    if not cap:
        return "momask clip missing humanml_prompt"
    if not _HUMANML_LEAD.match(cap):
        return f"caption is not HumanML (need 'a person …'): {cap[:80]!r}"
    return None


def qc_look(look: Any, humanml_prompt: str = "") -> List[str]:
    errs: List[str] = []
    plan = look if isinstance(look, LookPlan) else LookPlan.from_dict(look if isinstance(look, dict) else None)
    if plan.location not in LOOK_LOCATIONS and plan.location != "studio":
        errs.append(f"look.location not in enums: {plan.location!r}")
    if plan.time_of_day not in LOOK_TIMES:
        errs.append(f"look.time_of_day not in enums: {plan.time_of_day!r}")
    if plan.light_mood not in LOOK_MOODS:
        errs.append(f"look.light_mood not in enums: {plan.light_mood!r}")
    if plan.grade not in LOOK_GRADES:
        errs.append(f"look.grade not in enums: {plan.grade!r}")
    if _LOOK_POLLUTION.search(humanml_prompt or ""):
        errs.append(f"look tokens leaked into humanml_prompt: {(humanml_prompt or '')[:80]!r}")
    return errs


def qc_sil(clip: Dict[str, Any]) -> Optional[str]:
    """Motion-only: speech_frame_start must be 0 (no Parler / no S#)."""
    text = str(clip.get("text") or "").strip()
    spoken = str(clip.get("spoken") or "").strip()
    s0 = int(clip.get("speech_frame_start") or 0)
    sdur = float(clip.get("speech_duration_s") or 0.0)
    motion_only = bool(clip.get("motion_only")) or (not text and not spoken and sdur <= 0.04)
    if not motion_only:
        return None
    if s0 > 0:
        return f"motion-only clip has speech_frame_start={s0} (want 0)"
    return None


def qc_cut(clips: Sequence[Dict[str, Any]], edit_a: int, edit_b: int) -> Optional[str]:
    hit = range_overlaps_speech_interior(clips, edit_a, edit_b)
    if not hit:
        return None
    c, s0, s1 = hit
    return f"edit [{edit_a}-{edit_b}] cuts interior of S#{c.get('index')} [{s0}-{s1}]"


def qc_id(armature: str = "", bodymesh: str = "") -> Optional[str]:
    arm = (armature or ARMATURE_NAME).strip()
    mesh = (bodymesh or BODYMESH_NAME).strip()
    if arm != ARMATURE_NAME:
        return f"armature {arm!r} != {ARMATURE_NAME}"
    if mesh and mesh != BODYMESH_NAME:
        return f"mesh {mesh!r} != {BODYMESH_NAME}"
    return None


def qc_aud(clips: Sequence[Dict[str, Any]], *, master_fps: float = MASTER_FPS) -> List[str]:
    errs: List[str] = []
    for c in clips or []:
        s0 = int((c or {}).get("speech_frame_start") or 0)
        if s0 <= 0:
            continue
        got = (c or {}).get("adelay_ms")
        want = adelay_ms(s0, master_fps)
        if got is not None and int(got) != want:
            errs.append(
                f"clip#{(c or {}).get('index')} adelay_ms={got} want {want} "
                f"(speech_frame_start={s0} @ {master_fps:g}fps)"
            )
    return errs


def qc_lip(clip: Dict[str, Any], *, wav_tol_s: float = 0.05) -> Optional[str]:
    """
    v1: if a face timeline path is recorded, the file must exist and WAV
    duration must match speech_duration ±50 ms (hold-padded WAVs compared
    against speech_duration_s). Missing timeline on chat clips is a warning
    (see run_cpu_gates), not a hard error — Peak-lag vs RMS is deferred.
    """
    s0 = int(clip.get("speech_frame_start") or 0)
    s1 = int(clip.get("speech_frame_end") or 0)
    if s1 < s0 or s0 <= 0:
        return None
    face = str(clip.get("face_timeline_path") or "").strip()
    wav = str(clip.get("audio_path") or "").strip()
    if face and not Path(face).is_file():
        return f"clip#{clip.get('index')} face timeline missing on disk: {face}"
    if wav and not Path(wav).is_file():
        return f"clip#{clip.get('index')} missing WAV {wav}"
    if not wav:
        return None
    try:
        import soundfile as sf
        info = sf.info(wav)
        wav_s = float(info.duration)
    except Exception:
        wav_s = float(clip.get("speech_duration_s") or 0.0)
    sdur = float(clip.get("speech_duration_s") or 0.0)
    if sdur > 0.04 and wav_s > 0.04 and abs(wav_s - sdur) > wav_tol_s + 0.35:
        # Take WAV may include hold_before/after; allow that pad.
        span_s = (s1 - s0 + 1) / MASTER_FPS
        if abs(sdur - span_s) > wav_tol_s + 0.15:
            return (
                f"clip#{clip.get('index')} WAV {wav_s:.3f}s vs speech {sdur:.3f}s "
                f"span {span_s:.3f}s (tol {wav_tol_s}s)"
            )
    return None


def run_cpu_gates(
    *,
    clips: Sequence[Dict[str, Any]] | None = None,
    shots: Sequence[Any] | None = None,
    armature: str = ARMATURE_NAME,
    bodymesh: str = BODYMESH_NAME,
    edit_range: Optional[tuple] = None,
) -> Dict[str, Any]:
    """
    Returns {ok, errors, warnings, where}.
    Movie export refuses on errors. Preview may warn only.
    """
    errors: List[str] = []
    warnings: List[str] = []
    clips = list(clips or [])

    _gate("QC-ID", qc_id(armature, bodymesh) is None, qc_id(armature, bodymesh) or "", errors, warnings)

    rows = clips
    if shots:
        for sh in shots:
            d = sh.to_dict() if hasattr(sh, "to_dict") else dict(sh)
            rows.append(d)

    for c in rows:
        eng = str(c.get("engine") or c.get("body_mode") or "")
        hml = str(c.get("humanml_prompt") or c.get("prompt") or "")
        msg = qc_hml(hml, engine=eng)
        if msg and eng.lower() in ("momask", "both"):
            errors.append(f"QC-HML clip#{c.get('index') or c.get('shot_id')}: {msg}")
        look = c.get("look")
        for e in qc_look(look, hml):
            errors.append(f"QC-LOOK clip#{c.get('index') or c.get('shot_id')}: {e}")
        sil = qc_sil(c)
        if sil:
            errors.append(f"QC-SIL clip#{c.get('index')}: {sil}")
        lip = qc_lip(c)
        if lip:
            errors.append(f"QC-LIP {lip}")
        s0 = int(c.get("speech_frame_start") or 0)
        if s0 > 0 and not str(c.get("face_timeline_path") or "").strip():
            warnings.append(
                f"QC-LIP clip#{c.get('index')}: dialogue has no face timeline path (scrub lips may be missing)"
            )

    for e in qc_aud(clips):
        errors.append(f"QC-AUD {e}")

    if edit_range is not None:
        a, b = int(edit_range[0]), int(edit_range[1])
        cut = qc_cut(clips, a, b)
        if cut:
            errors.append(f"QC-CUT {cut}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "where": "cpu",
        "n_clips": len(clips),
        "speech_blocks": speech_blocks(clips),
    }
