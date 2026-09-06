"""
Movie Director — intelligent Stage 1–2 planner.

Design for memory-less LLMs:
  - ONE primary call plans the whole sequence (preferred)
  - ContinuityBoard is fully injected every call (external memory)
  - Director decides: take length, camera role/size/move, body engine intent, pace
  - Workers (camera_agent, motion_router, baker) execute geometry & bake

Rules fallback when LLM unavailable or parse fails.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .continuity_memory import (
    CAMERA_MOVES,
    CAMERA_ROLES,
    CAMERA_SIZES,
    CLIP_POLICIES,
    ContinuityBoard,
    ShotMemory,
)
from ..look_schema import LookPlan
from .shot_schema import ShotDetail
from .stage_speech import clean_spoken_for_tts, split_stage_and_spoken
from .world_state import WorldState
from ..motion_router import route_motion


DIRECTOR_SYSTEM = """You are the FILM DIRECTOR for a single-character 3D avatar movie pipeline.

You plan CINEMATIC takes. For ~120s prefer 6–10 medium takes (not 20 tiny cuts).
Each take may contain 1–3 spoken sentences. Mix CATALOG gestures with MOMASK open motion.

══════════════════════════════════════
BODY ENGINES (pick per shot)
══════════════════════════════════════
body_mode=catalog  — known clips only:
  wave, shrug, talk_open, talk_emphasize, celebrate, nod_yes, shake_no,
  idle / walk loop, simple stand/sit talk. Use for greetings and talk beats.
body_mode=momask   — open full-body from HumanML3D-style captions (MoMask t2m).
  Use for: walk/run with path, jump, sit-down/stand-up, multi-step actions,
  dance, kicks, pick up, stumble, novel compound motion.
clip_policy: loop | hold_end | stretch | momask_match
  (use momask_match when body_mode=momask)

══════════════════════════════════════
humanml_prompt (REQUIRED when body_mode=momask)
══════════════════════════════════════
Write like official MoMask / HumanML3D training text — NOT chat, NOT dialogue:

  GOOD (MoMask style):
    "a person walks forward at a steady pace"
    "a person jumps up and then lands"
    "a person sits down carefully then stands back up"
    "a person raises the left hand upwards then moves both hands aside"
    "a person bends down and picks something up with the right hand"
    "a man walks forward, turns around, and walks back"
    "a person waves with the right hand while standing still"

  BAD (never use as humanml_prompt):
    "hi" / "hello everyone" / "say welcome" / "walk" / "jump" / ["wave"]
    spoken dialogue alone / stage directions mixed into speech

Rules for humanml_prompt:
  1. Third person: start with "a person" or "a man" / "the person"
  2. Concrete body verbs (walks, jumps, sits, raises, bends, turns…)
  3. One continuous motion description; 8–30 words typical
  4. NO quotation of dialogue; speech goes only in "spoken"
  5. Prefer left/right when hands matter

══════════════════════════════════════
stage vs spoken
══════════════════════════════════════
  stage  = physical action only (Walk forward; raise left hand)
  spoken = words TTS will say only ("Welcome everyone.")
  NEVER put walk/wave/jump instructions into spoken.

══════════════════════════════════════
Also decide
══════════════════════════════════════
  target_duration_s, hold_before_s / hold_after_s
  camera_role, camera_shot, camera_move, pace, clip_speed
  emotion, intensity, state, actions, transition_in
  For ~2 min films: mix several catalog talk/wave shots with 2–4 momask
  locomotion or special actions so both engines appear.

Do NOT output 3D coordinates. NEVER ECU + aggressive dolly_in into the face.

Output ONLY valid JSON:
{
  "style": "cinematic",
  "target_total_s": 120,
  "director_notes": "short note",
  "shots": [
    {
      "shot_id": "shot_01",
      "text": "Walk forward while saying welcome everyone.",
      "stage": "Walk forward at a steady pace",
      "spoken": "Welcome everyone.",
      "summary": "open walk + greeting",
      "emotion": "happy",
      "intensity": 0.8,
      "state": "walking",
      "actions": ["walk", "talk_open"],
      "body_mode": "momask",
      "humanml_prompt": "a person walks forward at a steady pace swinging the arms naturally",
      "camera_role": "env_cam",
      "camera_shot": "WS",
      "camera_move": "follow",
      "transition_in": "cut",
      "target_duration_s": 10.0,
      "hold_before_s": 0.3,
      "hold_after_s": 0.5,
      "pace": "medium",
      "clip_speed": 1.0,
      "clip_policy": "momask_match",
      "director_notes": "open on locomotion"
    },
    {
      "shot_id": "shot_02",
      "text": "Wave and say hello friend.",
      "stage": "Wave with the right hand",
      "spoken": "Hello friend!",
      "summary": "catalog wave",
      "emotion": "happy",
      "intensity": 0.85,
      "state": "standing",
      "actions": ["wave", "talk_open"],
      "body_mode": "catalog",
      "humanml_prompt": "",
      "camera_role": "A_cam",
      "camera_shot": "MS",
      "camera_move": "static",
      "transition_in": "cut",
      "target_duration_s": 6.0,
      "hold_before_s": 0.2,
      "hold_after_s": 0.4,
      "pace": "medium",
      "clip_speed": 1.0,
      "clip_policy": "hold_end",
      "director_notes": "social catalog beat"
    }
  ]
}
"""


def _words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def _estimate_speech_s(text: str) -> float:
    # ~2.5 words/sec + base
    w = _words(text)
    return max(2.5, 0.9 + w / 2.4)


def _guess_emotion(text: str) -> Tuple[str, float]:
    low = (text or "").lower()
    pairs = [
        (("happy", "happily", "won", "yeh", "yeah", "great", "love", "yay", "celebrate"), "happy", 0.88),
        (("sad", "terrible", "sorry", "cry", "lonely", "failed"), "sad", 0.82),
        (("angry", "furious", "hate", "unacceptable"), "angry", 0.9),
        (("wow", "surprise", "shocked", "no idea"), "surprised", 0.88),
        (("worried", "afraid", "fear", "scared"), "fearful", 0.8),
        (("think", "explain", "carefully", "how"), "thinking", 0.75),
        (("thank", "welcome", "watching"), "encouraging", 0.8),
    ]
    for keys, emo, inten in pairs:
        if any(k in low for k in keys):
            return emo, inten
    return "neutral", 0.72


def _guess_state(text: str, default: str = "standing") -> str:
    from face_agents.director_schema import is_motion_caption

    low = (text or "").lower()
    if re.search(r"\b(sit|sitting|seated)\b", low):
        return "sitting"
    if re.search(r"\b(dance|dancing)\b", low):
        return "dancing"
    if re.search(r"\b(walk|walking)\b", low):
        return "walking"
    if is_motion_caption(text or ""):
        return "locomotion"
    return default


def _director_clip_speed(
    *,
    emotion: str,
    intensity: float,
    pace: str,
    camera_move: str,
    state: str,
    text: str,
    explicit: float | None = None,
) -> float:
    """
    Director-owned body playback rate.
    <1 slow-mo, 1 normal, >1 faster. Clamped to safe Blender range later.
    """
    if explicit is not None and explicit > 0.05:
        return float(max(0.35, min(1.6, explicit)))

    emo = (emotion or "neutral").lower()
    pace = (pace or "medium").lower()
    move = (camera_move or "static").lower()
    low = (text or "").lower()
    inten = float(intensity or 0.7)

    # Explicit language
    if re.search(r"(?i)\b(slow[\s-]?mo|slow\s+motion|in\s+slow)\b", low):
        return 0.5
    if re.search(r"(?i)\b(speed\s+up|fast[\s-]?forward|quickly)\b", low):
        return 1.25

    # Emotional / cinematic defaults
    if emo in ("sad", "apologetic", "concerned", "thinking") or pace == "slow":
        base = 0.72 if emo in ("sad", "apologetic") else 0.85
    elif emo in ("angry", "surprised", "fearful") and inten >= 0.85:
        base = 1.12
    elif emo in ("happy", "encouraging") and inten >= 0.9:
        base = 1.05
    else:
        base = 1.0

    # Camera language: crane / reveal / long orbit → slightly slower body for weight
    if move in ("crane_up", "crane_down", "reveal", "orbit") and pace in ("slow", "medium"):
        base = min(base, 0.88)
    if move == "handheld" and emo in ("fearful", "angry"):
        base = max(base, 1.08)
    if state == "walking" and pace == "slow":
        base = min(base, 0.8)
    if state == "walking" and pace == "fast":
        base = max(base, 1.15)

    return float(max(0.4, min(1.5, base)))


def _needs_open_momask(stage: str, spoken: str, state: str, actions: list) -> Tuple[bool, str]:
    """
    When t2m is needed vs catalog is enough.
    Catalog covers known clips + tiny social gestures.
    Any leftover / multi-step / full-body class goes to MoMask.
    """
    from face_agents.director_schema import is_full_body_state, is_motion_caption
    from face_agents.motion_router import _catalog_index, _load_catalog, _match_catalog, _stage_needs_open_vocab, _tokenize

    stage = (stage or "").strip()
    acts = {str(a).lower() for a in (actions or [])}
    multi = bool(re.search(r"(?i)\b(and then|then|after that|finally|while .+ and)\b", stage))
    compound_acts = len([a for a in acts if a not in ("talk_open", "talk_emphasize", "none", "")]) >= 2
    if multi or compound_acts:
        return True, "open/multi-step stage → momask t2m"
    if not stage:
        if is_full_body_state(state):
            return True, "full-body class → momask"
        return False, "no stage → catalog"
    rows = _catalog_index((_load_catalog().get("clips") or {}))
    hits = _match_catalog(_tokenize(stage), list(acts), rows)
    if _stage_needs_open_vocab(stage, _tokenize(stage), hits):
        return True, "stage residual → momask t2m"
    if is_motion_caption(stage) and not (hits and hits[0]["score"] >= 1.2):
        return True, "uncatalogued motion caption → momask"
    if is_full_body_state(state) and not (hits and hits[0]["score"] >= 1.2):
        return True, "full-body class → momask"
    return False, "catalog covers stage/gesture"


def _creative_camera(
    *,
    emotion: str,
    intensity: float,
    state: str,
    text: str,
    duration_s: float,
    shot_index: int,
    prev_shot: str,
    prev_move: str,
    prev_role: str,
) -> Tuple[str, str, str, str]:
    """
    Rules director: (size, move, role, pace) with variety and movie pacing.
    Avoid reusing same move/size when possible; multi-cam for coverage.
    """
    emo = (emotion or "neutral").lower()
    st = (state or "standing").lower()
    inten = float(intensity or 0.7)
    low = (text or "").lower()
    pace = "medium"
    role = "A_cam"

    # Any traveling / generated full-body → env or A wide + follow/truck
    from face_agents.director_schema import is_full_body_state, is_motion_caption

    if is_full_body_state(st) or is_motion_caption(text or ""):
        size, move = "WS", "follow" if duration_s >= 6 else "truck"
        role = "env_cam" if shot_index == 0 else "A_cam"
        pace = "medium"
    elif "wave" in low or "hello" in low or "welcome" in low:
        size, move = "MS", "static" if duration_s < 5 else "dolly_in"
        role = "A_cam"
    elif emo in ("angry",) and inten >= 0.75:
        size, move = "CU", "dolly_in" if duration_s < 8 else "orbit"
        role = "A_cam" if shot_index % 2 == 0 else "B_cam"
        pace = "fast"
    elif emo in ("surprised", "fearful") and inten >= 0.7:
        size, move = "CU", "static" if duration_s < 4 else "handheld"
        role = "C_cam"
        pace = "fast"
    elif emo in ("sad", "apologetic", "concerned"):
        size, move = "MCU", "dolly_out" if duration_s >= 5 else "static"
        role = "A_cam"
        pace = "slow"
    elif emo in ("thinking", "sarcastic"):
        size, move = "MCU", "orbit" if duration_s >= 6 else "pan_right"
        role = "B_cam"
        pace = "slow"
    elif emo in ("happy", "encouraging") and inten >= 0.85:
        size, move = "MS", "crane_up" if "won" in low or "celebrate" in low else "arc"
        role = "A_cam"
        pace = "medium"
    elif duration_s >= 10:
        size, move = "MS", "dolly_in"
        role = "A_cam"
        pace = "slow"
    else:
        size, move = "MS", "static"
        role = "A_cam"

    # Avoid ECU by default (head penetration risk); only short static detail
    if size == "ECU" and move not in ("static", "pan_left", "pan_right"):
        size, move = "CU", "static"

    # Variety vs previous
    if size == prev_shot and move == prev_move and shot_index > 0:
        alts = [
            ("MCU", "orbit", "B_cam"),
            ("MS", "dolly_in", "A_cam"),
            ("MLS", "truck_right", "env_cam"),
            ("CU", "static", "C_cam"),
            ("MS", "arc", "B_cam"),
        ]
        size, move, role = alts[shot_index % len(alts)]

    # Long takes: slower moves
    if duration_s >= 8 and move in ("dolly_in", "dolly_out"):
        pace = "slow"
    if duration_s >= 8 and move == "static" and emo not in ("sad",):
        move = "dolly_in" if shot_index % 2 == 0 else "orbit"

    # Don't always stick to same role
    if role == prev_role and shot_index > 0 and shot_index % 3 == 0:
        role = "B_cam" if prev_role != "B_cam" else "A_cam"

    return size, move, role, pace


def _merge_for_cinematic(lines: List[str], *, target_total_s: float = 60.0) -> List[str]:
    """Merge short sentences into longer cinematic takes."""
    if not lines:
        return lines
    # Aim for ~5–7 takes for 60s
    target_takes = max(4, min(8, int(round(target_total_s / 10.0))))
    if len(lines) <= target_takes:
        return lines

    # Greedy merge until near target_takes
    merged: List[str] = []
    buf = lines[0]
    for line in lines[1:]:
        starts_new = bool(re.match(r"(?i)^(then|finally|next|after that|later)\b", line.strip()))
        walk_break = bool(re.search(r"(?i)\b(walk|celebrate|wave)\b", line)) and _words(buf) >= 12
        if starts_new or walk_break or (_words(buf) >= 28 and len(merged) + 1 + (len(lines) - lines.index(line)) > target_takes):
            # soft: if still too many remaining, keep merging
            remaining = len(lines) - lines.index(line)
            if len(merged) + 1 + remaining <= target_takes + 1 and not starts_new:
                buf = buf.rstrip() + " " + line
            else:
                merged.append(buf)
                buf = line
        else:
            buf = buf.rstrip() + " " + line
    if buf:
        merged.append(buf)

    # If still too many, pair adjacent
    while len(merged) > target_takes + 1:
        nxt = []
        i = 0
        while i < len(merged):
            if i + 1 < len(merged) and len(nxt) + (len(merged) - i) // 1 > target_takes:
                nxt.append(merged[i] + " " + merged[i + 1])
                i += 2
            else:
                nxt.append(merged[i])
                i += 1
        merged = nxt
    return merged


def _split_story(story: str) -> List[str]:
    raw = (story or "").strip()
    if not raw:
        return []
    if re.search(r"(?im)^\s*(shot|beat|scene)\s*\d+", raw):
        parts = re.split(r"(?im)^\s*(?:shot|beat|scene)\s*\d+\s*[:.\-)]\s*", raw)
        return [p.strip() for p in parts if p and p.strip()]
    paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(paras) > 1:
        return paras
    sents = re.split(r"(?<=[.!?])\s+", raw)
    sents = [s.strip() for s in sents if s.strip()]
    # Comma stage beats: "Walk on a sunny street, wave hello, say goodbye."
    if len(sents) <= 1 and "," in raw:
        verb = (
            r"walk|wave|say|run|stop|look|sit|stand|turn|point|celebrate|"
            r"nod|shrug|jump|dance|talk|speak|greet|leave|enter"
        )
        parts = re.split(
            rf"\s*,\s*(?=(?:and\s+)?(?:then\s+)?(?:{verb})\b)",
            raw,
            flags=re.I,
        )
        parts = [p.strip(" ,.") for p in parts if p and p.strip(" ,.")]
        if len(parts) >= 2:
            return parts
    return sents if sents else [raw]


def _normalize_shot_fields(raw: Dict[str, Any], i: int) -> Dict[str, Any]:
    size = str(raw.get("camera_shot") or raw.get("shot") or "MS").upper()
    if size not in CAMERA_SIZES:
        size = "MS"
    move = str(raw.get("camera_move") or raw.get("move") or "static").lower()
    if move not in CAMERA_MOVES:
        move = "static"
    role = str(raw.get("camera_role") or raw.get("role") or "A_cam")
    if role not in CAMERA_ROLES:
        role = "A_cam"
    # Safety: never ECU + aggressive push
    if size == "ECU" and move in ("dolly_in", "follow", "handheld"):
        size, move = "CU", "static" if move == "dolly_in" else move
    clip = str(raw.get("clip_policy") or "hold_end").lower()
    if clip not in CLIP_POLICIES or clip == "stretch":
        clip = "hold_end"
    pace = str(raw.get("pace") or "medium").lower()
    if pace not in ("slow", "medium", "fast"):
        pace = "medium"
    body = str(raw.get("body_mode") or "auto").lower()
    if body not in ("catalog", "momask", "both", "auto", "procedural"):
        body = "auto"
    text = str(raw.get("text") or raw.get("dialogue") or "")
    # Stage / spoken — director or split from text
    stage = str(raw.get("stage") or "").strip()
    spoken = str(raw.get("spoken") or raw.get("dialogue_text") or "").strip()
    if not spoken or (stage and spoken == text):
        stg, spk = split_stage_and_spoken(text)
        stage = stage or stg
        spoken = spoken or spk
    spoken = clean_spoken_for_tts(spoken)
    if not spoken:
        spoken = clean_spoken_for_tts(text) or text

    emo = str(raw.get("emotion") or "neutral")
    inten = float(raw.get("intensity") or 0.75)
    state = str(raw.get("state") or "standing")
    target = float(raw.get("target_duration_s") or 0.0)
    if target < 1.0:
        target = max(6.0, _estimate_speech_s(spoken or text) + 1.5)
    min_take = float(os.environ.get("MOVIE_MIN_TAKE_S", "5.5"))
    target = max(min_take, target)

    speed_raw = raw.get("clip_speed")
    try:
        speed_explicit = float(speed_raw) if speed_raw is not None and str(speed_raw) != "" else None
    except (TypeError, ValueError):
        speed_explicit = None
    clip_speed = _director_clip_speed(
        emotion=emo,
        intensity=inten,
        pace=pace,
        camera_move=move,
        state=state,
        text=text,
        explicit=speed_explicit,
    )

    return {
        "shot_id": str(raw.get("shot_id") or raw.get("id") or f"shot_{i:02d}"),
        "text": text,
        "stage": stage,
        "spoken": spoken,
        "summary": str(raw.get("summary") or "")[:140],
        "emotion": emo,
        "intensity": inten,
        "state": state,
        "actions": list(raw.get("actions") or []),
        "body_mode": body,
        "humanml_prompt": str(raw.get("humanml_prompt") or ""),
        "camera_shot": size,
        "camera_move": move,
        "camera_role": role,
        "transition_in": str(raw.get("transition_in") or "cut"),
        "target_duration_s": target,
        "hold_before_s": float(raw.get("hold_before_s") or 0.25),
        "hold_after_s": float(raw.get("hold_after_s") or 0.5),
        "pace": pace,
        "clip_speed": clip_speed,
        "clip_policy": clip,
        "director_notes": str(raw.get("director_notes") or "")[:200],
        "location": str(raw.get("location") or "default"),
        "look": raw.get("look") if isinstance(raw.get("look"), dict) else {},
    }


def _sanitize_humanml(prompt: str, stage: str = "", state: str = "") -> str:
    """Force MoMask/HumanML style; never leave dialogue or bare tags."""
    try:
        from face_agents.momask_body_pipeline import (
            _clean_caption,
            build_humanml_prompt,
            to_humanml_caption,
        )
    except Exception:
        def _clean_caption(s: str) -> str:
            s = (s or "").strip()
            if s and not s.lower().startswith("a person"):
                s = "a person " + s
            return s[:240]

        def build_humanml_prompt(**kw) -> str:
            return _clean_caption(kw.get("humanml_prompt") or kw.get("state") or "stands")

        def to_humanml_caption(s: str, **_kw) -> str:
            return _clean_caption(s)

    p = (prompt or "").strip()
    # Reject dialogue-like prompts
    pl = p.lower()
    if (
        not p
        or len(p) < 12
        or " " not in p
        or pl in ("hi", "hello", "hey")
        or pl.startswith("say ")
        or (pl.startswith("\"") and pl.endswith("\""))
    ):
        p = build_humanml_prompt(
            humanml_prompt="",
            state=state or "standing",
            text=stage or "",
        )
        if stage and len(stage) > 8:
            p = to_humanml_caption(stage, state=state or "")
    else:
        p = to_humanml_caption(p, state=state or "")
    return p


def _to_shot_detail(n: Dict[str, Any]) -> ShotDetail:
    body_mode = n["body_mode"]
    hml = n.get("humanml_prompt") or ""
    if str(body_mode).lower() in ("momask", "both"):
        hml = _sanitize_humanml(hml, stage=n.get("stage") or "", state=n.get("state") or "")
    return ShotDetail(
        shot_id=n["shot_id"],
        text=n["text"],
        stage=n.get("stage") or "",
        spoken=n.get("spoken") or "",
        emotion=n["emotion"],
        intensity=n["intensity"],
        state=n["state"],
        actions=list(n["actions"]),
        body_mode=body_mode,
        humanml_prompt=hml,
        camera_shot=n["camera_shot"],
        camera_move=n["camera_move"],
        camera_role=n["camera_role"],
        transition_in=n["transition_in"],
        location=n.get("location") or "default",
        summary=n["summary"],
        target_duration_s=n["target_duration_s"],
        hold_before_s=n["hold_before_s"],
        hold_after_s=n["hold_after_s"],
        pace=n["pace"],
        clip_speed=float(n.get("clip_speed") or 1.0),
        clip_policy=n["clip_policy"],
        director_notes=n["director_notes"],
        look=LookPlan.from_dict(n.get("look") if isinstance(n.get("look"), dict) else None),
    )


def _apply_motion_router(
    shots: List[ShotDetail],
    *,
    llm_provider: Any = None,
    world: Optional[WorldState] = None,
    board: Optional[ContinuityBoard] = None,
) -> List[ShotDetail]:
    world = world or WorldState()
    board = board or ContinuityBoard()
    prefer_open = os.environ.get("MOMASK_PREFER_OPEN", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )
    # Movie bake: allow MoMask gen when director/router requests open motion
    # (set MOMASK_SYNC=0 to force catalog-only offline)
    if "MOMASK_SYNC" not in os.environ:
        os.environ["MOMASK_SYNC"] = "1"

    for sh in shots:
        ch = world.ensure_character("hero")
        # Ensure stage/spoken filled
        if not sh.spoken or not sh.stage:
            stg, spk = split_stage_and_spoken(sh.text)
            sh.stage = sh.stage or stg
            sh.spoken = clean_spoken_for_tts(sh.spoken or spk or sh.text)

        route_input = sh.stage.strip()
        if sh.spoken:
            route_input = (route_input + f" while saying {sh.spoken}").strip() if route_input else sh.spoken
        if not route_input:
            route_input = sh.text

        mplan = route_motion(
            user_input=route_input,
            spoken=sh.spoken,
            emotion=sh.emotion,
            intensity=sh.intensity,
            actions=sh.actions,
            state=sh.state or ch.base_state or "standing",
            llm_provider=None,
        )

        need_open, open_reason = _needs_open_momask(
            sh.stage, sh.spoken, sh.state, sh.actions or mplan.catalog_actions
        )
        from face_agents.director_schema import is_full_body_state as _full_body

        if prefer_open and _full_body(sh.state) and sh.target_duration_s >= 6:
            need_open, open_reason = True, "MOMASK_PREFER_OPEN + loco take"

        # Explicit director momask / policy
        if sh.body_mode in ("momask", "both") and (sh.humanml_prompt or need_open or mplan.humanml_prompt):
            sh.body_mode = "momask" if sh.body_mode != "both" else "momask"
            sh.allow_sync_gen = True
            sh.humanml_prompt = _sanitize_humanml(
                sh.humanml_prompt or mplan.humanml_prompt or "",
                stage=sh.stage or "",
                state=sh.state or "",
            )
            sh.motion_reason = sh.director_notes or "director momask"
            sh.motion_engines = ["momask"]
            if sh.clip_policy == "loop":
                sh.clip_policy = "momask_match"
        elif sh.clip_policy == "momask_match":
            sh.body_mode = "momask"
            sh.allow_sync_gen = True
            sh.humanml_prompt = _sanitize_humanml(
                sh.humanml_prompt or mplan.humanml_prompt or "",
                stage=sh.stage or sh.state or "",
                state=sh.state or "",
            )
            sh.motion_reason = "director clip_policy=momask_match"
            sh.motion_engines = ["momask"]
        elif need_open or (mplan.body_mode in ("momask", "both") and mplan.allow_sync_gen):
            # Intelligent: use t2m only when needed
            sh.body_mode = "momask"
            sh.allow_sync_gen = True
            sh.humanml_prompt = _sanitize_humanml(
                sh.humanml_prompt
                or mplan.humanml_prompt
                or (sh.stage if sh.stage else f"moves looking {sh.emotion}"),
                stage=sh.stage or "",
                state=sh.state or "",
            )
            sh.clip_policy = "momask_match"
            sh.motion_reason = open_reason or mplan.reason
            sh.motion_engines = ["momask"]
            if mplan.catalog_actions:
                sh.actions = list(dict.fromkeys(list(sh.actions) + list(mplan.catalog_actions)))
        else:
            # Catalog path
            mode = mplan.body_mode
            if mode in ("both", "momask") and not need_open:
                # Router said momask but director heuristics say catalog covers — stay catalog
                mode = "catalog"
            if mode == "procedural":
                mode = "catalog"
            if sh.body_mode in ("", "auto"):
                sh.body_mode = mode if mode in ("catalog", "momask") else "catalog"
            if sh.body_mode != "momask":
                sh.body_mode = "catalog"
                sh.allow_sync_gen = False
                sh.motion_reason = mplan.reason or open_reason
                sh.motion_engines = ["catalog"]
                if mplan.catalog_actions and not sh.actions:
                    sh.actions = list(mplan.catalog_actions)
                if not sh.actions and sh.spoken:
                    sh.actions = ["talk_open"]
            if mplan.state and sh.state == "standing":
                sh.state = mplan.state

        # Clip speed final pass if still default unset
        if not sh.clip_speed or abs(float(sh.clip_speed) - 1.0) < 1e-6:
            # only recompute if pace suggests otherwise
            if sh.pace in ("slow", "fast") or sh.emotion in ("sad", "angry"):
                sh.clip_speed = _director_clip_speed(
                    emotion=sh.emotion,
                    intensity=sh.intensity,
                    pace=sh.pace,
                    camera_move=sh.camera_move,
                    state=sh.state,
                    text=sh.text,
                    explicit=None,
                )
        sh.clip_speed = float(max(0.35, min(1.6, sh.clip_speed or 1.0)))

        ch.emotion = sh.emotion
        ch.intensity = sh.intensity
        ch.base_state = sh.state
        ch.body_mode = sh.body_mode

        board.record_shot_plan(
            ShotMemory(
                shot_id=sh.shot_id,
                summary=sh.summary,
                text=sh.text,
                emotion=sh.emotion,
                intensity=sh.intensity,
                state=sh.state,
                body_mode=sh.body_mode,
                actions=list(sh.actions),
                humanml_prompt=sh.humanml_prompt,
                camera_shot=sh.camera_shot,
                camera_move=sh.camera_move,
                camera_role=sh.camera_role,
                transition_in=sh.transition_in,
                target_duration_s=sh.target_duration_s,
                hold_before_s=sh.hold_before_s,
                hold_after_s=sh.hold_after_s,
                pace=sh.pace,
                clip_policy=sh.clip_policy,
                director_notes=(sh.director_notes or "") + f" | speed={sh.clip_speed:.2f}",
            )
        )
    return shots


def plan_rules(
    story: str = "",
    *,
    script_shots: Optional[List[Dict[str, Any]]] = None,
    title: str = "",
    board: Optional[ContinuityBoard] = None,
    world: Optional[WorldState] = None,
    llm_provider: Any = None,
    target_total_s: float = 60.0,
) -> Tuple[List[ShotDetail], ContinuityBoard]:
    """Offline intelligent rules director (+ optional script merge)."""
    board = board or ContinuityBoard()
    board.title = title or board.title
    board.story = story or board.story
    board.target_total_s = target_total_s
    board.director_source = "rules"
    world = world or WorldState()
    world.ensure_character("hero")

    if script_shots:
        # Enrich script with director timing/camera if missing
        shots: List[ShotDetail] = []
        prev_s, prev_m, prev_r = board.last_camera_shot, board.last_camera_move, board.last_camera_role
        for i, raw in enumerate(script_shots, 1):
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or raw.get("dialogue") or "")
            emo = str(raw.get("emotion") or "")
            inten = float(raw.get("intensity") or 0)
            if not emo:
                emo, inten = _guess_emotion(text)
            elif inten <= 0:
                _, inten = _guess_emotion(text)
            state = str(raw.get("state") or _guess_state(text))
            speech_est = _estimate_speech_s(text)
            target = float(raw.get("target_duration_s") or 0.0)
            if target < 1.0:
                # Director: speech + holds, movie floor
                target = max(6.0, speech_est + 1.8)
            size = str(raw.get("camera_shot") or "")
            move = str(raw.get("camera_move") or "")
            role = str(raw.get("camera_role") or "")
            pace = str(raw.get("pace") or "")
            if not size or not move:
                size, move, role2, pace2 = _creative_camera(
                    emotion=emo, intensity=inten, state=state, text=text,
                    duration_s=target, shot_index=i - 1,
                    prev_shot=prev_s, prev_move=prev_m, prev_role=prev_r,
                )
                size = size or size
                move = move or move
                role = role or role2
                pace = pace or pace2
            if not role:
                role = "A_cam"
            if not pace:
                pace = "medium"
            n = _normalize_shot_fields({
                **raw,
                "emotion": emo,
                "intensity": inten,
                "state": state,
                "camera_shot": size,
                "camera_move": move,
                "camera_role": role,
                "pace": pace,
                "target_duration_s": target,
                "hold_before_s": raw.get("hold_before_s", 0.25),
                "hold_after_s": raw.get("hold_after_s", 0.55),
                "clip_policy": raw.get("clip_policy") or ("loop" if state in ("walking", "standing") else "hold_end"),
            }, i)
            shots.append(_to_shot_detail(n))
            prev_s, prev_m, prev_r = n["camera_shot"], n["camera_move"], n["camera_role"]
        shots = _apply_motion_router(shots, world=world, board=board, llm_provider=llm_provider)
        return shots, board

    # Story → merge → director assign
    lines = _merge_for_cinematic(_split_story(story), target_total_s=target_total_s)
    if not lines:
        lines = ["Hello. This is a short cinematic test."]

    shots = []
    prev_s, prev_m, prev_r = "MS", "static", "A_cam"
    for i, line in enumerate(lines, 1):
        emo, inten = _guess_emotion(line)
        state = _guess_state(line, world.ensure_character("hero").base_state or "standing")
        speech_est = _estimate_speech_s(line)
        target = max(6.5, speech_est + 2.0)
        # distribute toward target_total
        size, move, role, pace = _creative_camera(
            emotion=emo, intensity=inten, state=state, text=line,
            duration_s=target, shot_index=i - 1,
            prev_shot=prev_s, prev_move=prev_m, prev_role=prev_r,
        )
        n = _normalize_shot_fields({
            "shot_id": f"shot_{i:02d}",
            "text": line,
            "summary": line[:80],
            "emotion": emo,
            "intensity": inten,
            "state": state,
            "camera_shot": size,
            "camera_move": move,
            "camera_role": role,
            "pace": pace,
            "target_duration_s": target,
            "hold_before_s": 0.3,
            "hold_after_s": 0.65 if emo in ("sad", "thinking") else 0.45,
            "clip_policy": "loop" if state in ("walking", "standing", "sitting") else "hold_end",
            "actions": ["wave", "talk_open"] if "wave" in line.lower() else (["talk_open"] if "say" in line.lower() or len(line.split()) > 4 else []),
            "director_notes": f"rules take {i}/{len(lines)}",
        }, i)
        shots.append(_to_shot_detail(n))
        prev_s, prev_m, prev_r = size, move, role

    shots = _apply_motion_router(shots, world=world, board=board, llm_provider=llm_provider)
    return shots, board


def plan_llm(
    story: str,
    *,
    title: str = "",
    board: Optional[ContinuityBoard] = None,
    world: Optional[WorldState] = None,
    llm_provider: Any = None,
    target_total_s: float = 60.0,
    script_hint: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[ShotDetail], ContinuityBoard]:
    """One LLM call with full ContinuityBoard inject. Falls back to rules."""
    board = board or ContinuityBoard()
    board.story = story or board.story
    board.title = title or board.title
    board.target_total_s = target_total_s
    world = world or WorldState()

    if llm_provider is None:
        return plan_rules(story, title=title, board=board, world=world, target_total_s=target_total_s,
                          script_shots=script_hint)

    user = (
        f"CONTINUITY_BOARD (external memory — trust this):\n{board.prompt_pack()}\n\n"
        f"TITLE: {title or 'untitled'}\n"
        f"TARGET_TOTAL_S: {target_total_s}\n"
        f"STORY / SCRIPT:\n{story}\n\n"
    )
    if script_hint:
        user += "OPTIONAL_HINT_SHOTS (you may merge/enrich, not forced):\n"
        user += json.dumps(script_hint, indent=2)[:4000] + "\n\n"
    user += (
        "Plan a cinematic multi-shot sequence. "
        "Create cameras via camera_role when you need different angles. "
        "Return ONLY the JSON object."
    )

    try:
        data = None
        if hasattr(llm_provider, "chat_json"):
            from llm_fw.providers.base import LLMMessage
            data = llm_provider.chat_json(
                [LLMMessage(role="user", content=user)],
                system=DIRECTOR_SYSTEM,
            )
        elif hasattr(llm_provider, "chat"):
            from llm_fw.providers.base import LLMMessage
            resp = llm_provider.chat(
                [LLMMessage(role="user", content=user)],
                system=DIRECTOR_SYSTEM,
            )
            raw = getattr(resp, "content", None) or str(resp)
            m = re.search(r"\{[\s\S]*\}", raw)
            data = json.loads(m.group(0)) if m else None
        elif callable(llm_provider):
            raw = llm_provider(DIRECTOR_SYSTEM + "\n\n" + user)
            m = re.search(r"\{[\s\S]*\}", str(raw))
            data = json.loads(m.group(0)) if m else None

        if not isinstance(data, dict) or data.get("parse_error"):
            raise ValueError("LLM plan parse failed")

        items = data.get("shots") or data.get("beats") or []
        if not items:
            raise ValueError("empty shots")

        board.style = str(data.get("style") or "cinematic")
        board.target_total_s = float(data.get("target_total_s") or target_total_s)
        if data.get("director_notes"):
            board.notes.append(str(data.get("director_notes")))
        board.director_source = "llm"

        shots: List[ShotDetail] = []
        for i, raw in enumerate(items, 1):
            if not isinstance(raw, dict):
                continue
            n = _normalize_shot_fields(raw, i)
            shots.append(_to_shot_detail(n))

        if not shots:
            raise ValueError("no valid shots")

        shots = _apply_motion_router(shots, world=world, board=board, llm_provider=None)
        print(f"[movie_director] LLM plan → {len(shots)} shot(s) style={board.style}")
        return shots, board
    except Exception as e:
        print(f"[movie_director] LLM failed ({e}) — rules fallback")
        return plan_rules(
            story, title=title, board=board, world=world,
            target_total_s=target_total_s, script_shots=script_hint,
        )


def direct_production(
    *,
    story: str = "",
    script: Optional[Dict[str, Any] | List[Any]] = None,
    title: str = "",
    llm_provider: Any = None,
    use_llm: Optional[bool] = None,
    world: Optional[WorldState] = None,
    target_total_s: float = 60.0,
) -> Tuple[List[ShotDetail], ContinuityBoard]:
    """
    Main entry: intelligent director for story or script.
    use_llm: default True if provider given, else rules.
    """
    board = ContinuityBoard(title=title, story=story, target_total_s=target_total_s)
    world = world or WorldState()
    world.ensure_character("hero")

    script_shots = None
    if script is not None:
        if isinstance(script, list):
            script_shots = script
        else:
            script_shots = script.get("shots") or script.get("beats") or script.get("sentences") or []
            if not story:
                story = str(script.get("story") or script.get("title") or title or "")
            if script.get("title") and not title:
                title = str(script.get("title"))
                board.title = title

    if use_llm is None:
        use_llm = llm_provider is not None and os.environ.get("MOVIE_DIRECTOR_LLM", "1") not in (
            "0", "false", "no", "off",
        )

    if use_llm and llm_provider is not None:
        # For scripts: pass as hint so director can merge + enrich durations/cameras
        if script_shots:
            # Build pseudo-story from texts for board
            board.story = story or "\n".join(
                str(s.get("text") or "") for s in script_shots if isinstance(s, dict)
            )
            return plan_llm(
                board.story,
                title=title,
                board=board,
                world=world,
                llm_provider=llm_provider,
                target_total_s=target_total_s,
                script_hint=script_shots,
            )
        return plan_llm(
            story, title=title, board=board, world=world,
            llm_provider=llm_provider, target_total_s=target_total_s,
        )

    return plan_rules(
        story, script_shots=script_shots, title=title,
        board=board, world=world, target_total_s=target_total_s,
    )
