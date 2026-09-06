"""
orchestrator_agents.py

Multi-agent face pipeline for realistic expression while talking.

  python orchestrator_agents.py

Architecture (scene FIRST, then performance):
    1) LookAgent → UDP type=look → Blender bpy builds Look_Set
       (HDRI + ground + trees/shops/platform/… — any location kit)
    2) Parler TTS → WAV  (only if there is dialogue)
    3) FaceCoordinator + body (catalog OR MoMask T2M)
         ↓
    play → UDP → Blender
         look packets: type=look (structures via bpy API)
         face packets: viseme / emotion / head
         body packets: type=body action=…

  MoMask (optional, USE_MOMASK=1):
    Director humanml_prompt → TTS duration → gen_t2m --motion_length (20fps)
    → rest→motion→rest BVH → Rokoko direct → Action
    play speed synced so body ends with speech; then A-pose idle settle.
    Face pipeline is unchanged (lips + expressions stay independent).

  Movie camera (default ON, USE_MOVIE_CAMERA=1):
    CameraAgent rules → shot/move keyframes → UDP type=camera → MovieCam in Blender
    Additive track; does not change face or body.

Classic single-pipeline (no agents):  python orchestrator.py
Full Brain dual path:                 python orchestrator_brain.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from face_agents import FaceCoordinator, FeedbackLogger
import wav2arkit

# LLM chat mode
LLM_MODE = os.environ.get("LLM_CHAT", "0").strip() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 9001
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)
FPS = 30.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use Brain encoder for expression agents (set USE_BRAIN=0 to disable)
USE_BRAIN = os.environ.get("USE_BRAIN", "1").strip() not in ("0", "false", "False", "no")

VALID_EMOTIONS = {
    "neutral", "happy", "sad", "angry", "surprised",
    "disgusted", "fearful", "sarcastic", "thinking",
    "calm", "apologetic", "assertive", "concerned", "encouraging",
    "surprise", "fear", "disgust",
}


def get_llm_provider():
    """Load LLM provider from config."""
    try:
        from llm_fw.providers import get_provider
        cfg_path = Path("llm_fw/config.json")
        if not cfg_path.exists():
            print("[WARN] llm_fw/config.json not found, LLM chat disabled")
            return None
        cfg = json.load(open(cfg_path))
        return get_provider(cfg)
    except Exception as e:
        print(f"[WARN] LLM provider load failed: {e}")
        return None


def _normalize_director_items(items: list) -> list:
    """
    Normalize sentences/beats into face pipeline dicts + optional body actions.
    Body actions are logged/stored; body mesh playback comes later.
    """
    try:
        from face_agents.director_schema import parse_beat
    except Exception:
        parse_beat = None

    cleaned = []
    for s in items:
        if not isinstance(s, dict):
            continue
        if parse_beat is not None:
            beat = parse_beat(s, apply_emotion_defaults=True)
            if not (
                beat.text
                or beat.humanml_prompt
                or beat.body_mode in ("momask", "both")
                or getattr(beat, "motion_only", False)
            ):
                continue
            emo = beat.emotion if beat.emotion in VALID_EMOTIONS else "neutral"
            cleaned.append({
                "text": beat.text,
                "emotion": emo,
                "intensity": beat.intensity,
                "actions": beat.actions,
                "action_timing": beat.action_timing,
                "state": beat.state,
                "gesture_target": beat.gesture_target,
                "hand": beat.hand,
                "body_mode": getattr(beat, "body_mode", "auto") or "auto",
                "humanml_prompt": getattr(beat, "humanml_prompt", "") or "",
                "motion_duration_s": getattr(beat, "motion_duration_s", None),
                "motion_length": getattr(beat, "motion_length", None),
                "allow_sync_gen": bool(getattr(beat, "allow_sync_gen", False)),
                "motion_only": bool(getattr(beat, "motion_only", False)),
                "motion_engines": list(getattr(beat, "motion_engines", []) or []),
                "motion_reason": getattr(beat, "motion_reason", "") or "",
                "camera_shot": getattr(beat, "camera_shot", "") or "",
                "camera_move": getattr(beat, "camera_move", "") or "",
                "camera_role": getattr(beat, "camera_role", "") or "",
                "look": (
                    beat.look.to_dict()
                    if getattr(beat, "look", None) is not None
                    else (s.get("look") if isinstance(s.get("look"), dict) else {})
                ),
                "clip_policy": getattr(beat, "clip_policy", "") or s.get("clip_policy") or "",
            })
            continue
        text = str(s.get("text", "")).strip()
        emotion = str(s.get("emotion", "neutral")).lower().strip()
        try:
            intensity = float(s.get("intensity", 0.7))
        except (TypeError, ValueError):
            intensity = 0.7
        aliases = {"surprise": "surprised", "fear": "fearful", "disgust": "disgusted"}
        emotion = aliases.get(emotion, emotion)
        hml = str(s.get("humanml_prompt") or "")
        mode = str(s.get("body_mode") or "auto")
        if text or hml or mode in ("momask", "both"):
            cleaned.append({
                "text": text,
                "emotion": emotion if emotion in VALID_EMOTIONS else "neutral",
                "intensity": max(0.0, min(1.0, intensity)),
                "actions": [],
                "action_timing": "during",
                "state": str(s.get("state") or "standing"),
                "gesture_target": "none",
                "hand": "right",
                "body_mode": mode,
                "humanml_prompt": hml,
                "motion_duration_s": s.get("motion_duration_s") or s.get("duration_s"),
                "motion_length": s.get("motion_length") or s.get("momask_frames"),
                "allow_sync_gen": bool(s.get("allow_sync_gen", False)),
                "motion_only": bool(s.get("motion_only")),
                "camera_shot": str(s.get("camera_shot") or s.get("shot") or ""),
                "camera_move": str(s.get("camera_move") or s.get("move") or ""),
                "camera_role": str(s.get("camera_role") or ""),
                "look": s.get("look") if isinstance(s.get("look"), dict) else {},
                "clip_policy": str(s.get("clip_policy") or ""),
            })
    return cleaned


# Persistent director across chat turns (state continuity)
_BODY_DIRECTOR = None


def _get_body_director(llm_provider):
    global _BODY_DIRECTOR
    from face_agents.body_director_agent import BodyDirectorAgent
    if _BODY_DIRECTOR is None or getattr(_BODY_DIRECTOR, "llm", None) is not llm_provider:
        _BODY_DIRECTOR = BodyDirectorAgent(llm_provider=llm_provider)
    return _BODY_DIRECTOR


def _maybe_run_momask_body(
    *,
    body_mode: str,
    humanml_prompt: str,
    body_state: str,
    actions: list,
    gesture_target: str,
    hand: str,
    emotion: str,
    text: str,
    duration_s: float,
    seed: int,
    allow_sync_gen: bool = False,
    motion_length: int = 0,
) -> dict | None:
    """
    Body MoMask worker (mavie Stage 5).

    - MotionRouter / director decide body_mode + humanml_prompt.
    - duration_s (speech or director) → MoMask --motion_length @ 20fps for sync.
    - Cache hit → fast. Agent-requested open motion → may wait for gen_t2m.
    - Face never uses this path.
    """
    try:
        from face_agents.momask_body_pipeline import (
            build_humanml_prompt,
            ensure_action_in_blender_via_packet_hint,
            generate_body_action,
            lookup_cached_action,
            momask_sync_generate,
            should_use_momask,
            speech_to_motion_length,
        )
    except Exception as e:
        print(f"  [momask] import failed: {e}")
        return None

    # Never generate MoMask for tiny social lines (hi / hello)
    _low = " ".join((text or "").lower().split())
    if _low in {
        "hi", "hello", "hey", "hiya", "yo", "hello there", "hi there",
        "hey there", "thanks", "thank you", "bye", "ok", "okay",
    }:
        print("  [momask] skip — short greeting uses catalog (wave/talk)")
        return None
    if body_mode in ("catalog", "procedural"):
        # Mixamo catalog walks are in-place — open HumanML locomotion must use MoMask
        try:
            from face_agents.director_schema import is_motion_caption
            open_loco = bool((humanml_prompt or "").strip()) or is_motion_caption(text or "")
        except Exception:
            open_loco = bool((humanml_prompt or "").strip())
        if not open_loco:
            return None
        print("  [momask] override catalog — open locomotion caption (not in-place walk)")
        body_mode = "auto"

    if not should_use_momask(
        body_mode=body_mode,
        humanml_prompt=humanml_prompt,
        state=body_state,
        actions=actions,
    ):
        return None

    prompt = build_humanml_prompt(
        humanml_prompt=humanml_prompt,
        state=body_state,
        actions=actions,
        gesture_target=gesture_target,
        hand=hand,
        emotion=emotion,
        text=text,
    )

    ml = int(motion_length or 0)
    if ml <= 0:
        ml = speech_to_motion_length(duration_s, prompt=prompt)

    cached = lookup_cached_action(
        prompt, duration_s=duration_s, seed=seed, motion_length=ml
    )
    if cached and cached.ok:
        print(
            f"  [momask] cache hit → {cached.action_name} "
            f"(speech={float(duration_s):.2f}s frames={ml}@20fps)"
        )
        return ensure_action_in_blender_via_packet_hint(
            cached, speech_duration_s=float(duration_s)
        )

    # Router said open model for this beat, or env MOMASK_SYNC=1
    if not momask_sync_generate(allow_sync_gen=bool(allow_sync_gen)):
        print(
            "  [momask] no cache — catalog fallback "
            f"(allow_sync_gen={allow_sync_gen}, set MOMASK_SYNC=1 to always wait)."
        )
        return None

    print(
        f"  [momask] generate (sync): prompt={prompt!r} "
        f"speech={float(duration_s):.2f}s → motion_length={ml} "
        f"(prompt floor if auto)"
    )
    t0 = __import__("time").time()
    result = generate_body_action(
        prompt,
        duration_s=duration_s,
        seed=seed,
        motion_length=ml,
        use_cache=True,
    )
    if not result.ok:
        print(f"  [momask] FAILED → catalog/procedural: {result.error}")
        return None
    print(
        f"  [momask] OK action={result.action_name} "
        f"in {__import__('time').time() - t0:.1f}s blend={result.blend_path}"
    )
    return ensure_action_in_blender_via_packet_hint(
        result, speech_duration_s=float(duration_s)
    )


def chat_to_sentences(user_input: str, llm_provider) -> list:
    """
    Body Director agent (LLM + menus) → layered body beats.

    Guide workflow:
      user text → director JSON (base_state, upper_gesture_target, emotion)
               → normalize → TTS + face + body UDP layers
    """
    director = _get_body_director(llm_provider)
    try:
        print("  [body_director] directing performance…")
        beats = director.direct(user_input)
        if not beats:
            return [{
                "text": "I'm not sure how to respond.",
                "emotion": "thinking",
                "intensity": 0.5,
                "actions": ["think_chin"],
                "action_timing": "during",
                "state": "standing",
                "gesture_target": "chin",
                "hand": "right",
            }]
        for i, b in enumerate(beats, 1):
            print(
                f"  [director] beat {i}: emo={b.get('emotion')} "
                f"state={b.get('state')} gesture={b.get('gesture_target')}/{b.get('hand')} "
                f"actions={b.get('actions')}"
            )
        return beats
    except Exception as e:
        print(f"[body_director error] {e}")
        return [{
            "text": "Sorry, I had trouble processing that.",
            "emotion": "apologetic",
            "intensity": 0.6,
            "actions": ["slump"],
            "action_timing": "during",
            "state": "standing",
            "gesture_target": "none",
            "hand": "right",
        }]


_STDIN_Q = None
_STDIN_THREAD = None


def _ensure_stdin_thread():
    """Terminal typing still works while we also poll the Blender sidebar inbox."""
    global _STDIN_Q, _STDIN_THREAD
    import threading
    import queue as _q
    if _STDIN_Q is not None:
        return
    _STDIN_Q = _q.Queue()

    def _reader():
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                _STDIN_Q.put(None)
                break
            _STDIN_Q.put(line)

    _STDIN_THREAD = threading.Thread(target=_reader, daemon=True)
    _STDIN_THREAD.start()


def get_chat_input(llm_provider):
    """Get natural language input from Blender sidebar and/or this terminal."""
    from face_agents.blender_chat_bridge import (
        get_pipeline_mode,
        pop_inbox_record,
        push_outbox,
        set_pipeline_mode,
        ui_is_active,
    )

    mode_now = get_pipeline_mode()
    print("\n" + "=" * 55)
    print("DIRECTOR CHAT — Blender sidebar (N-panel) or this terminal")
    print(f"  mode={mode_now}  (switch: 'mode scene' | 'mode full')")
    print("  Full: motion + speech + scene   |   Scene: Look_Set only (no T2M/speech)")
    print("  Examples:  park  |  scene forest  |  a person runs forward")
    print("Type 'quit' | 'json' for JSON paste mode")
    print("=" * 55)
    push_outbox(
        "sys",
        f"Waiting for chat… mode={mode_now} "
        f"({'scene only' if mode_now == 'scene' else 'full performance'})",
    )
    _ensure_stdin_thread()

    user_input = ""
    inbox_mode = mode_now
    while True:
        rec = pop_inbox_record()
        if rec:
            user_input = str(rec.get("text") or "")
            inbox_mode = str(rec.get("mode") or mode_now)
            print(f"You (blender/{inbox_mode}): {user_input}")
            break
        try:
            line = _STDIN_Q.get(timeout=0.25)
        except Exception:
            line = "__empty__"
        if line is None:
            blender_sidebar = os.environ.get("BLENDER_SIDEBAR", "").strip() in (
                "1", "true", "yes",
            )
            if blender_sidebar or ui_is_active(max_age_s=1800.0):
                continue
            return None
        if line != "__empty__":
            user_input = str(line).strip()
            if user_input:
                inbox_mode = get_pipeline_mode()
                print(f"You: {user_input}")
                break

    if user_input.lower() in ("quit", "exit", "q"):
        return None
    if user_input.lower() == "json":
        return "SWITCH_JSON"
    if not user_input:
        return get_chat_input(llm_provider)

    # Mode switches (sticky until changed)
    low = user_input.strip().lower()
    if low in ("mode scene", "scene mode", "mode look", "look mode"):
        set_pipeline_mode("scene")
        push_outbox("sys", "Mode → SCENE (only updates Look_Set / background)")
        print("[mode] SCENE — only Blender API scene builds (no T2M/speech)")
        return [{
            "text": "__mode_ack__",
            "emotion": "neutral",
            "intensity": 0.5,
            "actions": [],
            "body_mode": "catalog",
            "humanml_prompt": "",
            "allow_sync_gen": False,
            "pipeline_mode": "scene",
            "_mode_switch": "scene",
        }]
    if low in ("mode full", "mode chat", "full mode", "chat mode", "mode performance"):
        set_pipeline_mode("full")
        push_outbox("sys", "Mode → FULL (scene + motion + speech)")
        print("[mode] FULL — scene then T2M/speech")
        return [{
            "text": "__mode_ack__",
            "emotion": "neutral",
            "intensity": 0.5,
            "actions": [],
            "body_mode": "catalog",
            "humanml_prompt": "",
            "allow_sync_gen": False,
            "pipeline_mode": "full",
            "_mode_switch": "full",
        }]

    # Session edits always run even while sticky scene mode is on.
    if _is_session_edit_command(user_input):
        push_outbox("user", user_input)
        return [{
            "text": user_input,
            "emotion": "neutral",
            "intensity": 0.5,
            "actions": [],
            "action_timing": "during",
            "state": "standing",
            "gesture_target": "none",
            "hand": "right",
            "body_mode": "catalog",
            "humanml_prompt": "",
            "allow_sync_gen": False,
            "pipeline_mode": "full",
        }]

    # Sticky scene mode: prefer per-message mode from Blender/terminal.
    effective_mode = str(inbox_mode or get_pipeline_mode()).strip().lower()
    if effective_mode in ("look", "set"):
        effective_mode = "scene"
    if effective_mode == "scene":
        set_pipeline_mode("scene")
        push_outbox("user", f"[scene] {user_input}")
        return [{
            "text": user_input,
            "emotion": "neutral",
            "intensity": 0.5,
            "actions": [],
            "action_timing": "during",
            "state": "standing",
            "gesture_target": "none",
            "hand": "right",
            "body_mode": "catalog",
            "humanml_prompt": "",
            "allow_sync_gen": False,
            "pipeline_mode": "scene",
        }]
    set_pipeline_mode("full")

    push_outbox("user", user_input)
    stripped = (user_input or "").strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            beats = _load_director_json(stripped)
            print(f"  [json] {len(beats)} beat(s) from studio/paste")
            for b in beats:
                if isinstance(b, dict):
                    b["pipeline_mode"] = "full"
            return beats
        except Exception as e:
            print(f"  [json] parse failed ({e}) — treating as chat")
    beats = chat_to_sentences(user_input, llm_provider)
    for b in beats:
        if isinstance(b, dict):
            b["pipeline_mode"] = "full"
    return beats


def _is_session_edit_command(text: str) -> bool:
    """Inpaint / delete / reset — do not run through the body director."""
    from face_agents.session_state import is_reset_command

    t = (text or "").strip()
    if is_reset_command(t):
        return True
    if re.match(r"(?is)^\s*inpaint\s+", t):
        return True
    if re.match(r"(?is)^\s*delete(?:\s+range)?\s+", t):
        return True
    if re.match(r"(?is)^\s*trim\s+", t):
        return True
    return False


def _sanitize_json_text(raw: str) -> str:
    """Fix common paste damage from Word/chat/Windows terminals."""
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")
    # Smart / curly quotes → ASCII (very common cause of "Expecting ',' delimiter")
    for a, b in (
        ("\u201c", '"'), ("\u201d", '"'), ("\u201e", '"'), ("\u201f", '"'),
        ("\u2018", "'"), ("\u2019", "'"), ("\u2032", "'"), ("\u00b4", "'"),
        ("\u2013", "-"), ("\u2014", "-"), ("\u2026", "..."),
    ):
        raw = raw.replace(a, b)
    # Strip markdown fences if user pasted a fenced block
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
    raw = re.sub(r"\s*```\s*$", "", raw.strip())
    raw = raw.strip()
    # Common paste truncations: missing opening brace / quote
    #   sentences":[...]}  →  {"sentences":[...]}
    #   "sentences":[...]} →  {"sentences":[...]}
    #   [{...}]            →  {"sentences":[{...}]}
    if raw.startswith("sentences\""):
        raw = "{\"" + raw
    elif raw.startswith("\"sentences\"") or raw.startswith("'sentences'"):
        raw = "{" + raw
    elif raw.startswith("beats\""):
        raw = "{\"" + raw
    elif raw.startswith("\"beats\""):
        raw = "{" + raw
    elif raw.startswith("["):
        raw = '{"sentences":' + raw + "}"
    # Missing closing brace
    if raw.startswith("{") and raw.count("{") > raw.count("}"):
        raw = raw + ("}" * (raw.count("{") - raw.count("}")))
    return raw.strip()


def _load_director_json(raw: str):
    """Parse director JSON string → cleaned beat list, or raise."""
    raw = _sanitize_json_text(raw)
    data = json.loads(raw)
    items = data.get("beats") or data.get("sentences") or []
    if not isinstance(items, list) or not items:
        raise ValueError("Need a non-empty 'beats' or 'sentences' array.")
    cleaned = _normalize_director_items(items)
    if not cleaned:
        raise ValueError("No valid beats after cleaning.")
    return cleaned


def get_sentences_from_raw_json():
    """Paste full JSON, or load a file with @path / path.json. Blank line to submit."""
    print("\n" + "=" * 55)
    print("JSON MODE — options:")
    print("  1) Type a file path, e.g.  @temp/director_actions_test.json")
    print("  2) Paste FULL JSON (prefer ONE line). Empty line = submit.")
    print("  quit / chat  — exit or switch mode")
    print("=" * 55)

    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return None
        stripped = line.strip()
        if stripped.lower() in ("quit", "exit", "q"):
            return None
        if stripped.lower() == "chat":
            return "SWITCH_CHAT"
        if stripped == "":
            break
        lines.append(line)

    raw = "\n".join(lines).strip()
    if not raw:
        print("No content. Try again.")
        return get_sentences_from_raw_json()

    # File path: @temp/foo.json  or  temp/foo.json  or  absolute path
    path_candidate = raw.strip().lstrip("@").strip().strip('"').strip("'")
    if (
        len(lines) == 1
        and (path_candidate.lower().endswith(".json") or raw.strip().startswith("@"))
    ):
        p = Path(path_candidate)
        if not p.is_file():
            # also try relative to project root
            root = Path(__file__).resolve().parent
            alt = root / path_candidate
            if alt.is_file():
                p = alt
        if p.is_file():
            try:
                cleaned = _load_director_json(p.read_text(encoding="utf-8"))
                print(f"  [director] loaded {len(cleaned)} beat(s) from {p}")
                for i, b in enumerate(cleaned, 1):
                    acts = b.get("actions") or []
                    print(
                        f"    beat {i}: emo={b['emotion']} state={b.get('state')} "
                        f"gesture={b.get('gesture_target')}/{b.get('hand')}"
                        + (f" actions={acts}" if acts else "")
                    )
                return cleaned
            except Exception as e:
                print(f"Failed to load {p}: {e}. Try again.")
                return get_sentences_from_raw_json()
        else:
            print(f"File not found: {path_candidate}")
            return get_sentences_from_raw_json()

    try:
        cleaned = _load_director_json(raw)
        print(f"  [director] {len(cleaned)} beat(s) loaded")
        for i, b in enumerate(cleaned, 1):
            acts = b.get("actions") or []
            print(
                f"    beat {i}: emo={b['emotion']} state={b.get('state')} "
                f"gesture={b.get('gesture_target')}/{b.get('hand')}"
                + (f" actions={acts}" if acts else "")
            )
        return cleaned
    except json.JSONDecodeError as e:
        print(f"Invalid JSON ({e}). Try again.")
        print("  Tip: use  @temp/director_actions_test.json  instead of pasting.")
        return get_sentences_from_raw_json()
    except ValueError as e:
        print(f"{e}. Try again.")
        return get_sentences_from_raw_json()


def _is_humanml_caption(text: str) -> bool:
    """True if text is a body caption (must not be spoken by TTS)."""
    try:
        from face_agents.director_schema import is_motion_caption
        return is_motion_caption(text)
    except Exception:
        t = (text or "").strip().lower()
        if not t:
            return False
        return bool(
            re.match(
                r"^(a\s+person|someone|a\s+man|a\s+woman|a\s+figure|"
                r"the\s+person|the\s+man|the\s+woman)\b",
                t,
            )
        )


def _write_silence_wav(path: str, duration_s: float = 3.0, sr: int = 22050):
    """Silent WAV so motion-only beats still have a body timeline."""
    n = max(1, int(float(duration_s) * int(sr)))
    audio = np.zeros(n, dtype=np.float32)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        from parler_voice import _safe_write_wav
        _safe_write_wav(path, audio, sr)
    except Exception:
        sf.write(path, audio, sr)
    return audio, sr


def tts_to_wav(text: str, emotion: str, intensity: float, path: str):
    """TTS for clean speech only — never pass [token] markers or reaction words to Parler."""
    from face_agents.expression_tokens import (
        strip_tokens, TOKEN_RECIPES, TOKEN_SOUNDS, _resolve_alias,
    )
    clean = strip_tokens(text).strip()
    # Refuse pure reaction words if something slipped through
    low = re.sub(r"[^a-z]", "", clean.lower())
    if low and (_resolve_alias(low) in TOKEN_RECIPES or low in TOKEN_SOUNDS):
        print(f"  [TTS-SKIP] refusing to Parler-speak token word '{clean}' — use vocal clip")
        return _write_silence_wav(path, 0.05)
    # Motion-only / empty / HumanML caption → silence (body uses free MoMask length)
    if not clean or _is_humanml_caption(clean):
        if _is_humanml_caption(clean):
            print(f"  [TTS-SKIP] motion caption not spoken: {clean[:60]!r}")
        else:
            print("  [TTS-SKIP] empty dialogue → silence (motion-only beat)")
        return _write_silence_wav(path, 3.5)

    print(f"  [TTS speech-only] {clean[:70]}{'...' if len(clean) > 70 else ''}")
    from parler_voice import build_voice_style, generate_speech
    style = build_voice_style(emotion, intensity)
    try:
        audio, sr = generate_speech(
            text=clean,
            voice_style=style,
            output_path=path,
            play_audio=False,
        )
        return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)
    except Exception as e:
        print(f"  [TTS] generate/write failed ({e}) — using silence fallback")
        return _write_silence_wav(path, 2.0)


def _attach_and_save_movie_plan(
    *,
    idx: int,
    text: str,
    emotion: str,
    intensity: float,
    body_state: str,
    actions: list,
    body_mode: str,
    humanml_prompt: str,
    audio_path: str,
    duration_s: float,
    ctx,
) -> None:
    """Plan camera for beat, attach to ctx, write temp/movie_timeline_beat_N.json."""
    try:
        from face_agents.movie_timeline import MovieTimeline, MovieBeat
        from face_agents.camera_agent import movie_camera_enabled, plan_camera_for_beat

        if not movie_camera_enabled():
            return
        extras = getattr(ctx, "extras", None) or {}
        cam_dict = None
        if movie_camera_enabled():
            plan = plan_camera_for_beat(
                duration_s=duration_s,
                emotion=emotion,
                intensity=intensity,
                state=body_state,
                actions=list(actions or []),
                fps=FPS,
                shot=extras.get("camera_shot") or None,
                move_type=extras.get("camera_move") or None,
                camera_role=str(extras.get("camera_role") or ""),
                pace=str(extras.get("camera_pace") or "medium"),
                humanml_prompt=humanml_prompt or "",
                session_clip_index=int(extras.get("session_clip_index") or 0),
                last_shot=str(extras.get("last_camera_shot") or ""),
                last_role=str(extras.get("last_camera_role") or ""),
            )
            cam_dict = plan.to_dict()
        beat = MovieBeat(
            id=f"beat_{idx}",
            t0=0.0,
            t1=max(0.1, float(duration_s)),
            text=text,
            emotion=emotion,
            intensity=intensity,
            state=body_state,
            actions=list(actions or []),
            body_mode=body_mode,
            humanml_prompt=humanml_prompt,
            audio_path=audio_path,
            camera=cam_dict,
        )
        if beat.camera and getattr(ctx, "extras", None) is not None:
            ctx.extras["camera_plan"] = beat.camera
        tl = MovieTimeline(fps=FPS, title=f"beat_{idx}", beats=[beat])
        out = TEMP_DIR / f"movie_timeline_beat_{idx}.json"
        tl.save(out)
        cam = beat.camera or {}
        print(
            f"  [movie] timeline → {out.name}  "
            f"camera={cam.get('shot')}/{cam.get('move_type')} "
            f"keys={len(cam.get('keyframes') or [])}"
        )
    except Exception as e:
        print(f"  [movie] timeline save skipped: {e}")


def _resolve_and_send_look(coord, sess, sentence: dict, user_text: str = "") -> dict:
    """
    STEP 1 — Scene before performance.
    LLM/LookAgent plan → UDP type=look → Blender bpy builds structures.
    Returns the look dict applied (also locks sess.locked_look).
    """
    from face_agents.look_agent import plan_look
    from face_agents.look_schema import LookPlan, movie_look_enabled

    look = sentence.get("look") if isinstance(sentence.get("look"), dict) else {}
    prev = getattr(sess, "locked_look", None)
    if not look:
        look = plan_look(
            user_text or str(sentence.get("text") or ""),
            prev_look=prev if isinstance(prev, LookPlan) else None,
        ).to_dict()
    else:
        # Merge free-text inference on top of explicit so "park" always wins
        inferred = plan_look(
            user_text or str(sentence.get("text") or ""),
            prev_look=None,
            explicit=look,
        )
        look = inferred.to_dict()

    plan = LookPlan.from_dict(look)
    sess.locked_look = plan
    try:
        sess.cast.set_hero_outfit(plan.wardrobe_id)
        sess.cast.extras_count = int(plan.extras_count or 0)
        sess.cast.extras_preset = str(plan.extras_preset or "none")
    except Exception:
        pass
    try:
        sess.save()
    except Exception:
        pass

    if movie_look_enabled():
        # Clothes-only: skip full set rebuild when possible
        ut = (user_text or "").strip().lower()
        wear_only = bool(re.match(r"(?is)^\s*(wear|outfit|clothes|clothing)\s+", ut))
        people_only = bool(
            re.match(r"(?is)^\s*(add|spawn)\s+\d+\s+(people|persons|extras)", ut)
            or re.match(r"(?is)^\s*(crowd|clear\s+people|no\s+people|remove\s+extras)\s*$", ut)
        )
        if wear_only and not people_only:
            coord.send_udp({
                "type": "wardrobe",
                "op": "apply",
                "actor": "hero",
                "outfit_id": plan.wardrobe_id,
                "session_id": getattr(sess, "session_id", "") or "",
            })
            print(f"  [wardrobe] hero → {plan.wardrobe_id}")
            time.sleep(0.15)
        elif people_only and not wear_only:
            if int(plan.extras_count or 0) <= 0 or plan.extras_preset == "none":
                coord.send_udp({
                    "type": "cast",
                    "op": "clear_extras",
                    "session_id": getattr(sess, "session_id", "") or "",
                })
                print("  [cast] cleared extras")
            else:
                coord.send_udp({
                    "type": "cast",
                    "op": "spawn_extras",
                    "count": int(plan.extras_count or 0),
                    "preset": plan.extras_preset or "sidewalk",
                    "outfit_id": "casual_01",
                    "session_id": getattr(sess, "session_id", "") or "",
                })
                print(
                    f"  [cast] extras={plan.extras_count} preset={plan.extras_preset}"
                )
            time.sleep(0.2)
        else:
            coord.send_udp({
                "type": "look",
                "op": "apply",
                "session_id": getattr(sess, "session_id", "") or "",
                "look": plan.to_dict(),
            })
            print(
                f"  [look/SCENE] Blender API build "
                f"{plan.location}/{plan.time_of_day}/{plan.set_preset} "
                f"wardrobe={plan.wardrobe_id} extras={plan.extras_count} "
                f"(structures first; T2M/speech after)"
            )
            # Give Blender a beat to clear/rebuild Look_Set before body plays
            time.sleep(0.35)
    return plan.to_dict()


def _is_scene_only_command(text: str) -> bool:
    """User only wants a set built — no speech / no new motion clip."""
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()
    # Explicit: "scene park", "set forest", "background street", "look beach"
    if re.match(
        r"(?is)^\s*(scene|set|background|backdrop|look|location|environment)\s+[\w\s\-]+$",
        t,
    ):
        return True
    # Bare location words only
    if re.match(
        r"(?is)^\s*(studio|park|forest|woods|playground|street|city|"
        r"station|platform|beach|office|room|interior|corridor|stage|"
        r"garden|outdoors?)\s*$",
        t,
    ):
        return True
    # Wardrobe / crowd only (LookAgent — not BodyDirector)
    if re.match(
        r"(?is)^\s*(wear|outfit|clothes|clothing)\s+[\w\s\-]+$",
        t,
    ):
        return True
    if re.match(
        r"(?is)^\s*(add|spawn)\s+\d+\s+(people|persons|extras|npcs|pedestrians)\s*$",
        t,
    ):
        return True
    if re.match(
        r"(?is)^\s*(crowd|clear\s+people|no\s+people|remove\s+extras)\s*$",
        t,
    ):
        return True
    return False


def _is_wardrobe_only_command(text: str) -> bool:
    t = (text or "").strip().lower()
    return bool(re.match(r"(?is)^\s*(wear|outfit|clothes|clothing)\s+[\w\s\-]+$", t))


def process_sentence(
    coord: FaceCoordinator,
    sentence: dict,
    idx: int,
    session=None,
) -> None:
    """
    Order of work (product):
      1) SCENE — LookAgent → Blender bpy builds Look_Set (any kit)
      2) SPEECH — Parler TTS (if dialogue)
      3) MOTION — catalog or MoMask T2M
      4) PLAY — face + body + camera on the prepared set

    Segment-aware speech:
      speech text → Parler TTS + lip sync
      [token]     → pre-recorded vocal clip + expression (NOT Parler)
    """
    from face_agents.expression_tokens import split_token_segments, strip_tokens
    from face_agents.session_state import get_session, is_reset_command

    text = sentence["text"]
    emotion = sentence["emotion"]
    intensity = sentence["intensity"]
    actions = sentence.get("actions") or []
    action_timing = sentence.get("action_timing") or "during"
    body_state = sentence.get("state") or "standing"
    gesture_target = sentence.get("gesture_target") or "none"
    hand = sentence.get("hand") or "right"
    body_mode = str(sentence.get("body_mode") or "auto")
    humanml_prompt = str(sentence.get("humanml_prompt") or "")
    allow_sync_gen = bool(sentence.get("allow_sync_gen"))
    motion_reason = str(sentence.get("motion_reason") or "")

    # Session timeline (append clips across chat turns)
    sess = session if session is not None else get_session()

    # Sticky mode ack from get_chat_input ("mode scene" / "mode full") — no pipeline work.
    if sentence.get("_mode_switch") or (text or "").strip() == "__mode_ack__":
        return

    # Sticky SCENE mode: every chat line only rebuilds Look_Set (no T2M / speech).
    pipe_mode = str(sentence.get("pipeline_mode") or "").strip().lower()
    if pipe_mode in ("scene", "look", "set"):
        print(f"\n{'=' * 50}")
        print(f"[SCENE MODE] {text}")
        print(f"{'=' * 50}")
        look = _resolve_and_send_look(coord, sess, sentence, user_text=text)
        try:
            from face_agents.blender_chat_bridge import push_outbox
            push_outbox(
                "sys",
                f"Scene updated: {look.get('location')}/{look.get('set_preset')} "
                f"(scene mode — no motion/speech). Switch to Full for performance.",
            )
        except Exception:
            pass
        return

    low_cmd = (text or "").strip().lower()
    if low_cmd in ("preview", "export preview"):
        from face_agents.blender_chat_bridge import push_command, push_outbox
        push_command("export_preview")
        push_outbox("sys", "Preview MP4 queued in Blender")
        return
    if low_cmd in ("export movie", "export mp4", "export"):
        from face_agents.blender_chat_bridge import push_command, push_outbox
        push_command("export_movie")
        push_outbox("sys", "Export queued in Blender (movie if available, else preview)")
        return
    _trim = re.match(
        r"(?is)^\s*trim\s+(\d+(?:\.\d+)?)(s)?\s*[-:]\s*(\d+(?:\.\d+)?)(s)?\s*$",
        text or "",
    )
    if _trim:
        from face_agents.blender_chat_bridge import push_command, push_outbox
        a, a_sec, b, b_sec = _trim.groups()
        af, bf = float(a), float(b)
        if a_sec or b_sec:
            af, bf = af * float(sess.fps or 20), bf * float(sess.fps or 20)
        push_command("trim", start=int(min(af, bf)), end=int(max(af, bf)))
        push_outbox("sys", f"Trim {int(min(af,bf))}-{int(max(af,bf))} queued (speech-safe)")
        return

    # Speech-safe delete: "delete 40-80" or "delete 1.0s-2.5s"
    _del = re.match(
        r"(?is)^\s*delete(?:\s+range)?\s+(\d+(?:\.\d+)?)(s)?\s*[-:]\s*(\d+(?:\.\d+)?)(s)?\s*$",
        text or "",
    )
    if _del:
        a, a_sec, b, b_sec = _del.groups()
        af, bf = float(a), float(b)
        if a_sec or b_sec:
            af, bf = af * sess.fps, bf * sess.fps
        f0, f1 = int(min(af, bf)), int(max(af, bf))
        print(f"\n[delete] frames {f0}-{f1} (speech-safe whole clips)")
        removed = sess.remove_clips_overlapping(f0, f1)
        sess.needs_movie_rerender = True
        sess.save()
        coord.send_udp({
            "type": "body",
            "op": "session_delete_range",
            "frame_start": f0,
            "frame_end": f1,
            "session_id": sess.session_id,
        })
        print(f"  [delete] removed {len(removed)} clip(s): {[c.index for c in removed]}")
        return

    # MoMask inpaint: "inpaint 40-80 a person waves" or "inpaint 1.0s-2.5s ..."
    _inp = re.match(
        r"(?is)^\s*inpaint\s+(\d+(?:\.\d+)?)(s)?\s*[-:]\s*(\d+(?:\.\d+)?)(s)?\s+(.+)$",
        text or "",
    )
    if _inp:
        a, a_sec, b, b_sec, cap = _inp.groups()
        af = float(a)
        bf = float(b)
        if a_sec or b_sec:
            af, bf = af * 20.0, bf * 20.0
        print(f"\n[inpaint] frames {int(af)}-{int(bf)} prompt={cap.strip()!r}")
        try:
            from face_agents.momask_body_pipeline import (
                inpaint_body_action,
                ensure_action_in_blender_via_packet_hint,
                to_humanml_caption,
            )
            hml = to_humanml_caption(cap.strip())
            res = inpaint_body_action(hml, start_frame=int(af), end_frame=int(bf))
            if not res.ok:
                print(f"  [inpaint] FAILED: {res.error}")
                return
            hint = ensure_action_in_blender_via_packet_hint(res, speech_duration_s=0.0)
            print(f"  [inpaint] OK action={res.action_name} — play next as a new clip")
            # v1 two-step: speech-safe delete overlapping take(s), then new body-only clip
            try:
                from face_agents.speech_safe import snap_range_speech_safe
                clips_d = [c.to_dict() for c in sess.clips]
                f0, f1, note = snap_range_speech_safe(clips_d, int(af), int(bf))
                if note:
                    print(f"  [inpaint] {note}")
                removed = sess.remove_clips_overlapping(f0, f1)
                sess.needs_movie_rerender = True
                sess.save()
                coord.send_udp({
                    "type": "body",
                    "op": "session_delete_range",
                    "frame_start": f0,
                    "frame_end": f1,
                    "session_id": sess.session_id,
                })
                print(f"  [inpaint] deleted {len(removed)} overlapping clip(s)")
            except Exception as e:
                print(f"  [inpaint] delete-range skip: {e}")
            # Treat as a body-only beat
            sentence = dict(sentence)
            sentence["text"] = hml
            sentence["body_mode"] = "momask"
            sentence["humanml_prompt"] = hml
            sentence["allow_sync_gen"] = False
            sentence["_inpaint_action"] = (hint or {}).get("action") or res.action_name
            sentence["_inpaint_meta"] = hint or {}
        except Exception as e:
            print(f"  [inpaint] error: {e}")
            return

    if is_reset_command(text):
        sess.reset()
        sess.save()
        coord.send_udp({
            "type": "body",
            "session_reset": True,
            "rest": True,
            "session_id": sess.session_id,
            "session_name": sess.session_id,
        })
        # Fresh session → default studio set via Blender API
        _resolve_and_send_look(
            coord, sess,
            {"look": {"location": "studio", "time_of_day": "day", "set_preset": "studio_cyc"}},
            user_text="studio",
        )
        print(f"\n[session] RESET → new timeline {sess.session_id} + studio Look_Set")
        return

    # Scene-only: "park" / "scene forest" / "set street" — build set, skip T2M/speech
    if _is_scene_only_command(text):
        print(f"\n{'=' * 50}")
        print(f"[SCENE ONLY] {text}")
        print(f"{'=' * 50}")
        look = _resolve_and_send_look(coord, sess, sentence, user_text=text)
        try:
            from face_agents.blender_chat_bridge import push_outbox
            push_outbox(
                "sys",
                f"Scene ready: {look.get('location')}/{look.get('set_preset')} "
                f"(Blender API). Say a motion/line next for T2M+speech.",
            )
        except Exception:
            pass
        return

    try:
        from face_agents.policy_bridge import get_policy
        pol = get_policy()
        if pol.get("notes") or pol.get("mouth_gain", 1.0) != 1.0 or pol.get("brow_scale", 1.0) != 1.0:
            emotion = pol.get("emotion", emotion)
            intensity = float(pol.get("intensity", intensity))
            print(f"  [policy] emotion={emotion} intensity={intensity:.2f}")
    except Exception:
        pass

    print(f"\n{'=' * 50}")
    print(f"[Director Beat {idx}] {text}")
    print(
        f"  session={sess.session_id} clip_next=#{sess.clip_count + 1} "
        f"t_end={sess.t_end_s:.2f}s frames→{sess.frame_cursor}"
    )
    print(f"  emotion={emotion} intensity={intensity:.2f} brain={coord.use_brain}")
    print(f"  body state={body_state} gesture={gesture_target}/{hand}")
    print(f"  body_mode={body_mode} allow_sync_gen={allow_sync_gen}")
    if motion_reason:
        print(f"  motion_reason={motion_reason}")
    if humanml_prompt:
        print(f"  humanml_prompt={humanml_prompt[:80]}{'…' if len(humanml_prompt) > 80 else ''}")
    if actions:
        print(f"  actions={actions} timing={action_timing}")
    print(f"{'=' * 50}")

    # STEP 1 — SCENE (Blender API) before MoMask / Parler
    look_applied = _resolve_and_send_look(coord, sess, sentence, user_text=text)
    sentence["look"] = look_applied

    segments = split_token_segments(text)
    has_tokens = any(s.kind == "token" for s in segments)

    if not has_tokens:
        # Perfect speech↔motion sync (AFTER scene is up):
        #   1) TTS → exact duration
        #   2) face bake || MoMask(gen length = speech @ 20fps)
        #   3) play with speed so rest→motion→rest finishes with audio
        from concurrent.futures import ThreadPoolExecutor

        clean = strip_tokens(text) or text
        wav_path = str(TEMP_DIR / f"agents_sentence_{idx}.wav")
        # Director/timeline may override length (seconds or MoMask frames)
        dir_dur = sentence.get("motion_duration_s") or sentence.get("duration_s")
        dir_ml = sentence.get("motion_length") or sentence.get("momask_frames")

        t0 = time.time()
        spoken = (strip_tokens(text) or "").strip()
        motion_only = (not spoken) or _is_humanml_caption(spoken)
        try:
            body_ml = int(dir_ml) if dir_ml not in (None, "", 0) else 0
        except (TypeError, ValueError):
            body_ml = 0
        # Start MoMask after scene. Director length or 4s.
        try:
            body_dur = float(dir_dur) if dir_dur not in (None, "", 0, 0.0) else 4.0
        except (TypeError, ValueError):
            body_dur = 4.0

        def _body_job():
            if sentence.get("_inpaint_meta"):
                return sentence.get("_inpaint_meta")
            return _maybe_run_momask_body(
                body_mode=body_mode,
                humanml_prompt=humanml_prompt,
                body_state=body_state,
                actions=actions,
                gesture_target=gesture_target,
                hand=hand,
                emotion=emotion,
                text=clean,
                duration_s=body_dur,
                seed=idx * 101 + 7,
                allow_sync_gen=allow_sync_gen,
                motion_length=body_ml,
            )

        def _face_job(audio_sr_dur):
            _audio, _sr, _dur = audio_sr_dur
            return coord.prepare_sentence(
                text=clean,
                emotion=emotion,
                intensity=intensity,
                audio_path=wav_path,
                duration=_dur,
                sample_rate=_sr,
                body_actions=actions,
                action_timing=action_timing,
                body_state=body_state,
                gesture_target=gesture_target,
                hand=hand,
                body_mode=body_mode,
                humanml_prompt=humanml_prompt,
                momask_action="",  # filled after join
                momask_library="",
                camera_shot=str(sentence.get("camera_shot") or sentence.get("shot") or ""),
                camera_move=str(sentence.get("camera_move") or sentence.get("move") or ""),
            )

        t1 = time.time()
        if motion_only:
            print(f"  [body-only] skip face bake / Parler  MoMask {body_dur:.2f}s")
            audio, sr = tts_to_wav(clean, emotion, intensity, wav_path)
            duration = len(audio) / float(sr)
            momask_meta = _body_job()
            from face_agents.base import FaceContext
            ctx = FaceContext(
                t=0.0,
                duration=float(duration),
                is_speaking=False,
                emotion=emotion,
                intensity=intensity,
                text=clean,
                audio_path=wav_path,
                sample_rate=int(sr),
            )
            ctx.energy_envelope = np.zeros(1, dtype=np.float32)
        else:
            # Speech TTS and MoMask gen at the same time; face bake after WAV exists.
            with ThreadPoolExecutor(max_workers=3) as ex:
                print(
                    f"  [parallel] TTS || MoMask({body_dur:.2f}s) → then face bake"
                )
                fut_body = ex.submit(_body_job)
                fut_tts = ex.submit(tts_to_wav, clean, emotion, intensity, wav_path)
                audio, sr = fut_tts.result()
                duration = len(audio) / float(sr)
                print(
                    f"  [TTS] {duration:.2f}s @ {sr} Hz  ({time.time()-t0:.2f}s) "
                    f"→ MoMask body_dur={body_dur:.2f}s frames={body_ml or 'auto'}"
                )
                fut_face = ex.submit(_face_job, (audio, sr, duration))
                momask_meta = fut_body.result()
                ctx = fut_face.result()
        print(f"  [prep] face||body wall={time.time()-t1:.2f}s total={time.time()-t0:.2f}s")

        # Attach MoMask result: one-shot; duration may outlast short speech
        if momask_meta:
            act = momask_meta.get("action") or ""
            lib = momask_meta.get("library_blend") or ""
            if act:
                ctx.extras["momask_action"] = act
                ctx.extras["momask_library"] = lib
                sp = float(momask_meta.get("speed") or 1.0)
                cfp = float(momask_meta.get("clip_fps") or 20.0)
                sdelay = float(momask_meta.get("speech_delay_s") or (1 + 6) / 20.0)
                body_play = float(
                    momask_meta.get("duration")
                    or momask_meta.get("body_play_s")
                    or 0.0
                )
                act_frames = int(momask_meta.get("action_frames") or 0)
                ctx.extras["motion_plan"] = {
                    "clip_id": act,
                    "action_name": act,
                    "engine": momask_meta.get("engine") or "momask",
                    "loop": False,
                    "clip_policy": "momask_match",
                    "library_blend": lib,
                    "speed": sp,
                    "clip_fps": cfp,
                    "speech_delay_s": sdelay,
                    "duration": body_play if body_play > 0.05 else None,
                    "body_play_s": body_play if body_play > 0.05 else None,
                    "action_frames": act_frames,
                    "amp": 1.0,
                }
                ctx.extras["speech_delay_s"] = sdelay
                ctx.extras["body_play_s"] = body_play
                print(
                    f"  [body] action={act} speech_delay={sdelay:.2f}s "
                    f"body_play={body_play:.2f}s frames={act_frames} "
                    f"(rest→motion first, then speech) clip_fps={cfp:.0f}"
                )

        # Session: NLA append in Blender (cursor 133→134). Live play = this clip only.
        ctx.extras["body_state"] = body_state
        ctx.extras["body_actions"] = actions
        ctx.extras["humanml_prompt"] = humanml_prompt
        ctx.extras["append_timeline"] = True
        # Next clip starts where previous rest left the character (no teleport to origin)
        ctx.extras["continue_root"] = True
        ctx.extras["session_clip_index"] = sess.clip_count
        ctx.extras["face_clear_previous"] = sess.clip_count == 0
        ctx.extras["session_id"] = sess.session_id
        ctx.extras["session_name"] = sess.session_id
        ctx.extras["session_frame_start"] = int(sess.frame_cursor or 1)
        ctx.extras["session_frame_cursor"] = int(sess.frame_cursor or 1)
        ctx.extras["camera_session_frame_start"] = int(sess.frame_cursor or 1)
        ctx.extras["clip_label"] = humanml_prompt or clean or (actions[0] if actions else body_state)
        ctx.extras["audio_path"] = wav_path
        ctx.extras["speech_duration_s"] = float(duration)
        ctx.extras["text"] = clean
        ctx.extras["last_camera_shot"] = sess.last_camera_shot
        ctx.extras["last_camera_role"] = sess.last_camera_role or ""
        ctx.extras["subject_relative"] = True
        ctx.extras["camera_pace"] = "medium"
        if sentence.get("camera_shot"):
            ctx.extras["camera_shot"] = sentence.get("camera_shot")
        if sentence.get("camera_move"):
            ctx.extras["camera_move"] = sentence.get("camera_move")
        if sentence.get("camera_role"):
            ctx.extras["camera_role"] = sentence.get("camera_role")

        # Look already applied in STEP 1 (scene before T2M/speech)
        look = sentence.get("look") if isinstance(sentence.get("look"), dict) else {}
        if not look:
            try:
                look = sess.locked_look.to_dict() if getattr(sess, "locked_look", None) else {}
            except Exception:
                look = {}
        ctx.extras["look"] = look or {}
        prev_loc = ""
        try:
            prev_loc = str((sess.locked_look.location if sess.locked_look else "") or "")
        except Exception:
            prev_loc = ""
        new_loc = str((look or {}).get("location") or prev_loc or "studio")
        loc_changed = bool(prev_loc) and new_loc != prev_loc
        ctx.extras["look_location_changed"] = loc_changed or (sess.clip_count == 0)
        asked_pol = str(sentence.get("clip_policy") or "").lower()
        if asked_pol not in ("hold_end", "momask_match", "loop"):
            asked_pol = ""
        ctx.extras["clip_policy"] = asked_pol

        cam_dur = float(duration)
        try:
            bp = float(ctx.extras.get("body_play_s") or 0.0)
            if bp > cam_dur:
                cam_dur = bp
        except (TypeError, ValueError):
            pass
        ctx.extras["camera_duration_s"] = cam_dur
        _attach_and_save_movie_plan(
            idx=idx,
            text=clean,
            emotion=emotion,
            intensity=intensity,
            body_state=body_state,
            actions=actions,
            body_mode=body_mode,
            humanml_prompt=humanml_prompt,
            audio_path=wav_path,
            duration_s=cam_dur,
            ctx=ctx,
        )
        coord.play_sentence(ctx, wav_path, sr, audio_data=audio)

        # Record clip on session timeline (10 chats → 10 clips)
        try:
            plan = ctx.extras.get("motion_plan") or {}
            cam = ctx.extras.get("camera_plan") or {}
            act_name = (
                ctx.extras.get("momask_action")
                or plan.get("action_name")
                or plan.get("clip_id")
                or (actions[0] if actions else body_state)
                or "idle"
            )
            body_play = float(
                ctx.extras.get("body_play_s")
                or plan.get("duration")
                or (duration + float(ctx.extras.get("speech_delay_s") or 0.0))
            )
            eng = str(plan.get("engine") or ("momask" if ctx.extras.get("momask_action") else "catalog"))
            pol = str(plan.get("clip_policy") or ctx.extras.get("clip_policy") or "")
            clip = sess.append_clip(
                action_name=str(act_name),
                duration_s=body_play,
                prompt=humanml_prompt or clean,
                engine=eng,
                camera_shot=str(cam.get("shot") or ctx.extras.get("camera_shot") or sess.last_camera_shot),
                camera_move=str(cam.get("move_type") or ctx.extras.get("camera_move") or ""),
                camera_role=str(cam.get("camera_role") or ctx.extras.get("camera_role") or sess.last_camera_role or "A_cam"),
                camera_anchor=str(cam.get("subject_anchor") or ctx.extras.get("camera_anchor") or "chest"),
                text=clean,
                emotion=emotion,
                body_state=body_state,
                action_frames=int(plan.get("action_frames") or 0),
                clip_fps=float(plan.get("clip_fps") or 20.0),
                audio_path=str(wav_path or ""),
                speech_delay_s=float(ctx.extras.get("speech_delay_s") or plan.get("speech_delay_s") or 0.0),
                speech_duration_s=float(duration),
                look=ctx.extras.get("look") or {},
                clip_policy=pol,
                motion_only=bool(sentence.get("motion_only")) or (float(duration) <= 0.04),
                face_timeline_path=str(ctx.extras.get("face_timeline_path") or ""),
            )
            sess.save()
            print(
                f"  [session] appended clip#{clip.index} label={clip.label!r} "
                f"action={clip.action_name!r} "
                f"t={clip.t0:.2f}-{clip.t1:.2f}s frames={clip.frame_start}-{clip.frame_end} "
                f"speech=f{clip.speech_frame_start}-{clip.speech_frame_end} ({clip.speech_duration_s:.2f}s) "
                f"cam={clip.camera_shot}/{clip.camera_move} anchor={clip.camera_anchor} "
                f"(total clips={sess.clip_count})"
            )
        except Exception as e:
            print(f"  [session] append failed: {e}")
        return

    # Segment path: speech / token / speech / ...
    seg_desc = []
    for s in segments:
        if s.kind == "token":
            seg_desc.append(f"[{s.value}]")
        else:
            seg_desc.append(f'"{s.value[:28]}{"…" if len(s.value) > 28 else ""}"')
    print(f"  [segments] {' → '.join(seg_desc)}")

    part_i = 0
    for seg in segments:
        if seg.kind == "token":
            print(f"  >> TOKEN [{seg.value}] = pre-recorded vocal + lips + expression (NO Parler)")
            coord.play_token_reaction(
                seg.value,
                base_emotion=emotion,
                intensity=intensity,
            )
            continue

        # Speech segment only
        part_i += 1
        speech = seg.value.strip()
        if not speech:
            continue
        # Double-check: never TTS a bare token word
        from face_agents.expression_tokens import TOKEN_RECIPES, TOKEN_SOUNDS, _resolve_alias
        bare = re.sub(r"[^a-z]", "", speech.lower())
        if bare and (_resolve_alias(bare) in TOKEN_RECIPES or bare in TOKEN_SOUNDS) and len(speech.split()) <= 2:
            print(f"  >> SPEECH skip (looks like token word '{speech}') → treat as token")
            coord.play_token_reaction(bare, base_emotion=emotion, intensity=intensity)
            continue

        wav_path = str(TEMP_DIR / f"agents_sentence_{idx}_p{part_i}.wav")
        print(f"  >> SPEECH part {part_i}: {speech[:60]}{'…' if len(speech) > 60 else ''}")
        audio, sr = tts_to_wav(speech, emotion, intensity, wav_path)
        if len(audio) < int(0.08 * sr):
            print(f"  [TTS] empty/stub — skip part {part_i}")
            continue
        duration = len(audio) / float(sr)
        print(f"  [TTS] part {part_i}: {duration:.2f}s @ {sr} Hz")
        momask_meta = None
        if part_i == 1:
            momask_meta = _maybe_run_momask_body(
                body_mode=body_mode,
                humanml_prompt=humanml_prompt,
                body_state=body_state,
                actions=actions,
                gesture_target=gesture_target,
                hand=hand,
                emotion=emotion,
                text=speech,
                duration_s=duration,
                seed=idx * 101 + 7,
            )
        ctx = coord.prepare_sentence(
            text=speech,  # clean speech only — no [tokens]
            emotion=emotion,
            intensity=intensity,
            audio_path=wav_path,
            duration=duration,
            sample_rate=sr,
            body_actions=actions if part_i == 1 else [],
            action_timing=action_timing,
            body_state=body_state,
            gesture_target=gesture_target if part_i == 1 else "none",
            hand=hand,
            body_mode=body_mode if part_i == 1 else "catalog",
            humanml_prompt=humanml_prompt if part_i == 1 else "",
            momask_action=(momask_meta or {}).get("action") or "",
            momask_library=(momask_meta or {}).get("library_blend") or "",
            camera_shot=str(sentence.get("camera_shot") or sentence.get("shot") or ""),
            camera_move=str(sentence.get("camera_move") or sentence.get("move") or ""),
        )
        if part_i == 1:
            _attach_and_save_movie_plan(
                idx=idx,
                text=speech,
                emotion=emotion,
                intensity=intensity,
                body_state=body_state,
                actions=actions,
                body_mode=body_mode,
                humanml_prompt=humanml_prompt,
                audio_path=wav_path,
                duration_s=duration,
                ctx=ctx,
            )
        # Ensure prepare does not re-introduce token mixing (speech has no markers)
        coord.play_sentence(ctx, wav_path, sr, audio_data=audio)


def main() -> None:
    print("=" * 60)
    print("ORCHESTRATOR_AGENTS — optimized bake-then-play")
    print("  face: lips=wav2arkit || Brain(A2E) in parallel after TTS")
    print("  body: catalog by default; MoMask only walk/dance + cache (no 60s block)")
    print("  camera: MovieCam (USE_MOVIE_CAMERA=1) | face timeline scrub (USE_FACE_TIMELINE=1)")
    print(
        f"  USE_BRAIN={USE_BRAIN}  USE_MOMASK={os.environ.get('USE_MOMASK', '1')}  "
        f"MOMASK_ALL={os.environ.get('MOMASK_ALL', '0')}  "
        f"MOMASK_SYNC={os.environ.get('MOMASK_SYNC', '0')}  "
        f"USE_MOVIE_CAMERA={os.environ.get('USE_MOVIE_CAMERA', '1')}  device={DEVICE}"
    )
    print("=" * 60)
    print(">>> Blender: blender_receiver.py → stream_receiver")
    print(">>> Live body: catalog talk (fast). Walk uses MoMask cache only unless MOMASK_SYNC=1")
    print(">>> Force all-MoMask (slow): MOMASK_ALL=1 MOMASK_SYNC=1")
    print("=" * 60)

    if DEVICE == "cuda":
        try:
            free, total = torch.cuda.mem_get_info()
            print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM free {free/1e9:.2f}/{total/1e9:.2f} GB")
        except Exception:
            pass

    print(">>> Starting (no wait). Ensure Blender stream_receiver is already running.")
    print(">>> Parler-TTS loads on first spoken line (motion-only captions skip it).")
    print()

    try:
        wav2arkit.load_session()
        print("[OK] wav2arkit ready (lips agent)")
    except Exception as e:
        print(f"[WARN] wav2arkit: {e} — speech lips disabled until fixed")

    if USE_BRAIN:
        try:
            import brain_inference
            brain_inference.load_brain_model(DEVICE)
            print("[OK] Brain encoder ready (expression agents)")
        except Exception as e:
            print(f"[WARN] Brain load failed ({e}) — agents use emotion_map only")

    coord = FaceCoordinator(
        udp_ip=UDP_IP,
        udp_port=UDP_PORT,
        fps=FPS,
        use_brain=USE_BRAIN,
        device=DEVICE,
    )
    print("[OK] FaceCoordinator online")
    fb = FeedbackLogger()
    COLLECT_FEEDBACK = os.environ.get("FACE_FEEDBACK", "0").strip() in ("1", "true", "yes")
    print("     lips → mouth | eyes → blink/gaze/brain | brows/cheeks → brain+preset")
    print("     head → nod/breath/energy")

    import emotion_map
    coord.send_udp({
        "type": "rest_pose",
        "smooth": False,
        "blendshapes": emotion_map.NEUTRAL_REST.copy(),
    })

    # Director chat mode: natural language → state + gesture + dialogue
    # Works with LLM if configured; otherwise rule-based BodyDirector still runs.
    llm_provider = None
    use_chat_mode = LLM_MODE
    llm_provider = get_llm_provider()
    if use_chat_mode:
        if llm_provider:
            print(f"[OK] Director chat mode (LLM: {llm_provider.model})")
        else:
            print("[OK] Director chat mode (rules only — set LLM_CHAT=1 + llm_fw config for LLM)")
    else:
        print("[INFO] JSON mode default. Type 'chat' for natural language director.")
        if llm_provider:
            print(f"       LLM available ({llm_provider.model}) when you switch to chat.")

    from face_agents.session_state import get_session, reset_session

    session = get_session(reset=True)
    coord.send_udp({
        "type": "body",
        "session_begin": True,
        "session_id": session.session_id,
        "session_name": session.session_id,
    })
    print(
        f"[OK] Session timeline created id={session.session_id}\n"
        f"     Blender Action Session_{session.session_id[:40]} is empty until first clip.\n"
        f"     Each chat turn APPENDs a labeled range (continuation from current pose).\n"
        f"     Type 'reset' / 'new scene' to clear timeline & root."
    )

    n = 0
    while True:
        if use_chat_mode:
            sentences = get_chat_input(llm_provider)  # None provider → rule director
            if sentences == "SWITCH_JSON":
                use_chat_mode = False
                print("\n[Switched to JSON mode]")
                continue
        else:
            sentences = get_sentences_from_raw_json()
            if sentences == "SWITCH_CHAT":
                use_chat_mode = True
                llm_provider = llm_provider or get_llm_provider()
                if llm_provider:
                    print(f"\n[Director chat — LLM {llm_provider.model}]")
                else:
                    print("\n[Director chat — rule-based (no LLM). Examples: say hi while sitting]")
                continue

        if sentences is None:
            print("Goodbye.")
            break
        
        for sent in sentences:
            n += 1
            process_sentence(coord, sent, n, session=session)
            try:
                from face_agents.blender_chat_bridge import push_outbox
                import json as _json
                mode = sent.get("body_mode") or sent.get("emotion") or "ok"
                push_outbox(
                    "assistant",
                    f"clip #{n}: {sent.get('text', '')[:80]}  ({mode})",
                    kind="chat",
                )
                dump = {
                    "clip": n,
                    "text": sent.get("text") or "",
                    "body_mode": sent.get("body_mode") or "",
                    "emotion": sent.get("emotion") or "",
                    "action": sent.get("action") or sent.get("clip_id") or "",
                    "humanml_prompt": sent.get("humanml_prompt") or "",
                    "state": sent.get("state") or "",
                }
                push_outbox(
                    "json",
                    _json.dumps(dump, ensure_ascii=False),
                    kind="json",
                    payload=_json.dumps(sent, ensure_ascii=False, default=str),
                )
            except Exception:
                pass
            if COLLECT_FEEDBACK:
                fb.collect(
                    emotion=sent.get("emotion", "neutral"),
                    intensity=float(sent.get("intensity", 0.9)),
                    text=sent.get("text", ""),
                    audio_path=str(TEMP_DIR / f"agents_sentence_{n}.wav"),
                )
        time.sleep(0.15)


if __name__ == "__main__":
    main()
