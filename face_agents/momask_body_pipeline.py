"""
MoMask body pipeline — open-text full-body motion for the orchestrator.

Face pipeline stays independent (lipsync + expressions).
This module only produces a SMPL-X Action name for body playback.

Flow (official + our working retarget):
  HumanML3D caption
    → MoMask gen_t2m.py  (joints + *_ik.bvh)
    → KeeMap BVH→Mixamo  (tools/momask_keemap_exact.py)
    → Mixamo→SMPL-X      (tools/retarget_keemap_mixamo_rsl.py + working map)
    → Action on library blend  →  blender_receiver plays via UDP type=body

Env:
  USE_MOMASK=1          enable MoMask for open body (default off until ready)
  MOMASK_FORCE=1        always momask even for simple gestures
  MOMASK_PYTHON=path    python with momask deps (default third_party/momask_venv)
  BLENDER_EXE=path      Blender 5.x executable
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
MOMASK_CODES = ROOT / "third_party" / "momask-codes"
DEFAULT_VENV_PY = ROOT / "third_party" / "momask_venv" / "Scripts" / "python.exe"
CACHE_DIR = ROOT / "body_motion" / "momask_cache"
LIBRARY_BLEND = ROOT / "body_motion" / "momask_actions_library.blend"
REPORT_DIR = ROOT / "body_motion" / "momask_cache"
# Rokoko custom-names map (BVH → SMPL-X). Prefer user's modified mapping.
OUR_BVH_ROKOKO_MAP = ROOT / "our modified bhv mapping to smplx.json"
HML3D_TO_SMPL_MAP = ROOT / "final hml3dto smpl.json"
WORKING_MAP_FALLBACK = ROOT / "working maximo to smplx.json"
BASE_BLEND = ROOT / "whole_body_retargeted.blend"
# Armature-only host for bake (no mesh/textures/Rokoko). Built on first use.
RETARGET_HOST = ROOT / "body_motion" / "retarget_host.blend"
# Direct BVH → SMPL-X (Rokoko-style bake, NO KeeMap / NO Mixamo)
DIRECT_BVH_SCRIPT = ROOT / "tools" / "bvh_rokoko_direct_to_smplx.py"


def rokoko_map_path() -> Path:
    """Prefer our modified BVH→SMPL-X Rokoko map (manual path that works)."""
    if OUR_BVH_ROKOKO_MAP.is_file():
        return OUR_BVH_ROKOKO_MAP
    if HML3D_TO_SMPL_MAP.is_file():
        return HML3D_TO_SMPL_MAP
    if WORKING_MAP_FALLBACK.is_file():
        return WORKING_MAP_FALLBACK
    return ROOT / "body_motion" / "FINAL_BONE_MAP.json"


def _log(msg: str) -> None:
    print(f"[momask_body] {msg}", flush=True)


def momask_enabled() -> bool:
    # Testing default: ON (set USE_MOMASK=0 to force catalog-only)
    return os.environ.get("USE_MOMASK", "1").strip().lower() in ("1", "true", "yes", "on")


def blender_exe() -> str:
    env = os.environ.get("BLENDER_EXE", "").strip()
    if env and Path(env).is_file():
        return env
    candidates = [
        r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    ]
    for c in candidates:
        if Path(c).is_file():
            return c
    return "blender"


def momask_python() -> str:
    env = os.environ.get("MOMASK_PYTHON", "").strip()
    if env and Path(env).is_file():
        return env
    if DEFAULT_VENV_PY.is_file():
        return str(DEFAULT_VENV_PY)
    return sys.executable


@dataclass
class MomaskBodyResult:
    ok: bool
    action_name: str = ""
    engine: str = "momask"
    prompt: str = ""
    bvh_path: str = ""
    blend_path: str = ""
    frames: List[int] = field(default_factory=list)
    cached: bool = False
    error: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── HumanML3D-style prompt builder (from director menus if needed) ─────────

def build_humanml_prompt(
    *,
    humanml_prompt: str = "",
    state: str = "standing",
    actions: Optional[List[str]] = None,
    gesture_target: str = "none",
    hand: str = "right",
    emotion: str = "neutral",
    text: str = "",
) -> str:
    """
    Prefer explicit HumanML3D caption from director.
    Else rewrite menus into a caption MoMask was trained on.
    Never send bare tags like 'walk' alone.
    """
    explicit = (humanml_prompt or "").strip()
    if len(explicit) >= 12 and " " in explicit:
        return to_humanml_caption(explicit, emotion=emotion, state=state)

    parts: List[str] = ["a person"]
    st = (state or "standing").lower()
    acts = [str(a).lower() for a in (actions or [])]

    loco = {
        "walking": "walks forward",
        "sitting": "is sitting",
        "dancing": "is dancing",
        "standing": "stands",
        "locomotion": "moves through the space",
        "grounded": "moves close to the floor",
    }.get(st, "moves through the space" if st else "stands")
    parts.append(loco)

    # upper gestures from menu
    g = (gesture_target or "none").lower()
    h = (hand or "right").lower()
    if g in ("chin", "chest", "forehead", "temple", "mouth", "ear", "stomach", "hip"):
        parts.append(f"with the {h} hand near the {g}")
    elif g in ("forward", "camera"):
        parts.append(f"reaching {g} with the {h} hand")

    act_phrases = {
        "wave": "waving a hand",
        "shrug": "shrugging the shoulders",
        "point_forward": "pointing forward",
        "think_chin": "thinking with a hand on the chin",
        "recoil": "pulling back slightly",
        "celebrate": "celebrating with open arms",
        "hands_reject": "pushing both hands forward in rejection",
        "talk_open": "gesturing while talking",
        "talk_emphasize": "emphasizing with hand gestures",
        "nod_yes": "nodding yes",
        "shake_no": "shaking the head no",
        "slump": "with a slumped posture",
        "tense": "with a tense upright posture",
    }
    for a in acts:
        if a in act_phrases and act_phrases[a] not in " ".join(parts):
            parts.append(act_phrases[a])

    emo = (emotion or "neutral").lower()
    if emo in ("happy", "sad", "angry", "fearful", "thinking") and emo != "neutral":
        parts.append(f"looking {emo}")

    caption = " ".join(parts)
    # lightly ground in dialogue topic without quoting speech
    if text and len(text) < 80:
        caption = caption  # keep motion-only; speech is face path
    return _clean_caption(caption)


_HUMANML_LEAD = re.compile(
    r"^(a\s+person|someone|a\s+man|a\s+woman|a\s+figure|"
    r"the\s+person|the\s+man|the\s+woman)\b",
    re.I,
)


def _third_person_verb(word: str) -> str:
    w = (word or "").lower()
    if not w:
        return "moves"
    if w in ("is", "are", "was", "were"):
        return "is"
    if w.endswith("ing") or w.endswith("ed"):
        return w
    if w in ("do", "does"):
        return "does"
    if w in ("have", "has"):
        return "has"
    if w in ("go", "goes"):
        return "goes"
    if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    if w.endswith(("s", "x", "z", "ch", "sh", "o")):
        return w if w.endswith("s") else w + "es"
    if not w.endswith("s"):
        return w + "s"
    return w


def to_humanml_caption(text: str, *, emotion: str = "", state: str = "") -> str:
    """
    Director rewrite: any stage/natural line → official HumanML3D caption.
    Never send raw chat / imperatives to gen_t2m.
    """
    raw = re.sub(r"\s+", " ", (text or "").strip().strip("\"'"))
    raw = re.sub(r"(?i)\b(please\s+)?(say|tell)\b.*$", "", raw).strip(" ,.;")
    if not raw:
        return build_humanml_prompt(state=state or "standing", emotion=emotion)

    if _HUMANML_LEAD.match(raw):
        return _clean_caption(raw)

    words = raw.split()
    w0 = words[0].lower()
    rest = " ".join(words[1:])
    if w0 in ("do", "does", "please"):
        if w0 == "please" and rest:
            return to_humanml_caption(rest, emotion=emotion, state=state)
        caption = f"a person does {rest}" if rest else "a person moves"
    else:
        caption = f"a person {_third_person_verb(w0)}"
        if rest:
            caption = f"{caption} {rest}"
    emo = (emotion or "").lower()
    if emo in ("happy", "sad", "angry", "surprised", "fearful") and emo not in caption.lower():
        caption = caption.rstrip(".") + f" looking {emo}"
    return _clean_caption(caption)


def _clean_caption(s: str) -> str:
    """Normalize to MoMask-friendly HumanML caption (not dialogue)."""
    s = re.sub(r"\s+", " ", (s or "").strip())
    s = s.strip("\"'")
    # Strip accidental dialogue wrappers / stage junk
    low = s.lower()
    for bad_prefix in ("say ", "says ", "speaking ", "dialogue:", "line:"):
        if low.startswith(bad_prefix):
            s = s[len(bad_prefix) :].strip()
            low = s.lower()
    # Bare greetings are useless for MoMask — expand to a stand+wave caption
    if low in ("hi", "hello", "hey", "hello!", "hi!", "hey there"):
        s = "a person stands and waves with the right hand"
        low = s.lower()
    # Single-token motion tags
    tag_map = {
        "walk": "a person walks forward at a steady pace",
        "walking": "a person walks forward at a steady pace",
        "run": "a person runs forward",
        "running": "a person runs forward",
        "jump": "a person jumps up and then lands",
        "sit": "a person sits down carefully",
        "wave": "a person stands and waves with the right hand",
        "dance": "a person dances in place with lively arm movements",
    }
    if low in tag_map:
        s = tag_map[low]
        low = s.lower()
    if not low.startswith("a person") and not low.startswith("someone") and not low.startswith("a man") and not low.startswith("the person"):
        if s:
            s = "a person " + s[0].lower() + s[1:] if len(s) > 1 else "a person " + s
    return s[:240]


def should_use_momask(
    *,
    body_mode: str = "",
    humanml_prompt: str = "",
    state: str = "standing",
    actions: Optional[List[str]] = None,
) -> bool:
    """
    Whether MoMask worker is on the plan (MotionRouter / director body_mode).

    catalog  → never
    momask / both → yes
    auto → any full-body class, or a filled HumanML caption, or MOMASK_ALL
    """
    from .director_schema import is_full_body_state

    mode = (body_mode or "").strip().lower()
    if not momask_enabled():
        return False
    # Director catalog wins — MOMASK_ALL must not slam greetings/talk into t2m
    if mode in ("catalog", "procedural"):
        return False
    if os.environ.get("MOMASK_ALL", "0").strip().lower() in ("1", "true", "yes", "on"):
        return True
    if mode in ("momask", "both"):
        return True
    hml = (humanml_prompt or "").strip()
    if hml and len(hml) >= 12 and mode in ("auto", "momask", "both", ""):
        return True
    if mode == "auto" and is_full_body_state(state):
        return True
    return False


def momask_sync_generate(allow_sync_gen: bool = False) -> bool:
    """
    Block for gen_t2m?
      - MOMASK_SYNC=1 → always allow wait
      - allow_sync_gen from MotionRouter (agent chose open model) → wait
      - else cache-only (no block)
    """
    raw = os.environ.get("MOMASK_SYNC", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off") and not allow_sync_gen:
        return False
    # Agent/router requested open generation for this beat
    if allow_sync_gen:
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return False


# Custom rest is applied AFTER retarget on SMPL-X (overwrite first/last frames),
# not by padding BVH (that broke feet / path). Matches apply_custom_rest_pads n_in.
REST_EASE_N_IN = 8
REST_EASE_N_OUT = 12
REST_EASE_PAD_FRAMES = 0  # no extra BVH frames; rest blends in-place on SMPL-X
MOMASK_FPS = 20.0
# Speech starts after custom-rest → motion blend at Action start
SPEECH_DELAY_FRAMES = REST_EASE_N_IN  # 8 → 0.40s @ 20fps
HARD_MAX_BODY_FRAMES = 196


def speech_delay_s() -> float:
    """Seconds of body rest/ease before speech should start."""
    return float(SPEECH_DELAY_FRAMES) / MOMASK_FPS


def speech_to_motion_length(
    duration_s: float,
    *,
    prompt: str = "",
    include_rest_pads: bool = True,
    min_frames: int = 32,
    max_frames: int = HARD_MAX_BODY_FRAMES,
) -> int:
    """
    Optional helper when director sets duration_s only.
    Prefer free MoMask length (motion_length=0) when director does not care.
    """
    dur = max(0.5, float(duration_s))
    total = int(round(dur * MOMASK_FPS))
    # Rest no longer adds BVH frames; include_rest_pads kept for API compat
    body = total
    # Locomotion needs enough frames or MoMask emits a 1.5s stub that pops
    low = (prompt or "").lower()
    if any(
        w in low
        for w in (
            "walk", "run", "jump", "jog", "sprint", "dance",
            "crawl", "crouch", "kneel", "sit",
        )
    ):
        min_frames = max(int(min_frames), 64)
    body = max(int(min_frames), min(int(max_frames), int(body)))
    body = max(int(min_frames), (int(body) // 4) * 4)
    return int(body)


def expected_action_frames(motion_length: int, *, with_rest_ease: bool = True) -> int:
    """BVH/Action frame count after convert (no BVH rest pad frames)."""
    ml = int(motion_length)
    return ml


def natural_action_duration_s(
    action_frames: int,
    *,
    clip_fps: float = MOMASK_FPS,
) -> float:
    """Wall time to play full Action (rest→motion→rest) at 1× clip rate."""
    frames = max(2, int(action_frames))
    return float(frames) / max(1.0, float(clip_fps))


def body_play_duration_s(
    action_frames: int,
    speech_duration_s: float,
    *,
    clip_fps: float = MOMASK_FPS,
) -> float:
    """
    Playback window for a one-shot MoMask Action.

    - If speech is longer → stretch body across speech + lead-in.
    - If speech is short / motion-only → use natural action length so a jump
      is not squashed into ~1s (jump then instant rest).
    """
    natural = natural_action_duration_s(action_frames, clip_fps=clip_fps)
    speech = max(0.0, float(speech_duration_s))
    speech_window = speech + speech_delay_s() if speech > 0.05 else 0.0
    return float(max(natural, speech_window, 0.5))


def sync_play_speed(
    action_frames: int,
    play_duration_s: float,
    *,
    clip_fps: float = MOMASK_FPS,
) -> float:
    """
    Speed so the full action finishes in play_duration_s when not using
    duration-stretch scrub. Prefer ~1.0 when play window ≈ natural length.
    receiver non-loop + duration path ignores speed (uses u = t/duration).
    """
    play = max(0.35, float(play_duration_s))
    frames = max(2, int(action_frames))
    natural_s = frames / max(1.0, float(clip_fps))
    # speed = natural / play → >1 only if we must finish sooner than natural
    sp = natural_s / play
    return float(max(0.35, min(2.5, sp)))


def _find_generation_ric(bvh: Path) -> Optional[Path]:
    """Locate *_ric.npy next to official gen_t2m output."""
    try:
        anim_dir = Path(bvh).resolve().parent
        # .../generation/<ext>/animations/<k>/file.bvh
        ext_root = anim_dir.parent.parent
        k = anim_dir.name
        joint_dir = ext_root / "joints" / k
        if joint_dir.is_dir():
            hits = sorted(joint_dir.glob("*_ric.npy"))
            if hits:
                return hits[-1]
        # same folder fallback
        hits = sorted(anim_dir.glob("*_ric.npy"))
        return hits[-1] if hits else None
    except Exception:
        return None


def latest_cached_ric() -> Optional[Path]:
    if not CACHE_DIR.is_dir():
        return None
    hits = sorted(CACHE_DIR.glob("*_ric.npy"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def inpaint_body_action(
    prompt: str,
    *,
    start_frame: int,
    end_frame: int,
    source_ric: Optional[Path] = None,
    source_action: str = "",
    seed: int = 7,
) -> MomaskBodyResult:
    """
    MoMask temporal inpaint: keep the source clip, regenerate only [start, end].

    Official: third_party/momask-codes/edit_t2m.py
      --source_motion <ric.npy> --text_prompt "..." --mask_edit_section start,end
    """
    prompt = to_humanml_caption(prompt)
    if not prompt:
        return MomaskBodyResult(ok=False, error="empty inpaint prompt")
    ric = Path(source_ric) if source_ric else None
    if ric is None and source_action:
        cand = CACHE_DIR / f"{source_action}_ric.npy"
        if cand.is_file():
            ric = cand
    if ric is None:
        ric = latest_cached_ric()
    if ric is None or not ric.is_file():
        return MomaskBodyResult(
            ok=False,
            error="no source RIC npy — generate a clip first (needs *_ric.npy in momask_cache)",
        )
    f0 = max(0, int(start_frame))
    f1 = max(f0 + 4, int(end_frame))
    py = momask_python()
    edit = MOMASK_CODES / "edit_t2m.py"
    if not edit.is_file():
        return MomaskBodyResult(ok=False, error=f"missing {edit}")
    ext = f"inpaint_{int(time.time())}"
    gpu = os.environ.get("MOMASK_GPU_ID", "").strip() or "-1"
    cmd = [
        py, str(edit),
        "--gpu_id", gpu,
        "--ext", ext,
        "--source_motion", str(ric),
        "--text_prompt", prompt,
        "--mask_edit_section", f"{f0},{f1}",
        "--seed", str(int(seed)),
    ]
    _log(f"inpaint {ric.name} [{f0}-{f1}] prompt={prompt!r}")
    r = _run(cmd, cwd=MOMASK_CODES, timeout=1200)
    if r.returncode != 0:
        return MomaskBodyResult(
            ok=False,
            error=f"edit_t2m failed: {(r.stderr or r.stdout or '')[-1500:]}",
        )
    anim_root = MOMASK_CODES / "editing" / ext / "animations"
    ik = sorted(anim_root.rglob("*_ik.bvh"))
    if not ik:
        ik = sorted(anim_root.rglob("*.bvh"))
    if not ik:
        return MomaskBodyResult(ok=False, error=f"no inpaint BVH under {anim_root}")
    bvh = ik[-1]
    action_name = _cache_key(f"inpaint:{prompt}:{f0}-{f1}", f1 - f0, seed)
    out_blend = CACHE_DIR / f"{action_name}.blend"
    try:
        retarget_bvh_to_smplx_action(bvh, action_name, out_blend)
    except Exception as e:
        return MomaskBodyResult(ok=False, error=f"inpaint retarget: {e}")
    act_frames = _count_bvh_frames(bvh)
    meta = {
        "ok": True,
        "action_name": action_name,
        "prompt": prompt,
        "inpaint": True,
        "edit_range": [f0, f1],
        "source_ric": str(ric),
        "blend_path": str(out_blend),
        "bvh_path": str(bvh),
        "action_frames": int(act_frames),
        "clip_fps": float(MOMASK_FPS),
    }
    (CACHE_DIR / f"{action_name}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return MomaskBodyResult(
        ok=True,
        action_name=action_name,
        prompt=prompt,
        bvh_path=str(bvh),
        blend_path=str(out_blend),
        frames=[1, int(act_frames)],
        notes=[f"inpaint {f0}-{f1}", f"source={ric.name}"],
    )


def _count_bvh_frames(bvh: Path) -> int:
    """Count motion frames in a BVH (lines after 'Frame Time:')."""
    try:
        text = Path(bvh).read_text(encoding="utf-8", errors="ignore")
        n = 0
        in_motion = False
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("Frames:"):
                try:
                    return max(2, int(s.split(":", 1)[1].strip()))
                except Exception:
                    pass
            if s.lower().startswith("frame time"):
                in_motion = True
                continue
            if in_motion and s and not s[0].isalpha():
                n += 1
        return max(2, n) if n else 0
    except Exception:
        return 0


def _norm_prompt_key(prompt: str) -> str:
    """Compare captions case/space-insensitively for cache memory."""
    return re.sub(r"\s+", " ", (prompt or "").strip().lower())


def _is_usable_cache_blend(blend: Path) -> bool:
    if not blend.is_file():
        return False
    try:
        # Slim Action-only ~0.1–2MB; skip legacy full-scene dumps
        prefer_slim = os.environ.get("MOMASK_FULL_SCENE_CACHE", "0").strip().lower() not in (
            "1", "true", "yes", "on",
        )
        if prefer_slim and blend.stat().st_size > 8 * 1024 * 1024:
            return False
        return blend.stat().st_size > 1024
    except OSError:
        return False


def _result_from_meta(
    meta: dict,
    *,
    prompt: str,
    out_blend: Path,
    note: str,
) -> Optional[MomaskBodyResult]:
    if not meta.get("ok") or not meta.get("action_name"):
        return None
    if not _is_usable_cache_blend(out_blend):
        return None
    return MomaskBodyResult(
        ok=True,
        action_name=str(meta["action_name"]),
        prompt=prompt or str(meta.get("prompt") or ""),
        bvh_path=str(meta.get("bvh_path") or ""),
        blend_path=str(out_blend),
        frames=list(meta.get("frames") or []),
        cached=True,
        notes=[note, f"format={meta.get('cache_format', 'action_only')}"],
    )


def find_cache_by_prompt(
    prompt: str,
    *,
    motion_length: int = 0,
) -> Optional[MomaskBodyResult]:
    """
    User-style memory: if this HumanML caption was already generated, reuse it.

    Scans momask_cache/*.json for matching cleaned prompt (ignores seed).
    Prefer exact action-name key first, then any meta with same prompt text.
    """
    prompt = _clean_caption(prompt)
    if not prompt:
        return None
    want = _norm_prompt_key(prompt)
    key_ml = int(motion_length) if motion_length and int(motion_length) > 0 else "free"

    # 1) Canonical key (prompt + length mode, seed not in identity)
    action_name = _cache_key(prompt, key_ml, 0)
    meta_path = CACHE_DIR / f"{action_name}.json"
    out_blend = CACHE_DIR / f"{action_name}.blend"
    if meta_path.is_file() and out_blend.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            hit = _result_from_meta(meta, prompt=prompt, out_blend=out_blend, note="cache_hit_key")
            if hit:
                return hit
        except Exception:
            pass

    # 2) Scan all metas — any prior seed/key with same caption text
    if not CACHE_DIR.is_dir():
        return None
    best: Optional[MomaskBodyResult] = None
    best_mtime = -1.0
    for jp in CACHE_DIR.glob("momask_*.json"):
        try:
            meta = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not meta.get("ok"):
            continue
        mp = _norm_prompt_key(str(meta.get("prompt") or ""))
        if mp != want:
            continue
        # length mode: if we asked free, accept free or missing; if fixed, match frames
        meta_ml = meta.get("motion_length")
        if key_ml == "free":
            pass  # any length for this caption is fine
        else:
            try:
                if int(meta_ml or 0) != int(key_ml):
                    continue
            except (TypeError, ValueError):
                continue
        blend = Path(str(meta.get("blend_path") or ""))
        if not blend.is_file():
            blend = jp.with_suffix(".blend")
        hit = _result_from_meta(
            meta, prompt=prompt, out_blend=blend, note="cache_hit_prompt_scan"
        )
        if not hit:
            continue
        try:
            mt = blend.stat().st_mtime
        except OSError:
            mt = 0.0
        if mt >= best_mtime:
            best_mtime = mt
            best = hit
    return best


def lookup_cached_action(
    prompt: str,
    *,
    duration_s: float = 4.0,
    seed: int = 42,
    motion_length: int = 0,
) -> Optional[MomaskBodyResult]:
    """
    Realtime-safe: return cache only, never run gen/retarget.

    Matches like a user reusing memory: same cleaned prompt → hit
    (seed does not force a miss if that caption already exists).
    """
    prompt = _clean_caption(prompt)
    if not prompt:
        return None
    # Prefer prompt identity (user expectation); seed ignored for lookup
    hit = find_cache_by_prompt(prompt, motion_length=motion_length)
    if hit:
        _log(f"cache hit (prompt) → {hit.action_name}")
        return hit
    return None


def _cache_key(prompt: str, frames, seed: int = 0) -> str:
    """
    Cache identity = cleaned caption + length mode.
    Seed is stored in meta for regen variety but does NOT fragment the cache
    (same prompt always maps to the same Action file).
    """
    ml = frames if frames not in (None, "", 0, "0") else "free"
    h = hashlib.md5(f"{_clean_caption(str(prompt))}|{ml}".encode("utf-8")).hexdigest()[:12]
    return f"momask_{h}"


# Pipeline temp lives under the project (not C:\Users\...\AppData\Local\Temp).
# Each Blender retarget otherwise drops ~70–80MB autosave*.blend on C: and those
# pile up to multiple GB across runs.
PIPELINE_TMP = ROOT / "body_motion" / "_pipeline_tmp"


def _pipeline_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Env for subprocesses: keep Blender/torch temp off the system C: Temp."""
    PIPELINE_TMP.mkdir(parents=True, exist_ok=True)
    tmp = str(PIPELINE_TMP.resolve())
    env = {**os.environ}
    # Windows + Unix temp vars — Blender autosave + numpy memmaps honor these
    for key in ("TEMP", "TMP", "TMPDIR"):
        env[key] = tmp
    if extra:
        env.update(extra)
    return env


def cleanup_pipeline_disk(*, max_cache_blends: int = 12, max_tmp_age_hours: float = 24.0) -> Dict[str, Any]:
    """
    Reclaim disk after MoMask/Blender runs.

    Why C: / disk used to grow:
      - Blender autosave*.blend (~70MB) in %TEMP% (now redirected to _pipeline_tmp)
      - Legacy momask_cache full-scene .blend (~74MB); new cache is Action-only slim
      - huggingface/torch under user profile (not deleted here)

    Safe cleanup: project _pipeline_tmp + old system temp autosaves + fat legacy cache.
    """
    report: Dict[str, Any] = {"freed_mb": 0.0, "removed": []}
    freed = 0

    def _rm(p: Path) -> None:
        nonlocal freed
        try:
            if p.is_file():
                sz = p.stat().st_size
                p.unlink()
                freed += sz
                report["removed"].append(str(p))
        except OSError:
            pass

    # 1) Project pipeline tmp (preferred location)
    if PIPELINE_TMP.is_dir():
        for p in PIPELINE_TMP.rglob("*"):
            if p.is_file() and (
                "autosave" in p.name.lower()
                or p.name.lower().startswith("quit.blend")
                or p.suffix.lower() in (".blend1", ".tmp")
            ):
                _rm(p)

    # 2) System Temp leftover Blender autosaves from older runs (C: bloat)
    sys_temps = []
    for key in ("TEMP", "TMP"):
        v = os.environ.get(key, "")
        if v:
            sys_temps.append(Path(v))
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        sys_temps.append(Path(local) / "Temp")
    patterns = (
        "*autosave*.blend",
        "quit.blend",
        "quit.blend1",
        "whole_body*.blend",
        "momask_*.blend",
        "walk_*.blend",
    )
    now = time.time()
    max_age = max_tmp_age_hours * 3600.0
    for td in sys_temps:
        if not td.is_dir():
            continue
        # Don't wipe our project tmp twice
        try:
            if td.resolve() == PIPELINE_TMP.resolve():
                continue
        except OSError:
            pass
        for pat in patterns:
            for p in td.glob(pat):
                try:
                    age = now - p.stat().st_mtime
                except OSError:
                    continue
                # Always drop autosave/quit; other names only if older than max_age
                if "autosave" in p.name.lower() or p.name.lower().startswith("quit.blend"):
                    _rm(p)
                elif age > max_age:
                    _rm(p)

    # 3) Cap momask_cache blends (keep newest). Drop *old* fat leftovers only —
    #    never delete a file written in the last 30 minutes (full-scene verify
    #    and just-finished retargets live here).
    blends = sorted(
        CACHE_DIR.glob("momask_*.blend"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    keep_new_s = 30 * 60.0
    for i, old in enumerate(blends):
        try:
            st = old.stat()
            fat = st.st_size > 8 * 1024 * 1024
            age = now - st.st_mtime
        except OSError:
            fat = False
            age = 0.0
        if age < keep_new_s:
            continue
        if i >= int(max_cache_blends) or fat:
            meta = old.with_suffix(".json")
            bvh_c = old.with_suffix(".bvh")
            _rm(old)
            if meta.is_file():
                _rm(meta)
            if bvh_c.is_file():
                _rm(bvh_c)

    report["freed_mb"] = round(freed / (1024 * 1024), 1)
    if report["freed_mb"] > 0:
        _log(f"disk cleanup freed {report['freed_mb']} MB ({len(report['removed'])} files)")
    return report


def _run(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 900) -> subprocess.CompletedProcess:
    _log(" ".join(cmd[:6]) + (" …" if len(cmd) > 6 else ""))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_pipeline_env(),
    )


def _cached_momask_gpu_id(py: str) -> str:
    """Reuse last CUDA probe. Spawning torch every clip costs tens of seconds."""
    env = os.environ.get("MOMASK_GPU_ID", "").strip()
    if env:
        return env
    PIPELINE_TMP.mkdir(parents=True, exist_ok=True)
    cache = PIPELINE_TMP / "momask_gpu_id.txt"
    try:
        if cache.is_file() and (time.time() - cache.stat().st_mtime) < 86400:
            val = cache.read_text(encoding="utf-8").strip()
            if val:
                return val
    except OSError:
        pass
    gpu = "-1"
    try:
        chk = subprocess.run(
            [py, "-c", "import torch; print(1 if torch.cuda.is_available() else 0)"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        gpu = "0" if (chk.returncode == 0 and chk.stdout.strip() == "1") else "-1"
    except Exception:
        gpu = "-1"
    if gpu not in ("-1",) and gpu.lstrip("-").isdigit() is False:
        gpu = "-1"
    try:
        cache.write_text(gpu, encoding="utf-8")
    except OSError:
        pass
    return gpu


def generate_official_bvh(
    prompt: str,
    *,
    ext: str,
    seed: int = 42,
    motion_length: int = 0,
) -> Path:
    """Run official gen_t2m.py → return path to *_ik.bvh."""
    py = momask_python()
    gen = MOMASK_CODES / "gen_t2m.py"
    if not gen.is_file():
        raise FileNotFoundError(f"missing {gen}")

    gpu = _cached_momask_gpu_id(py)
    _log(f"gen_t2m gpu_id={gpu} (set MOMASK_GPU_ID to override)")
    cmd = [
        py,
        str(gen),
        "--gpu_id",
        gpu,
        "--ext",
        ext,
        "--text_prompt",
        prompt,
        "--repeat_times",
        "1",
        "--seed",
        str(int(seed)),
    ]
    # motion_length > 0 → explicit. motion_length == 0 → MoMask free estimate (model decides).
    ml = int(motion_length or 0)
    if ml > 0:
        ml = max(4, min(HARD_MAX_BODY_FRAMES, (ml // 4) * 4))
        cmd.extend(["--motion_length", str(ml)])
        _log(f"gen_t2m --motion_length {ml} (director explicit)")
    else:
        _log("gen_t2m free length (MoMask decides — no length limit from pipeline)")

    r = _run(cmd, cwd=MOMASK_CODES, timeout=1200)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-2000:]
        raise RuntimeError(f"gen_t2m failed ({r.returncode}): {tail}")

    anim_root = MOMASK_CODES / "generation" / ext / "animations"
    ik_files = sorted(anim_root.rglob("*_ik.bvh"))
    if not ik_files:
        # fallback non-ik
        bvh_files = sorted(anim_root.rglob("*.bvh"))
        if not bvh_files:
            raise FileNotFoundError(f"no BVH under {anim_root}")
        return bvh_files[-1]
    return ik_files[-1]


def ensure_retarget_host() -> Path:
    """Armature-only .blend for bake. Avoids loading 76MB production + Rokoko."""
    host = RETARGET_HOST
    ver_p = host.with_suffix(".ver")
    host_ok = False
    try:
        if host.is_file() and host.stat().st_size > 50_000:
            ver = ver_p.read_text(encoding="utf-8").strip() if ver_p.is_file() else ""
            fresh = (not BASE_BLEND.is_file()) or host.stat().st_mtime >= BASE_BLEND.stat().st_mtime
            host_ok = ver == "mesh-sole-1" and fresh
    except OSError:
        host_ok = False
    if host_ok:
        return host
    if not BASE_BLEND.is_file():
        return BASE_BLEND
    bl = blender_exe()
    host.parent.mkdir(parents=True, exist_ok=True)
    snippet = (
        "import bpy\n"
        "keep={'SMPL-X_Armature','BodyMesh'}\n"
        "for o in list(bpy.data.objects):\n"
        "    if o.name not in keep:\n"
        "        bpy.data.objects.remove(o, do_unlink=True)\n"
        "for coll_name in ('meshes','materials','images','textures','lights','cameras','curves'):\n"
        "    coll=getattr(bpy.data, coll_name, None)\n"
        "    if coll is None: continue\n"
        "    for b in list(coll):\n"
        "        if getattr(b,'users',1)==0:\n"
        "            try: coll.remove(b)\n"
        "            except Exception: pass\n"
        "try:\n"
        "    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)\n"
        "except Exception:\n"
        "    pass\n"
        f"bpy.ops.wm.save_as_mainfile(filepath=r'{str(host.resolve())}', compress=True)\n"
    )
    _log(f"building slim retarget host → {host.name}")
    r = subprocess.run(
        [
            bl,
            "--factory-startup",
            "--background",
            str(BASE_BLEND),
            "--python-expr",
            snippet,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        env=_pipeline_env(),
    )
    if host.is_file() and host.stat().st_size > 50_000:
        try:
            ver_p.write_text("mesh-sole-1", encoding="utf-8")
        except OSError:
            pass
        _log(f"retarget host ready {host.stat().st_size // 1024} KB (armature+BodyMesh)")
        return host
    _log(f"retarget host build failed (rc={r.returncode}) — using full scene")
    return BASE_BLEND


def retarget_bvh_to_smplx_action(
    bvh: Path,
    action_name: str,
    out_blend: Path,
) -> Dict[str, Any]:
    """
    Single Blender process: BVH → SMPL-X Action via Rokoko-style map only.
    NO KeeMap / NO Mixamo Idle (same as manual Rokoko path that works).
    """
    bl = blender_exe()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not BASE_BLEND.is_file():
        raise FileNotFoundError(f"missing {BASE_BLEND}")
    if not DIRECT_BVH_SCRIPT.is_file():
        raise FileNotFoundError(f"missing {DIRECT_BVH_SCRIPT}")
    host = ensure_retarget_host()
    map_path = rokoko_map_path()
    # Default: Action-only slim .blend (~0.2–2 MB). Full scene only if env set.
    full_scene = os.environ.get("MOMASK_FULL_SCENE_CACHE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    auto_scale = os.environ.get("MOMASK_AUTO_SCALE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )
    # Joint floor plant stays opt-in (puts soles through mesh). Mesh sole always on.
    # Foot lock (anti-slide contact IK) defaults ON so pipeline matches good manual
    # Rokoko feet; set MOMASK_FOOT_LOCK=0 to disable.
    want_floor = os.environ.get("MOMASK_FLOOR_PLANT", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )
    no_align = os.environ.get("MOMASK_ALIGN", "1").strip().lower() in (
        "0", "false", "no", "off",
    )
    _fl_raw = os.environ.get("MOMASK_FOOT_LOCK", "1").strip().lower()
    want_foot = _fl_raw not in ("0", "false", "no", "off")
    _log(
        f"direct BVH→SMPL-X map={map_path.name} "
        f"auto_scale={auto_scale} floor={want_floor} align={not no_align} "
        f"foot_lock={want_foot} "
        f"cache={'full_scene' if full_scene else 'action_only_slim'} "
        f"host={host.name}"
    )
    cmd = [
        bl,
        "--factory-startup",
        "--background",
        str(host),
        "--python",
        str(DIRECT_BVH_SCRIPT),
        "--python-exit-code",
        "1",
        "--",
        "--bvh",
        str(bvh),
        "--action",
        action_name,
        "--out",
        str(out_blend),
        "--rokoko_map",
        str(map_path),
        "--root",
        "location",
    ]
    if auto_scale:
        cmd.append("--auto-scale")
    if want_floor:
        cmd.append("--floor-plant")
    if no_align:
        cmd.append("--no-align")
    if want_foot:
        cmd.append("--foot-lock")
    if full_scene:
        cmd.append("--full-scene")
    r = _run(cmd, cwd=ROOT, timeout=600)
    # Blender addons often exit 1 even when the script saved. Trust the file +
    # the [bvh_rokoko] {"ok": true} line.
    log = (r.stdout or "") + "\n" + (r.stderr or "")
    saved = out_blend.is_file() and out_blend.stat().st_size > 1024
    ok_line = '"ok": true' in log or '"ok":true' in log
    if not saved or not ok_line:
        raise RuntimeError(
            f"SMPL-X retarget failed (rc={r.returncode} saved={saved}): {log[-2000:]}"
        )
    return {"out_blend": str(out_blend), "method": "direct_bvh_rokoko", "map": str(map_path)}


def generate_body_action(
    prompt: str,
    *,
    duration_s: float = 4.0,
    seed: int = 42,
    motion_length: int = 0,
    use_cache: bool = True,
) -> MomaskBodyResult:
    """
    End-to-end: HumanML3D caption → SMPL-X Action name.

    Returns MomaskBodyResult; face pipeline never calls this for mouth/eyes.
    """
    prompt = _clean_caption(prompt)
    if not prompt:
        return MomaskBodyResult(ok=False, error="empty prompt")

    # Explicit director length only; 0 = MoMask free estimate (model decides length).
    if motion_length and int(motion_length) > 0:
        motion_length = max(4, min(HARD_MAX_BODY_FRAMES, (int(motion_length) // 4) * 4))
    else:
        motion_length = 0

    key_ml = motion_length if motion_length > 0 else "free"
    action_name = _cache_key(prompt, key_ml, seed)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE_DIR / f"{action_name}.json"
    out_blend = CACHE_DIR / f"{action_name}.blend"

    # User-style memory: reuse any existing clip for this caption before gen_t2m
    if use_cache and os.environ.get("MOMASK_FORCE_REGEN", "0").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        hit = find_cache_by_prompt(prompt, motion_length=motion_length)
        if hit and hit.ok:
            sz = Path(hit.blend_path).stat().st_size if hit.blend_path and Path(hit.blend_path).is_file() else 0
            _log(
                f"cache hit (reuse existing prompt) {hit.action_name} "
                f"size={sz / 1024:.0f} KB — skip MoMask gen"
            )
            hit.notes = list(hit.notes or []) + ["cache_hit_before_gen", f"size_kb={sz // 1024}"]
            return hit

    t0 = time.time()
    notes: List[str] = []
    try:
        ext = f"orch_{action_name}"
        speech_delay = speech_delay_s()
        _log(
            f"gen_t2m prompt={prompt!r} motion_length="
            f"{'FREE' if motion_length <= 0 else motion_length} "
            f"speech={float(duration_s):.2f}s"
        )
        bvh = generate_official_bvh(
            prompt, ext=ext, seed=seed, motion_length=motion_length
        )
        notes.append(f"bvh={bvh.name}")
        notes.append(
            f"motion_length={'free' if motion_length <= 0 else motion_length}"
        )
        notes.append("no_intelligent_cut")
        _log(f"official BVH → {bvh}")

        # Real frame count from BVH (after free gen, this is the truth)
        act_frames = _count_bvh_frames(bvh)
        if act_frames < 2 and motion_length > 0:
            act_frames = expected_action_frames(motion_length, with_rest_ease=True)
        act_frames = max(int(act_frames), 16)
        body_play_s = body_play_duration_s(
            act_frames, float(duration_s), clip_fps=MOMASK_FPS
        )
        sync_sp = sync_play_speed(act_frames, body_play_s, clip_fps=MOMASK_FPS)
        notes.append(f"action_frames={act_frames}")
        notes.append(f"speech_delay_s={speech_delay:.3f}")
        notes.append(f"body_play_s={body_play_s:.3f}")

        retarget_bvh_to_smplx_action(bvh, action_name, out_blend)
        notes.append("retarget_ok")

        # Keep a light BVH copy next to the Action cache (regen/debug; not full scene)
        cached_bvh = CACHE_DIR / f"{action_name}.bvh"
        try:
            import shutil

            if bvh.is_file():
                shutil.copy2(bvh, cached_bvh)
                notes.append("bvh_cached")
            ric_src = _find_generation_ric(bvh)
            if ric_src and ric_src.is_file():
                ric_dst = CACHE_DIR / f"{action_name}_ric.npy"
                shutil.copy2(ric_src, ric_dst)
                notes.append("ric_cached")
        except Exception as e:
            notes.append(f"bvh_cache_skip:{e}")
            cached_bvh = Path("")

        # Optional merge into shared library (still Action-only source)
        if os.environ.get("MOMASK_LIBRARY_MERGE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            try:
                _merge_action_to_library(out_blend, action_name, LIBRARY_BLEND)
                notes.append("library_updated")
            except Exception as e:
                notes.append(f"library_skip:{e}")

        size_b = out_blend.stat().st_size if out_blend.is_file() else 0
        # Compact meta: useful for body play + speech/camera sync; no redundant noise
        meta = {
            "ok": True,
            "cache_format": "action_only",  # Action datablock only; camera is live UDP
            "action_name": action_name,
            "prompt": prompt,
            "seed": int(seed),
            "motion_length": int(motion_length),  # 0 = free at gen time
            "action_frames": int(act_frames),
            "frames": [1, int(act_frames)],
            "clip_fps": float(MOMASK_FPS),
            # Timing used by coordinator/camera speech_delay alignment
            "speech_duration_s": round(float(duration_s), 4),
            "speech_delay_s": round(float(speech_delay), 4),
            "body_play_s": round(float(body_play_s), 4),
            "sync_speed": round(float(sync_sp), 4),
            # Paths (slim Action blend + optional BVH copy)
            "blend_path": str(out_blend),
            "bvh_path": str(cached_bvh) if cached_bvh and Path(cached_bvh).is_file() else str(bvh),
            "size_bytes": int(size_b),
            "size_kb": round(size_b / 1024, 1),
            "elapsed_s": round(time.time() - t0, 2),
            # Camera is NOT stored in body cache — session MovieCam via camera_agent
            "camera": "live_udp_session",
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _log(
            f"DONE action={action_name} in {meta['elapsed_s']}s "
            f"slim_cache={meta['size_kb']} KB sync_speed={sync_sp:.2f}"
        )
        return MomaskBodyResult(
            ok=True,
            action_name=action_name,
            prompt=prompt,
            bvh_path=str(meta["bvh_path"]),
            blend_path=str(out_blend),
            frames=[1, act_frames],
            cached=False,
            notes=notes + [f"slim_kb={meta['size_kb']}"],
        )
    except Exception as e:
        _log(f"FAILED: {e}")
        return MomaskBodyResult(
            ok=False,
            prompt=prompt,
            error=str(e),
            notes=notes,
        )


def _merge_action_to_library(src_blend: Path, action_name: str, lib_blend: Path) -> None:
    """Append Action from src_blend into library blend (background Blender)."""
    bl = blender_exe()
    code = f"""
import bpy
from pathlib import Path
src = r"{src_blend.as_posix()}"
act_name = r"{action_name}"
lib = r"{lib_blend.as_posix()}"
# start empty or open lib
p = Path(lib)
if p.is_file():
    bpy.ops.wm.open_mainfile(filepath=lib)
else:
    bpy.ops.wm.read_homefile(use_empty=True)
# append action
with bpy.data.libraries.load(src, link=False) as (df, dt):
    names = [n for n in df.actions if n == act_name or act_name in n]
    dt.actions = names or list(df.actions)
for a in bpy.data.actions:
    a.use_fake_user = True
bpy.ops.wm.save_as_mainfile(filepath=lib)
print("library saved", lib, "actions", [a.name for a in bpy.data.actions])
"""
    script = CACHE_DIR / "_merge_action.py"
    script.write_text(code, encoding="utf-8")
    r = _run([bl, "--background", "--python", str(script)], cwd=ROOT, timeout=180)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "")[-800:])


def ensure_action_in_blender_via_packet_hint(
    result: MomaskBodyResult,
    *,
    speech_duration_s: float = 0.0,
) -> Dict[str, Any]:
    """Extra fields for UDP so blender_receiver can load Action from cache blend."""
    # Prefer meta sync fields when regenerating; recompute from frames if cache hit
    act_frames = 0
    if result.frames and len(result.frames) >= 2:
        act_frames = int(result.frames[-1]) - int(result.frames[0]) + 1
    if act_frames < 2 and result.blend_path:
        # fallback estimate from notes
        for n in result.notes or []:
            if str(n).startswith("action_frames≈"):
                try:
                    act_frames = int(float(str(n).split("≈", 1)[1]))
                except Exception:
                    pass
    speech = float(speech_duration_s or 0.0)
    # Read compact meta: body timing + speech delay (camera plans use same delay)
    sync_speed = 1.0
    clip_fps = MOMASK_FPS
    meta_body_play = 0.0
    speech_delay = speech_delay_s()
    meta_path = CACHE_DIR / f"{result.action_name}.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            act_frames = int(meta.get("action_frames") or act_frames or 0)
            if speech <= 0.05:
                speech = float(meta.get("speech_duration_s") or 0.0)
            sync_speed = float(meta.get("sync_speed") or 0.0)
            clip_fps = float(meta.get("clip_fps") or MOMASK_FPS)
            meta_body_play = float(meta.get("body_play_s") or 0.0)
            if meta.get("speech_delay_s") is not None:
                speech_delay = float(meta["speech_delay_s"])
        except Exception:
            pass
    if act_frames < 2:
        act_frames = expected_action_frames(
            speech_to_motion_length(speech or 4.0, prompt=result.prompt or ""),
            with_rest_ease=True,
        )
    # Natural action length when speech is short — do not squash jump into TTS
    body_play = body_play_duration_s(act_frames, speech, clip_fps=clip_fps)
    if meta_body_play > body_play:
        body_play = meta_body_play
    sync_speed = sync_play_speed(act_frames, body_play, clip_fps=clip_fps)

    # Full BVH (no cut). Body starts at rest; speech delayed until motion begins.
    # duration = full rest→motion→rest window (may outlast short speech).
    motion_length_meta = 0
    if meta_path.is_file():
        try:
            motion_length_meta = int(
                json.loads(meta_path.read_text(encoding="utf-8")).get("motion_length") or 0
            )
        except Exception:
            motion_length_meta = 0

    return {
        "action": result.action_name,
        "clip_id": result.action_name,
        "engine": "momask",
        "library_blend": result.blend_path or str(LIBRARY_BLEND),
        "loop": False,
        "force": True,
        "speed": sync_speed,
        "clip_fps": clip_fps,
        "duration": body_play,
        "body_play_s": body_play,
        "action_frames": act_frames,
        "motion_length": motion_length_meta,
        "speech_duration_s": speech,
        "speech_delay_s": speech_delay,
        "sync_mode": "body_first_then_speech",
    }
