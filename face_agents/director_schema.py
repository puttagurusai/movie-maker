"""
director_schema.py — Director-level beat format for the avatar.

A beat = what to SAY + how to FEEL (face) + body STATE + optional GESTURE.

Body control model (layering):
  - state          → base layer (standing / sitting / walking / dancing)
  - gesture_target → upper-body procedural IK (chin, chest, forward, …)
  - actions[]      → optional upper-body catalog verbs (wave, shrug, …)
  - body_mode      → catalog | momask | auto
  - humanml_prompt → free-text HumanML3D caption when body_mode=momask
  - face           → always separate (ARKit / agents) — NEVER driven by MoMask

LLM never outputs bone angles — only words / captions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .look_schema import LookPlan


# ── Action catalog (upper-body verbs / complex clips) ───────────────────────
ACTION_CATALOG: Dict[str, Dict[str, Any]] = {
    "idle": {"layer": "body", "desc": "Neutral hold", "procedural": "idle"},
    "talk_open": {
        "layer": "body",
        "desc": "Open hand gestures while speaking",
        "procedural": "talk_open",
    },
    "talk_emphasize": {
        "layer": "body",
        "desc": "Beat gesture on speech energy peaks",
        "procedural": "talk_emphasize",
    },
    "wave": {
        "layer": "body",
        "desc": "Greeting wave (complex upper clip)",
        "procedural": "wave",
        "duration_s": 1.8,
    },
    "nod_yes": {
        "layer": "both",
        "desc": "Agreement nod",
        "face_extra": "token_nod",
        "procedural": "nod_yes",
    },
    "shake_no": {
        "layer": "both",
        "desc": "Disagreement head shake",
        "procedural": "shake_no",
    },
    "shrug": {
        "layer": "body",
        "desc": "I don't know shrug",
        "procedural": "shrug",
        "duration_s": 1.35,
    },
    "point_forward": {
        "layer": "body",
        "desc": "Point forward",
        "procedural": "point",
        "duration_s": 1.25,
        "gesture_target": "forward",
        "hand": "right",
    },
    "recoil": {
        "layer": "body",
        "desc": "Pull back (disgust/surprise)",
        "procedural": "recoil",
        "duration_s": 1.1,
    },
    "celebrate": {
        "layer": "body",
        "desc": "Soft open arms (happy)",
        "procedural": "celebrate",
        "duration_s": 1.5,
    },
    "slump": {"layer": "body", "desc": "Sad/tired posture", "procedural": "slump"},
    "tense": {"layer": "body", "desc": "Angry/assertive upright", "procedural": "tense"},
    "think_chin": {
        "layer": "body",
        "desc": "Thinking hand near chin",
        "procedural": "think_chin",
        "duration_s": 1.5,
        "gesture_target": "chin",
        "hand": "right",
    },
    "hands_reject": {
        "layer": "body",
        "desc": "Push-away hands",
        "procedural": "hands_reject",
        "duration_s": 0.8,
        "gesture_target": "forward",
        "hand": "both",
    },
    "look_camera": {
        "layer": "both",
        "desc": "Orient toward camera",
        "procedural": "look_camera",
    },
    "look_left": {"layer": "both", "desc": "Glance left", "procedural": "look_left"},
    "look_right": {"layer": "both", "desc": "Glance right", "procedural": "look_right"},
    "weight_shift": {
        "layer": "body",
        "desc": "Idle weight shift",
        "procedural": "weight_shift",
    },
}

VALID_ACTION_TIMINGS = frozenset({"start", "during", "end"})

# Base layer — coarse posture / travel classes (not a verb menu).
# walking/dancing stay as catalog-compatible names; locomotion/grounded
# cover every other full-body performance without naming each action.
BODY_STATES = frozenset({
    "standing",
    "sitting",
    "walking",
    "dancing",
    "locomotion",
    "grounded",
})
# Frame the whole figure (CameraAgent / MoMask auto).
FULL_BODY_STATES = frozenset({"walking", "dancing", "locomotion", "grounded"})

_HUMANML_PREFIX = re.compile(
    r"^(a\s+person|someone|a\s+man|a\s+woman|a\s+figure|"
    r"the\s+person|the\s+man|the\s+woman)\b",
    re.I,
)
_SPATIAL_HINT = re.compile(
    r"(?i)\b(forward|backward|backwards|left|right|around|across|toward|"
    r"towards|onto|into|down|up|floor|ground|knees|hands|feet|arms|"
    r"slowly|quickly|in\s+place|in\s+a\s+circle)\b",
)
_DIALOGUE_SUBJECT = re.compile(
    r"(?i)\b(i|i'm|im|me|my|myself|we|our|you\s+know)\b",
)

# Overlay layer — semantic IK targets (LLM menu; guide: upper_gesture_target)
GESTURE_TARGETS = frozenset({
    "none",
    "chin",
    "chest",
    "forehead",
    "temple",
    "mouth",
    "ear",
    "hip",
    "stomach",
    "forward",
    "camera",
    "rest_at_side",
})
GESTURE_HANDS = frozenset({"left", "right", "both"})

# Emotion → default upper actions when director omits actions
EMOTION_DEFAULT_ACTIONS: Dict[str, List[str]] = {
    "neutral": ["talk_open"],
    "happy": ["talk_open", "celebrate"],
    "sad": ["slump", "talk_open"],
    "angry": ["tense", "talk_emphasize"],
    "surprised": ["recoil"],
    "disgusted": ["recoil", "hands_reject"],
    "fearful": ["recoil", "tense"],
    "sarcastic": ["shrug", "talk_open"],
    "thinking": ["think_chin"],
    "calm": ["idle", "talk_open"],
    "apologetic": ["slump", "talk_open"],
    "assertive": ["tense", "talk_emphasize"],
    "concerned": ["talk_open"],
    "encouraging": ["talk_open", "nod_yes"],
}

# Emotion → default base state
EMOTION_DEFAULT_STATE: Dict[str, str] = {
    "sad": "standing",
    "thinking": "standing",
    "happy": "standing",
    "angry": "standing",
}


BODY_MODES = frozenset({"catalog", "momask", "auto", "both", "procedural"})


@dataclass
class DirectorBeat:
    """One directed performance unit."""

    text: str
    emotion: str = "neutral"
    intensity: float = 0.7
    actions: List[str] = field(default_factory=list)
    action_timing: str = "during"
    # Layered body control
    state: str = "standing"
    gesture_target: str = "none"
    hand: str = "right"
    # MoMask open-text body (HumanML3D-style caption). Face never uses this.
    body_mode: str = "auto"  # catalog | momask | auto | both
    humanml_prompt: str = ""
    # Optional: force MoMask length (else speech duration from TTS)
    motion_duration_s: Optional[float] = None  # seconds
    motion_length: Optional[int] = None  # MoMask frames @ 20fps (multiple of 4)
    # MotionRouter (mavie Stage 5 worker pick)
    motion_engines: List[str] = field(default_factory=list)
    motion_reason: str = ""
    allow_sync_gen: bool = False
    motion_only: bool = False
    camera_shot: str = ""
    camera_move: str = ""
    camera_role: str = ""
    look: LookPlan = field(default_factory=LookPlan)
    clip_policy: str = "hold_end"

    def to_face_sentence(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "emotion": self.emotion,
            "intensity": self.intensity,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "emotion": self.emotion,
            "intensity": self.intensity,
            "actions": list(self.actions),
            "action_timing": self.action_timing,
            "state": self.state,
            "gesture_target": self.gesture_target,
            "hand": self.hand,
            "body_mode": self.body_mode,
            "humanml_prompt": self.humanml_prompt,
            "motion_duration_s": self.motion_duration_s,
            "motion_length": self.motion_length,
            "motion_engines": list(self.motion_engines),
            "motion_reason": self.motion_reason,
            "allow_sync_gen": bool(self.allow_sync_gen),
            "motion_only": bool(self.motion_only),
            "camera_shot": self.camera_shot,
            "camera_move": self.camera_move,
            "camera_role": self.camera_role,
            "look": self.look.to_dict() if self.look else LookPlan().to_dict(),
            "clip_policy": self.clip_policy,
        }


def normalize_action(name: str) -> Optional[str]:
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "point": "point_forward",
        "nod": "nod_yes",
        "shake": "shake_no",
        "no": "shake_no",
        "yes": "nod_yes",
        "think": "think_chin",
        "reject": "hands_reject",
        "camera": "look_camera",
    }
    key = aliases.get(key, key)
    return key if key in ACTION_CATALOG else None


def normalize_state(name: str) -> str:
    key = (name or "standing").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "stand": "standing",
        "idle": "standing",
        "standing_idle": "standing",
        "sit": "sitting",
        "seated": "sitting",
        "sitting_idle": "sitting",
        "walk": "walking",
        "walk_loop": "walking",
        "run": "locomotion",
        "running": "locomotion",
        "dance": "dancing",
        "dancing_wave": "dancing",
        "wave_dance": "dancing",
        "loco": "locomotion",
        "travel": "locomotion",
        "moving": "locomotion",
        "low": "grounded",
        "floor": "grounded",
    }
    key = aliases.get(key, key)
    if key in BODY_STATES:
        return key
    # Unknown director state → full-body class, never slam to standing talk
    return "locomotion" if key else "standing"


def is_full_body_state(state: str) -> bool:
    return (state or "").strip().lower() in FULL_BODY_STATES


def is_motion_caption(text: str) -> bool:
    """
    True for a body/stage caption that must not be spoken as dialogue.

    Uses HumanML prefixes, spatial/body language, and short non-dialogue
    instructions — not a closed list of action names.
    """
    t = (text or "").strip()
    if not t:
        return False
    if _HUMANML_PREFIX.match(t):
        return True
    if re.match(r"^[\(\[].+[\)\]]$", t):
        return True
    words = t.split()
    if len(words) > 18 or t.count(".") + t.count("!") + t.count("?") >= 2:
        return False
    if "?" in t:
        return False
    greet = re.sub(r"[^a-z]+", " ", t.lower()).strip()
    if greet in {
        "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
        "yes", "no", "bye", "goodbye",
    }:
        return False
    if re.match(r"(?i)^(hello|hi|hey|thanks|thank|sorry|please)\b", t):
        return False
    if _DIALOGUE_SUBJECT.search(t) and not re.search(
        r"(?i)\b(your|you\s+(walk|sit|stand|go))\b", t
    ):
        return False
    if re.search(r"(?i)\b(say|tell|asking|asked)\b", t):
        return False
    if _SPATIAL_HINT.search(t) and len(words) <= 16:
        return True
    # Short unlabeled physical line (no copula / no spoken-clause shape)
    if (
        1 <= len(words) <= 8
        and not re.search(r"(?i)\b(is|are|was|were|that's|it's|its)\b", t)
        and not re.match(r"(?i)^(that|this|it)\b", t)
    ):
        return True
    return False


def normalize_gesture_target(name: str) -> str:
    key = (name or "none").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "jaw": "chin",
        "face": "chin",
        "head": "forehead",
        "torso": "chest",
        "heart": "chest",
        "hand_to_chest": "chest",
        "hand_to_chin": "chin",
        "hand_to_temple": "temple",
        "hand_to_stomach": "stomach",
        "temple": "temple",
        "stomach": "stomach",
        "belly": "stomach",
        "side": "rest_at_side",
        "rest": "rest_at_side",
        "rest_at_side": "rest_at_side",
        "point": "forward",
        "front": "forward",
        "out": "forward",
        "null": "none",
        "": "none",
        "none": "none",
    }
    key = aliases.get(key, key)
    # rest_at_side = no reach (arms down); treat as none for IK
    if key == "rest_at_side":
        return "none"
    return key if key in GESTURE_TARGETS else "none"


def normalize_hand(name: str) -> str:
    key = (name or "right").strip().lower()
    aliases = {"r": "right", "l": "left", "both_hands": "both", "two": "both"}
    key = aliases.get(key, key)
    return key if key in GESTURE_HANDS else "right"


def gesture_from_actions(actions: Sequence[str]) -> Tuple[str, str]:
    """Derive IK target from known action verbs when LLM only sent actions[]."""
    for a in actions:
        meta = ACTION_CATALOG.get(a) or {}
        gt = meta.get("gesture_target")
        if gt:
            return str(gt), str(meta.get("hand") or "right")
    return "none", "right"


def parse_beat(raw: Dict[str, Any], apply_emotion_defaults: bool = True) -> DirectorBeat:
    # Guide-compatible field aliases (Component A)
    text = str(
        raw.get("text")
        or raw.get("dialogue_text")
        or raw.get("dialogue")
        or raw.get("line")
        or ""
    ).strip()
    emotion = str(
        raw.get("emotion")
        or raw.get("emotion_label")
        or raw.get("mood")
        or "neutral"
    ).lower().strip()
    # common aliases
    emotion = {
        "disgust": "disgusted",
        "surprise": "surprised",
        "fear": "fearful",
    }.get(emotion, emotion)

    try:
        intensity = float(raw.get("intensity", 0.7))
    except (TypeError, ValueError):
        intensity = 0.7
    intensity = max(0.0, min(1.0, intensity))

    timing = str(raw.get("action_timing", "during") or "during").lower().strip()
    if timing not in VALID_ACTION_TIMINGS:
        timing = "during"

    actions_in = raw.get("actions") or raw.get("action") or raw.get("upper_actions") or []
    if isinstance(actions_in, str):
        actions_in = [actions_in]
    actions: List[str] = []
    for a in actions_in:
        n = normalize_action(str(a))
        if n and n not in actions:
            actions.append(n)

    motion_only_hint = bool(raw.get("motion_only")) or (
        not text and bool(raw.get("humanml_prompt") or raw.get("motion_prompt") or "")
    )
    if apply_emotion_defaults and not actions and not motion_only_hint:
        actions = list(EMOTION_DEFAULT_ACTIONS.get(emotion, ["talk_open"]))

    # Base state (guide: base_state)
    state_raw = (
        raw.get("state")
        or raw.get("base_state")
        or raw.get("body_state")
        or raw.get("pose_state")
    )
    if state_raw:
        state = normalize_state(str(state_raw))
    else:
        state = normalize_state(EMOTION_DEFAULT_STATE.get(emotion, "standing"))

    # Semantic gesture (guide: upper_gesture_target / body_target)
    bt = raw.get("body_target") or raw.get("gesture") or {}
    if isinstance(bt, str):
        bt = {"target": bt}
    if not isinstance(bt, dict):
        bt = {}

    g_raw = (
        raw.get("gesture_target")
        or raw.get("upper_gesture_target")
        or raw.get("target")
        or bt.get("target")
        or bt.get("gesture_target")
        or "none"
    )
    h_raw = raw.get("hand") or bt.get("hand") or "right"

    gesture_target = normalize_gesture_target(str(g_raw))
    hand = normalize_hand(str(h_raw))

    # If still none, infer from actions that map to IK
    if gesture_target == "none":
        g2, h2 = gesture_from_actions(actions)
        gesture_target, hand = g2, h2

    body_mode = str(
        raw.get("body_mode") or raw.get("motion_engine") or "auto"
    ).strip().lower()
    if body_mode not in BODY_MODES:
        body_mode = "auto"
    humanml_prompt = str(
        raw.get("humanml_prompt")
        or raw.get("motion_prompt")
        or raw.get("body_prompt")
        or ""
    ).strip()
    motion_engines = raw.get("motion_engines") or raw.get("engines") or []
    if not isinstance(motion_engines, list):
        motion_engines = []
    motion_reason = str(raw.get("motion_reason") or raw.get("reason") or "")
    allow_sync = raw.get("allow_sync_gen")
    if allow_sync is None:
        allow_sync = body_mode in ("momask", "both")
    allow_sync = bool(allow_sync)

    motion_duration_s = raw.get("motion_duration_s") or raw.get("duration_s")
    try:
        motion_duration_s = (
            float(motion_duration_s) if motion_duration_s not in (None, "") else None
        )
    except (TypeError, ValueError):
        motion_duration_s = None
    motion_length = raw.get("motion_length") or raw.get("momask_frames")
    try:
        motion_length = int(motion_length) if motion_length not in (None, "") else None
    except (TypeError, ValueError):
        motion_length = None

    motion_only = bool(raw.get("motion_only"))
    if not motion_only:
        motion_only = (not text) and bool(humanml_prompt or body_mode in ("momask", "both"))
    camera_shot = str(raw.get("camera_shot") or raw.get("shot") or "").strip()
    camera_move = str(raw.get("camera_move") or raw.get("move") or "").strip()
    camera_role = str(raw.get("camera_role") or raw.get("role") or "").strip()

    look_raw = raw.get("look") if isinstance(raw.get("look"), dict) else None
    look = LookPlan.from_dict(look_raw, **raw)
    clip_policy = str(raw.get("clip_policy") or "hold_end").strip().lower()
    if clip_policy not in ("hold_end", "momask_match", "loop", "stretch"):
        clip_policy = "hold_end"
    if clip_policy == "stretch":
        clip_policy = "hold_end"

    return DirectorBeat(
        text=text,
        emotion=emotion,
        intensity=intensity,
        actions=actions,
        action_timing=timing,
        state=state,
        gesture_target=gesture_target,
        hand=hand,
        body_mode=body_mode if body_mode != "both" else "momask",
        humanml_prompt=humanml_prompt,
        motion_duration_s=motion_duration_s,
        motion_length=motion_length,
        motion_engines=[str(e) for e in motion_engines],
        motion_reason=motion_reason,
        allow_sync_gen=allow_sync,
        motion_only=motion_only,
        camera_shot=camera_shot,
        camera_move=camera_move,
        camera_role=camera_role,
        look=look,
        clip_policy=clip_policy,
    )


def parse_director_script(
    data: Dict[str, Any] | List[Any],
    apply_emotion_defaults: bool = True,
) -> List[DirectorBeat]:
    """
    Accepts:
      {"beats": [ ... ]}
      {"sentences": [ ... ]}
      [ {beat}, ... ]
    """
    if isinstance(data, list):
        raw_list = data
    elif isinstance(data, dict):
        raw_list = data.get("beats") or data.get("sentences") or []
    else:
        raw_list = []

    beats: List[DirectorBeat] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        beat = parse_beat(item, apply_emotion_defaults=apply_emotion_defaults)
        if (
            beat.text
            or beat.humanml_prompt
            or beat.body_mode in ("momask", "both")
            or beat.motion_only
        ):
            beats.append(beat)
    return beats


def beats_to_face_sentences(beats: Sequence[DirectorBeat]) -> List[Dict[str, Any]]:
    return [b.to_face_sentence() for b in beats]


def describe_catalog() -> str:
    lines = [
        "Director body model:",
        f"  states: {', '.join(sorted(BODY_STATES))}",
        f"  gesture_targets: {', '.join(sorted(GESTURE_TARGETS))}",
        f"  hands: {', '.join(sorted(GESTURE_HANDS))}",
        "Actions (upper / complex):",
    ]
    for name, meta in sorted(ACTION_CATALOG.items()):
        lines.append(f"  {name:16} [{meta['layer']}] {meta['desc']}")
    return "\n".join(lines)
