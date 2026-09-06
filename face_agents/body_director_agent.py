"""
body_director_agent.py — LLM (or rule) Director for layered body + dialogue.

User talks in natural language / stage directions. The agent fills:

  {
    "text": "what the avatar SAYS",
    "state": "sitting",           # base layer
    "gesture_target": "chin",     # upper IK
    "hand": "right",
    "emotion": "thinking",
    "intensity": 0.7,
    "actions": ["talk_open"]
  }

Examples of user chat (no JSON needed):
  "say hello while sitting"
  "walk over there and think about it with your hand on your chin"
  "while sitting, tell them that's disgusting"
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .director_schema import (
    ACTION_CATALOG,
    BODY_STATES,
    EMOTION_DEFAULT_ACTIONS,
    GESTURE_TARGETS,
    is_motion_caption,
    parse_director_script,
)

_VALID_EMOTIONS = frozenset({
    "neutral", "happy", "sad", "angry", "surprised", "disgusted", "fearful",
    "sarcastic", "thinking", "calm", "apologetic", "assertive", "concerned",
    "encouraging",
})

DIRECTOR_SYSTEM_PROMPT = """You are the PERFORMANCE DIRECTOR for a talking full-body avatar.

The USER gives natural language or stage directions. YOU figure out:
  - What the character should SAY (dialogue only — clean spoken words)
  - BODY base_state (posture/locomotion)
  - BODY upper_gesture_target (hand reach IK)
  - hand, emotion, intensity, optional actions

You NEVER output bone angles or 3D coordinates.

BODY has TWO engines (pick correctly):
  A) catalog  — known clips (wave, shrug, talk_open, idle sit/stand)
  B) momask   — open full-body from HumanML3D-style English captions
Face / lips are ALWAYS separate. Never put speech words into body captions only.

══════════════════════════════════════
MENUS
══════════════════════════════════════
body_mode:
  catalog | momask | auto
  - catalog: simple greetings, talk gestures, known upper clips
  - momask: walk/run/sit transitions, novel or compound full-body from language
  - auto: system chooses (prefer momask for any full-body / novel motion)

base_state (coarse class, not an action name):
  standing | sitting | walking | dancing | locomotion | grounded

upper_gesture_target:
  none | chin | chest | forehead | temple | mouth | ear | hip | stomach
  | forward | camera | rest_at_side

hand: left | right | both

actions (0–2; catalog upper verbs — NOT a substitute for momask captions):
  talk_open, talk_emphasize, wave, shrug, recoil, celebrate, slump, tense,
  nod_yes, shake_no, point_forward, think_chin, hands_reject, look_camera

humanml_prompt (REQUIRED when body_mode=momask):
  Official MoMask / HumanML3D style ONLY — third-person motion English, e.g.:
    "a person walks forward slowly with the right hand on the chin"
    "a person jumps up and then lands"
    "a person raises the left hand upwards then moves both hands aside"
    "a person bends down and picks something up with the left hand"
    "a man walks forward, turns around, and walks back"
  NOT tags like "walk" or ["walk"]. NOT "hi" / dialogue / "say hello".

motion length (IMPORTANT for momask — no post-cut):
  Prefer motion_length (frames @ 20fps, multiple of 4) or motion_duration_s.
  That value is MoMask --motion_length ONLY (full BVH, nothing trimmed).
  Example jump: motion_length 96–120 or motion_duration_s 5–6.
  If empty, pipeline uses max(speech frames, prompt floor) — jump/walk never
  shrink to a 1–2s stub just because dialogue is short.

emotion_label:
  neutral, happy, sad, angry, surprised, disgusted, fearful, sarcastic,
  thinking, calm, apologetic, assertive, concerned, encouraging

══════════════════════════════════════
EXAMPLES
══════════════════════════════════════
  "say hello while sitting"
    → body_mode: catalog, base_state: sitting, actions: ["wave","talk_open"]
    → dialogue_text: "Hello!"

  "walk over and think with hand on chin"
    → body_mode: momask
    → humanml_prompt: "a person walks forward thoughtfully with one hand on the chin"
    → base_state: walking, emotion_label: thinking
    → dialogue_text: a short thoughtful spoken line

  "just say hi"
    → body_mode: catalog, actions: ["wave"], base_state: standing

SCENE / LOOK (FIRST — Blender builds the set via API before T2M/speech):
  Put place in nested look, NEVER in humanml_prompt. Examples:
    park / garden / outdoors → look.location=park, set_preset=exterior_park
    forest / woods → forest / exterior_forest
    street / city / alley → street / exterior_street
    playground → playground / exterior_playground
    station / train platform → station / exterior_station
    beach → beach / exterior_beach
    room / office → interior_room / interior_simple
    studio / stage → studio or stage / studio_cyc
  User can also say only "scene park" (no motion) — pipeline builds set only.

RULES:
1. dialogue_text = ONLY words the avatar speaks.
2. Face emotion is independent of MoMask; still set emotion_label for lips/face.
3. When body_mode=momask, always fill humanml_prompt (full sentence) with MOTION only (no park/forest/street words).
4. Prefer catalog for tiny social gestures; momask for ANY full-body or novel motion.
5. Prefer 1 beat unless user clearly wants several lines.
6. Optional face tokens only: [eww] [laugh] [hmm] [shocked] [sigh]
7. Always fill look when user mentions a place; scene is built in Blender before body/speech.

Output ONLY valid JSON:
{"beats":[
  {
    "dialogue_text": "Hello!",
    "emotion_label": "happy",
    "intensity": 0.8,
    "body_mode": "catalog",
    "humanml_prompt": "",
    "base_state": "standing",
    "upper_gesture_target": "none",
    "hand": "right",
    "actions": ["wave", "talk_open"],
    "action_timing": "start"
  }
]}

Respond ONLY with JSON."""


def _extract_stage_and_line(user_input: str) -> Dict[str, Any]:
    """
    Parse stage directions out of natural language.
    Returns state, gesture, hand, emotion hints, and cleaned spoken line if found.
    """
    raw = (user_input or "").strip()
    low = raw.lower()
    out: Dict[str, Any] = {
        "state": None,
        "gesture": None,
        "hand": None,
        "emotion": None,
        "intensity": None,
        "actions": [],
        "spoken": None,
        "timing": "during",
    }

    # ── coarse base_state only (engine choice is MotionRouter, not verb lists) ──
    if re.search(
        r"\b(while\s+sitting|sit\s+down|seated|on\s+the\s+chair|from\s+your\s+seat|"
        r"sitting\s+down|remain\s+seated|stay\s+seated)\b",
        low,
    ) or re.search(r"\bsitting\b", low):
        out["state"] = "sitting"
    elif re.search(
        r"\b(while\s+walking|walk\s+over|walk\s+forward|walk\s+a\s+few|"
        r"walking|walk\s+to|go\s+over|come\s+here|"
        r"walk\s+and|start\s+walking|as\s+you\s+walk|\bwalks?\b)\b",
        low,
    ):
        out["state"] = "walking"
    elif re.search(r"\b(dance|dancing|party\s+move|boogie)\b", low):
        out["state"] = "dancing"
    elif re.search(r"\b(stand\s+up|standing|while\s+standing)\b", low):
        out["state"] = "standing"
    elif is_motion_caption(raw):
        # Any other physical stage (not a named verb list) → traveling class
        out["state"] = "locomotion"

    # ── gesture_target + hand ─────────────────────────────────────────────
    if re.search(
        r"\b(hand\s+on\s+(your\s+)?chin|to\s+(your\s+)?chin|touch\s+(your\s+)?chin|"
        r"chin\s+gesture|think.*chin|chin\s+think)\b",
        low,
    ):
        out["gesture"] = "chin"
        out["hand"] = "right"
        out["emotion"] = out["emotion"] or "thinking"
    elif re.search(
        r"\b(hand\s+on\s+(your\s+)?chest|to\s+(your\s+)?chest|hand\s+to\s+chest|"
        r"clutch\s+(your\s+)?chest|over\s+(your\s+)?heart)\b",
        low,
    ):
        out["gesture"] = "chest"
        out["hand"] = "right"
    elif re.search(r"\b(hand\s+on\s+(your\s+)?(forehead|temple)|facepalm)\b", low):
        out["gesture"] = "temple"
        out["hand"] = "right"
    elif re.search(r"\b(wave|waving|greet)\b", low):
        out["gesture"] = "camera"
        out["actions"] = ["wave", "talk_open"]
        out["emotion"] = out["emotion"] or "happy"
        out["timing"] = "start"
    elif re.search(r"\b(shrug|whatever)\b", low):
        out["gesture"] = "none"
        out["actions"] = ["shrug", "talk_open"]
        out["emotion"] = out["emotion"] or "sarcastic"
    elif re.search(r"\b(both\s+hands|hands\s+up|push\s+away)\b", low):
        out["gesture"] = "forward"
        out["hand"] = "both"
        out["actions"] = ["hands_reject"]
    elif re.search(r"\b(nod|yes\s+i\s+agree|i\s+agree)\b", low):
        out["actions"] = ["nod_yes", "talk_open"]
        out["emotion"] = out["emotion"] or "encouraging"
    elif re.search(r"\b(shake\s+(your\s+)?head|no\s+way|disagree)\b", low):
        out["actions"] = ["shake_no", "talk_open"]
        out["emotion"] = out["emotion"] or "assertive"
    elif re.search(r"\b(look\s+at\s+(the\s+)?camera|face\s+the\s+camera|look\s+here)\b", low):
        out["gesture"] = "camera"
        out["actions"] = ["look_camera", "talk_open"]
    elif re.search(r"\b(look\s+left|glance\s+left|to\s+the\s+left)\b", low):
        out["actions"] = ["look_left", "talk_open"]
    elif re.search(r"\b(look\s+right|glance\s+right|to\s+the\s+right)\b", low):
        out["actions"] = ["look_right", "talk_open"]
    elif re.search(r"\b(shift\s+(your\s+)?weight|weight\s+shift)\b", low):
        out["actions"] = ["weight_shift", "talk_open"]
    elif re.search(r"\b(point\s+(at|to|forward)|pointing)\b", low):
        out["gesture"] = "forward"
        out["hand"] = "right"
        out["actions"] = ["point_forward", "talk_open"]

    if re.search(r"\b(left\s+hand)\b", low):
        out["hand"] = "left"
    elif re.search(r"\b(right\s+hand)\b", low):
        out["hand"] = "right"
    elif re.search(r"\b(both\s+hands)\b", low):
        out["hand"] = "both"

    # ── emotion keywords ──────────────────────────────────────────────────
    emo_map = [
        (("eww", "disgust", "gross", "nasty", "sick", "revolting"), "disgusted", 0.9, "chest", ["recoil"]),
        (("angry", "furious", "mad", "unacceptable", "outrage"), "angry", 0.85, "forward", ["tense", "talk_emphasize"]),
        (("scared", "afraid", "fear", "terrified", "nervous"), "fearful", 0.8, "chest", ["recoil"]),
        (("think", "hmm", "ponder", "consider", "let me think"), "thinking", 0.65, "chin", ["talk_open"]),
        (("haha", "funny", "hilarious", "lol", "joke"), "happy", 0.75, "none", ["celebrate", "talk_open"]),
        (("sorry", "apolog"), "apologetic", 0.65, "none", ["slump", "talk_open"]),
        (("sad", "unhappy", "depressed"), "sad", 0.7, "none", ["slump", "talk_open"]),
        (("hello", "hi ", "hey", "welcome", "greetings"), "happy", 0.8, "camera", ["wave", "talk_open"]),
        (("surprising", "shocked", "oh my god", "wow"), "surprised", 0.85, "chest", ["recoil"]),
    ]
    for keys, emo, inten, gest, acts in emo_map:
        if any(k in low for k in keys):
            out["emotion"] = out["emotion"] or emo
            out["intensity"] = out["intensity"] or inten
            if out["gesture"] is None and gest != "none":
                out["gesture"] = gest
            if not out["actions"]:
                out["actions"] = list(acts)
            if emo in ("happy", "surprised", "disgusted", "angry", "fearful") and keys[0] in (
                "hello", "hi ", "hey", "welcome", "eww", "angry", "scared", "surprising",
            ):
                out["timing"] = "start"
            break

    # ── extract spoken line from "say …" / "tell them …" ──────────────────
    spoken = None
    patterns = [
        # say "quoted"
        r"""(?i)(?:please\s+)?say\s+["'](.+?)["']""",
        r"""(?i)(?:please\s+)?say\s+(?:that\s+)?(.+?)(?:\s+while\s+|\s+as\s+you\s+|\s+and\s+(?:sit|walk|stand|dance)|$)""",
        # "while saying <dialogue>" / "saying <dialogue>"
        r"""(?i)while\s+saying\s+(.+)$""",
        r"""(?i)(?<![a-z])saying\s+(.+)$""",
        r"""(?i)tell\s+(?:them|him|her|everyone|us)\s+["'](.+?)["']""",
        r"""(?i)tell\s+(?:them|him|her|everyone|us)\s+(.+?)(?:\s+while\s+|\s+as\s+you\s+|$)""",
        r"""(?i)speak\s+(?:the\s+words?\s+)?["'](.+?)["']""",
        r"""(?i)(?:reply|respond|answer)\s+(?:with\s+)?["'](.+?)["']""",
    ]
    for pat in patterns:
        m = re.search(pat, raw)
        if m:
            spoken = m.group(1).strip()
            # strip trailing stage bits that leaked in
            spoken = re.sub(
                r"(?i)\s*(while\s+(sitting|walking|standing|dancing).*)\s*$",
                "",
                spoken,
            ).strip()
            spoken = re.sub(
                r"(?i)\s*(with\s+(your\s+)?hand\s+on.*)\s*$",
                "",
                spoken,
            ).strip(" ,.;")
            if spoken:
                break

    # "while sitting, <dialogue>" or "while walking: <dialogue>"
    if not spoken:
        m = re.search(
            r"(?i)while\s+(sitting|walking|standing|dancing)\s*[,:-]\s*(.+)$",
            raw,
        )
        if m:
            out["state"] = {
                "sitting": "sitting",
                "walking": "walking",
                "standing": "standing",
                "dancing": "dancing",
            }[m.group(1).lower()]
            spoken = m.group(2).strip().strip("\"'")

    # If user only gave stage direction without dialogue, invent a short line later
    out["spoken"] = spoken
    return out


def _default_line_for(emotion: str, state: str) -> str:
    if emotion == "happy":
        return "Hello — great to see you!"
    if emotion == "thinking":
        return "Hmm, let me think about that for a second."
    if emotion == "disgusted":
        return "Eww, that is disgusting."
    if emotion == "angry":
        return "That is completely unacceptable."
    if emotion == "fearful":
        return "I'm not sure this is safe."
    if emotion == "sarcastic":
        return "Oh sure, that makes perfect sense."
    if state == "sitting":
        return "I'm comfortable right here."
    if state == "walking":
        return "I'm on my way."
    return "Okay, got it."


class BodyDirectorAgent:
    """Natural language / stage directions → pipeline beats."""

    name = "body_director"

    def __init__(self, llm_provider: Any = None) -> None:
        self.llm = llm_provider
        self._prev_state = "standing"
        self._prev_look: Optional[Dict[str, Any]] = None

    def direct(
        self,
        user_input: str,
        *,
        previous_state: Optional[str] = None,
        scene_note: str = "",
    ) -> List[Dict[str, Any]]:
        if previous_state:
            self._prev_state = previous_state

        text = (user_input or "").strip()
        if not text:
            return [self._fallback_beat("...", "neutral", 0.5)]

        # Raw JSON paste
        if text.lstrip().startswith("{") or text.lstrip().startswith("["):
            try:
                data = json.loads(text)
                items = self._normalize_items(
                    data if isinstance(data, list)
                    else (data.get("beats") or data.get("sentences") or [data])
                )
                return [self._ensure_look(b, text) for b in items]
            except json.JSONDecodeError:
                pass

        stage = _extract_stage_and_line(text)

        if self.llm is not None:
            try:
                beats = self._direct_llm(text, scene_note=scene_note, stage=stage)
                if beats:
                    beats = [self._merge_stage(b, stage) for b in beats]
                    beats = [self._ensure_humanml(b, text) for b in beats]
                    beats = [self._ensure_look(b, text) for b in beats]
                    self._prev_state = beats[-1].get("state", self._prev_state)
                    return beats
            except Exception as e:
                print(f"[body_director] LLM failed ({e}) — rule fallback")

        beats = self._direct_rules(text, stage)
        return [self._ensure_look(b, text) for b in beats]

    def _ensure_look(self, beat: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Attach LookPlan (scene/lights). Never writes into humanml_prompt."""
        from .look_agent import plan_look
        from .look_schema import LookPlan

        b = dict(beat)
        # parse_beat always fills a default studio look — only treat as explicit
        # when the user/LLM set a non-default location/time/mood/hdri.
        explicit = None
        ld = b.get("look") if isinstance(b.get("look"), dict) else None
        if ld:
            loc = str(ld.get("location") or "studio")
            tod = str(ld.get("time_of_day") or "day")
            mood = str(ld.get("light_mood") or "soft")
            if (
                loc not in ("studio", "default", "")
                or tod not in ("day", "")
                or mood not in ("soft", "neutral", "")
                or ld.get("notes")
                or ld.get("hdri_name")
            ):
                explicit = ld
        prev = None
        if self._prev_look:
            try:
                prev = LookPlan.from_dict(self._prev_look)
            except Exception:
                prev = None
        look = plan_look(user_input, prev_look=prev, explicit=explicit)
        b["look"] = look.to_dict()
        self._prev_look = look.to_dict()
        print(
            f"[body_director] look ← {look.location}/{look.time_of_day}/"
            f"{look.light_mood} set={look.set_preset}"
        )
        return b

    def _ensure_humanml(self, beat: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """MoMask never sees raw chat — director always writes HumanML3D English."""
        b = dict(beat)
        mode = str(b.get("body_mode") or "auto").lower()
        hml = str(b.get("humanml_prompt") or "").strip()
        if mode in ("catalog", "procedural") and not b.get("motion_only"):
            return b
        want = (
            mode in ("momask", "both")
            or bool(b.get("motion_only"))
            or is_motion_caption(user_input)
            or is_motion_caption(hml)
        )
        if not want:
            return b
        from .momask_body_pipeline import to_humanml_caption

        src = hml or user_input
        rewritten = to_humanml_caption(
            src, emotion=str(b.get("emotion") or ""), state=str(b.get("state") or ""),
        )
        if rewritten != hml:
            print(f"[body_director] humanml ← {rewritten!r} (from {src[:48]!r})")
        b["humanml_prompt"] = rewritten
        if b.get("body_mode") == "auto":
            b["body_mode"] = "momask"
        return b

    def _direct_llm(
        self,
        user_input: str,
        scene_note: str = "",
        stage: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        from llm_fw.providers import LLMMessage

        hints = []
        if scene_note:
            hints.append(f"Scene note: {scene_note}")
        if self._prev_state:
            hints.append(f"Previous base_state: {self._prev_state}")
        if stage:
            if stage.get("state"):
                hints.append(f"Detected stage base_state hint: {stage['state']}")
            if stage.get("gesture"):
                hints.append(
                    f"Detected gesture hint: {stage['gesture']} hand={stage.get('hand') or 'right'}"
                )
            if stage.get("spoken"):
                hints.append(f"Detected spoken line hint: {stage['spoken']}")

        preamble = ("\n".join(hints) + "\n\n") if hints else ""
        user_msg = (
            preamble
            + "USER DIRECTION (parse stage directions + invent clean dialogue if needed):\n"
            + user_input
        )

        messages = [LLMMessage(role="user", content=user_msg)]
        result = self.llm.chat_json(messages, system=DIRECTOR_SYSTEM_PROMPT)

        if isinstance(result, dict) and "parse_error" in result:
            return []

        items: List[Any] = []
        if isinstance(result, dict):
            items = result.get("beats") or result.get("sentences") or []
            if not items and (
                result.get("dialogue_text") or result.get("text") or result.get("emotion_label")
            ):
                items = [result]
        return self._normalize_items(items)

    def _merge_stage(self, beat: Dict[str, Any], stage: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure rule-detected sit/walk/chin are not dropped if LLM forgot them."""
        b = dict(beat)
        if stage.get("state"):
            # Stage direction wins for locomotion/posture
            b["state"] = stage["state"]
        if stage.get("gesture") and (
            not b.get("gesture_target") or b.get("gesture_target") == "none"
        ):
            b["gesture_target"] = stage["gesture"]
        if stage.get("hand"):
            b["hand"] = stage["hand"]
        if stage.get("spoken"):
            # Prefer extracted spoken line if LLM left stage words in dialogue
            t = (b.get("text") or "").lower()
            if "while sit" in t or "while walk" in t or t.startswith("say "):
                b["text"] = stage["spoken"]
        if stage.get("emotion") and b.get("emotion") in (None, "neutral"):
            b["emotion"] = stage["emotion"]
        return b

    def _direct_rules(
        self,
        user_input: str,
        stage: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        stage = stage or _extract_stage_and_line(user_input)
        low = user_input.lower()

        state = stage.get("state") or (
            self._prev_state if self._prev_state in BODY_STATES else "standing"
        )
        # Dialogue-only chat should not keep a previous traveling state forever
        if stage.get("state") is None:
            if is_motion_caption(user_input):
                state = "locomotion"
            elif not re.search(r"(?i)\b(say|tell|while)\b", low):
                if not is_motion_caption(user_input):
                    state = "standing"

        emotion = stage.get("emotion") or "neutral"
        intensity = float(stage.get("intensity") or 0.7)
        gesture = stage.get("gesture") or "none"
        hand = stage.get("hand") or "right"
        actions = list(stage.get("actions") or [])
        timing = stage.get("timing") or "during"

        if not actions and not is_motion_caption(user_input):
            actions = list(EMOTION_DEFAULT_ACTIONS.get(emotion, ["talk_open"]))

        # Spoken line
        line = stage.get("spoken")
        generic = {
            "something", "it", "this", "that", "stuff", "things", "a thing",
            "anything", "whatever", "words", "a line", "the line",
        }
        if line and line.lower().strip(" !.?,\"'") in generic:
            line = None

        if not line:
            # Pure motion caption → silent, NOT speak the caption
            if is_motion_caption(user_input):
                line = ""
            elif not re.search(r"(?i)\b(say|tell|while)\b", user_input):
                line = user_input.strip()
            else:
                line = _default_line_for(emotion, state)

        # Never speak stage / HumanML captions as dialogue
        if line and is_motion_caption(line):
            line = ""

        # Capitalize lightly (skip empty motion-only)
        if line:
            if line[0].islower():
                line = line[0].upper() + line[1:]
            if line[-1] not in ".!?":
                if len(line.split()) > 2:
                    line = line + (
                        "!" if emotion in ("happy", "angry", "surprised") else "."
                    )

        # ── Stage-5 worker selection (mavie.txt): MotionRouter ────────────
        # Picks catalog / momask / both / procedural from catalog capabilities
        # + open-vocab residual — NOT a hard-coded jump/run if-list.
        from .motion_router import route_motion

        mplan = route_motion(
            user_input=user_input,
            spoken=line or "",
            emotion=emotion,
            intensity=intensity,
            actions=actions,
            state=state,
            llm_provider=getattr(self, "llm", None),
        )
        body_mode = mplan.body_mode if mplan.body_mode in (
            "catalog", "momask", "both", "auto", "procedural",
        ) else "catalog"
        # Pipeline understands catalog | momask; "both" → momask primary + catalog acts
        if body_mode == "both":
            body_mode = "momask"
        if body_mode == "procedural":
            body_mode = "catalog"
        humanml_prompt = mplan.humanml_prompt or ""
        # Short spoken greetings: never wait on MoMask even if LLM said momask
        _greet = re.sub(r"[^a-z]+", " ", (line or user_input or "").lower()).strip()
        if _greet in {
            "hi", "hello", "hey", "hiya", "yo", "hello there", "hi there", "hey there",
        }:
            body_mode = "catalog"
            humanml_prompt = ""
            mplan.allow_sync_gen = False
            if not actions:
                actions = ["wave", "talk_open"]
        # Motion caption → director HumanML (never raw user text to gen_t2m)
        if is_motion_caption(user_input) and body_mode != "catalog":
            from .momask_body_pipeline import to_humanml_caption
            humanml_prompt = to_humanml_caption(
                humanml_prompt or user_input, emotion=emotion, state=state,
            )
            if body_mode not in ("momask", "both"):
                body_mode = "momask"
            if not line:
                actions = [a for a in actions if a not in ("talk_open", "talk_emphasize")]
        if mplan.catalog_actions:
            # merge router catalog picks with menu actions (unique, preserve order)
            merged = list(mplan.catalog_actions)
            for a in actions:
                if a not in merged:
                    merged.append(a)
            actions = merged
        if (not line) and (humanml_prompt or is_motion_caption(user_input)):
            actions = [a for a in actions if a not in ("talk_open", "talk_emphasize")]
        if mplan.state in BODY_STATES:
            state = mplan.state
        elif mplan.state:
            state = "locomotion"

        beat = {
            "text": (line or "")[:300],
            "emotion": emotion if emotion in _VALID_EMOTIONS else "neutral",
            "intensity": max(0.0, min(1.0, intensity)),
            "actions": actions,
            "action_timing": timing,
            "state": state if state in BODY_STATES else "locomotion",
            "gesture_target": gesture if gesture in GESTURE_TARGETS or gesture == "none" else "none",
            "hand": hand if hand in ("left", "right", "both") else "right",
            "body_mode": body_mode,
            "humanml_prompt": humanml_prompt,
            "motion_engines": list(mplan.engines),
            "motion_reason": mplan.reason,
            "allow_sync_gen": (
                bool(mplan.allow_sync_gen) or is_motion_caption(user_input)
            ) and body_mode not in ("catalog", "procedural"),
            "motion_only": (not line) and bool(humanml_prompt or is_motion_caption(user_input)),
        }
        print(
            f"[body_director] motion_router → mode={body_mode} engines={mplan.engines} "
            f"reason={mplan.reason!r} hml={(humanml_prompt[:60] + '…') if len(humanml_prompt) > 60 else humanml_prompt!r}"
        )
        cleaned = self._normalize_items([self._ensure_humanml(beat, user_input)])
        if cleaned:
            self._prev_state = cleaned[-1].get("state", self._prev_state)
            return cleaned
        self._prev_state = beat["state"]
        return [self._ensure_humanml(beat, user_input)]

    def _normalize_items(self, items: Sequence[Any]) -> List[Dict[str, Any]]:
        if not items:
            return []
        beats = parse_director_script({"beats": list(items)}, apply_emotion_defaults=True)
        out: List[Dict[str, Any]] = []
        for b in beats:
            d = b.to_dict()
            if d.get("emotion") not in _VALID_EMOTIONS:
                d["emotion"] = "neutral"
            out.append(d)
        return out

    @staticmethod
    def _fallback_beat(text: str, emotion: str, intensity: float) -> Dict[str, Any]:
        return {
            "text": text,
            "emotion": emotion if emotion in _VALID_EMOTIONS else "neutral",
            "intensity": max(0.0, min(1.0, intensity)),
            "actions": list(EMOTION_DEFAULT_ACTIONS.get(emotion, ["talk_open"])),
            "action_timing": "during",
            "state": "standing",
            "gesture_target": "none",
            "hand": "right",
            "body_mode": "catalog",
            "humanml_prompt": "",
        }


def director_system_prompt() -> str:
    return DIRECTOR_SYSTEM_PROMPT


def list_menus() -> Dict[str, List[str]]:
    return {
        "base_state": sorted(BODY_STATES),
        "upper_gesture_target": sorted(GESTURE_TARGETS),
        "actions": sorted(ACTION_CATALOG.keys()),
        "emotions": sorted(_VALID_EMOTIONS),
    }
