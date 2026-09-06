"""
motion_router.py — Stage-5 worker selection (see mavie.txt).

NOT a whitelist of verbs (jump/run/…). Decides which *engines* to call:

  catalog     → closed Action library (fast, known clips)
  momask      → open-vocab HumanML text→motion (t2m)
  both        → need catalog clip(s) AND open-vocab motion
                (e.g. talk gesture from catalog + loco from MoMask,
                 or multi-part stage where one part is covered and one is not)
  procedural  → body_agent only (rare fallback)

Principle (mavie Stage 5):
  Director plans WHAT; workers are Brain / catalog / MoMask / procedural.
  This router only answers: which worker(s) for body this beat.

Inputs:
  user_input, spoken line, emotion, candidate actions[], base state
  catalog.json capability index

Outputs:
  body_mode, humanml_prompt, catalog_actions, engines[], reason, allow_sync_gen
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "body_motion" / "catalog.json"


@dataclass
class MotionPlan:
    """Per-beat body engine plan."""

    body_mode: str = "catalog"  # catalog | momask | both | procedural
    engines: List[str] = field(default_factory=lambda: ["catalog"])
    catalog_actions: List[str] = field(default_factory=list)
    humanml_prompt: str = ""
    state: str = "standing"
    # When True, orchestrator may block on gen_t2m (agent chose open model)
    allow_sync_gen: bool = False
    reason: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_catalog() -> Dict[str, Any]:
    if not CATALOG_PATH.is_file():
        return {"clips": {}}
    try:
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"clips": {}}


def _catalog_index(clips: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten clips into searchable capability records."""
    rows = []
    for clip_id, meta in (clips or {}).items():
        if not isinstance(meta, dict):
            continue
        tags = set()
        tags.add(str(clip_id).lower())
        tags.add(str(meta.get("action") or clip_id).lower())
        for d in meta.get("director") or []:
            tags.add(str(d).lower().replace(" ", "_"))
        layer = str(meta.get("layer") or "full").lower()
        rows.append({
            "id": clip_id,
            "action": str(meta.get("action") or clip_id),
            "layer": layer,  # base | upper | full
            "tags": tags,
            "loop": bool(meta.get("loop")),
        })
    return rows


def _split_stage_and_speech(user_input: str) -> tuple[str, str]:
    """Rough split: stage direction vs spoken line (no verb whitelist for engines)."""
    raw = (user_input or "").strip()
    if not raw:
        return "", ""
    # "… say/says/saying <dialogue>"
    m = re.search(
        r"(?is)^(.*?)(?:\b(?:please\s+)?say(?:ing)?\s+(?:this\s+dialog(?:ue)?\s+)?|"
        r"\btell\s+(?:them|him|her|everyone)\s+)(.+)$",
        raw,
    )
    if m:
        return m.group(1).strip(" ,.;"), m.group(2).strip(" \"'")
    # quoted dialogue
    m2 = re.search(r'["\'](.+?)["\']\s*$', raw)
    if m2 and len(m2.group(1).split()) >= 2:
        stage = raw[: m2.start()].strip(" ,.;")
        return stage, m2.group(1).strip()
    # No say-clause — caller decides dialogue-only vs stage-only
    return "", ""


def _looks_like_stage_direction(text: str) -> bool:
    """
    True if text is mostly a physical performance direction (not a spoken monologue).
    Capability-oriented: short imperative / body-centric, not a verb whitelist.
    """
    from .director_schema import is_motion_caption

    t = (text or "").strip()
    if not t:
        return False
    if is_motion_caption(t):
        return True
    words = t.split()
    # Long multi-sentence prose → dialogue
    if len(words) > 18 or t.count(".") + t.count("!") + t.count("?") >= 2:
        return False
    # Catalog capability match (any clip tag — not a hard-coded action list)
    tokens = _tokenize(t)
    physical_hints = 0
    catalog = _load_catalog()
    tags: Set[str] = set()
    for meta in (catalog.get("clips") or {}).values():
        if isinstance(meta, dict):
            tags.add(str(meta.get("action") or "").lower())
            for d in meta.get("director") or []:
                tags.add(str(d).lower())
    for tok in tokens:
        if tok in tags or any(tok in g or g in tok for g in tags if len(g) > 2):
            physical_hints += 1
    if len(words) <= 14 and physical_hints >= 1:
        return True
    if len(words) <= 16 and re.search(r"(?i)\b(and then|then|while|after)\b", t):
        return True
    return False


def _tokenize(text: str) -> Set[str]:
    toks = re.findall(r"[a-z0-9']+", (text or "").lower())
    stop = {
        "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "for", "with",
        "your", "you", "i", "am", "is", "are", "this", "that", "it", "my", "me",
        "while", "then", "last", "at", "say", "says", "saying", "dialog", "dialogue",
        "please", "do", "does", "be", "as", "from",
    }
    return {t for t in toks if t not in stop and len(t) > 1}


def _match_catalog(
    stage_tokens: Set[str],
    actions: Sequence[str],
    catalog_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Capability match: which catalog clips cover parts of the request?
    Score by tag overlap — not a fixed verb→clip table for every word.
    """
    hits = []
    act_set = {str(a).lower().replace(" ", "_") for a in (actions or [])}
    for row in catalog_rows:
        score = 0.0
        # explicit action list from director menus
        if row["action"].lower() in act_set or row["id"].lower() in act_set:
            score += 1.0
        # tag overlap with stage tokens
        overlap = stage_tokens & row["tags"]
        # also soft: token contained in tag or tag in token
        soft = 0
        for t in stage_tokens:
            for tag in row["tags"]:
                if t == tag or (len(t) > 3 and (t in tag or tag in t)):
                    soft += 1
                    break
        score += 0.45 * len(overlap) + 0.25 * min(3, soft)
        if score >= 0.45:
            hits.append({**row, "score": score})
    hits.sort(key=lambda r: -r["score"])
    return hits


def _stage_needs_open_vocab(
    stage: str,
    stage_tokens: Set[str],
    catalog_hits: List[Dict[str, Any]],
) -> bool:
    """
    Open-vocab if stage describes motion that catalog cannot fully cover.

    Heuristics (capability-based, not jump-specific):
      - multi-step stage (and/then/after/finally + enough content)
      - many unmatched stage tokens after removing catalog-covered tags
      - long free-form stage without a strong single catalog hit
    """
    stage = (stage or "").strip()
    if not stage:
        return False

    covered: Set[str] = set()
    for h in catalog_hits[:5]:
        if h["score"] >= 0.7:
            covered |= set(h["tags"])

    residual = set()
    for t in stage_tokens:
        if t in covered:
            continue
        if any(t in c or c in t for c in covered if len(c) > 2):
            continue
        residual.add(t)

    multi_step = bool(
        re.search(r"(?i)\b(and then|then|after that|finally|at last|after)\b", stage)
    ) and len(stage_tokens) >= 3

    strong_hit = bool(catalog_hits and catalog_hits[0]["score"] >= 1.2)
    weak_coverage = len(residual) >= 2 and not strong_hit
    long_stage = len(stage.split()) >= 6 and not strong_hit

    # Dialogue-only residual words (happy, game, yeh) shouldn't force momask alone —
    # residual must look motion-like: verbs/adverbs leftover after speech strip
    speechy = {
        "happily", "happy", "sad", "angry", "game", "won", "win", "yeh", "yeah",
        "hello", "hi", "today", "everyone", "really", "very", "so",
    }
    motion_residual = residual - speechy

    if multi_step and motion_residual:
        return True
    # A single leftover physical token (backflip, cartwheel, …) is enough —
    # catalog already failed to cover it.
    if motion_residual and not strong_hit:
        return True
    if long_stage and motion_residual:
        return True
    if weak_coverage and motion_residual and len(stage.split()) >= 4:
        return True
    return False


def _build_humanml_from_stage(stage: str, emotion: str, spoken: str) -> str:
    """Free-form stage → HumanML-style caption (open vocab for t2m)."""
    stage = re.sub(r"\s+", " ", (stage or "").strip())
    if not stage:
        # emotion-only motion
        emo = (emotion or "neutral").lower()
        if emo in ("happy", "sad", "angry"):
            return f"a person stands and talks looking {emo}"
        return "a person stands gesturing while talking"
    from .momask_body_pipeline import to_humanml_caption

    stage = re.sub(r"(?i)\b(say|tell)\b.*$", "", stage).strip(" ,.;")
    return to_humanml_caption(stage, emotion=emotion)


def route_motion(
    *,
    user_input: str,
    spoken: str = "",
    emotion: str = "neutral",
    intensity: float = 0.7,
    actions: Optional[Sequence[str]] = None,
    state: str = "standing",
    llm_provider: Any = None,
) -> MotionPlan:
    """
    Decide catalog / momask / both / procedural for this beat.
    Optional llm_provider for free-form planning (mavie Stage 2 style).
    """
    acts = [str(a) for a in (actions or []) if a and str(a) != "none"]
    stage, speech_from_split = _split_stage_and_speech(user_input)
    spoken = (spoken or speech_from_split or "").strip()

    # If no "say …" clause, treat full input as dialogue unless it is stage-like
    if not speech_from_split and not spoken:
        if _looks_like_stage_direction(user_input):
            stage = (user_input or "").strip()
            spoken = ""
        else:
            stage = ""
            spoken = (user_input or "").strip()

    # Short greetings must NEVER wait on MoMask gen/retarget
    spoken_low = re.sub(r"[^a-z]+", " ", (spoken or user_input or "").lower()).strip()
    if spoken_low in {
        "hi", "hello", "hey", "hiya", "yo", "sup", "hello there", "hi there",
        "hey there", "good morning", "good evening", "good night", "thanks",
        "thank you", "bye", "goodbye", "ok", "okay", "yes", "no",
    }:
        greet_acts = acts or ["wave", "talk_open"]
        return MotionPlan(
            body_mode="catalog",
            engines=["catalog"],
            catalog_actions=greet_acts,
            humanml_prompt="",
            state=state or "standing",
            allow_sync_gen=False,
            reason="short greeting → catalog wave (skip MoMask)",
            confidence=0.95,
        )

    # Pure dialogue (no stage): catalog talk
    if not stage or stage.lower() == (spoken or "").lower():
        if not acts:
            acts = ["talk_open"]
        return MotionPlan(
            body_mode="catalog",
            engines=["catalog"],
            catalog_actions=acts,
            humanml_prompt="",
            state=state or "standing",
            allow_sync_gen=False,
            reason="dialogue-only → catalog talk/gesture",
            confidence=0.85,
        )

    catalog = _load_catalog()
    rows = _catalog_index(catalog.get("clips") or {})
    stage_tokens = _tokenize(stage)
    hits = _match_catalog(stage_tokens, acts, rows)

    need_open = _stage_needs_open_vocab(stage, stage_tokens, hits)
    covered_actions = [h["action"] for h in hits if h["score"] >= 0.7][:4]
    # keep director-suggested upper actions that are in catalog
    for a in acts:
        if a not in covered_actions:
            # if it's a known catalog id, keep
            if any(r["action"] == a or r["id"] == a for r in rows):
                covered_actions.append(a)

    # LLM refinement (optional) — capability list, not verb hardcoding
    if llm_provider is not None and stage:
        try:
            plan = _route_llm(
                llm_provider,
                user_input=user_input,
                stage=stage,
                spoken=spoken,
                emotion=emotion,
                catalog_rows=rows,
                rule_hint={
                    "need_open": need_open,
                    "catalog_hits": covered_actions,
                },
            )
            if plan is not None:
                return plan
        except Exception as e:
            print(f"[motion_router] LLM route failed ({e}) — rules")

    # Rule composition
    if need_open and covered_actions:
        # both: open full-body sequence + named upper clips available
        hml = _build_humanml_from_stage(stage, emotion, spoken)
        return MotionPlan(
            body_mode="both",
            engines=["momask", "catalog"],
            catalog_actions=covered_actions,
            humanml_prompt=hml,
            state=_infer_state(state, stage_tokens, hits),
            allow_sync_gen=True,
            reason="stage partly catalog-covered + open/residual motion → both",
            confidence=0.7,
        )

    if need_open:
        hml = _build_humanml_from_stage(stage, emotion, spoken)
        return MotionPlan(
            body_mode="momask",
            engines=["momask"],
            catalog_actions=[],
            humanml_prompt=hml,
            state=_infer_state(state, stage_tokens, hits),
            allow_sync_gen=True,
            reason="open-vocab / multi-step stage → MoMask worker",
            confidence=0.75,
        )

    if covered_actions or acts:
        use = covered_actions or acts or ["talk_open"]
        # if only upper + talking, ensure talk_open
        layers = {h["layer"] for h in hits if h["action"] in use or h["id"] in use}
        if spoken and "upper" in layers and "talk_open" not in use:
            use = list(use) + ["talk_open"]
        return MotionPlan(
            body_mode="catalog",
            engines=["catalog"],
            catalog_actions=use,
            humanml_prompt="",
            state=_infer_state(state, stage_tokens, hits),
            allow_sync_gen=False,
            reason="catalog capability match → closed clips",
            confidence=0.8 if covered_actions else 0.55,
        )

    # fallback standing talk
    return MotionPlan(
        body_mode="catalog",
        engines=["catalog"],
        catalog_actions=acts or ["talk_open"],
        humanml_prompt="",
        state=state or "standing",
        allow_sync_gen=False,
        reason="fallback catalog talk_open",
        confidence=0.4,
    )


def _infer_state(
    state: str,
    stage_tokens: Set[str],
    hits: List[Dict[str, Any]],
) -> str:
    """Map to coarse posture/travel class (not a verb menu)."""
    from .director_schema import BODY_STATES, normalize_state

    st = normalize_state(state or "standing")
    for h in hits[:3]:
        if h["score"] < 0.9:
            continue
        tags = {str(t).lower() for t in (h.get("tags") or [])}
        if tags & {"sit", "sitting", "seated"}:
            return "sitting"
        if tags & {"walk", "walking", "run", "running", "jog", "dance", "dancing"}:
            if tags & {"dance", "dancing"}:
                return "dancing"
            return "walking"
    if "sit" in stage_tokens or "sitting" in stage_tokens:
        return "sitting"
    if stage_tokens & {"walk", "walking"}:
        return "walking"
    if any("danc" in t for t in stage_tokens):
        return "dancing"
    # Open stage leftover → traveling full-body class
    if stage_tokens:
        return "locomotion" if st in ("standing", "locomotion") else st
    return st if st in BODY_STATES else "standing"


def _route_llm(
    llm_provider: Any,
    *,
    user_input: str,
    stage: str,
    spoken: str,
    emotion: str,
    catalog_rows: List[Dict[str, Any]],
    rule_hint: Dict[str, Any],
) -> Optional[MotionPlan]:
    """
    One LLM call: pick workers from capability list (mavie Stage 2/5 spirit).
    """
    clip_lines = []
    for r in catalog_rows[:40]:
        clip_lines.append(
            f"- id={r['id']} action={r['action']} layer={r['layer']} tags={sorted(list(r['tags']))[:8]}"
        )
    catalog_txt = "\n".join(clip_lines)
    prompt = f"""You are the body motion ROUTER for a 3D avatar (not the dialogue writer).

Available BODY WORKERS:
1) catalog — play a named clip from the library (fast, fixed motion)
2) momask — text-to-motion model; free-form HumanML caption (slower, open vocabulary)
3) both — use catalog for covered parts AND momask for open/novel full-body motion
4) procedural — simple procedural pose only (last resort)

CATALOG CAPABILITIES (only these names exist for catalog):
{catalog_txt}

USER REQUEST:
{user_input}

Parsed stage (non-speech): {stage!r}
Spoken line: {spoken!r}
Emotion: {emotion}
Rule hint (optional): {json.dumps(rule_hint)}

Choose engines by CAPABILITY fit, not by memorizing verbs.
- If the whole stage is expressible as catalog clip(s) → catalog
- If the stage is novel, multi-step, or not in catalog → momask (humanml_prompt required)
- If some parts are catalog and some need open motion → both

Return ONLY JSON:
{{
  "body_mode": "catalog"|"momask"|"both"|"procedural",
  "engines": ["catalog"] or ["momask"] or ["momask","catalog"],
  "catalog_actions": ["action_id", ...],
  "humanml_prompt": "a person ...",
  "state": "standing"|"sitting"|"walking"|"dancing",
  "allow_sync_gen": true/false,
  "reason": "one short sentence"
}}
"""
    # provider APIs vary
    text = None
    if hasattr(llm_provider, "complete"):
        text = llm_provider.complete(prompt)
    elif hasattr(llm_provider, "chat"):
        text = llm_provider.chat([{"role": "user", "content": prompt}])
    elif callable(llm_provider):
        text = llm_provider(prompt)
    if not text:
        return None
    if isinstance(text, dict):
        data = text
    else:
        raw = str(text).strip()
        # extract JSON object
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        data = json.loads(m.group(0))
    mode = str(data.get("body_mode") or "catalog").lower()
    if mode not in ("catalog", "momask", "both", "procedural"):
        mode = "catalog"
    engines = data.get("engines") or (
        ["catalog"] if mode == "catalog"
        else ["momask"] if mode == "momask"
        else ["momask", "catalog"] if mode == "both"
        else ["procedural"]
    )
    hml = str(data.get("humanml_prompt") or "").strip()
    if mode in ("momask", "both") and len(hml) < 8:
        hml = _build_humanml_from_stage(stage, emotion, spoken)
    allow = bool(data.get("allow_sync_gen", mode in ("momask", "both")))
    return MotionPlan(
        body_mode=mode,
        engines=[str(e) for e in engines],
        catalog_actions=[str(a) for a in (data.get("catalog_actions") or [])],
        humanml_prompt=hml,
        state=str(data.get("state") or "standing"),
        allow_sync_gen=allow,
        reason=str(data.get("reason") or "llm router"),
        confidence=0.82,
    )
