"""
motion_controller.py — light policy + clip catalog.

Brain (director) sends menus. Controller resolves:
  clip_id, layer, intensity, seed, loop, fallback

Does not load Blender; emits a MotionPlan for viewers / Blender / hybrid pipeline.
Variation: intensity + seed change phase/amp so same action is not identical.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "body_motion" / "catalog.json"


@dataclass
class MotionPlan:
    """Resolved plan for one beat / frame group."""

    clip_id: str
    action_name: str
    layer: str = "full"  # base | upper | head | full
    loop: bool = False
    intensity: float = 0.75
    seed: int = 0
    # variation parameters derived from seed/intensity
    amp: float = 1.0
    phase: float = 0.0  # 0..1 time offset into clip
    speed: float = 1.0
    hand: str = "right"
    gesture_target: str = "none"
    base_state: str = "standing"
    fallback_clip: str = "idle"
    source_fbx: str = ""
    engine: str = "clip_catalog"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MotionController:
    """
    Small policy over body_motion/catalog.json.

    resolve(command) → MotionPlan
    """

    def __init__(self, catalog_path: Optional[Path] = None) -> None:
        path = catalog_path or DEFAULT_CATALOG
        self.catalog_path = path
        self._clips: Dict[str, Dict[str, Any]] = {}
        self._director_index: Dict[str, str] = {}
        self._load(path)

    def _load(self, path: Path) -> None:
        if not path.is_file():
            self._clips = {}
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self._clips = dict(data.get("clips") or {})
        self._director_index = {}
        for cid, meta in self._clips.items():
            self._director_index[cid.lower()] = cid
            for d in meta.get("director") or []:
                self._director_index[str(d).lower()] = cid
            act = meta.get("action") or cid
            self._director_index[str(act).lower()] = cid

    def reload(self) -> None:
        self._load(self.catalog_path)

    def list_clips(self) -> List[str]:
        return sorted(self._clips.keys())

    def has_clip(self, name: str) -> bool:
        return self._resolve_clip_id(name) is not None

    def _resolve_clip_id(self, name: str) -> Optional[str]:
        if not name:
            return None
        key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "standing": "idle",
            "stand": "idle",
            "walking": "walk",
            "sitting": "sit_idle",
            "sit": "sit_idle",
            "point_forward": "point",
            "think_chin": "think",
            "nod": "nod_yes",
            "yes": "nod_yes",
            "no": "shake_no",
            "angry": "tense",
            "sad": "slump",
            "apologetic": "bow",
            "looking": "look_around",
            "look_camera": "look_around",
        }
        key = aliases.get(key, key)
        if key in self._clips:
            return key
        return self._director_index.get(key)

    def _seed_from(self, text: str, explicit: Optional[int]) -> int:
        if explicit is not None:
            return int(explicit) & 0x7FFFFFFF
        h = hashlib.md5((text or str(random.random())).encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def resolve(
        self,
        *,
        actions: Optional[Sequence[str]] = None,
        state: str = "standing",
        gesture_target: str = "none",
        hand: str = "right",
        emotion: str = "neutral",
        intensity: float = 0.75,
        text: str = "",
        seed: Optional[int] = None,
        prefer: Optional[str] = None,
    ) -> MotionPlan:
        """
        Policy:
          1. prefer explicit clip
          2. base_state locomotion (walk/sit) BEFORE talk_open  ← fixes walk beats
          3. first non-talk action in actions[]
          4. talk_open / emotion / gesture
          5. idle
        """
        intensity = max(0.2, min(1.2, float(intensity)))
        sid = self._seed_from(text or "|".join(actions or []) or state, seed)
        rng = random.Random(sid)

        notes: List[str] = []
        clip_id: Optional[str] = None

        if prefer:
            clip_id = self._resolve_clip_id(prefer)
            if clip_id:
                notes.append(f"prefer:{prefer}")

        # Locomotion / posture from state first (walking must not lose to talk_open)
        st = (state or "standing").lower()
        st_map = {
            "walking": "walk",
            "walk": "walk",
            "sitting": "sit_idle",
            "sit": "sit_idle",
            "dancing": "celebrate",
            "standing": "idle",
            "idle": "idle",
        }
        if not clip_id and st in ("walking", "walk", "sitting", "sit", "dancing"):
            clip_id = self._resolve_clip_id(st_map.get(st, "idle"))
            if clip_id:
                notes.append(f"state_priority:{state}")

        if not clip_id and actions:
            # Prefer non-talk gestures first
            ordered = sorted(
                actions,
                key=lambda a: (0 if "talk" not in str(a).lower() else 1),
            )
            for a in ordered:
                cid = self._resolve_clip_id(str(a))
                if cid:
                    clip_id = cid
                    notes.append(f"action:{a}")
                    break

        if not clip_id and emotion:
            emo_map = {
                "angry": "tense",
                "assertive": "tense",
                "sad": "slump",
                "apologetic": "bow",
                "happy": "celebrate",
                "thinking": "think",
                "disgusted": "hands_reject",
                "surprised": "recoil",
                "fearful": "recoil",
            }
            cand = emo_map.get(emotion.lower())
            if cand and self._resolve_clip_id(cand):
                if not actions:
                    clip_id = self._resolve_clip_id(cand)
                    notes.append(f"emotion:{emotion}")

        if not clip_id and gesture_target and gesture_target not in ("none", "", "rest_at_side"):
            gmap = {
                "chin": "think",
                "forehead": "think",
                "temple": "think",
                "forward": "point",
                "camera": "look_around",
            }
            cand = gmap.get(gesture_target.lower())
            if cand:
                clip_id = self._resolve_clip_id(cand)
                notes.append(f"gesture:{gesture_target}")

        if not clip_id:
            clip_id = self._resolve_clip_id(st_map.get(st, "idle"))
            notes.append(f"state:{state}")

        if not clip_id or clip_id not in self._clips:
            clip_id = "idle" if "idle" in self._clips else next(iter(self._clips), "idle")
            notes.append("fallback:idle")

        meta = self._clips.get(clip_id) or {}
        action_name = meta.get("action") or clip_id
        layer = meta.get("layer") or "full"
        loop = bool(meta.get("loop"))
        fbx = meta.get("fbx") or ""

        # variation
        amp = 0.55 + 0.55 * intensity + rng.uniform(-0.05, 0.08)
        amp = max(0.4, min(1.25, amp))
        phase = rng.random() * (0.35 if loop else 0.15)
        speed = 0.85 + 0.35 * intensity + rng.uniform(-0.08, 0.08)
        speed = max(0.7, min(1.35, speed))

        # base state for viewer layers
        base = "standing"
        if clip_id in ("walk", "walk_back", "run"):
            base = "walking" if clip_id != "run" else "walking"
        elif clip_id in ("sit_idle", "sit"):
            base = "sitting"

        return MotionPlan(
            clip_id=clip_id,
            action_name=action_name,
            layer=layer,
            loop=loop,
            intensity=intensity,
            seed=sid,
            amp=amp,
            phase=phase,
            speed=speed,
            hand=hand or "right",
            gesture_target=gesture_target or "none",
            base_state=base,
            fallback_clip="idle",
            source_fbx=fbx,
            engine="clip_catalog",
            notes=notes,
        )


def plan_from_director_beat(beat: Any, controller: Optional[MotionController] = None) -> MotionPlan:
    """Accept DirectorBeat or dict."""
    ctrl = controller or MotionController()
    if isinstance(beat, dict):
        return ctrl.resolve(
            actions=beat.get("actions") or [],
            state=str(beat.get("state") or beat.get("base_state") or "standing"),
            gesture_target=str(beat.get("gesture_target") or "none"),
            hand=str(beat.get("hand") or "right"),
            emotion=str(beat.get("emotion") or "neutral"),
            intensity=float(beat.get("intensity") or 0.75),
            text=str(beat.get("text") or beat.get("dialogue_text") or ""),
            seed=beat.get("seed"),
            prefer=beat.get("clip_id") or beat.get("prefer_clip"),
        )
    return ctrl.resolve(
        actions=getattr(beat, "actions", None) or [],
        state=getattr(beat, "state", "standing"),
        gesture_target=getattr(beat, "gesture_target", "none"),
        hand=getattr(beat, "hand", "right"),
        emotion=getattr(beat, "emotion", "neutral"),
        intensity=float(getattr(beat, "intensity", 0.75) or 0.75),
        text=getattr(beat, "text", "") or "",
    )


if __name__ == "__main__":
    c = MotionController()
    print("clips:", c.list_clips())
    for sample in (
        {"text": "Hello!", "actions": ["wave"], "emotion": "happy", "intensity": 0.9},
        {"text": "Hmm.", "actions": ["think_chin"], "state": "standing"},
        {"text": "ok", "state": "walking", "actions": []},
        {"text": "eww", "emotion": "disgusted", "actions": ["recoil"]},
    ):
        p = c.resolve(**{k: sample[k] for k in sample if k != "text"}, text=sample["text"])
        print(sample["text"], "→", p.clip_id, "amp", round(p.amp, 2), "phase", round(p.phase, 2), p.notes)
