"""
Session continuity for interactive chat.

One chat session accumulates body clips on a timeline (append, not replace)
and remembers world root / last camera so the next turn continues from
where the character stopped.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cast_schema import CastState
from .look_schema import LookPlan

ROOT = Path(__file__).resolve().parents[1]
SESSION_DIR = ROOT / "temp" / "sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SessionClip:
    """One chat turn's body (+ camera meta) on the session timeline."""

    index: int
    action_name: str = ""
    label: str = ""
    prompt: str = ""
    engine: str = "catalog"  # momask | catalog | procedural
    duration_s: float = 0.0
    t0: float = 0.0  # seconds on session timeline
    t1: float = 0.0
    frame_start: int = 1
    frame_end: int = 1
    camera_shot: str = "MS"
    camera_move: str = "static"
    camera_role: str = "A_cam"
    camera_anchor: str = "chest"  # head | chest | pelvis | full_body
    root_end: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    text: str = ""
    emotion: str = "neutral"
    body_state: str = "standing"
    audio_path: str = ""
    speech_delay_s: float = 0.0
    speech_duration_s: float = 0.0
    speech_frame_start: int = 0
    speech_frame_end: int = 0
    clip_fps: float = 20.0
    look: LookPlan = field(default_factory=LookPlan)
    clip_policy: str = "hold_end"
    action_frames: int = 0
    motion_only: bool = False
    face_timeline_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["root_end"] = list(self.root_end)
        d["look"] = self.look.to_dict() if isinstance(self.look, LookPlan) else dict(self.look or {})
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionClip":
        re = d.get("root_end") or [0.0, 0.0, 0.0]
        look_raw = d.get("look") if isinstance(d.get("look"), dict) else None
        return cls(
            index=int(d.get("index") or 0),
            action_name=str(d.get("action_name") or ""),
            label=str(d.get("label") or d.get("prompt") or d.get("action_name") or ""),
            prompt=str(d.get("prompt") or ""),
            engine=str(d.get("engine") or "catalog"),
            duration_s=float(d.get("duration_s") or 0.0),
            t0=float(d.get("t0") or 0.0),
            t1=float(d.get("t1") or 0.0),
            frame_start=int(d.get("frame_start") or 1),
            frame_end=int(d.get("frame_end") or 1),
            camera_shot=str(d.get("camera_shot") or "MS"),
            camera_move=str(d.get("camera_move") or "static"),
            camera_role=str(d.get("camera_role") or "A_cam"),
            camera_anchor=str(d.get("camera_anchor") or "chest"),
            root_end=(float(re[0]), float(re[1]), float(re[2])),
            text=str(d.get("text") or ""),
            emotion=str(d.get("emotion") or "neutral"),
            body_state=str(d.get("body_state") or "standing"),
            audio_path=str(d.get("audio_path") or ""),
            speech_delay_s=float(d.get("speech_delay_s") or 0.0),
            speech_duration_s=float(d.get("speech_duration_s") or 0.0),
            speech_frame_start=int(d.get("speech_frame_start") or 0),
            speech_frame_end=int(d.get("speech_frame_end") or 0),
            clip_fps=float(d.get("clip_fps") or 20.0),
            look=LookPlan.from_dict(look_raw, **d),
            clip_policy=str(d.get("clip_policy") or "hold_end"),
            action_frames=int(d.get("action_frames") or 0),
            motion_only=bool(d.get("motion_only")),
            face_timeline_path=str(d.get("face_timeline_path") or ""),
        )


@dataclass
class SessionState:
    """
    Persistent across chat turns in one process (and optionally disk).

    - clips: appended each turn (10 chats → 10 clips)
    - root_world: last known character world root (for continuity)
    - frame_cursor: next free frame on the Blender session NLA/timeline
    """

    session_id: str = ""
    clips: List[SessionClip] = field(default_factory=list)
    root_world: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_state: str = "standing"
    last_emotion: str = "neutral"
    last_action: str = ""
    last_camera_shot: str = "MS"
    last_camera_move: str = "static"
    last_camera_role: str = "A_cam"
    last_camera_anchor: str = "chest"
    t_end_s: float = 0.0
    frame_cursor: int = 1  # next NLA start frame
    fps: float = 20.0
    created_at: float = 0.0
    locked_look: LookPlan = field(default_factory=LookPlan)
    cast: CastState = field(default_factory=CastState)
    needs_movie_rerender: bool = False

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        if not self.created_at:
            self.created_at = time.time()

    @property
    def clip_count(self) -> int:
        return len(self.clips)

    @property
    def is_first_turn(self) -> bool:
        return len(self.clips) == 0

    def next_frame_start(self) -> int:
        return max(1, int(self.frame_cursor))

    def append_clip(
        self,
        *,
        action_name: str,
        duration_s: float,
        prompt: str = "",
        engine: str = "catalog",
        camera_shot: str = "MS",
        camera_move: str = "static",
        camera_role: str = "A_cam",
        camera_anchor: str = "chest",
        root_end: Optional[Tuple[float, float, float]] = None,
        text: str = "",
        emotion: str = "neutral",
        body_state: str = "standing",
        action_frames: int = 0,
        clip_fps: float = 20.0,
        audio_path: str = "",
        speech_delay_s: float = 0.0,
        speech_duration_s: float = 0.0,
        look: Any = None,
        clip_policy: str = "",
        motion_only: bool = False,
        face_timeline_path: str = "",
    ) -> SessionClip:
        """Append a finished (or about-to-play) clip onto the session timeline."""
        dur = max(0.05, float(duration_s))
        t0 = float(self.t_end_s)
        t1 = t0 + dur
        f0 = self.next_frame_start()
        # ONLY director/pipeline frame count — never duration*fps (that made 3000+ frames)
        if action_frames and int(action_frames) > 1:
            span = min(214, max(2, int(action_frames)))
        else:
            span = 80  # small default; Blender uses same director rule
        f1 = f0 + span - 1
        fps = float(clip_fps) if float(clip_fps) >= 1.0 else 20.0
        sdelay = max(0.0, float(speech_delay_s or 0.0))
        sdur = max(0.0, float(speech_duration_s or 0.0))
        if sdur <= 0.01 and text:
            sdur = max(0.0, dur - sdelay)
        sf0 = f0 + int(round(sdelay * fps)) if sdur > 0.04 else 0
        sf1 = sf0 + max(1, int(round(sdur * fps))) - 1 if sf0 > 0 else 0
        if sf1 > f1:
            sf1 = f1
        re = root_end if root_end is not None else self.root_world
        look_plan = look if isinstance(look, LookPlan) else LookPlan.from_dict(
            look if isinstance(look, dict) else None
        )
        eng = str(engine or "catalog")
        pol = str(clip_policy or "").lower()
        if pol not in ("hold_end", "momask_match", "loop"):
            if eng == "momask":
                pol = "momask_match"
            elif body_state in ("walking", "sitting", "dancing"):
                pol = "loop"
            else:
                pol = "hold_end"
        if eng == "momask" and pol == "loop":
            pol = "momask_match"
        clip = SessionClip(
            index=len(self.clips) + 1,
            action_name=str(action_name or ""),
            label=str(prompt or action_name or f"clip{len(self.clips) + 1}")[:80],
            prompt=str(prompt or ""),
            engine=eng,
            duration_s=dur,
            t0=t0,
            t1=t1,
            frame_start=f0,
            frame_end=f1,
            camera_shot=str(camera_shot or "MS"),
            camera_move=str(camera_move or "static"),
            camera_role=str(camera_role or "A_cam"),
            camera_anchor=str(camera_anchor or "chest"),
            root_end=(float(re[0]), float(re[1]), float(re[2])),
            text=str(text or "")[:200],
            emotion=str(emotion or "neutral"),
            body_state=str(body_state or "standing"),
            audio_path=str(audio_path or ""),
            speech_delay_s=sdelay,
            speech_duration_s=sdur,
            speech_frame_start=int(sf0),
            speech_frame_end=int(sf1),
            clip_fps=fps,
            look=look_plan,
            clip_policy=pol,
            action_frames=int(action_frames or span),
            motion_only=bool(motion_only) or (sdur <= 0.04 and not text),
            face_timeline_path=str(face_timeline_path or ""),
        )
        self.clips.append(clip)
        self.locked_look = look_plan
        try:
            self.cast.set_hero_outfit(look_plan.wardrobe_id)
            self.cast.extras_count = int(look_plan.extras_count or 0)
            self.cast.extras_preset = str(look_plan.extras_preset or "none")
        except Exception:
            pass
        self.t_end_s = t1
        # gap of 1 frame between strips
        self.frame_cursor = f1 + 2
        self.last_action = clip.action_name
        self.last_emotion = clip.emotion
        self.base_state = clip.body_state
        self.last_camera_shot = clip.camera_shot
        self.last_camera_move = clip.camera_move
        self.last_camera_role = clip.camera_role
        self.last_camera_anchor = clip.camera_anchor
        if root_end is not None:
            self.root_world = (float(re[0]), float(re[1]), float(re[2]))
        return clip

    def remove_clips_overlapping(self, frame_start: int, frame_end: int) -> List["SessionClip"]:
        """
        Drop every clip whose motion span overlaps [frame_start, frame_end].
        Speech-safe: if the range hits a spoken line, the whole clip goes.
        """
        a, b = int(min(frame_start, frame_end)), int(max(frame_start, frame_end))
        keep: List[SessionClip] = []
        removed: List[SessionClip] = []
        for c in self.clips:
            s0 = int(c.speech_frame_start or 0)
            s1 = int(c.speech_frame_end or 0)
            hit_speech = s1 >= s0 > 0 and a < s1 and b > s0
            hit_body = c.frame_end >= a and c.frame_start <= b
            if hit_speech or hit_body:
                removed.append(c)
            else:
                keep.append(c)
        self.clips = keep
        if keep:
            last = keep[-1]
            self.t_end_s = float(last.t1)
            self.frame_cursor = int(last.frame_end) + 2
            self.last_action = last.action_name
            self.last_emotion = last.emotion
            self.base_state = last.body_state
            self.last_camera_shot = last.camera_shot
            self.last_camera_move = last.camera_move
            self.last_camera_role = last.camera_role
            self.last_camera_anchor = last.camera_anchor
        else:
            self.t_end_s = 0.0
            self.frame_cursor = 1
            self.last_action = ""
            self.last_camera_shot = "MS"
            self.last_camera_move = "static"
            self.last_camera_role = "A_cam"
            self.last_camera_anchor = "chest"
        return removed

    def reset(self) -> None:
        """New scene: clear clips, root, cameras continuity."""
        sid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.session_id = sid
        self.clips.clear()
        self.root_world = (0.0, 0.0, 0.0)
        self.base_state = "standing"
        self.last_emotion = "neutral"
        self.last_action = ""
        self.last_camera_shot = "MS"
        self.last_camera_move = "static"
        self.last_camera_role = "A_cam"
        self.last_camera_anchor = "chest"
        self.t_end_s = 0.0
        self.frame_cursor = 1
        self.created_at = time.time()
        self.locked_look = LookPlan()
        self.cast = CastState()
        self.needs_movie_rerender = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "clips": [c.to_dict() for c in self.clips],
            "root_world": list(self.root_world),
            "base_state": self.base_state,
            "last_emotion": self.last_emotion,
            "last_action": self.last_action,
            "last_camera_shot": self.last_camera_shot,
            "last_camera_move": self.last_camera_move,
            "last_camera_role": self.last_camera_role,
            "last_camera_anchor": self.last_camera_anchor,
            "t_end_s": self.t_end_s,
            "frame_cursor": self.frame_cursor,
            "fps": self.fps,
            "created_at": self.created_at,
            "clip_count": self.clip_count,
            "locked_look": self.locked_look.to_dict() if self.locked_look else LookPlan().to_dict(),
            "cast": self.cast.to_dict() if self.cast else CastState().to_dict(),
            "needs_movie_rerender": bool(self.needs_movie_rerender),
        }

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (SESSION_DIR / f"{self.session_id}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        # also write "current" pointer for debugging
        cur = SESSION_DIR / "current_session.json"
        cur.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionState":
        rw = d.get("root_world") or [0.0, 0.0, 0.0]
        st = cls(
            session_id=str(d.get("session_id") or ""),
            root_world=(float(rw[0]), float(rw[1]), float(rw[2])),
            base_state=str(d.get("base_state") or "standing"),
            last_emotion=str(d.get("last_emotion") or "neutral"),
            last_action=str(d.get("last_action") or ""),
            last_camera_shot=str(d.get("last_camera_shot") or "MS"),
            last_camera_move=str(d.get("last_camera_move") or "static"),
            last_camera_role=str(d.get("last_camera_role") or "A_cam"),
            last_camera_anchor=str(d.get("last_camera_anchor") or "chest"),
            t_end_s=float(d.get("t_end_s") or 0.0),
            frame_cursor=int(d.get("frame_cursor") or 1),
            fps=float(d.get("fps") or 20.0),
            created_at=float(d.get("created_at") or 0.0),
            locked_look=LookPlan.from_dict(d.get("locked_look") if isinstance(d.get("locked_look"), dict) else None),
            cast=CastState.from_dict(d.get("cast") if isinstance(d.get("cast"), dict) else None),
            needs_movie_rerender=bool(d.get("needs_movie_rerender")),
        )
        st.clips = [SessionClip.from_dict(c) for c in (d.get("clips") or [])]
        # Keep cast hero outfit aligned with locked look when present.
        try:
            wid = str(getattr(st.locked_look, "wardrobe_id", "") or "")
            if wid:
                st.cast.set_hero_outfit(wid)
            st.cast.extras_count = int(getattr(st.locked_look, "extras_count", st.cast.extras_count) or 0)
            st.cast.extras_preset = str(getattr(st.locked_look, "extras_preset", st.cast.extras_preset) or "none")
        except Exception:
            pass
        return st


# Process-global session (one interactive chat pipeline)
_CURRENT: Optional[SessionState] = None


def get_session(*, reset: bool = False) -> SessionState:
    global _CURRENT
    if reset or _CURRENT is None:
        _CURRENT = SessionState()
        _CURRENT.save()
    return _CURRENT


def reset_session() -> SessionState:
    return get_session(reset=True)


def is_reset_command(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {
        "reset", "reset scene", "new scene", "clear timeline",
        "restart session", "/reset", "start over",
    }
