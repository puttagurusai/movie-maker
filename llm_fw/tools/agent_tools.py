"""
Wrappers around existing face_agents so any LLM uses the same specialists
ChatGPT-style: brain picks tools; tools call Look / Body / Camera / speech / face.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from .base import Tool, ToolRegistry, ToolResult


def _coord():
    from face_agents.coordinator import FaceCoordinator

    return FaceCoordinator(
        udp_ip=os.environ.get("FACE_UDP_IP", "127.0.0.1"),
        udp_port=int(os.environ.get("FACE_UDP_PORT", "9001")),
    )


def register_agent_tools(reg: ToolRegistry, *, agent_names: Optional[List[str]] = None) -> None:
    agents = agent_names if agent_names is not None else ["movie_os", "director"]

    def look_plan_from_text(text: str, apply: bool = True) -> ToolResult:
        """Use LookAgent to infer location/set/wardrobe/extras from natural language."""
        try:
            from face_agents.look_agent import plan_look

            plan = plan_look(text or "")
            data = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan or {})
            if apply:
                _coord().send_udp({"type": "look", "op": "apply", "look": data})
                time.sleep(0.15)
            return ToolResult(ok=True, data=data)
        except Exception as e:
            return ToolResult(ok=False, error=str(e))

    def body_direct(text: str, duration: float = 3.0, append: bool = True) -> ToolResult:
        """
        Route stage text through motion router / BodyDirector intent → catalog or MoMask hint,
        then optionally append to Session.
        """
        try:
            from face_agents.motion_router import route_motion

            mplan = route_motion(
                user_input=text or "",
                spoken="",
                emotion="neutral",
                intensity=0.7,
                actions=[],
                state="standing",
                llm_provider=None,
            )
            if hasattr(mplan, "to_dict"):
                md = mplan.to_dict()
            elif isinstance(mplan, dict):
                md = dict(mplan)
            else:
                md = {}
            acts = [str(a).lower() for a in (md.get("catalog_actions") or md.get("actions") or [])]
            mode = str(md.get("body_mode") or "catalog").lower()
            eng = "momask" if mode in ("momask", "both") else "catalog"
            act = ""
            for prefer in ("wave", "walk", "run", "talk_open", "celebrate", "shrug", "idle"):
                if prefer in acts:
                    act = prefer
                    break
            if not act:
                st = str(md.get("state") or "").lower()
                low = (text or "").lower()
                if "wave" in low:
                    act = "wave"
                elif "walk" in low or "walk" in st:
                    act = "walk"
                elif "run" in low:
                    act = "run"
                else:
                    act = "talk_open"
            # Session append needs a catalog Action name unless MoMask Action already exists
            if eng == "momask":
                eng = "catalog"  # sync gen is movie_run_story's job; tools stay fast
            out = {"motion_plan": md, "action": act, "engine": eng}
            if append:
                fps = 20.0 if eng == "momask" else 30.0
                frames = max(8, int(round(float(duration) * 20.0)))
                pkt = {
                    "type": "body",
                    "action": act,
                    "duration": float(duration),
                    "append_timeline": True,
                    "append_only": True,
                    "live": False,
                    "restart": True,
                    "action_frames": frames,
                    "motion_length": frames,
                    "engine": eng if eng != "open" else "catalog",
                    "clip_policy": "loop" if act in ("idle", "talk_open", "wave") else "hold_end",
                    "clip_label": act,
                    "clip_fps": fps,
                }
                _coord().send_udp(pkt)
                out["appended"] = True
            return ToolResult(ok=True, data=out)
        except Exception as e:
            return ToolResult(ok=False, error=str(e))

    def camera_direct(
        text: str = "",
        duration: float = 4.0,
        shot: str = "",
        move_type: str = "",
        emotion: str = "neutral",
        state: str = "standing",
    ) -> ToolResult:
        """CameraAgent plan_camera_for_beat → SessionCam bake."""
        try:
            from face_agents.camera_agent import plan_camera_for_beat

            cam = plan_camera_for_beat(
                duration_s=float(duration),
                emotion=emotion or "neutral",
                intensity=0.7,
                state=state or "standing",
                actions=[],
                fps=20.0,
                shot=shot or None,
                move_type=move_type or None,
                track_head=False,
                humanml_prompt=text or "",
            )
            cam_d = cam.to_dict() if hasattr(cam, "to_dict") else dict(cam or {})
            pkt = {
                "type": "camera",
                "op": "plan",
                "duration": float(duration),
                "fps": 20.0,
                "shot": cam_d.get("shot") or shot or "MS",
                "move_type": cam_d.get("move_type") or move_type or "static",
                "track_head": False,
                "camera_name": cam_d.get("camera_name") or "MovieCam_A",
                "look_at_name": cam_d.get("look_at_name") or "MovieCam_A_LookAt",
                "camera_role": cam_d.get("camera_role") or "A_cam",
                "set_scene_camera": True,
                "bake_keyframes": True,
                "clear_previous": False,
                "hold_after": False,
                "live": False,
                "keyframes": list(cam_d.get("keyframes") or []),
            }
            c = _coord()
            c.send_udp(pkt)
            c.send_udp({"type": "camera", "op": "rest", "hold_after": False})
            return ToolResult(ok=True, data={"camera": cam_d, "sent": True})
        except Exception as e:
            return ToolResult(ok=False, error=str(e))

    def speech_say(
        text: str,
        emotion: str = "neutral",
        intensity: float = 0.72,
        body_action: str = "talk_open",
        hold_before_s: float = 0.2,
    ) -> ToolResult:
        """
        TTS for one line + Session talk body append.
        For full multi-beat films with face timelines prefer movie_run_story.
        """
        try:
            from pathlib import Path
            import soundfile as sf
            import numpy as np
            from parler_voice import build_voice_style, generate_speech, load_parler

            speak = (text or "").strip()
            if not speak:
                return ToolResult(ok=False, error="empty text")
            out_dir = Path(__file__).resolve().parents[2] / "temp" / "agentic_speech"
            out_dir.mkdir(parents=True, exist_ok=True)
            wav_path = out_dir / f"say_{int(time.time())}.wav"
            load_parler()
            style = build_voice_style(emotion, float(intensity))
            generate_speech(speak, style, str(wav_path), play_audio=False)
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            speech_dur = float(len(audio) / float(sr))
            n_before = int(round(max(0.0, float(hold_before_s)) * sr))
            if n_before > 0:
                audio = np.concatenate([np.zeros(n_before, dtype=np.float32), np.asarray(audio, dtype=np.float32)])
                sf.write(str(wav_path), audio, sr)
            take_dur = float(len(audio) / float(sr))

            frames = max(8, int(round(take_dur * 20.0)))
            _coord().send_udp({
                "type": "body",
                "action": body_action or "talk_open",
                "duration": take_dur,
                "append_timeline": True,
                "append_only": True,
                "live": False,
                "restart": True,
                "action_frames": frames,
                "motion_length": frames,
                "engine": "catalog",
                "clip_policy": "loop",
                "text": speak,
                "audio_path": str(wav_path),
                "speech_duration_s": speech_dur,
                "speech_delay_s": float(hold_before_s),
                "clip_label": "speech_say",
                "clip_fps": 30.0,
            })
            return ToolResult(
                ok=True,
                data={
                    "spoken": speak,
                    "speech_duration_s": speech_dur,
                    "take_duration_s": take_dur,
                    "audio_path": str(wav_path),
                    "body_action": body_action or "talk_open",
                    "note": "Use movie_run_story for full lips/face timeline bake",
                },
            )
        except Exception as e:
            return ToolResult(ok=False, error=str(e))

    def face_emotion(emotion: str = "neutral", intensity: float = 0.7) -> ToolResult:
        """Update face policy emotion (region agents / Brain)."""
        try:
            from llm_fw.tools.face_tools import FACE_POLICY, get_face_policy

            FACE_POLICY["emotion"] = (emotion or "neutral").lower().strip()
            FACE_POLICY["intensity"] = max(0.0, min(1.0, float(intensity)))
            return ToolResult(ok=True, data=get_face_policy())
        except Exception as e:
            return ToolResult(ok=False, error=str(e))

    specs = [
        (
            "look_plan_from_text",
            "LookAgent: infer and optionally apply Look_Set from natural language",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "apply": {"type": "boolean"},
                },
                "required": ["text"],
            },
            look_plan_from_text,
        ),
        (
            "body_direct",
            "BodyDirector/motion router: choose action from stage text and append to Session",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "duration": {"type": "number"},
                    "append": {"type": "boolean"},
                },
                "required": ["text"],
            },
            body_direct,
        ),
        (
            "camera_direct",
            "CameraAgent: plan shot/move and bake SessionCam",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "duration": {"type": "number"},
                    "shot": {"type": "string"},
                    "move_type": {"type": "string"},
                    "emotion": {"type": "string"},
                    "state": {"type": "string"},
                },
            },
            camera_direct,
        ),
        (
            "speech_say",
            "TTS + talk body append for one spoken line (use movie_run_story for multi-beat films)",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "emotion": {"type": "string"},
                    "intensity": {"type": "number"},
                    "body_action": {"type": "string"},
                    "hold_before_s": {"type": "number"},
                },
                "required": ["text"],
            },
            speech_say,
        ),
        (
            "face_emotion",
            "Set face emotion/intensity policy for subsequent speech/face bake",
            {
                "type": "object",
                "properties": {
                    "emotion": {"type": "string"},
                    "intensity": {"type": "number"},
                },
            },
            face_emotion,
        ),
    ]

    for name, desc, params, handler in specs:
        reg.register(
            Tool(
                name=name,
                description=desc,
                parameters=params,
                handler=handler,
                allowed_agents=agents,
            )
        )
