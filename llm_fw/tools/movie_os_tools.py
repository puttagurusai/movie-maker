"""
Movie OS tools — model-independent APIs any LLM agent can call.

Attach Grok / OpenAI / Anthropic / local models to the same surface:
  look, body, camera, face, cast, wardrobe, bpy scene code, story→Session.
Typed tools first; freeform bpy when kits cannot express the set.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, ToolRegistry, ToolResult

ROOT = Path(__file__).resolve().parents[2]


def _coord():
    from face_agents.coordinator import FaceCoordinator

    return FaceCoordinator(
        udp_ip=os.environ.get("FACE_UDP_IP", "127.0.0.1"),
        udp_port=int(os.environ.get("FACE_UDP_PORT", "9001")),
    )


def register_movie_os_tools(reg: ToolRegistry, *, agent_names: Optional[List[str]] = None) -> None:
    """Register full Movie OS control surface.

    Default allowed agents: movie_os (+ director). Pass agent_names=None to open
    the surface to every agent (any LLM attach / AGI harness).
    """
    if agent_names is None:
        agents = ["movie_os", "director"]
    else:
        agents = agent_names

    def movie_run_story(
        story: str,
        title: str = "movie",
        target_s: float = 60.0,
        use_llm: bool = True,
        join: bool = True,
    ) -> ToolResult:
        """Plan + bake + Session install a story (product path)."""
        try:
            from face_agents.movie_production.pipeline import MovieProductionPipeline

            llm = None
            if use_llm:
                try:
                    from llm_fw.providers import get_provider

                    cfg_path = ROOT / "llm_fw" / "config.json"
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
                    cfg = dict(cfg)
                    cfg["max_tokens"] = max(int(cfg.get("max_tokens") or 1024), 4096)
                    llm = get_provider(cfg)
                except Exception as e:
                    llm = None
                    print(f"[movie_os] LLM skip: {e}")
            pipe = MovieProductionPipeline(
                use_brain=True,
                llm_provider=llm,
                target_total_s=float(target_s),
            )
            pkg = pipe.run_story(story, title=title or "movie", play=False)
            report = None
            if join:
                pipe.install_master_timeline(pkg)
                report = {"joined": True}
            return ToolResult(
                ok=bool((pkg.validation or {}).get("ok", True)),
                data={
                    "title": pkg.title,
                    "duration_s": pkg.duration_s,
                    "shots": len(pkg.shots),
                    "director_source": pkg.director_source,
                    "output_dir": str(pkg.output_dir),
                    "body": [
                        {
                            "shot": s.shot_id,
                            "body_action": s.body_action,
                            "actions": list(s.actions or []),
                            "state": s.state,
                        }
                        for s in pkg.shots
                    ],
                    "join": report,
                },
            )
        except Exception as e:
            return ToolResult(ok=False, error=str(e))

    def look_apply(
        location: str = "street",
        time_of_day: str = "day",
        set_preset: str = "exterior_street",
        wardrobe_id: str = "hero_default",
        extras_count: int = 3,
        extras_preset: str = "sidewalk",
        light_mood: str = "soft",
    ) -> ToolResult:
        look = {
            "location": location,
            "time_of_day": time_of_day,
            "set_preset": set_preset,
            "wardrobe_id": wardrobe_id,
            "extras_count": int(extras_count),
            "extras_preset": extras_preset,
            "light_mood": light_mood,
        }
        _coord().send_udp({"type": "look", "op": "apply", "look": look})
        return ToolResult(ok=True, data=look)

    def body_append(
        action: str,
        duration: float = 2.5,
        append_timeline: bool = True,
        clip_policy: str = "hold_end",
        text: str = "",
        engine: str = "catalog",
    ) -> ToolResult:
        fps = 20.0
        frames = max(8, int(round(float(duration) * fps)))
        pkt = {
            "type": "body",
            "action": action,
            "duration": float(duration),
            "append_timeline": bool(append_timeline),
            "append_only": True,
            "live": False,
            "restart": True,
            "action_frames": frames,
            "motion_length": frames,
            "clip_policy": clip_policy,
            "loop": clip_policy == "loop",
            "engine": engine,
            "text": text,
            "clip_label": action,
            "clip_fps": 30.0 if engine != "momask" else 20.0,
        }
        _coord().send_udp(pkt)
        return ToolResult(ok=True, data=pkt)

    def session_bind() -> ToolResult:
        _coord().send_udp({"type": "body", "op": "session_bind"})
        return ToolResult(ok=True, data={"op": "session_bind"})

    def session_reset(session_id: str = "movie") -> ToolResult:
        _coord().send_udp({
            "type": "body",
            "session_reset": True,
            "rest": True,
            "session_id": session_id,
        })
        return ToolResult(ok=True, data={"session_id": session_id})

    def camera_plan(
        shot: str = "MS",
        move_type: str = "static",
        duration: float = 4.0,
        camera_name: str = "MovieCam_A",
        clear_previous: bool = False,
    ) -> ToolResult:
        pkt = {
            "type": "camera",
            "op": "plan",
            "duration": float(duration),
            "fps": 20.0,
            "shot": shot,
            "move_type": move_type,
            "track_head": False,
            "camera_name": camera_name,
            "look_at_name": camera_name + "_LookAt",
            "set_scene_camera": True,
            "bake_keyframes": True,
            "clear_previous": bool(clear_previous),
            "hold_after": False,
            "live": False,
        }
        c = _coord()
        c.send_udp(pkt)
        c.send_udp({"type": "camera", "op": "rest", "hold_after": False})
        return ToolResult(ok=True, data=pkt)

    def cast_spawn(count: int = 3, preset: str = "sidewalk", outfit_id: str = "casual_01") -> ToolResult:
        pkt = {
            "type": "cast",
            "op": "spawn_extras",
            "count": int(count),
            "preset": preset,
            "outfit_id": outfit_id,
        }
        _coord().send_udp(pkt)
        return ToolResult(ok=True, data=pkt)

    def wardrobe_apply(wardrobe_id: str = "casual_01") -> ToolResult:
        pkt = {"type": "wardrobe", "op": "apply", "wardrobe_id": wardrobe_id}
        _coord().send_udp(pkt)
        return ToolResult(ok=True, data=pkt)

    def bpy_exec(code: str) -> ToolResult:
        """Freeform bpy for sets/props kits cannot express. No OS/network."""
        if not (code or "").strip():
            return ToolResult(ok=False, error="empty code")
        _coord().send_udp({"type": "bpy", "code": code})
        time.sleep(0.05)
        return ToolResult(ok=True, data={"bytes": len(code)})

    def list_capabilities() -> ToolResult:
        return ToolResult(
            ok=True,
            data={
                "performance": [
                    "body_append", "body_direct", "speech_say", "face_emotion",
                    "session_bind", "session_reset", "camera_plan", "camera_direct",
                ],
                "scene": [
                    "look_apply", "look_plan_from_text", "cast_spawn", "wardrobe_apply",
                    "light_set", "material_set", "prop_add", "npc_motion", "bpy_exec",
                ],
                "inspect": ["session_inspect", "scene_inspect", "list_capabilities"],
                "film": ["movie_run_story", "movie_play"],
                "product": "3D Session Play in Blender (not auto-MP4)",
                "attach": "Set any LLM in llm_fw/config.json — same tool belt (ChatGPT pattern)",
            },
        )

    specs = [
        ("list_capabilities", "List Movie OS tools and what they control", {}, list_capabilities),
        (
            "movie_run_story",
            "Plan+bake+install a natural-language story into one Blender Session",
            {
                "type": "object",
                "properties": {
                    "story": {"type": "string"},
                    "title": {"type": "string"},
                    "target_s": {"type": "number"},
                    "use_llm": {"type": "boolean"},
                    "join": {"type": "boolean"},
                },
                "required": ["story"],
            },
            movie_run_story,
        ),
        (
            "look_apply",
            "Build Look_Set (location, lighting, extras, wardrobe intent)",
            {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "time_of_day": {"type": "string"},
                    "set_preset": {"type": "string"},
                    "wardrobe_id": {"type": "string"},
                    "extras_count": {"type": "integer"},
                    "extras_preset": {"type": "string"},
                    "light_mood": {"type": "string"},
                },
            },
            look_apply,
        ),
        (
            "body_append",
            "Append a catalog/MoMask body Action onto the Session timeline",
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "duration": {"type": "number"},
                    "append_timeline": {"type": "boolean"},
                    "clip_policy": {"type": "string"},
                    "text": {"type": "string"},
                    "engine": {"type": "string"},
                },
                "required": ["action"],
            },
            body_append,
        ),
        (
            "session_bind",
            "Bind Session Action to hero armature for Space/Play scrub",
            {"type": "object", "properties": {}},
            session_bind,
        ),
        (
            "session_reset",
            "Clear Session timeline and rest hero",
            {"type": "object", "properties": {"session_id": {"type": "string"}}},
            session_reset,
        ),
        (
            "camera_plan",
            "Bake a SessionCam shot/move (no live follow)",
            {
                "type": "object",
                "properties": {
                    "shot": {"type": "string"},
                    "move_type": {"type": "string"},
                    "duration": {"type": "number"},
                    "camera_name": {"type": "string"},
                    "clear_previous": {"type": "boolean"},
                },
            },
            camera_plan,
        ),
        (
            "cast_spawn",
            "Spawn sidewalk/extras NPCs",
            {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "preset": {"type": "string"},
                    "outfit_id": {"type": "string"},
                },
            },
            cast_spawn,
        ),
        (
            "wardrobe_apply",
            "Apply wardrobe preset if fit gate passes",
            {
                "type": "object",
                "properties": {"wardrobe_id": {"type": "string"}},
                "required": ["wardrobe_id"],
            },
            wardrobe_apply,
        ),
        (
            "bpy_exec",
            "Freeform bpy for missing sets/props (Look_Set only; no OS/network)",
            {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            bpy_exec,
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
