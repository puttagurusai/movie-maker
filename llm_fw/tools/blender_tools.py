"""
Blender-facing tools: inspect, lights, materials, NPC motion, hardened bpy.
Typed first; bpy_exec for geometry kits cannot express.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, ToolRegistry, ToolResult

ROOT = Path(__file__).resolve().parents[2]
INSPECT_PATH = ROOT / "temp" / "movie_os_inspect.json"


def _coord():
    from face_agents.coordinator import FaceCoordinator

    return FaceCoordinator(
        udp_ip=os.environ.get("FACE_UDP_IP", "127.0.0.1"),
        udp_port=int(os.environ.get("FACE_UDP_PORT", "9001")),
    )


def _read_inspect(timeout_s: float = 1.2) -> Dict[str, Any]:
    INSPECT_PATH.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + max(0.2, float(timeout_s))
    last_mtime = INSPECT_PATH.stat().st_mtime if INSPECT_PATH.is_file() else 0.0
    _coord().send_udp({
        "type": "inspect",
        "path": str(INSPECT_PATH.resolve()),
    })
    while time.time() < deadline:
        if INSPECT_PATH.is_file():
            try:
                mtime = INSPECT_PATH.stat().st_mtime
                if mtime >= last_mtime - 0.05 or last_mtime == 0.0:
                    data = json.loads(INSPECT_PATH.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and data.get("ok"):
                        return data
            except Exception:
                pass
        time.sleep(0.08)
    if INSPECT_PATH.is_file():
        try:
            return json.loads(INSPECT_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            return {"ok": False, "error": f"read inspect: {e}"}
    return {"ok": False, "error": "inspect timeout — is Blender stream_receiver running?"}


def register_blender_tools(reg: ToolRegistry, *, agent_names: Optional[List[str]] = None) -> None:
    agents = agent_names if agent_names is not None else ["movie_os", "director"]

    def session_inspect() -> ToolResult:
        data = _read_inspect()
        ok = bool(data.get("ok"))
        return ToolResult(ok=ok, data=data, error=None if ok else str(data.get("error") or "inspect failed"))

    def scene_inspect() -> ToolResult:
        data = _read_inspect()
        if not data.get("ok"):
            return ToolResult(ok=False, error=str(data.get("error") or "inspect failed"), data=data)
        slim = {
            "look_objects": data.get("look_objects"),
            "npcs": data.get("npcs"),
            "cameras": data.get("cameras"),
            "scene_camera": data.get("scene_camera"),
            "bound_action": data.get("bound_action"),
            "clip_count": data.get("clip_count"),
            "clips": data.get("clips"),
            "frame_end": data.get("frame_end"),
        }
        return ToolResult(ok=True, data=slim)

    def light_set(
        mood: str = "soft",
        world_strength: float = -1.0,
        key_energy: float = -1.0,
        fill_energy: float = -1.0,
        rim_energy: float = -1.0,
    ) -> ToolResult:
        pkt: Dict[str, Any] = {"type": "light", "op": "mood", "mood": mood or "soft"}
        if world_strength is not None and float(world_strength) >= 0:
            pkt["world_strength"] = float(world_strength)
        if key_energy is not None and float(key_energy) >= 0:
            pkt["key_energy"] = float(key_energy)
        if fill_energy is not None and float(fill_energy) >= 0:
            pkt["fill_energy"] = float(fill_energy)
        if rim_energy is not None and float(rim_energy) >= 0:
            pkt["rim_energy"] = float(rim_energy)
        _coord().send_udp(pkt)
        return ToolResult(ok=True, data=pkt)

    def material_set(
        target: str = "Look_Ground",
        color_r: float = 0.35,
        color_g: float = 0.35,
        color_b: float = 0.38,
        roughness: float = 0.65,
        metallic: float = 0.0,
    ) -> ToolResult:
        pkt = {
            "type": "material",
            "target": target or "Look_Ground",
            "color": [float(color_r), float(color_g), float(color_b), 1.0],
            "roughness": float(roughness),
            "metallic": float(metallic),
        }
        _coord().send_udp(pkt)
        return ToolResult(ok=True, data=pkt)

    def npc_motion(action: str = "idle", duration: float = 2.0) -> ToolResult:
        """
        Best-effort: apply a simple Look/Cast note. Hero Session is untouched.
        Full NPC Action retarget is limited; this tags intent + optional bpy pose hold.
        """
        act = (action or "idle").lower().strip()
        # Document intent via inspect-friendly UDP cast op if supported; else bpy note
        code = f"""
import bpy
n = 0
for obj in bpy.data.objects:
    if str(obj.get('cast_role') or '') != 'extra':
        continue
    obj['npc_action'] = {act!r}
    obj['npc_duration'] = {float(duration)!r}
    n += 1
result = {{'tagged': n, 'action': {act!r}}}
"""
        _coord().send_udp({"type": "bpy", "code": code})
        return ToolResult(
            ok=True,
            data={
                "action": act,
                "duration": float(duration),
                "note": "NPCs tagged; catalog retarget on extras is best-effort",
            },
        )

    def prop_add(
        kind: str = "cube",
        name: str = "Look_Prop",
        x: float = 0.0,
        y: float = -2.0,
        z: float = 0.5,
        scale: float = 0.5,
    ) -> ToolResult:
        """Add a simple prop into Look_Set (safer than freeform bpy)."""
        safe = "".join(c for c in (name or "Look_Prop") if c.isalnum() or c in ("_",))[:40] or "Look_Prop"
        kind_l = (kind or "cube").lower()
        ops = {
            "cube": "bpy.ops.mesh.primitive_cube_add",
            "cylinder": "bpy.ops.mesh.primitive_cylinder_add",
            "sphere": "bpy.ops.mesh.primitive_uv_sphere_add",
            "plane": "bpy.ops.mesh.primitive_plane_add",
            "cone": "bpy.ops.mesh.primitive_cone_add",
        }
        op = ops.get(kind_l, ops["cube"])
        code = f"""
import bpy
col = look_collection
old = bpy.data.objects.get({safe!r})
if old is not None:
    bpy.data.objects.remove(old, do_unlink=True)
{op}(location=({float(x)}, {float(y)}, {float(z)}))
obj = bpy.context.active_object
obj.name = {safe!r}
obj.scale = ({float(scale)}, {float(scale)}, {float(scale)})
if col is not None:
    try:
        if obj.name not in col.objects:
            col.objects.link(obj)
    except Exception:
        pass
result = {{'name': obj.name}}
"""
        _coord().send_udp({"type": "bpy", "code": code})
        time.sleep(0.05)
        return ToolResult(ok=True, data={"kind": kind_l, "name": safe, "xyz": [x, y, z], "scale": scale})

    def movie_play() -> ToolResult:
        """Ask Blender to Play Session (bind + animation_play)."""
        code = """
import bpy
try:
    bpy.ops.session.play_timeline()
    result = {'played': True}
except Exception as e:
    result = {'played': False, 'error': str(e)}
"""
        _coord().send_udp({"type": "body", "op": "session_bind"})
        time.sleep(0.1)
        _coord().send_udp({"type": "bpy", "code": code})
        return ToolResult(ok=True, data={"op": "movie_play"})

    specs = [
        (
            "session_inspect",
            "Inspect Session SoT: clips, bound action, frames (requires Blender receiver)",
            {"type": "object", "properties": {}},
            session_inspect,
        ),
        (
            "scene_inspect",
            "Inspect Look_Set objects, NPCs, cameras, Session clip summary",
            {"type": "object", "properties": {}},
            scene_inspect,
        ),
        (
            "light_set",
            "Set Look lighting mood (soft/bright/dramatic/night/golden) or energies",
            {
                "type": "object",
                "properties": {
                    "mood": {"type": "string"},
                    "world_strength": {"type": "number"},
                    "key_energy": {"type": "number"},
                    "fill_energy": {"type": "number"},
                    "rim_energy": {"type": "number"},
                },
            },
            light_set,
        ),
        (
            "material_set",
            "Set simple Principled material on Look_Ground or named mesh",
            {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "color_r": {"type": "number"},
                    "color_g": {"type": "number"},
                    "color_b": {"type": "number"},
                    "roughness": {"type": "number"},
                    "metallic": {"type": "number"},
                },
            },
            material_set,
        ),
        (
            "npc_motion",
            "Tag extras NPCs with an intended action (idle/walk) — does not touch hero Session",
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "duration": {"type": "number"},
                },
            },
            npc_motion,
        ),
        (
            "prop_add",
            "Add a simple mesh prop into Look_Set (cube/cylinder/sphere/plane/cone)",
            {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "name": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                    "scale": {"type": "number"},
                },
            },
            prop_add,
        ),
        (
            "movie_play",
            "Bind Session and press Play in Blender",
            {"type": "object", "properties": {}},
            movie_play,
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
