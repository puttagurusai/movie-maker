"""
Movie OS agent — attach ANY LLM; mediates 3D Session via tools.

Prefers native OpenAI-style tool_calls; falls back to JSON action protocol
when the model/provider does not support tools.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..providers.base import LLMMessage, LLMProvider, LLMResponse
from ..tools.base import ToolRegistry


SYSTEM = """You are the Movie OS director agent inside an editable Blender film engine.

Goal: beat text-to-video black boxes by building an editable 3D Session the user can Play.

You control tools — not pixels. Prefer typed tools; use bpy_exec / prop_add only when kits cannot build the set.

Performance (always typed agents):
- movie_run_story for multi-beat films
- body_direct / body_append, speech_say, camera_direct / camera_plan
- session_bind then movie_play when the take is ready

Scene:
- look_plan_from_text / look_apply, cast_spawn, wardrobe_apply
- light_set, material_set, prop_add, bpy_exec (escape hatch)

Inspect before claiming done:
- session_inspect / scene_inspect

Rules:
1. Prefer typed tools over bpy_exec.
2. After body clips, call session_bind (and movie_play if user wants playback).
3. Never invent shell commands or absolute OS paths outside the project.
4. When finished, stop calling tools and give a short status of what Play Session should show.
"""


class MovieOSAgent:
    name = "movie_os"

    def __init__(self, provider: LLMProvider, tools: ToolRegistry, *, max_rounds: int = 16):
        self.provider = provider
        self.tools = tools
        self.max_rounds = max(3, int(max_rounds))
        self.history: List[LLMMessage] = []

    def reset(self) -> None:
        self.history.clear()

    def _openai_tools(self) -> List[Dict[str, Any]]:
        return self.tools.openai_tools_for_agent(self.name)

    def _catalog_text(self) -> str:
        return json.dumps(self.tools.schemas_for_agent(self.name), indent=2)

    def _run_tool(self, tool_name: str, args: Dict[str, Any], tool_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = self.tools.call(self.name, tool_name, **(args or {}))
        entry = {
            "tool": tool_name,
            "args": args or {},
            "result": result.to_dict(),
        }
        tool_trace.append(entry)
        return result.to_dict()

    def run(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx_txt = json.dumps(context or {}, ensure_ascii=False, indent=2)
        system = f"{SYSTEM}\n\nAgent: {self.name}\n"
        self.history.append(
            LLMMessage(
                role="user",
                content=f"Context:\n{ctx_txt}\n\nRequest:\n{user_message}",
            )
        )
        tool_trace: List[Dict[str, Any]] = []
        openai_tools = self._openai_tools()
        use_native = bool(openai_tools) and hasattr(self.provider, "chat_with_tools")

        for _ in range(self.max_rounds):
            if use_native:
                try:
                    resp = self.provider.chat_with_tools(
                        self.history,
                        tools=openai_tools,
                        system=system,
                        tool_choice="auto",
                    )
                except Exception as e:
                    err = str(e)
                    low = err.lower()
                    # Fatal auth/model errors — do not pretend JSON will work
                    if any(x in low for x in ("404", "401", "403", "model_not_found", "invalid api key")):
                        return {
                            "agent": self.name,
                            "ok": False,
                            "message": f"LLM provider error: {err}",
                            "tools": tool_trace,
                            "mode": "provider_error",
                        }
                    # Tool-schema unsupported → JSON protocol
                    use_native = False
                    self.history.append(
                        LLMMessage(
                            role="user",
                            content=f"(native tools failed: {err}; switch to JSON tool protocol)",
                        )
                    )
                    continue

                if resp.tool_calls:
                    # Record assistant turn with tool_calls
                    self.history.append(
                        LLMMessage(
                            role="assistant",
                            content=resp.content or "",
                            tool_calls=resp.tool_calls,
                        )
                    )
                    for tc in resp.tool_calls:
                        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                        tool_name = str(fn.get("name") or "")
                        raw_args = fn.get("arguments") or "{}"
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                        except json.JSONDecodeError:
                            args = {}
                        if not isinstance(args, dict):
                            args = {}
                        result_d = self._run_tool(tool_name, args, tool_trace)
                        self.history.append(
                            LLMMessage(
                                role="tool",
                                content=json.dumps(result_d),
                                name=tool_name,
                                tool_call_id=str(tc.get("id") or tool_name),
                            )
                        )
                    continue

                # Final text response
                msg = (resp.content or "").strip()
                self.history.append(LLMMessage(role="assistant", content=msg))
                return {
                    "agent": self.name,
                    "ok": True,
                    "message": msg,
                    "policy": {},
                    "tools": tool_trace,
                    "mode": "native_tools",
                    "raw": getattr(resp, "raw", None),
                }

            # ----- JSON fallback (models without tool_calls) -----
            system_json = (
                system
                + "\nAvailable tools (JSON):\n"
                + self._catalog_text()
                + "\nAlways reply with one JSON object:\n"
                + '  {"action":"tool","tool":"name","args":{}}\n'
                + '  {"action":"respond","message":"..."}\n'
            )
            try:
                data = self.provider.chat_json(self.history, system=system_json)
            except Exception as e:
                return {
                    "agent": self.name,
                    "ok": False,
                    "message": f"LLM provider error: {e}",
                    "tools": tool_trace,
                    "mode": "provider_error",
                }
            if data.get("parse_error"):
                return {
                    "agent": self.name,
                    "ok": False,
                    "message": data.get("raw_text", "parse error"),
                    "tools": tool_trace,
                    "mode": "json_fallback",
                }

            action = (data.get("action") or "respond").lower()
            if action == "tool":
                tool_name = str(data.get("tool") or "")
                args = data.get("args") if isinstance(data.get("args"), dict) else {}
                result_d = self._run_tool(tool_name, args, tool_trace)
                self.history.append(LLMMessage(role="assistant", content=json.dumps(data)))
                self.history.append(
                    LLMMessage(role="user", content="Tool result:\n" + json.dumps(result_d))
                )
                continue

            msg = data.get("message") or data.get("summary") or ""
            self.history.append(LLMMessage(role="assistant", content=json.dumps(data)))
            return {
                "agent": self.name,
                "ok": True,
                "message": msg,
                "policy": data.get("policy") if isinstance(data.get("policy"), dict) else {},
                "tools": tool_trace,
                "mode": "json_fallback",
                "raw": data,
            }

        return {
            "agent": self.name,
            "ok": False,
            "message": "Hit tool-round limit without final respond",
            "tools": tool_trace,
        }
