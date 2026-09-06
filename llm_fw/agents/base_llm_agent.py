"""
Base LLM agent — same loop for every model.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..providers.base import LLMMessage, LLMProvider
from ..tools.base import ToolRegistry


class BaseLLMAgent:
    name: str = "base"
    system_prompt: str = "You are a helpful agent. Reply with JSON only."

    def __init__(self, provider: LLMProvider, tools: ToolRegistry):
        self.provider = provider
        self.tools = tools
        self.history: List[LLMMessage] = []

    def reset(self) -> None:
        self.history.clear()

    def _tool_catalog(self) -> str:
        schemas = self.tools.schemas_for_agent(self.name)
        return json.dumps(schemas, indent=2)

    def run(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        One turn: LLM returns JSON with either:
          {"action": "respond", "message": "...", "policy": {...optional}}
          {"action": "tool", "tool": "name", "args": {...}}
        Tool results are fed back once for a final respond.
        """
        ctx_txt = json.dumps(context or {}, ensure_ascii=False, indent=2)
        system = (
            f"{self.system_prompt}\n\n"
            f"Your agent name: {self.name}\n"
            f"Available tools (JSON):\n{self._tool_catalog()}\n\n"
            "Always reply with a single JSON object, no markdown, one of:\n"
            '  {"action":"respond","message":"string","policy":{}}\n'
            '  {"action":"tool","tool":"tool_name","args":{}}\n'
            "After tools run you will get results; then finish with action=respond."
        )
        user = f"Context:\n{ctx_txt}\n\nUser / director request:\n{user_message}"
        self.history.append(LLMMessage(role="user", content=user))

        # Allow up to 3 tool rounds
        for _ in range(3):
            data = self.provider.chat_json(self.history, system=system)
            if data.get("parse_error"):
                return {
                    "agent": self.name,
                    "ok": False,
                    "message": data.get("raw_text", "parse error"),
                    "policy": {},
                }

            action = (data.get("action") or "respond").lower()
            if action == "tool":
                tool_name = data.get("tool") or ""
                args = data.get("args") or {}
                if not isinstance(args, dict):
                    args = {}
                result = self.tools.call(self.name, tool_name, **args)
                self.history.append(
                    LLMMessage(
                        role="assistant",
                        content=json.dumps(data),
                    )
                )
                self.history.append(
                    LLMMessage(
                        role="user",
                        content="Tool result:\n" + json.dumps(result.to_dict()),
                    )
                )
                continue

            # respond
            msg = data.get("message") or data.get("summary") or ""
            policy = data.get("policy") if isinstance(data.get("policy"), dict) else {}
            self.history.append(LLMMessage(role="assistant", content=json.dumps(data)))
            return {
                "agent": self.name,
                "ok": True,
                "message": msg,
                "policy": policy,
                "raw": data,
            }

        return {
            "agent": self.name,
            "ok": False,
            "message": "Too many tool rounds without final respond",
            "policy": {},
        }
