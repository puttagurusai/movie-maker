"""
Tools are model-independent. Any LLM agent can call the same tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "data": self.data, "error": self.error}


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]  # simple JSON-schema-ish
    handler: Callable[..., ToolResult]
    # which agents may use this tool
    allowed_agents: Optional[List[str]] = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_for_agent(self, agent_name: str) -> List[Tool]:
        out = []
        for t in self._tools.values():
            if t.allowed_agents is None or agent_name in t.allowed_agents:
                out.append(t)
        return out

    def schemas_for_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self.list_for_agent(agent_name)
        ]

    def openai_tools_for_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        """OpenAI / Groq Chat Completions `tools` array (function calling)."""
        out: List[Dict[str, Any]] = []
        for t in self.list_for_agent(agent_name):
            params = t.parameters if isinstance(t.parameters, dict) else {}
            if not params:
                params = {"type": "object", "properties": {}}
            elif "type" not in params:
                params = {"type": "object", **params}
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or t.name,
                        "parameters": params,
                    },
                }
            )
        return out

    def call(self, agent_name: str, tool_name: str, **kwargs) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(ok=False, error=f"Unknown tool: {tool_name}")
        if tool.allowed_agents is not None and agent_name not in tool.allowed_agents:
            return ToolResult(
                ok=False,
                error=f"Agent '{agent_name}' may not call tool '{tool_name}'",
            )
        try:
            return tool.handler(**kwargs)
        except Exception as e:
            return ToolResult(ok=False, error=str(e))
