"""
Director agent — plans work and delegates to region agents (conceptually).
In this framework it returns a plan JSON that the session executes.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..providers.base import LLMMessage, LLMProvider
from ..tools.base import ToolRegistry
from .base_llm_agent import BaseLLMAgent


class DirectorAgent(BaseLLMAgent):
    name = "director"
    system_prompt = """You are the DIRECTOR of a multi-agent talking-head system.
Specialist agents: lips, eyes, brows.
You may use tools (including write_file for code under workspace rules).
When planning, prefer responding with JSON:

{
  "action": "respond",
  "message": "human summary",
  "policy": {},
  "plan": [
    {"agent": "lips", "task": "..."},
    {"agent": "eyes", "task": "..."},
    {"agent": "brows", "task": "..."}
  ]
}

Only include agents that need to act. Keep tasks short and concrete.
Emotion for a spoken line should be set via tools or plan for brows/eyes."""

    def plan(self, user_message: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """High-level plan + optional tool use, then list of sub-tasks."""
        result = self.run(user_message, context=context)
        raw = result.get("raw") or {}
        plan = raw.get("plan")
        if not isinstance(plan, list):
            # heuristic fallback: always consult all three for speaking tasks
            plan = [
                {"agent": "lips", "task": user_message},
                {"agent": "eyes", "task": user_message},
                {"agent": "brows", "task": user_message},
            ]
            result["plan"] = plan
        else:
            # normalize
            cleaned: List[Dict[str, str]] = []
            for step in plan:
                if not isinstance(step, dict):
                    continue
                ag = str(step.get("agent", "")).lower()
                task = str(step.get("task", user_message))
                if ag in ("lips", "eyes", "brows"):
                    cleaned.append({"agent": ag, "task": task})
            result["plan"] = cleaned or [
                {"agent": "lips", "task": user_message},
                {"agent": "brows", "task": user_message},
            ]
        return result
