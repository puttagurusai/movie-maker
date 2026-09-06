"""
AgentSession — wires provider + tools + director + region agents.
Independent of which LLM is configured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..agents.director_agent import DirectorAgent
from ..agents.movie_os_agent import MovieOSAgent
from ..agents.region_agents import BrowsLLMAgent, EyesLLMAgent, LipsLLMAgent
from ..providers.base import LLMProvider
from ..providers.registry import get_provider, load_config
from ..tools.base import ToolRegistry
from ..tools.face_tools import get_face_policy, register_face_tools
from ..tools.file_tools import register_file_tools
from ..tools.movie_os_tools import register_movie_os_tools
from ..tools.agent_tools import register_agent_tools
from ..tools.blender_tools import register_blender_tools


class AgentSession:
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[Dict[str, Any]] = None,
        workspace: Optional[Path] = None,
    ):
        self.config = config or load_config()
        self.provider = provider or get_provider(self.config)
        root = Path(self.config.get("workspace_root") or ".").resolve()
        self.workspace = (workspace or root).resolve()

        self.tools = ToolRegistry()
        register_face_tools(self.tools)
        register_file_tools(self.tools, self.workspace)
        # ChatGPT-style Movie OS tool belt (any LLM attaches here)
        register_movie_os_tools(self.tools)
        register_agent_tools(self.tools)
        register_blender_tools(self.tools)

        self.director = DirectorAgent(self.provider, self.tools)
        self.movie_os = MovieOSAgent(self.provider, self.tools)
        self.specialists = {
            "lips": LipsLLMAgent(self.provider, self.tools),
            "eyes": EyesLLMAgent(self.provider, self.tools),
            "brows": BrowsLLMAgent(self.provider, self.tools),
        }

    def info(self) -> Dict[str, Any]:
        return {
            "provider": getattr(self.provider, "name", type(self.provider).__name__),
            "model": self.provider.model,
            "workspace": str(self.workspace),
            "agents": ["movie_os", "director", "lips", "eyes", "brows"],
            "movie_os_tools": [t.name for t in self.tools.list_for_agent("movie_os")],
            "openai_tool_count": len(self.tools.openai_tools_for_agent("movie_os")),
            "face_policy": get_face_policy(),
            "pattern": "ChatGPT-style: any LLM + tool belt → Blender Session Play",
        }

    def handle(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Full multi-agent turn:
          1) Director plans
          2) Each listed specialist runs its task
          3) Return combined report + face policy
        """
        ctx = dict(context or {})
        ctx["face_policy"] = get_face_policy()

        director_out = self.director.plan(user_message, context=ctx)
        plan = director_out.get("plan") or []

        specialist_results = []
        for step in plan:
            ag = step.get("agent")
            task = step.get("task") or user_message
            agent = self.specialists.get(ag)
            if not agent:
                continue
            sub_ctx = {
                "face_policy": get_face_policy(),
                "director_message": director_out.get("message"),
                "user_message": user_message,
            }
            out = agent.run(task, context=sub_ctx)
            specialist_results.append(out)

        return {
            "director": {
                "message": director_out.get("message"),
                "plan": plan,
                "ok": director_out.get("ok", True),
            },
            "specialists": specialist_results,
            "face_policy": get_face_policy(),
        }

    def handle_movie_os(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """AGI attach point: LLM mediates the full Movie OS tool surface."""
        ctx = dict(context or {})
        ctx.setdefault("product", "3D Session Play in Blender")
        return self.movie_os.run(user_message, context=ctx)

    def handle_single_agent(self, agent_name: str, user_message: str) -> Dict[str, Any]:
        """Talk to one region agent only."""
        if agent_name in ("movie_os", "movie"):
            return self.handle_movie_os(user_message)
        if agent_name == "director":
            return self.director.plan(user_message, context={"face_policy": get_face_policy()})
        agent = self.specialists.get(agent_name)
        if not agent:
            return {"ok": False, "error": f"Unknown agent {agent_name}"}
        return agent.run(user_message, context={"face_policy": get_face_policy()})
