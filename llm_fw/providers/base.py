"""
Abstract LLM provider — every model behind the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMMessage:
    role: str  # system | user | assistant | tool
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    # OpenAI-style assistant tool_calls payload (list of dicts)
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class LLMResponse:
    content: str
    raw: Any = None
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    # Native function calling (OpenAI / Groq / etc.)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""


class LLMProvider(ABC):
    """One implementation per API family; agents never import vendor SDKs."""

    name: str = "base"

    def __init__(self, model: str, temperature: float = 0.4, max_tokens: int = 1024, **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

    @abstractmethod
    def chat(self, messages: List[LLMMessage], system: Optional[str] = None) -> LLMResponse:
        """Send messages, return assistant text."""
        raise NotImplementedError

    def chat_with_tools(
        self,
        messages: List[LLMMessage],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """
        Chat with OpenAI-style tools. Default falls back to plain chat
        (providers that support tools override this).
        """
        return self.chat(messages, system=system)

    def chat_json(self, messages: List[LLMMessage], system: Optional[str] = None) -> Dict[str, Any]:
        """Ask for JSON; parse robustly."""
        import json
        import re

        resp = self.chat(messages, system=system)
        text = (resp.content or "").strip()
        # strip markdown fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # try extract first {...}
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                return json.loads(m.group(0))
            return {"raw_text": resp.content, "parse_error": True}
