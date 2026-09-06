"""
Anthropic Messages API (Claude).
Optional — only used when provider=anthropic.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import List, Optional

from .base import LLMMessage, LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        temperature: float = 0.4,
        max_tokens: int = 1024,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.api_key = api_key or os.environ.get(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(f"Set {api_key_env} for Anthropic provider")

    def chat(self, messages: List[LLMMessage], system: Optional[str] = None) -> LLMResponse:
        # Anthropic wants system separate; messages only user/assistant
        anth_msgs = []
        for m in messages:
            role = m.role if m.role in ("user", "assistant") else "user"
            anth_msgs.append({"role": role, "content": m.content})

        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": anth_msgs or [{"role": "user", "content": "Hello"}],
        }
        if system:
            body["system"] = system

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=data,
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("x-api-key", self.api_key)
        req.add_header("anthropic-version", "2023-06-01")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic HTTP {e.code}: {err}") from e

        parts = raw.get("content") or []
        text = ""
        for p in parts:
            if isinstance(p, dict) and p.get("type") == "text":
                text += p.get("text", "")
        usage = raw.get("usage") or {}
        return LLMResponse(
            content=text,
            raw=raw,
            model=raw.get("model", self.model),
            usage={
                "prompt_tokens": int(usage.get("input_tokens", 0)),
                "completion_tokens": int(usage.get("output_tokens", 0)),
            },
        )
