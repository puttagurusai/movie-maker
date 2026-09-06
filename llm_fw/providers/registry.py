"""
Provider factory — switch models without changing agent code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .base import LLMProvider
from .openai_compat import OpenAICompatProvider


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(path) if path else root / "config.json"
    if not cfg_path.is_file():
        example = root / "config.example.json"
        if example.is_file():
            return json.loads(example.read_text(encoding="utf-8"))
        return {
            "provider": "openai_compat",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
        }
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def get_provider(config: Optional[Dict[str, Any]] = None) -> LLMProvider:
    cfg = config or load_config()
    provider = (cfg.get("provider") or "openai_compat").lower().strip()
    model = cfg.get("model") or "gpt-4o-mini"
    temperature = float(cfg.get("temperature", 0.4))
    max_tokens = int(cfg.get("max_tokens", 1024))
    api_key_env = cfg.get("api_key_env") or "OPENAI_API_KEY"
    # Paste key in config.json as "api_key": "gsk_..."  (preferred) or use env var
    api_key = (cfg.get("api_key") or "").strip() or None
    if api_key and api_key.upper().startswith("PASTE_"):
        api_key = None  # ignore placeholder text
    base_url = cfg.get("base_url")

    if provider in (
        "openai_compat", "openai", "grok", "groq", "ollama", "openrouter", "vllm"
    ):
        return OpenAICompatProvider(
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model=model,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(
        f"Unknown provider '{provider}'. Use openai_compat or anthropic. "
        f"See llm_fw/config.example.json"
    )
