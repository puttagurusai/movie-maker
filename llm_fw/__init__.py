"""
llm_fw — model-agnostic multi-agent framework for TalkFace.

- Providers: swap Claude / OpenAI / Grok / Ollama / any OpenAI-compatible API
- Agents: director, lips, eyes, brows (each owns a region)
- Tools: same for every model (files + face control params)
- Runtime face pipeline stays in face_agents/; LLMs set *policy*, not 30fps mesh

Run:
    python run_llm_agents.py
"""

from .providers.registry import get_provider
from .runtime.session import AgentSession

__all__ = ["get_provider", "AgentSession"]
