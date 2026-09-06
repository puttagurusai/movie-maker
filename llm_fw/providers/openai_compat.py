"""
OpenAI-compatible Chat Completions API.

Works with:
  - Groq   (base_url=https://api.groq.com/openai/v1)
  - OpenAI
  - xAI Grok (base_url=https://api.x.ai/v1)
  - Ollama (base_url=http://127.0.0.1:11434/v1)
  - OpenRouter, vLLM, LM Studio, etc.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .base import LLMMessage, LLMProvider, LLMResponse

# Cloudflare (error 1010) blocks the default Python-urllib User-Agent.
# Use a normal client signature so Groq/Cloudflare accepts the request.
_DEFAULT_UA = (
    "TalkFace-llm_fw/1.0 (+https://github.com/local) "
    "Python-urllib compatible; Mozilla/5.0"
)


class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 1024,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.api_key = (api_key or os.environ.get(api_key_env, "") or "").strip()
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._api_key_env = api_key_env

    def _auth_headers(self) -> Dict[str, str]:
        if not self.api_key and "11434" not in self.base_url:
            raise RuntimeError(
                "No API key set. Paste your Groq key into llm_fw/config.json "
                'as "api_key": "gsk_..." '
                f"(or set env {self._api_key_env})"
            )
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": _DEFAULT_UA,
            "Accept": "application/json",
        }

    @staticmethod
    def _serialize_messages(
        messages: List[LLMMessage],
        system: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        payload_msgs: List[Dict[str, Any]] = []
        if system:
            payload_msgs.append({"role": "system", "content": system})
        for m in messages:
            msg: Dict[str, Any] = {"role": m.role, "content": m.content if m.content is not None else ""}
            if m.role == "assistant" and m.tool_calls:
                msg["tool_calls"] = m.tool_calls
                # Some APIs require content null when tool_calls present
                if not (m.content or "").strip():
                    msg["content"] = None
            if m.role == "tool":
                if m.tool_call_id:
                    msg["tool_call_id"] = m.tool_call_id
                if m.name:
                    msg["name"] = m.name
            elif m.name and m.role != "assistant":
                msg["name"] = m.name
            payload_msgs.append(msg)
        return payload_msgs

    @staticmethod
    def _parse_response(raw: dict, default_model: str) -> LLMResponse:
        try:
            choice = raw["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            finish = str(choice.get("finish_reason") or "")
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected LLM response: {raw}") from e
        usage = raw.get("usage") or {}
        return LLMResponse(
            content=content if isinstance(content, str) else (content or ""),
            raw=raw,
            model=raw.get("model", default_model),
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            },
            tool_calls=list(tool_calls) if isinstance(tool_calls, list) else [],
            finish_reason=finish,
        )

    def chat(self, messages: List[LLMMessage], system: Optional[str] = None) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": self._serialize_messages(messages, system=system),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(body).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        raw = self._post_json(url, data, self._auth_headers())
        return self._parse_response(raw, self.model)

    def chat_with_tools(
        self,
        messages: List[LLMMessage],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        """Native OpenAI-compatible function calling."""
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": self._serialize_messages(messages, system=system),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        data = json.dumps(body).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        try:
            raw = self._post_json(url, data, self._auth_headers())
            return self._parse_response(raw, self.model)
        except RuntimeError as e:
            err = str(e).lower()
            # Fatal: bad model/key — do not mask as "no tools"
            if any(x in err for x in ("401", "403", "404", "model_not_found", "invalid api")):
                raise
            # Some local models reject tools — fall back to plain chat
            if tools and any(x in err for x in ("tool", "function", "400", "unsupported")):
                return self.chat(messages, system=system)
            raise

    def _post_json(self, url: str, data: bytes, headers: dict) -> dict:
        # Prefer requests if installed (more reliable with Cloudflare)
        try:
            import requests

            resp = requests.post(url, data=data, headers=headers, timeout=120)
            if resp.status_code >= 400:
                raise RuntimeError(self._format_http_error(resp.status_code, resp.text))
            return resp.json()
        except ImportError:
            pass
        except RuntimeError:
            raise
        except Exception as e:
            # fall through to urllib
            last_req_err = e
        else:
            last_req_err = None

        req = urllib.request.Request(url, data=data, method="POST")
        for k, v in headers.items():
            req.add_header(k, v)

        # Default SSL context
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(self._format_http_error(e.code, err)) from e
        except urllib.error.URLError as e:
            extra = f" (requests fallback failed: {last_req_err})" if last_req_err else ""
            raise RuntimeError(
                f"LLM connection failed ({self.base_url}): {e}{extra}"
            ) from e

    @staticmethod
    def _format_http_error(code: int, body: str) -> str:
        hint = ""
        if code == 401:
            hint = (
                "\n→ Invalid API key. Check llm_fw/config.json \"api_key\" "
                "(should start with gsk_ for Groq)."
            )
        elif code == 403:
            if "1010" in body:
                hint = (
                    "\n→ Cloudflare 1010: request blocked as bot. "
                    "Fixed by User-Agent in latest code; also try: pip install requests\n"
                    "→ Or regenerate key at https://console.groq.com (login first) → API Keys"
                )
            else:
                hint = (
                    "\n→ Access denied. Key may lack permission, or model not allowed on free tier. "
                    "Try model llama-3.1-8b-instant"
                )
        elif code == 404:
            hint = (
                "\n→ Wrong URL or model name. base_url must be "
                "https://api.groq.com/openai/v1  (not console.groq.com)"
            )
        elif code == 429:
            hint = "\n→ Rate limited. Wait a few seconds and retry."
        return f"LLM HTTP {code}: {body}{hint}"
