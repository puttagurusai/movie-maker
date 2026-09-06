#!/usr/bin/env python
"""
run_agentic_movie.py — ChatGPT pattern: any LLM + Movie OS tools → Blender Session.

Set the model in llm_fw/config.json (Groq / OpenAI / Grok / Ollama / …).
The same tool belt is used regardless of provider.

Examples:
  python run_agentic_movie.py --info
  python run_agentic_movie.py --story "Walk on a sunny street, wave hello, say goodbye."
  python run_agentic_movie.py --ask "Sunny street, 3 NPCs, hero waves and says hello. Then session_bind."
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Agentic Movie OS — any LLM drives film tools (ChatGPT pattern)"
    )
    ap.add_argument("--story", type=str, default="", help="Story → prefer movie_run_story tool")
    ap.add_argument("--ask", type=str, default="", help="Freeform request for MovieOSAgent")
    ap.add_argument("--info", action="store_true", help="Print provider + tool catalog")
    ap.add_argument("--list-tools", action="store_true", help="Print OpenAI-style tool schemas")
    args = ap.parse_args()

    from llm_fw.runtime.session import AgentSession

    sess = AgentSession()
    info = sess.info()

    if args.info:
        print(json.dumps(info, indent=2))
        return 0

    if args.list_tools:
        schemas = sess.tools.openai_tools_for_agent("movie_os")
        print(json.dumps(schemas, indent=2))
        print(f"\n# {len(schemas)} tools for model={info.get('model')}")
        return 0

    ask = (args.ask or "").strip()
    if args.story and not ask:
        ask = (
            "Film this story into one Blender Session. Prefer movie_run_story "
            "(join=true). Then session_inspect and session_bind. "
            "Confirm body clips include walk/wave/talk when present. "
            "Do not export MP4.\n\nSTORY:\n"
            + args.story.strip()
        )
    if not ask:
        ask = (
            "Film: Walk on a sunny street, wave hello, say goodbye. "
            "Use movie_run_story then session_bind. Then session_inspect."
        )

    print(f"[agentic] provider={info.get('provider')} model={info.get('model')}")
    print(f"[agentic] tools={len(info.get('movie_os_tools') or [])} "
          f"openai_functions={info.get('openai_tool_count')}")
    print(f"[agentic] {info.get('pattern')}")
    print(">>> Blender face.stream_receiver must be running.\n")

    out = sess.handle_movie_os(ask)
    summary = {
        "ok": out.get("ok"),
        "mode": out.get("mode"),
        "message": out.get("message"),
        "tools": [
            {
                "tool": t.get("tool"),
                "ok": (t.get("result") or {}).get("ok"),
                "error": (t.get("result") or {}).get("error"),
            }
            for t in (out.get("tools") or [])
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
