"""
run_llm_agents.py — interactive multi-agent console (model-agnostic).

Setup:
  1. Copy llm_fw/config.example.json → llm_fw/config.json (already present)
  2. Set API key for your provider
  3. python run_llm_agents.py

Commands:
  /info              show provider + policy
  /agent lips|eyes|brows|director   talk to one agent
  /policy            show FACE_POLICY
  /quit              exit
  otherwise          director multi-agent turn
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ensure project root on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_fw.runtime.session import AgentSession
from llm_fw.tools.face_tools import get_face_policy


def main() -> None:
    print("=" * 60)
    print("TalkFace LLM multi-agent framework")
    print("Providers are swappable via llm_fw/config.json")
    print("=" * 60)

    try:
        session = AgentSession(workspace=ROOT)
    except Exception as e:
        print(f"Failed to start session: {e}")
        print("Edit llm_fw/config.json and set API keys.")
        sys.exit(1)

    print(json.dumps(session.info(), indent=2))
    print("\nType a goal (e.g. 'Happy energetic line, lips clearer, calmer eyes')")
    print("Commands: /info  /policy  /agent lips|eyes|brows|director  /quit\n")

    mode = "multi"  # multi | lips | eyes | brows | director

    while True:
        try:
            line = input(f"[{mode}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line:
            continue
        if line.lower() in ("/quit", "/exit", "quit", "exit"):
            break
        if line.lower() == "/info":
            print(json.dumps(session.info(), indent=2))
            continue
        if line.lower() == "/policy":
            print(json.dumps(get_face_policy(), indent=2))
            continue
        if line.lower().startswith("/agent"):
            parts = line.split()
            if len(parts) >= 2 and parts[1] in ("lips", "eyes", "brows", "director", "multi"):
                mode = parts[1]
                print(f"Mode set to: {mode}")
            else:
                print("Usage: /agent lips|eyes|brows|director|multi")
            continue

        try:
            if mode == "multi":
                result = session.handle(line)
            else:
                result = session.handle_single_agent(mode, line)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
