"""
agent_comm.py — file-based communication layer for the face agents framework.

Reads USER: commands from agents.txt (project root) and writes AGENT: responses.
Also parses SET key=value policy overrides that policy_bridge can consume.

Protocol in agents.txt:
  USER: <your instruction>
  AGENT [timestamp]: <agent response>
  [PROCESSED]

SET overrides (parsed by get_policy_overrides, no [PROCESSED] needed):
  SET mouth_gain=1.3
  SET micro_intensity=0.5
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


AGENTS_TXT: Path = Path(__file__).resolve().parents[1] / "agents.txt"

_USER_PREFIX = "USER:"
_AGENT_PREFIX = "AGENT"
_PROCESSED_MARKER = "[PROCESSED]"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_new_commands() -> List[Tuple[int, str]]:
    """
    Return list of (line_index, command_text) for unprocessed USER: commands.

    A command is unprocessed if no AGENT: line or [PROCESSED] marker follows
    it before the next USER: block or end of file.
    """
    if not AGENTS_TXT.is_file():
        return []

    lines = AGENTS_TXT.read_text(encoding="utf-8").splitlines()
    commands: List[Tuple[int, str]] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.upper().startswith(_USER_PREFIX):
            cmd_text = stripped[len(_USER_PREFIX):].strip()
            cmd_line = i
            has_response = False
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()
                if nxt.upper().startswith(_USER_PREFIX):
                    break
                if (nxt.upper().startswith(_AGENT_PREFIX)
                        or nxt == _PROCESSED_MARKER):
                    has_response = True
                    break
                j += 1
            if not has_response and cmd_text:
                commands.append((cmd_line, cmd_text))
        i += 1
    return commands


def write_response(command_line: int, response: str) -> None:
    """
    Insert AGENT response + [PROCESSED] directly after command_line.
    """
    if not AGENTS_TXT.is_file():
        return

    lines = AGENTS_TXT.read_text(encoding="utf-8").splitlines()
    ts = _now()
    insert = [
        f"AGENT [{ts}]: {response}",
        _PROCESSED_MARKER,
        "",
    ]
    # Insert after the command line
    idx = command_line + 1
    lines = lines[:idx] + insert + lines[idx:]
    AGENTS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_agent_message(message: str, tag: str = "INFO") -> None:
    """Append a standalone AGENT message block (not in response to a command)."""
    ts = _now()
    block = f"\nAGENT [{ts}] [{tag}]: {message}\n"
    with AGENTS_TXT.open("a", encoding="utf-8") as f:
        f.write(block)


def get_policy_overrides() -> Dict[str, object]:
    """
    Scan agents.txt for SET key=value lines and return the final value
    for each key (last SET wins).  Values are cast to float if possible.

    Example lines in agents.txt:
      SET mouth_gain=1.3
      SET micro_intensity=0.55
    """
    if not AGENTS_TXT.is_file():
        return {}

    overrides: Dict[str, object] = {}
    text = AGENTS_TXT.read_text(encoding="utf-8")
    for m in re.finditer(r"^SET\s+(\w+)\s*=\s*(.+)$", text, re.MULTILINE):
        key = m.group(1).strip()
        raw = m.group(2).strip()
        try:
            overrides[key] = float(raw)
        except ValueError:
            overrides[key] = raw
    return overrides


def read_all() -> str:
    """Return full content of agents.txt (for agent introspection)."""
    if not AGENTS_TXT.is_file():
        return ""
    return AGENTS_TXT.read_text(encoding="utf-8")
