"""
File bridge: Blender N-panel chat ↔ orchestrator.

  temp/blender_chat/inbox.jsonl   Blender → orchestrator (user lines)
  temp/blender_chat/outbox.jsonl  orchestrator → Blender (log / replies)
  temp/blender_chat/ui_active     touched while the Blender panel is registered
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
CHAT_DIR = ROOT / "temp" / "blender_chat"
INBOX = CHAT_DIR / "inbox.jsonl"
OUTBOX = CHAT_DIR / "outbox.jsonl"
UI_FLAG = CHAT_DIR / "ui_active"
COMMANDS = CHAT_DIR / "commands.jsonl"
SESSION_SNAP = CHAT_DIR / "session.json"
MODE_FILE = CHAT_DIR / "pipeline_mode.txt"  # "full" | "scene"


def ensure_dir() -> Path:
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    return CHAT_DIR


def mark_ui_active() -> None:
    ensure_dir()
    UI_FLAG.write_text(str(time.time()), encoding="utf-8")


def ui_is_active(max_age_s: float = 120.0) -> bool:
    try:
        if not UI_FLAG.is_file():
            return False
        age = time.time() - UI_FLAG.stat().st_mtime
        return age <= float(max_age_s)
    except Exception:
        return False


def push_command(op: str, **fields) -> None:
    ensure_dir()
    rec = {"t": time.time(), "op": str(op or "").strip(), **fields}
    with COMMANDS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_pipeline_mode() -> str:
    """Return 'full' (chat+motion+speech) or 'scene' (Look_Set only)."""
    try:
        if MODE_FILE.is_file():
            m = MODE_FILE.read_text(encoding="utf-8").strip().lower()
            if m in ("scene", "look", "set"):
                return "scene"
            if m in ("full", "chat", "performance"):
                return "full"
    except Exception:
        pass
    return "full"


def set_pipeline_mode(mode: str) -> str:
    ensure_dir()
    m = str(mode or "full").strip().lower()
    if m in ("scene", "look", "set"):
        m = "scene"
    else:
        m = "full"
    MODE_FILE.write_text(m, encoding="utf-8")
    return m


def push_inbox(text: str, *, source: str = "blender", mode: str = "") -> None:
    ensure_dir()
    rec = {
        "t": time.time(),
        "text": str(text or "").strip(),
        "src": source,
        "mode": str(mode or get_pipeline_mode()),
    }
    with INBOX.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def pop_inbox() -> Optional[str]:
    """Pop the oldest pending user line. None if empty.

    Also returns a mode hint via pop_inbox_record for callers that need it;
    this wrapper keeps backward-compatible text-only API.
    """
    rec = pop_inbox_record()
    if rec is None:
        return None
    return str(rec.get("text") or "").strip() or None


def pop_inbox_record() -> Optional[dict]:
    """Pop oldest inbox row as dict {text, mode, src}."""
    if not INBOX.is_file():
        return None
    try:
        rows = [ln for ln in INBOX.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return None
    if not rows:
        return None
    first, rest = rows[0], rows[1:]
    try:
        INBOX.write_text(("\n".join(rest) + ("\n" if rest else "")), encoding="utf-8")
    except Exception:
        return None
    try:
        data = json.loads(first)
        text = str(data.get("text") or "").strip()
        if not text:
            return None
        mode = str(data.get("mode") or get_pipeline_mode()).strip().lower()
        if mode in ("look", "set"):
            mode = "scene"
        if mode not in ("full", "scene"):
            mode = get_pipeline_mode()
        return {"text": text, "mode": mode, "src": str(data.get("src") or "")}
    except Exception:
        t = first.strip()
        return {"text": t, "mode": get_pipeline_mode(), "src": ""} if t else None


def _looks_like_json(text: str) -> bool:
    s = str(text or "").strip()
    return (s.startswith("{") and s.endswith("}")) or (
        s.startswith("[") and s.endswith("]")
    )


def push_outbox(
    role: str,
    text: str,
    *,
    kind: str = "",
    payload: str = "",
) -> None:
    ensure_dir()
    raw = str(text or "")
    role_s = str(role or "sys")
    kind_s = str(kind or "").strip().lower()
    if not kind_s:
        if role_s == "json" or _looks_like_json(raw):
            kind_s = "json"
        elif role_s == "user":
            kind_s = "chat"
        elif role_s == "sys":
            kind_s = "sys"
        else:
            kind_s = "chat"
    rec = {
        "t": time.time(),
        "role": role_s,
        "kind": kind_s,
        "text": raw,
        "payload": str(payload or ""),
    }
    line = json.dumps(rec, ensure_ascii=False)
    with OUTBOX.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        rows = OUTBOX.read_text(encoding="utf-8").splitlines()
        if len(rows) > 120:
            OUTBOX.write_text("\n".join(rows[-120:]) + "\n", encoding="utf-8")
    except Exception:
        pass
    # Mirror into the durable sidebar history (same file the Blender panel reads).
    try:
        hist = CHAT_DIR / "history.jsonl"
        with hist.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        rows = hist.read_text(encoding="utf-8").splitlines()
        if len(rows) > 160:
            hist.write_text("\n".join(rows[-160:]) + "\n", encoding="utf-8")
    except Exception:
        pass


def read_outbox(limit: int = 24) -> List[dict]:
    if not OUTBOX.is_file():
        return []
    try:
        rows = [ln for ln in OUTBOX.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return []
    out = []
    for ln in rows[-int(limit) :]:
        try:
            out.append(json.loads(ln))
        except Exception:
            out.append({"role": "sys", "text": ln})
    return out
