"""
Scoped file tools — any LLM can read/write project files within workspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import Tool, ToolRegistry, ToolResult

# Deny writing huge / binary / secrets paths
_DENY_WRITE_PARTS = (
    "models/",
    ".git/",
    "__pycache__/",
    "temp/",
    ".env",
    "voices-",
    ".onnx",
    ".safetensors",
    ".bin",
)


def register_file_tools(reg: ToolRegistry, workspace: Path) -> None:
    workspace = workspace.resolve()

    def _safe(path: str) -> Path:
        p = (workspace / path).resolve()
        if not str(p).startswith(str(workspace)):
            raise ValueError("Path escapes workspace")
        return p

    def _deny_write(p: Path) -> Optional[str]:
        rel = str(p.relative_to(workspace)).replace("\\", "/")
        for d in _DENY_WRITE_PARTS:
            if d in rel or rel.startswith(d.rstrip("/")):
                return f"Write denied for: {rel}"
        return None

    def list_dir(path: str = ".") -> ToolResult:
        p = _safe(path)
        if not p.is_dir():
            return ToolResult(ok=False, error="Not a directory")
        items = []
        for c in sorted(p.iterdir())[:200]:
            items.append({"name": c.name, "type": "dir" if c.is_dir() else "file"})
        return ToolResult(ok=True, data=items)

    def read_file(path: str, max_chars: int = 12000) -> ToolResult:
        p = _safe(path)
        if not p.is_file():
            return ToolResult(ok=False, error="Not a file")
        if p.stat().st_size > 2_000_000:
            return ToolResult(ok=False, error="File too large")
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]..."
        return ToolResult(ok=True, data={"path": str(path), "content": text})

    def write_file(path: str, content: str) -> ToolResult:
        p = _safe(path)
        reason = _deny_write(p)
        if reason:
            return ToolResult(ok=False, error=reason)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return ToolResult(ok=True, data={"path": str(path), "bytes": len(content.encode("utf-8"))})

    reg.register(Tool(
        name="list_dir",
        description="List files in a workspace-relative directory.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
        handler=list_dir,
        allowed_agents=["director", "lips", "eyes", "brows"],
    ))
    reg.register(Tool(
        name="read_file",
        description="Read a text file from the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer"},
            },
            "required": ["path"],
        },
        handler=read_file,
        allowed_agents=["director", "lips", "eyes", "brows"],
    ))
    reg.register(Tool(
        name="write_file",
        description="Write a text file (code/config only; models/temp denied).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        handler=write_file,
        allowed_agents=["director"],  # only director can write files by default
    ))
