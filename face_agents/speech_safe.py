"""
Speech-safe edit helpers (pure Python, no bpy).

If an edit range hits the interior of a spoken block S#, snap to the full
speech span so we never cut a line mid-word.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def range_overlaps_speech_interior(
    clips: Sequence[Dict[str, Any]],
    a: int,
    b: int,
) -> Optional[Tuple[Dict[str, Any], int, int]]:
    """Return (clip, s0, s1) if [a,b] cuts through the middle of a spoken line."""
    a, b = int(min(a, b)), int(max(a, b))
    for c in clips or []:
        s0 = int((c or {}).get("speech_frame_start") or 0)
        s1 = int((c or {}).get("speech_frame_end") or 0)
        if s1 < s0 or s0 <= 0:
            continue
        if a < s1 and b > s0 and (a > s0 or b < s1):
            return c, s0, s1
    return None


def snap_range_speech_safe(
    clips: Sequence[Dict[str, Any]],
    a: int,
    b: int,
) -> Tuple[int, int, str]:
    """
    Never split a spoken line.
    If the edit hits speech interior, snap to the FULL speech block.
    """
    hit = range_overlaps_speech_interior(clips, a, b)
    if not hit:
        return int(a), int(b), ""
    c, s0, s1 = hit
    idx = (c or {}).get("index")
    return (
        min(int(a), s0),
        max(int(b), s1),
        f"snapped to speech S#{idx} [{s0}-{s1}] (do not cut mid-line)",
    )


def speech_blocks(clips: Sequence[Dict[str, Any]]) -> List[Tuple[int, int, int]]:
    """(clip_index, speech_frame_start, speech_frame_end) for dialogue clips."""
    out: List[Tuple[int, int, int]] = []
    for c in clips or []:
        s0 = int((c or {}).get("speech_frame_start") or 0)
        s1 = int((c or {}).get("speech_frame_end") or 0)
        if s1 >= s0 > 0:
            out.append((int((c or {}).get("index") or 0), s0, s1))
    return out
