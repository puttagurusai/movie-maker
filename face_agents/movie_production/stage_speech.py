"""
Split stage directions from spoken dialogue.

JSON text is often:
  "Walk forward while saying welcome to our showcase."

TTS must hear only spoken words; body/router use stage + full text.
"""

from __future__ import annotations

import re
from typing import Tuple


def split_stage_and_spoken(text: str) -> Tuple[str, str]:
    """
    Returns (stage, spoken).

    Prefer explicit markers: say / saying / tell / speak / quoted.
    If no marker, treat full string as spoken (dialogue-only beat).
    """
    raw = (text or "").strip()
    if not raw:
        return "", ""

    # Quoted dialogue at end
    m_q = re.search(r"""(?is)^(?P<stage>.*?)(?P<q>["'])(?P<spoken>.+?)(?P=q)\s*$""", raw)
    if m_q and len(m_q.group("spoken").split()) >= 2:
        stage = m_q.group("stage").strip(" ,.;:-")
        spoken = m_q.group("spoken").strip()
        stage = re.sub(r"(?i)\b(while\s+)?(say|saying|tell|speak)\b\s*$", "", stage).strip(" ,.;")
        return stage, spoken

    patterns = [
        # while saying X / saying X
        r"(?is)^(?P<stage>.*?)\bwhile\s+saying\s+(?P<spoken>.+)$",
        r"(?is)^(?P<stage>.*?)\bsaying\s+(?P<spoken>.+)$",
        # say X / please say X
        r"(?is)^(?P<stage>.*?)(?:\bplease\s+)?\bsay(?:s|ing)?\s+(?:this\s+dialog(?:ue)?\s+)?(?P<spoken>.+)$",
        r"(?is)^(?P<stage>.*?)\btell\s+(?:them|him|her|everyone|us)\s+(?P<spoken>.+)$",
        r"(?is)^(?P<stage>.*?)\bspeak\s+(?:the\s+words?\s+)?(?P<spoken>.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, raw)
        if not m:
            continue
        stage = (m.group("stage") or "").strip(" ,.;:-")
        stage = re.sub(r"(?i)\s+\band\b\s*$", "", stage).strip(" ,.;")
        spoken = (m.group("spoken") or "").strip().strip("\"'")
        # strip leaked trailing stage
        spoken = re.sub(
            r"(?i)\s*(while\s+(sitting|walking|standing|dancing).*)\s*$",
            "",
            spoken,
        ).strip(" ,.;")
        if spoken:
            return stage, spoken

    # "while walking, <dialogue>"
    m2 = re.search(
        r"(?is)^while\s+(sitting|walking|standing|dancing)\s*[,:-]\s*(?P<spoken>.+)$",
        raw,
    )
    if m2:
        return f"while {m2.group(1)}", m2.group("spoken").strip().strip("\"'")

    # No marker: full text is spoken
    return "", raw


def clean_spoken_for_tts(spoken: str) -> str:
    """Remove residual stage verbs if they leaked into spoken."""
    s = (spoken or "").strip()
    # Drop leading stage leftovers
    s = re.sub(
        r"(?i)^(walk|walking|run|running|wave|waving|celebrate|sit|sitting)\s+"
        r"(forward|over|in|up|down|to\s+\w+)?\s*(while\s+saying\s+)?",
        "",
        s,
    ).strip(" ,.;")
    return s or spoken.strip()
