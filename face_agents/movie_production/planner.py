"""
Stages 1–2 — Story / script → ShotDetail list via Movie Director.

Intelligence lives in movie_director (LLM + ContinuityBoard).
This module is the thin public API used by the pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .continuity_memory import ContinuityBoard
from .movie_director import direct_production
from .shot_schema import ShotDetail
from .world_state import WorldState


def plan_shots_from_story(
    story: str,
    *,
    title: str = "",
    llm_provider: Any = None,
    world: Optional[WorldState] = None,
    target_total_s: float = 60.0,
    board: Optional[ContinuityBoard] = None,
) -> List[ShotDetail]:
    shots, _board = direct_production(
        story=story,
        title=title,
        llm_provider=llm_provider,
        world=world,
        target_total_s=target_total_s,
    )
    if board is not None:
        board.__dict__.update(_board.__dict__)
    return shots


def plan_shots_from_script(
    script: Dict[str, Any] | List[Any],
    *,
    world: Optional[WorldState] = None,
    llm_provider: Any = None,
    target_total_s: float = 60.0,
    title: str = "",
) -> List[ShotDetail]:
    shots, _ = direct_production(
        script=script,
        title=title or (script.get("title") if isinstance(script, dict) else "") or "",
        llm_provider=llm_provider,
        world=world,
        target_total_s=target_total_s,
    )
    return shots


def plan_with_board(
    *,
    story: str = "",
    script: Optional[Dict[str, Any] | List[Any]] = None,
    title: str = "",
    llm_provider: Any = None,
    world: Optional[WorldState] = None,
    target_total_s: float = 60.0,
) -> Tuple[List[ShotDetail], ContinuityBoard]:
    """Same as direct_production — returns board for pipeline persistence."""
    return direct_production(
        story=story,
        script=script,
        title=title,
        llm_provider=llm_provider,
        world=world,
        target_total_s=target_total_s,
    )
