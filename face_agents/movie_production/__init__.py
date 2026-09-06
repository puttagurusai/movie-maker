"""
movie_production — production movie-clip pipeline (mavie.txt).

Intelligent director + external ContinuityBoard (for memory-less LLMs):
  1-2  Director (LLM or rules) → takes with duration/camera_role/move/pace
  3    World + ContinuityBoard
  4-6  Bake: TTS pad, face, body, multi-cam, face timeline
  7    Package / master timeline
  8    Validation

Entry:
  python run_movie_pipeline.py --story "..." --llm
  python run_movie_pipeline.py --script path.json
"""

from .pipeline import MovieProductionPipeline, run_movie_from_story
from .continuity_memory import ContinuityBoard
from .movie_director import direct_production

__all__ = [
    "MovieProductionPipeline",
    "run_movie_from_story",
    "ContinuityBoard",
    "direct_production",
]
