"""
feedback_logger.py — lightweight human feedback collector for expression quality.

Usage (in orchestrator_agents.py after each sentence):
    from face_agents.feedback_logger import FeedbackLogger
    fb = FeedbackLogger()
    fb.collect(emotion, intensity, text, audio_path, non_blocking=True)

Feedback is stored in temp/feedback_log.json and summarized on request.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "temp" / "feedback_log.json"


class FeedbackLogger:
    def __init__(self, log_path: Optional[str] = None) -> None:
        self._path = Path(log_path) if log_path else LOG_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._log: list = self._load()
        self._pending_thread: Optional[threading.Thread] = None

    def _load(self) -> list:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(self._log, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def collect(
        self,
        emotion: str,
        intensity: float,
        text: str,
        audio_path: str = "",
        non_blocking: bool = True,
    ) -> None:
        """
        Prompt user for a 1-5 rating after a sentence. Non-blocking runs in a thread.
        If the user skips (Enter), no entry is recorded.
        """
        def _prompt() -> None:
            try:
                print(f"\n  [feedback] Rate expression realism (1-5, Enter=skip): ", end="", flush=True)
                raw = input().strip()
                if not raw:
                    return
                rating = int(raw)
                if not 1 <= rating <= 5:
                    return
                entry = {
                    "ts": datetime.now().isoformat(),
                    "emotion": emotion,
                    "intensity": round(float(intensity), 3),
                    "text_snippet": text[:60],
                    "rating": rating,
                    "audio": os.path.basename(audio_path),
                }
                self._log.append(entry)
                self._save()
                print(f"  [feedback] Saved: {emotion}@{intensity:.2f} → {rating}/5")
            except Exception:
                pass

        if non_blocking:
            self._pending_thread = threading.Thread(target=_prompt, daemon=True)
            self._pending_thread.start()
        else:
            _prompt()

    def wait(self, timeout: float = 8.0) -> None:
        """Wait for pending feedback input to complete."""
        if self._pending_thread and self._pending_thread.is_alive():
            self._pending_thread.join(timeout=timeout)

    def summary(self) -> str:
        """Return a human-readable summary of collected feedback."""
        if not self._log:
            return "No feedback collected yet."

        from collections import defaultdict
        by_emo: dict = defaultdict(list)
        for e in self._log:
            by_emo[e["emotion"]].append(e["rating"])

        lines = [f"Feedback log: {len(self._log)} entries ({self._path.name})"]
        for emo, ratings in sorted(by_emo.items()):
            avg = sum(ratings) / len(ratings)
            lines.append(f"  {emo:15s}: avg={avg:.1f}/5  n={len(ratings)}  ratings={ratings}")
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())

    def get_low_rated(self, threshold: int = 2) -> list:
        """Return entries with rating <= threshold for review."""
        return [e for e in self._log if e["rating"] <= threshold]
