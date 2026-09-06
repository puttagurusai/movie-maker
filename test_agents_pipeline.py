"""
test_agents_pipeline.py — test face_agents with simulated input, no Blender/TTS needed.

Validates: coordinator, all agents, micro-expressions, policy bridge, per-emotion outputs.
Simulates what orchestrator_agents.py does per sentence, using synthetic audio energy.

Usage:
    python test_agents_pipeline.py
"""

from __future__ import annotations

import sys
import os
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Import agents (no Blender, no TTS needed)
from face_agents.base import FaceContext
from face_agents.brows_agent import BrowsAgent
from face_agents.eyes_agent import EyesAgent
from face_agents.cheeks_agent import CheeksAgent
from face_agents.head_agent import HeadAgent
from face_agents.micro_expression_agent import MicroExpressionAgent
from face_agents.policy_bridge import get_policy
import emotion_map

# Test sentences (same format as orchestrator_agents.py JSON input)
TEST_INPUT = {
    "sentences": [
        {"text": "Hello, great to see you today!",                   "emotion": "happy",     "intensity": 0.90},
        {"text": "I am really worried about what might happen.",      "emotion": "fearful",   "intensity": 0.80},
        {"text": "That was completely and utterly wrong.",            "emotion": "angry",     "intensity": 0.90},
        {"text": "Oh wow, I had no idea that was possible!",          "emotion": "surprised", "intensity": 0.85},
        {"text": "I feel terrible about what happened.",              "emotion": "sad",       "intensity": 0.85},
        {"text": "Let me think about this for a moment...",          "emotion": "thinking",  "intensity": 0.75},
        {"text": "Sure, whatever you say.",                          "emotion": "sarcastic", "intensity": 0.80},
        {"text": "I apologize for the misunderstanding.",            "emotion": "neutral",   "intensity": 0.70},
    ]
}

FPS = 30.0
SAMPLE_DURATION = 3.0  # simulate 3s per sentence
SAMPLE_FRAMES = int(SAMPLE_DURATION * FPS)


def make_synthetic_energy(duration: float, fps: float = 30.0) -> np.ndarray:
    """Synthetic speech energy envelope with natural variation."""
    n = int(duration * fps)
    t = np.linspace(0, duration, n)
    # Speech-like envelope: ramp in, peaks, ramp out
    env = (np.sin(t * 3.5) * 0.3 + 0.5
           + np.sin(t * 7.1) * 0.15
           + np.random.randn(n) * 0.08)
    env = np.clip(env, 0.0, 1.0).astype(np.float32)
    return env


def make_ctx(text: str, emotion: str, intensity: float, duration: float = SAMPLE_DURATION) -> FaceContext:
    ctx = FaceContext(
        t=0.0,
        duration=duration,
        is_speaking=True,
        emotion=emotion,
        intensity=intensity,
        text=text,
        audio_path=None,
        sample_rate=16000,
    )
    ctx.energy_envelope = make_synthetic_energy(duration)
    ctx.brain_enabled = False
    ctx.brain_frames = []
    ctx.lips_frames = []
    return ctx


# Per-emotion minimum expected blendshape values (key sanity check)
EMOTION_EXPECTATIONS = {
    "happy":     {"mouthSmileLeft": 0.30, "cheekSquintLeft": 0.30},
    "sad":       {"mouthFrownLeft": 0.30, "browInnerUp": 0.30},
    "angry":     {"browDownLeft": 0.30,   "eyeSquintLeft": 0.20},
    "surprised": {"browInnerUp": 0.30,   "eyeWideLeft": 0.25},
    "fearful":   {"eyeWideLeft": 0.25,   "browInnerUp": 0.20},
    "disgusted": {"noseSneerLeft": 0.20},
    "sarcastic": {"eyeSquintLeft": 0.15},
    "thinking":  {"browDownLeft": 0.10},
    "neutral":   {},
}


def run_sentence(sentence: dict, idx: int) -> dict:
    text = sentence["text"]
    emotion = sentence["emotion"]
    intensity = float(sentence.get("intensity", 0.7))

    print(f"\n{'='*55}")
    print(f"[Sentence {idx}] {text[:60]}")
    print(f"  emotion={emotion}  intensity={intensity:.2f}")
    print(f"{'='*55}")

    ctx = make_ctx(text, emotion, intensity)

    agents = {
        "brows":  BrowsAgent(),
        "eyes":   EyesAgent(),
        "cheeks": CheeksAgent(),
        "micro":  MicroExpressionAgent(),
    }
    head = HeadAgent()

    for ag in agents.values():
        ag.prepare(ctx)
    head.prepare(ctx)

    # Sample across the sentence at key moments
    sample_times = [0.15, 0.50, 1.00, 1.50, 2.00, 2.50]
    frame_data = []
    micro_active_frames = 0

    for t in sample_times:
        ctx.t = min(t, ctx.duration - 0.01)
        ctx.is_speaking = True

        upper: dict[str, float] = {}
        for ag_name, ag in agents.items():
            part = ag.sample(ctx)
            if ag_name == "micro":
                for k, v in part.items():
                    upper[k] = min(1.0, max(upper.get(k, 0.0), float(v)))
                if any(v > 0.02 for v in part.values()):
                    micro_active_frames += 1
            else:
                upper.update(part)

        pitch, yaw, roll = head.get_head(ctx)
        frame_data.append({
            "t": t,
            "upper": {k: round(v, 3) for k, v in upper.items() if v > 0.005},
            "head": {"pitch": round(pitch, 4), "yaw": round(yaw, 4), "roll": round(roll, 4)},
        })

    # Verify expectations at t=1.0
    frame_1s = next((f for f in frame_data if abs(f["t"] - 1.0) < 0.01), frame_data[2])
    upper_1s = frame_1s["upper"]

    expectations = EMOTION_EXPECTATIONS.get(emotion, {})
    pass_count = 0
    fail_list = []
    for key, min_val in expectations.items():
        actual = upper_1s.get(key, 0.0)
        if actual >= min_val:
            pass_count += 1
        else:
            fail_list.append(f"    FAIL {key}: got {actual:.3f} (need >= {min_val:.3f})")

    print(f"\n  t=1.0s blendshapes: {upper_1s}")
    print(f"  head at t=1.0s: {frame_1s['head']}")
    print(f"  micro triggered: {micro_active_frames}/{len(sample_times)} samples")

    if fail_list:
        print("  EXPRESSION CHECKS: FAIL")
        for f in fail_list:
            print(f)
        status = "FAIL"
    else:
        print(f"  EXPRESSION CHECKS: PASS ({pass_count}/{len(expectations)} requirements met)")
        status = "PASS"

    return {
        "idx": idx,
        "emotion": emotion,
        "status": status,
        "frame_at_1s": upper_1s,
        "micro_hits": micro_active_frames,
    }


def test_policy_bridge():
    print("\n--- POLICY BRIDGE TEST ---")
    pol = get_policy()
    required = ["mouth_gain", "blink_scale", "brow_scale", "micro_intensity", "micro_rate"]
    missing = [k for k in required if k not in pol]
    if missing:
        print(f"  FAIL: missing keys: {missing}")
        return False
    print(f"  mouth_gain={pol['mouth_gain']}  brow_scale={pol['brow_scale']}")
    print(f"  micro_intensity={pol['micro_intensity']}  micro_rate={pol['micro_rate']}")
    print("  [PASS]")
    return True


def main():
    print("=" * 60)
    print("FACE AGENTS PIPELINE TEST")
    print("Tests: all agents, micro-expressions, policy bridge")
    print("Input: test JSON (same format as orchestrator_agents.py)")
    print("=" * 60)

    sentences = TEST_INPUT["sentences"]
    results = []

    for i, sent in enumerate(sentences, 1):
        r = run_sentence(sent, i)
        results.append(r)

    policy_ok = test_policy_bridge()

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    micro_total = sum(r["micro_hits"] for r in results)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Sentences: {len(sentences)}")
    print(f"Expression checks: {passed} PASS / {failed} FAIL")
    print(f"Policy bridge: {'OK' if policy_ok else 'FAIL'}")
    print(f"Micro-expression activations: {micro_total} across all sentences")
    print(f"\nAll agents that would run at 30fps:")
    print(f"  lips  → wav2arkit (needs audio file — skipped in dry-run)")
    print(f"  brows → preset + brain encoder (brain skipped — no model path)")
    print(f"  eyes  → blink + gaze + squint/wide")
    print(f"  cheeks→ smile/frown/nose + brain")
    print(f"  head  → nod + breath + energy peaks")
    print(f"  micro → [NEW] fleeting micro-expressions")
    print(f"\n{'OVERALL: ALL PASS' if failed == 0 and policy_ok else 'OVERALL: SOME ISSUES'}")

    # Write results
    out_path = ROOT / "test_agents_pipeline_report.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"FACE AGENTS PIPELINE TEST REPORT\n{'='*60}\n\n")
        for r in results:
            f.write(f"[{r['status']}] Sentence {r['idx']}: {r['emotion']}\n")
            f.write(f"  t=1.0s: {r['frame_at_1s']}\n")
            f.write(f"  micro: {r['micro_hits']} activations\n\n")
        f.write(f"Policy bridge: {'OK' if policy_ok else 'FAIL'}\n")
        f.write(f"Micro total: {micro_total}\n")
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
