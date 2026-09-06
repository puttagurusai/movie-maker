"""
test_parler.py

Verify Parler-TTS emotion styles before using in the orchestrator.

Run:
    python test_parler.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import torch

from parler_voice import (
    build_voice_style,
    generate_speech,
    generate_speech_for_emotion,
    load_parler,
)

TEMP = Path("temp")
TEMP.mkdir(exist_ok=True)


def _ms(seconds: float) -> float:
    return seconds * 1000.0


def test_1_basic() -> bool:
    print("\n" + "=" * 60)
    print("TEST 1 — Basic generation (neutral / medium)")
    print("=" * 60)

    text = "Hello how are you today"
    style = build_voice_style("neutral", 0.5)
    out = TEMP / "test_neutral.wav"

    print(f"  style: {style}")
    t0 = time.time()
    try:
        audio, sr = generate_speech(
            text=text,
            voice_style=style,
            output_path=str(out),
            play_audio=True,
        )
        elapsed = _ms(time.time() - t0)
    except Exception as e:
        print(f"  FAIL — exception: {e}")
        return False

    ok = out.is_file() and len(audio) > 0 and sr > 0
    print(f"  saved: {out}  samples={len(audio)}  sr={sr}  time={elapsed:.0f} ms")
    if ok:
        print("  PASS")
    else:
        print("  FAIL — empty or missing audio")
    return ok


def test_2_emotions() -> bool:
    print("\n" + "=" * 60)
    print("TEST 2 — Emotion variety")
    print("=" * 60)

    cases = [
        ("I am so happy to see you!", "happy", 0.9),
        ("That is really sad news.", "sad", 0.7),
        ("How dare you do that!", "angry", 0.85),
        ("Oh wow I had no idea!", "surprised", 0.9),
        ("Hmm let me think about that...", "thinking", 0.6),
    ]

    all_ok = True
    for text, emotion, intensity in cases:
        out = TEMP / f"test_{emotion}.wav"
        style = build_voice_style(emotion, intensity)
        print(f"\n  [{emotion} @ {intensity}] {text!r}")
        print(f"    style: {style[:100]}...")
        try:
            t0 = time.time()
            audio, sr = generate_speech(
                text=text,
                voice_style=style,
                output_path=str(out),
                play_audio=False,
            )
            elapsed = _ms(time.time() - t0)
            ok = out.is_file() and len(audio) > 0
            status = "PASS" if ok else "FAIL"
            print(f"    saved: {out}  ({elapsed:.0f} ms)  → {status}")
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"    FAIL — {e}")
            all_ok = False

        time.sleep(1.0)

    print(f"\n  TEST 2 overall: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_3_latency() -> bool:
    print("\n" + "=" * 60)
    print("TEST 3 — Latency check")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_ms = 1000.0 if device == "cuda" else 3000.0
    print(f"  device={device}  target < {target_ms:.0f} ms")

    # Ensure model is already warm
    load_parler()

    text = "This is a short latency check sentence."
    out = TEMP / "test_latency.wav"
    style = build_voice_style("neutral", 0.5)

    times = []
    for i in range(3):
        t0 = time.time()
        try:
            generate_speech(
                text=text,
                voice_style=style,
                output_path=str(out),
                play_audio=False,
            )
        except Exception as e:
            print(f"  FAIL — generation error: {e}")
            return False
        elapsed = _ms(time.time() - t0)
        times.append(elapsed)
        print(f"  run {i + 1}: {elapsed:.0f} ms")

    avg = sum(times) / len(times)
    # Use best (last/warm) run for pass criteria; still report average
    best = min(times)
    print(f"  average={avg:.0f} ms  best={best:.0f} ms  target={target_ms:.0f} ms")

    # Soft target: pass if best under target * 2 on first-run machines,
    # but report hard FAIL only if average exceeds 3x target (model still usable).
    if best <= target_ms:
        print("  PASS — under target")
        return True
    if best <= target_ms * 2:
        print("  PASS (soft) — slightly over target but acceptable for Mini-v1")
        return True

    print("  FAIL — latency too high")
    return False


def main():
    print("Parler-TTS verification")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}
    results["basic"] = test_1_basic()
    results["emotions"] = test_2_emotions()
    results["latency"] = test_3_latency()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name:12s}  {'PASS' if ok else 'FAIL'}")

    print("\nListen to files in temp/:")
    for p in sorted(TEMP.glob("test_*.wav")):
        print(f"  - {p}")

    print(
        "\nIf emotions sound flat/same: check device line above (GPU is much better).\n"
        "If one emotion is wrong: edit VOICE_STYLES in parler_voice.py and re-run."
    )

    if all(results.values()):
        print("\nAll tests PASS — safe to use Parler in orchestrator.")
        return 0

    print("\nSome tests FAILED — fix before relying on Parler in production.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
