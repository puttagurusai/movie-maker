"""
test_wav2arkit.py — verify ONNX audio→ARKit model.

Usage:
  python test_wav2arkit.py
  python test_wav2arkit.py temp/test_neutral.wav
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from wav2arkit import audio_file_to_frames, infer_blendshapes, load_session

TEMP = Path("temp")


def main() -> int:
    print("=== wav2arkit (myned-ai) test ===")
    t0 = time.time()
    load_session()
    print(f"  load: {(time.time() - t0) * 1000:.0f} ms")

    # Dummy 1s
    dummy = (np.random.randn(16000) * 0.02).astype(np.float32)
    t0 = time.time()
    out = infer_blendshapes(dummy)
    ms = (time.time() - t0) * 1000
    print(f"  dummy 1s audio → shape {out.shape} in {ms:.0f} ms")
    ok = out.ndim == 2 and out.shape[1] == 52 and out.shape[0] >= 25
    print(f"  shape check: {'PASS' if ok else 'FAIL'}")

    wav = None
    if len(sys.argv) > 1:
        wav = Path(sys.argv[1])
    else:
        for candidate in [
            TEMP / "test_neutral.wav",
            TEMP / "test_happy.wav",
            TEMP / "sentence_1.wav",
        ]:
            if candidate.is_file():
                wav = candidate
                break

    if wav and wav.is_file():
        print(f"\n  Running on: {wav}")
        t0 = time.time()
        frames, fps, raw = audio_file_to_frames(str(wav), mouth_only=True)
        ms = (time.time() - t0) * 1000
        jaw = raw[:, 24]  # jawOpen index per config
        print(f"  frames={len(frames)} fps={fps} time={ms:.0f} ms")
        print(f"  jawOpen min/mean/max = {jaw.min():.3f} / {jaw.mean():.3f} / {jaw.max():.3f}")
        print(f"  sample frame[0] keys: {list(frames[0].keys())[:8]}...")
        print("  PASS (file inference)" if frames else "  FAIL")
    else:
        print("\n  No test WAV found (optional). Generate with test_parler.py first.")

    print("\nDone. Next: set LIPSYNC_ENGINE='wav2arkit' in orchestrator and run pipeline.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
