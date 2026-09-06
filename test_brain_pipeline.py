"""
test_brain_pipeline.py — verify real emotion_26d + Brain + optional short TTS

Run:
  python test_brain_pipeline.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

import brain_inference
import prosody_gpu
from emotion_manager import EmotionManager


def test_emotion_26d_real():
    print("\n=== TEST 1 — real emotion_26d (not zeros) ===")
    mgr = EmotionManager("models/brain/emotion_stats.json")
    vecs = []
    for i in range(5):
        t = mgr.get_emotion_26d_tensor("happy", 0.8, device="cpu", noise_scale=0.25)
        assert t.shape == (1, 26), t.shape
        s = float(t.abs().sum())
        assert s > 1e-4, "emotion_26d is zeros — bad stats"
        vecs.append(t.numpy().copy())
        print(f"  run {i+1}: sum={s:.4f} mean={float(t.mean()):.4f} L2={float(t.norm()):.4f}")
    # slight variation across runs
    import numpy as np
    var = sum(float(np.abs(vecs[i] - vecs[0]).sum()) for i in range(1, 5))
    print(f"  noise variation sum={var:.4f} (should be > 0)")
    assert var > 0, "noise not varying"
    # mapping
    assert mgr.resolve_label("surprised") == "surprise"
    assert mgr.resolve_label("thinking") == "neutral"
    print("  PASS")


def test_prosody(wav: str):
    print("\n=== TEST 2 — prosody_gpu ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = prosody_gpu.extract_prosody(wav, device=device)
    print(f"  shape={tuple(p.shape)} device={p.device}")
    assert p.ndim == 2 and p.shape[1] == 3
    assert float(p.min()) >= -1e-5 and float(p.max()) <= 1.0 + 1e-5
    assert float(p.abs().sum()) > 0
    print(f"  ranges ch0=[{p[:,0].min():.3f},{p[:,0].max():.3f}] "
          f"ch1=[{p[:,1].min():.3f},{p[:,1].max():.3f}] "
          f"ch2=[{p[:,2].min():.3f},{p[:,2].max():.3f}]")
    print("  PASS")


def test_run_brain(wav: str):
    print("\n=== TEST 3 — run_brain full 52 ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    frames, fps, raw = brain_inference.run_brain(
        wav, "happy", 0.8, device=device, mouth_only=False
    )
    dt = time.perf_counter() - t0
    print(f"  frames={len(frames)} raw={raw.shape} fps={fps} time={dt:.2f}s")
    assert raw.shape[1] == 52
    assert len(frames) == raw.shape[0]
    assert len(frames[0]) == 52
    jaw = raw[:, 24]
    brow = raw[:, 2]
    print(f"  jawOpen  min/mean/max = {jaw.min():.3f}/{jaw.mean():.3f}/{jaw.max():.3f}")
    print(f"  browInnerUp min/mean/max = {brow.min():.3f}/{brow.mean():.3f}/{brow.max():.3f}")
    assert jaw.max() > jaw.min() or jaw.max() > 0.01
    print("  PASS")


def main():
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    test_emotion_26d_real()

    wav = Path("temp/test_neutral.wav")
    if not wav.is_file():
        # synthesize a short tone WAV for tests
        import numpy as np
        import soundfile as sf
        Path("temp").mkdir(exist_ok=True)
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        y = 0.2 * np.sin(2 * np.pi * 180 * t).astype("float32")
        sf.write(str(wav), y, sr)
        print(f"Created synthetic {wav}")

    test_prosody(str(wav))
    test_run_brain(str(wav))

    print("\n=== ALL UNIT TESTS PASSED ===")
    print("Next: Blender receiver + python orchestrator_brain.py")


if __name__ == "__main__":
    main()
