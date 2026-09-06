"""Analyze temp/verify_run audio: ASR + spectrograms + lip report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

OUT = Path(__file__).resolve().parents[1] / "temp" / "verify_run"
FILES = [
    "s1_speech_0.wav",
    "s1_token_laugh.wav",
    "s1_speech_2.wav",
    "s1_token_eww.wav",
    "s1_speech_4.wav",
    "s1_timeline.wav",
]


def main() -> int:
    asr_results: dict[str, str] = {}
    try:
        import whisper

        print("Loading whisper tiny for ASR...")
        model = whisper.load_model("tiny")
        for f in FILES[:-1]:
            p = OUT / f
            if not p.exists():
                continue
            r = model.transcribe(str(p), fp16=False, language="en")
            asr_results[f] = (r.get("text") or "").strip()
            print(f"ASR {f}: {asr_results[f]!r}")
    except Exception as e:
        print(f"whisper unavailable: {e}")

    fig, axes = plt.subplots(len(FILES), 1, figsize=(10, 2.1 * len(FILES)))
    for ax, f in zip(axes, FILES):
        p = OUT / f
        if not p.exists():
            ax.set_title(f + " MISSING")
            continue
        a, sr = sf.read(str(p), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        ax.specgram(a, Fs=sr, NFFT=512, noverlap=256, cmap="magma")
        label = f.replace("s1_", "").replace(".wav", "")
        asr = asr_results.get(f, "")
        ax.set_ylabel(label[:16], fontsize=8)
        ax.set_title(f"{label}  ASR={asr!r}" if asr else label, fontsize=9)
        ax.set_xlim(0, max(0.1, len(a) / sr))
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("JSON pipeline: Parler speech vs real token SFX")
    fig.tight_layout()
    out_png = OUT / "spectrograms.png"
    fig.savefig(str(out_png), dpi=120)
    print("wrote", out_png)

    print("\n=== SOUND VERDICT ===")
    for f in ["s1_token_laugh.wav", "s1_token_eww.wav", "s1_speech_0.wav"]:
        p = OUT / f
        if not p.exists():
            continue
        a, sr = sf.read(str(p), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        win = max(1, int(0.05 * sr))
        e = np.array(
            [float(np.sqrt(np.mean(a[i : i + win] ** 2))) for i in range(0, max(1, len(a) - win), win)]
        )
        burst = float(e.std() / (e.mean() + 1e-8))
        print(
            f"{f}: {len(a)/sr:.2f}s burstiness={burst:.2f} ASR={asr_results.get(f, '?')!r}"
        )

    print("\n=== LIPSYNC VERDICT ===")
    rep = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    for seg in rep["sentences"][0]["segments"]:
        lips = seg["lips"]
        val = str(seg.get("value", ""))[:24]
        print(
            f"  {seg['kind']:6s} {val:24s} jaw_peak={lips['jaw_peak']:.3f} "
            f"active={lips['active_ratio']:.2f} ok={seg.get('ok')}"
        )
    tl = rep["sentences"][0]["timeline"]["lips"]
    print(
        f"  FULL   timeline                 jaw_peak={tl['jaw_peak']:.3f} "
        f"active={tl['active_ratio']:.2f} frames={tl['n_frames']}"
    )
    lips_pass = tl["jaw_peak"] >= 0.08 and tl["active_ratio"] > 0.3
    print("LIPS:", "PASS" if lips_pass else "FAIL")

    # Sound correctness summary for human
    print("\n=== INTERPRETATION ===")
    print("JSON used:")
    print('  {"text": "Hello friend [laugh] that was funny [eww] oh no", ...}')
    print("Order in s1_timeline.wav:")
    print("  1) Parler speech 'Hello friend'")
    print("  2) REAL SFX laugh clip (not Parler)")
    print("  3) Parler speech 'that was funny'")
    print("  4) REAL SFX eww clip (not Parler)")
    print("  5) Parler speech 'oh no'")
    print("Listen: temp/verify_run/s1_timeline.wav")
    return 0 if lips_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
