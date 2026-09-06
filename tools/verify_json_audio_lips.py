"""
End-to-end verify: JSON → segmented audio (Parler speech + real vocal clips) → lips.

  python tools/verify_json_audio_lips.py
  python tools/verify_json_audio_lips.py temp/test_pipeline_input.json

Outputs under temp/verify_run/:
  part_*.wav, token_*.wav, timeline.wav, report.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

OUT = ROOT / "temp" / "verify_run"
OUT.mkdir(parents=True, exist_ok=True)


def load_sentences(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "sentences" in data:
        return data["sentences"]
    if isinstance(data, list):
        return data
    raise ValueError("JSON must be {sentences:[...]} or a list")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def zcr(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.sign(x)))) / 2.0)


def analyze_clip(path: Path) -> dict:
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = np.asarray(a, dtype=np.float32)
    dur = len(a) / float(sr)
    # spectral centroid proxy via FFT
    n = min(len(a), int(sr * 0.5))
    if n > 64:
        spec = np.abs(np.fft.rfft(a[:n] * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        cent = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
    else:
        cent = 0.0
    return {
        "path": str(path),
        "duration_s": round(dur, 3),
        "sr": int(sr),
        "rms": round(rms(a), 4),
        "peak": round(float(np.abs(a).max()), 4),
        "zcr": round(zcr(a), 4),
        "spectral_centroid_hz": round(cent, 1),
    }


def try_transcribe(path: Path) -> str | None:
    """Optional: if whisper available, ASR the clip."""
    try:
        import whisper  # type: ignore
    except Exception:
        return None
    try:
        model = whisper.load_model("tiny")
        r = model.transcribe(str(path), fp16=False, language="en")
        return (r.get("text") or "").strip()
    except Exception as e:
        print(f"  [whisper] fail: {e}")
        return None


def bake_lips(path: Path) -> dict:
    import wav2arkit

    frames, fps, raw = wav2arkit.audio_file_to_frames(
        str(path), mouth_only=True, enhance_mouth=True
    )
    jaws = [float(fr.get("jawOpen", 0.0)) for fr in frames]
    return {
        "n_frames": len(frames),
        "fps": float(fps),
        "jaw_peak": round(max(jaws) if jaws else 0.0, 4),
        "jaw_mean": round(float(np.mean(jaws)) if jaws else 0.0, 4),
        "jaw_std": round(float(np.std(jaws)) if jaws else 0.0, 4),
        "active_ratio": round(
            float(np.mean([1.0 if j > 0.04 else 0.0 for j in jaws])) if jaws else 0.0,
            3,
        ),
    }


def main() -> int:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "temp" / "test_pipeline_input.json"
    if not json_path.is_file():
        print(f"Missing {json_path}")
        return 1

    sentences = load_sentences(json_path)
    print("=" * 60)
    print("VERIFY JSON → AUDIO → LIPSYNC")
    print(f"Input: {json_path}")
    print("=" * 60)

    from face_agents.expression_tokens import (
        split_token_segments,
        prepare_token_clip,
        strip_tokens,
    )
    from parler_voice import load_parler, build_voice_style, generate_speech
    import wav2arkit

    print("\n[load] Parler + wav2arkit...")
    t0 = time.time()
    load_parler()
    wav2arkit.load_session()
    print(f"[load] ready in {time.time()-t0:.1f}s")

    report = {"input": str(json_path), "sentences": []}
    all_ok = True

    for si, sent in enumerate(sentences, 1):
        text = sent.get("text", "")
        emotion = sent.get("emotion", "neutral")
        intensity = float(sent.get("intensity", 0.7))
        print(f"\n--- Sentence {si}: {text!r} emotion={emotion} ---")
        segs = split_token_segments(text)
        print("  segments:", [(s.kind, s.value) for s in segs])

        timeline = []  # list of (label, audio, sr)
        sent_report = {"text": text, "segments": []}

        for pi, seg in enumerate(segs):
            if seg.kind == "token":
                print(f"\n  [TOKEN] [{seg.value}] — must use pre-recorded clip, NOT Parler")
                prep = prepare_token_clip(seg.value)
                if prep is None:
                    print("    FAIL: no clip on disk")
                    all_ok = False
                    sent_report["segments"].append({"kind": "token", "value": seg.value, "ok": False})
                    continue
                audio, sr, clip_path = prep
                out = OUT / f"s{si}_token_{seg.value}.wav"
                sf.write(str(out), audio, sr)
                meta = analyze_clip(out)
                lips = bake_lips(out)
                # Critical: SOURCE must be real SFX / OpenVoice VC, never pure Parler stubs.
                source = (ROOT / "temp" / "vocal_sounds" / "SOURCE.txt")
                src_txt = source.read_text(encoding="utf-8", errors="ignore") if source.is_file() else ""
                src_low = src_txt.lower()
                src_ok = source.is_file() and (
                    "not parler" in src_low
                    or "mixkit" in src_low
                    or "openvoice" in src_low
                    or "voice-convert" in src_low
                    or "voice convert" in src_low
                ) and "generate_missing=true" not in src_low
                # lips must move for laugh/gasp
                lips_ok = lips["jaw_peak"] >= 0.08 or lips["active_ratio"] >= 0.15
                print(f"    audio: {meta['duration_s']}s peak={meta['peak']} zcr={meta['zcr']} cent={meta['spectral_centroid_hz']}Hz")
                print(f"    lips:  jaw_peak={lips['jaw_peak']} mean={lips['jaw_mean']} active={lips['active_ratio']} frames={lips['n_frames']}")
                print(f"    source_ok={src_ok} lips_ok={lips_ok}")
                # optional ASR — should NOT clearly be the word alone if speech TTS waste
                asr = try_transcribe(out)
                if asr is not None:
                    print(f"    ASR: {asr!r}")
                seg_ok = src_ok and lips_ok and meta["duration_s"] > 0.15
                if not seg_ok:
                    all_ok = False
                sent_report["segments"].append({
                    "kind": "token",
                    "value": seg.value,
                    "audio": meta,
                    "lips": lips,
                    "asr": asr,
                    "ok": seg_ok,
                })
                timeline.append((f"[{seg.value}]", audio, sr))
            else:
                speech = seg.value.strip()
                print(f"\n  [SPEECH] Parler TTS: {speech!r}")
                # refuse if pure token word
                from face_agents.expression_tokens import TOKEN_RECIPES, TOKEN_SOUNDS, _resolve_alias
                bare = "".join(c for c in speech.lower() if c.isalpha())
                if bare and (_resolve_alias(bare) in TOKEN_RECIPES or bare in TOKEN_SOUNDS) and len(speech.split()) <= 2:
                    print("    FAIL: speech segment looks like a token word — would be Parler waste")
                    all_ok = False
                    sent_report["segments"].append({"kind": "speech", "value": speech, "ok": False, "error": "token word"})
                    continue
                out = OUT / f"s{si}_speech_{pi}.wav"
                style = build_voice_style(emotion, intensity)
                audio, sr = generate_speech(
                    text=speech,
                    voice_style=style,
                    output_path=str(out),
                    play_audio=False,
                )
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                meta = analyze_clip(out)
                lips = bake_lips(out)
                asr = try_transcribe(out)
                # speech should have lip activity
                lips_ok = lips["jaw_peak"] >= 0.08
                print(f"    audio: {meta['duration_s']}s peak={meta['peak']}")
                print(f"    lips:  jaw_peak={lips['jaw_peak']} active={lips['active_ratio']}")
                if asr is not None:
                    print(f"    ASR: {asr!r}")
                    # should not be empty for speech
                if not lips_ok:
                    print("    WARN: weak lips on speech (may still be OK for short words)")
                sent_report["segments"].append({
                    "kind": "speech",
                    "value": speech,
                    "audio": meta,
                    "lips": lips,
                    "asr": asr,
                    "ok": meta["duration_s"] > 0.1,
                })
                if meta["duration_s"] <= 0.1:
                    all_ok = False
                timeline.append((speech, audio, sr))

        # stitch timeline for listening
        if timeline:
            # resample all to 22050
            target_sr = 22050
            parts = []
            gap = np.zeros(int(0.12 * target_sr), dtype=np.float32)
            for label, a, sr in timeline:
                a = np.asarray(a, dtype=np.float32).reshape(-1)
                if sr != target_sr:
                    try:
                        import scipy.signal as sig
                        a = sig.resample(a, int(len(a) * target_sr / sr)).astype(np.float32)
                    except Exception:
                        pass
                parts.append(a)
                parts.append(gap)
            full = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
            tl_path = OUT / f"s{si}_timeline.wav"
            sf.write(str(tl_path), full, target_sr)
            print(f"\n  [timeline] wrote {tl_path} ({len(full)/target_sr:.2f}s)")
            # bake lips on full timeline
            tl_lips = bake_lips(tl_path)
            print(f"  [timeline lips] jaw_peak={tl_lips['jaw_peak']} active={tl_lips['active_ratio']} frames={tl_lips['n_frames']}")
            sent_report["timeline"] = {
                "path": str(tl_path),
                "duration_s": round(len(full) / target_sr, 3),
                "lips": tl_lips,
            }
            # full timeline should have lip motion
            if tl_lips["jaw_peak"] < 0.08:
                all_ok = False
                print("  FAIL timeline lips too flat")
            else:
                print("  PASS timeline lips move")

        report["sentences"].append(sent_report)

    rep_path = OUT / "report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"Report: {rep_path}")
    print(f"Audio dir: {OUT}")
    print("RESULT:", "PASS" if all_ok else "FAIL (see above)")
    print("=" * 60)
    print("\nListen to: temp/verify_run/s1_timeline.wav")
    print("  Order: speech → real laugh clip → speech → eww clip → speech")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import time
    raise SystemExit(main())
