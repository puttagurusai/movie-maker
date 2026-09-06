"""
test_face_only_triple.py — face-only bake + metrics for:
  1) wav2arkit  (lips / lip-sync)
  2) Audio2Emotion-v2.2  (emotion_26d)
  3) Brain  (upper-face ARKit track conditioned by A2E + HuBERT)

No MoMask, no movie camera, no body Action required for bake.
Optional --play streams face UDP to Blender (body packets suppressed).

Usage:
  # Offline bake + metrics (recommended first)
  python test_face_only_triple.py

  # Reuse existing WAVs under temp/ (skip TTS)
  python test_face_only_triple.py --no-tts

  # Play face only to Blender (receiver must be running)
  python test_face_only_triple.py --play

  # Single emotion
  python test_face_only_triple.py --emotion happy --play

Writes:
  temp/face_triple_report.json
  temp/face_triple_report.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

TEMP = ROOT / "temp"
TEMP.mkdir(exist_ok=True)
REPORT_JSON = TEMP / "face_triple_report.json"
REPORT_TXT = TEMP / "face_triple_report.txt"

# Force face-only product defaults for this test process
os.environ["USE_MOMASK"] = "0"
os.environ["MOMASK_ALL"] = "0"
os.environ["USE_MOVIE_CAMERA"] = "0"
os.environ.setdefault("USE_BRAIN", "1")
os.environ.setdefault("EMOTION_26D_SOURCE", "a2e")

# Emotion suite — director-style labels
SENTENCES: List[Dict[str, Any]] = [
    {
        "id": "happy",
        "text": "Hello everyone. Today I will show you our avatar pipeline.",
        "emotion": "happy",
        "intensity": 0.85,
    },
    {
        "id": "sad",
        "text": "I feel terrible about what happened yesterday.",
        "emotion": "sad",
        "intensity": 0.85,
    },
    {
        "id": "angry",
        "text": "That was completely and utterly unacceptable.",
        "emotion": "angry",
        "intensity": 0.90,
    },
    {
        "id": "surprised",
        "text": "Oh wow, I had no idea that was even possible!",
        "emotion": "surprised",
        "intensity": 0.88,
    },
    {
        "id": "fearful",
        "text": "I am really worried about what might happen next.",
        "emotion": "fearful",
        "intensity": 0.80,
    },
    {
        "id": "neutral",
        "text": "Let me explain how this works carefully.",
        "emotion": "neutral",
        "intensity": 0.65,
    },
]

# Signature upper keys per emotion for scoring
EMOTION_SIGNS: Dict[str, List[Tuple[str, float]]] = {
    "happy": [("mouthSmileLeft", 0.15), ("cheekSquintLeft", 0.10), ("eyeSquintLeft", 0.08)],
    "sad": [("mouthFrownLeft", 0.12), ("browInnerUp", 0.15)],
    "angry": [("browDownLeft", 0.20), ("eyeSquintLeft", 0.10), ("noseSneerLeft", 0.05)],
    "surprised": [("browInnerUp", 0.20), ("eyeWideLeft", 0.15), ("browOuterUpLeft", 0.10)],
    "fearful": [("eyeWideLeft", 0.15), ("browInnerUp", 0.12)],
    "neutral": [],
}


def _peak(frames: List[Dict[str, float]], key: str) -> float:
    if not frames:
        return 0.0
    return max(float(f.get(key, 0.0)) for f in frames)


def _mean_energy_corr(lips: List[Dict], energy: np.ndarray, fps: float = 30.0) -> float:
    """Correlation of jawOpen with speech energy (lip-sync quality proxy)."""
    if not lips or energy is None or len(energy) < 3:
        return 0.0
    n = min(len(lips), len(energy))
    j = np.array([float(lips[i].get("jawOpen", 0.0)) for i in range(n)], dtype=np.float64)
    e = np.asarray(energy[:n], dtype=np.float64)
    if j.std() < 1e-6 or e.std() < 1e-6:
        return 0.0
    j = (j - j.mean()) / (j.std() + 1e-9)
    e = (e - e.mean()) / (e.std() + 1e-9)
    return float(np.clip(np.mean(j * e), -1.0, 1.0))


def ensure_wav(
    item: Dict[str, Any],
    *,
    use_tts: bool,
    idx: int,
) -> Tuple[str, float, int]:
    """Return path, duration, sr. Prefer cached test_*.wav / face_triple_*.wav."""
    wid = item["id"]
    candidates = [
        TEMP / f"face_triple_{wid}.wav",
        TEMP / f"test_{wid}.wav",
        TEMP / f"agents_sentence_1.wav" if wid == "happy" else None,
    ]
    if not use_tts:
        for c in candidates:
            if c and c.is_file():
                audio, sr = sf.read(str(c), dtype="float32")
                if getattr(audio, "ndim", 1) > 1:
                    audio = audio.mean(axis=1)
                return str(c), float(len(audio) / sr), int(sr)

    out = TEMP / f"face_triple_{wid}.wav"
    if out.is_file() and not use_tts:
        audio, sr = sf.read(str(out), dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        return str(out), float(len(audio) / sr), int(sr)

    from parler_voice import load_parler, generate_speech, build_voice_style

    load_parler()
    style = build_voice_style(item["emotion"], float(item["intensity"]))
    print(f"  [TTS] {wid}: {item['text'][:50]}…")
    generate_speech(item["text"], style, str(out), play_audio=False)
    audio, sr = sf.read(str(out), dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return str(out), float(len(audio) / sr), int(sr)


def run_a2e(wav: str, emotion: str, intensity: float) -> Dict[str, Any]:
    t0 = time.perf_counter()
    import audio2emotion as a2e

    if not a2e.is_available():
        a2e.ensure_model()
    # Prefer tensor path used by brain
    try:
        t = a2e.infer_emotion_26d_tensor(
            wav,
            preferred_emotion=emotion,
            preferred_strength=float(os.environ.get("A2E_PREFERRED_STRENGTH", "0.35")),
            intensity=float(intensity),
            device="cpu",
        )
        vec = t.detach().cpu().numpy().astype(np.float32)
        meta = {"preferred": emotion, "tensor": True}
    except TypeError:
        vec = a2e.infer_emotion_26d(wav)
        meta = {"preferred": emotion, "tensor": False}
    except Exception as e:
        return {"ok": False, "error": str(e), "ms": (time.perf_counter() - t0) * 1000}

    # Top explicit-ish magnitude
    l2 = float(np.linalg.norm(vec))
    mx = float(np.max(np.abs(vec)))
    return {
        "ok": True,
        "ms": (time.perf_counter() - t0) * 1000,
        "l2": l2,
        "max_abs": mx,
        "dim": int(vec.shape[0]),
        "meta": meta,
        "vec_head": [round(float(x), 4) for x in vec[:8].tolist()],
    }


def run_wav2arkit(wav: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    import wav2arkit

    frames, fps, raw = wav2arkit.audio_file_to_frames(wav, mouth_only=True, enhance_mouth=True)
    ms = (time.perf_counter() - t0) * 1000
    jaw = _peak(frames, "jawOpen")
    # activity: fraction of frames with jaw > 0.05
    active = sum(1 for f in frames if float(f.get("jawOpen", 0)) > 0.05) / max(1, len(frames))
    return {
        "ok": True,
        "ms": ms,
        "n_frames": len(frames),
        "fps": float(fps),
        "jawOpen_peak": jaw,
        "jaw_active_frac": float(active),
        "mouthUpperUp_peak": _peak(frames, "mouthUpperUpLeft"),
        "frames": frames,  # kept for play/score; stripped in report
    }


def run_brain(wav: str, emotion: str, intensity: float, device: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    import brain_inference

    frames, fps, raw = brain_inference.run_brain(
        wav_path=wav,
        emotion_label=emotion,
        intensity=float(intensity),
        device=device,
        mouth_only=False,
        postprocess=True,
    )
    ms = (time.perf_counter() - t0) * 1000
    # Upper peaks
    upper_keys = [
        "browInnerUp", "browDownLeft", "browOuterUpLeft",
        "eyeWideLeft", "eyeSquintLeft",
        "mouthSmileLeft", "mouthFrownLeft", "cheekSquintLeft", "noseSneerLeft",
    ]
    peaks = {k: _peak(frames, k) for k in upper_keys}
    jaw_brain = 0.0
    if raw is not None and raw.ndim == 2:
        try:
            ji = brain_inference.NVIDIA_ARKIT_ORDER.index("jawOpen")
            jaw_brain = float(raw[:, ji].max())
        except Exception:
            pass
    return {
        "ok": bool(frames),
        "ms": ms,
        "n_frames": len(frames),
        "fps": float(fps),
        "upper_peaks": peaks,
        "jawOpen_peak_brain": jaw_brain,
        "frames": frames,
    }


def score_emotion(emotion: str, brain_peaks: Dict[str, float], preset_peaks: Dict[str, float]) -> Dict[str, Any]:
    signs = EMOTION_SIGNS.get(emotion, [])
    hits = []
    miss = []
    for key, thr in signs:
        bp = float(brain_peaks.get(key, 0.0))
        pp = float(preset_peaks.get(key, 0.0))
        # pass if either brain or (when mixed) preset would show
        ok = bp >= thr * 0.5 or pp >= thr
        (hits if ok else miss).append({"key": key, "thr": thr, "brain": round(bp, 3), "preset": round(pp, 3)})
    score = 1.0 if not signs else len(hits) / len(signs)
    return {"score": score, "hits": hits, "miss": miss}


def score_lips(w2a: Dict[str, Any], energy: np.ndarray) -> Dict[str, Any]:
    jaw = float(w2a.get("jawOpen_peak") or 0)
    active = float(w2a.get("jaw_active_frac") or 0)
    corr = _mean_energy_corr(w2a.get("frames") or [], energy)
    # Heuristic quality 0–1
    # good: jaw peak 0.18–0.45, active 0.25–0.7, corr > 0.15
    jaw_s = 1.0 if 0.18 <= jaw <= 0.50 else (0.6 if jaw > 0.10 else 0.2)
    act_s = 1.0 if 0.20 <= active <= 0.75 else 0.5
    corr_s = float(np.clip((corr + 0.2) / 0.6, 0, 1))  # corr -0.2..0.4 → 0..1
    total = 0.45 * jaw_s + 0.25 * act_s + 0.30 * corr_s
    return {
        "jawOpen_peak": jaw,
        "jaw_active_frac": active,
        "energy_corr": round(corr, 3),
        "score": round(total, 3),
        "verdict": "good" if total >= 0.65 else ("ok" if total >= 0.45 else "weak"),
    }


def emotion_map_peaks(emotion: str, intensity: float) -> Dict[str, float]:
    import emotion_map

    full = emotion_map.get_blendshapes(emotion, intensity)
    keys = [
        "browInnerUp", "browDownLeft", "browOuterUpLeft",
        "eyeWideLeft", "eyeSquintLeft",
        "mouthSmileLeft", "mouthFrownLeft", "cheekSquintLeft", "noseSneerLeft",
    ]
    return {k: float(full.get(k, 0.0)) for k in keys}


def bake_one(item: Dict[str, Any], device: str, use_tts: bool, idx: int) -> Dict[str, Any]:
    print(f"\n{'='*60}\n[{item['id']}] emotion={item['emotion']} I={item['intensity']}\n  {item['text']}\n{'='*60}")
    wav, dur, sr = ensure_wav(item, use_tts=use_tts, idx=idx)
    print(f"  wav={Path(wav).name}  {dur:.2f}s @ {sr}Hz")

    from face_agents.brain_track import compute_energy_envelope

    audio, _ = sf.read(wav, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    energy = compute_energy_envelope(audio, int(sr), fps=30.0)

    # --- 1) A2E ---
    print("  [1/3] Audio2Emotion…")
    a2e = run_a2e(wav, item["emotion"], item["intensity"])
    print(f"       A2E ok={a2e.get('ok')} L2={a2e.get('l2')} ms={a2e.get('ms', 0):.0f}")

    # --- 2) Brain (uses A2E inside) ---
    print("  [2/3] Brain (A2E + HuBERT)…")
    brain = run_brain(wav, item["emotion"], item["intensity"], device)
    print(
        f"       Brain frames={brain.get('n_frames')} "
        f"smile={brain.get('upper_peaks', {}).get('mouthSmileLeft', 0):.3f} "
        f"browIn={brain.get('upper_peaks', {}).get('browInnerUp', 0):.3f} "
        f"ms={brain.get('ms', 0):.0f}"
    )

    # --- 3) wav2arkit lips ---
    print("  [3/3] wav2arkit lips…")
    lips = run_wav2arkit(wav)
    print(
        f"       lips n={lips.get('n_frames')} jawPeak={lips.get('jawOpen_peak', 0):.3f} "
        f"ms={lips.get('ms', 0):.0f}"
    )

    preset = emotion_map_peaks(item["emotion"], item["intensity"])
    lip_score = score_lips(lips, energy)
    emo_score = score_emotion(
        item["emotion"],
        brain.get("upper_peaks") or {},
        preset,
    )

    # Lipsync recommendation: always prefer wav2arkit over Brain jaw for speech
    lip_rec = "wav2arkit"
    # Emotion: if brain signature score high, prefer a2e+brain mix; else boost preset weight
    if emo_score["score"] >= 0.66:
        emo_rec = "a2e+brain_mix (preset~0.55–0.70, brain~0.30–0.45)"
    elif emo_score["score"] >= 0.34:
        emo_rec = "preset_heavy (preset~0.80, brain~0.20) — brain weak on signature"
    else:
        emo_rec = "preset_only (brain miss) — check A2E preferred strength / intensity"

    row = {
        "id": item["id"],
        "text": item["text"],
        "emotion": item["emotion"],
        "intensity": item["intensity"],
        "wav": wav,
        "duration_s": dur,
        "a2e": {k: v for k, v in a2e.items() if k != "vec"},
        "brain": {
            k: v for k, v in brain.items() if k != "frames"
        },
        "lips": {k: v for k, v in lips.items() if k != "frames"},
        "preset_peaks": preset,
        "lip_score": lip_score,
        "emotion_score": emo_score,
        "recommend": {
            "lipsync": lip_rec,
            "emotion": emo_rec,
        },
        # keep frames in memory only for play
        "_lips_frames": lips.get("frames"),
        "_brain_frames": brain.get("frames"),
        "_energy": energy,
        "_sr": sr,
    }
    print(
        f"  → lipsync score={lip_score['score']} ({lip_score['verdict']}) "
        f"corr={lip_score['energy_corr']} | emotion score={emo_score['score']:.2f}"
    )
    print(f"  → REC lips={lip_rec}")
    print(f"  → REC emotion={emo_rec}")
    return row


def play_face_only(rows: List[Dict[str, Any]], device: str) -> None:
    """Stream face UDP only (no body/camera/momask)."""
    from face_agents.coordinator import FaceCoordinator
    import emotion_map

    coord = FaceCoordinator(
        udp_ip="127.0.0.1",
        udp_port=9001,
        fps=30.0,
        use_brain=True,
        device=device,
    )

    # Monkey-patch: suppress body + camera during this face-only session
    _orig_send = coord.send_udp

    def _face_only_send(packet: dict) -> None:
        ptype = packet.get("type")
        if ptype in ("body", "camera"):
            return
        _orig_send(packet)

    coord.send_udp = _face_only_send  # type: ignore
    coord._send_body_sync_start = lambda *a, **k: None  # type: ignore
    coord._send_camera_plan = lambda *a, **k: None  # type: ignore

    print("\n>>> FACE-ONLY PLAY (body/camera UDP suppressed)")
    print(">>> Blender: blender_receiver running; watch face only (no wait)\n")

    for row in rows:
        print(f"\n--- PLAY {row['id']} / {row['emotion']} ---")
        ctx = coord.prepare_sentence(
            text=row["text"],
            emotion=row["emotion"],
            intensity=row["intensity"],
            audio_path=row["wav"],
            duration=row["duration_s"],
            sample_rate=int(row.get("_sr") or 44100),
            body_actions=[],  # no body menu
            body_state="standing",
            body_mode="catalog",
        )
        # Ensure no momask
        ctx.extras.pop("momask_action", None)
        ctx.extras.pop("motion_plan", None)
        audio, sr = sf.read(row["wav"], dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        coord.play_sentence(ctx, row["wav"], int(sr), audio_data=audio)
        time.sleep(0.4)

    print("\n[face-only] play complete.")


def aggregate_recommendation(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    lip_scores = [r["lip_score"]["score"] for r in rows]
    emo_scores = [r["emotion_score"]["score"] for r in rows]
    mean_lip = float(np.mean(lip_scores)) if lip_scores else 0.0
    mean_emo = float(np.mean(emo_scores)) if emo_scores else 0.0

    # Policy recommendation
    if mean_emo >= 0.66:
        policy = {
            "expr_preset_weight": 0.60,
            "brain_brow_weight": 0.40,
            "brain_cheek_weight": 0.38,
            "brain_eye_weight": 0.40,
            "mouth_gain": 1.15,
            "mouth_open_boost": 1.25,
            "target_jaw_peak": 0.22,
            "face_lip_hold_s": 0.14,
            "A2E_PREFERRED_STRENGTH": 0.40,
            "notes": "Brain+A2E carrying emotion well — balanced mix",
        }
    elif mean_emo >= 0.4:
        policy = {
            "expr_preset_weight": 0.78,
            "brain_brow_weight": 0.22,
            "brain_cheek_weight": 0.22,
            "brain_eye_weight": 0.30,
            "mouth_gain": 1.20,
            "mouth_open_boost": 1.30,
            "target_jaw_peak": 0.24,
            "face_lip_hold_s": 0.14,
            "A2E_PREFERRED_STRENGTH": 0.45,
            "notes": "Preset-led emotion; Brain for micro motion",
        }
    else:
        policy = {
            "expr_preset_weight": 0.90,
            "brain_brow_weight": 0.15,
            "brain_cheek_weight": 0.15,
            "brain_eye_weight": 0.20,
            "mouth_gain": 1.20,
            "mouth_open_boost": 1.30,
            "target_jaw_peak": 0.24,
            "face_lip_hold_s": 0.12,
            "A2E_PREFERRED_STRENGTH": 0.50,
            "notes": "Brain weak on signatures — lean presets; keep A2E for 26d conditioning",
        }

    return {
        "mean_lip_score": round(mean_lip, 3),
        "mean_emotion_score": round(mean_emo, 3),
        "lipsync_choice": "wav2arkit (always for speech mouth — never Brain jaw as primary)",
        "emotion_choice": (
            "Audio2Emotion → emotion_26d + Brain upper ARKit, mixed with emotion_map presets"
        ),
        "recommended_policy": policy,
        "disable_for_face_only_live": {
            "USE_MOMASK": "0",
            "USE_MOVIE_CAMERA": "0",
            "USE_BRAIN": "1",
            "EMOTION_26D_SOURCE": "a2e",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Face-only: wav2arkit + A2E + Brain test")
    ap.add_argument("--play", action="store_true", help="Play face UDP to Blender after bake")
    ap.add_argument("--no-tts", action="store_true", help="Prefer existing WAVs (skip TTS when possible)")
    ap.add_argument("--tts", action="store_true", help="Force TTS regenerate")
    ap.add_argument("--emotion", type=str, default="", help="Run only this emotion id")
    ap.add_argument("--device", type=str, default="", help="cuda|cpu (default auto)")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_tts = bool(args.tts) or not args.no_tts
    if args.no_tts:
        use_tts = False

    items = SENTENCES
    if args.emotion:
        items = [s for s in SENTENCES if s["id"] == args.emotion or s["emotion"] == args.emotion]
        if not items:
            print(f"Unknown emotion {args.emotion!r}")
            return 2

    print("=" * 64)
    print("FACE-ONLY TRIPLE TEST")
    print("  1) wav2arkit  → lipsync")
    print("  2) Audio2Emotion-v2.2 → emotion_26d")
    print("  3) Brain → upper-face ARKit (uses A2E + HuBERT)")
    print(f"  device={device}  tts={use_tts}  play={args.play}")
    print("  body/MoMask/camera: OFF for this test")
    print("=" * 64)

    # Preload Brain once
    if os.environ.get("USE_BRAIN", "1") not in ("0", "false"):
        try:
            import brain_inference

            brain_inference.load_brain_model(device)
            print("[OK] Brain loaded")
        except Exception as e:
            print(f"[WARN] Brain load: {e}")

    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        try:
            rows.append(bake_one(item, device=device, use_tts=use_tts, idx=i))
        except Exception as e:
            print(f"  FAIL {item['id']}: {e}")
            rows.append({"id": item["id"], "error": str(e), "ok": False})

    ok_rows = [r for r in rows if "lip_score" in r]
    rec = aggregate_recommendation(ok_rows) if ok_rows else {}

    # Strip heavy frames for disk report
    disk_rows = []
    for r in rows:
        d = {k: v for k, v in r.items() if not k.startswith("_")}
        disk_rows.append(d)

    report = {
        "device": device,
        "sentences": disk_rows,
        "recommendation": rec,
        "stack": {
            "lipsync": "wav2arkit ONNX (models/wav2arkit_cpu)",
            "emotion_26d": "NVIDIA Audio2Emotion-v2.2 (models/audio2emotion_v2.2)",
            "upper_face": "Brain + HuBERT (models/brain) conditioned by A2E 26d",
            "emotion_preset": "emotion_map.py mixed in brows/eyes/cheeks agents",
        },
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "=" * 64,
        "FACE-ONLY TRIPLE REPORT",
        f"device={device}",
        "=" * 64,
        "",
        "Per sentence:",
    ]
    for r in disk_rows:
        if "error" in r:
            lines.append(f"  {r.get('id')}: ERROR {r['error']}")
            continue
        ls = r["lip_score"]
        es = r["emotion_score"]
        lines.append(
            f"  {r['id']:10} lips={ls['score']:.2f}({ls['verdict']}) "
            f"jaw={ls['jawOpen_peak']:.2f} corr={ls['energy_corr']} | "
            f"emo={es['score']:.2f} | a2e_L2={r.get('a2e', {}).get('l2')} "
            f"brain_ms={r.get('brain', {}).get('ms', 0):.0f}"
        )
    lines.append("")
    lines.append("RECOMMENDATION")
    lines.append(f"  lipsync : {rec.get('lipsync_choice')}")
    lines.append(f"  emotion : {rec.get('emotion_choice')}")
    lines.append(f"  mean lip score     = {rec.get('mean_lip_score')}")
    lines.append(f"  mean emotion score = {rec.get('mean_emotion_score')}")
    pol = rec.get("recommended_policy") or {}
    lines.append("  suggested policy:")
    for k, v in pol.items():
        lines.append(f"    {k}: {v}")
    lines.append("")
    lines.append(f"JSON: {REPORT_JSON}")
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))

    if args.play and ok_rows:
        play_face_only(ok_rows, device=device)

    return 0 if ok_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
