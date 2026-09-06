"""
Replace Parler-generated "vocal" clips (spoken words) with REAL human SFX.

Why: Parler is text-to-speech. It cannot produce real laughs/gasps/sighs —
only spoken "ha ha ha" which breaks immersion.

What this does:
  1. Backs up current temp/vocal_sounds/ → temp/vocal_sounds_parler_backup/
  2. Downloads free Mixkit royalty-free human vocal SFX (when IDs resolve)
  3. Converts to mono WAV 44.1kHz, trims, normalizes
  4. Writes into temp/vocal_sounds/vocal_*.wav used by the token pipeline

Matching Parler voice exactly:
  - Real SFX will NOT perfectly match Parler speaker identity.
  - Best same-voice options (manual):
      a) Record yourself doing laugh/eww/gasp (same mic), drop into vocal_sounds/
      b) Use RVC/voice conversion later to map real SFX → your speaker
  - Optional light pitch shift toward a reference Parler clip (--match-pitch)

Usage:
  python tools/build_real_vocals.py
  python tools/build_real_vocals.py --match-pitch temp/agents_sentence_1.wav
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("pip install soundfile")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
SOUND_DIR = ROOT / "temp" / "vocal_sounds"
BACKUP_DIR = ROOT / "temp" / "vocal_sounds_parler_backup"

# Mixkit free SFX: active_storage IDs (royalty-free under Mixkit License)
# Map token → list of candidate Mixkit asset IDs (first that downloads wins)
# IDs verified pattern: https://assets.mixkit.co/active_storage/sfx/{id}/{id}-preview.mp3
MIXKIT_CANDIDATES: dict[str, list[int]] = {
    # Will be filled after scrape + manual known-good IDs
}

# Direct preview URLs that work without login (filled at runtime by scrape)
SCRAPE_PAGES = [
    "https://mixkit.co/free-sound-effects/laugh/",
    "https://mixkit.co/free-sound-effects/gasp/",
    "https://mixkit.co/free-sound-effects/human/",
    "https://mixkit.co/free-sound-effects/voices/",
    "https://mixkit.co/free-sound-effects/cough/",
    "https://mixkit.co/free-sound-effects/breath/",
    "https://mixkit.co/free-sound-effects/scream/",
    "https://mixkit.co/free-sound-effects/cry/",
]

# Prefer these keywords when assigning scraped assets to tokens
TOKEN_KEYWORDS: dict[str, list[str]] = {
    "laugh": ["male casual laugh", "human male casual laugh", "person nasal laugh", "male laugh", "laugh"],
    "chuckle": ["giggle", "chuckle", "nasal laugh", "small", "laugh"],
    "gasp": ["gasp", "astonished", "female astonished gasp", "surprise"],
    "sigh": ["sigh", "breath"],
    "cough": ["cough", "ahem"],
    "sneeze": ["sneeze"],
    "yawn": ["yawn"],
    "cry": ["cry", "crying", "sob"],
    "sob": ["sob", "cry"],
    "hmm": ["hmm", "thinking", "hum"],
    "eww": ["disgust", "eww", "yuck", "ugh"],
    "ugh": ["ugh", "disgust", "groan"],
    "wow": ["wow", "amazed"],
    "groan": ["groan", "moan"],
    "sniff": ["sniff", "sniffle"],
    "nervous": ["nervous", "giggle"],
    "grunt": ["grunt", "effort"],
    "scoff": ["scoff", "pff", "snort"],
    "wince": ["pain", "ouch", "wince"],
    "smirk": ["chuckle", "smirk", "heh"],
}


def _http_get(url: str, timeout: int = 30) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        print(f"  fail {url}: {e}")
        return None


def scrape_mixkit_assets() -> list[dict]:
    """Return list of {id, title, url} from Mixkit free SFX pages."""
    assets = []
    seen = set()
    for page in SCRAPE_PAGES:
        print(f"[scrape] {page}")
        raw = _http_get(page)
        if not raw:
            continue
        html = raw.decode("utf-8", errors="ignore")
        # Titles are in <h2> in grid order; IDs appear as data-audio-player-item-id-value
        titles = [
            re.sub(r"\s+", " ", t).strip()
            for t in re.findall(r"<h2[^>]*>\s*([^<]+?)\s*</h2>", html, flags=re.I)
        ]
        ids = re.findall(
            r'data-audio-player-item-id-value="(\d+)"',
            html,
        )
        if not ids:
            ids = re.findall(r"active_storage/sfx/(\d+)/\1-preview\.mp3", html)
        # zip by order (grid order)
        n = min(len(titles), len(ids))
        for i in range(n):
            aid = int(ids[i])
            if aid in seen:
                continue
            seen.add(aid)
            title = titles[i] or f"sfx-{aid}"
            url = f"https://assets.mixkit.co/active_storage/sfx/{aid}/{aid}-preview.mp3"
            assets.append({"id": aid, "title": title, "url": url})
        # leftover ids without titles
        for sid in ids[n:]:
            aid = int(sid)
            if aid in seen:
                continue
            seen.add(aid)
            url = f"https://assets.mixkit.co/active_storage/sfx/{aid}/{aid}-preview.mp3"
            assets.append({"id": aid, "title": f"sfx-{aid}", "url": url})
        print(f"  found {len(seen)} unique so far (titles={len(titles)} ids={len(ids)})")
    return assets


def assign_assets(assets: list[dict]) -> dict[str, dict]:
    """Pick best asset per token by keyword match on title (unique assets preferred)."""
    chosen: dict[str, dict] = {}
    used_ids: set[int] = set()
    # assign more specific tokens first
    order = sorted(TOKEN_KEYWORDS.keys(), key=lambda t: -len(TOKEN_KEYWORDS[t][0]) if TOKEN_KEYWORDS[t] else 0)
    # priority tokens first
    priority = [
        "laugh", "chuckle", "gasp", "sigh", "cough", "sneeze", "yawn",
        "cry", "sob", "eww", "ugh", "hmm", "wow", "groan", "sniff",
        "grunt", "wince", "scoff", "smirk", "nervous",
    ]
    for token in priority:
        if token not in TOKEN_KEYWORDS:
            continue
        kws = TOKEN_KEYWORDS[token]
        best = None
        best_score = -1
        for a in assets:
            if a["id"] in used_ids:
                continue
            t = a["title"].lower()
            score = 0
            for i, kw in enumerate(kws):
                if kw in t:
                    score += 12 - i
            if "male" in t or "man" in t or "person" in t or "human" in t:
                score += 6
            if "crowd" in t or "audience" in t or "cartoon" in t or "child" in t or "kid" in t or "creature" in t:
                score -= 10
            if "female" in t or "woman" in t:
                score -= 3
            # require at least one keyword hit
            if not any(kw in t for kw in kws):
                score = -1
            if score > best_score:
                best_score = score
                best = a
        if best and best_score > 0:
            chosen[token] = best
            used_ids.add(best["id"])
            print(f"  [{token}] ← {best['title']} (id={best['id']}, score={best_score})")
        else:
            print(f"  [{token}] no good match")
    return chosen


def load_audio_bytes(data: bytes, ext: str = ".mp3") -> tuple[np.ndarray, int] | None:
    """Decode mp3/wav bytes via soundfile if possible, else scipy/temp file."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        # soundfile may not read mp3; try pydub/ffmpeg or librosa
        try:
            a, sr = sf.read(tmp, dtype="float32")
        except Exception:
            try:
                from pydub import AudioSegment
                seg = AudioSegment.from_file(tmp)
                sr = seg.frame_rate
                samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                if seg.channels > 1:
                    samples = samples.reshape(-1, seg.channels).mean(axis=1)
                samples /= 32768.0
                a, sr = samples, sr
            except Exception as e:
                print(f"  decode failed: {e}")
                return None
        if a.ndim > 1:
            a = a.mean(axis=1)
        return np.ascontiguousarray(a, dtype=np.float32), int(sr)
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def process_clip(audio: np.ndarray, sr: int, max_s: float = 2.5) -> tuple[np.ndarray, int]:
    """Trim silence/energy, cap length, normalize."""
    if len(audio) < 10:
        return audio, sr
    # mono already
    win = max(1, int(0.02 * sr))
    n = max(1, len(audio) // win)
    energy = np.array([
        float(np.sqrt(np.mean(audio[i * win:(i + 1) * win] ** 2)) + 1e-12)
        for i in range(n)
    ])
    thr = max(float(energy.max()) * 0.10, 1e-4)
    active = np.where(energy >= thr)[0]
    if len(active):
        a0 = int(active[0] * win)
        a1 = int(min(len(audio), (active[-1] + 1) * win))
        audio = audio[a0:a1]
    max_n = int(max_s * sr)
    if len(audio) > max_n:
        hop = max(1, win)
        best_i, best_e = 0, -1.0
        for i in range(0, len(audio) - max_n + 1, hop):
            e = float(np.mean(audio[i:i + max_n] ** 2))
            if e > best_e:
                best_e, best_i = e, i
        audio = audio[best_i:best_i + max_n]
    fade = min(int(0.03 * sr), max(1, len(audio) // 5))
    if fade > 1 and len(audio) > fade * 2:
        audio = audio.copy()
        audio[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        audio[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    peak = float(np.abs(audio).max()) if len(audio) else 0
    if peak > 1e-6:
        audio = np.clip(audio * (0.85 / peak), -1, 1)
    # resample to 44100 if needed for consistency
    target_sr = 44100
    if sr != target_sr and len(audio) > 0:
        try:
            import scipy.signal as sig
            audio = sig.resample(audio, int(len(audio) * target_sr / sr)).astype(np.float32)
            sr = target_sr
        except Exception:
            pass
    return audio, sr


def estimate_f0(audio: np.ndarray, sr: int) -> float:
    """Very rough pitch estimate via autocorrelation (Hz)."""
    if len(audio) < sr // 10:
        return 0.0
    # use middle 0.5s
    n = min(len(audio), int(0.5 * sr))
    mid = len(audio) // 2
    x = audio[max(0, mid - n // 2): mid + n // 2]
    x = x - x.mean()
    if np.allclose(x, 0):
        return 0.0
    corr = np.correlate(x, x, mode="full")[len(x) - 1:]
    # search 80–350 Hz
    lo, hi = int(sr / 350), int(sr / 80)
    lo, hi = max(1, lo), min(len(corr) - 1, hi)
    if hi <= lo:
        return 0.0
    peak = lo + int(np.argmax(corr[lo:hi]))
    return float(sr / peak) if peak > 0 else 0.0


def pitch_shift_simple(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Crude pitch shift via resample (changes duration — we fix length)."""
    if abs(n_steps) < 0.1 or len(audio) < 64:
        return audio
    factor = 2 ** (n_steps / 12.0)
    new_len = max(1, int(len(audio) / factor))
    try:
        import scipy.signal as sig
        shifted = sig.resample(audio, new_len).astype(np.float32)
        # time-stretch back to original length
        return sig.resample(shifted, len(audio)).astype(np.float32)
    except Exception:
        return audio


def backup_parler_clips():
    SOUND_DIR.mkdir(parents=True, exist_ok=True)
    if BACKUP_DIR.exists():
        print(f"[backup] already exists: {BACKUP_DIR}")
        return
    if any(SOUND_DIR.glob("*.wav")):
        shutil.copytree(SOUND_DIR, BACKUP_DIR)
        print(f"[backup] Parler clips → {BACKUP_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-pitch", type=str, default="", help="Reference WAV (Parler speech) to rough pitch-match")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("BUILD REAL VOCAL LIBRARY (not Parler speech)")
    print("=" * 60)
    print("""
Parler CANNOT make real non-speech vocals (laughs/gasps).
Those clips must be REAL recordings or SFX.

Same-voice options:
  1) BEST: record short reactions yourself → drop into temp/vocal_sounds/
  2) GOOD: free SFX (this script) — clear reactions, different speaker
  3) LATER: RVC voice conversion SFX → Parler speaker
""")

    backup_parler_clips()

    assets = scrape_mixkit_assets()
    if not assets:
        print("[error] Could not scrape Mixkit. Download manually from:")
        print("  https://mixkit.co/free-sound-effects/laugh/")
        print("  https://mixkit.co/free-sound-effects/gasp/")
        print("  Place as temp/vocal_sounds/vocal_laugh.wav etc.")
        # Still disable parler regen
        _write_source_marker("manual")
        return 1

    chosen = assign_assets(assets)
    if not chosen:
        print("[error] No keyword matches — write README for manual fill")
        _write_source_marker("empty")
        return 1

    ref_f0 = 0.0
    if args.match_pitch and os.path.isfile(args.match_pitch):
        ref_a, ref_sr = sf.read(args.match_pitch, dtype="float32")
        if ref_a.ndim > 1:
            ref_a = ref_a.mean(axis=1)
        ref_f0 = estimate_f0(ref_a, ref_sr)
        print(f"[pitch] reference F0 ≈ {ref_f0:.1f} Hz from {args.match_pitch}")

    SOUND_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for token, asset in chosen.items():
        print(f"[dl] [{token}] {asset['title']}")
        if args.dry_run:
            continue
        data = _http_get(asset["url"])
        if not data:
            continue
        decoded = load_audio_bytes(data, ".mp3")
        if not decoded:
            continue
        audio, sr = decoded
        audio, sr = process_clip(audio, sr, max_s=2.4)
        if ref_f0 > 50:
            f0 = estimate_f0(audio, sr)
            if f0 > 50:
                # semitone steps toward reference
                steps = 12 * np.log2(ref_f0 / f0)
                steps = float(np.clip(steps, -5, 5))  # limit
                if abs(steps) > 0.5:
                    print(f"  pitch shift {steps:+.1f} st ({f0:.0f}→{ref_f0:.0f} Hz)")
                    audio = pitch_shift_simple(audio, sr, steps)
        out = SOUND_DIR / f"vocal_{token}.wav"
        sf.write(str(out), audio, sr)
        print(f"  wrote {out.name} ({len(audio)/sr:.2f}s)")
        ok += 1
        # aliases that share file
        aliases = {
            "laugh": ["giggle"],  # chuckle has own if found
            "gasp": ["shocked", "surprise"],
            "cry": ["crying"],
            "sob": [],
        }
        for alt in aliases.get(token, []):
            alt_path = SOUND_DIR / f"vocal_{alt}.wav"
            shutil.copy2(out, alt_path)
            print(f"  alias → {alt_path.name}")

    # Map gasp→shocked file names used by expression_tokens
    for src, dst in [
        ("vocal_gasp.wav", "vocal_gasp.wav"),
        ("vocal_laugh.wav", "vocal_laugh.wav"),
    ]:
        pass

    # shocked uses vocal_gasp in TOKEN_SOUNDS already
    if (SOUND_DIR / "vocal_gasp.wav").exists():
        # also copy as needed — TOKEN uses vocal_gasp for shocked
        pass

    _write_source_marker("mixkit_real_sfx")
    print(f"\n[done] {ok} real clips written to {SOUND_DIR}")
    print("Parler backup at:", BACKUP_DIR)
    print("Re-run orchestrator_agents.py — tokens use these files, not Parler.")
    return 0 if ok else 1


def _write_source_marker(tag: str):
    SOUND_DIR.mkdir(parents=True, exist_ok=True)
    (SOUND_DIR / "SOURCE.txt").write_text(
        f"source={tag}\n"
        "These should be REAL non-speech vocals (laugh/gasp/sigh), NOT Parler TTS.\n"
        "Parler only does speech. Do not re-run ensure_vocal_sounds(generate_missing=True).\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
