"""
Record a real vocal reaction into temp/vocal_sounds/ (same mic → closer to agent).

Usage:
  python tools/record_vocal_token.py laugh
  python tools/record_vocal_token.py eww --seconds 2.5

Requires: sounddevice, soundfile
  pip install sounddevice soundfile

Play a Parler line first to match pitch/energy, then perform the reaction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "temp" / "vocal_sounds"

# token → filename (matches expression_tokens.TOKEN_SOUNDS)
TOKEN_FILES = {
    "laugh": "vocal_laugh.wav",
    "chuckle": "vocal_chuckle.wav",
    "giggle": "vocal_chuckle.wav",
    "eww": "vocal_eww.wav",
    "ew": "vocal_ew.wav",
    "ugh": "vocal_ugh.wav",
    "sigh": "vocal_sigh.wav",
    "gasp": "vocal_gasp.wav",
    "shocked": "vocal_gasp.wav",
    "hmm": "vocal_hmm.wav",
    "wow": "vocal_wow.wav",
    "ooh": "vocal_wow.wav",
    "smirk": "vocal_scoff.wav",
    "scoff": "vocal_scoff.wav",
    "wince": "vocal_wince.wav",
    "groan": "vocal_groan.wav",
    "cough": "vocal_cough.wav",
    "yawn": "vocal_yawn.wav",
    "cry": "vocal_cry.wav",
    "sob": "vocal_sob.wav",
    "sniff": "vocal_sniff.wav",
    "sneeze": "vocal_sneeze.wav",
    "nervous": "vocal_nervous.wav",
    "grunt": "vocal_grunt.wav",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Record real vocal for a token")
    ap.add_argument("token", help="e.g. laugh, eww, gasp, sigh, hmm")
    ap.add_argument("--seconds", type=float, default=2.0, help="record length")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--device", type=int, default=None, help="sounddevice input device index")
    args = ap.parse_args()

    token = args.token.lower().strip()
    fname = TOKEN_FILES.get(token)
    if not fname:
        print(f"Unknown token {token!r}. Known: {', '.join(sorted(TOKEN_FILES))}")
        return 1

    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print("pip install sounddevice soundfile")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / fname

    print("=" * 50)
    print(f"Recording [{token}] → {out}")
    print(f"Duration: {args.seconds:.1f}s  sample_rate={args.sr}")
    print("Tip: match energy/pitch of your Parler dialogue voice.")
    print("Press ENTER then perform the reaction...")
    input()

    print("REC...")
    audio = sd.rec(
        int(args.seconds * args.sr),
        samplerate=args.sr,
        channels=1,
        dtype="float32",
        device=args.device,
    )
    sd.wait()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    # trim silence edges
    thr = max(float(np.abs(audio).max()) * 0.05, 1e-4)
    idx = np.where(np.abs(audio) > thr)[0]
    if len(idx) > 10:
        pad = int(0.03 * args.sr)
        a0 = max(0, int(idx[0]) - pad)
        a1 = min(len(audio), int(idx[-1]) + pad)
        audio = audio[a0:a1]

    peak = float(np.abs(audio).max()) if len(audio) else 0
    if peak > 1e-6:
        audio = np.clip(audio * (0.85 / peak), -1, 1)

    # backup existing
    if out.exists():
        bak = out.with_suffix(".bak.wav")
        out.replace(bak)
        print(f"Backed up old file → {bak.name}")

    sf.write(str(out), audio, args.sr)
    print(f"Saved {out} ({len(audio)/args.sr:.2f}s)")
    print("Pipeline will use this file on next [token] — no Parler.")
    # write marker
    src = OUT_DIR / "SOURCE.txt"
    src.write_text(
        "source=user_recorded\n"
        "Real mic recordings for reaction tokens. Not Parler TTS.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
