"""
expression_tokens.py — timed expression directives embedded in TTS text.

Usage in text:
  "That's really great [laugh] I love it!"
  "Ugh [ugh] this is terrible [wince] why does it keep doing that"
  "Oh [shocked] I had no idea!"

Supported tokens (case-insensitive):
  [eww]       nose sneer + mouth frown (disgust reaction)
  [ugh]       eye squint + brow down + mouth press (frustration)
  [laugh]     wide smile + cheek squint + slight eye squint (amusement burst)
  [sigh]      brow inner up + slight frown (resignation)
  [shocked]   eyes wide + brows up + jaw drop (surprise burst)
  [smirk]     one-sided smile left + brow arch left (sarcasm flash)
  [wince]     eye squint + nose sneer (pain/cringe)
  [hmm]       brow down left + eye squint left (one-sided thinking)
  [smile]     bilateral smile + cheek squint flash
  [frown]     mouth frown + brow inner up (quick sad flash)
  [nod]       head nod trigger (pitch bump — handled via extras)
  [disgust]   alias for [eww]
  [surprise]  alias for [shocked]
  [ooh]       raised brows + wide eyes + slight jaw (impressed/surprised-soft)
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Token recipes — blendshape values at peak of expression burst
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Vocal sound files for each token (relative to temp/vocal_sounds/)
# ─────────────────────────────────────────────────────────────────────────────
TOKEN_SOUNDS: Dict[str, str] = {
    "eww":      "vocal_eww.wav",
    "disgust":  "vocal_ew.wav",
    "ugh":      "vocal_ugh.wav",
    "laugh":    "vocal_laugh.wav",
    "chuckle":  "vocal_chuckle.wav",
    "giggle":   "vocal_chuckle.wav",
    "sigh":     "vocal_sigh.wav",
    "shocked":  "vocal_gasp.wav",
    "surprise": "vocal_gasp.wav",
    "gasp":     "vocal_gasp.wav",
    "wince":    "vocal_wince.wav",
    "cringe":   "vocal_grunt.wav",
    "hmm":      "vocal_hmm.wav",
    "smirk":    "vocal_scoff.wav",
    "scoff":    "vocal_scoff.wav",
    "ooh":      "vocal_wow.wav",
    "wow":      "vocal_wow.wav",
    "frown":    "vocal_groan.wav",
    "groan":    "vocal_groan.wav",
    "cough":    "vocal_cough.wav",
    "yawn":     "vocal_yawn.wav",
    "cry":      "vocal_cry.wav",
    "sob":      "vocal_sob.wav",
    "sniff":    "vocal_sniff.wav",
    "sneeze":   "vocal_sneeze.wav",
    "nervous":  "vocal_nervous.wav",
    "grunt":    "vocal_grunt.wav",
}


TOKEN_RECIPES: Dict[str, Dict[str, float]] = {
    "eww": {
        "noseSneerLeft": 0.85, "noseSneerRight": 0.80,
        "mouthFrownLeft": 0.55, "mouthFrownRight": 0.55,
        "browDownLeft": 0.45, "browDownRight": 0.45,
        "eyeSquintLeft": 0.35, "eyeSquintRight": 0.35,
    },
    "ugh": {
        "browDownLeft": 0.65, "browDownRight": 0.60,
        "eyeSquintLeft": 0.55, "eyeSquintRight": 0.55,
        "mouthPressLeft": 0.50, "mouthPressRight": 0.50,
        "mouthFrownLeft": 0.30, "mouthFrownRight": 0.30,
        "noseSneerLeft": 0.30,
    },
    "laugh": {
        "mouthSmileLeft": 0.80, "mouthSmileRight": 0.80,
        "cheekSquintLeft": 0.70, "cheekSquintRight": 0.70,
        "eyeSquintLeft": 0.55, "eyeSquintRight": 0.55,
        "browOuterUpLeft": 0.25, "browOuterUpRight": 0.25,
    },
    "sigh": {
        "browInnerUp": 0.55,
        "mouthFrownLeft": 0.35, "mouthFrownRight": 0.35,
        "eyeSquintLeft": 0.20, "eyeSquintRight": 0.20,
    },
    "shocked": {
        "eyeWideLeft": 0.95, "eyeWideRight": 0.95,
        "browInnerUp": 0.80,
        "browOuterUpLeft": 0.65, "browOuterUpRight": 0.65,
        "jawOpen": 0.45,
    },
    "smirk": {
        "mouthSmileLeft": 0.65,
        "browOuterUpLeft": 0.50,
        "eyeSquintRight": 0.40,
        "mouthFrownRight": 0.20,
    },
    "wince": {
        "eyeSquintLeft": 0.75, "eyeSquintRight": 0.75,
        "noseSneerLeft": 0.55, "noseSneerRight": 0.50,
        "browDownLeft": 0.35, "browDownRight": 0.35,
        "mouthPressLeft": 0.30, "mouthPressRight": 0.30,
    },
    "hmm": {
        "browDownLeft": 0.55,
        "browInnerUp": 0.30,
        "eyeSquintLeft": 0.45,
        "mouthPressLeft": 0.25, "mouthPressRight": 0.15,
    },
    "smile": {
        "mouthSmileLeft": 0.70, "mouthSmileRight": 0.70,
        "cheekSquintLeft": 0.55, "cheekSquintRight": 0.55,
    },
    "frown": {
        "mouthFrownLeft": 0.60, "mouthFrownRight": 0.60,
        "browInnerUp": 0.45,
    },
    "ooh": {
        "browOuterUpLeft": 0.60, "browOuterUpRight": 0.60,
        "browInnerUp": 0.40,
        "eyeWideLeft": 0.55, "eyeWideRight": 0.55,
        "jawOpen": 0.20,
    },
    "nod": {},  # handled by extras (head pitch bump)
}

# Aliases → canonical recipe + sound
TOKEN_RECIPES["disgust"]  = TOKEN_RECIPES["eww"]
TOKEN_RECIPES["surprise"] = TOKEN_RECIPES["shocked"]
TOKEN_RECIPES["cringe"]   = TOKEN_RECIPES["wince"]
TOKEN_RECIPES["chuckle"]  = TOKEN_RECIPES["laugh"]
TOKEN_RECIPES["giggle"]   = TOKEN_RECIPES["laugh"]
TOKEN_RECIPES["gasp"]     = TOKEN_RECIPES["shocked"]
TOKEN_RECIPES["scoff"]    = TOKEN_RECIPES["smirk"]
TOKEN_RECIPES["wow"]      = TOKEN_RECIPES["ooh"]
TOKEN_RECIPES["groan"]    = TOKEN_RECIPES["frown"]
TOKEN_RECIPES["cough"]    = {
    "eyeSquintLeft": 0.4, "eyeSquintRight": 0.4,
    "browDownLeft": 0.3, "browDownRight": 0.3,
    "jawOpen": 0.25, "mouthFunnel": 0.3,
}
TOKEN_RECIPES["yawn"] = {
    "jawOpen": 0.75, "eyeSquintLeft": 0.5, "eyeSquintRight": 0.5,
    "browInnerUp": 0.25, "mouthFunnel": 0.35,
}
TOKEN_RECIPES["cry"] = {
    "browInnerUp": 0.7, "mouthFrownLeft": 0.55, "mouthFrownRight": 0.55,
    "eyeSquintLeft": 0.4, "eyeSquintRight": 0.4,
}
TOKEN_RECIPES["sob"] = TOKEN_RECIPES["cry"]
TOKEN_RECIPES["sniff"] = {
    "noseSneerLeft": 0.4, "noseSneerRight": 0.4,
    "browInnerUp": 0.25, "eyeSquintLeft": 0.2, "eyeSquintRight": 0.2,
}
TOKEN_RECIPES["sneeze"] = {
    "eyeSquintLeft": 0.9, "eyeSquintRight": 0.9,
    "browDownLeft": 0.5, "browDownRight": 0.5,
    "jawOpen": 0.35, "noseSneerLeft": 0.5, "noseSneerRight": 0.5,
}
TOKEN_RECIPES["nervous"] = {
    "mouthSmileLeft": 0.25, "mouthSmileRight": 0.2,
    "eyeSquintLeft": 0.25, "browInnerUp": 0.35,
}
TOKEN_RECIPES["grunt"] = TOKEN_RECIPES["ugh"]
TOKEN_RECIPES["eyeroll"] = TOKEN_RECIPES["smirk"]
# Body-only gesture tokens (face recipe optional / light)
TOKEN_RECIPES["shrug"] = {
    "browInnerUp": 0.25,
    "mouthSmileLeft": 0.15, "mouthSmileRight": 0.15,
}
TOKEN_RECIPES["wave"] = {
    "mouthSmileLeft": 0.45, "mouthSmileRight": 0.45,
    "browOuterUpLeft": 0.2, "browOuterUpRight": 0.2,
}
TOKEN_RECIPES["point"] = {
    "browOuterUpLeft": 0.2, "browOuterUpRight": 0.15,
    "eyeSquintLeft": 0.1,
}
TOKEN_RECIPES["wink"] = {
    "eyeBlinkLeft": 1.0, "mouthSmileLeft": 0.45, "browOuterUpLeft": 0.3,
}

# Duration of each token burst (seconds): attack, hold, release
# hold stretches to match clip length at runtime when clip is longer
TOKEN_DURATION: Dict[str, tuple] = {
    "shocked":  (0.08, 0.40, 0.30),
    "laugh":    (0.10, 0.90, 0.35),
    "chuckle":  (0.10, 0.70, 0.30),
    "eww":      (0.08, 0.45, 0.30),
    "ugh":      (0.10, 0.50, 0.30),
    "sigh":     (0.15, 0.90, 0.40),
    "smirk":    (0.12, 0.45, 0.35),
    "wince":    (0.06, 0.35, 0.25),
    "hmm":      (0.12, 0.70, 0.35),
    "smile":    (0.10, 0.35, 0.30),
    "frown":    (0.10, 0.45, 0.30),
    "ooh":      (0.08, 0.40, 0.30),
    "nod":      (0.05, 0.15, 0.10),
    "cough":    (0.05, 0.40, 0.25),
    "yawn":     (0.15, 0.90, 0.40),
    "cry":      (0.12, 0.90, 0.40),
    "sob":      (0.12, 0.80, 0.40),
    "sniff":    (0.08, 0.40, 0.25),
    "sneeze":   (0.05, 0.35, 0.30),
    "nervous":  (0.10, 0.50, 0.30),
    "grunt":    (0.08, 0.40, 0.25),
    "gasp":     (0.06, 0.40, 0.25),
    "wow":      (0.08, 0.40, 0.30),
    "giggle":   (0.10, 0.70, 0.30),
    "scoff":    (0.08, 0.40, 0.25),
    "groan":    (0.12, 0.70, 0.35),
    "eyeroll":  (0.10, 0.40, 0.25),
    "wink":     (0.05, 0.20, 0.15),
}
_DEFAULT_DUR = (0.10, 0.45, 0.30)

# Max seconds of a reaction clip to play (find loudest window inside long files)
TOKEN_MAX_CLIP_S: Dict[str, float] = {
    "laugh": 2.4, "chuckle": 1.6, "giggle": 1.6,
    "sigh": 2.2, "hmm": 2.0, "yawn": 2.0, "cry": 2.2, "sob": 2.0,
    "wince": 1.4, "smirk": 1.2, "scoff": 1.2, "groan": 1.8,
    "sneeze": 1.5, "eww": 1.2, "ugh": 1.3, "shocked": 1.1, "gasp": 1.1,
}
_DEFAULT_MAX_CLIP = 1.8


class ExpressionEvent(NamedTuple):
    time: float           # when to start (seconds into sentence audio)
    token: str            # canonical token name
    recipe: Dict[str, float]
    attack: float         # ramp-up duration
    hold: float           # hold duration
    release: float        # ramp-down duration


# ─────────────────────────────────────────────────────────────────────────────
# Text parsing
# ─────────────────────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r'\[([a-z_]+)\]', re.IGNORECASE)


def strip_tokens(text: str) -> str:
    """Remove all [token] markers from text before sending to TTS."""
    return _TOKEN_RE.sub('', text).strip()


class TextSegment(NamedTuple):
    """One piece of a sentence: spoken words OR a reaction token."""
    kind: str   # "speech" | "token"
    value: str  # clean speech text, or canonical token name (e.g. "eww")


# Optional sentence-level emotion override while a token reaction plays
TOKEN_EMOTIONS: Dict[str, str] = {
    "eww": "disgusted", "disgust": "disgusted",
    "ugh": "angry", "grunt": "angry",
    "laugh": "happy", "chuckle": "happy", "giggle": "happy", "smile": "happy",
    "sigh": "sad", "frown": "sad", "cry": "sad", "sob": "sad",
    "shocked": "surprised", "surprise": "surprised", "gasp": "surprised",
    "ooh": "surprised", "wow": "surprised",
    "wince": "disgusted", "cringe": "disgusted",
    "hmm": "thinking", "nervous": "fearful",
    "smirk": "sarcastic", "scoff": "sarcastic", "eyeroll": "sarcastic",
    "nod": "neutral", "cough": "neutral", "yawn": "calm",
    "sniff": "sad", "sneeze": "surprised", "wink": "happy", "groan": "sad",
}


def split_token_segments(text: str) -> List[TextSegment]:
    """
    Split LLM/sentence text into ordered speech + token segments.

    Example:
      "Oh no [eww] that is bad [laugh] right?"
      → speech "Oh no" | token eww | speech "that is bad" | token laugh | speech "right?"

    Speech segments are what Parler TTS should generate.
    Token segments play pre-recorded vocal_sounds + expression recipes (never TTS).
    """
    if not text or not text.strip():
        return []

    segs: List[TextSegment] = []
    last = 0
    for match in _TOKEN_RE.finditer(text):
        before = text[last:match.start()]
        speech = re.sub(r"\s+", " ", before).strip(" \t\n\r,.;:")
        # keep trailing punctuation attached to speech when possible
        speech = re.sub(r"\s+", " ", before).strip()
        if speech:
            segs.append(TextSegment("speech", speech))

        raw = match.group(1).lower()
        canonical = _resolve_alias(raw)
        # NEVER turn tokens into Parler speech (that produces waste "laugh" readings)
        if canonical in TOKEN_RECIPES or canonical in TOKEN_SOUNDS:
            segs.append(TextSegment("token", canonical if canonical in TOKEN_RECIPES else raw))
        else:
            # Unknown marker: skip audio speech, still try as token face-only
            segs.append(TextSegment("token", canonical))

        last = match.end()

    after = re.sub(r"\s+", " ", text[last:]).strip()
    if after:
        segs.append(TextSegment("speech", after))

    # If no tokens found, single speech segment (strip any leftover)
    if not segs:
        clean = strip_tokens(text)
        if clean:
            segs.append(TextSegment("speech", clean))

    # Drop speech that is only punctuation/whitespace (e.g. trailing "!")
    cleaned_segs: List[TextSegment] = []
    for s in segs:
        if s.kind == "speech":
            if not re.sub(r"[\s\W_]+", "", s.value):
                continue
            # merge leading punctuation onto previous speech if any
            if cleaned_segs and cleaned_segs[-1].kind == "speech" and re.match(r"^[\W\s]+", s.value):
                prev = cleaned_segs[-1]
                cleaned_segs[-1] = TextSegment("speech", (prev.value + " " + s.value).strip())
                continue
        cleaned_segs.append(s)
    return cleaned_segs


def resolve_sound_path(token: str, sounds_dir: Optional[str] = None) -> Optional[str]:
    """Absolute path to pre-recorded clip for token, or None."""
    import os
    if sounds_dir is None:
        sounds_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "temp", "vocal_sounds"
        )
    name = TOKEN_SOUNDS.get(token)
    if not name:
        return None
    path = os.path.join(sounds_dir, name)
    return path if os.path.isfile(path) else None


def parse_tokens(text: str, audio_duration: float) -> List[ExpressionEvent]:
    """
    Parse [token] markers from text and assign timing based on character position.

    Character position → fractional time (linear approximation).
    Returns sorted list of ExpressionEvent.
    """
    if not text or audio_duration <= 0:
        return []

    total_chars = len(text)
    events: List[ExpressionEvent] = []
    char_offset = 0

    # Build a clean version to measure positions
    clean_chars = 0
    for match in _TOKEN_RE.finditer(text):
        token_name = match.group(1).lower()
        canonical = _resolve_alias(token_name)
        if canonical not in TOKEN_RECIPES:
            continue

        # Position: count non-token characters up to this match
        chars_before = len(_TOKEN_RE.sub('', text[:match.start()]))
        total_clean = len(_TOKEN_RE.sub('', text))
        frac = chars_before / max(1, total_clean)

        # Clamp to [5%, 92%] of duration so bursts don't clip
        t = max(0.05 * audio_duration, min(0.92 * audio_duration, frac * audio_duration))

        recipe = dict(TOKEN_RECIPES[canonical])
        atk, hld, rel = TOKEN_DURATION.get(canonical, _DEFAULT_DUR)

        events.append(ExpressionEvent(
            time=t,
            token=canonical,
            recipe=recipe,
            attack=atk,
            hold=hld,
            release=rel,
        ))

    events.sort(key=lambda e: e.time)
    return events


def _resolve_alias(name: str) -> str:
    aliases = {
        "disgust": "eww",
        "surprise": "shocked",
        "cringe": "wince",
        "gasp": "shocked",
        "chuckle": "chuckle",  # own sound; recipe copies laugh
        "giggle": "giggle",
        "lol": "laugh",
        "haha": "laugh",
        "ha": "laugh",
        "eyeroll": "eyeroll",
        "eye_roll": "eyeroll",
        "wink": "wink",
        "wow": "wow",
        "scoff": "scoff",
        "groan": "groan",
    }
    return aliases.get(name, name)


def prepare_token_clip(
    token: str,
    sounds_dir: Optional[str] = None,
    out_path: Optional[str] = None,
) -> Optional[tuple]:
    """
    Load pre-recorded reaction clip (never Parler).
    Returns (float32 mono audio, sample_rate, path_written) or None.

    Long raw recordings are trimmed to the loudest active window so lips
    and expression stay synced to the real vocal burst.
    """
    import os
    import numpy as np

    try:
        import soundfile as sf
    except ImportError:
        return None

    path = resolve_sound_path(token, sounds_dir)
    if not path:
        # try alias sound
        alt = _resolve_alias(token)
        path = resolve_sound_path(alt, sounds_dir)
    if not path:
        return None

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if len(audio) < int(0.05 * sr):
        return None

    max_s = float(TOKEN_MAX_CLIP_S.get(token, _DEFAULT_MAX_CLIP))
    max_n = int(max_s * sr)

    # Energy-based trim: keep loudest window (real laugh/gasp, not silence lead-in)
    win = max(1, int(0.02 * sr))
    n_frames = max(1, len(audio) // win)
    energy = np.array(
        [float(np.sqrt(np.mean(audio[i * win:(i + 1) * win] ** 2)) + 1e-12)
         for i in range(n_frames)],
        dtype=np.float32,
    )
    thr = max(float(energy.max()) * 0.12, 1e-4)
    active = np.where(energy >= thr)[0]
    if len(active) > 0:
        a0 = int(active[0] * win)
        a1 = int(min(len(audio), (active[-1] + 1) * win))
        audio = audio[a0:a1]

    if len(audio) > max_n:
        # Slide max_n window to peak energy
        hop = max(1, win)
        best_i, best_e = 0, -1.0
        for i in range(0, len(audio) - max_n + 1, hop):
            e = float(np.mean(audio[i:i + max_n] ** 2))
            if e > best_e:
                best_e, best_i = e, i
        audio = audio[best_i:best_i + max_n].copy()
        fade = min(int(0.04 * sr), len(audio) // 4)
        if fade > 1:
            audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    peak = float(np.abs(audio).max())
    if peak > 1e-6:
        audio = np.clip(audio * (0.88 / peak), -1.0, 1.0)

    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "temp",
            f"token_{token}_ready.wav",
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, audio, int(sr))
    return audio, int(sr), out_path


# ─────────────────────────────────────────────────────────────────────────────
# Runtime: evaluate all active token bursts at time t
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_tokens(
    events: List[ExpressionEvent],
    t: float,
) -> Dict[str, float]:
    """
    Given the list of events and current time t, compute the combined
    blendshape contribution from all active token bursts.
    Uses max-merge across simultaneous events.
    """
    out: Dict[str, float] = {}
    for ev in events:
        start = ev.time
        end   = start + ev.attack + ev.hold + ev.release
        if t < start or t > end:
            continue

        dt = t - start
        # Ramp envelope
        if dt < ev.attack:
            weight = dt / max(ev.attack, 1e-4)
        elif dt < ev.attack + ev.hold:
            weight = 1.0
        else:
            remaining = end - t
            weight = remaining / max(ev.release, 1e-4)

        weight = max(0.0, min(1.0, weight))
        weight = weight * weight * (3.0 - 2.0 * weight)  # smoothstep

        for k, v in ev.recipe.items():
            val = float(v) * weight
            out[k] = min(1.0, max(out.get(k, 0.0), val))

    return out


def has_nod(events: List[ExpressionEvent], t: float, window: float = 0.05) -> bool:
    """True if a [nod] token is active near time t."""
    return any(
        ev.token == "nod" and abs(t - ev.time) < window
        for ev in events
    )


def mix_token_sounds(
    audio_path: str,
    events: List[ExpressionEvent],
    sounds_dir: Optional[str] = None,
    mix_gain: float = 0.70,
    out_path: Optional[str] = None,
) -> str:
    """
    Mix vocal reaction clips into the main audio at each token's time position.
    Wav2arkit is then run on this mixed audio so lip sync covers both speech
    AND the reaction vocalization.

    Returns the path to the mixed audio file (or original if nothing to mix).
    """
    import os
    import numpy as np

    try:
        import soundfile as sf
    except ImportError:
        return audio_path

    if sounds_dir is None:
        sounds_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "temp", "vocal_sounds"
        )

    # Filter events that have a sound file
    to_mix = [
        ev for ev in events
        if ev.token in TOKEN_SOUNDS and
           os.path.exists(os.path.join(sounds_dir, TOKEN_SOUNDS[ev.token]))
    ]
    if not to_mix:
        return audio_path

    # Load main audio
    main_audio, sr = sf.read(audio_path, dtype="float32")
    if main_audio.ndim > 1:
        main_audio = main_audio.mean(axis=1)
    mixed = main_audio.copy()

    for ev in to_mix:
        snd_path = os.path.join(sounds_dir, TOKEN_SOUNDS[ev.token])
        snd, snd_sr = sf.read(snd_path, dtype="float32")
        if snd.ndim > 1:
            snd = snd.mean(axis=1)

        # Resample if needed
        if snd_sr != sr:
            try:
                import scipy.signal as sig
                snd = sig.resample(snd, int(len(snd) * sr / snd_sr))
            except Exception:
                continue

        # Trim to expression burst duration so long clips don't bleed into next speech.
        # Max cap = attack + hold + release for this token, hard-capped at 1.5s.
        atk, hld, rel = TOKEN_DURATION.get(ev.token, _DEFAULT_DUR)
        max_clip_s = min(atk + hld + rel, 1.50)
        max_samples = int(max_clip_s * sr)
        if len(snd) > max_samples:
            snd = snd[:max_samples]
            # 30ms fade-out to avoid click at trim point
            fade_n = min(int(0.030 * sr), len(snd))
            fade = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
            snd[-fade_n:] *= fade

        # Normalise reaction sound to mix_gain
        peak = float(np.abs(snd).max())
        if peak > 1e-6:
            snd = snd * (mix_gain / peak)

        # Insert at token time position — only within main audio boundaries
        start_sample = int(ev.time * sr)
        end_sample = min(len(mixed), start_sample + len(snd))
        n = end_sample - start_sample
        if n > 0:
            mixed[start_sample:end_sample] = np.clip(
                mixed[start_sample:end_sample] + snd[:n], -1.0, 1.0
            )

    # Save mixed audio
    if out_path is None:
        base = os.path.splitext(audio_path)[0]
        out_path = base + "_mixed.wav"

    sf.write(out_path, mixed, sr)
    return out_path
