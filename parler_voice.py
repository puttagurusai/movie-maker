"""
parler_voice.py

Parler-TTS Mini v1 helper for TalkFace.
Emotion-aware speech from text + style descriptions.

Model downloads into: models/parler-tts-mini-v1/ (~880 MB–1.2 GB, project folder)

Named parler_voice.py so it does NOT shadow the installed package `parler_tts`.
Import this module as:
    from parler_voice import generate_speech, build_voice_style, load_parler
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

# transformers / parler_tts are imported inside load_parler() so starting
# the orchestrator does not spend a minute importing TTS weights.

# HuggingFace repo id (used only when downloading into the project models folder)
HF_REPO_ID = "parler-tts/parler-tts-mini-v1"

# Local project folder — all Parler weights live HERE, not in ~/.cache/huggingface
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
LOCAL_MODEL_DIR = MODELS_DIR / "parler-tts-mini-v1"

QUALITY_SUFFIX = "The audio is high quality with  no background noise."


def _safe_write_wav(output_path: str, audio: np.ndarray, sample_rate: int) -> None:
    """
    Write mono float32 audio as PCM16 WAV.
    Handles locked files, bad paths, and libsndfile 'System error' by retry/fallback.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    path = os.path.abspath(output_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Prefer unique path if target is locked
    candidates = [path]
    stem, ext = os.path.splitext(path)
    for i in range(1, 4):
        candidates.append(f"{stem}_{i}{ext or '.wav'}")

    last_err: Exception | None = None
    for cand in candidates:
        try:
            # Remove stale/partial file first
            if os.path.isfile(cand):
                try:
                    os.remove(cand)
                except OSError:
                    pass
            sf.write(cand, audio, int(sample_rate), subtype="PCM_16")
            if cand != path:
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                    os.replace(cand, path)
                except OSError:
                    # leave on alternate name; caller may still use returned audio buffer
                    print(f"[parler_voice] WARN wrote alternate path {cand}")
            return
        except Exception as e:
            last_err = e
            print(f"[parler_voice] sf.write failed ({cand}): {e}")

    # Fallback: scipy / wave module
    try:
        from scipy.io import wavfile

        pcm = (audio * 32767.0).astype(np.int16)
        wavfile.write(path, int(sample_rate), pcm)
        print(f"[parler_voice] saved via scipy.io.wavfile → {path}")
        return
    except Exception as e2:
        last_err = e2

    try:
        import wave
        import struct

        pcm = (audio * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm.tobytes())
        print(f"[parler_voice] saved via wave module → {path}")
        return
    except Exception as e3:
        raise RuntimeError(
            f"Failed to write WAV {path}: {last_err}; wave fallback: {e3}"
        ) from e3

# Fixed speaker identity prepended to every style description.
# "Thomas" is a stable male voice in Parler-TTS Mini v1 training data.
# This ensures the same actor / timbre across all emotions and intensity bands.
SPEAKER_PREFIX = "Thomas speaks with"

# Module-level cache (in-memory after load — not the disk "cache" folder)
_model = None
_tokenizer = None
_device: Optional[str] = None
_warmed_up = False
_filler_cache: dict[str, Tuple[str, int]] = {}  # name -> (wav_path, sample_rate)


def _model_is_complete(model_dir: Path) -> bool:
    """True if local folder has the main weight file and config (not a partial download)."""
    if not model_dir.is_dir():
        return False
    config_ok = (model_dir / "config.json").is_file()
    weights_ok = (
        (model_dir / "model.safetensors").is_file()
        or (model_dir / "pytorch_model.bin").is_file()
        or any(model_dir.glob("model*.safetensors"))
    )
    # Incomplete HF temp files mean download did not finish
    incomplete = list(model_dir.rglob("*.incomplete"))
    return config_ok and weights_ok and not incomplete


def ensure_local_model() -> str:
    """
    Make sure models/parler-tts-mini-v1 exists with full weights.
    Downloads from HuggingFace into the project folder (not user cache).
    Returns local path string for from_pretrained().
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if _model_is_complete(LOCAL_MODEL_DIR):
        print(f"[parler_voice] Using local model: {LOCAL_MODEL_DIR}")
        return str(LOCAL_MODEL_DIR)

    print(f"[parler_voice] Model not complete at: {LOCAL_MODEL_DIR}")
    print(f"[parler_voice] Downloading {HF_REPO_ID} into project models/ folder...")
    print("[parler_voice] Size ~880MB–1.2GB. This is a one-time download.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download the model.\n"
            "  pip install huggingface_hub"
        ) from e

    # local_dir = project folder with a clear name (no user-profile cache)
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(LOCAL_MODEL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    if not _model_is_complete(LOCAL_MODEL_DIR):
        raise RuntimeError(
            f"Download finished but model looks incomplete under {LOCAL_MODEL_DIR}.\n"
            "Delete that folder and try again when your network is stable."
        )

    print(f"[parler_voice] Download complete → {LOCAL_MODEL_DIR}")
    return str(LOCAL_MODEL_DIR)


# ---------------------------------------------------------------------------
# Voice style descriptions (emotion × intensity band) — exact project mapping
# ---------------------------------------------------------------------------
VOICE_STYLES = {
    "neutral": {
        "low": (
            "Calm and clear delivery, moderate pace, "
            "natural conversational tone, no particular emphasis"
        ),
        "medium": (
            "Clear and engaging delivery, steady pace, "
            "warm conversational tone"
        ),
        "high": (
            "Very clear and composed delivery, measured pace, "
            "professional tone"
        ),
    },
    "happy": {
        "low": (
            "Warm and friendly tone, slight smile in the voice, "
            "gentle upward inflection"
        ),
        "medium": (
            "Cheerful and bright delivery, energetic pace, "
            "upbeat natural tone"
        ),
        "high": (
            "Very excited and joyful delivery, fast energetic pace, "
            "bright enthusiastic tone, rising inflection on key words"
        ),
    },
    "sad": {
        "low": (
            "Slightly subdued tone, slower pace, "
            "gentle falling intonation"
        ),
        "medium": (
            "Heavy and slow delivery, low energy, "
            "falling intonation, quiet and somber"
        ),
        "high": (
            "Very slow and heavy delivery, low pitch, long pauses, "
            "deeply somber and tired tone"
        ),
    },
    "angry": {
        "low": (
            "Firm and direct delivery, clipped words, "
            "slightly tense tone"
        ),
        "medium": (
            "Sharp and forceful delivery, fast clipped pace, "
            "hard emphasis on key words"
        ),
        "high": (
            "Very intense and forceful delivery, aggressive clipped speech, "
            "strong emphasis, tight jaw quality in voice"
        ),
    },
    "surprised": {
        "low": (
            "Slightly raised pitch, mild upward inflection, "
            "gentle breathiness"
        ),
        "medium": (
            "Noticeably raised pitch, faster pace, "
            "breathless quality, rising intonation"
        ),
        "high": (
            "Very fast and breathless delivery, high pitch, "
            "strong rising intonation, genuine astonishment in voice"
        ),
    },
    "fearful": {
        "low": (
            "Slightly tense delivery, careful pacing, "
            "quiet and cautious tone"
        ),
        "medium": (
            "Tense and quiet delivery, uneven pace, "
            "slightly shaky quality"
        ),
        "high": (
            "Very quiet and tense delivery, fast uncertain pace, "
            "shaky breathless quality, whisper-like intensity"
        ),
    },
    "disgusted": {
        "low": (
            "Flat and dry delivery, slightly slow pace, "
            "dismissive tone"
        ),
        "medium": (
            "Heavy and contemptuous tone, slow deliberate pace, "
            "strong distaste in delivery"
        ),
        "high": (
            "Very heavy and contemptuous tone, slow and deliberate, "
            "strong revulsion quality, drawn out vowels"
        ),
    },
    "sarcastic": {
        "low": (
            "Slightly dry delivery, mild exaggerated inflection, "
            "understated ironic tone"
        ),
        "medium": (
            "Clearly ironic delivery, exaggerated stress on key words, "
            "dry wit in tone"
        ),
        "high": (
            "Very exaggerated ironic delivery, strong emphasis on sarcastic words, "
            "drawn out stressed syllables, obvious deadpan quality"
        ),
    },
    "thinking": {
        "low": (
            "Slightly slower thoughtful pace, gentle hesitations, "
            "contemplative tone"
        ),
        "medium": (
            "Slow and deliberate delivery, noticeable pauses between thoughts, "
            "searching quality in voice"
        ),
        "high": (
            "Very slow and careful delivery, long thoughtful pauses, "
            "quiet and introspective tone, trailing off at end of sentences"
        ),
    },
}


def _intensity_band(intensity: float) -> str:
    """
    Intensity mapping:
      0.0 - 0.33  → low
      0.34 - 0.66 → medium
      0.67 - 1.0  → high
    """
    try:
        i = float(intensity)
    except (TypeError, ValueError):
        i = 0.5
    i = max(0.0, min(1.0, i))
    if i <= 0.33:
        return "low"
    if i <= 0.66:
        return "medium"
    return "high"


def build_voice_style(emotion: str, intensity: float) -> str:
    """
    Build a Parler voice description from emotion + intensity.
    SPEAKER_PREFIX ensures the same male voice (Thomas) on every sentence.
    Always appends the high-quality audio suffix.
    """
    emotion_key = (emotion or "neutral").lower().strip()
    if emotion_key not in VOICE_STYLES:
        emotion_key = "neutral"

    band = _intensity_band(intensity)
    style_core = VOICE_STYLES[emotion_key][band]
    return f"{SPEAKER_PREFIX} {style_core.lower()}. {QUALITY_SUFFIX}"


def load_parler():
    """
    Load Parler-TTS Mini v1 once (GPU preferred).
    Optimizations: float16 on CUDA, optional torch.compile, warmup generate.
    Returns (model, tokenizer, device).
    """
    global _model, _tokenizer, _device, _warmed_up

    if _model is not None and _tokenizer is not None and _device is not None:
        return _model, _tokenizer, _device

    try:
        from transformers import AutoTokenizer
        from parler_tts import ParlerTTSForConditionalGeneration
    except ImportError as e:
        raise ImportError(
            "parler-tts is not installed.\n"
            "Install with:\n"
            "  pip install git+https://github.com/huggingface/parler-tts.git\n"
            "  pip install transformers accelerate torch soundfile sounddevice numpy"
        ) from e

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    local_path = ensure_local_model()
    print(f"[parler_voice] Loading from: {local_path}")
    print(f"[parler_voice] Device: {_device}")

    _model = ParlerTTSForConditionalGeneration.from_pretrained(local_path)
    _tokenizer = AutoTokenizer.from_pretrained(local_path)
    _model.to(_device)
    _model.eval()

    # float16 on CUDA — less VRAM, faster inference
    if _device == "cuda":
        try:
            _model = _model.half()
            print("[parler_voice] model.half() applied (float16)")
        except Exception as e:
            print(f"[parler_voice] model.half() skipped: {e}")

    # torch.compile is OFF by default: on Windows + short utterances it often
    # makes TTS *slower* (10s+). Set PARLER_TORCH_COMPILE=1 to enable.
    import os as _os
    if _os.environ.get("PARLER_TORCH_COMPILE", "0") == "1":
        if hasattr(torch, "compile") and int(torch.__version__.split(".")[0]) >= 2:
            try:
                _model = torch.compile(_model)
                print("[parler_voice] torch.compile() applied")
            except Exception as e:
                print(f"[parler_voice] torch.compile() skipped: {e}")
    else:
        print("[parler_voice] torch.compile disabled (faster short sentences)")

    # Short warmup (1–2 words) — enough to allocate kernels without long delay
    if not _warmed_up:
        try:
            print("[parler_voice] Warmup generation (discarded)...")
            warm_style = build_voice_style("neutral", 0.5)
            desc = _tokenizer(warm_style, return_tensors="pt").input_ids.to(_device)
            prompt = _tokenizer("Hi.", return_tensors="pt").input_ids.to(_device)
            with torch.no_grad():
                _ = _model.generate(input_ids=desc, prompt_input_ids=prompt)
            _warmed_up = True
            print("[parler_voice] Warmup complete")
        except Exception as e:
            print(f"[parler_voice] Warmup skipped: {e}")

    print(f"[parler_voice] Model ready on {_device}")
    return _model, _tokenizer, _device


def _trim_trailing_silence(audio: np.ndarray, sample_rate: int, threshold: float = 0.015) -> np.ndarray:
    """
    Strip trailing silence from audio array.
    threshold: fraction of peak amplitude below which samples are considered silent.
    Keeps a 50ms tail after the last loud sample so we do not clip the natural decay.
    """
    if len(audio) == 0:
        return audio
    peak = float(np.abs(audio).max())
    if peak < 1e-6:
        return audio
    loud = np.where(np.abs(audio) > peak * threshold)[0]
    if len(loud) == 0:
        return audio
    tail_samples = int(sample_rate * 0.05)
    end = min(len(audio), loud[-1] + tail_samples + 1)
    return audio[:end]


def generate_speech(
    text: str,
    voice_style: str,
    output_path: str,
    play_audio: bool = False,
    max_new_tokens: int | None = None,
    trim_silence: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Generate speech with Parler-TTS and save a WAV file.

    Official Parler API:
      input_ids        = description (voice style)
      prompt_input_ids = text to speak

    Uses the model defaults (no forced min_new / soft_max from movie timing).
    Short lines like "Hi" stop at EOS and stay fast.

    max_new_tokens: optional hard ceiling (vocal stubs only).
    trim_silence:   strip trailing silence (vocal stubs).

    Returns (audio_float32, sample_rate).
    """
    model, tokenizer, device = load_parler()

    description_ids = tokenizer(voice_style, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    # Attention masks (pad==eos on this tokenizer → missing mask can cut speech early)
    description_mask = tokenizer(voice_style, return_tensors="pt").attention_mask.to(device)
    prompt_mask = tokenizer(text, return_tensors="pt").attention_mask.to(device)

    # Simple generate path (chat TTS). No min_new_tokens floor (that forced long
    # audio for "Hi"). Optional max_new_tokens for vocal stubs; otherwise a light
    # word-based CEILING only so EOS-late rambles cannot burn 30s of tokens.
    gen_kwargs: dict = dict(
        input_ids=description_ids,
        attention_mask=description_mask,
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
    )
    _n_words = max(1, len((text or "").split()))
    if max_new_tokens is not None:
        _cap = max(32, int(max_new_tokens))
    else:
        # ~0.55s/word + 1.2s headroom @ DAC ~86 Hz; short lines stay quick
        _frame_rate = 86.0
        try:
            _frame_rate = float(
                getattr(getattr(model.config, "audio_encoder", None), "frame_rate", None)
                or 86
            )
        except Exception:
            pass
        _max_sec = min(25.0, max(1.6, _n_words * 0.55 + 1.2))
        _cap = max(48, int(_max_sec * _frame_rate))
    gen_kwargs["max_new_tokens"] = _cap
    print(
        f"[parler_voice] generate max_new_tokens={_cap} "
        f"(words={_n_words} text={text[:40]!r})"
    )

    with torch.no_grad():
        try:
            generation = model.generate(**gen_kwargs)
        except TypeError:
            # Older parler-tts builds may not accept prompt_attention_mask
            _fb: dict = dict(
                input_ids=description_ids,
                prompt_input_ids=prompt_ids,
                max_new_tokens=_cap,
            )
            generation = model.generate(**_fb)

    audio = generation.cpu().float().numpy().squeeze()
    sample_rate = int(model.config.sampling_rate)
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    # Sanitize for libsndfile (NaN/Inf/empty → System error on write)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).any():
        print("[parler_voice] WARN empty/non-finite audio — writing short silence")
        audio = np.zeros(int(0.25 * max(1, sample_rate)), dtype=np.float32)
    else:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    if trim_silence:
        before = len(audio)
        audio = _trim_trailing_silence(audio, sample_rate)
        if audio.size < int(0.05 * sample_rate):
            audio = np.zeros(int(0.15 * sample_rate), dtype=np.float32)
        print(f"[parler_voice] silence trim: {before/sample_rate:.2f}s -> {len(audio)/sample_rate:.2f}s")

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    # 16-bit PCM for Rhubarb / PocketSphinx; robust write (disk/path/lock safe)
    _safe_write_wav(output_path, audio, sample_rate)
    print(
        f"[parler_voice] Saved: {output_path}  "
        f"({len(audio) / sample_rate:.2f}s @ {sample_rate} Hz)"
    )

    if play_audio:
        sd.play(audio, sample_rate)
        sd.wait()

    return audio, sample_rate


def generate_speech_for_emotion(
    text: str,
    emotion: str,
    intensity: float,
    output_path: str,
    play_audio: bool = False,
) -> Tuple[np.ndarray, int]:
    """Build style from emotion/intensity, then generate."""
    style = build_voice_style(emotion, intensity)
    print(f"[parler_voice] emotion={emotion!r} intensity={float(intensity):.2f} band={_intensity_band(intensity)}")
    print(f"[parler_voice] style: {style[:140]}...")
    return generate_speech(text, style, output_path, play_audio=play_audio)


# Text and emotion to pass to Parler for each vocal token.
# These produce the actual SOUND, not the spoken word.
VOCAL_SOUND_TEXTS: dict[str, Tuple[str, str, float]] = {
    "eww":     ("eww!",          "disgusted",  0.95),
    "ew":      ("ew!",           "disgusted",  0.90),
    "laugh":   ("ha ha ha ha!",  "happy",      0.95),
    "chuckle": ("heh heh heh.",  "happy",      0.80),
    "sigh":    ("ahhhh.",        "sad",        0.70),
    "gasp":    ("oh!",           "surprised",  0.95),
    "hmm":     ("hmm...",        "thinking",   0.65),
    "ugh":     ("ugh!",          "disgusted",  0.90),
    "wow":     ("wow!",          "surprised",  0.85),
    "cry":     ("oh... oh...",   "sad",        0.90),
    "crying":  ("oh... oh...",   "sad",        0.90),
    "sob":     ("oh no... oh.", "sad",         0.95),
    "scoff":   ("pff!",          "disgusted",  0.75),
    "groan":   ("uhhhh...",      "disgusted",  0.85),
    "yawn":    ("ahhhhh...",     "neutral",    0.50),
    "wince":   ("oh!",           "fearful",    0.85),
    "smirk":   ("hmm.",          "sarcastic",  0.65),
    "grunt":   ("grr!",          "angry",      0.95),
    "sniff":   ("sniff...",      "sad",        0.70),
    "nervous": ("heh... um...",  "fearful",    0.70),
    "sneeze":  ("achoo!",        "surprised",  0.90),
    "cough":   ("ahem!",         "neutral",    0.55),
}

_vocal_sound_cache: dict[str, Tuple[str, int]] = {}  # token -> (wav_path, sr)


def ensure_vocal_sounds(
    temp_dir: str | Path = "temp",
    *,
    generate_missing: bool = False,
    only_tokens: Optional[list] = None,
) -> dict[str, Tuple[str, int]]:
    """
    Index short vocalization clips under temp/vocal_sounds/.

    IMPORTANT: generate_missing defaults to False.
    Parler is TTS — it cannot produce real laughs/gasps/sighs, only spoken
    words like "ha ha ha". Those clips sound wrong for reaction tokens.

    Use REAL reaction audio instead:
      python tools/build_real_vocals.py
      # or drop your own recordings into temp/vocal_sounds/vocal_laugh.wav etc.

    Set generate_missing=True only for emergency stubs (not recommended).
    """
    global _vocal_sound_cache

    temp_dir = Path(temp_dir)
    sound_dir = temp_dir / "vocal_sounds"
    sound_dir.mkdir(parents=True, exist_ok=True)

    # Prefer real SFX library marker
    source_txt = sound_dir / "SOURCE.txt"
    if source_txt.is_file():
        print(f"[parler_voice] Vocal library: {source_txt.read_text(encoding='utf-8', errors='ignore').strip().splitlines()[0]}")

    _frame_rate_default = 86.0
    tokens = only_tokens if only_tokens is not None else list(VOCAL_SOUND_TEXTS.keys())

    for token in tokens:
        if token not in VOCAL_SOUND_TEXTS:
            continue
        if token in _vocal_sound_cache:
            continue
        path = str(sound_dir / f"vocal_{token}.wav")
        text, emotion, intensity = VOCAL_SOUND_TEXTS[token]
        _n_words = max(1, len((text or "").split()))
        _max_tok = min(180, int(_n_words * 1.8 * _frame_rate_default))
        _max_sec = (_max_tok / _frame_rate_default) * 1.25

        if os.path.isfile(path) and os.path.getsize(path) > 1000:
            try:
                data, sr = sf.read(path, dtype="float32")
                _vocal_sound_cache[token] = (path, int(sr))
                continue
            except Exception:
                pass

        if not generate_missing:
            # Do NOT invent Parler speech for reactions
            continue

        # Legacy fallback (discouraged): Parler speaks a short word
        try:
            print(
                f"[parler_voice] WARNING: generating STUB vocal [{token}] with Parler "
                f"(not a real laugh/gasp). Prefer tools/build_real_vocals.py"
            )
            style = build_voice_style(emotion, intensity)
            audio, sr = generate_speech(
                text,
                style,
                path,
                play_audio=False,
                max_new_tokens=_max_tok,
                trim_silence=True,
            )
            _vocal_sound_cache[token] = (path, int(sr))
        except Exception as e:
            print(f"[parler_voice] WARNING: vocal clip failed for [{token}]: {e}")

    return _vocal_sound_cache


def ensure_filler_sounds(temp_dir: str | Path = "temp") -> dict[str, Tuple[str, int]]:
    """
    Pre-generate short thinking fillers (play while waiting on LLM).
    Returns dict name -> (wav_path, sample_rate). Cached after first call.
    """
    global _filler_cache
    if _filler_cache:
        return _filler_cache

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(exist_ok=True)

    fillers = {
        "hmm": ("Hmm...", "thinking", 0.5),
        "let_me_think": ("Let me think...", "thinking", 0.6),
    }

    print("[parler_voice] Pre-generating filler / thinking sounds...")
    for name, (text, emotion, intensity) in fillers.items():
        path = str(temp_dir / f"filler_{name}.wav")
        if os.path.isfile(path) and os.path.getsize(path) > 1000:
            # Reuse existing file; still need sample rate
            data, sr = sf.read(path, dtype="float32")
            _filler_cache[name] = (path, int(sr))
            print(f"[parler_voice] Filler cached on disk: {path}")
            continue
        audio, sr = generate_speech_for_emotion(
            text=text,
            emotion=emotion,
            intensity=intensity,
            output_path=path,
            play_audio=False,
        )
        _filler_cache[name] = (path, sr)

    return _filler_cache


def play_filler(name: str = "hmm") -> None:
    """Play a pre-generated filler sound (non-blocking if already loaded)."""
    cache = ensure_filler_sounds()
    if name not in cache:
        name = next(iter(cache), None)
        if name is None:
            return
    path, sr = cache[name]
    audio, _ = sf.read(path, dtype="float32")
    sd.play(audio, sr)
    # Non-blocking — caller can continue LLM work; use sd.wait() if needed


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out = os.path.join("temp", "parler_test.wav")
    os.makedirs("temp", exist_ok=True)

    print("=== Parler-TTS self-test ===")
    t0 = time.time()
    generate_speech_for_emotion(
        text="Hello there. This is a short test of the talking face voice.",
        emotion="happy",
        intensity=0.7,
        output_path=out,
        play_audio=True,
    )
    print(f"Elapsed: {(time.time() - t0) * 1000:.0f} ms")
    print("Done.")
