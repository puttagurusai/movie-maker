"""
Option C: Convert real reaction SFX → Parler agent voice (OpenVoice tone-color VC).

Uses tools/ov_env (Python 3.11) when available — OpenVoice is brittle on 3.13.

Pipeline:
  1) Build / reuse Parler reference (temp/voice_ref_parler.wav)
  2) Load OpenVoice ToneColorConverter (no Whisper / no full pip -e install)
  3) Convert each vocal_*.wav from mixkit backup → agent timbre
  4) Write to temp/vocal_sounds/

Usage:
  tools\\ov_env\\Scripts\\python.exe tools\\voice_convert_vocals.py
  tools\\ov_env\\Scripts\\python.exe tools\\voice_convert_vocals.py --ref temp\\voice_ref_parler.wav
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
OV_DIR = TOOLS / "OpenVoice"
OV_ENV_PY = TOOLS / "ov_env" / "Scripts" / "python.exe"
CKPT_V2 = OV_DIR / "checkpoints_v2" / "converter"
CKPT_V1 = OV_DIR / "checkpoints" / "converter"
CKPT_ZIP = TOOLS / "checkpoints_v2_0417.zip"
CKPT_ZIP_URLS = [
    "https://huggingface.co/cqchangm/openvoice2/resolve/main/checkpoints_v2_0417.zip",
    "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip",
]
SOUND_DIR = ROOT / "temp" / "vocal_sounds"
MIXKIT_BAK = ROOT / "temp" / "vocal_sounds_mixkit"
PRE_VC_BACKUP = ROOT / "temp" / "vocal_sounds_pre_vc"
REF_PATH = ROOT / "temp" / "voice_ref_parler.wav"
OUT_DIR = ROOT / "temp" / "vocal_sounds_vc"
PROCESSED = ROOT / "temp" / "openvoice_processed"


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_running_under_ov_env() -> None:
    """Re-exec under tools/ov_env if this is the system Python."""
    if not OV_ENV_PY.is_file():
        return
    try:
        cur = Path(sys.executable).resolve()
        want = OV_ENV_PY.resolve()
        if cur == want:
            return
    except Exception:
        pass
    # Only re-exec if we're not already in a venv that has openvoice path working
    if "ov_env" in str(sys.executable).lower():
        return
    log(f"[env] Re-launching under {OV_ENV_PY}")
    os.execv(str(OV_ENV_PY), [str(OV_ENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])


def download_file(url: str, out: Path) -> bool:
    import urllib.request

    log(f"[dl] {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r, open(out, "wb") as f:
            shutil.copyfileobj(r, f)
        log(f"[dl] OK {out} ({out.stat().st_size} bytes)")
        return True
    except Exception as e:
        log(f"[dl] FAIL: {e}")
        if out.is_file():
            out.unlink(missing_ok=True)
        return False


def ensure_checkpoints() -> Path:
    """Return converter dir with config.json + checkpoint.pth."""
    for ck in (CKPT_V2, CKPT_V1):
        if (ck / "checkpoint.pth").is_file() and (ck / "config.json").is_file():
            log(f"[ckpt] Using {ck}")
            return ck

    if not CKPT_ZIP.is_file() or CKPT_ZIP.stat().st_size < 1_000_000:
        ok = False
        for url in CKPT_ZIP_URLS:
            if download_file(url, CKPT_ZIP):
                ok = True
                break
        if not ok:
            raise FileNotFoundError("Could not download OpenVoice checkpoints")

    log(f"[ckpt] Extracting {CKPT_ZIP}")
    with zipfile.ZipFile(CKPT_ZIP, "r") as z:
        z.extractall(OV_DIR)

    for ck in (CKPT_V2, CKPT_V1):
        if (ck / "checkpoint.pth").is_file():
            log(f"[ckpt] Using {ck}")
            return ck

    # search
    for p in OV_DIR.rglob("converter/checkpoint.pth"):
        log(f"[ckpt] Found {p.parent}")
        return p.parent
    raise FileNotFoundError("OpenVoice converter checkpoint not found after extract")


def build_parler_reference(out_path: Path, seconds_target: float = 25.0) -> Path:
    if out_path.is_file() and out_path.stat().st_size > 50_000:
        log(f"[ref] Using existing {out_path}")
        return out_path

    import numpy as np
    import soundfile as sf

    parts = sorted((ROOT / "temp" / "verify_run").glob("s1_speech_*.wav"))
    if len(parts) < 2:
        parts = sorted((ROOT / "temp").glob("**/s*_speech_*.wav"))
    if len(parts) >= 1:
        log(f"[ref] Concatenating {len(parts)} existing speech clips")
        audios = []
        sr0 = None
        for p in parts:
            a, sr = sf.read(str(p), dtype="float32")
            if a.ndim > 1:
                a = a.mean(axis=1)
            if sr0 is None:
                sr0 = sr
            elif sr != sr0:
                import scipy.signal as sig

                a = sig.resample(a, int(len(a) * sr0 / sr)).astype(np.float32)
            audios.append(a)
            audios.append(np.zeros(int(0.15 * sr0), dtype=np.float32))
        full = np.concatenate(audios)
        need = int(seconds_target * sr0)
        if len(full) < need:
            full = np.tile(full, int(np.ceil(need / max(len(full), 1))))[:need]
        else:
            full = full[:need]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), full, sr0)
        log(f"[ref] Wrote {out_path} ({len(full) / sr0:.1f}s)")
        return out_path

    log("[ref] No existing speech clips; leave ref missing")
    return out_path


def content_source_dir() -> Path:
    """Prefer original Mixkit SFX, not already-VC'd audio."""
    if MIXKIT_BAK.is_dir() and any(MIXKIT_BAK.glob("vocal_*.wav")):
        return MIXKIT_BAK
    if PRE_VC_BACKUP.is_dir() and any(PRE_VC_BACKUP.glob("vocal_*.wav")):
        src = (PRE_VC_BACKUP / "SOURCE.txt").read_text(encoding="utf-8", errors="ignore") if (PRE_VC_BACKUP / "SOURCE.txt").is_file() else ""
        if "mixkit" in src.lower() or "openvoice" not in src.lower():
            return PRE_VC_BACKUP
    return SOUND_DIR


def convert_all(ref_path: Path) -> int:
    import numpy as np
    import soundfile as sf
    import torch

    if str(OV_DIR) not in sys.path:
        sys.path.insert(0, str(OV_DIR))

    from openvoice.api import ToneColorConverter

    ckpt_dir = ensure_checkpoints()
    conv_cfg = ckpt_dir / "config.json"
    conv_ckpt = ckpt_dir / "checkpoint.pth"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"[vc] device={device}")
    log(f"[vc] converter={ckpt_dir}")

    # Avoid wavmark: OpenVoiceBaseClass rejects enable_watermark kwarg on some versions
    # (ToneColorConverter passes **kwargs to super). Patch after init instead.
    import openvoice.api as ov_api

    _orig_tc_init = ov_api.ToneColorConverter.__init__

    def _tc_init_no_wm(self, *a, **kw):
        kw.pop("enable_watermark", None)
        # Call grandparent-style: Base init only, then set watermark off
        ov_api.OpenVoiceBaseClass.__init__(self, *a, **kw)
        self.watermark_model = None
        self.version = getattr(self.hps, "_version_", "v1")

    ov_api.ToneColorConverter.__init__ = _tc_init_no_wm
    converter = ToneColorConverter(str(conv_cfg), device=device)
    converter.load_ckpt(str(conv_ckpt))

    # Direct SE extract — skip se_extractor (needs Whisper / silero VAD)
    log(f"[vc] Target SE from {ref_path}")
    target_se = converter.extract_se([str(ref_path)], se_save_path=str(PROCESSED / "target_se.pth"))
    log("[vc] Target SE ready")

    content_dir = content_source_dir()
    log(f"[vc] Content source: {content_dir}")

    # Backup current if not mixkit backup yet
    if SOUND_DIR.is_dir() and not MIXKIT_BAK.is_dir():
        source_txt = SOUND_DIR / "SOURCE.txt"
        src = source_txt.read_text(encoding="utf-8", errors="ignore") if source_txt.is_file() else ""
        if "mixkit" in src.lower() or "openvoice" not in src.lower():
            shutil.copytree(SOUND_DIR, MIXKIT_BAK)
            log(f"[backup] mixkit → {MIXKIT_BAK}")

    if SOUND_DIR.is_dir() and not PRE_VC_BACKUP.is_dir():
        shutil.copytree(SOUND_DIR, PRE_VC_BACKUP)
        log(f"[backup] pre_vc → {PRE_VC_BACKUP}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOUND_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    wavs = sorted(content_dir.glob("vocal_*.wav"))
    if not wavs:
        log("[vc] No vocal_*.wav found")
        return 1

    ok = 0
    for src in wavs:
        try:
            a, sr = sf.read(str(src), dtype="float32")
            if a.ndim > 1:
                a = a.mean(axis=1)
            if len(a) < max(int(sr * 0.12), 1000):
                log(f"[skip] {src.name} too short")
                continue
        except Exception as e:
            log(f"[skip] {src.name}: {e}")
            continue

        dst_tmp = OUT_DIR / src.name
        dst_final = SOUND_DIR / src.name
        log(f"[vc] {src.name} ...")
        try:
            src_se = converter.extract_se([str(src)])
            converter.convert(
                audio_src_path=str(src),
                src_se=src_se,
                tgt_se=target_se,
                output_path=str(dst_tmp),
                message="@MyShell",
                tau=0.3,
            )
            a, sr = sf.read(str(dst_tmp), dtype="float32")
            if a.ndim > 1:
                a = a.mean(axis=1)
            peak = float(np.abs(a).max()) if len(a) else 0.0
            if peak > 1e-6:
                a = np.clip(a * (0.85 / peak), -1.0, 1.0)
            sf.write(str(dst_final), a.astype(np.float32), int(sr))
            log(f"  → {dst_final.name} ({len(a) / sr:.2f}s)")
            ok += 1
        except Exception as e:
            log(f"  FAIL {src.name}: {e}")
            # keep content original so tokens still work
            try:
                shutil.copy2(src, dst_final)
            except Exception:
                pass

    (SOUND_DIR / "SOURCE.txt").write_text(
        "source=openvoice_vc_to_parler_ref\n"
        f"reference={ref_path}\n"
        f"content={content_dir}\n"
        "Content: real reaction SFX (Mixkit). Timbre: OpenVoice tone-color toward agent.\n"
        "NOT Parler TTS speech stubs — real non-speech vocals voice-converted only.\n",
        encoding="utf-8",
    )
    log(f"\n[done] Converted {ok}/{len(wavs)} clips → {SOUND_DIR}")
    log("Test: python tools/verify_json_audio_lips.py temp/test_pipeline_input.json")
    return 0 if ok else 1


def main() -> int:
    ensure_running_under_ov_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=str, default=str(REF_PATH))
    ap.add_argument("--skip-ref-gen", action="store_true")
    args = ap.parse_args()

    log("=" * 60)
    log("OPTION C: Voice-convert reaction SFX → Parler agent timbre")
    log(f"Python: {sys.version.split()[0]}  exe={sys.executable}")
    log("=" * 60)

    if not OV_DIR.is_dir():
        log(f"Missing OpenVoice clone at {OV_DIR}")
        log("Run: git clone --depth 1 https://github.com/myshell-ai/OpenVoice.git tools/OpenVoice")
        return 1

    ref = Path(args.ref)
    if not args.skip_ref_gen:
        ref = build_parler_reference(ref)
    if not ref.is_file():
        log(f"Missing reference: {ref}")
        return 1

    return convert_all(ref)


if __name__ == "__main__":
    raise SystemExit(main())
