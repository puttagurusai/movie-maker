#!/usr/bin/env python
"""
export_movie_preview_mp4.py — mesh/Workbench preview MP4 of a baked movie package.

Optimized for short AND long inputs:
  - FAST   (default for E2E): 1× Blender + frame-step (e.g. every 3rd frame) + JPEG
           ~3–8× fewer stills than full 30fps PNG long mode
  - SHORT  : one Blender pass for whole film
  - LONG   : per-shot Blender (8 process launches — slow; only for very long films)
  - Never keeps full-film still libraries unless --keep-frames

Time math (75s film @ 30fps):
  full long:  ~2250 frames × ~0.45s + 8× Blender load ≈ 20+ min
  fast step3: ~750 frames × ~0.35s + 1× Blender load  ≈ 4–6 min

Requires: imageio, imageio-ffmpeg  (pip install -r requirements-movie.txt)

Examples:
  python export_movie_preview_mp4.py --mode fast
  python export_movie_preview_mp4.py --package temp/movies/demo_xxx/movie_package.json --mode fast --frame-step 3
  python export_movie_preview_mp4.py --mode short --frame-step 1 --still-format PNG   # full quality
  python run_movie_pipeline.py --story "..." --preview-mp4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Auto long-form when total duration exceeds this (seconds)
DEFAULT_LONG_THRESHOLD_S = 45.0


def _find_blender() -> str:
    env = os.environ.get("BLENDER_EXE", "").strip()
    if env and Path(env).is_file():
        return env
    for c in (
        r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    ):
        if Path(c).is_file():
            return c
    return "blender"


def _find_ffmpeg() -> Optional[str]:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    for c in (
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ):
        if Path(c).is_file():
            return c
    return None


def _find_latest_package() -> Optional[Path]:
    root = ROOT / "temp" / "movies"
    if not root.is_dir():
        return None
    pkgs = sorted(root.glob("*/movie_package.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pkgs[0] if pkgs else None


def _normalize_clip(c: Dict[str, Any], raw_shot: Optional[Dict] = None) -> Dict[str, Any]:
    out = dict(c)
    raw = raw_shot or {}
    if not out.get("body_action"):
        acts = [str(a).lower() for a in (out.get("actions") or raw.get("actions") or [])]
        for prefer in ("wave", "talk_open", "celebrate", "shrug", "idle", "walk", "run"):
            if prefer in acts:
                out["body_action"] = prefer
                break
        if not out.get("body_action"):
            out["body_action"] = "talk_open"
    if not out.get("audio"):
        out["audio"] = raw.get("audio_path") or ""
    if not out.get("face_timeline"):
        out["face_timeline"] = raw.get("face_timeline_path") or ""
    # Chat-parity body playback fields (do not stretch walk to fill take)
    if not out.get("body_mode"):
        out["body_mode"] = raw.get("body_mode") or ""
    if not out.get("clip_policy"):
        out["clip_policy"] = raw.get("clip_policy") or ""
    if out.get("clip_speed") is None:
        out["clip_speed"] = raw.get("clip_speed") or 1.0
    ba = str(out.get("body_action") or "")
    bm = str(out.get("body_mode") or "").lower()
    if not out.get("clip_fps"):
        if bm in ("momask", "both") or ba.startswith("momask_"):
            out["clip_fps"] = 20.0
        else:
            out["clip_fps"] = float(raw.get("clip_fps") or 30.0)
    if not out.get("clip_policy"):
        out["clip_policy"] = (
            "momask_match" if (bm in ("momask", "both") or ba.startswith("momask_")) else "loop"
        )
    return out


def _load_clips(package: Path) -> Tuple[List[Dict[str, Any]], float, float]:
    data = json.loads(package.read_text(encoding="utf-8"))
    fps = float(data.get("fps") or 30)
    if "clips" in data:
        clips = [_normalize_clip(c) for c in data["clips"]]
        dur = float(data.get("duration_s") or max((float(c.get("t1") or 0) for c in clips), default=0))
        return clips, fps, dur
    clips = []
    for s in data.get("shots") or []:
        clips.append(
            _normalize_clip(
                {
                    "shot_id": s.get("shot_id"),
                    "t0": s.get("t0"),
                    "t1": s.get("t1"),
                    "duration_s": s.get("duration_s"),
                    "audio": s.get("audio_path") or s.get("audio"),
                    "face_timeline": s.get("face_timeline_path") or s.get("face_timeline"),
                    "body_action": s.get("body_action") or "",
                    "body_library": s.get("body_library") or "",
                    "body_mode": s.get("body_mode") or "",
                    "clip_policy": s.get("clip_policy") or "",
                    "clip_speed": s.get("clip_speed") if s.get("clip_speed") is not None else 1.0,
                    "clip_fps": s.get("clip_fps"),
                    "camera": s.get("camera"),
                    "actions": s.get("actions") or [],
                },
                raw_shot=s,
            )
        )
    dur = float(data.get("duration_s") or max((float(c.get("t1") or 0) for c in clips), default=0))
    return clips, fps, dur


def stitch_audio_wav(clips: list, out_wav: Path) -> Optional[Path]:
    """
    Lay shot WAVs on the master timeline with silence for:
      - gaps before each shot t0
      - remaining shot length after speech ends (until t1)
    So audio duration matches picture (~sum of shot durations), not just spoken seconds.
    """
    import audioop

    segments: List[bytes] = []
    target_rate = target_width = target_ch = None
    cursor_s = 0.0

    def _silence(seconds: float) -> bytes:
        if not target_rate or seconds <= 0.0005:
            return b""
        n = int(round(seconds * target_rate))
        return b"\x00" * (n * target_width * target_ch)

    for c in clips:
        path = c.get("audio") or ""
        t0 = float(c.get("t0") or 0.0)
        t1 = float(c.get("t1") or 0.0)
        # Pad silence up to this shot's start
        gap = t0 - cursor_s
        if gap > 0.001 and target_rate:
            segments.append(_silence(gap))
            cursor_s = t0
        elif gap > 0.001 and target_rate is None:
            # First clip starts later — need rate first; create after open
            pass

        speech_s = 0.0
        if path and Path(path).is_file():
            with wave.open(path, "rb") as w:
                rate, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
                frames = w.readframes(w.getnframes())
                speech_s = w.getnframes() / float(rate) if rate else 0.0
                if target_rate is None:
                    target_rate, target_width, target_ch = rate, width, ch
                    # Late first-shot gap (story starts at t0>0)
                    if t0 > 0.001:
                        segments.insert(0, _silence(t0))
                        cursor_s = t0
                elif (rate, width, ch) != (target_rate, target_width, target_ch):
                    if width == target_width and ch == target_ch and rate != target_rate:
                        frames, _ = audioop.ratecv(frames, width, ch, rate, target_rate, None)
                        speech_s = (len(frames) // (target_width * target_ch)) / float(target_rate)
                    else:
                        print(f"[audio] skip incompatible {path}")
                        frames = b""
                        speech_s = 0.0
                if frames:
                    segments.append(frames)
                    cursor_s = t0 + speech_s
        # Pad silence to end of shot (hold / motion after speech)
        shot_end = t1 if t1 > t0 else (t0 + max(speech_s, 0.05))
        pad = shot_end - cursor_s
        if pad > 0.001 and target_rate:
            segments.append(_silence(pad))
            cursor_s = shot_end
        elif t1 > cursor_s and target_rate:
            segments.append(_silence(t1 - cursor_s))
            cursor_s = t1

    if not segments or not target_rate:
        return None
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as out:
        out.setnchannels(target_ch)
        out.setsampwidth(target_width)
        out.setframerate(target_rate)
        out.writeframes(b"".join(segments))
    print(f"[audio] stitched → {out_wav} ({out_wav.stat().st_size/1024:.1f} KB)")
    return out_wav


def find_blend() -> Path:
    for rel in (
        "whole_body_production_ready.blend",
        "whole_body_retargeted.blend",
        "body_motion/whole_body_with_clips.blend",
    ):
        p = ROOT / rel
        if p.is_file():
            return p
    raise FileNotFoundError("No production .blend found")


def _run_blender_frames(
    *,
    package: Path,
    frames_dir: Path,
    fps: int,
    res: str,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
    clip_filter: Optional[str] = None,
    frame_step: int = 1,
    still_format: str = "JPEG",
) -> int:
    """Invoke Blender Workbench frame dump (optional frame range / single shot)."""
    blender = _find_blender()
    blend = find_blend()
    script = ROOT / "tools" / "blender_movie_preview.py"
    cmd = [
        blender,
        str(blend),
        "--background",
        "--python",
        str(script),
        "--",
        "--package",
        str(package.resolve()),
        "--out",
        str((frames_dir.parent / "preview_mesh.mp4").resolve()),
        "--fps",
        str(fps),
        "--res",
        res,
        "--frames-dir",
        str(frames_dir.resolve()),
        "--frame-step",
        str(max(1, int(frame_step))),
        "--still-format",
        str(still_format or "JPEG"),
    ]
    if frame_start is not None:
        cmd += ["--frame-start", str(frame_start)]
    if frame_end is not None:
        cmd += ["--frame-end", str(frame_end)]
    if clip_filter:
        cmd += ["--clip-id", clip_filter]
    print(
        f"[export] Blender step={frame_step} fmt={still_format} res={res} "
        f"{'range '+str(frame_start)+'-'+str(frame_end) if frame_start else 'full'}"
    )
    r = subprocess.run(cmd, cwd=str(ROOT))
    return r.returncode


def _list_stills(frames_dir: Path) -> List[Path]:
    stills = sorted(frames_dir.glob("f*.jpg")) + sorted(frames_dir.glob("f*.jpeg"))
    if not stills:
        stills = sorted(frames_dir.glob("f*.png"))
    return stills


def _encode_frames_to_mp4(
    frames_dir: Path,
    out_mp4: Path,
    *,
    fps: float,
    audio_wav: Optional[Path] = None,
    delete_frames: bool = True,
) -> Optional[Path]:
    """Encode f#####.jpg/png → mp4; optionally mux audio; optionally delete stills."""
    stills = _list_stills(frames_dir)
    if len(stills) < 2:
        print(f"[export] not enough frames in {frames_dir}")
        return None

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ff = _find_ffmpeg()
    ext = stills[0].suffix.lstrip(".") or "jpg"
    # f%05d.jpg / .png
    pattern = str(frames_dir / f"f%05d.{ext}")
    if stills[0].suffix.lower() == ".jpeg":
        pattern = str(frames_dir / "f%05d.jpeg")

    # Prefer ffmpeg (system or imageio_ffmpeg binary)
    if ff:
        first = stills[0].stem  # f00001
        try:
            start_num = int(first.replace("f", ""))
        except ValueError:
            start_num = 1
        cmd = [
            ff, "-y",
            "-framerate", str(fps),
            "-start_number", str(start_num),
            "-i", pattern,
        ]
        if audio_wav and audio_wav.is_file():
            # Do NOT use -shortest: short speech WAVs used to truncate the picture.
            # Audio is padded to full shot timeline by stitch_audio_wav.
            cmd += [
                "-i", str(audio_wav),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k",
                "-af", "apad", "-shortest",
            ]
        else:
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23"]
        cmd.append(str(out_mp4))
        print(f"[export] ffmpeg encode → {out_mp4.name} ({len(stills)} frames @ {fps:.1f}fps)")
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-800:] if r.stderr else "ffmpeg failed")
        elif out_mp4.is_file():
            if delete_frames:
                _rm_tree(frames_dir)
            return out_mp4

    # imageio fallback
    try:
        import imageio.v2 as iio

        print(f"[export] imageio encode → {out_mp4.name}")
        writer = iio.get_writer(
            str(out_mp4), fps=float(fps), codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=1
        )
        for p in stills:
            writer.append_data(iio.imread(p))
        writer.close()
        if audio_wav and audio_wav.is_file() and ff:
            muxed = out_mp4.parent / (out_mp4.stem + "_audio.mp4")
            subprocess.run(
                [ff, "-y", "-i", str(out_mp4), "-i", str(audio_wav),
                 "-c:v", "copy", "-c:a", "aac", "-shortest", str(muxed)],
                check=False,
            )
            if muxed.is_file():
                if delete_frames:
                    _rm_tree(frames_dir)
                return muxed
        if delete_frames:
            _rm_tree(frames_dir)
        return out_mp4 if out_mp4.is_file() else None
    except Exception as e:
        print(f"[export] encode failed: {e}")
        return None


def _rm_tree(p: Path) -> None:
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            print(f"[export] cleaned temp frames {p.name}")
    except Exception:
        pass


def _concat_mp4s(parts: List[Path], out_mp4: Path, audio_wav: Optional[Path]) -> Optional[Path]:
    ff = _find_ffmpeg()
    if not ff or not parts:
        return None
    lst = out_mp4.parent / "concat_list.txt"
    lines = []
    for p in parts:
        # ffmpeg concat demuxer needs escaped paths
        path = str(p.resolve()).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{path}'")
    lst.write_text("\n".join(lines), encoding="utf-8")
    video_only = out_mp4.parent / (out_mp4.stem + "_vid.mp4")
    cmd = [ff, "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(video_only)]
    print(f"[export] concat {len(parts)} shot videos…")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not video_only.is_file():
        print(r.stderr[-600:] if r.stderr else "concat failed")
        return None
    if audio_wav and audio_wav.is_file():
        cmd2 = [
            ff, "-y", "-i", str(video_only), "-i", str(audio_wav),
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_mp4),
        ]
        subprocess.run(cmd2, check=False)
        try:
            video_only.unlink(missing_ok=True)
        except Exception:
            pass
        return out_mp4 if out_mp4.is_file() else video_only
    shutil.move(str(video_only), str(out_mp4))
    return out_mp4


def _mux_video_audio(video: Path, audio: Optional[Path], out_mp4: Path) -> Optional[Path]:
    ff = _find_ffmpeg()
    if not ff or not video.is_file():
        return video if video.is_file() else None
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if audio and audio.is_file():
        cmd = [
            ff, "-y", "-i", str(video), "-i", str(audio),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            "-shortest", str(out_mp4),
        ]
        print(f"[export] mux audio → {out_mp4.name}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and out_mp4.is_file():
            return out_mp4
        print(r.stderr[-400:] if r.stderr else "mux failed")
    # video only
    try:
        if video.resolve() != out_mp4.resolve():
            shutil.copy2(video, out_mp4)
        return out_mp4 if out_mp4.is_file() else video
    except Exception:
        return video


def export_short(
    package: Path,
    clips: list,
    *,
    fps: int,
    res: str,
    out_mp4: Path,
    audio_wav: Path,
    keep_frames: bool,
    frame_step: int = 1,
    still_format: str = "JPEG",
) -> Optional[Path]:
    """
    One Blender pass: prefer DIRECT Workbench→FFMPEG (no stills), then mux audio.
    Falls back to JPEG stills + encode if direct path fails.
    """
    frames_dir = package.parent / "_tmp_preview_frames"
    _rm_tree(frames_dir)
    step = max(1, int(frame_step))
    # Clear prior direct markers
    marker = package.parent / "preview_direct_video.txt"
    for p in package.parent.glob("*_vid.mp4"):
        try:
            p.unlink()
        except Exception:
            pass
    if marker.is_file():
        try:
            marker.unlink()
        except Exception:
            pass

    code = _run_blender_frames(
        package=package,
        frames_dir=frames_dir,
        fps=fps,
        res=res,
        frame_step=step,
        still_format=still_format,
    )
    if code != 0:
        print(f"[export] Blender exit {code}")

    # Direct video from Blender FFMPEG writer?
    direct = None
    if marker.is_file():
        try:
            direct = Path(marker.read_text(encoding="utf-8").strip())
        except Exception:
            direct = None
    if direct is None or not direct.is_file():
        cands = sorted(package.parent.glob("*_vid.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        direct = cands[0] if cands else None
    if direct is not None and direct.is_file() and direct.stat().st_size > 2000:
        print(f"[export] using DIRECT Blender video ({direct.stat().st_size/1024:.0f} KB) — skip still encode")
        final = _mux_video_audio(direct, audio_wav if audio_wav.is_file() else None, out_mp4)
        if not keep_frames:
            _rm_tree(frames_dir)
        return final

    encode_fps = max(1.0, float(fps) / float(step))
    return _encode_frames_to_mp4(
        frames_dir,
        out_mp4,
        fps=encode_fps,
        audio_wav=audio_wav,
        delete_frames=not keep_frames,
    )


def export_long(
    package: Path,
    clips: list,
    *,
    fps: int,
    res: str,
    out_mp4: Path,
    audio_wav: Path,
    keep_frames: bool,
    frame_step: int = 1,
    still_format: str = "JPEG",
) -> Optional[Path]:
    """
    Per-shot: render only that shot's frame range → encode shot MP4 → delete frames.
    Then concat shot MP4s + master audio. Scales to long films without huge PNG trees.
    """
    work = package.parent / "_tmp_shots"
    work.mkdir(parents=True, exist_ok=True)
    shot_mp4s: List[Path] = []
    step = max(1, int(frame_step))
    encode_fps = max(1.0, float(fps) / float(step))

    for i, c in enumerate(clips, 1):
        sid = str(c.get("shot_id") or f"shot_{i:02d}")
        t0 = float(c.get("t0") or 0)
        t1 = float(c.get("t1") or t0)
        f0 = max(1, int(round(t0 * fps)) + 1)
        f1 = max(f0 + 1, int(round(t1 * fps)) + 1)
        n_full = f1 - f0 + 1
        n_est = max(2, (n_full + step - 1) // step)
        print(f"\n[export] LONG shot {sid} frames {f0}-{f1} (~{n_est} stills step={step}) ({t1-t0:.1f}s)")

        frames_dir = work / f"{sid}_frames"
        _rm_tree(frames_dir)

        code = _run_blender_frames(
            package=package,
            frames_dir=frames_dir,
            fps=fps,
            res=res,
            frame_start=f0,
            frame_end=f1,
            clip_filter=sid,
            frame_step=step,
            still_format=still_format,
        )
        if code != 0:
            print(f"[export] Blender fail on {sid}")

        shot_vid = work / f"{sid}.mp4"
        shot_audio = Path(c.get("audio") or "")
        enc = _encode_frames_to_mp4(
            frames_dir,
            shot_vid,
            fps=encode_fps,
            audio_wav=shot_audio if shot_audio.is_file() else None,
            delete_frames=not keep_frames,
        )
        if enc and enc.is_file():
            shot_mp4s.append(enc)
            print(f"[export] shot video {enc.name}")
        else:
            print(f"[export] WARN: no video for {sid}")

    if not shot_mp4s:
        return None
    final = _concat_mp4s(shot_mp4s, out_mp4, audio_wav if audio_wav.is_file() else None)
    if not keep_frames:
        _rm_tree(work)
    return final


def main() -> int:
    ap = argparse.ArgumentParser(description="Optimized mesh preview MP4 (short + long + fast)")
    ap.add_argument("--package", type=str, default="", help="movie_package.json")
    ap.add_argument("--out", type=str, default="", help="Output mp4")
    ap.add_argument(
        "--res",
        type=str,
        default="640x360",
        help="WxH (default 640x360 for fast preview; use 960x540 for share quality)",
    )
    ap.add_argument("--fps", type=int, default=0, help="Timeline fps (package default 30)")
    ap.add_argument(
        "--mode",
        choices=("auto", "short", "long", "fast"),
        default="fast",
        help="fast=default (1× Blender + frame-step). auto|short|long also available.",
    )
    ap.add_argument(
        "--long-threshold",
        type=float,
        default=DEFAULT_LONG_THRESHOLD_S,
        help="Seconds above which auto uses per-shot mode (default 45)",
    )
    ap.add_argument(
        "--frame-step",
        type=int,
        default=0,
        help="Render every Nth timeline frame (0=auto: 1 full, 2–3 for fast). ~Nx fewer stills.",
    )
    ap.add_argument(
        "--still-format",
        type=str,
        default="JPEG",
        choices=("JPEG", "PNG", "JPG"),
        help="Still format (JPEG much faster than PNG)",
    )
    ap.add_argument("--keep-frames", action="store_true", help="Keep still temps for debug")
    args = ap.parse_args()

    # Ensure encode deps
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        print("[export] Installing imageio imageio-ffmpeg…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio", "imageio-ffmpeg", "-q"])

    pkg = Path(args.package) if args.package else _find_latest_package()
    if args.package and "*" in args.package:
        matches = sorted(ROOT.glob(args.package), key=lambda p: p.stat().st_mtime, reverse=True)
        pkg = matches[0] if matches else None
    if pkg is None or not Path(pkg).is_file():
        print("No movie_package.json — run run_movie_pipeline.py first.")
        return 1
    pkg = Path(pkg)
    print(f"[export] package={pkg}")

    clips, pkg_fps, duration = _load_clips(pkg)
    fps = args.fps or int(pkg_fps) or 30
    out_mp4 = Path(args.out) if args.out else (pkg.parent / "preview_mesh_audio.mp4")
    if not out_mp4.is_absolute():
        out_mp4 = ROOT / out_mp4

    mode = args.mode
    # Fast path: single Blender launch + subsampled frames (biggest win)
    if mode == "fast":
        mode = "short"
        if not args.frame_step or args.frame_step < 1:
            # 2 min @ 30fps → step 4 ≈ 15fps encode feel with ~75% fewer stills
            if duration >= 90:
                args.frame_step = 4
            elif duration >= 40:
                args.frame_step = 3
            else:
                args.frame_step = 2
        if not args.res or args.res in ("960x540", "1280x720", "1920x1080"):
            args.res = "640x360" if duration >= 45 else "720x405"
        print(
            f"[export] FAST playblast: 1× Blender + step={args.frame_step} "
            f"res={args.res} JPEG (preview quality, not final)"
        )
    elif mode == "auto":
        # Default auto → same as fast for anything under ~4 min
        if duration >= 240:
            mode = "long"
            if not args.frame_step:
                args.frame_step = 3
            if not args.res or args.res == "960x540":
                args.res = "640x360"
        else:
            mode = "short"
            if not args.frame_step or args.frame_step < 1:
                args.frame_step = 4 if duration >= 90 else (3 if duration >= 40 else 2)
            if not args.res or args.res in ("960x540", "1280x720"):
                args.res = "640x360"

    frame_step = max(1, int(args.frame_step or 1))
    still_format = "JPEG" if args.still_format.upper() in ("JPEG", "JPG") else "PNG"
    encode_fps = max(1.0, float(fps) / float(frame_step))
    n_full = int(round(duration * fps)) + 1
    n_est = max(2, (n_full + frame_step - 1) // frame_step)
    print(
        f"[export] duration={duration:.1f}s mode={mode} timeline_fps={fps} "
        f"frame_step={frame_step} → ~{n_est} stills @ {encode_fps:.1f}fps  res={args.res} fmt={still_format}"
    )
    print(
        f"[export] cost model: ~{n_full} full frames → ~{n_est} rendered "
        f"({100.0 * n_est / max(1, n_full):.0f}% of full; ~{n_full / max(1, n_est):.1f}× fewer)"
    )

    audio_wav = pkg.parent / "preview_audio_stitched.wav"
    stitch_audio_wav(clips, audio_wav)

    if mode == "long":
        res = args.res
        if res == "960x540" and duration >= args.long_threshold:
            res = "640x360"
            print(f"[export] long mode: res → {res} (override with --res)")
        final = export_long(
            pkg, clips, fps=fps, res=res, out_mp4=out_mp4,
            audio_wav=audio_wav, keep_frames=args.keep_frames,
            frame_step=frame_step, still_format=still_format,
        )
    else:
        final = export_short(
            pkg, clips, fps=fps, res=args.res, out_mp4=out_mp4,
            audio_wav=audio_wav, keep_frames=args.keep_frames,
            frame_step=frame_step, still_format=still_format,
        )

    if not final or not Path(final).is_file():
        print("[export] FAILED — no MP4")
        return 1

    mb = Path(final).stat().st_size / (1024 * 1024)
    print("=" * 60)
    print("PREVIEW READY (mesh Workbench + pipeline audio)")
    print(f"  video : {final}  ({mb:.1f} MB)")
    print(f"  audio : {audio_wav}")
    print(f"  mode  : {mode}  duration={duration:.1f}s  step={frame_step} encode_fps={encode_fps:.1f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
