"""
Movie render + mix helpers (Blender and ffmpeg).

Master evaluate/render: 20 fps (Session_Timeline 1:1 keys).
Delivered picture: ffmpeg fps=24.
Mix adelay uses (speech_frame_start - 1) / 20.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

MASTER_FPS = 20.0
PICTURE_FPS = 24

# Compositor lift/gamma/gain + mild bloom. none/neutral = identity (nodes still exist).
GRADE_PRESETS: Dict[str, Optional[Dict[str, Any]]] = {
    "none": None,
    "neutral": None,
    "warm": {
        "lift": (0.04, 0.018, 0.0),
        "gamma": (0.98, 1.0, 1.04),
        "gain": (1.08, 1.02, 0.95),
        "bloom": 0.14,
        "sat": 1.0,
    },
    "cool": {
        "lift": (0.0, 0.012, 0.04),
        "gamma": (1.04, 1.0, 0.98),
        "gain": (0.95, 1.0, 1.08),
        "bloom": 0.08,
        "sat": 1.0,
    },
    "film_contrast": {
        "lift": (-0.03, -0.028, -0.02),
        "gamma": (1.05, 1.04, 1.03),
        "gain": (1.12, 1.08, 1.04),
        "bloom": 0.12,
        "sat": 0.95,
    },
    "bleach": {
        "lift": (0.05, 0.05, 0.05),
        "gamma": (0.92, 0.92, 0.94),
        "gain": (1.04, 1.04, 1.02),
        "bloom": 0.05,
        "sat": 0.62,
    },
}


def adelay_ms(speech_frame_start: int, master_fps: float = MASTER_FPS) -> int:
    s0 = int(speech_frame_start or 0)
    if s0 <= 0:
        return 0
    fps = max(1.0, float(master_fps))
    return int(round((s0 - 1) / fps * 1000.0))


def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def mix_session_audio(
    clips: Sequence[Dict[str, Any]],
    video_path: Path | str,
    out_path: Path | str,
    *,
    master_fps: float = MASTER_FPS,
    picture_seconds: Optional[float] = None,
) -> Path:
    """
    Mix clip WAVs onto video. Delays from speech_frame_start at master_fps.
    No -shortest; pad audio to picture duration.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[movie_render] ffmpeg missing — copy video only")
        try:
            shutil.copy2(video_path, out_path)
        except Exception:
            return video_path
        return out_path

    audios: List[Tuple[Path, int]] = []
    fps = max(1.0, float(master_fps))
    for c in clips or []:
        ap = str((c or {}).get("audio_path") or "")
        if not ap or not Path(ap).is_file():
            continue
        s0 = int((c or {}).get("speech_frame_start") or 0)
        if s0 <= 0:
            continue
        audios.append((Path(ap), s0))

    if not audios:
        shutil.copy2(video_path, out_path)
        return out_path

    args = [ffmpeg, "-y", "-i", str(video_path)]
    filt = []
    amix = []
    for i, (wav, s0) in enumerate(audios):
        args.extend(["-i", str(wav)])
        ms = adelay_ms(s0, fps)
        filt.append(
            f"[{i+1}:a]adelay={ms}|{ms},aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]"
        )
        amix.append(f"[a{i}]")
    filt.append(f"{''.join(amix)}amix=inputs={len(audios)}:normalize=0,apad[aout]")
    dur = picture_seconds
    if dur is None or dur <= 0:
        # fall back: last clip end / master fps
        try:
            f1 = max(int(c.get("frame_end") or 1) for c in (clips or [{}]))
            dur = max(0.1, f1 / fps)
        except Exception:
            dur = None
    args.extend([
        "-filter_complex", ";".join(filt),
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
    ])
    if dur:
        args.extend(["-t", f"{float(dur):.4f}"])
    args.append(str(out_path))
    subprocess.run(args, check=False, capture_output=True)
    if out_path.is_file():
        return out_path
    return video_path


def conform_fps(src: Path | str, dest: Path | str, fps: int = PICTURE_FPS) -> Path:
    src, dest = Path(src), Path(dest)
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        shutil.copy2(src, dest)
        return dest
    subprocess.run(
        [ffmpeg, "-y", "-i", str(src), "-filter:v", f"fps={int(fps)}", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-an", str(dest)],
        check=False,
        capture_output=True,
    )
    return dest if dest.is_file() else src


def set_eevee_engine(scene) -> str:
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = eng
            return str(scene.render.engine)
        except Exception:
            continue
    raise RuntimeError("no EEVEE engine on this Blender")


def enable_camera_dof(cam, shot: str = "MS") -> None:
    if cam is None or getattr(cam, "type", "") != "CAMERA":
        return
    data = cam.data
    data.dof.use_dof = True
    look = None
    name = cam.name or ""
    if name.endswith("_LookAt"):
        look_name = name
    else:
        look_name = name + "_LookAt"
    try:
        import bpy
        look = bpy.data.objects.get(look_name)
    except Exception:
        look = None
    if look is not None:
        data.dof.focus_object = look
    sh = (shot or "MS").upper()
    fstop = 2.8 if sh in ("CU", "ECU") else 5.6 if sh in ("MS", "MCU") else 8.0
    data.dof.aperture_fstop = float(fstop)


def apply_compositor_grade(scene, grade: str = "neutral") -> str:
    """
    v1 look finish: ColorBalance lift/gamma/gain + mild glare bloom.
    none/neutral = identity graph (timing/lips unchanged). Neural restyle is not this.
    """
    g = str(grade or "neutral").strip().lower()
    if g not in GRADE_PRESETS:
        g = "neutral"
    preset = GRADE_PRESETS.get(g)
    try:
        scene.use_nodes = True
        scene.render.use_compositing = True
    except Exception:
        pass
    tree = getattr(scene, "node_tree", None)
    if tree is None:
        print(f"[movie_render] no compositor tree; skip grade={g}")
        return g
    nodes, links = tree.nodes, tree.links

    def _get(name: str, bl_idname: str, loc=(0, 0)):
        n = nodes.get(name)
        if n is None:
            n = nodes.new(bl_idname)
            n.name = name
            n.label = name
        try:
            n.location = loc
        except Exception:
            pass
        return n

    rl = None
    for n in nodes:
        if n.bl_idname in ("CompositorNodeRLayers", "CompositorNodeRLayers"):
            rl = n
            break
    if rl is None:
        rl = _get("MovieGrade_RLayers", "CompositorNodeRLayers", (-600, 0))
    comp = nodes.get("MovieGrade_Composite")
    if comp is None:
        for n in nodes:
            if n.bl_idname == "CompositorNodeComposite":
                comp = n
                break
    if comp is None:
        comp = _get("MovieGrade_Composite", "CompositorNodeComposite", (700, 0))

    bal = _get("MovieGrade_Balance", "CompositorNodeColorBalance", (-80, 40))
    glare = _get("MovieGrade_Glare", "CompositorNodeGlare", (220, 40))
    hue = _get("MovieGrade_Hue", "CompositorNodeHueSat", (420, 40))

    lift = (0.0, 0.0, 0.0)
    gamma = (1.0, 1.0, 1.0)
    gain = (1.0, 1.0, 1.0)
    bloom = 0.0
    sat = 1.0
    if preset:
        lift = tuple(preset.get("lift") or lift)
        gamma = tuple(preset.get("gamma") or gamma)
        gain = tuple(preset.get("gain") or gain)
        bloom = float(preset.get("bloom") or 0.0)
        sat = float(preset.get("sat") or 1.0)

    try:
        bal.correction_method = "LIFT_GAMMA_GAIN"
    except Exception:
        pass
    for attr, val in (("lift", lift), ("gamma", gamma), ("gain", gain)):
        try:
            col = getattr(bal, attr)
            col[0], col[1], col[2] = float(val[0]), float(val[1]), float(val[2])
        except Exception:
            try:
                inp = bal.inputs.get(attr.title()) or bal.inputs.get(attr)
                if inp is not None:
                    inp.default_value[0] = float(val[0])
                    inp.default_value[1] = float(val[1])
                    inp.default_value[2] = float(val[2])
            except Exception:
                pass

    try:
        glare.glare_type = "FOG_GLOW"
    except Exception:
        try:
            glare.glare_type = "FOG_GLOW"
        except Exception:
            pass
    try:
        glare.mix = float(max(-1.0, min(1.0, bloom - 1.0))) if bloom <= 0 else float(min(0.5, bloom) - 1.0)
        # mix=-1 is original; mix closer to 0 adds glow. Identity: mix=-1
        glare.mix = -1.0 if bloom <= 0.001 else float(-1.0 + min(0.85, bloom * 2.5))
        glare.threshold = 0.85 if bloom > 0.001 else 10.0
        glare.size = 7
    except Exception:
        pass
    try:
        hue.inputs["Saturation"].default_value = float(sat)
        hue.inputs["Fac"].default_value = 0.0 if abs(sat - 1.0) < 1e-3 else 1.0
    except Exception:
        pass

    def _link(a, a_sock, b, b_sock):
        try:
            sa = a.outputs.get(a_sock) or (a.outputs[0] if a.outputs else None)
            sb = b.inputs.get(b_sock) or (b.inputs[0] if b.inputs else None)
            if sa is None or sb is None:
                return
            for ln in list(sb.links):
                links.remove(ln)
            links.new(sa, sb)
        except Exception:
            pass

    img = "Image"
    _link(rl, img, bal, img)
    _link(bal, img, glare, img)
    _link(glare, img, hue, img)
    _link(hue, img, comp, img)
    print(f"[movie_render] compositor grade={g} bloom={bloom:.2f} sat={sat:.2f}")
    return g


def configure_movie_render(scene, *, evaluate_fps: float = MASTER_FPS, grade: str = "neutral") -> str:
    """1080p, master fps, EEVEE, motion blur, compositor grade. Returns engine id actually set."""
    eng = set_eevee_engine(scene)
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = int(round(float(evaluate_fps)))
    scene.render.fps_base = 1.0
    scene.render.image_settings.file_format = "FFMPEG"
    ff = scene.render.ffmpeg
    ff.format = "MPEG4"
    try:
        ff.codec = "H264"
    except Exception:
        pass
    try:
        scene.render.film_transparent = False
    except Exception:
        pass
    ee = getattr(scene, "eevee", None)
    if ee is not None:
        if hasattr(ee, "use_motion_blur"):
            ee.use_motion_blur = True
        if hasattr(ee, "motion_blur_shutter"):
            ee.motion_blur_shutter = 0.5
        elif hasattr(scene.render, "motion_blur_shutter"):
            scene.render.use_motion_blur = True
            scene.render.motion_blur_shutter = 0.5
        if hasattr(ee, "use_shadows"):
            ee.use_shadows = True
        if hasattr(ee, "taa_render_samples"):
            ee.taa_render_samples = 32
    try:
        scene.view_settings.view_transform = "AgX"
    except Exception:
        try:
            scene.view_settings.view_transform = "Filmic"
        except Exception:
            pass
    try:
        apply_compositor_grade(scene, grade)
    except Exception as e:
        print(f"[movie_render] grade skip: {e}")
    print(f"[movie_render] engine={eng} evaluate_fps={int(evaluate_fps)} 1920x1080 grade={grade}")
    return eng
