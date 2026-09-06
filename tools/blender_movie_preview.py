"""
blender_movie_preview.py — viewport/OpenGL mesh preview of a movie package.

Run via:
  blender production.blend --background --python tools/blender_movie_preview.py -- \\
      --package temp/movies/.../movie_package.json --out temp/movies/.../preview.mp4

- Applies face_timeline keyframes (lips/expression)
- Applies camera keyframes
- Evaluates body Actions per shot (catalog/momask library)
- Places audio on VSE for mux
- Renders Workbench/OpenGL animation (mesh preview, not Cycles)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _argv_after_dd():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return []


def _parse_args(argv):
    pkg = None
    out = None
    fps = 30
    res_x, res_y = 960, 540
    frames_dir = None
    frame_start = None
    frame_end = None
    clip_id = None
    frame_step = 1
    still_format = "JPEG"  # JPEG much faster than PNG for analysis
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--package" and i + 1 < len(argv):
            pkg = argv[i + 1]
            i += 2
        elif a == "--out" and i + 1 < len(argv):
            out = argv[i + 1]
            i += 2
        elif a == "--fps" and i + 1 < len(argv):
            fps = int(float(argv[i + 1]))
            i += 2
        elif a == "--res" and i + 1 < len(argv):
            parts = argv[i + 1].lower().split("x")
            res_x, res_y = int(parts[0]), int(parts[1])
            i += 2
        elif a == "--frames-dir" and i + 1 < len(argv):
            frames_dir = argv[i + 1]
            i += 2
        elif a == "--frame-start" and i + 1 < len(argv):
            frame_start = int(argv[i + 1])
            i += 2
        elif a == "--frame-end" and i + 1 < len(argv):
            frame_end = int(argv[i + 1])
            i += 2
        elif a == "--clip-id" and i + 1 < len(argv):
            clip_id = argv[i + 1]
            i += 2
        elif a == "--frame-step" and i + 1 < len(argv):
            frame_step = max(1, int(argv[i + 1]))
            i += 2
        elif a == "--still-format" and i + 1 < len(argv):
            still_format = str(argv[i + 1]).upper()
            i += 2
        else:
            i += 1
    return (
        pkg, out, fps, res_x, res_y, frames_dir, frame_start, frame_end,
        clip_id, frame_step, still_format,
    )


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_movie_cam():
    scene = bpy.context.scene
    cam = bpy.data.objects.get("MovieCam")
    if cam is None or cam.type != "CAMERA":
        data = bpy.data.cameras.new("MovieCam")
        cam = bpy.data.objects.new("MovieCam", data)
        scene.collection.objects.link(cam)
    look = bpy.data.objects.get("MovieCam_LookAt")
    if look is None:
        look = bpy.data.objects.new("MovieCam_LookAt", None)
        look.empty_display_type = "PLAIN_AXES"
        scene.collection.objects.link(look)
        look.location = (0, 0, 1.48)
    scene.camera = cam
    return cam, look


def _look_at_rot(cam_loc, target):
    direction = Vector(target) - Vector(cam_loc)
    if direction.length < 1e-6:
        return (math.radians(90), 0, 0)
    return direction.to_track_quat("-Z", "Y").to_euler()


def _fov_to_lens(fov_deg, sensor=36.0):
    fov = max(5.0, min(120.0, float(fov_deg)))
    return float(sensor) / (2.0 * math.tan(math.radians(fov) * 0.5))


def _resolve_shape_key_name(mp_name, key_blocks):
    if mp_name in key_blocks:
        return mp_name
    low = mp_name.lower()
    for kb in key_blocks:
        if kb.name.lower() == low:
            return kb.name
    # Faceit-style
    for kb in key_blocks:
        if kb.name.lower().replace("_", "") == low.replace("_", ""):
            return kb.name
    return None


def _clear_shape_anim(sk):
    if sk is None:
        return
    try:
        if sk.animation_data:
            sk.animation_data_clear()
    except Exception:
        pass


def _apply_face_timeline(path: Path, mesh_names=("head_lod0_ORIGINAL",)):
    if not path.is_file():
        print(f"[preview] face timeline missing: {path}")
        return
    track = _load_json(path)
    kfs = track.get("keyframes") or []
    if not kfs:
        return
    obj = None
    for name in mesh_names:
        o = bpy.data.objects.get(name)
        if o and o.type == "MESH" and o.data and o.data.shape_keys:
            obj = o
            break
    if obj is None:
        # auto pick
        for o in bpy.data.objects:
            if o.type == "MESH" and o.data and o.data.shape_keys:
                n = len(o.data.shape_keys.key_blocks)
                if n > 20:
                    obj = o
                    break
    if obj is None:
        print("[preview] no face mesh with shape keys")
        return

    sk = obj.data.shape_keys
    _clear_shape_anim(sk)
    kb = sk.key_blocks
    n = 0
    for kf in kfs:
        fr = int(kf.get("frame") or 1)
        for mp, val in (kf.get("blendshapes") or {}).items():
            name = _resolve_shape_key_name(mp, kb)
            if not name:
                continue
            try:
                kb[name].value = float(val)
                kb[name].keyframe_insert(data_path="value", frame=fr)
                n += 1
            except Exception:
                pass
        # secondary
        for sec_name in ("eyeLeft_ORIGINAL", "eyeRight_ORIGINAL", "teeth_ORIGINAL"):
            sec = bpy.data.objects.get(sec_name)
            if not sec or not sec.data or not sec.data.shape_keys:
                continue
            skb = sec.data.shape_keys.key_blocks
            for mp, val in (kf.get("blendshapes") or {}).items():
                name = _resolve_shape_key_name(mp, skb)
                if not name:
                    continue
                try:
                    skb[name].value = float(val)
                    skb[name].keyframe_insert(data_path="value", frame=fr)
                except Exception:
                    pass
    print(f"[preview] face keyframes on {obj.name}: ~{n} inserts, {len(kfs)} frames")


def _subject_anchor_world(anchor: str = "chest"):
    """Live head/chest/pelvis world point for framing (not static origin)."""
    arm = bpy.data.objects.get("SMPL-X_Armature")
    if arm is None:
        for o in bpy.data.objects:
            if o.type == "ARMATURE":
                arm = o
                break
    if arm is None:
        return None
    bone_pref = {
        "head": ("head", "neck", "spine3"),
        "chest": ("spine3", "spine2", "spine1", "head"),
        "pelvis": ("pelvis", "spine1"),
        "full_body": ("pelvis", "spine2"),
    }.get((anchor or "chest").lower(), ("spine3", "head", "pelvis"))
    for bn in bone_pref:
        pb = arm.pose.bones.get(bn)
        if pb is not None:
            return (arm.matrix_world @ pb.head).copy()
    return arm.matrix_world.translation.copy()


def _apply_camera_plan(cam, look, camera_dict, fps: float, store_rel=None):
    """
    Store *relative* cam offsets when possible so preview can track the live
    character (face/chest). Absolute baked keys alone frame the wrong height
    when the body root moves or jumps.
    """
    if not camera_dict:
        return
    kfs = camera_dict.get("keyframes") or []
    if store_rel is not None:
        for kf in kfs:
            # Prefer explicit relative offsets from package
            rc = kf.get("rel_cam") or kf.get("location")
            rl = kf.get("rel_look") or kf.get("look_at")
            fr = kf.get("frame_abs")
            if fr is None and kf.get("t_abs") is not None:
                fr = int(round(float(kf["t_abs"]) * fps)) + 1
            if fr is None:
                fr = int(kf.get("frame") or 1)
            store_rel.append({
                "frame": int(fr),
                "rel_cam": list(rc or [0.0, -2.4, 0.35]),
                "rel_look": list(rl or [0.0, 0.0, 0.25]),
                "fov": float(kf.get("fov_deg") or 45),
                "anchor": str(camera_dict.get("subject_anchor") or "chest"),
            })
        print(f"[preview] camera plan stored rel keys={len(kfs)} (live subject follow)")
        return

    if cam.animation_data:
        try:
            cam.animation_data_clear()
        except Exception:
            pass
    if look.animation_data:
        try:
            look.animation_data_clear()
        except Exception:
            pass
    for kf in kfs:
        if "frame_abs" in kf:
            fr = int(kf["frame_abs"])
        elif "t_abs" in kf:
            fr = int(round(float(kf["t_abs"]) * fps)) + 1
        else:
            fr = int(kf.get("frame") or 1)
        loc = list(kf.get("location") or [0, -2.4, 1.48])
        la = list(kf.get("look_at") or [0, 0, 1.55])
        # Bias look toward head if look_at is near ground (hips framing bug)
        if la[2] < 1.0:
            la[2] = 1.55
        fov = float(kf.get("fov_deg") or 45)
        look.location = Vector(la)
        look.keyframe_insert(data_path="location", frame=fr)
        cam.location = Vector(loc)
        cam.rotation_euler = _look_at_rot(loc, la)
        cam.keyframe_insert(data_path="location", frame=fr)
        cam.keyframe_insert(data_path="rotation_euler", frame=fr)
        if cam.data:
            cam.data.lens = _fov_to_lens(fov)
            cam.data.keyframe_insert(data_path="lens", frame=fr)
    print(f"[preview] camera keyframes: {len(kfs)}")


def _setup_live_camera_follow(cam, look, rel_keys, fps: float):
    """Each frame: cam/look = live chest/head + relative offset (tracks jumps safely)."""
    if not rel_keys:
        return
    # Sort by frame for lerp
    keys = sorted(rel_keys, key=lambda k: int(k["frame"]))

    def _lerp_key(f: int):
        if f <= keys[0]["frame"]:
            return keys[0]
        if f >= keys[-1]["frame"]:
            return keys[-1]
        for i in range(len(keys) - 1):
            a, b = keys[i], keys[i + 1]
            if a["frame"] <= f <= b["frame"]:
                span = max(1, b["frame"] - a["frame"])
                u = (f - a["frame"]) / span
                rc = [
                    a["rel_cam"][j] * (1 - u) + b["rel_cam"][j] * u for j in range(3)
                ]
                rl = [
                    a["rel_look"][j] * (1 - u) + b["rel_look"][j] * u for j in range(3)
                ]
                # Prefer chest/head bias on look Z
                if rl[2] < 0.1:
                    rl[2] = 0.25
                return {
                    "rel_cam": rc,
                    "rel_look": rl,
                    "fov": a["fov"] * (1 - u) + b["fov"] * u,
                    "anchor": a.get("anchor") or "chest",
                }
        return keys[-1]

    def _on_cam(scene, depsgraph=None):
        f = int(scene.frame_current)
        k = _lerp_key(f)
        anchor = k.get("anchor") or "chest"
        # Head for CU-ish, chest default — never pelvis-only for face framing
        sub = _subject_anchor_world(anchor if anchor != "pelvis" else "chest")
        if sub is None:
            sub = Vector((0.0, 0.0, 1.5))
        rc, rl = k["rel_cam"], k["rel_look"]
        # If offset was stored as absolute world (large Z), convert to offset from sub
        if abs(rc[2]) > 0.8 and abs(rc[1]) > 1.0:
            # likely absolute cam from bake — re-anchor: keep X/Y offset from sub, face height
            loc = Vector((sub.x + 0.0, sub.y - 2.6, sub.z + 0.15))
            la = Vector((sub.x, sub.y, sub.z + 0.12))
        else:
            loc = Vector((sub.x + rc[0], sub.y + rc[1], sub.z + rc[2]))
            la = Vector((sub.x + rl[0], sub.y + rl[1], sub.z + max(rl[2], 0.12)))
        # Keep cam in front of subject on -Y
        if loc.y > sub.y - 0.9:
            loc.y = sub.y - 2.4
        look.location = la
        cam.location = loc
        cam.rotation_euler = _look_at_rot(loc, la)
        if cam.data:
            cam.data.lens = _fov_to_lens(k.get("fov") or 45)

    to_remove = []
    for h in bpy.app.handlers.frame_change_pre:
        if getattr(h, "_movie_preview_cam", False):
            to_remove.append(h)
    for h in to_remove:
        bpy.app.handlers.frame_change_pre.remove(h)
    _on_cam._movie_preview_cam = True  # type: ignore
    bpy.app.handlers.frame_change_pre.append(_on_cam)
    print(f"[preview] live camera→subject follow ({len(keys)} keys, jump-safe)")


def _iter_action_fcurves(action):
    if action is None:
        return
    try:
        fcus = getattr(action, "fcurves", None)
        if fcus is not None and len(fcus) > 0:
            for fcu in fcus:
                yield fcu
            return
    except Exception:
        pass
    try:
        for layer in getattr(action, "layers", []) or []:
            for strip in getattr(layer, "strips", []) or []:
                bags = getattr(strip, "channelbags", None) or []
                for bag in bags:
                    for fcu in bag.fcurves:
                        yield fcu
    except Exception:
        return


def _apply_action_frame(arm, action, frame: float) -> bool:
    import re

    if arm is None or action is None:
        return False
    pat = re.compile(
        r'^pose\.bones\["([^"]+)"\]\.(location|rotation_quaternion|rotation_euler|scale)$'
    )
    ok = False
    for fcu in _iter_action_fcurves(action):
        m = pat.match(fcu.data_path or "")
        if not m:
            continue
        bname, prop = m.group(1), m.group(2)
        pb = arm.pose.bones.get(bname)
        if pb is None:
            continue
        try:
            val = float(fcu.evaluate(float(frame)))
        except Exception:
            continue
        idx = int(fcu.array_index)
        if prop == "location" and 0 <= idx < 3:
            pb.location[idx] = val
            ok = True
        elif prop == "rotation_quaternion":
            pb.rotation_mode = "QUATERNION"
            if 0 <= idx < 4:
                pb.rotation_quaternion[idx] = val
                ok = True
        elif prop == "rotation_euler":
            pb.rotation_mode = "XYZ"
            if 0 <= idx < 3:
                pb.rotation_euler[idx] = val
                ok = True
        elif prop == "scale" and 0 <= idx < 3:
            pb.scale[idx] = val
            ok = True
    return ok


def _try_load_action(name: str, library: str):
    if not name:
        return None
    act = bpy.data.actions.get(name)
    if act:
        return act
    if library and Path(library).is_file():
        try:
            with bpy.data.libraries.load(library, link=False) as (src, dst):
                names = [n for n in (src.actions or []) if n == name or name in n]
                dst.actions = names[:1]
            act = bpy.data.actions.get(name) or (
                bpy.data.actions.get(names[0]) if names else None
            )
            if act:
                act.use_fake_user = True
                print(f"[preview] loaded action {act.name} from {Path(library).name}")
                return act
        except Exception as e:
            print(f"[preview] library load fail: {e}")
    # catalog library fallbacks
    roots = [
        Path(bpy.data.filepath).parent if bpy.data.filepath else Path("."),
        Path(r"C:\me\proj\projface_v1"),
    ]
    for root in roots:
        for rel in (
            "whole_body_retargeted.blend",
            "body_motion/whole_body_with_clips.blend",
        ):
            p = root / rel
            if not p.is_file():
                continue
            try:
                with bpy.data.libraries.load(str(p), link=False) as (src, dst):
                    names = [n for n in (src.actions or []) if n == name or n.lower() == name.lower()]
                    if not names:
                        # common catalog names
                        names = [n for n in (src.actions or []) if name.lower() in n.lower()]
                    dst.actions = names[:1]
                for a in bpy.data.actions:
                    if a.name == name or a.name.lower() == name.lower():
                        a.use_fake_user = True
                        return a
            except Exception:
                continue
    return None


def _build_body_schedule(clips, fps: float):
    """
    Schedule like chat-mode body play — NOT time-stretched to fill the take.

    Chat (blender_receiver): MoMask Action plays at native ~20fps + optional speed.
    Old movie bug: mapped whole take u∈[0,1] → full action (stretch/slow walk).
    """
    schedule = []
    for c in clips:
        t0 = float(c.get("t0") or 0)
        t1 = float(c.get("t1") or t0)
        f0 = int(round(t0 * fps)) + 1
        f1 = max(f0 + 1, int(round(t1 * fps)) + 1)
        act = (c.get("body_action") or "").strip()
        if not act:
            act = "talk_open"
        body_mode = str(c.get("body_mode") or "").lower()
        eng = "momask" if (
            body_mode in ("momask", "both")
            or act.startswith("momask_")
        ) else "catalog"
        # Native clip fps: MoMask BVH/Action is 20fps; catalog often 30
        clip_fps = float(c.get("clip_fps") or (20.0 if eng == "momask" else float(fps)))
        policy = str(c.get("clip_policy") or ("momask_match" if eng == "momask" else "loop")).lower()
        speed = float(c.get("clip_speed") or 1.0)
        speed = max(0.35, min(1.6, speed))
        schedule.append({
            "f0": f0,
            "f1": f1,
            "action": act,
            "library": c.get("body_library") or "",
            "engine": eng,
            "clip_fps": clip_fps,
            "clip_policy": policy,
            "clip_speed": speed,
            "scene_fps": float(fps),
        })
    return schedule


def _local_frame_for_shot(s: dict, act, scene_frame: int) -> float:
    """
    Chat-parity local Action frame for a global scene frame.

    - momask_match / hold_end: play 1:1 in time (clip_fps * speed), then HOLD last
    - loop: loop at native rate (walk cycles)
    - stretch: only when director asked to fill take (rare)
    """
    try:
        fr = act.frame_range
        af0, af1 = float(fr[0]), float(fr[1])
    except Exception:
        af0, af1 = 1.0, 60.0
    if af1 <= af0:
        af1 = af0 + 1.0
    span = max(1.0, af1 - af0)
    policy = (s.get("clip_policy") or "hold_end").lower()
    scene_fps = float(s.get("scene_fps") or 30.0)
    clip_fps = float(s.get("clip_fps") or 20.0)
    speed = float(s.get("clip_speed") or 1.0)
    # seconds into this shot on the movie timeline
    t = max(0.0, (float(scene_frame) - float(s["f0"])) / max(1e-6, scene_fps))
    # native action frames advanced at clip rate (chat uses wall-clock × clip fps)
    local = af0 + t * clip_fps * speed

    if policy == "stretch":
        # Explicit stretch-to-fill (director only)
        u = (scene_frame - s["f0"]) / max(1, s["f1"] - s["f0"])
        u = max(0.0, min(1.0, u))
        return af0 + u * span

    if policy == "loop":
        # Loop gait without slowing the step cycle to fill the take
        return af0 + ((local - af0) % span)

    # hold_end / momask_match / default: natural speed, freeze on last pose
    if local >= af1:
        return af1
    if local < af0:
        return af0
    return local


def _setup_body_handler(schedule, fps: float):
    arm = bpy.data.objects.get("SMPL-X_Armature")
    if arm is None:
        for o in bpy.data.objects:
            if o.type == "ARMATURE":
                arm = o
                break
    if arm is None:
        print("[preview] no armature — body skipped")
        return

    # Preload actions
    cache = {}
    for s in schedule:
        name = s["action"]
        if name not in cache:
            cache[name] = _try_load_action(name, s.get("library") or "")
            if cache[name] is None and name != "talk_open":
                cache[name] = _try_load_action("talk_open", "")
            if cache[name] is None and name != "wave":
                cache[name] = _try_load_action("wave", "")

    # Clear action for evaluation via handler
    if arm.animation_data:
        arm.animation_data.action = None

    def _on_frame(scene, depsgraph=None):
        f = int(scene.frame_current)
        for s in schedule:
            if s["f0"] <= f <= s["f1"]:
                act = cache.get(s["action"])
                if act is None:
                    return
                local = _local_frame_for_shot(s, act, f)
                if arm.animation_data and arm.animation_data.action:
                    arm.animation_data.action = None
                _apply_action_frame(arm, act, local)
                return

    # remove old handlers with same name tag
    to_remove = []
    for h in bpy.app.handlers.frame_change_pre:
        if getattr(h, "_movie_preview_body", False):
            to_remove.append(h)
    for h in to_remove:
        bpy.app.handlers.frame_change_pre.remove(h)

    _on_frame._movie_preview_body = True  # type: ignore
    bpy.app.handlers.frame_change_pre.append(_on_frame)
    modes = [f"{s['action'][:16]}@{s.get('clip_policy')}/{s.get('clip_fps')}fps" for s in schedule]
    print(f"[preview] body schedule (chat-parity): {len(schedule)} shots on {arm.name}")
    print(f"[preview]   " + " | ".join(modes[:8]))


def _setup_vse_audio(clips, fps: float):
    scene = bpy.context.scene
    if not scene.sequence_editor:
        scene.sequence_editor_create()
    sed = scene.sequence_editor
    # Blender 5: strips / strips_all (not sequences_all)
    strips = getattr(sed, "strips", None) or getattr(sed, "sequences", None)
    all_strips = getattr(sed, "strips_all", None) or getattr(sed, "sequences_all", None) or strips
    if all_strips is not None:
        for s in list(all_strips):
            try:
                if hasattr(strips, "remove"):
                    strips.remove(s)
                elif hasattr(sed, "sequences") and hasattr(sed.sequences, "remove"):
                    sed.sequences.remove(s)
            except Exception:
                pass
    ch = 1
    for c in clips:
        wav = c.get("audio") or ""
        if not wav or not Path(wav).is_file():
            print(f"[preview] missing audio {wav}")
            continue
        t0 = float(c.get("t0") or 0)
        frame_start = max(1, int(round(t0 * fps)) + 1)
        try:
            if strips is not None and hasattr(strips, "new_sound"):
                strips.new_sound(
                    str(c.get("shot_id") or "snd")[:60],
                    wav,
                    ch,
                    frame_start,
                )
                print(f"[preview] audio {Path(wav).name} @ f{frame_start}")
            else:
                print(f"[preview] VSE new_sound unavailable; audio muxed after encode")
        except Exception as e:
            print(f"[preview] VSE audio fail: {e} (audio still muxed by export script)")
        ch = 1


def _render_direct_ffmpeg(
    out_path: Path,
    *,
    fps: int,
    res_x: int,
    res_y: int,
    f0: int,
    f1: int,
    frame_step: int = 1,
) -> bool:
    """
    One-shot Workbench → H.264 MP4 via Blender's FFMPEG writer.
    No per-frame JPEG dump (major speed win). Handlers still drive body+cam.
    """
    scene = bpy.context.scene
    step = max(1, int(frame_step))
    scene.frame_start = int(f0)
    scene.frame_end = int(f1)
    scene.frame_step = step
    scene.render.fps = int(fps)
    scene.render.resolution_x = int(res_x)
    scene.render.resolution_y = int(res_y)
    scene.render.resolution_percentage = 100
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.shading.show_shadows = False
        if hasattr(scene.display, "render_aa"):
            scene.display.render_aa = "OFF"
    except Exception:
        pass
    _hide_heavy_preview_objects()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer writing a video-only file; export script muxes audio
    vid_path = out_path.parent / (out_path.stem + "_vid.mp4")
    fp = str(vid_path.resolve()).replace("\\", "/")
    try:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        try:
            scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        except Exception:
            pass
        try:
            scene.render.ffmpeg.ffmpeg_preset = "REALTIME"
        except Exception:
            try:
                scene.render.ffmpeg.ffmpeg_preset = "GOOD"
            except Exception:
                pass
        scene.render.ffmpeg.audio_codec = "NONE"
        scene.render.filepath = fp
    except Exception as e:
        print(f"[preview] FFMPEG setup failed: {e}")
        return False

    n_est = max(1, (int(f1) - int(f0)) // step + 1)
    print(
        f"[preview] DIRECT Workbench→FFMPEG {f0}-{f1} step={step} "
        f"(~{n_est} frames) {res_x}x{res_y} → {vid_path.name}"
    )
    try:
        bpy.ops.render.render(animation=True, write_still=False)
    except Exception as e:
        print(f"[preview] animation render failed: {e}")
        return False
    # Blender may append extension
    candidates = [
        vid_path,
        Path(fp + ".mp4"),
        out_path.parent / (out_path.stem + "_vid.mp4"),
        out_path.parent / (out_path.stem + "_vid.ffmpeg"),
    ]
    found = next((p for p in candidates if p.is_file() and p.stat().st_size > 1000), None)
    if found is None:
        # any new mp4 in folder
        mp4s = sorted(out_path.parent.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        found = mp4s[0] if mp4s else None
    if found is None:
        print("[preview] no FFMPEG output file found")
        return False
    # Normalize name for export_movie_preview_mp4 mux path
    try:
        if found.resolve() != vid_path.resolve():
            import shutil

            shutil.copy2(found, vid_path)
    except Exception:
        pass
    print(f"[preview] wrote direct video {found} ({found.stat().st_size/1024:.0f} KB)")
    # Write marker so export script can mux without stills
    marker = out_path.parent / "preview_direct_video.txt"
    marker.write_text(str(found.resolve()), encoding="utf-8")
    return True


def _apply_feature_environment(look: dict) -> None:
    """
    Build Look_Set + wardrobe + cast extras inside the preview Blender process.
    Required so comparison MP4s are not a naked character on a void stage.
    """
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import importlib.util

    br_path = root / "blender_receiver.py"
    spec = importlib.util.spec_from_file_location("blender_receiver_preview", br_path)
    br = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(br)
    plan = dict(look or {})
    plan.setdefault("location", "street")
    plan.setdefault("time_of_day", "day")
    plan.setdefault("set_preset", "exterior_street")
    plan.setdefault("wardrobe_id", "casual_01")
    plan.setdefault("extras_count", 3)
    plan.setdefault("extras_preset", "sidewalk")
    print(
        f"[preview] FEATURE SET look={plan.get('location')} "
        f"wardrobe={plan.get('wardrobe_id')} extras={plan.get('extras_count')}"
    )
    br._apply_look_packet({"op": "apply", "look": plan})
    # Force visibility of set / clothes / NPCs for render
    for obj in bpy.data.objects:
        n = obj.name
        if (
            n.startswith(("Look_", "NPC_", "Cloth_", "Cast_"))
            or n.startswith("Wardrobe")
            or "Wardrobe_" in n
        ):
            try:
                obj.hide_set(False)
                obj.hide_viewport = False
                obj.hide_render = False
            except Exception:
                pass
    print("[preview] feature environment applied (Look_Set + wardrobe + NPCs)")


def _hide_heavy_preview_objects():
    """Hide non-essential high-poly junk for faster Workbench playblast."""
    hide_sub = (
        "hair", "eyelash", "brow", "particle", "proxy", "collision",
        "shadowcatcher", "lightprobe",
    )
    # NEVER hide product features needed for comparison demos
    keep_tokens = (
        "look_", "npc_", "cloth_", "wardrobe", "cast_", "smpl", "bodymesh",
        "head_", "moviecam", "facade", "curb", "ground", "asphalt",
    )
    n = 0
    for obj in bpy.data.objects:
        try:
            low = obj.name.lower()
            if any(tok in low for tok in keep_tokens):
                obj.hide_render = False
                continue
            if any(s in low for s in hide_sub) and obj.type in ("MESH", "CURVE", "VOLUME"):
                obj.hide_render = True
                obj.hide_viewport = True
                n += 1
        except Exception:
            pass
    if n:
        print(f"[preview] hid {n} heavy objects for speed")


def _configure_render_stills(
    still_dir: Path,
    fps: int,
    res_x: int,
    res_y: int,
    f0: int,
    f1: int,
    still_format: str = "JPEG",
):
    """Workbench mesh frames as JPEG (fast playblast — not Cycles/EEVEE final)."""
    scene = bpy.context.scene
    scene.frame_start = f0
    scene.frame_end = f1
    scene.render.fps = fps
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.shading.show_shadows = False
        scene.display.shading.show_cavity = False
        # Cheapest AA for analysis previews
        if hasattr(scene.display, "render_aa"):
            scene.display.render_aa = "OFF"
        if hasattr(scene.display, "viewport_aa"):
            scene.display.viewport_aa = "OFF"
    except Exception:
        pass
    # Simplify Workbench
    try:
        wb = scene.display
        if hasattr(wb, "shading"):
            pass
    except Exception:
        pass
    still_dir.mkdir(parents=True, exist_ok=True)
    fmt = (still_format or "JPEG").upper()
    if fmt not in ("JPEG", "JPG", "PNG"):
        fmt = "JPEG"
    if fmt in ("JPEG", "JPG"):
        scene.render.image_settings.file_format = "JPEG"
        try:
            scene.render.image_settings.quality = 72  # smaller/faster I/O
        except Exception:
            pass
    else:
        scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(still_dir / "f")
    _hide_heavy_preview_objects()
    return scene


def main():
    argv = _argv_after_dd()
    (
        pkg_path, out_path, fps, res_x, res_y,
        frames_dir_arg, frame_start, frame_end, clip_id,
        frame_step, still_format,
    ) = _parse_args(argv)
    if not pkg_path:
        print("Usage: --package movie_package.json --out preview.mp4")
        return 1
    pkg_path = Path(pkg_path)
    if not pkg_path.is_file():
        print(f"[preview] package not found: {pkg_path}")
        return 1

    data = _load_json(pkg_path)
    if "clips" in data:
        clips = data["clips"]
        fps = int(float(data.get("fps") or fps))
    else:
        shots = data.get("shots") or []
        clips = []
        for s in shots:
            acts = [str(a).lower() for a in (s.get("actions") or [])]
            body_act = (s.get("body_action") or "").strip()
            if not body_act:
                for prefer in ("wave", "talk_open", "celebrate", "shrug", "walk", "run", "idle"):
                    if prefer in acts:
                        body_act = prefer
                        break
                body_act = body_act or "talk_open"
            bm = str(s.get("body_mode") or "").lower()
            clips.append({
                "shot_id": s.get("shot_id"),
                "t0": s.get("t0"),
                "t1": s.get("t1"),
                "duration_s": s.get("duration_s"),
                "audio": s.get("audio_path") or s.get("audio"),
                "face_timeline": s.get("face_timeline_path") or s.get("face_timeline"),
                "body_action": body_act,
                "body_library": s.get("body_library"),
                "body_mode": bm or s.get("body_mode") or "",
                "clip_policy": s.get("clip_policy") or (
                    "momask_match" if (bm in ("momask", "both") or body_act.startswith("momask_")) else "loop"
                ),
                "clip_speed": s.get("clip_speed") if s.get("clip_speed") is not None else 1.0,
                "clip_fps": s.get("clip_fps") or (
                    20.0 if (bm in ("momask", "both") or body_act.startswith("momask_")) else 30.0
                ),
                "camera": s.get("camera"),
            })
        fps = int(float(data.get("fps") or fps))

    if clip_id:
        clips = [c for c in clips if str(c.get("shot_id")) == str(clip_id)] or clips

    if not out_path:
        out_path = str(pkg_path.parent / "preview_mesh.mp4")
    out_path = Path(out_path)

    duration = 0.0
    for c in clips:
        duration = max(duration, float(c.get("t1") or 0))
    f0 = int(frame_start) if frame_start is not None else 1
    f1 = int(frame_end) if frame_end is not None else max(2, int(round(duration * fps)) + 1)
    if f1 < f0:
        f1 = f0 + 1
    print(f"[preview] clips={len(clips)} duration={duration:.2f}s frames={f0}-{f1} @ {fps}fps")

    # --- PRODUCT FEATURES: rebuild Look_Set + wardrobe + NPCs in THIS blend ---
    # Background preview used to open a bare .blend (no street/clothes/crowd).
    look_plan = {}
    if isinstance(data.get("look"), dict):
        look_plan = dict(data.get("look") or {})
    if not look_plan:
        # Defaults for feature comparison demos
        look_plan = {
            "location": "street",
            "time_of_day": "day",
            "set_preset": "exterior_street",
            "wardrobe_id": "casual_01",
            "extras_count": 3,
            "extras_preset": "sidewalk",
        }
    try:
        _apply_feature_environment(look_plan)
    except Exception as e:
        print(f"[preview] feature environment FAILED: {e}")

    cam, look = _ensure_movie_cam()
    # Live subject-relative camera (tracks head/chest through walks AND jumps)
    cam_rel_keys: list = []
    for c in clips:
        ft = c.get("face_timeline") or ""
        if ft:
            _apply_face_timeline(Path(ft))
        if c.get("camera"):
            _apply_camera_plan(cam, look, c["camera"], fps, store_rel=cam_rel_keys)

    schedule = _build_body_schedule(clips if not clip_id else clips, fps)
    _setup_body_handler(schedule, fps)
    if cam_rel_keys:
        _setup_live_camera_follow(cam, look, cam_rel_keys, fps)
    try:
        _setup_vse_audio(clips, fps)
    except Exception as e:
        print(f"[preview] VSE audio skipped: {e}")

    still_dir = Path(frames_dir_arg) if frames_dir_arg else (out_path.parent / "preview_frames")
    step = max(1, int(frame_step or 1))
    out_fps = max(1.0, float(fps) / float(step))

    # Direct FFMPEG animation (default) — one stream to MP4, no thousands of stills
    use_direct = os.environ.get("PREVIEW_STILLS", "0").strip().lower() not in (
        "1", "true", "yes", "on",
    )
    meta = {
        "fps": fps,
        "preview_fps": out_fps,
        "frame_step": step,
        "still_format": still_format,
        "f0": f0,
        "f1": f1,
        "out_mp4": str(out_path),
        "mode": "direct_ffmpeg" if use_direct else "stills",
        "camera": "live_subject_follow",
        "clip_id": clip_id,
    }
    (out_path.parent / "preview_render_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    if use_direct:
        ok = _render_direct_ffmpeg(
            out_path, fps=fps, res_x=res_x, res_y=res_y, f0=f0, f1=f1, frame_step=step
        )
        if ok:
            print(f"[preview] DIRECT FFMPEG done → {out_path}")
            return 0
        print("[preview] direct FFMPEG failed — falling back to stills")

    scene = _configure_render_stills(still_dir, fps, res_x, res_y, f0, f1, still_format)
    frame_list = list(range(f0, f1 + 1, step))
    if frame_list[-1] != f1:
        frame_list.append(f1)
    print(
        f"[preview] Workbench stills {f0}-{f1} step={step} → {len(frame_list)} "
        f"@ ~{out_fps:.1f}fps  {res_x}x{res_y}"
    )
    seq = 1
    total = len(frame_list)
    for f in frame_list:
        scene.frame_set(f)
        bpy.context.view_layer.update()
        scene.render.filepath = str(still_dir / f"f{seq:05d}")
        bpy.ops.render.render(write_still=True)
        if seq == 1 or seq == total or seq % 40 == 0:
            print(f"  frame {seq}/{total} (scene f{f})")
        seq += 1
    n_still = len(list(still_dir.glob("f*.*")))
    print(f"[preview] wrote {n_still} stills (step={step})")
    return 0 if n_still > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main() or 0)
