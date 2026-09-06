"""
Stages 4–6 — Resolve absolute times + bake workers per shot (mavie.txt).

Honors director decisions:
  - target_duration_s / hold_before / hold_after (pad take beyond speech)
  - camera_role / size / move / pace
  - clip_policy for body loop vs hold
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import soundfile as sf

from ..camera_agent import plan_camera_for_beat
from ..coordinator import FaceCoordinator
from .continuity_memory import ContinuityBoard
from .shot_schema import ShotDetail
from .stage_speech import clean_spoken_for_tts, split_stage_and_spoken
from .world_state import WorldState

ROOT = Path(__file__).resolve().parents[2]


def _resolve_body_action(
    *,
    body_mode: str,
    humanml_prompt: str,
    body_state: str,
    actions: list,
    emotion: str,
    text: str,
    duration_s: float,
    seed: int,
    allow_sync_gen: bool,
) -> dict | None:
    try:
        from face_agents.momask_body_pipeline import (
            build_humanml_prompt,
            ensure_action_in_blender_via_packet_hint,
            generate_body_action,
            lookup_cached_action,
            momask_sync_generate,
            should_use_momask,
        )
    except Exception as e:
        print(f"  [body] import failed: {e}")
        return None

    if not should_use_momask(
        body_mode=body_mode,
        humanml_prompt=humanml_prompt,
        state=body_state,
        actions=actions,
    ):
        return None

    prompt = build_humanml_prompt(
        humanml_prompt=humanml_prompt,
        state=body_state,
        actions=actions,
        emotion=emotion,
        text=text,
    )
    # Same caption as a previous shot / session → reuse slim Action (no gen).
    # Seed is ignored for hits so shot_index does not bust the memory.
    cached = lookup_cached_action(prompt, duration_s=duration_s, seed=seed)
    if cached and cached.ok:
        print(
            f"  [momask] CACHE HIT (prompt already generated) → {cached.action_name} "
            f"— skip gen_t2m/retarget"
        )
        return ensure_action_in_blender_via_packet_hint(cached)

    if not momask_sync_generate(allow_sync_gen=allow_sync_gen):
        print(f"  [momask] no cache + sync=0 → catalog fallback  prompt={prompt[:60]!r}")
        return None

    print(f"  [momask] CACHE MISS → generate once: {prompt!r}")
    t0 = time.time()
    # Stable seed from caption so re-runs write the same cache slot
    import hashlib

    stable = int(hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8], 16) % 100_000
    result = generate_body_action(
        prompt, duration_s=duration_s, seed=stable, use_cache=True
    )
    if not result.ok:
        print(f"  [momask] FAIL: {result.error}")
        return None
    tag = "cache_hit" if result.cached else "generated"
    print(f"  [momask] OK ({tag}) {result.action_name} in {time.time()-t0:.1f}s")
    return ensure_action_in_blender_via_packet_hint(result)


def _tts(text: str, emotion: str, intensity: float, wav_path: Path) -> Tuple[np.ndarray, int, float]:
    from parler_voice import build_voice_style, generate_speech, load_parler

    load_parler()
    style = build_voice_style(emotion, intensity)
    generate_speech(text, style, str(wav_path), play_audio=False)
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    dur = float(len(audio) / float(sr))
    return np.asarray(audio, dtype=np.float32), int(sr), dur


def _pad_audio(
    audio: np.ndarray,
    sr: int,
    *,
    hold_before_s: float,
    hold_after_s: float,
    target_duration_s: float,
    speech_dur: float,
) -> Tuple[np.ndarray, float]:
    """
    Director timing: pad silence before/after speech so take length matches
    target_duration when speech is shorter (cinematic breathing room).
    """
    before = max(0.0, float(hold_before_s or 0.0))
    after = max(0.0, float(hold_after_s or 0.0))
    target = float(target_duration_s or 0.0)

    n_before = int(round(before * sr))
    n_after = int(round(after * sr))
    core = float(speech_dur) + before + after

    if target > core + 0.05:
        # extend hold_after so total ≈ target
        extra = target - core
        n_after += int(round(extra * sr))
        after += extra

    pieces = []
    if n_before > 0:
        pieces.append(np.zeros(n_before, dtype=np.float32))
    pieces.append(np.asarray(audio, dtype=np.float32))
    if n_after > 0:
        pieces.append(np.zeros(n_after, dtype=np.float32))
    out = np.concatenate(pieces) if pieces else np.asarray(audio, dtype=np.float32)
    return out, float(len(out) / float(sr))


def _clip_loop_flag(shot: ShotDetail) -> bool:
    """MoMask never loops. Catalog walk/idle may. stretch is ignored (forbidden)."""
    mode = str(shot.body_mode or "").lower()
    if mode in ("momask", "both") or (shot.humanml_prompt or "").strip():
        return False
    pol = (shot.clip_policy or "hold_end").lower()
    if pol in ("hold_end", "stretch", "momask_match"):
        return False
    if pol == "loop":
        return True
    return str(shot.state or "") in ("walking", "standing", "sitting", "dancing")


def bake_shot(
    shot: ShotDetail,
    *,
    coord: FaceCoordinator,
    world: WorldState,
    out_dir: Path,
    t0: float,
    fps: float = 20.0,
    shot_index: int = 1,
    use_brain: bool = True,
    board: Optional[ContinuityBoard] = None,
) -> ShotDetail:
    t_wall0 = time.perf_counter()
    shot.bake_errors = []
    shot.world_in = world.snapshot()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage vs spoken — never TTS stage directions or HumanML captions
    if not shot.spoken or not shot.stage:
        stg, spk = split_stage_and_spoken(shot.text)
        shot.stage = shot.stage or stg
        if not (shot.spoken or "").strip():
            try:
                from face_agents.director_schema import is_motion_caption
                if (is_motion_caption(shot.text) or is_motion_caption(spk)) and not stg:
                    shot.spoken = ""
                    shot.stage = shot.stage or shot.text
                else:
                    shot.spoken = clean_spoken_for_tts(spk or "")
            except Exception:
                shot.spoken = clean_spoken_for_tts(spk or "")
    speak = clean_spoken_for_tts(shot.spoken or "").strip()
    wav_path = out_dir / f"{shot.shot_id}.wav"
    motion_only = not speak
    if motion_only:
        take_guess = float(shot.target_duration_s or shot.motion_duration_s or 3.5)
        sr = 22050
        audio = np.zeros(max(1, int(take_guess * sr)), dtype=np.float32)
        sf.write(str(wav_path), audio, sr)
        speech_dur = 0.0
        print(f"  [speech] motion-only silence {take_guess:.2f}s (no Parler)")
    else:
        print(
            f"  [speech] stage={shot.stage[:50]!r}… spoken={speak[:60]!r}…"
            if len(shot.stage or "") > 50 or len(speak) > 60
            else f"  [speech] stage={shot.stage!r} spoken={speak!r}"
        )
        try:
            audio, sr, speech_dur = _tts(speak, shot.emotion, shot.intensity, wav_path)
        except Exception as e:
            shot.bake_errors.append(f"tts: {e}")
            shot.bake_ok = False
            shot.bake_ms = (time.perf_counter() - t_wall0) * 1000
            return shot

    shot.speech_duration_s = float(speech_dur)

    # Director timing pad
    audio, take_dur = _pad_audio(
        audio,
        sr,
        hold_before_s=shot.hold_before_s,
        hold_after_s=shot.hold_after_s,
        target_duration_s=shot.target_duration_s,
        speech_dur=speech_dur,
    )
    try:
        sf.write(str(wav_path), audio, sr)
    except Exception as e:
        shot.bake_errors.append(f"wav_write: {e}")

    shot.audio_path = str(wav_path)
    shot.duration_s = float(take_dur)
    shot.t0 = float(t0)
    shot.t1 = float(t0) + float(take_dur)

    # Face track uses speech length; hold_before offsets start within the take
    face_dur = float(speech_dur)
    hold_before = max(0.0, float(shot.hold_before_s or 0.0))

    clip_speed = float(max(0.35, min(1.6, shot.clip_speed or 1.0)))

    def _body():
        # MoMask / open path uses full stage+intent; duration matches take
        body_text = shot.stage or shot.text
        return _resolve_body_action(
            body_mode=shot.body_mode,
            humanml_prompt=shot.humanml_prompt,
            body_state=shot.state,
            actions=shot.actions,
            emotion=shot.emotion,
            text=body_text,
            duration_s=take_dur,
            seed=shot_index * 101 + 7,
            allow_sync_gen=bool(shot.allow_sync_gen),
        )

    def _silent_face():
        from ..base import FaceContext

        ctx = FaceContext(
            duration=float(take_dur),
            is_speaking=False,
            emotion=shot.emotion,
            intensity=shot.intensity,
            text="",
            audio_path=str(wav_path),
            sample_rate=sr,
        )
        ctx.extras["motion_only"] = True
        return ctx

    def _face():
        return coord.prepare_sentence(
            text=speak,
            emotion=shot.emotion,
            intensity=shot.intensity,
            audio_path=str(wav_path),
            duration=face_dur,
            sample_rate=sr,
            body_actions=shot.actions,
            body_state=shot.state,
            body_mode=shot.body_mode,
            humanml_prompt=shot.humanml_prompt,
            momask_action="",
            momask_library="",
            camera_shot=shot.camera_shot,
            camera_move=shot.camera_move,
        )

    try:
        if motion_only:
            # Never call prepare_sentence / wav2arkit / Brain on silence.
            with ThreadPoolExecutor(max_workers=1) as ex:
                body_meta = ex.submit(_body).result()
            ctx = _silent_face()
        else:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_b = ex.submit(_body)
                fut_f = ex.submit(_face)
                body_meta = fut_b.result()
                ctx = fut_f.result()
    except Exception as e:
        shot.bake_errors.append(f"prep: {e}")
        shot.bake_ok = False
        shot.bake_ms = (time.perf_counter() - t_wall0) * 1000
        return shot

    loop_body = _clip_loop_flag(shot)
    if body_meta:
        shot.body_action = str(
            body_meta.get("action")
            or body_meta.get("action_name")
            or body_meta.get("clip_id")
            or ""
        )
        shot.body_library = str(body_meta.get("library_blend") or "")
        if shot.body_action:
            ctx.extras["momask_action"] = shot.body_action
            ctx.extras["momask_library"] = shot.body_library
            ctx.extras["motion_plan"] = {
                "clip_id": shot.body_action,
                "action_name": shot.body_action,
                "engine": body_meta.get("engine") or "momask",
                "loop": bool(body_meta.get("loop", loop_body)),
                "library_blend": shot.body_library,
                "speed": clip_speed,
                "amp": 1.0,
            }
    else:
        # catalog path — director speed + policy for receiver
        plan = dict(ctx.extras.get("motion_plan") or {})
        plan["loop"] = loop_body
        plan["clip_policy"] = shot.clip_policy
        plan["speed"] = clip_speed
        plan["engine"] = plan.get("engine") or "clip_catalog"
        # Resolve Action name so Session install / MP4 never see empty body_action
        act = str(plan.get("action_name") or plan.get("clip_id") or "").strip()
        if not act:
            acts = [str(a).lower() for a in (shot.actions or [])]
            for prefer in (
                "wave", "talk_open", "talk_emphasize", "celebrate", "shrug",
                "look_around", "run", "walk", "nod_yes", "shake_no", "idle",
            ):
                if prefer in acts:
                    act = prefer
                    break
        if not act:
            st = str(shot.state or "").lower()
            act = "run" if any(x in st for x in ("walk", "loco", "run")) else "talk_open"
        plan["action_name"] = act
        plan["clip_id"] = act
        shot.body_action = act
        ctx.extras["motion_plan"] = plan

    # Final safety: never leave body_action blank after a successful bake prep
    if not str(shot.body_action or "").strip():
        acts = [str(a).lower() for a in (shot.actions or [])]
        shot.body_action = next(
            (p for p in ("wave", "talk_open", "run", "look_around", "idle") if p in acts),
            "talk_open",
        )

    ctx.extras["clip_policy"] = shot.clip_policy
    ctx.extras["clip_speed"] = clip_speed
    ctx.extras["hold_before_s"] = hold_before
    ctx.extras["take_duration_s"] = take_dur
    ctx.extras["spoken"] = speak
    ctx.extras["stage"] = shot.stage

    # Camera over FULL take (including holds) — cinematic settle
    try:
        cam = plan_camera_for_beat(
            duration_s=take_dur,
            emotion=shot.emotion,
            intensity=shot.intensity,
            state=shot.state,
            actions=shot.actions,
            fps=fps,
            shot=shot.camera_shot or None,
            move_type=shot.camera_move or None,
            camera_role=shot.camera_role or "",
            pace=shot.pace or "medium",
            humanml_prompt=shot.humanml_prompt or "",
            session_clip_index=max(0, int(shot_index) - 1),
            # Soft track only; hard track caused look into mesh on ECU
            track_head=(shot.camera_shot or "").upper() not in ("ECU", "CU"),
        )
        cam_d = cam.to_dict()
        for kf in cam_d.get("keyframes") or []:
            kf["t_abs"] = float(shot.t0) + float(kf.get("t") or 0.0)
            kf["frame_abs"] = int(round(kf["t_abs"] * fps)) + 1
        shot.camera = cam_d
        shot.camera_role = cam.camera_role or shot.camera_role
        shot.camera_shot = cam.shot or shot.camera_shot
        shot.camera_move = cam.move_type or shot.camera_move
        ctx.extras["camera_plan"] = cam_d
        print(
            f"  [camera] role={cam.camera_role} {cam.shot}/{cam.move_type} "
            f"pace={cam.pace} keys={len(cam.keyframes)} take={take_dur:.2f}s"
        )
    except Exception as e:
        shot.bake_errors.append(f"camera: {e}")

    look_d = {}
    try:
        look_obj = getattr(shot, "look", None)
        if look_obj is not None and hasattr(look_obj, "to_dict"):
            look_d = look_obj.to_dict()
        elif isinstance(look_obj, dict):
            look_d = dict(look_obj)
    except Exception:
        look_d = {}
    if look_d:
        ctx.extras["look"] = look_d

    # Face timeline — skip on motion-only (no lips). Dialogue: map to absolute.
    if motion_only or face_dur <= 0.04:
        shot.face_timeline_path = ""
        ctx.extras["motion_only"] = True
        print(f"  [face] skip timeline (motion-only)")
    else:
        try:
            track = coord._build_face_timeline(ctx, face_dur)
            f0_abs = int(round((shot.t0 + hold_before) * fps)) + 1
            for kf in track.get("keyframes") or []:
                local = int(kf.get("frame") or 1) - 1
                kf["frame"] = f0_abs + local
                kf["t_abs"] = float(shot.t0) + hold_before + float(kf.get("t") or 0.0)
            track["frame_start"] = f0_abs
            track["frame_end"] = f0_abs + int(track.get("n_frames") or 1) - 1
            track["shot_id"] = shot.shot_id
            track["t0"] = shot.t0
            track["t1"] = shot.t1
            track["hold_before_s"] = hold_before
            fpath = out_dir / f"{shot.shot_id}_face_timeline.json"
            import json

            fpath.write_text(json.dumps(track), encoding="utf-8")
            shot.face_timeline_path = str(fpath)
            ctx.extras["face_timeline_path"] = str(fpath)
            ctx.extras["face_timeline"] = {
                "path": str(fpath),
                "frame_start": track["frame_start"],
                "frame_end": track["frame_end"],
            }
        except Exception as e:
            shot.bake_errors.append(f"face_timeline: {e}")

    shot._ctx = ctx  # type: ignore[attr-defined]
    shot._audio = audio  # type: ignore[attr-defined]
    shot._sr = sr  # type: ignore[attr-defined]

    pos_delta = (0.0, 0.0, 0.0)
    if shot.state == "walking" or "walk" in (shot.humanml_prompt or "").lower():
        pos_delta = (0.0, 0.35 * max(1.0, take_dur / 4.0), 0.0)
    world.apply_shot_end(
        character_id="hero",
        base_state=shot.state,
        emotion=shot.emotion,
        intensity=shot.intensity,
        last_action=shot.body_action or (shot.actions[0] if shot.actions else ""),
        body_mode=shot.body_mode,
        position_delta=pos_delta,
        note=f"{shot.shot_id} baked role={shot.camera_role}",
    )
    shot.world_out = world.snapshot()

    if board is not None:
        board.record_bake(
            shot.shot_id,
            actual_duration_s=shot.duration_s,
            t0=shot.t0,
            t1=shot.t1,
            bake_ok=not shot.bake_errors,
            state=shot.state,
        )

    shot.bake_ok = not shot.bake_errors
    shot.bake_ms = (time.perf_counter() - t_wall0) * 1000
    print(
        f"  [bake] {shot.shot_id} ok={shot.bake_ok} "
        f"speech={speech_dur:.2f}s take={shot.duration_s:.2f}s "
        f"t=[{shot.t0:.2f},{shot.t1:.2f}] "
        f"cam={shot.camera_role}/{shot.camera_shot}/{shot.camera_move} "
        f"body={shot.body_mode}/{shot.body_action or 'catalog'} "
        f"speed={clip_speed:.2f} "
        f"ms={shot.bake_ms:.0f}"
    )
    return shot


def bake_sequence(
    shots: List[ShotDetail],
    *,
    coord: FaceCoordinator,
    world: WorldState,
    out_dir: Path,
    fps: float = 20.0,
    board: Optional[ContinuityBoard] = None,
) -> List[ShotDetail]:
    t_cursor = 0.0
    baked: List[ShotDetail] = []
    for i, sh in enumerate(shots, 1):
        print(f"\n=== BAKE {sh.shot_id} ({i}/{len(shots)}) ===")
        print(f"  text={sh.text[:70]}{'…' if len(sh.text)>70 else ''}")
        print(
            f"  emo={sh.emotion} body={sh.body_mode} clip={sh.clip_policy} "
            f"speed={sh.clip_speed:.2f} target={sh.target_duration_s:.1f}s "
            f"cam={sh.camera_role}/{sh.camera_shot}/{sh.camera_move} pace={sh.pace}"
        )
        print(f"  reason={sh.motion_reason!r}")
        sh = bake_shot(
            sh,
            coord=coord,
            world=world,
            out_dir=out_dir,
            t0=t_cursor,
            fps=fps,
            shot_index=i,
            board=board,
        )
        t_cursor = float(sh.t1)
        baked.append(sh)
    return baked
