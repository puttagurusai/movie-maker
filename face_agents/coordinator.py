"""
coordinator.py — multi-agent talking face.

  lips   → wav2arkit (speech mouth only)
  eyes   → blink/gaze + strong emotion + Brain
  brows  → strong emotion_map + Brain + energy
  cheeks → smile/frown/nose (expression corners)
  head   → nod/breath/energy
  body   → SMPL-X bone actions (wave, shrug, talk, …)

Bake once, sample @ 30fps with lip hold for A/V sync.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

import emotion_map

from .base import BaseFaceAgent, FaceContext
from .brain_track import bake_brain_expression, compute_energy_envelope
from .brows_agent import BrowsAgent
from .cheeks_agent import CheeksAgent
from .eyes_agent import EyesAgent
from .head_agent import HeadAgent
from .body_agent import BodyAgent
from .lips_agent import LipsAgent
from .micro_expression_agent import MicroExpressionAgent
from .expression_mix import blend_emotions
from .expression_tokens import (
    parse_tokens,
    evaluate_tokens,
    strip_tokens,
    has_nod,
    mix_token_sounds,
    TOKEN_RECIPES,
    TOKEN_DURATION,
    TOKEN_EMOTIONS,
    TOKEN_SOUNDS,
    resolve_sound_path,
    prepare_token_clip,
    _resolve_alias,
)

# Default lip lag so mouth is NOT ahead of speakers (Windows ~100–160ms)
_ENV_HOLD = os.environ.get("FACE_LIP_HOLD_MS")


def _device_latency_s() -> float:
    try:
        dev = sd.query_devices(kind="output")
        lat = dev.get("default_high_output_latency") or dev.get("default_low_output_latency")
        if lat is not None and float(lat) > 0:
            return float(lat)
    except Exception:
        pass
    return 0.10


def _lip_hold_s() -> float:
    """Seconds to sample lips BEHIND audio wall-clock."""
    try:
        from .policy_bridge import get_policy
        pol = get_policy()
        if "face_lip_hold_s" in pol:
            return max(0.04, min(0.28, float(pol["face_lip_hold_s"])))
    except Exception:
        pass
    if _ENV_HOLD is not None and str(_ENV_HOLD).strip() != "":
        return max(0.04, min(0.28, float(_ENV_HOLD) / 1000.0))
    # Auto: device latency + small model lead
    return max(0.12, min(0.22, _device_latency_s() + 0.04))


class FaceCoordinator:
    def __init__(
        self,
        udp_ip: str = "127.0.0.1",
        udp_port: int = 9001,
        fps: float = 30.0,
        use_brain: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.fps = fps
        self.use_brain = use_brain
        self.device = device
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.lips = LipsAgent()
        self.eyes = EyesAgent()
        self.brows = BrowsAgent()
        self.cheeks = CheeksAgent()
        self.head = HeadAgent()
        self.body = BodyAgent()
        self.micro = MicroExpressionAgent()
        self._last_upper: Dict[str, float] = {}  # upper-face values from previous sentence

        self.agents: List[BaseFaceAgent] = [
            self.lips,
            self.eyes,
            self.brows,
            self.cheeks,
            self.head,
            self.body,
            self.micro,
        ]

    def send_udp(self, packet: dict) -> None:
        try:
            self._sock.sendto(
                json.dumps(packet).encode("utf-8"),
                (self.udp_ip, self.udp_port),
            )
        except Exception as e:
            print(f"[coordinator] UDP failed: {e}")

    def _send_body_sync_start(
        self,
        ctx: FaceContext,
        duration: float,
        play_t0: float,
    ) -> None:
        """
        One forced body packet at audio start so Blender locks body clock to speech.
        Face/lips still stream every frame; body Action advances on the same wall clock.
        """
        plan = ctx.extras.get("motion_plan") or {}
        action_label = self.body.current_action_label(ctx)
        momask_act = str(ctx.extras.get("momask_action") or plan.get("action_name") or "")
        if momask_act and (
            plan.get("engine") == "momask" or ctx.extras.get("body_mode") == "momask"
        ):
            action_label = momask_act
        st = str(ctx.extras.get("body_state") or "standing")
        eng = str(plan.get("engine") or "").lower()
        al = str(action_label or "").lower()
        is_momask = eng == "momask" or al.startswith("momask_") or bool(momask_act)
        # MoMask = one-shot only. Catalog walk/idle may loop to fill speech.
        if is_momask:
            use_loop = False
            eng = "momask"
        else:
            default_loop = st in ("walking", "sitting", "standing", "dancing") or (
                al in ("talk_open", "talk_emphasize", "idle", "walk", "run", "sit_idle")
            )
            use_loop = bool(plan.get("loop") if "loop" in plan else default_loop)
        pkt = {
            "type": "body",
            "bones": {},
            "action": action_label,
            "clip_id": plan.get("clip_id") or action_label,
            "t": 0.0,
            "state": st,
            "gesture_target": ctx.extras.get("gesture_target", "none"),
            "hand": ctx.extras.get("hand", "right"),
            "energy": 0.0,
            "intensity": float(ctx.intensity or 0.7),
            "duration": float(duration) if duration > 0.05 else None,
            "speed": float(
                plan.get("speed")
                if plan.get("speed") is not None
                else (0.85 + 0.4 * float(ctx.intensity or 0.7))
            ),
            "loop": use_loop,
            "amp": float(plan.get("amp") or 1.0),
            "engine": eng or plan.get("engine") or "clip_catalog",
            "library_blend": plan.get("library_blend")
            or ctx.extras.get("momask_library")
            or "",
            "play_t0": float(play_t0),
            # MoMask BVH is 20fps; catalog clips often 30fps — wrong fps causes double-cycle feel
            "clip_fps": float(
                plan.get("clip_fps")
                or (20.0 if is_momask else (self.fps or 30.0))
            ),
            "sync": True,
            "force": True,
            # Session continuity: append clip + root-align (do not wipe prior timeline)
            # NLA session append (lightweight). Live play = current clip only.
            "append_timeline": bool(ctx.extras.get("append_timeline", True)),
            "continue_root": bool(ctx.extras.get("continue_root", True)),
            "session_clip_index": int(ctx.extras.get("session_clip_index") or 0),
            "session_id": str(ctx.extras.get("session_id") or ""),
            "session_name": str(ctx.extras.get("session_name") or ctx.extras.get("session_id") or ""),
            "clip_label": str(
                ctx.extras.get("clip_label")
                or plan.get("clip_label")
                or ctx.extras.get("humanml_prompt")
                or plan.get("prompt")
                or action_label
                or ""
            ),
            "prompt": str(
                ctx.extras.get("humanml_prompt")
                or plan.get("prompt")
                or ""
            ),
            "text": str(getattr(ctx, "text", "") or ctx.extras.get("text") or "")[:200],
            "action_frames": int(
                plan.get("action_frames")
                or ctx.extras.get("action_frames")
                or 0
            ),
            "motion_length": int(
                plan.get("motion_length")
                or ctx.extras.get("motion_length")
                or 0
            ),
            "speech_delay_s": float(ctx.extras.get("speech_delay_s") or plan.get("speech_delay_s") or 0.0),
            "speech_duration_s": float(
                ctx.extras.get("speech_duration_s")
                or plan.get("speech_duration_s")
                or duration
                or 0.0
            ),
            "audio_path": str(ctx.extras.get("audio_path") or ""),
            "look": ctx.extras.get("look") if isinstance(ctx.extras.get("look"), dict) else {},
            "clip_policy": str(
                ctx.extras.get("clip_policy")
                or plan.get("clip_policy")
                or ("momask_match" if is_momask else "hold_end")
            ),
        }
        self.send_udp(pkt)
        ctx.extras["_last_body_send"] = {"action": action_label, "t": 0.0}
        # Pass session slot start so camera keys align on timeline (no wipe)
        if ctx.extras.get("session_frame_start"):
            ctx.extras["camera_session_frame_start"] = ctx.extras["session_frame_start"]

    def _build_face_timeline(
        self,
        ctx: FaceContext,
        duration: float,
    ) -> Dict[str, Any]:
        """
        Offline sample of lips + upper face + head for the full sentence.
        Used for Blender shape-key keyframes so scrub/replay has moving lips.
        Content time (no lip-hold) so timeline frame f maps to speech time (f-1)/fps.
        """
        fps = float(self.fps or 30.0)
        dur = max(0.05, float(duration))
        n = max(2, int(round(dur * fps)) + 1)
        # Sample every 2nd frame for speed; Blender LINEAR interp fills gaps
        step = 2 if n > 40 else 1
        frame0 = 1
        was_speaking = ctx.is_speaking
        ctx.is_speaking = True
        keyframes: List[Dict[str, Any]] = []
        try:
            for i in range(0, n, step):
                t = min(dur, i / fps)
                ctx.t = t
                mouth, upper, head, _body = self._merge_frame(ctx)
                bs: Dict[str, float] = {}
                for k, v in {**mouth, **upper}.items():
                    fv = float(v)
                    if abs(fv) >= 0.004:
                        bs[k] = round(fv, 4)
                pitch, yaw, roll = head
                keyframes.append({
                    "frame": frame0 + i,
                    "t": round(t, 4),
                    "blendshapes": bs,
                    "head": {
                        "pitch": round(float(pitch), 5),
                        "yaw": round(float(yaw), 5),
                        "roll": round(float(roll), 5),
                    },
                })
            # Always include last frame
            if keyframes and keyframes[-1]["frame"] != frame0 + n - 1:
                t = dur
                ctx.t = t
                mouth, upper, head, _body = self._merge_frame(ctx)
                bs = {
                    k: round(float(v), 4)
                    for k, v in {**mouth, **upper}.items()
                    if abs(float(v)) >= 0.004
                }
                pitch, yaw, roll = head
                keyframes.append({
                    "frame": frame0 + n - 1,
                    "t": round(t, 4),
                    "blendshapes": bs,
                    "head": {
                        "pitch": round(float(pitch), 5),
                        "yaw": round(float(yaw), 5),
                        "roll": round(float(roll), 5),
                    },
                })
        finally:
            ctx.is_speaking = was_speaking
            ctx.t = 0.0

        return {
            "fps": fps,
            "frame_start": frame0,
            "frame_end": frame0 + n - 1,
            "duration_s": dur,
            "n_frames": n,
            "sample_step": step,
            "keyframes": keyframes,
        }

    def _send_face_timeline_bake(self, ctx: FaceContext, duration: float) -> None:
        """
        Bake face track to disk + tell Blender to keyframe shape keys.
        Env USE_FACE_TIMELINE=0 disables. Default ON.
        ctx.extras['skip_face_timeline_bake']=True → skip (master film already joined).
        ctx.extras['face_clear_previous']=False → append without wiping prior shots.
        """
        if os.environ.get("USE_FACE_TIMELINE", "1").strip().lower() in (
            "0", "false", "no", "off",
        ):
            return
        if ctx.extras.get("skip_face_timeline_bake"):
            return
        try:
            t0 = time.perf_counter()
            track = self._build_face_timeline(ctx, duration)
            sfs = int(
                ctx.extras.get("session_frame_start")
                or ctx.extras.get("camera_session_frame_start")
                or 1
            )
            if sfs > 1:
                delta = sfs - 1
                for kf in track.get("keyframes") or []:
                    kf["frame"] = int(kf.get("frame") or 1) + delta
                track["frame_start"] = int(track.get("frame_start") or 1) + delta
                track["frame_end"] = int(track.get("frame_end") or 1) + delta
            out_dir = Path("temp")
            out_dir.mkdir(exist_ok=True)
            # unique-ish name per sentence
            stamp = int(time.time() * 1000) % 100000000
            path = out_dir / f"face_timeline_{stamp}.json"
            path.write_text(json.dumps(track), encoding="utf-8")
            # Append face keys on the session; only the first clip (or reset) clears
            clear_prev = bool(ctx.extras.get("face_clear_previous", False))
            if int(ctx.extras.get("session_clip_index") or 0) > 0:
                clear_prev = False
            self.send_udp({
                "type": "face_keyframes",
                "path": str(path.resolve()),
                "fps": track["fps"],
                "frame_start": track["frame_start"],
                "frame_end": track["frame_end"],
                "set_frame_range": True,
                "clear_previous": clear_prev,
            })
            ctx.extras["face_timeline_path"] = str(path)
            ctx.extras["face_timeline"] = {
                "fps": track["fps"],
                "frame_start": track["frame_start"],
                "frame_end": track["frame_end"],
                "n_frames": track["n_frames"],
                "path": str(path),
            }
            print(
                f"[coordinator] FACE TIMELINE baked {track['n_frames']} fr "
                f"f{track['frame_start']}-{track['frame_end']} → {path.name} "
                f"({(time.perf_counter()-t0)*1000:.0f}ms)  (scrub will replay lips)"
            )
        except Exception as e:
            print(f"[coordinator] face timeline bake failed: {e}")

    def _send_camera_plan(self, ctx: FaceContext, duration: float) -> None:
        """
        Plan cinematic camera for this sentence and send to Blender.
        Subject-relative (FilmAgent-style anchors: head/chest/full_body — not always hip).
        Disabled with USE_MOVIE_CAMERA=0. Never touches face/body.
        """
        try:
            from .camera_agent import movie_camera_enabled, plan_camera_for_beat
        except Exception as e:
            print(f"[coordinator] camera_agent import failed: {e}")
            return
        if not movie_camera_enabled():
            return
        # Allow beat/orchestrator to force shot
        shot = ctx.extras.get("camera_shot")
        move = ctx.extras.get("camera_move")
        try:
            duration = float(ctx.extras.get("camera_duration_s") or duration)
        except (TypeError, ValueError):
            pass
        try:
            # Reuse plan already attached (e.g. from movie_timeline save path)
            existing = ctx.extras.get("camera_plan")
            if isinstance(existing, dict) and existing.get("keyframes"):
                cam_name = existing.get("camera_name") or "MovieCam_A"
                look_name = existing.get("look_at_name") or (cam_name + "_LookAt")
                pkt = {
                    "type": "camera",
                    "op": "plan",
                    "duration": float(existing.get("duration_s") or duration),
                    "fps": 20.0,
                    "shot": existing.get("shot") or "",
                    "move_type": existing.get("move_type") or "",
                    "track_head": bool(existing.get("track_head", False)),
                    "camera_name": cam_name,
                    "look_at_name": look_name,
                    "camera_role": existing.get("camera_role") or "",
                    "set_scene_camera": True,
                    "bake_keyframes": True,
                    "clear_previous": False,
                    "min_cam_dist": float(existing.get("min_cam_dist") or 1.08),
                    "subject_relative": bool(existing.get("subject_relative", True)),
                    "subject_anchor": existing.get("subject_anchor") or "chest",
                    "follow_lag": float(existing.get("follow_lag") or 0.38),
                    "hold_after": bool(existing.get("hold_after", True)),
                    "clear_previous": False,
                    "force_camera_role": True,
                    "keyframes": existing["keyframes"],
                }
                sfs = (
                    ctx.extras.get("session_frame_start")
                    or ctx.extras.get("camera_session_frame_start")
                    or ctx.extras.get("session_frame_cursor")
                )
                if sfs:
                    pkt["session_frame_start"] = int(sfs)
                self.send_udp(pkt)
                print(
                    f"[coordinator] CAMERA (cached) role={pkt.get('camera_role')} "
                    f"cam={cam_name} shot={pkt.get('shot')} move={pkt.get('move_type')} "
                    f"anchor={pkt.get('subject_anchor')} keys={len(pkt['keyframes'])}"
                )
                return

            role = str(ctx.extras.get("camera_role") or "")
            sh = str(shot) if shot else None
            # Soft track only on wider sizes — CU/ECU head track caused inside-mesh
            track = (sh or "MS").upper() not in ("ECU", "CU")
            plan = plan_camera_for_beat(
                duration_s=float(duration),
                emotion=str(ctx.emotion or "neutral"),
                intensity=float(ctx.intensity or 0.7),
                state=str(ctx.extras.get("body_state") or "standing"),
                actions=list(ctx.extras.get("body_actions") or []),
                fps=20.0,
                shot=sh,
                move_type=str(move) if move else None,
                track_head=track,
                camera_role=role,
                pace=str(ctx.extras.get("camera_pace") or "medium"),
                humanml_prompt=str(ctx.extras.get("humanml_prompt") or ""),
                session_clip_index=int(ctx.extras.get("session_clip_index") or 0),
                subject_relative=bool(ctx.extras.get("subject_relative", True)),
                subject_anchor=ctx.extras.get("subject_anchor"),
                last_shot=str(ctx.extras.get("last_camera_shot") or ""),
                last_role=str(ctx.extras.get("last_camera_role") or ""),
                location_changed=bool(ctx.extras.get("look_location_changed")),
            )
            # CameraAgent owns role/shot/move. Do not slam back to A_cam/static.
            plan.follow_lag = max(0.16, min(0.38, float(plan.follow_lag or 0.22)))
            pkt = plan.udp_packet()
            pkt["clear_previous"] = False
            pkt["bake_keyframes"] = True
            pkt["follow_lag"] = float(plan.follow_lag)
            # Align baked cam keys to session body slot (cursor if start unknown)
            sfs = (
                ctx.extras.get("session_frame_start")
                or ctx.extras.get("camera_session_frame_start")
                or ctx.extras.get("session_frame_cursor")
            )
            if sfs:
                pkt["session_frame_start"] = int(sfs)
            self.send_udp(pkt)
            ctx.extras["camera_plan"] = plan.to_dict()
            ctx.extras["camera_shot"] = plan.shot
            ctx.extras["camera_move"] = plan.move_type
            ctx.extras["camera_role"] = plan.camera_role
            ctx.extras["camera_anchor"] = plan.subject_anchor
            print(
                f"[coordinator] CAMERA role={plan.camera_role} cam={plan.camera_name} "
                f"shot={plan.shot} move={plan.move_type} anchor={plan.subject_anchor} "
                f"subj_rel={plan.subject_relative} keys={len(plan.keyframes)} "
                f"notes={plan.notes}"
            )
            look = ctx.extras.get("look")
            if isinstance(look, dict) and look:
                try:
                    from .look_schema import movie_look_enabled
                    if movie_look_enabled():
                        self.send_udp({
                            "type": "look",
                            "op": "apply",
                            "look": look,
                            "session_id": str(ctx.extras.get("session_id") or ""),
                        })
                except Exception as e:
                    print(f"[coordinator] look send failed: {e}")
        except Exception as e:
            print(f"[coordinator] camera plan failed: {e}")

    def prepare_sentence(
        self,
        text: str,
        emotion: str,
        intensity: float,
        audio_path: str,
        duration: float,
        sample_rate: int,
        body_actions: Optional[List[str]] = None,
        action_timing: str = "during",
        body_state: str = "standing",
        gesture_target: str = "none",
        hand: str = "right",
        body_mode: str = "auto",
        humanml_prompt: str = "",
        momask_action: str = "",
        momask_library: str = "",
        camera_shot: str = "",
        camera_move: str = "",
    ) -> FaceContext:
        ctx = FaceContext(
            t=0.0,
            duration=duration,
            is_speaking=False,
            emotion=emotion,
            intensity=intensity,
            text=text,
            audio_path=audio_path,
            sample_rate=sample_rate,
        )
        if body_actions:
            ctx.extras["body_actions"] = list(body_actions)
        ctx.extras["action_timing"] = action_timing or "during"
        ctx.extras["body_state"] = body_state or "standing"
        ctx.extras["gesture_target"] = gesture_target or "none"
        ctx.extras["hand"] = hand or "right"
        # MoMask body (Action name) — never used for mouth/visemes
        ctx.extras["body_mode"] = body_mode or "auto"
        ctx.extras["humanml_prompt"] = humanml_prompt or ""
        if camera_shot:
            ctx.extras["camera_shot"] = str(camera_shot)
        if camera_move:
            ctx.extras["camera_move"] = str(camera_move)
        if momask_action:
            ctx.extras["momask_action"] = momask_action
            ctx.extras["momask_library"] = momask_library or ""
            ctx.extras["motion_plan"] = {
                "clip_id": momask_action,
                "action_name": momask_action,
                "engine": "momask",
                "loop": False,
                "library_blend": momask_library or "",
                "speed": 0.85 + 0.4 * float(intensity or 0.7),
                "amp": 1.0,
            }

        try:
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            ctx.sample_rate = int(sr)
            ctx.energy_envelope = compute_energy_envelope(audio, int(sr), fps=self.fps)
            ctx.duration = float(len(audio) / float(sr))
        except Exception as e:
            print(f"[coordinator] energy failed: {e}")
            ctx.energy_envelope = np.zeros(1, dtype=np.float32)

        # Tokens are handled as SEPARATE segments by orchestrator (play_token_reaction).
        raw_text = text
        clean_text = strip_tokens(text)
        ctx.extras["expression_events"] = []
        if "[" in raw_text and "]" in raw_text:
            ctx.extras["expression_events"] = parse_tokens(raw_text, ctx.duration)
            if ctx.extras["expression_events"]:
                print(
                    f"[coordinator] WARN: tokens in prepare text "
                    f"(should be segmented upstream): "
                    f"{[e.token for e in ctx.extras['expression_events']]}"
                )
        ctx.text = clean_text or raw_text

        # ── Parallel bake: Brain (A2E+HuBERT) || wav2arkit lips ────────────
        # These are independent after WAV exists; wall time ≈ max(brain, lips).
        ctx.brain_enabled = False
        ctx.brain_frames = []
        t_bake0 = time.perf_counter()

        def _job_brain():
            if not self.use_brain:
                return None, 30.0, None
            print("[coordinator] Baking Brain expression (upper face)…")
            return bake_brain_expression(
                audio_path, emotion, intensity, device=self.device
            )

        def _job_lips():
            print("[coordinator] Baking lips (wav2arkit)…")
            self.lips.prepare(ctx)
            return len(ctx.lips_frames or [])

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_b = ex.submit(_job_brain)
            fut_l = ex.submit(_job_lips)
            brain_result = fut_b.result()
            lips_n = fut_l.result()

        if brain_result is not None:
            frames, bfps, _raw = brain_result
            if frames:
                ctx.brain_frames = frames
                ctx.brain_fps = bfps
                ctx.brain_enabled = True
            else:
                print("[coordinator] Brain bake empty — strong presets only")

        # Fast agents (no heavy ML) — sequential is fine
        print("[coordinator] Preparing expression agents…")
        for agent in self.agents:
            if agent.name == "lips":
                continue  # already prepared
            agent.prepare(ctx)

        print(
            f"[coordinator] Ready brain={ctx.brain_enabled} "
            f"lips={len(ctx.lips_frames)} brain_fr={len(ctx.brain_frames)} "
            f"energy_n={len(ctx.energy_envelope) if ctx.energy_envelope is not None else 0} "
            f"bake_wall={(time.perf_counter()-t_bake0):.2f}s (parallel lips||brain)"
        )
        return ctx

    def _merge_frame(
        self, ctx: FaceContext
    ) -> Tuple[Dict[str, float], Dict[str, float], Tuple[float, float, float], Dict]:
        mouth: Dict[str, float] = {}
        upper: Dict[str, float] = {}

        for agent in self.agents:
            part = agent.sample(ctx)
            if agent.name == "lips":
                mouth.update(part)
            elif agent.name in ("head", "body"):
                continue
            elif agent.name == "micro":
                # Additive overlay — take max(existing, micro) per key
                # so micro-expressions can only amplify, never suppress
                for k, v in part.items():
                    upper[k] = min(1.0, max(upper.get(k, 0.0), float(v)))
            else:
                upper.update(part)

        # Never put expression corners on viseme (cheeks own them via emotion)
        for k in (
            "mouthSmileLeft", "mouthSmileRight",
            "mouthFrownLeft", "mouthFrownRight",
            "noseSneerLeft", "noseSneerRight",
            "browDownLeft", "browDownRight", "browInnerUp",
            "browOuterUpLeft", "browOuterUpRight",
            "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
        ):
            mouth.pop(k, None)

        # Inter-sentence blending: smooth from previous emotion for first blend_dur seconds
        blend_from = ctx.extras.get("blend_from", {})
        blend_dur  = float(ctx.extras.get("blend_dur", 0.0))
        if blend_from and blend_dur > 0 and ctx.t < blend_dur:
            upper = blend_emotions(blend_from, upper, ctx.t, blend_dur)

        # Expression tokens: timed bursts from [eww], [laugh], [shocked] etc.
        events = ctx.extras.get("expression_events", [])
        if events:
            token_bs = evaluate_tokens(events, ctx.t)
            for k, v in token_bs.items():
                if k != "jawOpen":  # jaw handled by lips_agent
                    upper[k] = min(1.0, max(upper.get(k, 0.0), float(v)))
                else:
                    mouth[k] = min(1.0, max(mouth.get(k, 0.0), float(v)))
            # [nod] token → inject into head via extras
            if has_nod(events, ctx.t):
                ctx.extras["token_nod"] = True

        head = self.head.get_head(ctx)
        # Skip heavy procedural body when using Action clips (catalog/momask) — realtime
        plan_peek = ctx.extras.get("motion_plan") or {}
        eng = plan_peek.get("engine") or ""
        use_clip = bool(
            ctx.extras.get("momask_action")
            or eng in ("momask", "clip_catalog")
            or ctx.extras.get("body_actions")
        )
        # Still resolve action label once; only compute bone dict if procedural needed
        if use_clip and not os.environ.get("BODY_PROCEDURAL", "0").strip() in (
            "1",
            "true",
            "yes",
        ):
            body = {}
            # ensure motion_plan / action label filled
            _ = self.body.current_action_label(ctx)
        else:
            body = self.body.get_body(ctx)
        return mouth, upper, head, body

    def _send_frame(
        self,
        ctx: FaceContext,
        mouth: Dict[str, float],
        upper: Dict[str, float],
        head: Tuple[float, float, float],
        body: Optional[Dict] = None,
    ) -> None:
        pitch, yaw, roll = head
        self.send_udp({"type": "head", "pitch": pitch, "yaw": yaw, "roll": roll})
        # Body: Action name + duration (audio-tied). Avoid re-sending huge bone dicts every frame.
        energy = 0.0
        try:
            energy = float(ctx.speech_energy()) if ctx.is_speaking else 0.0
        except Exception:
            energy = 0.0
        action_label = self.body.current_action_label(ctx)
        plan = ctx.extras.get("motion_plan") or {}
        momask_act = str(ctx.extras.get("momask_action") or plan.get("action_name") or "")
        if momask_act and (plan.get("engine") == "momask" or ctx.extras.get("body_mode") == "momask"):
            action_label = momask_act
        speech_dur = float(ctx.duration or 0.0)
        play_t0 = ctx.extras.get("play_t0")
        # Send body packet only when action changes or every ~0.25s (not 30Hz spam)
        last = ctx.extras.get("_last_body_send")
        now_t = float(ctx.t or 0.0)
        send_body = (
            last is None
            or last.get("action") != action_label
            or (now_t - float(last.get("t") or 0.0)) >= 0.25
            or now_t < 0.05
        )
        if send_body:
            ctx.extras["_last_body_send"] = {"action": action_label, "t": now_t}
            st = str(ctx.extras.get("body_state") or "standing")
            eng = str(plan.get("engine") or "").lower()
            al = str(action_label or "").lower()
            is_momask = (
                eng == "momask"
                or al.startswith("momask_")
                or bool(momask_act)
            )
            # NEVER loop MoMask — walking state alone used to force loop → 2 cycles
            if is_momask:
                use_loop = False
                eng = "momask"
                c_fps = float(plan.get("clip_fps") or 20.0)
            else:
                default_loop = st in ("walking", "sitting", "standing", "dancing") or (
                    al in ("talk_open", "talk_emphasize", "idle", "walk", "run", "sit_idle")
                )
                use_loop = bool(plan.get("loop") if "loop" in plan else default_loop)
                c_fps = float(plan.get("clip_fps") or self.fps or 30.0)
            # Prefer full body window for MoMask (not speech-only, which restarts math)
            body_win = (
                ctx.extras.get("body_play_s")
                or plan.get("body_play_s")
                or plan.get("duration")
                or speech_dur
            )
            try:
                body_win_f = float(body_win) if body_win not in (None, "") else speech_dur
            except (TypeError, ValueError):
                body_win_f = speech_dur
            # Mid-stream refine vs new clip: only skip Session append when the
            # SAME action is already live. New catalog clips (wave after walk)
            # must append or they play at Action origin.
            already = (
                last is not None
                and last.get("action") == action_label
                and now_t > 0.05
            )
            is_new_action = last is None or last.get("action") != action_label
            self.send_udp({
                "type": "body",
                # empty bones when using clips — receiver plays Action, not procedural
                "bones": body or {},
                "action": action_label,
                "clip_id": plan.get("clip_id") or action_label,
                "t": now_t,
                "state": st,
                "gesture_target": ctx.extras.get("gesture_target", "none"),
                "hand": ctx.extras.get("hand", "right"),
                "energy": energy,
                "intensity": float(ctx.intensity or 0.7),
                "duration": float(body_win_f) if body_win_f and body_win_f > 0.05 else (
                    speech_dur if speech_dur > 0.05 else None
                ),
                "speed": float(plan.get("speed") or (0.85 + 0.4 * float(ctx.intensity or 0.7))),
                "loop": False if is_momask else use_loop,
                "amp": float(plan.get("amp") or 1.0),
                "engine": eng or plan.get("engine") or "clip_catalog",
                "library_blend": plan.get("library_blend")
                or ctx.extras.get("momask_library")
                or "",
                # Shared wall clock with audio (Blender locks body t0 to this)
                "play_t0": play_t0,
                "clip_fps": c_fps,
                "sync": True,
                "continue_root": bool(ctx.extras.get("continue_root", True)),
                "append_timeline": bool(is_new_action),
                "force": bool(is_new_action),
                "restart": False,
            })
        if upper:
            self.send_udp({
                "type": "emotion",
                "emotion": ctx.emotion,
                "blendshapes": upper,
            })
        if mouth:
            self.send_udp({"type": "viseme", "blendshapes": mouth})

    def play_token_reaction(
        self,
        token: str,
        base_emotion: str = "neutral",
        intensity: float = 0.7,
    ) -> None:
        """
        Pre-recorded reaction clip + lip-sync + expression — NEVER Parler TTS.

        Sync:
          1) Load / energy-trim clip from temp/vocal_sounds/
          2) Bake wav2arkit mouth track on THAT clip
          3) Play clip while streaming lips(t) + expression envelope(t)
          4) Return so caller continues next speech segment
        """
        import sounddevice as sd

        try:
            import wav2arkit
        except ImportError:
            wav2arkit = None  # type: ignore

        token = _resolve_alias((token or "").lower().strip())
        recipe = dict(TOKEN_RECIPES.get(token, TOKEN_RECIPES.get(_resolve_alias(token), {})))
        atk, hld, rel = TOKEN_DURATION.get(token, (0.10, 0.45, 0.30))
        emo_label = TOKEN_EMOTIONS.get(token, base_emotion)

        prepared = prepare_token_clip(token)
        audio = None
        sr = 22050
        clip_path = None
        if prepared is not None:
            audio, sr, clip_path = prepared
        else:
            print(f"[coordinator] TOKEN [{token}] NO pre-recorded clip — face-only (Parler not used)")

        # --- Lip track from the REAL clip (sync mouth to laugh/gasp audio) ---
        lip_frames: List[Dict[str, float]] = []
        lip_fps = self.fps
        if clip_path and wav2arkit is not None:
            try:
                print(f"[coordinator] TOKEN [{token}] baking lips from clip {os.path.basename(clip_path)}")
                frames, fps, _raw = wav2arkit.audio_file_to_frames(
                    clip_path, mouth_only=True, enhance_mouth=True,
                )
                lip_frames = [
                    {k: float(v) for k, v in fr.items() if k in self.lips.owned_keys}
                    for fr in frames
                ]
                lip_fps = float(fps) if fps else self.fps
                print(f"[coordinator] TOKEN [{token}] lip frames={len(lip_frames)} @ {lip_fps:.0f}fps")
            except Exception as e:
                print(f"[coordinator] TOKEN [{token}] lip bake failed: {e}")
                lip_frames = []

        dur_audio = float(len(audio) / sr) if audio is not None and len(audio) else 0.0
        # Stretch expression hold to cover most of the clip
        if dur_audio > 0.2:
            hld = max(hld, max(0.15, dur_audio - atk - rel))
        expr_dur = float(atk + hld + rel)
        total = max(dur_audio, expr_dur * 0.9, 0.28)

        print(
            f"[coordinator] TOKEN [{token}] PRE-RECORDED "
            f"clip={os.path.basename(clip_path) if clip_path else 'NONE'} "
            f"audio={dur_audio:.2f}s lips={len(lip_frames)} play={total:.2f}s emotion={emo_label}"
        )

        if audio is not None and len(audio) > 0:
            sd.play(np.ascontiguousarray(audio, dtype=np.float32), int(sr), blocking=False)

        start = time.perf_counter()
        frame_dt = 1.0 / self.fps
        next_deadline = start + frame_dt

        while True:
            now = time.perf_counter()
            t = now - start
            if t >= total:
                break

            # Expression envelope across full play length
            if t < atk:
                w = t / max(atk, 1e-4)
            elif t < atk + hld:
                w = 1.0
            elif t < atk + hld + rel:
                w = (atk + hld + rel - t) / max(rel, 1e-4)
            else:
                w = 0.15 if t < dur_audio else 0.0
            w = max(0.0, min(1.0, w))
            w = w * w * (3.0 - 2.0 * w)
            w *= max(0.6, min(1.0, float(intensity) + 0.3))

            # Mouth from wav2arkit on the clip (primary) + recipe jaw extras
            mouth: Dict[str, float] = {}
            if lip_frames and wav2arkit is not None:
                mouth = dict(wav2arkit.frame_at_time(lip_frames, min(t, dur_audio), fps=lip_fps))
            for k, v in recipe.items():
                val = float(v) * w
                if k in self.lips.owned_keys or k in (
                    "jawOpen", "jawForward", "jawLeft", "jawRight",
                    "mouthClose", "mouthFunnel", "mouthPucker",
                ):
                    # max-merge so laugh smile recipe can open mouth with lips track
                    mouth[k] = max(float(mouth.get(k, 0.0)), val)

            # Upper face from recipe (laugh eyes, disgust sneer, etc.)
            upper: Dict[str, float] = {}
            for k, v in recipe.items():
                if k in mouth:
                    continue
                upper[k] = float(v) * w

            # Laugh: boost smile from recipe even if lips track is neutral
            if token in ("laugh", "chuckle", "giggle") and w > 0.2:
                upper["mouthSmileLeft"] = max(upper.get("mouthSmileLeft", 0), 0.75 * w)
                upper["mouthSmileRight"] = max(upper.get("mouthSmileRight", 0), 0.75 * w)
                upper["cheekSquintLeft"] = max(upper.get("cheekSquintLeft", 0), 0.65 * w)
                upper["cheekSquintRight"] = max(upper.get("cheekSquintRight", 0), 0.65 * w)
                upper["eyeSquintLeft"] = max(upper.get("eyeSquintLeft", 0), 0.5 * w)
                upper["eyeSquintRight"] = max(upper.get("eyeSquintRight", 0), 0.5 * w)

            if upper:
                self.send_udp({
                    "type": "emotion",
                    "emotion": emo_label,
                    "blendshapes": upper,
                })
            if mouth:
                self.send_udp({"type": "viseme", "blendshapes": mouth})

            sleep = next_deadline - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            next_deadline += frame_dt
            if next_deadline < time.perf_counter() - frame_dt:
                next_deadline = time.perf_counter() + frame_dt

        try:
            sd.wait()
        except Exception:
            pass

        # Soft settle before next speech segment
        for i in range(1, 8):
            frac = i / 7.0
            ease = frac * frac * (3.0 - 2.0 * frac)
            self.send_udp({
                "type": "viseme",
                "blendshapes": {"jawOpen": 0.0, "mouthClose": 0.04 + 0.02 * (1.0 - ease)},
            })
            upper = {}
            for k, v in recipe.items():
                if k.startswith("jaw") or k.startswith("mouth") and k not in (
                    "mouthSmileLeft", "mouthSmileRight",
                    "mouthFrownLeft", "mouthFrownRight",
                ):
                    continue
                if k in self.lips.owned_keys:
                    continue
                upper[k] = float(v) * (1.0 - ease) * 0.25
            if upper:
                self.send_udp({
                    "type": "emotion",
                    "emotion": emo_label,
                    "blendshapes": upper,
                })
            time.sleep(0.028)

        print(f"[coordinator] TOKEN [{token}] done (pre-recorded + lips) → next segment")

    def play_sentence(
        self,
        ctx: FaceContext,
        audio_path: str,
        sample_rate: int,
        audio_data: Optional[np.ndarray] = None,
    ) -> None:
        # Use mixed audio (speech + reaction sounds) if it was prepared
        _play_path = ctx.extras.get("mixed_audio_path", audio_path)
        if audio_data is not None:
            audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
            sr = int(sample_rate)
        else:
            audio, sr = sf.read(_play_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            sr = int(sr)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        duration = float(len(audio) / float(sr))
        ctx.duration = duration

        hold = _lip_hold_s()
        # Short speech parts (after token split) must not be eaten by lip hold
        if duration < 1.0:
            hold = min(hold, max(0.05, duration * 0.25))
        # Small tail pad so last syllable is not device-clipped
        pad_s = 0.10 if duration >= 0.6 else 0.06
        audio_play = np.concatenate([
            audio,
            np.zeros(int(round(pad_s * sr)), dtype=np.float32),
        ])
        play_dur = float(len(audio_play) / float(sr))

        # MoMask: body starts at A-pose rest first; speech delayed until motion begins
        plan = ctx.extras.get("motion_plan") or {}
        speech_delay = 0.0
        if str(plan.get("engine") or "") == "momask" or ctx.extras.get("momask_action"):
            speech_delay = float(
                plan.get("speech_delay_s")
                or ctx.extras.get("speech_delay_s")
                or 0.0
            )
            if speech_delay <= 0.0:
                # default = BVH rest hold + ease-in @ 20fps (1+6 frames)
                speech_delay = (1 + 6) / 20.0
            if speech_delay > 0.02:
                silence = np.zeros(int(round(speech_delay * sr)), dtype=np.float32)
                audio_play = np.concatenate([silence, audio_play])
                play_dur = float(len(audio_play) / float(sr))

        print(
            f"[coordinator] Play speech={duration:.2f}s pad={pad_s:.2f}s "
            f"body_lead={speech_delay:.2f}s (speech after motion starts) "
            f"lip_hold={hold*1000:.0f}ms brain={ctx.brain_enabled} "
            f"agents={[a.name for a in self.agents]}"
        )

        self.send_udp({
            "type": "rest_pose",
            "smooth": True,
            "blendshapes": emotion_map.NEUTRAL_REST.copy(),
        })

        # --- Movie camera: plan once before audio (additive; face/body unchanged) ---
        # Skip when master film already joined (avoids wiping multi-shot keys)
        if not ctx.extras.get("skip_camera_plan"):
            self._send_camera_plan(ctx, play_dur)

        # --- Face timeline bake (scrub/replay lips+expression after play) ---
        # Still bake-then-play: full face track is computed offline, then live UDP plays.
        # skip_face_timeline_bake: multi-shot master already installed
        self._send_face_timeline_bake(ctx, duration)

        # --- SYNC: body starts at t=0 (rest→motion); speech after delay ---
        ctx.extras["audio_duration"] = duration
        ctx.extras["speech_delay_s"] = speech_delay
        # Audio window (speech + lead silence + pad)
        body_dur = float(play_dur)
        # MoMask one-shots: keep natural action length when speech is short
        # (e.g. "a person jumps" must not squash jump into ~1s then rest)
        if str(plan.get("engine") or "") == "momask" or ctx.extras.get("momask_action"):
            plan_body = (
                plan.get("duration")
                or plan.get("body_play_s")
                or ctx.extras.get("body_play_s")
            )
            try:
                plan_body_f = float(plan_body) if plan_body not in (None, "") else 0.0
            except (TypeError, ValueError):
                plan_body_f = 0.0
            if plan_body_f <= 0.05:
                # Derive from action_frames @ clip_fps if plan omitted duration
                try:
                    af = int(plan.get("action_frames") or 0)
                    cfp = float(plan.get("clip_fps") or 20.0)
                    if af >= 2 and cfp > 1.0:
                        plan_body_f = float(af) / cfp
                except (TypeError, ValueError):
                    plan_body_f = 0.0
            if plan_body_f > body_dur:
                body_dur = plan_body_f
        sd.play(audio_play, sr, blocking=False)
        start = time.perf_counter()
        play_t0 = time.time()
        ctx.extras["play_t0"] = play_t0
        ctx.extras["body_play_s"] = body_dur
        self._send_body_sync_start(ctx, body_dur, play_t0)

        ctx.is_speaking = True
        ctx.extras["blend_from"] = dict(self._last_upper)
        ctx.extras["blend_dur"] = 0.25  # seconds to blend from prev emotion

        frame_dt = 1.0 / self.fps
        next_deadline = start + frame_dt
        frame_i = 0
        last_mouth: Dict[str, float] = {}
        last_upper: Dict[str, float] = {}

        print(
            f"[coordinator] SYNC play_t0={play_t0:.3f} body_dur={body_dur:.2f}s "
            f"audio={play_dur:.2f}s speech_delay={speech_delay:.2f}s fps={self.fps:.0f}"
        )

        # Run until both audio and body finish (body may outlast short speech)
        end_wall = max(float(play_dur), float(body_dur))
        while True:
            now = time.perf_counter()
            wall_t = now - start
            if wall_t >= end_wall:
                break

            # Face tracks *spoken* audio (after body lead silence); idle after audio
            face_t = wall_t - hold - speech_delay
            if wall_t >= play_dur:
                # Audio done; hold rest face while body finishes natural motion
                ctx.t = duration
                mouth = {k: 0.0 for k in last_mouth} if last_mouth else {"jawOpen": 0.0, "mouthClose": 0.04}
                upper = {k: float(v) * 0.5 for k, v in last_upper.items()} if last_upper else {}
                head = self.head.get_head(ctx)
                body = self.body.get_body(ctx)
            elif face_t < 0.0:
                ctx.t = 0.0
                mouth = {"jawOpen": 0.0, "mouthClose": 0.04}
                # still ramp upper slightly at t=0 after hold starts empty
                upper = {}
                head = self.head.get_head(ctx)
                body = self.body.get_body(ctx)
            elif wall_t >= duration + speech_delay:
                # audio tail pad: hold last face, soft close mouth
                ctx.t = duration
                mouth = {k: float(v) * 0.88 for k, v in last_mouth.items()} if last_mouth else {}
                upper = dict(last_upper)
                head = self.head.get_head(ctx)
                body = self.body.get_body(ctx)
            else:
                ctx.t = min(face_t, duration)
                mouth, upper, head, body = self._merge_frame(ctx)
                last_mouth, last_upper = mouth, upper

            self._send_frame(ctx, mouth, upper, head, body)

            if frame_i % 15 == 0:
                print(
                    f"[coordinator] audio_t={wall_t:.2f}s face_t={max(0, face_t):.2f}s "
                    f"jaw={last_mouth.get('jawOpen', mouth.get('jawOpen', 0)):.3f} "
                    f"smile={last_upper.get('mouthSmileLeft', 0):.2f} "
                    f"act={self.body.current_action_label(ctx)} "
                    f"E={ctx.speech_energy():.2f}"
                )

            frame_i += 1
            sleep = next_deadline - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            next_deadline += frame_dt
            if next_deadline < time.perf_counter() - frame_dt:
                next_deadline = time.perf_counter() + frame_dt

        sd.wait()
        ctx.is_speaking = False
        self._last_upper = dict(last_upper)  # save for next sentence blend
        self._settle(ctx, last_mouth, last_upper)

    def _settle(
        self,
        ctx: FaceContext,
        last_mouth: Dict[str, float],
        last_upper: Dict[str, float],
    ) -> None:
        """One smooth ease last pose → rest (no emotion flash)."""
        print("[coordinator] Smooth settle → rest")
        rest = emotion_map.NEUTRAL_REST
        steps = 16
        for i in range(1, steps + 1):
            frac = i / steps
            ease = frac * frac * (3.0 - 2.0 * frac)
            mouth = {}
            for k in set(last_mouth) | set(self.lips.owned_keys):
                a = float(last_mouth.get(k, 0.0))
                b = float(rest.get(k, 0.0))
                mouth[k] = a * (1.0 - ease) + b * ease
            upper = {}
            for k in set(last_upper) | set(rest.keys()):
                if k in self.lips.owned_keys:
                    continue
                a = float(last_upper.get(k, 0.0))
                b = float(rest.get(k, 0.0))
                upper[k] = a * (1.0 - ease) + b * ease
            if mouth:
                self.send_udp({"type": "viseme", "blendshapes": mouth})
            if upper:
                self.send_udp({
                    "type": "emotion",
                    "emotion": "neutral",
                    "blendshapes": upper,
                })
            time.sleep(0.033)

        self.send_udp({
            "type": "rest_pose",
            "smooth": True,
            "blendshapes": emotion_map.NEUTRAL_REST.copy(),
        })
        # Body: stop clip and ease to A-pose / hands-down idle at SAME location
        self.send_udp({
            "type": "body",
            "rest": True,
            "smooth": True,
            "rest_duration": 0.70,  # blend into hands-down A-pose idle (not T-pose)
        })
        # Camera: hold framing (no hard rest snap — that looked like flicker)
        self.send_udp({"type": "camera", "op": "rest", "hold_after": False})
        print("[coordinator] Done (face + body idle; camera hold continuous).")
