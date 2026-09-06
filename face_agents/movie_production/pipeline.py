"""
MovieProductionPipeline — end-to-end production runner (mavie.txt).

  Director (LLM + ContinuityBoard) → bake workers → validate → package
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..coordinator import FaceCoordinator
from .baker import bake_sequence
from .continuity_memory import ContinuityBoard
from .planner import plan_with_board
from .shot_schema import MoviePackage, ShotDetail
from .validate import validate_package
from .world_state import WorldState

ROOT = Path(__file__).resolve().parents[2]


class MovieProductionPipeline:
    def __init__(
        self,
        *,
        fps: float = 20.0,
        use_brain: bool = True,
        use_movie_camera: bool = True,
        use_face_timeline: bool = True,
        output_root: Optional[Path] = None,
        llm_provider: Any = None,
        device: Optional[str] = None,
        target_total_s: float = 60.0,
    ) -> None:
        self.fps = float(fps)
        self.use_brain = use_brain
        self.llm = llm_provider
        self.device = device
        self.target_total_s = float(target_total_s)
        self.output_root = Path(output_root or (ROOT / "temp" / "movies"))
        os.environ.setdefault("USE_MOVIE_CAMERA", "1" if use_movie_camera else "0")
        os.environ.setdefault("USE_FACE_TIMELINE", "1" if use_face_timeline else "0")
        os.environ.setdefault("MOMASK_ALL", "0")
        if "MOMASK_SYNC" not in os.environ:
            os.environ["MOMASK_SYNC"] = "0"

        self.coord = FaceCoordinator(
            udp_ip="127.0.0.1",
            udp_port=9001,
            fps=self.fps,
            use_brain=use_brain,
            device=device,
        )
        self.world = WorldState()
        self.world.ensure_character("hero")
        self.board = ContinuityBoard(target_total_s=self.target_total_s)

    def run_story(
        self,
        story: str,
        *,
        title: str = "movie",
        play: bool = False,
    ) -> MoviePackage:
        t0 = time.time()
        print("=" * 64)
        print("MOVIE PRODUCTION PIPELINE (intelligent director)")
        print("=" * 64)
        print(f"Story: {story[:120]}{'…' if len(story)>120 else ''}")
        print(f"  llm={'yes' if self.llm else 'no (rules)'}  target≈{self.target_total_s}s")

        print("\n[Stage 1-2] Director plan (ContinuityBoard + LLM/rules)…")
        shots, self.board = plan_with_board(
            story=story,
            title=title,
            llm_provider=self.llm,
            world=self.world,
            target_total_s=self.target_total_s,
        )
        print(f"  → {len(shots)} take(s) source={self.board.director_source}")
        for s in shots:
            print(
                f"     {s.shot_id}: {s.target_duration_s:.1f}s "
                f"{s.camera_role}/{s.camera_shot}/{s.camera_move} "
                f"body={s.body_mode} | {s.text[:45]}"
            )

        return self._bake_validate_play(shots, title=title, story=story, play=play, t0=t0)

    def run_script(
        self,
        script: Dict[str, Any] | List[Any],
        *,
        title: str = "movie",
        play: bool = False,
    ) -> MoviePackage:
        t0 = time.time()
        story = title
        if isinstance(script, dict):
            story = str(script.get("story") or script.get("title") or title)
            if script.get("title") and title == "movie":
                title = str(script.get("title"))
        print("=" * 64)
        print("MOVIE PRODUCTION PIPELINE (script + intelligent director)")
        print("=" * 64)
        print(f"  llm={'yes' if self.llm else 'no (rules)'}  target≈{self.target_total_s}s")

        print("\n[Stage 1-2] Director enrich script (durations, multi-cam, pace)…")
        shots, self.board = plan_with_board(
            script=script,
            title=title,
            llm_provider=self.llm,
            world=self.world,
            target_total_s=self.target_total_s,
        )
        print(f"  → {len(shots)} take(s) source={self.board.director_source}")
        for s in shots:
            print(
                f"     {s.shot_id}: {s.target_duration_s:.1f}s "
                f"{s.camera_role}/{s.camera_shot}/{s.camera_move} "
                f"pace={s.pace} | {s.text[:45]}"
            )
        return self._bake_validate_play(shots, title=title, story=story, play=play, t0=t0)

    def _bake_validate_play(
        self,
        shots: List[ShotDetail],
        *,
        title: str,
        story: str,
        play: bool,
        t0: float,
    ) -> MoviePackage:
        run_dir = self.output_root / f"{_slug(title)}_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print("\n[Stage 4-6] Bake workers (TTS | face || body | multi-cam | face timeline)…")
        if self.use_brain:
            try:
                import brain_inference
                import torch

                dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                brain_inference.load_brain_model(dev)
            except Exception as e:
                print(f"  [warn] brain preload: {e}")

        baked = bake_sequence(
            shots,
            coord=self.coord,
            world=self.world,
            out_dir=run_dir,
            fps=self.fps,
            board=self.board,
        )

        print("\n[Stage 8] Validate…")
        validation = validate_package(baked, world=self.world, fps=self.fps)
        print(f"  ok={validation['ok']} errors={len(validation['errors'])} warnings={len(validation['warnings'])}")
        for e in validation["errors"][:8]:
            print(f"  ERROR: {e}")
        for w in validation["warnings"][:8]:
            print(f"  WARN: {w}")

        # Persist continuity board (LLM external memory dump)
        self.board.save(run_dir / "continuity_board.json")

        pkg = MoviePackage(
            title=title,
            story=story,
            fps=self.fps,
            shots=baked,
            world_final=self.world.snapshot(),
            continuity=self.board.to_dict(),
            validation=validation,
            output_dir=str(run_dir),
            director_source=self.board.director_source,
        )
        master_path = run_dir / "movie_package.json"
        master_path.write_text(json.dumps(pkg.to_dict(), indent=2), encoding="utf-8")
        master_tl = {
            "title": title,
            "fps": self.fps,
            "duration_s": pkg.duration_s,
            "director_source": pkg.director_source,
            "clips": [
                {
                    "shot_id": s.shot_id,
                    "t0": s.t0,
                    "t1": s.t1,
                    "duration_s": s.duration_s,
                    "speech_duration_s": s.speech_duration_s,
                    "target_duration_s": s.target_duration_s,
                    "audio": s.audio_path,
                    "text": s.text,
                    "stage": s.stage,
                    "spoken": s.spoken,
                    "emotion": s.emotion,
                    "body_mode": s.body_mode,
                    "body_action": s.body_action,
                    "body_library": s.body_library,
                    "clip_policy": s.clip_policy,
                    "clip_speed": s.clip_speed,
                    # Native Action rate for preview (MoMask=20, catalog≈30) — do not stretch
                    "clip_fps": 20.0 if (
                        str(s.body_mode or "").lower() in ("momask", "both")
                        or str(s.body_action or "").startswith("momask_")
                    ) else float(pkg.fps or 30),
                    "allow_sync_gen": s.allow_sync_gen,
                    "humanml_prompt": s.humanml_prompt,
                    "motion_reason": s.motion_reason,
                    "face_timeline": s.face_timeline_path,
                    "camera": s.camera,
                    "camera_role": s.camera_role,
                    "camera_shot": s.camera_shot,
                    "camera_move": s.camera_move,
                    "pace": s.pace,
                    "transition_in": s.transition_in,
                    "director_notes": s.director_notes,
                }
                for s in baked
            ],
        }
        (run_dir / "master_timeline.json").write_text(
            json.dumps(master_tl, indent=2), encoding="utf-8"
        )
        print(f"\n[Stage 7] Package → {master_path}")
        print(f"  duration={pkg.duration_s:.2f}s  shots={len(baked)}  dir={run_dir}")
        print(f"  cameras_created={self.board.cameras_created} moves={self.board.moves_used}")
        print(f"  total wall={(time.time()-t0):.1f}s")

        if play:
            print("\n[Play] Sequential synced playback…")
            self.play_package(pkg)

        return pkg

    def install_master_timeline(self, pkg: MoviePackage) -> None:
        """
        Install the full film into Blender Session SoT (same path as chat).

        Body → Session append, face keyframes, SessionCam bake, Look_Set + NPCs.
        This IS the movie pipeline — not a side path.
        """
        from .session_install import install_package_into_session

        print("\n[Join] Session SoT install (body + face + camera + look)…")
        report = install_package_into_session(
            self.coord,
            pkg,
            reset=True,
            apply_look=True,
        )
        if not report.get("ok", True) or report.get("errors"):
            for e in report.get("errors") or []:
                print(f"  WARN: {e}")
        print(
            f"[Join] Session ready shots={report.get('shots')} "
            f"frames→{report.get('frame_end')} look={bool(report.get('look'))}"
        )

    def play_package(self, pkg: MoviePackage, *, join_timeline: bool = True) -> None:
        """
        Live-play all shots in order.

        join_timeline=True (default): first install full master face+camera keys so
        shot_01 is NOT wiped when shot_02 plays — scrub still has the whole film.
        """
        import soundfile as sf

        if join_timeline:
            self.install_master_timeline(pkg)

        for i, sh in enumerate(pkg.shots):
            if not sh.bake_ok or not sh.audio_path:
                print(f"  skip {sh.shot_id} (not baked)")
                continue
            print(f"\n--- PLAY {sh.shot_id} [{sh.t0:.2f}-{sh.t1:.2f}] "
                  f"{sh.camera_role}/{sh.camera_shot}/{sh.camera_move} ---")
            # During sequential live play: do NOT clear master keys again.
            # Only re-send face if join was skipped.
            if not join_timeline and sh.face_timeline_path and Path(sh.face_timeline_path).is_file():
                self.coord.send_udp({
                    "type": "face_keyframes",
                    "path": str(Path(sh.face_timeline_path).resolve()),
                    "set_frame_range": True,
                    "clear_previous": (i == 0),
                })
            if not join_timeline and sh.camera and os.environ.get("USE_MOVIE_CAMERA", "1") not in ("0", "false"):
                cam = sh.camera
                self.coord.send_udp({
                    "type": "camera",
                    "op": "plan",
                    "duration": float(sh.duration_s),
                    "fps": float(pkg.fps),
                    "shot": cam.get("shot") or sh.camera_shot or "",
                    "move_type": cam.get("move_type") or sh.camera_move or "",
                    "track_head": bool(cam.get("track_head", False)),
                    "camera_name": cam.get("camera_name") or "MovieCam_A",
                    "look_at_name": cam.get("look_at_name") or "MovieCam_A_LookAt",
                    "camera_role": cam.get("camera_role") or sh.camera_role,
                    "set_scene_camera": True,
                    "bake_keyframes": True,
                    "clear_previous": (i == 0),
                    "min_cam_dist": float(os.environ.get("MOVIE_CAM_MIN_DIST", "1.08")),
                    "keyframes": cam.get("keyframes") or [],
                })

            audio, sr = sf.read(sh.audio_path, dtype="float32")
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            ctx = getattr(sh, "_ctx", None)
            if ctx is None:
                ctx = self.coord.prepare_sentence(
                    text=sh.spoken or sh.text,
                    emotion=sh.emotion,
                    intensity=sh.intensity,
                    audio_path=sh.audio_path,
                    duration=sh.speech_duration_s or sh.duration_s,
                    sample_rate=int(sr),
                    body_actions=sh.actions,
                    body_state=sh.state,
                    body_mode=sh.body_mode,
                    humanml_prompt=sh.humanml_prompt,
                    momask_action=sh.body_action,
                    momask_library=sh.body_library,
                    camera_shot=sh.camera_shot,
                    camera_move=sh.camera_move,
                )
            else:
                if sh.body_action:
                    ctx.extras["momask_action"] = sh.body_action
                    ctx.extras["momask_library"] = sh.body_library or ""
                if sh.camera:
                    ctx.extras["camera_plan"] = sh.camera

            # Live stream must not clear the joined master face/camera timeline
            if join_timeline:
                ctx.extras["skip_face_timeline_bake"] = True
                ctx.extras["skip_camera_plan"] = True
                ctx.extras["face_clear_previous"] = False

            self.coord.play_sentence(ctx, sh.audio_path, int(sr), audio_data=audio)
            time.sleep(0.12)

        print("\n[Play] sequence complete — full film still on timeline (scrub frame 1…end).")


def run_movie_from_story(
    story: str,
    *,
    title: str = "movie",
    play: bool = False,
    use_brain: bool = True,
) -> MoviePackage:
    pipe = MovieProductionPipeline(use_brain=use_brain)
    return pipe.run_story(story, title=title, play=play)


def _slug(s: str) -> str:
    import re

    s = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "movie").strip())[:40]
    return s.strip("_").lower() or "movie"
