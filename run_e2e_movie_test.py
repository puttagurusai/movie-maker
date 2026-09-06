#!/usr/bin/env python
"""
run_e2e_movie_test.py — full pipeline performance test (~1 min meaningful clip).

Exercises:
  - multi-shot movie package (mavie stages 1–8)
  - MotionRouter (catalog / MoMask)
  - face: lips + Brain/A2E (optional --no-brain)
  - all camera sizes: ECU, CU, MCU, MS, MLS, WS
  - camera moves: static, dolly_in, dolly_out, truck, crane_up
  - master timeline + validation
  - mesh preview MP4 export

Writes a detailed timing log:
  temp/movies/e2e_*/E2E_PIPELINE_LOG.txt
  temp/movies/e2e_*/E2E_PIPELINE_LOG.json

Usage:
  python run_e2e_movie_test.py
  python run_e2e_movie_test.py --no-brain          # faster face
  python run_e2e_movie_test.py --skip-preview      # bake only
  python run_e2e_movie_test.py --play              # also live Blender play
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

SCRIPT_PATH = ROOT / "tests" / "movie_e2e_1min_script.json"


def re_search_stage_in_spoken(spoken: str) -> bool:
    """True if spoken still looks like stage direction leaked into TTS."""
    return bool(
        re.search(
            r"(?i)\b(walk forward|while saying|wave and say|celebrate and say)\b",
            spoken or "",
        )
    )


class StepLog:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.steps: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.meta: Dict[str, Any] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "cwd": str(ROOT),
        }

    def step(self, name: str, **extra: Any) -> "_StepCtx":
        return _StepCtx(self, name, extra)

    def add(self, name: str, status: str, duration_s: float, **extra: Any) -> None:
        row = {
            "step": name,
            "status": status,
            "duration_s": round(duration_s, 3),
            "t_since_start_s": round(time.perf_counter() - self.t0, 3),
            **extra,
        }
        self.steps.append(row)
        mark = "OK" if status == "ok" else status.upper()
        print(f"  [{mark}] {name}  {duration_s:.2f}s")

    def total_s(self) -> float:
        return time.perf_counter() - self.t0

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.meta,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "total_s": round(self.total_s(), 3),
            "steps": self.steps,
            "errors": self.errors,
        }

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        jpath = out_dir / "E2E_PIPELINE_LOG.json"
        tpath = out_dir / "E2E_PIPELINE_LOG.txt"
        data = self.to_dict()
        jpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
        lines = [
            "=" * 72,
            "E2E MOVIE PIPELINE TEST LOG",
            f"started : {data.get('started_at')}",
            f"finished: {data.get('finished_at')}",
            f"total   : {data.get('total_s')}s",
            "=" * 72,
            "",
            "STEPS:",
        ]
        for s in self.steps:
            lines.append(
                f"  {s['t_since_start_s']:>8.2f}s  +{s['duration_s']:>7.2f}s  "
                f"[{s['status']:6}] {s['step']}"
            )
            for k, v in s.items():
                if k in ("step", "status", "duration_s", "t_since_start_s"):
                    continue
                if isinstance(v, (dict, list)):
                    lines.append(f"           {k}: {json.dumps(v)[:200]}")
                else:
                    lines.append(f"           {k}: {v}")
        if self.errors:
            lines.append("")
            lines.append("ERRORS:")
            for e in self.errors:
                lines.append(f"  - {e}")
        lines.append("")
        lines.append(f"JSON: {jpath}")
        tpath.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[log] {tpath}")
        print(f"[log] {jpath}")


class _StepCtx:
    def __init__(self, log: StepLog, name: str, extra: Dict[str, Any]) -> None:
        self.log = log
        self.name = name
        self.extra = extra
        self.t0 = 0.0

    def __enter__(self) -> "_StepCtx":
        self.t0 = time.perf_counter()
        print(f"\n>>> START {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        dt = time.perf_counter() - self.t0
        if exc is not None:
            self.log.errors.append(f"{self.name}: {exc}")
            self.log.add(self.name, "fail", dt, error=str(exc), **self.extra)
            print(f">>> FAIL {self.name}: {exc}")
            traceback.print_exc()
            return True  # swallow so suite continues where possible
        self.log.add(self.name, "ok", dt, **self.extra)
        print(f">>> DONE {self.name} ({dt:.2f}s)")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="E2E ~1min movie pipeline test with full camera gallery")
    ap.add_argument("--script", type=str, default=str(SCRIPT_PATH), help="JSON shot list")
    ap.add_argument("--no-brain", action="store_true", help="Skip Brain/A2E for speed")
    ap.add_argument("--skip-preview", action="store_true", help="Skip MP4 export")
    ap.add_argument("--play", action="store_true", help="Live UDP play after bake")
    ap.add_argument("--llm", action="store_true", help="Use LLM movie director")
    ap.add_argument("--title", type=str, default="e2e_1min_camera_gallery")
    ap.add_argument("--res", type=str, default="640x360", help="Preview resolution")
    ap.add_argument(
        "--preview-mode",
        type=str,
        default="fast",
        choices=("fast", "short", "long", "auto", "full"),
        help="Preview export: fast (default, frame-step) | full (every frame, slow)",
    )
    ap.add_argument("--frame-step", type=int, default=0, help="Override preview frame step (fast default 3)")
    ap.add_argument("--target-s", type=float, default=60.0, help="Director target total duration")
    args = ap.parse_args()

    # Production-ish env for this test
    os.environ.setdefault("USE_MOVIE_CAMERA", "1")
    os.environ.setdefault("USE_FACE_TIMELINE", "1")
    os.environ.setdefault("MOMASK_ALL", "0")
    # Allow MoMask gen when director/router sets allow_sync_gen (needed t2m only)
    os.environ.setdefault("MOMASK_SYNC", "1")
    os.environ.setdefault("USE_MOMASK", "1")
    os.environ.setdefault("MOVIE_MIN_TAKE_S", "5.5")
    os.environ.setdefault("MOVIE_CAM_MIN_DIST", "1.08")
    if args.no_brain:
        os.environ["USE_BRAIN"] = "0"

    log = StepLog()
    log.meta.update({
        "title": args.title,
        "script": args.script,
        "use_brain": not args.no_brain,
        "use_llm": bool(args.llm),
        "skip_preview": args.skip_preview,
        "play": args.play,
        "target_s": args.target_s,
        "camera_sizes": ["ECU", "CU", "MCU", "MS", "MLS", "WS"],
        "camera_moves": [
            "static", "dolly_in", "dolly_out", "truck", "truck_left", "truck_right",
            "crane_up", "crane_down", "pan_left", "pan_right", "orbit", "arc",
            "tilt_up", "tilt_down", "reveal", "follow", "handheld",
        ],
        "camera_roles": ["A_cam", "B_cam", "C_cam", "env_cam"],
    })

    script_path = Path(args.script)
    if not script_path.is_file():
        print(f"Missing script: {script_path}")
        return 1

    with log.step("load_script", path=str(script_path)):
        script = json.loads(script_path.read_text(encoding="utf-8"))
        n_shots = len(script.get("shots") or [])
        cams = [(s.get("camera_shot"), s.get("camera_move")) for s in script.get("shots") or []]
        log.meta["n_shots_planned"] = n_shots
        log.meta["forced_cameras"] = cams

    print("=" * 72)
    print("E2E MOVIE PIPELINE TEST — ~1 min camera gallery")
    print(f"  shots={n_shots}  brain={not args.no_brain}  preview={not args.skip_preview}")
    print("  cameras forced:", cams)
    print("=" * 72)

    pkg = None
    llm = None
    with log.step("import_pipeline"):
        from face_agents.movie_production.pipeline import MovieProductionPipeline

    if args.llm:
        with log.step("load_llm_director"):
            from llm_fw.providers import get_provider
            cfg_path = ROOT / "llm_fw" / "config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
            cfg = dict(cfg)
            cfg["max_tokens"] = max(int(cfg.get("max_tokens") or 1024), 4096)
            llm = get_provider(cfg)
            log.meta["llm_model"] = getattr(llm, "model", "")

    with log.step("create_pipeline", use_brain=not args.no_brain, use_llm=bool(llm)):
        pipe = MovieProductionPipeline(
            use_brain=not args.no_brain,
            use_movie_camera=True,
            use_face_timeline=True,
            llm_provider=llm,
            target_total_s=float(args.target_s),
        )

    if args.play:
        print(">>> --play: ensure Blender stream_receiver is already running (no wait).")

    with log.step("bake_sequence", n_shots=n_shots):
        # Intelligent director enriches script (durations, multi-cam, pace, holds)
        pkg = pipe.run_script(script, title=args.title, play=False)

    # Enrich log with per-shot bake stats
    if pkg is not None:
        shot_stats = []
        cam_used = set()
        moves_used = set()
        roles_used = set()
        for s in pkg.shots:
            cam = s.camera or {}
            shot_stats.append({
                "shot_id": s.shot_id,
                "duration_s": round(s.duration_s, 3),
                "speech_duration_s": round(getattr(s, "speech_duration_s", 0) or 0, 3),
                "target_duration_s": round(getattr(s, "target_duration_s", 0) or 0, 3),
                "t0": round(s.t0, 3),
                "t1": round(s.t1, 3),
                "emotion": s.emotion,
                "stage": (getattr(s, "stage", "") or "")[:80],
                "spoken": (getattr(s, "spoken", "") or "")[:80],
                "body_mode": s.body_mode,
                "body_action": s.body_action,
                "allow_sync_gen": bool(getattr(s, "allow_sync_gen", False)),
                "humanml_prompt": (getattr(s, "humanml_prompt", "") or "")[:100],
                "clip_policy": getattr(s, "clip_policy", ""),
                "clip_speed": round(float(getattr(s, "clip_speed", 1.0) or 1.0), 3),
                "motion_reason": s.motion_reason,
                "camera_role": cam.get("camera_role") or getattr(s, "camera_role", ""),
                "camera_shot": cam.get("shot") or s.camera_shot,
                "camera_move": cam.get("move_type") or s.camera_move,
                "pace": getattr(s, "pace", ""),
                "bake_ok": s.bake_ok,
                "bake_ms": round(s.bake_ms, 1),
                "bake_errors": list(s.bake_errors),
                "audio": bool(s.audio_path),
                "face_timeline": bool(s.face_timeline_path),
            })
            if cam.get("shot"):
                cam_used.add(cam.get("shot"))
            if cam.get("move_type"):
                moves_used.add(cam.get("move_type"))
            role = cam.get("camera_role") or getattr(s, "camera_role", "")
            if role:
                roles_used.add(role)
        log.meta["output_dir"] = pkg.output_dir
        log.meta["duration_s"] = round(pkg.duration_s, 3)
        log.meta["validation"] = pkg.validation
        log.meta["director_source"] = getattr(pkg, "director_source", "")
        log.meta["shot_stats"] = shot_stats
        log.meta["cameras_observed"] = sorted(cam_used)
        log.meta["moves_observed"] = sorted(moves_used)
        log.meta["roles_observed"] = sorted(roles_used)
        log.meta["continuity"] = getattr(pkg, "continuity", None)
        log.steps[-1]["extra_summary"] = {
            "duration_s": pkg.duration_s,
            "n_shots": len(pkg.shots),
            "validation_ok": (pkg.validation or {}).get("ok"),
            "director_source": getattr(pkg, "director_source", ""),
        }

        planned = {s.get("shot_id"): s for s in script.get("shots") or []}
        for st in shot_stats:
            p = planned.get(st["shot_id"]) or {}
            st["camera_shot_forced"] = p.get("camera_shot")
            st["camera_move_forced"] = p.get("camera_move")

    preview_path = None
    if not args.skip_preview and pkg is not None:
        with log.step("export_preview_mp4", res=args.res):
            # fast = 1× Blender + frame-step (default). full = every timeline frame (slow).
            pmode = args.preview_mode
            if pmode == "full":
                pmode = "short"
                step = 1
            elif pmode == "fast":
                pmode = "fast"
                step = args.frame_step or 3
            else:
                step = args.frame_step or 0
            from export_movie_preview_mp4 import main as export_main
            import sys as _sys

            pkg_json = str(Path(pkg.output_dir) / "movie_package.json")
            out_mp4 = str(Path(pkg.output_dir) / "e2e_preview_mesh_audio.mp4")
            _sys.argv = [
                "export_movie_preview_mp4.py",
                "--package", pkg_json,
                "--out", out_mp4,
                "--mode", pmode,
                "--res", args.res,
                "--still-format", "JPEG",
            ]
            if step:
                _sys.argv += ["--frame-step", str(step)]
            code = export_main()
            if code != 0:
                raise RuntimeError(f"export_movie_preview_mp4 exit {code}")
            preview_path = out_mp4
            log.meta["preview_mp4"] = preview_path
            log.meta["preview_mode"] = args.preview_mode
            log.meta["preview_frame_step"] = step

    if args.play and pkg is not None:
        with log.step("live_play_sequence"):
            pipe.play_package(pkg)

    # Coverage report
    with log.step("coverage_report"):
        wanted_shots = {"ECU", "CU", "MCU", "MS", "MLS", "WS"}
        wanted_moves = {"static", "dolly_in", "dolly_out", "truck", "crane_up"}
        got_s = set(log.meta.get("cameras_observed") or [])
        got_m = set(log.meta.get("moves_observed") or [])
        got_r = set(log.meta.get("roles_observed") or [])
        forced_s = {c[0] for c in log.meta.get("forced_cameras") or [] if c[0]}
        forced_m = {c[1] for c in log.meta.get("forced_cameras") or [] if c[1]}
        # also from shot_stats (director may change)
        for st in log.meta.get("shot_stats") or []:
            if st.get("camera_shot"):
                got_s.add(st["camera_shot"])
            if st.get("camera_move"):
                got_m.add(st["camera_move"])
            if st.get("camera_role"):
                got_r.add(st["camera_role"])
        stats = log.meta.get("shot_stats") or [{"duration_s": 0}]
        min_take = min((st.get("duration_s") or 0) for st in stats)
        momask_n = sum(1 for st in stats if (st.get("body_mode") or "") == "momask")
        catalog_n = sum(1 for st in stats if (st.get("body_mode") or "") == "catalog")
        slow_n = sum(1 for st in stats if float(st.get("clip_speed") or 1) < 0.9)
        spoken_ok = sum(
            1 for st in stats
            if st.get("spoken") and not re_search_stage_in_spoken(st.get("spoken") or "")
        )
        coverage = {
            "shot_sizes_planned": sorted(forced_s),
            "shot_sizes_observed": sorted(got_s),
            "shot_sizes_missing": sorted(wanted_shots - (got_s | forced_s)),
            "moves_planned": sorted(forced_m),
            "moves_observed": sorted(got_m),
            "moves_missing": sorted(wanted_moves - (got_m | forced_m)),
            "roles_observed": sorted(got_r),
            "duration_s": log.meta.get("duration_s"),
            "min_take_s": min_take,
            "target_about_1_min": abs(float(log.meta.get("duration_s") or 0) - 60) < 25,
            "cinematic_takes": min_take >= 5.0,
            "director_source": log.meta.get("director_source"),
            "body_catalog_shots": catalog_n,
            "body_momask_t2m_shots": momask_n,
            "slow_mo_or_slow_clips": slow_n,
            "spoken_without_stage_verbs": spoken_ok,
            "n_shots": len(stats),
        }
        log.meta["coverage"] = coverage
        print("  coverage:", json.dumps(coverage, indent=2))

    # Write logs next to package
    out_dir = Path(pkg.output_dir) if pkg else (ROOT / "temp" / "movies" / "e2e_failed")
    log.meta["preview_mp4"] = preview_path
    log.write(out_dir)

    # Human summary
    print("\n" + "=" * 72)
    print("E2E SUMMARY")
    print(f"  total wall time : {log.total_s():.1f}s")
    print(f"  movie duration  : {log.meta.get('duration_s')}s")
    print(f"  shots           : {log.meta.get('n_shots_planned')}")
    print(f"  validation      : {(log.meta.get('validation') or {}).get('ok')}")
    print(f"  package         : {log.meta.get('output_dir')}")
    print(f"  preview mp4     : {preview_path}")
    print(f"  log             : {out_dir / 'E2E_PIPELINE_LOG.txt'}")
    print("=" * 72)
    return 0 if not log.errors else 1


if __name__ == "__main__":
    # Fix MovieProductionPipeline constructor if it doesn't take title_hint
    raise SystemExit(main())
