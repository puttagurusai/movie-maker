#!/usr/bin/env python
"""
run_movie_pipeline.py — movie-level 3D Session pipeline (primary product).

Goal: story → agentic director → bake → install ONE Blender Session you can Play.
MP4 export is optional and OFF by default (Play Session first).

Examples:
  # Agentic director + Session join (default when LLM config exists)
  python run_movie_pipeline.py --story "Walk on a sunny street, wave hello, then say goodbye."

  # Force rules director
  python run_movie_pipeline.py --story "..." --no-llm

  # Script beats
  python run_movie_pipeline.py --script temp/stories/demo_60s_features.json --title street_60s

  # Live play after install
  python run_movie_pipeline.py --story "..." --play

  # Optional MP4 ONLY after Session Play looks right
  python run_movie_pipeline.py --script ... --export-mp4 --preview-res 1280x720
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def _try_llm(*, force_off: bool, force_on: bool):
    """Default ON when provider/config works; --no-llm forces rules."""
    if force_off:
        return None
    # Default agentic: try load unless explicitly disabled via env
    if not force_on and os.environ.get("MOVIE_DIRECTOR_LLM", "1").strip().lower() in (
        "0", "false", "no", "off",
    ):
        return None
    try:
        from llm_fw.providers import get_provider

        cfg_path = ROOT / "llm_fw" / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        cfg = dict(cfg)
        cfg["max_tokens"] = max(int(cfg.get("max_tokens") or 1024), 4096)
        llm = get_provider(cfg)
        print(f"[llm] AGENTIC director on  model={getattr(llm, 'model', '?')}")
        print("[llm] ContinuityBoard injected each call (model has no memory)")
        return llm
    except Exception as e:
        print(f"[llm] unavailable ({e}) — rules director fallback")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Movie-level 3D Session pipeline (Play in Blender is the product)"
    )
    ap.add_argument("--story", type=str, default="", help="Story / multi-line premise")
    ap.add_argument("--script", type=str, default="", help="JSON shots/beats file")
    ap.add_argument("--title", type=str, default="movie", help="Package title")
    ap.add_argument("--play", action="store_true", help="Live-play after Session install")
    ap.add_argument(
        "--join-timeline",
        action="store_true",
        default=True,
        help="Install Session SoT after bake (default ON)",
    )
    ap.add_argument(
        "--no-join",
        action="store_true",
        help="Skip Session install (bake package only)",
    )
    ap.add_argument("--no-brain", action="store_true", help="Skip Brain/A2E (lips-only face)")
    ap.add_argument("--llm", action="store_true", help="Force try LLM director")
    ap.add_argument("--no-llm", action="store_true", help="Force rules director")
    ap.add_argument("--target-s", type=float, default=60.0, help="Target total duration (seconds)")
    ap.add_argument("--out", type=str, default="", help="Output root directory")
    # Export is OPTIONAL — not the product goal
    ap.add_argument(
        "--export-mp4",
        action="store_true",
        help="Optional: export MP4 AFTER Session install (off by default)",
    )
    ap.add_argument("--preview-mp4", action="store_true", help=argparse.SUPPRESS)  # legacy alias
    ap.add_argument("--preview-res", type=str, default="1280x720")
    ap.add_argument("--preview-step", type=int, default=1)
    args = ap.parse_args()

    if args.no_join:
        args.join_timeline = False
    if args.preview_mp4:
        args.export_mp4 = True

    if not args.story and not args.script:
        args.story = (
            "On a sunny city street, walk forward among people, "
            "stop and wave hello, say welcome friend, then wave goodbye."
        )
        print("[demo story]", args.story)

    llm = _try_llm(force_off=args.no_llm, force_on=args.llm)

    from face_agents.movie_production.pipeline import MovieProductionPipeline

    out = Path(args.out) if args.out else None
    pipe = MovieProductionPipeline(
        use_brain=not args.no_brain,
        llm_provider=llm,
        output_root=out,
        target_total_s=float(args.target_s),
    )

    print(">>> Blender face stream_receiver must be running for Session install/play.")

    if args.script:
        data = json.loads(Path(args.script).read_text(encoding="utf-8"))
        # play=False here; we install Session then optionally play
        pkg = pipe.run_script(data, title=args.title, play=False)
    else:
        pkg = pipe.run_story(args.story, title=args.title, play=False)

    # Default product path: one Session in Blender
    if args.join_timeline:
        print("\n[Product] Installing film into Blender Session SoT…")
        pipe.install_master_timeline(pkg)
        print("  → In Blender: Space / Play Session to review the 3D take")
        print("  → Export only when Play already looks right (--export-mp4)")

    if args.play:
        print("\n[Play] Sequential live play on installed Session…")
        pipe.play_package(pkg, join_timeline=False)

    print("\nDONE (3D Session is the deliverable)")
    print(f"  package: {pkg.output_dir}/movie_package.json")
    print(f"  timeline: {pkg.output_dir}/master_timeline.json")
    print(f"  duration: {pkg.duration_s:.2f}s  shots: {len(pkg.shots)}")
    print(f"  director: {pkg.director_source}")
    print(f"  validation ok: {(pkg.validation or {}).get('ok')}")
    for s in pkg.shots:
        print(
            f"    {s.shot_id}: body_action={s.body_action!r} "
            f"cam={s.camera_role}/{s.camera_shot} spoken={((s.spoken or '')[:40])!r}"
        )

    if args.export_mp4:
        print("\n[optional export] MP4 from package (Session Play should already be good)…")
        from export_movie_preview_mp4 import main as preview_main
        import sys as _sys

        pkg_json = str(Path(pkg.output_dir) / "movie_package.json")
        argv = [
            "export_movie_preview_mp4.py",
            "--package", pkg_json,
            "--mode", "short",
            "--res", str(args.preview_res or "1280x720"),
            "--frame-step", str(max(1, int(args.preview_step or 1))),
            "--still-format", "JPEG",
        ]
        _sys.argv = argv
        preview_main()

    return 0 if (pkg.validation or {}).get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
