"""CPU QC + speech-safe + mix delay (no GPU / no Blender)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from face_agents.qc_gates import (
    adelay_ms,
    qc_cut,
    qc_hml,
    qc_look,
    qc_sil,
    run_cpu_gates,
)
from face_agents.speech_safe import snap_range_speech_safe
from face_agents.look_schema import LookPlan
from tools.blender_movie_render import adelay_ms as mix_adelay_ms


def test_adelay_golden():
    # clip 1 speech_frame_start=8 → 350 ms @ 20 fps
    assert adelay_ms(8, 20.0) == 350
    assert mix_adelay_ms(8, 20.0) == 350
    # second clip speech_frame_start != 1
    assert adelay_ms(85, 20.0) == 4200
    assert adelay_ms(1, 20.0) == 0
    assert adelay_ms(0, 20.0) == 0
    print("  OK adelay golden 8→350ms 85→4200ms")


def test_speech_safe_snap():
    clips = [
        {
            "index": 1,
            "frame_start": 1,
            "frame_end": 80,
            "speech_frame_start": 8,
            "speech_frame_end": 60,
        }
    ]
    a, b, note = snap_range_speech_safe(clips, 20, 40)
    assert a == 8 and b == 60, (a, b, note)
    assert "snapped" in note
    a, b, note = snap_range_speech_safe(clips, 1, 4)
    assert a == 1 and b == 4 and note == ""
    cut = qc_cut(clips, 20, 40)
    assert cut and "S#1" in cut
    print("  OK speech-safe snap + QC-CUT")


def test_hml_and_look():
    assert qc_hml("a person walks forward", engine="momask") is None
    assert qc_hml("do a backflip", engine="momask")
    errs = qc_look({"location": "street", "time_of_day": "golden_hour", "grade": "warm"})
    assert not errs, errs
    leak = qc_look({"location": "studio"}, "a person walks at golden hour")
    assert leak, leak
    print("  OK QC-HML / QC-LOOK")


def test_sil_and_cpu_bundle():
    sil = {"index": 1, "speech_frame_start": 0, "text": "", "motion_only": True}
    assert qc_sil(sil) is None
    bad = {"index": 2, "speech_frame_start": 8, "text": "", "motion_only": True}
    assert qc_sil(bad)
    clips = [
        {
            "index": 1,
            "engine": "momask",
            "humanml_prompt": "a person waves with the right hand",
            "look": LookPlan().to_dict(),
            "speech_frame_start": 0,
            "motion_only": True,
            "text": "",
        },
        {
            "index": 2,
            "engine": "catalog",
            "speech_frame_start": 8,
            "speech_frame_end": 40,
            "speech_duration_s": 1.65,
            "look": LookPlan(grade="warm").to_dict(),
            "text": "hello",
        },
    ]
    r = run_cpu_gates(clips=clips)
    assert r["ok"], r["errors"]
    assert r["where"] == "cpu"
    print("  OK QC-SIL + run_cpu_gates")


def main() -> int:
    print("=" * 50)
    print("QC GATES / SPEECH-SAFE")
    print("=" * 50)
    test_adelay_golden()
    test_speech_safe_snap()
    test_hml_and_look()
    test_sil_and_cpu_bundle()
    print("=" * 50)
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
