"""Baker motion-only: silence WAV, no load_parler / prepare_sentence."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from face_agents.movie_production.shot_schema import ShotDetail
from face_agents.movie_production.world_state import WorldState
from face_agents.movie_production import baker


class _Boom(Exception):
    pass


def test_motion_only_silence(tmp_path: Path):
    shot = ShotDetail(
        shot_id="shot_01",
        text="a person walks forward at a steady pace",
        spoken="",
        stage="a person walks forward at a steady pace",
        humanml_prompt="a person walks forward at a steady pace",
        body_mode="momask",
        target_duration_s=3.5,
        clip_policy="hold_end",
    )
    world = WorldState()
    coord = MagicMock()
    coord.prepare_sentence.side_effect = _Boom("prepare_sentence must not run")
    coord._build_face_timeline.side_effect = _Boom("face timeline must not run")

    def _no_tts(*_a, **_k):
        raise _Boom("load_parler / _tts must not run")

    cam = MagicMock()
    cam.camera_role = "env_cam"
    cam.shot = "WS"
    cam.move_type = "static"
    cam.pace = "medium"
    cam.keyframes = []
    cam.to_dict.return_value = {"keyframes": [], "shot": "WS", "move_type": "static", "camera_role": "env_cam"}

    with patch.object(baker, "_tts", _no_tts), patch.object(
        baker, "_resolve_body_action", return_value=None
    ), patch.object(baker, "plan_camera_for_beat", return_value=cam):
        out = baker.bake_shot(
            shot, coord=coord, world=world, out_dir=tmp_path, t0=0.0, fps=20.0, shot_index=1
        )

    assert out.audio_path, out.bake_errors
    wav = Path(out.audio_path)
    assert wav.is_file()
    import soundfile as sf
    audio, sr = sf.read(str(wav), dtype="float32")
    import numpy as np
    assert float(np.max(np.abs(audio))) < 1e-6
    assert out.speech_duration_s == 0.0
    coord.prepare_sentence.assert_not_called()
    assert out.clip_policy == "hold_end"
    print(f"  OK motion-only silence {wav.name} dur={out.duration_s:.2f}s no Parler")


def main() -> int:
    print("=" * 50)
    print("BAKER MOTION-ONLY")
    print("=" * 50)
    tmp = ROOT / "temp" / "test_baker_motion_only"
    tmp.mkdir(parents=True, exist_ok=True)
    test_motion_only_silence(tmp)
    print("=" * 50)
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
