"""
Smoke test for the agents face pipeline (no Blender required).

  python tools/smoke_test_pipeline.py

Mocks Parler TTS + sounddevice. Uses real wav2arkit + vocal clips on disk.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os_chdir = ROOT
import os

os.chdir(ROOT)


def main() -> int:
    fails = 0

    def ok(name: str) -> None:
        print(f"  PASS  {name}")

    def bad(name: str, err: str) -> None:
        nonlocal fails
        fails += 1
        print(f"  FAIL  {name}: {err}")

    print("=" * 60)
    print("SMOKE TEST — face agents pipeline")
    print("=" * 60)

    # --- tokens ---
    print("\n[1] expression_tokens")
    try:
        from face_agents.expression_tokens import (
            split_token_segments,
            strip_tokens,
            prepare_token_clip,
            resolve_sound_path,
        )

        segs = split_token_segments("Hi [laugh] there [eww] bye")
        kinds = [s.kind for s in segs]
        assert kinds == ["speech", "token", "speech", "token", "speech"], kinds
        assert "[" not in strip_tokens("Hi [laugh] there")
        for tok in ("laugh", "eww", "gasp", "chuckle"):
            r = prepare_token_clip(tok)
            assert r is not None, f"no clip {tok}"
            a, sr, p = r
            assert 0.15 < len(a) / sr < 4.0
        ok("segment + clips")
    except Exception as e:
        bad("segment + clips", str(e))

    # --- mock devices ---
    class FakeSD:
        def play(self, *a, **k):
            pass

        def wait(self):
            pass

    sys.modules["sounddevice"] = FakeSD()  # type: ignore

    import parler_voice

    def fake_gen(text, voice_style=None, output_path=None, play_audio=False, **kw):
        # Guard: must never see bracket tokens
        assert "[" not in (text or ""), f"Parler got token text: {text!r}"
        sr = 22050
        n = int(0.35 * sr)
        audio = (np.random.randn(n).astype(np.float32) * 0.04)
        if output_path:
            sf.write(output_path, audio, sr)
        return audio, sr

    parler_voice.generate_speech = fake_gen  # type: ignore

    # --- tts guard ---
    print("\n[2] TTS refuses pure token words")
    try:
        import orchestrator_agents as oa

        oa.generate_speech = fake_gen  # type: ignore
        a, sr = oa.tts_to_wav("laugh", "happy", 0.7, "temp/_smoke_laugh.wav")
        assert len(a) / sr < 0.15
        ok("tts_to_wav skip token word")
    except Exception as e:
        bad("tts_to_wav skip token word", str(e))

    # --- full process_sentence ---
    print("\n[3] process_sentence Hello [laugh] world")
    try:
        from face_agents.coordinator import FaceCoordinator
        import orchestrator_agents as oa

        oa.generate_speech = fake_gen  # type: ignore
        coord = FaceCoordinator(
            udp_ip="127.0.0.1", udp_port=19099, fps=30, use_brain=False, device="cpu"
        )
        log = []
        coord.send_udp = lambda p: log.append(p.get("type"))  # type: ignore
        parler_calls = []

        def tracking_gen(text, voice_style=None, output_path=None, play_audio=False, **kw):
            parler_calls.append(text)
            return fake_gen(text, voice_style, output_path, play_audio, **kw)

        oa.generate_speech = tracking_gen  # type: ignore
        parler_voice.generate_speech = tracking_gen  # type: ignore

        oa.process_sentence(
            coord,
            {"text": "Hello [laugh] world", "emotion": "happy", "intensity": 0.7},
            42,
        )
        # Parler only speech parts
        joined = " | ".join(parler_calls)
        assert "laugh" not in joined.lower() or all(
            "laugh" not in t.lower() or len(t.split()) > 2 for t in parler_calls
        )
        for t in parler_calls:
            assert "[" not in t
            assert t.strip().lower() not in ("laugh", "eww", "gasp", "ha ha ha")
        assert "viseme" in log and "emotion" in log
        ok(f"process_sentence (parler_calls={parler_calls})")
    except Exception as e:
        bad("process_sentence", str(e))

    # --- play_token_reaction lips ---
    print("\n[4] play_token_reaction has lips+emotion")
    try:
        from face_agents.coordinator import FaceCoordinator

        coord = FaceCoordinator(
            udp_ip="127.0.0.1", udp_port=19099, fps=30, use_brain=False, device="cpu"
        )
        packets = []
        coord.send_udp = lambda p: packets.append(p)  # type: ignore
        coord.play_token_reaction("laugh", "happy", 0.8)
        types = [p.get("type") for p in packets]
        assert types.count("viseme") > 5
        assert types.count("emotion") > 5
        # smile in some emotion packet
        smiles = [
            p for p in packets
            if p.get("type") == "emotion"
            and (p.get("blendshapes") or {}).get("mouthSmileLeft", 0) > 0.2
        ]
        assert smiles, "laugh should raise smile"
        ok("token lips + smile")
    except Exception as e:
        bad("token lips + smile", str(e))

    # --- web assets ---
    print("\n[5] web_viewer assets")
    try:
        w = ROOT / "web_viewer"
        assert (w / "face_viewer.html").is_file()
        assert (w / "face.glb").is_file()
        assert (w / "ws_bridge.py").is_file()
        assert (w / "textures" / "scanstore_albedo_4k.png").is_file()
        ok("web files")
    except Exception as e:
        bad("web files", str(e))

    # --- vocal source ---
    print("\n[6] vocal library source")
    try:
        src = ROOT / "temp" / "vocal_sounds" / "SOURCE.txt"
        assert src.is_file(), "SOURCE.txt missing — run tools/build_real_vocals.py"
        text = src.read_text(encoding="utf-8")
        assert "parler" not in text.lower() or "NOT Parler" in text
        ok(f"SOURCE: {text.splitlines()[0]}")
    except Exception as e:
        bad("vocal SOURCE", str(e))

    print("\n" + "=" * 60)
    if fails:
        print(f"RESULT: {fails} FAILURE(S)")
        return 1
    print("RESULT: ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
