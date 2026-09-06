"""
export_brain_modules.py

Split monolithic brain_latest.pt / brain.pt into three inference packages:

  models/brain/shared_encoder.pt
  models/brain/face_head.pt
  models/brain/character_adapter.pt   (also saved as adapter.pt alias)

Usage:
  python export_brain_modules.py
  python export_brain_modules.py "brain output/results (1)/checkpoints/brain_latest.pt"
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from brain_model import (
    BrainModel,
    CharacterAdapter,
    FaceHeadBundle,
    SharedEncoderBundle,
    load_full_brain_state_dict,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_SRC = ROOT / "models" / "brain" / "brain.pt"
OUT_DIR = ROOT / "models" / "brain"


def export(src: Path, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading full Brain from {src} ...")
    full = load_full_brain_state_dict(src, device="cpu")

    # --- shared_encoder package ---
    shared = SharedEncoderBundle()
    shared.hubert.load_state_dict(full.hubert.state_dict())
    shared.audio_project.load_state_dict(full.audio_project.state_dict())
    shared.emotion_conditioning.load_state_dict(full.emotion_conditioning.state_dict())
    shared.fusion_input_project.load_state_dict(full.fusion_input_project.state_dict())
    shared.shared_encoder.load_state_dict(full.shared_encoder.state_dict())
    p1 = out_dir / "shared_encoder.pt"
    torch.save(shared.state_dict(), p1)
    print(f"  saved {p1}  ({p1.stat().st_size / 1e6:.1f} MB)")

    # --- face_head package ---
    face = FaceHeadBundle()
    face.face_head.load_state_dict(full.face_head.state_dict())
    face.lower_face_head.load_state_dict(full.lower_face_head.state_dict())
    face.upper_face_linear.load_state_dict(full.upper_face_linear.state_dict())
    face.upper_face_conv.load_state_dict(full.upper_face_conv.state_dict())
    p2 = out_dir / "face_head.pt"
    torch.save(face.state_dict(), p2)
    print(f"  saved {p2}  ({p2.stat().st_size / 1e6:.1f} MB)")

    # --- character adapter ---
    adapter = CharacterAdapter()
    adapter.load_state_dict(full.character_adapter.state_dict())
    p3 = out_dir / "character_adapter.pt"
    p3b = out_dir / "adapter.pt"
    torch.save(adapter.state_dict(), p3)
    torch.save(adapter.state_dict(), p3b)
    print(f"  saved {p3}")
    print(f"  saved {p3b} (alias)")

    print("Done. Inference can load the three modules via brain_inference.load_brain_model().")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    if not src.is_file():
        alt = ROOT / "brain output" / "results (1)" / "checkpoints" / "brain_latest.pt"
        if alt.is_file():
            src = alt
        else:
            raise SystemExit(f"Checkpoint not found: {src}")
    export(src)
