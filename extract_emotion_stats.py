"""
extract_emotion_stats.py

Offline extractor — run ONCE where train.parquet lives.

Reads train.parquet, groups by emotion_label, computes mean/std for emotion_26d,
writes emotion_stats.json (used at runtime by emotion_manager.py).

Usage:
  python extract_emotion_stats.py path/to/train.parquet
  python extract_emotion_stats.py path/to/train.parquet models/brain/emotion_stats.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT = Path(__file__).resolve().parent / "models" / "brain" / "emotion_stats.json"


def extract_emotion_stats(parquet_path: str, output_json_path: str = None):
    """
    Reads train.parquet, groups by emotion_label, computes mean and std vector
    for emotion_26d, and saves output to emotion_stats.json.
    """
    if output_json_path is None:
        output_json_path = str(DEFAULT_OUT)

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Could not find parquet file at: {parquet_path}")

    print(f"Loading dataset from '{parquet_path}'...")
    df = pd.read_parquet(parquet_path)

    if "emotion_label" not in df.columns:
        raise KeyError("Column 'emotion_label' not found in parquet")
    if "emotion_26d" not in df.columns:
        raise KeyError("Column 'emotion_26d' not found in parquet")

    emotion_stats = {}

    print("Processing emotion categories...")
    # Group by emotion_label
    for label, group in df.groupby("emotion_label"):
        # Stack list of vectors into a 2D numpy array [N, 26]
        vectors = np.stack(group["emotion_26d"].values)

        # Compute mean and standard deviation along column axis (axis=0)
        mean_vec = np.mean(vectors, axis=0)
        std_vec = np.std(vectors, axis=0)

        # Store as standard floats
        emotion_stats[str(label).lower()] = {
            "mean": mean_vec.tolist(),
            "std": std_vec.tolist(),
            "count": int(len(group)),
        }
        print(f"  Processed '{label}' ({len(group)} samples)  shape={vectors.shape}")

    # Save to JSON
    out = Path(output_json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(emotion_stats, f, indent=4)

    print(f"\nSuccessfully created '{out}'!")
    return str(out)


if __name__ == "__main__":
    # Point to your train.parquet file
    PARQUET_FILE = sys.argv[1] if len(sys.argv) > 1 else "train.parquet"
    OUT_JSON = sys.argv[2] if len(sys.argv) > 2 else str(DEFAULT_OUT)
    extract_emotion_stats(PARQUET_FILE, OUT_JSON)
