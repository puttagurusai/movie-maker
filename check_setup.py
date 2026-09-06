"""
check_setup.py

Run this before orchestrator.py to verify everything is in place.
"""

import os
import sys
from pathlib import Path

def check_file(path: str, min_mb: float = 0, desc: str = "") -> bool:
    p = Path(path)
    if not p.exists():
        print(f"❌ MISSING: {path}  ({desc})")
        return False
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb < min_mb:
        print(f"⚠️  TOO SMALL: {path} ({size_mb:.1f} MB, expected > {min_mb} MB)  ({desc})")
        return False
    print(f"✅ {path} ({size_mb:.1f} MB)  {desc}")
    return True

def main():
    print("=== Pipeline Setup Check ===\n")

    ok = True

    # Project models folder (clear names — not HuggingFace user cache)
    print("--- models/ (project folder) ---")
    models = Path("models")
    if models.is_dir():
        print(f"✅ models/ exists")
    else:
        print("❌ models/ missing — create it and download assets")
        ok = False

    # Optional / capture models
    check_file("models/face_landmarker.task", 1, "MediaPipe face landmarker (live capture)")
    check_file("models/kokoro-v1.0.onnx", 200, "Legacy Kokoro (optional, not used by orchestrator)")
    check_file("models/voices-v1.0.bin", 20, "Legacy Kokoro voices (optional)")

    # Parler local folder
    parler_dir = models / "parler-tts-mini-v1"
    parler_weights = parler_dir / "model.safetensors"
    parler_config = parler_dir / "config.json"
    incomplete = list(parler_dir.rglob("*.incomplete")) if parler_dir.is_dir() else []
    if parler_config.is_file() and (parler_weights.is_file() or any(parler_dir.glob("model*.safetensors"))) and not incomplete:
        size_mb = sum(f.stat().st_size for f in parler_dir.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f"✅ models/parler-tts-mini-v1/ complete ({size_mb:.0f} MB)")
    elif incomplete:
        print("❌ models/parler-tts-mini-v1/ has INCOMPLETE download (*.incomplete)")
        print("   Delete models/parler-tts-mini-v1 and re-run: python test_parler.py")
        ok = False
    else:
        print("ℹ️  models/parler-tts-mini-v1/ not downloaded yet")
        print("   Will download into models/ on first: python test_parler.py")

    # Rhubarb
    rhubarb = "rhubarb.exe" if os.name == "nt" else "rhubarb"
    ok &= check_file(rhubarb, 1, "Rhubarb lip-sync binary")

    # Python packages (basic import check)
    print("\n--- Python packages ---")
    packages = {
        "anthropic": "Claude API client (optional)",
        "sounddevice": "Audio playback",
        "soundfile": "WAV I/O",
        "torch": "PyTorch (Parler)",
        "transformers": "HuggingFace transformers",
        "huggingface_hub": "Model download into models/",
    }
    for mod, desc in packages.items():
        try:
            __import__(mod)
            print(f"✅ {mod}  ({desc})")
        except ImportError:
            print(f"❌ {mod} not installed  ({desc})")
            ok = False

    # Parler package (must not be shadowed by a local parler_tts.py)
    print("\n--- Parler-TTS ---")
    try:
        from parler_tts import ParlerTTSForConditionalGeneration  # noqa: F401
        print("✅ parler_tts package (ParlerTTSForConditionalGeneration)")
    except Exception as e:
        print(f"❌ parler_tts package import failed: {e}")
        print("   pip install git+https://github.com/huggingface/parler-tts.git")
        ok = False

    try:
        import parler_voice  # noqa: F401
        print("✅ parler_voice.py helper")
        print(f"   local path: {parler_voice.LOCAL_MODEL_DIR}")
    except Exception as e:
        print(f"❌ parler_voice import failed: {e}")
        ok = False

    # Rhubarb res/ (phonetic mode)
    print("\n--- Rhubarb resources ---")
    res_dir = Path("res") / "sphinx"
    if res_dir.is_dir():
        print(f"✅ res/sphinx/ present  (phonetic models)")
    else:
        print("❌ res/sphinx/ missing  (copy 'res' from the Rhubarb release next to rhubarb.exe)")
        ok = False

    # API key (optional in JSON-paste mode)
    print("\n--- Environment ---")
    if os.getenv("ANTHROPIC_API_KEY"):
        print("✅ ANTHROPIC_API_KEY is set")
    else:
        print("ℹ️  ANTHROPIC_API_KEY is NOT set (OK for JSON-paste mode)")
        print("   For live Claude later:  $env:ANTHROPIC_API_KEY = \"sk-ant-...\"")

    print("\n" + "="*40)
    if ok:
        print("✅ Required checks passed. Ready to run:")
        print("   1. In Blender: open blender_receiver.py → Run Script")
        print("   2. Then: bpy.ops.face.stream_receiver()")
        print("   3. Terminal: python orchestrator.py")
    else:
        print("❌ Some items are missing or incomplete.")
        print("   See README.md (One-time asset setup) and run .\\download_assets.ps1")
        sys.exit(1)

if __name__ == "__main__":
    main()
