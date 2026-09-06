"""
test_expressions.py — headless Blender validation of emotion expressions.
Run: blender --background face.blend --python test_expressions.py
Outputs test_expressions_report.txt with pass/fail for each expression.
"""

import bpy
import os
import sys

# Add project root to path so we can import emotion_map
sys.path.insert(0, os.path.dirname(bpy.data.filepath))
import emotion_map

OUT_PATH = os.path.join(os.path.dirname(bpy.data.filepath), "test_expressions_report.txt")

EMOTIONS_TO_TEST = [
    ("happy",     0.9),
    ("sad",       0.85),
    ("angry",     0.9),
    ("surprised", 0.85),
    ("fearful",   0.8),
    ("disgusted", 0.85),
    ("sarcastic", 0.8),
    ("thinking",  0.75),
    ("neutral",   1.0),
]

# Minimum expected values for each emotion (sanity checks)
EXPRESSION_EXPECTATIONS = {
    "happy": {
        "mouthSmileLeft": 0.5, "mouthSmileRight": 0.5,
        "cheekSquintLeft": 0.4, "cheekSquintRight": 0.4,
        "eyeSquintLeft": 0.3, "eyeSquintRight": 0.3,
    },
    "sad": {
        "browInnerUp": 0.5,
        "mouthFrownLeft": 0.5, "mouthFrownRight": 0.5,
    },
    "angry": {
        "browDownLeft": 0.5, "browDownRight": 0.5,
        "eyeSquintLeft": 0.3, "eyeSquintRight": 0.3,
    },
    "surprised": {
        "browInnerUp": 0.6,
        "eyeWideLeft": 0.5, "eyeWideRight": 0.5,
    },
    "fearful": {
        "eyeWideLeft": 0.5, "eyeWideRight": 0.5,
        "browInnerUp": 0.4,
    },
    "disgusted": {
        "noseSneerLeft": 0.4, "noseSneerRight": 0.3,
    },
}


def get_face_mesh():
    best, best_n = None, 0
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.data and obj.data.shape_keys:
            n = len(obj.data.shape_keys.key_blocks)
            if n > best_n:
                best_n, best = n, obj
    return best


def apply_blendshapes(obj, values_dict):
    kb = obj.data.shape_keys.key_blocks
    for k, v in values_dict.items():
        if k in kb:
            kb[k].value = float(v)


def reset_face(obj):
    kb = obj.data.shape_keys.key_blocks
    for block in kb:
        if block.name != "Basis":
            block.value = 0.0


def test_emotion(obj, emotion, intensity):
    reset_face(obj)
    blendshapes = emotion_map.get_blendshapes(emotion, intensity)
    apply_blendshapes(obj, blendshapes)
    kb = obj.data.shape_keys.key_blocks

    results = {}
    expectations = EXPRESSION_EXPECTATIONS.get(emotion, {})
    passed = 0
    failed = 0
    failures = []

    for key, min_val in expectations.items():
        actual = kb[key].value if key in kb else -1.0
        ok = actual >= min_val
        results[key] = {"expected_min": min_val, "actual": round(actual, 4), "pass": ok}
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append(f"  FAIL {key}: got {actual:.3f} < {min_val:.3f}")

    # Also log all non-zero values for inspection
    active_keys = {
        kb_name: round(kb[kb_name].value, 4)
        for kb_name in [b.name for b in kb]
        if kb_name != "Basis" and kb[kb_name].value > 0.005
    }

    return {
        "emotion": emotion,
        "intensity": intensity,
        "passed": passed,
        "failed": failed,
        "failures": failures,
        "active_keys": active_keys,
        "all_ok": failed == 0,
    }


def test_lip_keys(obj):
    """Check that major lip-sync keys exist and have correct range."""
    kb = obj.data.shape_keys.key_blocks
    lip_keys = [
        "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
        "mouthLowerDownLeft", "mouthLowerDownRight",
        "mouthUpperUpLeft", "mouthUpperUpRight",
        "mouthStretchLeft", "mouthStretchRight",
        "mouthRollLower", "mouthRollUpper",
    ]
    results = {}
    all_ok = True
    for k in lip_keys:
        present = k in kb
        if present:
            results[k] = f"present range=({kb[k].slider_min:.1f},{kb[k].slider_max:.1f})"
        else:
            results[k] = "MISSING"
            all_ok = False
    return results, all_ok


def test_secondary_meshes():
    """Verify secondary meshes exist and have expected keys."""
    secondary = {
        "eyeLeft_ORIGINAL": ["eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft"],
        "eyeRight_ORIGINAL": ["eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight"],
        "teeth_ORIGINAL": ["jawOpen", "mouthClose"],
    }
    results = {}
    for mesh_name, expected_keys in secondary.items():
        obj = bpy.data.objects.get(mesh_name)
        if obj is None:
            results[mesh_name] = "MISSING OBJECT"
            continue
        if obj.data is None or obj.data.shape_keys is None:
            results[mesh_name] = "NO SHAPE KEYS"
            continue
        kb = obj.data.shape_keys.key_blocks
        found = [k for k in expected_keys if k in kb]
        missing = [k for k in expected_keys if k not in kb]
        results[mesh_name] = {
            "found": found,
            "missing": missing,
            "status": "OK" if not missing else f"MISSING: {missing}",
        }
    return results


lines = []
lines.append("=" * 70)
lines.append("FACE EXPRESSION VALIDATION REPORT")
lines.append("=" * 70)

face_obj = get_face_mesh()
if face_obj is None:
    lines.append("ERROR: No face mesh found!")
else:
    lines.append(f"Face mesh: {face_obj.name!r} ({len(face_obj.data.shape_keys.key_blocks)} shape keys)\n")

    # Test emotions
    lines.append("EMOTION EXPRESSION TESTS:")
    lines.append("-" * 50)
    all_passed = True
    for emotion, intensity in EMOTIONS_TO_TEST:
        result = test_emotion(face_obj, emotion, intensity)
        status = "PASS" if result["all_ok"] else "FAIL"
        if not result["all_ok"]:
            all_passed = False
        lines.append(f"\n[{status}] {emotion.upper()} @ {intensity:.2f}")
        lines.append(f"  Active keys: {result['active_keys']}")
        if result["failures"]:
            for f in result["failures"]:
                lines.append(f)

    reset_face(face_obj)
    lines.append(f"\n\nOVERALL EMOTION TESTS: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    # Test lip keys
    lines.append("\nLIP SYNC KEY PRESENCE TEST:")
    lines.append("-" * 50)
    lip_results, lips_ok = test_lip_keys(face_obj)
    for k, v in lip_results.items():
        lines.append(f"  {k}: {v}")
    lines.append(f"Lips: {'OK' if lips_ok else 'MISSING KEYS'}")

    # Test secondary meshes
    lines.append("\nSECONDARY MESH TEST (eye balls + teeth):")
    lines.append("-" * 50)
    sec_results = test_secondary_meshes()
    for mesh_name, info in sec_results.items():
        lines.append(f"  {mesh_name}: {info}")

lines.append("\n" + "=" * 70)
report = "\n".join(lines)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(report)
print("[test_expressions] Done.")
print(report)
