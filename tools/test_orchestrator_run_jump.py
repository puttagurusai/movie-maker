"""
Test agents orchestrator body path with run+jump MoMask (updated our template).

1) Verifies Joint2BVHConvertor uses our Mixamo-named template
2) Pre-generates MoMask → SMPL-X Action (same as orchestrator)
3) Prints JSON to paste / path for blender receiver test

Usage:
  python tools/test_orchestrator_run_jump.py

Then:
  1. Open whole_body_retargeted.blend (or production blend)
  2. Run blender_receiver.py → bpy.ops.face.stream_receiver()
  3. python orchestrator_agents.py
     USE_MOMASK=1 MOMASK_ALL=1 MOMASK_SYNC=1
     paste temp/test_run_jump_dialog.json content when prompted
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("USE_MOMASK", "1")
os.environ.setdefault("MOMASK_ALL", "1")
os.environ.setdefault("MOMASK_SYNC", "1")
os.environ.setdefault("MOMASK_GPU_ID", "0")

PROMPT = "a person is running forward quickly and then jumps at the end"
DIALOG = "Look at me — I'm running as fast as I can, and then I jump!"
DURATION_S = 6.0
SEED = 42


def verify_template() -> None:
    """Fail hard if stock MoMask template.bvh is loaded instead of OUR avatar BVH."""
    momask = ROOT / "third_party" / "momask-codes"
    sys.path.insert(0, str(momask))
    os.chdir(momask)
    import visualization.BVH_mod as BVH
    from visualization.joints2bvh import Joint2BVHConvertor
    import numpy as np

    our_path = momask / "visualization" / "data" / "our_momask_template.bvh"
    stock_path = momask / "visualization" / "data" / "template.bvh"
    assert our_path.is_file(), f"missing OUR template {our_path}"

    c = Joint2BVHConvertor()  # must default to OUR file
    our = BVH.load(str(our_path).replace("\\", "/"), need_quater=True)
    stock = BVH.load(str(stock_path).replace("\\", "/"), need_quater=True)

    names = list(c.template.names)
    off0 = np.asarray(c.template.offsets[0], dtype=np.float64)
    d_our = float(np.linalg.norm(c.template.offsets - our.offsets))
    d_stock = float(np.linalg.norm(c.template.offsets - stock.offsets))

    print("[check] convertor path default → our_momask_template.bvh")
    print("[check] root name=", names[0], "offset0=", off0)
    print("[check] ||active-OUR||=", d_our, " ||active-STOCK||=", d_stock)
    print("[check] STOCK root offset (must NOT match):", stock.offsets[0])

    assert d_our < 1e-6, "Convertor is NOT using our updated BVH offsets"
    assert d_stock > 0.1, "Convertor still matches stock template — wrong file"
    assert names[0] == "Hips", "Need Mixamo names for agent KeeMap (still OUR lengths)"
    assert abs(float(off0[1]) - 0.992111) < 0.02, "hip height not our avatar"
    print("[check] PASS — ONLY our updated BVH (not stock template.bvh)")
    os.chdir(ROOT)


def main():
    print("=" * 60)
    print("TEST: orchestrator MoMask path — run + jump + dialog")
    print("=" * 60)
    verify_template()

    # Import module file directly (avoid face_agents.__init__ TTS/sounddevice deps)
    import importlib.util

    mp = ROOT / "face_agents" / "momask_body_pipeline.py"
    spec = importlib.util.spec_from_file_location("momask_body_pipeline", mp)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["momask_body_pipeline"] = mod
    spec.loader.exec_module(mod)

    print(f"[env] USE_MOMASK={os.environ.get('USE_MOMASK')} enabled={mod.momask_enabled()}")
    print(f"[gen] prompt={PROMPT!r} duration_s={DURATION_S}")

    result = mod.generate_body_action(
        PROMPT,
        duration_s=DURATION_S,
        seed=SEED,
        use_cache=True,
    )
    print(json.dumps(result.to_dict(), indent=2))

    report = {
        "ok": result.ok,
        "action_name": result.action_name,
        "prompt": result.prompt,
        "bvh_path": result.bvh_path,
        "blend_path": result.blend_path,
        "cached": result.cached,
        "error": result.error,
        "dialog": DIALOG,
        "template": "our_momask_template.bvh",
        "json_input": str(ROOT / "temp" / "test_run_jump_dialog.json"),
        "blender_steps": [
            "Open whole_body_retargeted.blend (or production blend with SMPL-X_Armature)",
            "Scripting: open blender_receiver.py → Run Script",
            "Console: bpy.ops.face.stream_receiver()",
            "Terminal: set USE_MOMASK=1 MOMASK_ALL=1 MOMASK_SYNC=1",
            "python orchestrator_agents.py",
            "Paste contents of temp/test_run_jump_dialog.json when prompted",
        ],
    }
    out = ROOT / "temp" / "test_run_jump_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[report] {out}")
    if result.ok:
        print("\nSUCCESS — body Action ready for receiver")
        print(f"  action: {result.action_name}")
        print(f"  blend:  {result.blend_path}")
        print(f"  bvh:    {result.bvh_path}")
        print("\n>>> Now open Blender + blender_receiver, then run orchestrator_agents.py")
        print(">>> Paste: temp/test_run_jump_dialog.json")
    else:
        print("\nFAILED:", result.error)
        sys.exit(1)


if __name__ == "__main__":
    main()
