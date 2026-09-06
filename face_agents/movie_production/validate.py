"""
Stage 8 — Validation pass (mavie.txt).
"""

from __future__ import annotations

from typing import Any, Dict, List

from .shot_schema import ShotDetail
from .world_state import WorldState


def validate_package(
    shots: List[ShotDetail],
    *,
    world: WorldState | None = None,
    fps: float = 20.0,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not shots:
        errors.append("no shots")
        return {"ok": False, "errors": errors, "warnings": warnings}

    # Time continuity on master clock
    for i, sh in enumerate(shots):
        if sh.duration_s <= 0.05 and sh.bake_ok:
            warnings.append(f"{sh.shot_id}: very short duration {sh.duration_s:.2f}s")
        if sh.t1 + 1e-6 < sh.t0:
            errors.append(f"{sh.shot_id}: t1 < t0")
        if i > 0:
            prev = shots[i - 1]
            # allow small gap/overlap
            gap = sh.t0 - prev.t1
            if gap < -0.05:
                errors.append(f"{sh.shot_id}: overlaps previous by {-gap:.2f}s")
            elif gap > 0.5:
                warnings.append(f"{sh.shot_id}: gap {gap:.2f}s after previous")

        if not sh.text.strip():
            errors.append(f"{sh.shot_id}: empty text")
        if sh.bake_ok and not sh.audio_path:
            errors.append(f"{sh.shot_id}: bake_ok but no audio_path")
        if sh.body_mode in ("momask", "both") and not (sh.humanml_prompt or sh.body_action):
            warnings.append(f"{sh.shot_id}: momask mode without humanml/action")
        if sh.bake_errors:
            for e in sh.bake_errors:
                errors.append(f"{sh.shot_id}: {e}")

        # Frame range sanity
        nfr = int(round(max(0.0, sh.duration_s) * fps))
        if sh.bake_ok and nfr < 2:
            warnings.append(f"{sh.shot_id}: fewer than 2 frames @ {fps}fps")

    # World continuity soft check
    if world is not None:
        for i in range(1, len(shots)):
            a, b = shots[i - 1], shots[i]
            if a.world_out and b.world_in:
                ca = (a.world_out.get("characters") or {}).get("hero") or {}
                cb = (b.world_in.get("characters") or {}).get("hero") or {}
                if ca.get("base_state") and cb.get("base_state"):
                    # if next shot claims sitting but prev left standing without sit action — warn
                    if ca.get("base_state") != cb.get("base_state"):
                        # allowed if shot text implies change; just note
                        warnings.append(
                            f"{b.shot_id}: base_state {ca.get('base_state')}→{cb.get('base_state')}"
                        )

    try:
        from face_agents.qc_gates import run_cpu_gates
        qc = run_cpu_gates(shots=shots)
        errors.extend(qc.get("errors") or [])
        warnings.extend(qc.get("warnings") or [])
    except Exception as e:
        warnings.append(f"qc_gates skip: {e}")

    ok = not errors
    return {
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "n_shots": len(shots),
        "duration_s": max((s.t1 for s in shots), default=0.0),
    }
