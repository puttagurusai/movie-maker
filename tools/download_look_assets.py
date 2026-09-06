#!/usr/bin/env python3
"""
Download Poly Haven (CC0) HDRIs, PBR textures, and glTF props for realistic Look_Set kits.

Run:  python tools/download_look_assets.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HDRI = ROOT / "assets" / "looks" / "hdri"
TEX = ROOT / "assets" / "looks" / "textures"
MODELS = ROOT / "assets" / "looks" / "models"
UA = {"User-Agent": "projface-look-downloader/1.0", "Referer": "https://polyhaven.com/"}

HDRIS = {
    "studio_soft.exr": "studio_small_09",
    "golden_hour.exr": "venice_sunset",
    "night_urban.exr": "dikhololo_night",
    "park_day.exr": "spruit_sunrise",
    "forest_day.exr": "forest_slope",
    "playground_day.exr": "kloppenheim_06",
    "street_day.exr": "courtyard",
}

# PBR texture packs (diff + nor_gl + rough @ 1k jpg)
TEXTURES = {
    "grass": "leafy_grass",
    "asphalt": "asphalt_02",
    "brick": "red_brick_03",
    "plaster": "plastered_wall_04",
    "wood": "wood_planks",
    "concrete": "concrete_floor",
    "sand": "coast_sand_rocks_02",
}

# Realistic props (glTF 1k)
MODELS_IDS = {
    "pine_tree": "pine_tree_01",
    "island_tree": "island_tree_02",
    "jacaranda": "jacaranda_tree",
    "urban_facade": "modular_urban_apartments_facade",
    "street_lamp": "street_lamp_01",
    "bench": "painted_wooden_bench",
    "grass_clump": "grass_medium_01",
    "picnic_table": "wooden_picnic_table",
    "chair": "WoodenChair_01",
    "table": "WoodenTable_01",
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def _api(path: str):
    return json.loads(_get(f"https://api.polyhaven.com{path}").decode())


def _save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 5000:
        print(f"  EXISTS {path.relative_to(ROOT)} ({path.stat().st_size})")
        return
    path.write_bytes(data)
    print(f"  OK {path.relative_to(ROOT)} ({len(data)})")


def download_hdris() -> None:
    print("=== HDRIs ===")
    HDRI.mkdir(parents=True, exist_ok=True)
    for fname, asset_id in HDRIS.items():
        out = HDRI / fname
        if out.is_file() and out.stat().st_size > 10_000:
            print(f"  EXISTS {fname}")
            continue
        files = _api(f"/files/{asset_id}")
        url = files["hdri"]["1k"]["exr"]["url"]
        print(f"  GET {fname}")
        _save(out, _get(url))


def download_textures() -> None:
    print("=== Textures (1k) ===")
    for folder, asset_id in TEXTURES.items():
        dest = TEX / folder
        dest.mkdir(parents=True, exist_ok=True)
        files = _api(f"/files/{asset_id}")
        for map_name, suffix in (("Diffuse", "diff"), ("nor_gl", "nor"), ("Rough", "rough")):
            if map_name not in files:
                continue
            meta = files[map_name].get("1k") or {}
            jpg = meta.get("jpg") or meta.get("png")
            if not jpg or "url" not in jpg:
                continue
            out = dest / f"{suffix}.jpg"
            print(f"  GET {folder}/{suffix}")
            _save(out, _get(jpg["url"]))


def download_models() -> None:
    print("=== Models (glTF 1k) ===")
    for folder, asset_id in MODELS_IDS.items():
        dest = MODELS / folder
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / f"{asset_id}.gltf"
        if marker.is_file() and marker.stat().st_size > 100:
            print(f"  EXISTS {folder}")
            continue
        files = _api(f"/files/{asset_id}")
        g = (files.get("gltf") or {}).get("1k") or (files.get("gltf") or {}).get("2k")
        if not g or "gltf" not in g:
            print(f"  SKIP {folder} (no gltf)")
            continue
        entry = g["gltf"]
        print(f"  GET {folder}/{asset_id}.gltf")
        _save(dest / f"{asset_id}.gltf", _get(entry["url"]))
        # companion .bin often next to includes
        for rel, meta in (entry.get("include") or {}).items():
            url = meta.get("url")
            if not url:
                continue
            out = dest / rel.replace("\\", "/")
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"  GET {folder}/{rel}")
            _save(out, _get(url))


def main() -> None:
    download_hdris()
    download_textures()
    download_models()
    print("DONE — realistic look assets ready under assets/looks/")


if __name__ == "__main__":
    main()
