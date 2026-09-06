#!/usr/bin/env python3
"""
Download wardrobe fabrics + NPC character packs for cast/clothing.

Sources (all free for commercial use as documented):
  - Poly Haven CC0 fabric PBR textures (denim / jersey / wool) → hero outfit materials
  - Quaternius Ultimate Animated Character Pack (CC0) via OpenGameArt → NPC meshes
  - Kenney Blocky Characters (CC0) via OpenGameArt → simple NPC fallback
  - Kenney Starter-Kit character.glb (CC0) via GitHub → single NPC glb

Run:
  python tools/download_cast_assets.py

Notes:
  Poly Haven does NOT ship SMPL-X fitted clothing. Real hero outfits that deform
  with the body still need artist clothing or a commercial SMPL cloth pack.
  This script gives: fabric maps for proxies + ready NPC people for crowds.
"""
from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAST = ROOT / "assets" / "cast"
FABRIC = CAST / "fabrics"
NPC = CAST / "npcs"
UA = {
    "User-Agent": "projface-cast-downloader/1.0",
    "Referer": "https://polyhaven.com/",
}

# Poly Haven fabric textures → wardrobe materials
FABRICS = {
    "denim": "denim_fabric",                 # casual jeans
    "jersey": "cotton_jersey",               # casual tee
    "suit_wool": "poly_wool_herringbone",    # formal
    "leather": "brown_leather",              # optional accents
}

# Direct zip / glb URLs (CC0)
NPC_PACKS = {
    "quaternius_ultimate": {
        "url": "https://opengameart.org/sites/default/files/ultimate_animated_character_pack_by_quaternius.zip",
        "kind": "zip",
    },
    "kenney_blocky": {
        "url": "https://opengameart.org/sites/default/files/kenney_blocky-characters_2.0.zip",
        "kind": "zip",
    },
    "kenney_platformer_character": {
        "url": "https://raw.githubusercontent.com/KenneyNL/Starter-Kit-3D-Platformer/main/models/character.glb",
        "kind": "glb",
        "out": "kenney_character.glb",
    },
    "itch_plewr_character": {
        "url": "https://plewr.itch.io/3d-rigged-character/file/123456",  # placeholder — resolved below if mirror fails
        "kind": "skip",
    },
}


def _ctx():
    # Some hosts (OGA CDN) fail strict CRL checks on Windows corporate SSL.
    try:
        return ssl.create_default_context()
    except Exception:
        return ssl._create_unverified_context()


def _get(url: str, *, referer: str = "") -> bytes:
    headers = dict(UA)
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300, context=_ctx()) as r:
            return r.read()
    except Exception:
        # Retry with ssl-no-verify style context
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=300, context=ctx) as r:
            return r.read()


def _api(path: str):
    return json.loads(_get(f"https://api.polyhaven.com{path}").decode())


def _save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 2000 and path.stat().st_size == len(data):
        print(f"  EXISTS {path.relative_to(ROOT)}")
        return
    if path.is_file() and path.stat().st_size > 50_000 and path.suffix in (".zip", ".glb", ".gltf"):
        print(f"  EXISTS {path.relative_to(ROOT)} ({path.stat().st_size})")
        return
    path.write_bytes(data)
    print(f"  OK {path.relative_to(ROOT)} ({len(data)})")


def download_fabrics() -> None:
    print("=== Poly Haven fabrics (wardrobe materials) ===")
    FABRIC.mkdir(parents=True, exist_ok=True)
    for folder, asset_id in FABRICS.items():
        dest = FABRIC / folder
        dest.mkdir(parents=True, exist_ok=True)
        try:
            files = _api(f"/files/{asset_id}")
        except Exception as e:
            print(f"  FAIL {folder}: {e}")
            continue
        for map_name, suffix in (("Diffuse", "diff"), ("nor_gl", "nor"), ("Rough", "rough")):
            if map_name not in files:
                # some packs use lowercase keys historically
                continue
            meta = files[map_name].get("1k") or files[map_name].get("2k") or {}
            jpg = meta.get("jpg") or meta.get("png")
            if not jpg or "url" not in jpg:
                continue
            out = dest / f"{suffix}.jpg"
            if out.is_file() and out.stat().st_size > 5000:
                print(f"  EXISTS {folder}/{suffix}")
                continue
            print(f"  GET {folder}/{suffix}")
            try:
                _save(out, _get(jpg["url"], referer="https://polyhaven.com/"))
            except Exception as e:
                print(f"  FAIL {folder}/{suffix}: {e}")


def _extract_zip(zpath: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(dest)
    print(f"  EXTRACTED → {dest.relative_to(ROOT)}")


def download_npc_packs() -> None:
    print("=== NPC character packs (CC0) ===")
    NPC.mkdir(parents=True, exist_ok=True)
    # Quaternius
    qzip = NPC / "quaternius_ultimate.zip"
    qdir = NPC / "quaternius_ultimate"
    if not (qdir.is_dir() and any(qdir.rglob("*.fbx"))):
        try:
            if not (qzip.is_file() and qzip.stat().st_size > 1_000_000):
                print("  GET quaternius_ultimate.zip (OpenGameArt)")
                _save(
                    qzip,
                    _get(
                        NPC_PACKS["quaternius_ultimate"]["url"],
                        referer="https://opengameart.org/",
                    ),
                )
            _extract_zip(qzip, qdir)
        except Exception as e:
            print(f"  FAIL quaternius: {e}")
    else:
        print("  EXISTS quaternius_ultimate/")

    # Kenney blocky
    kzip = NPC / "kenney_blocky.zip"
    kdir = NPC / "kenney_blocky"
    if not (kdir.is_dir() and any(kdir.rglob("*"))):
        try:
            if not (kzip.is_file() and kzip.stat().st_size > 100_000):
                print("  GET kenney_blocky.zip (OpenGameArt)")
                _save(
                    kzip,
                    _get(
                        NPC_PACKS["kenney_blocky"]["url"],
                        referer="https://opengameart.org/",
                    ),
                )
            _extract_zip(kzip, kdir)
        except Exception as e:
            print(f"  FAIL kenney_blocky: {e}")
    else:
        print("  EXISTS kenney_blocky/")

    # Single Kenney glb
    glb = NPC / "kenney_character.glb"
    if not (glb.is_file() and glb.stat().st_size > 10_000):
        try:
            print("  GET kenney_character.glb (GitHub)")
            _save(glb, _get(NPC_PACKS["kenney_platformer_character"]["url"]))
        except Exception as e:
            print(f"  FAIL kenney_character.glb: {e}")
    else:
        print("  EXISTS kenney_character.glb")


def organize_quaternius_clothes() -> None:
    """
    Map already-downloaded Quaternius FBX (clothes baked on body) into
    assets/cast/clothes/<outfit_id>/ — generic humanoid scale, not SMPL-specific.
    """
    print("=== Organize Quaternius outfits (similar-avatar clothes) ===")
    src_root = NPC / "quaternius_ultimate"
    if not src_root.is_dir():
        print("  SKIP (run NPC download first)")
        return
    mapping = {
        "casual_01": ["Casual_Male.fbx", "Casual2_Male.fbx", "Casual_Female.fbx", "Casual2_Female.fbx"],
        "formal_01": ["Suit_Male.fbx", "Suit_Female.fbx"],
        "worker_01": ["Worker_Male.fbx", "Worker_Female.fbx"],
        "hoodie_01": [],  # filled from poly.pizza glb below if present
    }
    fbx_by_name = {p.name: p for p in src_root.rglob("*.fbx")}
    clothes = CAST / "clothes"
    for oid, names in mapping.items():
        dest = clothes / oid
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = fbx_by_name.get(name)
            if not src:
                continue
            out = dest / name
            if out.is_file() and out.stat().st_size > 1000:
                print(f"  EXISTS clothes/{oid}/{name}")
                continue
            out.write_bytes(src.read_bytes())
            print(f"  OK clothes/{oid}/{name}")


def download_poly_pizza_outfit_glbs() -> None:
    """
    Quaternius modular men pieces on poly.pizza — thumbnails share UUID with .glb.
    These are generic humanoid outfits (casual / business / hoodie / worker).
    """
    print("=== Poly.pizza outfit GLBs (Quaternius modular men) ===")
    # UUID stems from poly.pizza bundle page thumbnails
    outfits = {
        "formal_01": [
            ("business_man.glb", "e599abbe-7d73-488c-9d7e-3ead281e705c"),
        ],
        "casual_01": [
            ("casual_character.glb", "90a9e2d4-053f-42f1-99a2-8f5e1180ea7f"),
            ("beach_character.glb", "f771a536-1c18-4a47-bb56-ceea4b603455"),
        ],
        "hoodie_01": [
            ("hoodie_character.glb", "bcd66ec5-5e81-4901-a222-47abc875fe2a"),
        ],
        "worker_01": [
            ("worker_character.glb", "3a5f3056-ffe6-42eb-bd52-122afcbd22b2"),
        ],
        "npc_variant_a": [
            ("punk_character.glb", "e56f23b5-3270-406f-8924-f77cad980c43"),
            ("adventurer.glb", "bbe369ee-a686-42c7-adad-14356f5f2f15"),
        ],
        "npc_variant_b": [
            ("farmer.glb", "81f2f0cf-6f53-4b57-92ea-dba0928620f2"),
            ("swat.glb", "713f6535-f4f3-4367-a4c6-ced126ae0936"),
        ],
    }
    clothes = CAST / "clothes"
    for oid, items in outfits.items():
        dest = clothes / oid
        dest.mkdir(parents=True, exist_ok=True)
        for fname, uid in items:
            out = dest / fname
            if out.is_file() and out.stat().st_size > 10_000:
                print(f"  EXISTS clothes/{oid}/{fname}")
                continue
            url = f"https://static.poly.pizza/{uid}.glb"
            print(f"  GET {oid}/{fname}")
            try:
                data = _get(url, referer="https://poly.pizza/")
                if len(data) < 500 or data[:4] == b"<!DO" or data[:1] == b"{":
                    print(f"  FAIL {fname}: not a glb ({len(data)} bytes)")
                    continue
                _save(out, data)
            except Exception as e:
                print(f"  FAIL {fname}: {e}")


def write_manifest() -> None:
    fabrics = {
        name: {
            "diff": (FABRIC / name / "diff.jpg").is_file(),
            "nor": (FABRIC / name / "nor.jpg").is_file(),
            "rough": (FABRIC / name / "rough.jpg").is_file(),
        }
        for name in FABRICS
    }
    npcs = {
        "quaternius_ultimate": (NPC / "quaternius_ultimate").is_dir(),
        "kenney_blocky": (NPC / "kenney_blocky").is_dir(),
        "kenney_character.glb": (NPC / "kenney_character.glb").is_file(),
    }
    man = {
        "fabrics": fabrics,
        "npcs": npcs,
        "wardrobe_map": {
            "casual_01": {"shirt": "jersey", "pants": "denim"},
            "formal_01": {"shirt": "suit_wool", "pants": "suit_wool"},
            "hero_default": {"shirt": "jersey", "pants": "denim"},
        },
        "notes": (
            "Hero SMPL-X fitted clothing is NOT in these packs. "
            "Use fabrics on proxy/wardrobe meshes; use NPC packs for Cast_Extras."
        ),
    }
    out = CAST / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"  wrote {out.relative_to(ROOT)}")


def main() -> None:
    CAST.mkdir(parents=True, exist_ok=True)
    download_fabrics()
    download_npc_packs()
    organize_quaternius_clothes()
    download_poly_pizza_outfit_glbs()
    write_manifest()
    print("DONE — assets under assets/cast/")
    print("  fabrics/  → PBR maps for wardrobe proxies")
    print("  npcs/     → Quaternius + Kenney people for extras")
    print("  clothes/  → generic humanoid outfits (casual/formal/worker/hoodie)")
    print("Note: these fit similar-scale avatars — SMPL-specific cloth NOT required.")


if __name__ == "__main__":
    main()
