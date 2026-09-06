"""
Export SMPL-X body as skinned GLB for the web viewer.

- Official layout: models/smplx/male/model.npz (locked-head pack)
- Also supports: models/smplx/SMPLX_MALE.npz
- T-pose, head collapsed for face.glb attach at neck

Usage:
  python tools/smplx_export_body.py
  python tools/smplx_export_body.py --gender male

Requires:
  pip install smplx torch numpy pygltflib
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "smplx"
OUT_GLB = ROOT / "web_viewer" / "body_assets" / "body_smplx.glb"
OUT_JSON = ROOT / "web_viewer" / "body_assets" / "smplx_rig.json"
VIEWER_BODY = ROOT / "web_viewer" / "body.glb"

# First 55 joints used by LBS weights
from smplx.joint_names import JOINT_NAMES

BODY_JOINTS = JOINT_NAMES[:55]


def resolve_model_file(model_dir: Path, gender: str) -> Path:
    """
    Official download (removed head bun) layout:
      models/smplx/male/model.npz
      models/smplx/female/model.npz
      models/smplx/neutral/model.npz

    Also accepts:
      models/smplx/SMPLX_MALE.npz
      nested smplx_locked_head/.../male/model.npz
    """
    g = gender.lower()
    candidates = [
        model_dir / g / "model.npz",
        model_dir / g / "model.pkl",
        model_dir / f"SMPLX_{g.upper()}.npz",
        model_dir / f"SMPLX_{g.upper()}.pkl",
        model_dir / "smplx_locked_head" / g / "model.npz",
        model_dir / "smplx_locked_head.tar" / "smplx_locked_head" / g / "model.npz",
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_size > 1_000_000:
            return c
    raise FileNotFoundError(
        f"No SMPL-X {gender} model under {model_dir}.\n"
        f"Expected e.g. {model_dir / g / 'model.npz'} from official download."
    )


def load_official_mesh(model_file: Path, betas: np.ndarray | None = None):
    """
    Load official locked-head SMPL-X NPZ (no full hand/face tensors).
    T-pose = v_template (+ optional shape blend).
    """
    print(f"[smplx] Loading official NPZ: {model_file}")
    data = np.load(str(model_file), allow_pickle=True)
    keys = set(data.keys())
    required = {"v_template", "f", "weights", "J_regressor"}
    if not required.issubset(keys):
        raise ValueError(f"Unexpected model keys {sorted(keys)}; need {required}")

    v_template = np.array(data["v_template"], dtype=np.float32)
    faces = np.array(data["f"], dtype=np.int32)
    weights = np.array(data["weights"], dtype=np.float32)
    J_reg = np.array(data["J_regressor"], dtype=np.float32)

    # Parents from kintree_table if present
    if "kintree_table" in keys:
        kt = np.array(data["kintree_table"])
        # row0 = parents (or row1 depending on convention)
        # SMPL: kintree_table[0] = parent indices
        parents = kt[0].astype(np.int32)
        parents[0] = -1
    else:
        # fallback SMPL-X body parents for 55 joints
        parents = np.array(
            [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
            + [20] * 15 + [21] * 18,
            dtype=np.int32,
        )[: weights.shape[1]]

    verts = v_template.copy()
    if betas is not None and "shapedirs" in keys:
        shapedirs = np.array(data["shapedirs"], dtype=np.float32)  # V,3,B
        b = np.asarray(betas, dtype=np.float32).reshape(-1)
        nb = min(b.shape[0], shapedirs.shape[-1])
        verts = verts + np.einsum("vcb,b->vc", shapedirs[:, :, :nb], b[:nb])

    # Rest joints: J_regressor @ verts (sparse or dense)
    if hasattr(J_reg, "toarray"):
        J_reg = J_reg.toarray()
    J_reg = np.asarray(J_reg, dtype=np.float32)
    if J_reg.ndim == 2 and J_reg.shape[0] == weights.shape[1]:
        joints = J_reg @ verts
    elif "J" in keys:
        joints = np.array(data["J"], dtype=np.float32)
    else:
        raise ValueError("Cannot compute rest joints")

    n_j = weights.shape[1]
    joints = joints[:n_j].astype(np.float32)
    parents = parents[:n_j].astype(np.int32)
    print(
        f"[smplx] official mesh verts={verts.shape[0]} faces={faces.shape[0]} "
        f"joints={n_j} shapedirs={'yes' if 'shapedirs' in keys else 'no'}"
    )
    return verts.astype(np.float32), faces, joints, weights, parents


def generate_mesh_from_dir(model_dir: Path, gender: str, betas: np.ndarray | None = None):
    """Prefer official male/model.npz; fall back to smplx library if full model."""
    src = resolve_model_file(model_dir, gender)
    # Official locked-head packs only have ~13 keys (no hands_components*)
    try:
        data = np.load(str(src), allow_pickle=True)
        keys = set(data.keys())
    except Exception:
        keys = set()

    if "v_template" in keys and "hands_componentsl" not in keys and "expr_dirs" not in keys:
        return load_official_mesh(src, betas=betas), str(src)

    # Full SMPL-X via smplx package
    import smplx

    # rename model.npz → SMPLX_MALE.npz for library
    flat = model_dir / f"SMPLX_{gender.upper()}.npz"
    if src.name == "model.npz":
        if (not flat.is_file()) or flat.stat().st_size != src.stat().st_size:
            shutil.copy2(src, flat)
        path = str(flat)
    else:
        path = str(src)
    print(f"[smplx] Loading via smplx library: {path}")
    model = smplx.create(
        path,
        model_type="smplx",
        gender=gender if gender != "neutral" else "neutral",
        use_pca=False,
        num_betas=10,
        ext="npz",
        flat_hand_mean=True,
    )
    device = torch.device("cpu")
    model = model.to(device)
    kwargs = {}
    if betas is not None:
        b = torch.tensor(betas, dtype=torch.float32).view(1, -1)
        nb = int(model.num_betas)
        if b.shape[1] < nb:
            b = torch.cat([b, torch.zeros(1, nb - b.shape[1])], dim=1)
        kwargs["betas"] = b[:, :nb]
    with torch.no_grad():
        out = model(return_verts=True, **kwargs)
    verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = out.joints[0].detach().cpu().numpy().astype(np.float32)[:55]
    faces = np.array(model.faces, dtype=np.int32)
    weights = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
    parents = model.parents.detach().cpu().numpy().astype(np.int32)[:55]
    return (verts, faces, joints, weights, parents), path


def collapse_head(verts: np.ndarray, joints: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Pull head/face vertices toward the neck so body head doesn't fight our face.glb.
    Joints: 12=neck, 15=head
    """
    v = verts.copy()
    neck = joints[12]
    head_w = weights[:, 15]  # head bone influence
    # Also jaw / eyes if present in first 55: jaw=22, eyes=23,24
    for ji in (15, 22, 23, 24):
        if ji < weights.shape[1]:
            head_w = np.maximum(head_w, weights[:, ji])

    # Vertices mostly driven by head
    mask = head_w > 0.35
    # Soft blend
    t = np.clip((head_w[mask] - 0.35) / 0.65, 0.0, 1.0)[:, None]
    v[mask] = v[mask] * (1.0 - t) + neck[None, :] * t

    # Anything above neck + margin that still sticks out
    y_cut = neck[1] + 0.04
    high = v[:, 1] > y_cut
    v[high, 1] = np.minimum(v[high, 1], y_cut)
    # Shrink XZ toward neck for high verts
    v[high, 0] = neck[0] + (v[high, 0] - neck[0]) * 0.15
    v[high, 2] = neck[2] + (v[high, 2] - neck[2]) * 0.15
    return v


def plant_feet(verts: np.ndarray, joints: np.ndarray):
    """Shift so lowest vertex y ≈ 0."""
    ymin = float(verts[:, 1].min())
    verts = verts.copy()
    joints = joints.copy()
    verts[:, 1] -= ymin
    joints[:, 1] -= ymin
    return verts, joints, ymin


def _pack_f32(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.float32).tobytes()


def _pack_u16(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.uint16).tobytes()


def _pack_u32(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.uint32).tobytes()


def export_skinned_glb(
    path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    weights: np.ndarray,
    parents: np.ndarray,
    joint_names: list[str],
    skin_color=(0.78, 0.58, 0.46),
):
    """
    Minimal glTF 2.0 skinned mesh (Y-up).
    JOINTS: 4 bone indices per vertex (uint16)
    WEIGHTS: 4 floats per vertex
    Inverse bind matrices for each joint
    """
    from pygltflib import (
        GLTF2,
        Scene,
        Node,
        Mesh,
        Primitive,
        Attributes,
        Buffer,
        BufferView,
        Accessor,
        Skin,
        Material,
        PbrMetallicRoughness,
        Asset,
        ARRAY_BUFFER,
        ELEMENT_ARRAY_BUFFER,
        FLOAT,
        UNSIGNED_SHORT,
        UNSIGNED_INT,
        VEC3,
        VEC4,
        MAT4,
        SCALAR,
    )

    V = verts.shape[0]
    J = joints.shape[0]

    # Top-4 influences per vertex
    top_idx = np.argsort(-weights, axis=1)[:, :4].astype(np.uint16)
    top_w = np.take_along_axis(weights, top_idx.astype(np.int64), axis=1).astype(np.float32)
    # Normalize
    s = top_w.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-8)
    top_w = top_w / s

    # Normals (area-weighted face normals)
    normals = np.zeros_like(verts)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces[:, i], fn)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen = np.maximum(nlen, 1e-8)
    normals = (normals / nlen).astype(np.float32)

    # Inverse bind: world-from-joint in rest is just translation at joint position
    # IBM = inverse(joint world matrix). Rest joint world = T(joint_pos)
    ibms = np.zeros((J, 4, 4), dtype=np.float32)
    for i in range(J):
        M = np.eye(4, dtype=np.float32)
        M[:3, 3] = joints[i]
        ibms[i] = np.linalg.inv(M)
    # glTF column-major
    ibm_bytes = ibms.transpose(0, 2, 1).tobytes()

    indices = faces.reshape(-1).astype(np.uint32)

    # Build blob: indices, POSITION, NORMAL, JOINTS_0, WEIGHTS_0, inverseBindMatrices
    parts = []
    def align4(b: bytes) -> bytes:
        pad = (4 - (len(b) % 4)) % 4
        return b + (b"\x00" * pad)

    idx_b = align4(_pack_u32(indices))
    pos_b = align4(_pack_f32(verts))
    nrm_b = align4(_pack_f32(normals))
    jnt_b = align4(_pack_u16(top_idx))
    wgt_b = align4(_pack_f32(top_w))
    ibm_b = align4(ibm_bytes)

    blobs = [idx_b, pos_b, nrm_b, jnt_b, wgt_b, ibm_b]
    offsets = []
    cursor = 0
    for b in blobs:
        offsets.append(cursor)
        cursor += len(b)
    bin_blob = b"".join(blobs)

    gltf = GLTF2(
        asset=Asset(version="2.0", generator="projface_smplx_export"),
        scenes=[Scene(nodes=[0])],  # root armature node
        scene=0,
    )

    # Nodes: 0 = root (skin root), 1 = mesh node, 2..2+J-1 = joints
    # Hierarchy from parents
    # Node layout:
    #   0: Armature (skin)
    #   1: Body mesh (skinned)
    #   2 + i: joint i

    joint_node_offset = 2
    nodes = []
    # 0 armature
    nodes.append(Node(name="SMPL-X_Armature", children=[1] + [joint_node_offset + i for i in range(J) if parents[i] < 0]))
    # 1 mesh
    nodes.append(Node(name="BodyMesh", mesh=0, skin=0))

    # Joint local transforms: relative to parent
    for i in range(J):
        p = int(parents[i])
        if p < 0:
            local = joints[i].copy()
        else:
            local = joints[i] - joints[p]
        node = Node(
            name=joint_names[i],
            translation=local.tolist(),
        )
        # children
        kids = [joint_node_offset + c for c in range(J) if int(parents[c]) == i]
        if kids:
            node.children = kids
        nodes.append(node)

    # Fix armature children: only root joints under armature; mesh also child
    root_joints = [joint_node_offset + i for i in range(J) if parents[i] < 0]
    nodes[0].children = [1] + root_joints

    gltf.nodes = nodes

    gltf.meshes = [
        Mesh(
            name="SMPL-X_Body",
            primitives=[
                Primitive(
                    attributes=Attributes(
                        POSITION=1,
                        NORMAL=2,
                        JOINTS_0=3,
                        WEIGHTS_0=4,
                    ),
                    indices=0,
                    material=0,
                )
            ],
        )
    ]

    gltf.skins = [
        Skin(
            name="SMPL-X_Skin",
            inverseBindMatrices=5,
            skeleton=root_joints[0] if root_joints else joint_node_offset,
            joints=[joint_node_offset + i for i in range(J)],
        )
    ]

    gltf.materials = [
        Material(
            name="Skin",
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorFactor=[skin_color[0], skin_color[1], skin_color[2], 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.55,
            ),
            doubleSided=True,
        )
    ]

    gltf.buffers = [Buffer(byteLength=len(bin_blob))]

    # BufferViews
    gltf.bufferViews = [
        BufferView(buffer=0, byteOffset=offsets[0], byteLength=len(idx_b), target=ELEMENT_ARRAY_BUFFER),
        BufferView(buffer=0, byteOffset=offsets[1], byteLength=len(pos_b), target=ARRAY_BUFFER),
        BufferView(buffer=0, byteOffset=offsets[2], byteLength=len(nrm_b), target=ARRAY_BUFFER),
        BufferView(buffer=0, byteOffset=offsets[3], byteLength=len(jnt_b), target=ARRAY_BUFFER),
        BufferView(buffer=0, byteOffset=offsets[4], byteLength=len(wgt_b), target=ARRAY_BUFFER),
        BufferView(buffer=0, byteOffset=offsets[5], byteLength=len(ibm_b)),
    ]

    def acc_minmax(data, component_type, type_name, buffer_view, count, byte_offset=0):
        a = Accessor(
            bufferView=buffer_view,
            byteOffset=byte_offset,
            componentType=component_type,
            count=count,
            type=type_name,
        )
        if type_name == VEC3 and component_type == FLOAT:
            a.max = data.reshape(-1, 3).max(axis=0).tolist()
            a.min = data.reshape(-1, 3).min(axis=0).tolist()
        if type_name == SCALAR and component_type == UNSIGNED_INT:
            a.max = [int(data.max())]
            a.min = [int(data.min())]
        return a

    gltf.accessors = [
        acc_minmax(indices, UNSIGNED_INT, SCALAR, 0, indices.size),
        acc_minmax(verts, FLOAT, VEC3, 1, V),
        acc_minmax(normals, FLOAT, VEC3, 2, V),
        Accessor(bufferView=3, componentType=UNSIGNED_SHORT, count=V, type=VEC4),
        Accessor(bufferView=4, componentType=FLOAT, count=V, type=VEC4),
        Accessor(bufferView=5, componentType=FLOAT, count=J, type=MAT4),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    gltf.set_binary_blob(bin_blob)
    gltf.save(str(path))
    print(f"[smplx] Wrote {path} ({path.stat().st_size} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--gender", type=str, default="male", choices=["neutral", "male", "female"])
    ap.add_argument("--out", type=str, default=str(OUT_GLB))
    ap.add_argument("--no-collapse-head", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    # Mean shape for official pack (betas optional; 0 = template)
    betas = None

    (verts, faces, joints, weights, parents), src = generate_mesh_from_dir(
        model_dir, args.gender, betas=betas
    )
    # joint name list length must match
    jnames = list(BODY_JOINTS[: joints.shape[0]])
    while len(jnames) < joints.shape[0]:
        jnames.append(f"joint_{len(jnames)}")

    print(f"[smplx] verts={verts.shape[0]} faces={faces.shape[0]} joints={joints.shape[0]}")
    print(f"[smplx] bounds Y {verts[:,1].min():.3f} .. {verts[:,1].max():.3f}")

    if not args.no_collapse_head:
        verts = collapse_head(verts, joints, weights)
        print("[smplx] head collapsed for face attach")

    verts, joints, ymin = plant_feet(verts, joints)
    print(f"[smplx] planted feet (shifted Y by {-ymin:.3f})")

    out = Path(args.out)
    export_skinned_glb(
        out,
        verts,
        faces,
        joints,
        weights,
        parents,
        jnames,
    )

    # Rig metadata for runtime
    neck_i = 12 if joints.shape[0] > 12 else 0
    head_i = 15 if joints.shape[0] > 15 else min(1, joints.shape[0] - 1)
    neck = joints[neck_i].tolist()
    head = joints[head_i].tolist()
    meta = {
        "model": "smplx_official_locked_head",
        "gender": args.gender,
        "source": src,
        "joint_names": jnames,
        "parents": parents.tolist(),
        "rest_joints": joints.tolist(),
        "neck_index": neck_i,
        "head_index": head_i,
        "neck_position": neck,
        "head_position": head,
        "units": "meters",
        "up": "Y",
        "note": "Head mesh collapsed; attach face.glb to joint 'neck'",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[smplx] Wrote {OUT_JSON}")

    # Copy to viewer body.glb
    import shutil

    shutil.copy2(out, VIEWER_BODY)
    print(f"[smplx] Copied → {VIEWER_BODY}")
    print("[smplx] DONE — open web_viewer/face_viewer.html (hard refresh)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
