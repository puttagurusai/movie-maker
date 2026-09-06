"""
Blender-only body motion library for whole body.blend

Creates named Actions (pose keyframes) on SMPL-X_Armature, renders
preview stills, and saves a working motion file.

Run (background):
  "C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe" ^
    --background "C:\\me\\proj\\projface_v1\\whole body.blend" ^
    --python tools/blender_body_motion.py

Interactive (open result):
  open  body_motion/whole_body_motions.blend
  select SMPL-X_Armature → Dope Sheet / Action Editor → pick Action

Axes (probed on this rig, local XYZ euler):
  right_shoulder raise  → Z negative
  left_shoulder  raise  → Z positive
  right_elbow    flex   → Z negative  (wrist up toward shoulder)
  left_elbow     flex   → Z positive
  shoulder forward      → Y (sign differs L/R — see helpers)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Euler

ROOT = Path(__file__).resolve().parents[1]
OUT_BLEND = ROOT / "body_motion" / "whole_body_motions.blend"
OUT_SS = ROOT / "body_motion" / "previews"
ARM_NAME = "SMPL-X_Armature"
FPS = 24

# ── bone helpers (local XYZ euler radians) ───────────────────────────

def _e(x=0.0, y=0.0, z=0.0):
    return (float(x), float(y), float(z))


def r_sh(raise_up=0.0, forward=0.0, twist=0.0):
    # raise: Z-, forward: Y- pulls arm toward body front on this export
    return _e(twist, -forward, -raise_up)


def l_sh(raise_up=0.0, forward=0.0, twist=0.0):
    return _e(twist, forward, raise_up)


def r_elb(flex=0.0, twist=0.0):
    return _e(twist, 0.0, -flex)


def l_elb(flex=0.0, twist=0.0):
    return _e(twist, 0.0, flex)


def rest_soft():
    """Slightly relaxed T-pose (arms a bit down, elbows soft)."""
    return {
        "left_shoulder": l_sh(raise_up=-0.25, forward=0.08),
        "right_shoulder": r_sh(raise_up=-0.25, forward=0.08),
        "left_elbow": l_elb(0.35),
        "right_elbow": r_elb(0.35),
        "spine3": _e(0.0, 0.0, 0.0),
        "spine2": _e(0.0, 0.0, 0.0),
        "neck": _e(0.0, 0.0, 0.0),
        "head": _e(0.0, 0.0, 0.0),
        "pelvis": _e(0.0, 0.0, 0.0),
        "left_hip": _e(0.0, 0.0, 0.0),
        "right_hip": _e(0.0, 0.0, 0.0),
        "left_knee": _e(0.0, 0.0, 0.0),
        "right_knee": _e(0.0, 0.0, 0.0),
    }


# Each motion: list of (frame, bone→euler)  (frames at FPS=24)
# Keys are absolute pose eulers (not deltas).
MOTIONS: dict[str, list[tuple[int, dict]]] = {
    "rest": [
        (1, rest_soft()),
        (24, rest_soft()),
    ],
    "wave": [
        (1, rest_soft()),
        (8, {
            **rest_soft(),
            "spine3": _e(0.0, 0.05, -0.06),
            "right_shoulder": r_sh(raise_up=0.9, forward=0.15),
            "right_elbow": r_elb(1.0),
            "right_wrist": _e(0.1, 0.2, -0.2),
            "left_shoulder": l_sh(raise_up=-0.3, forward=0.05),
        }),
        (14, {
            **rest_soft(),
            "spine3": _e(-0.02, 0.06, -0.06),
            "right_collar": _e(0.05, 0.05, -0.08),
            "right_shoulder": r_sh(raise_up=1.15, forward=0.2),
            "right_elbow": r_elb(1.15, twist=0.15),
            "right_wrist": _e(0.15, 0.45, -0.25),
            "neck": _e(0.0, 0.05, 0.0),
        }),
        (20, {
            **rest_soft(),
            "right_shoulder": r_sh(raise_up=1.1, forward=0.12),
            "right_elbow": r_elb(1.05, twist=-0.12),
            "right_wrist": _e(0.1, -0.35, -0.2),
        }),
        (26, {
            **rest_soft(),
            "right_shoulder": r_sh(raise_up=1.15, forward=0.2),
            "right_elbow": r_elb(1.1, twist=0.12),
            "right_wrist": _e(0.12, 0.4, -0.25),
        }),
        (36, {
            **rest_soft(),
            "right_shoulder": r_sh(raise_up=0.5, forward=0.1),
            "right_elbow": r_elb(0.6),
        }),
        (48, rest_soft()),
    ],
    "shrug": [
        (1, rest_soft()),
        (10, {
            **rest_soft(),
            # collar lift + small shoulder raise, elbows out (not arms overhead)
            "left_collar": _e(0.22, 0.05, 0.15),
            "right_collar": _e(0.22, -0.05, -0.15),
            "left_shoulder": l_sh(raise_up=0.22, forward=0.12),
            "right_shoulder": r_sh(raise_up=0.22, forward=0.12),
            "left_elbow": l_elb(1.05),
            "right_elbow": r_elb(1.05),
            "spine3": _e(0.05, 0.0, 0.0),
            "neck": _e(-0.06, 0.0, 0.0),
            "left_wrist": _e(0.15, 0.25, 0.1),
            "right_wrist": _e(0.15, -0.25, -0.1),
        }),
        (22, {
            **rest_soft(),
            "left_collar": _e(0.18, 0.04, 0.12),
            "right_collar": _e(0.18, -0.04, -0.12),
            "left_shoulder": l_sh(raise_up=0.18, forward=0.1),
            "right_shoulder": r_sh(raise_up=0.18, forward=0.1),
            "left_elbow": l_elb(0.95),
            "right_elbow": r_elb(0.95),
        }),
        (36, rest_soft()),
    ],
    "talk_open": [
        (1, {
            **rest_soft(),
            "left_shoulder": l_sh(raise_up=-0.1, forward=0.2),
            "right_shoulder": r_sh(raise_up=-0.1, forward=0.2),
            "left_elbow": l_elb(0.7),
            "right_elbow": r_elb(0.7),
        }),
        (12, {
            **rest_soft(),
            "left_shoulder": l_sh(raise_up=-0.05, forward=0.35),
            "right_shoulder": r_sh(raise_up=0.0, forward=0.45),
            "left_elbow": l_elb(0.85),
            "right_elbow": r_elb(0.55),
            "spine3": _e(-0.03, 0.04, -0.05),
            "right_wrist": _e(0.1, -0.1, -0.1),
        }),
        (24, {
            **rest_soft(),
            "left_shoulder": l_sh(raise_up=0.0, forward=0.4),
            "right_shoulder": r_sh(raise_up=-0.08, forward=0.25),
            "left_elbow": l_elb(0.6),
            "right_elbow": r_elb(0.8),
            "spine3": _e(-0.02, -0.03, 0.04),
        }),
        (36, {
            **rest_soft(),
            "left_shoulder": l_sh(raise_up=-0.1, forward=0.2),
            "right_shoulder": r_sh(raise_up=-0.1, forward=0.2),
            "left_elbow": l_elb(0.7),
            "right_elbow": r_elb(0.7),
        }),
    ],
    "point": [
        (1, rest_soft()),
        (10, {
            **rest_soft(),
            "spine3": _e(-0.04, 0.05, -0.08),
            "right_shoulder": r_sh(raise_up=0.35, forward=0.85),
            "right_elbow": r_elb(0.25),
            "right_wrist": _e(0.0, 0.0, -0.15),
            "left_shoulder": l_sh(raise_up=-0.28, forward=0.05),
            "left_elbow": l_elb(0.5),
            "neck": _e(0.0, 0.08, 0.0),
            "head": _e(0.0, 0.1, 0.0),
        }),
        (28, {
            **rest_soft(),
            "spine3": _e(-0.04, 0.05, -0.08),
            "right_shoulder": r_sh(raise_up=0.32, forward=0.9),
            "right_elbow": r_elb(0.2),
            "neck": _e(0.0, 0.08, 0.0),
        }),
        (40, rest_soft()),
    ],
    "think": [
        (1, rest_soft()),
        (14, {
            **rest_soft(),
            "spine3": _e(0.06, 0.04, -0.05),
            "neck": _e(0.08, 0.12, 0.0),
            "head": _e(0.1, 0.15, 0.05),
            "right_shoulder": r_sh(raise_up=0.55, forward=0.55),
            "right_elbow": r_elb(1.35),
            "right_wrist": _e(0.3, 0.2, -0.4),
            "left_shoulder": l_sh(raise_up=-0.3, forward=0.05),
            "left_elbow": l_elb(0.4),
        }),
        (40, {
            **rest_soft(),
            "spine3": _e(0.05, 0.03, -0.04),
            "neck": _e(0.06, 0.1, 0.0),
            "head": _e(0.08, 0.12, 0.04),
            "right_shoulder": r_sh(raise_up=0.5, forward=0.5),
            "right_elbow": r_elb(1.3),
            "right_wrist": _e(0.25, 0.15, -0.35),
        }),
        (56, rest_soft()),
    ],
    "recoil": [
        (1, rest_soft()),
        (6, {
            **rest_soft(),
            "pelvis": _e(0.08, 0.0, 0.0),
            "spine1": _e(0.1, 0.0, 0.05),
            "spine2": _e(0.12, 0.0, 0.06),
            "spine3": _e(0.15, 0.0, 0.08),
            "neck": _e(0.1, 0.0, 0.05),
            "head": _e(0.12, 0.0, 0.05),
            "left_shoulder": l_sh(raise_up=0.2, forward=-0.3),
            "right_shoulder": r_sh(raise_up=0.2, forward=-0.3),
            "left_elbow": l_elb(1.0),
            "right_elbow": r_elb(1.0),
            "left_hip": _e(0.15, 0.0, 0.0),
            "right_hip": _e(0.15, 0.0, 0.0),
        }),
        (18, {
            **rest_soft(),
            "pelvis": _e(0.04, 0.0, 0.0),
            "spine3": _e(0.06, 0.0, 0.03),
            "left_shoulder": l_sh(raise_up=0.05, forward=-0.1),
            "right_shoulder": r_sh(raise_up=0.05, forward=-0.1),
            "left_elbow": l_elb(0.7),
            "right_elbow": r_elb(0.7),
        }),
        (32, rest_soft()),
    ],
    "celebrate": [
        (1, rest_soft()),
        (10, {
            **rest_soft(),
            "spine3": _e(-0.08, 0.0, 0.0),
            "left_shoulder": l_sh(raise_up=1.2, forward=0.1),
            "right_shoulder": r_sh(raise_up=1.2, forward=0.1),
            "left_elbow": l_elb(0.4),
            "right_elbow": r_elb(0.4),
            "left_wrist": _e(0.0, 0.2, 0.1),
            "right_wrist": _e(0.0, -0.2, -0.1),
        }),
        (18, {
            **rest_soft(),
            "spine3": _e(-0.1, 0.05, 0.0),
            "left_shoulder": l_sh(raise_up=1.3, forward=0.15),
            "right_shoulder": r_sh(raise_up=1.1, forward=0.1),
            "left_elbow": l_elb(0.55),
            "right_elbow": r_elb(0.35),
        }),
        (26, {
            **rest_soft(),
            "spine3": _e(-0.1, -0.05, 0.0),
            "left_shoulder": l_sh(raise_up=1.1, forward=0.1),
            "right_shoulder": r_sh(raise_up=1.3, forward=0.15),
            "left_elbow": l_elb(0.35),
            "right_elbow": r_elb(0.55),
        }),
        (40, rest_soft()),
    ],
    "sit": [
        # Hip X+ = thigh forward; knee X- = shin back under chair (probed on this rig).
        # Pelvis location Z is keyed separately in make_action via _LOC_KEYS.
        (1, rest_soft()),
        (20, {
            **rest_soft(),
            "spine1": _e(0.12, 0.0, 0.0),
            "spine2": _e(0.1, 0.0, 0.0),
            "spine3": _e(0.06, 0.0, 0.0),
            "left_hip": _e(1.05, 0.0, 0.06),
            "right_hip": _e(1.05, 0.0, -0.06),
            "left_knee": _e(-1.25, 0.0, 0.0),
            "right_knee": _e(-1.25, 0.0, 0.0),
            "left_ankle": _e(0.35, 0.0, 0.0),
            "right_ankle": _e(0.35, 0.0, 0.0),
            "left_shoulder": l_sh(raise_up=-0.15, forward=0.12),
            "right_shoulder": r_sh(raise_up=-0.15, forward=0.12),
            "left_elbow": l_elb(0.55),
            "right_elbow": r_elb(0.55),
        }),
        (48, {
            **rest_soft(),
            "spine1": _e(0.12, 0.0, 0.0),
            "spine2": _e(0.1, 0.0, 0.0),
            "spine3": _e(0.06, 0.0, 0.0),
            "left_hip": _e(1.05, 0.0, 0.06),
            "right_hip": _e(1.05, 0.0, -0.06),
            "left_knee": _e(-1.25, 0.0, 0.0),
            "right_knee": _e(-1.25, 0.0, 0.0),
            "left_ankle": _e(0.35, 0.0, 0.0),
            "right_ankle": _e(0.35, 0.0, 0.0),
            "left_shoulder": l_sh(raise_up=-0.15, forward=0.12),
            "right_shoulder": r_sh(raise_up=-0.15, forward=0.12),
            "left_elbow": l_elb(0.55),
            "right_elbow": r_elb(0.55),
        }),
    ],
    "nod": [
        (1, {**rest_soft(), "neck": _e(0.0, 0.0, 0.0), "head": _e(0.0, 0.0, 0.0)}),
        (8, {**rest_soft(), "neck": _e(0.25, 0.0, 0.0), "head": _e(0.2, 0.0, 0.0)}),
        (16, {**rest_soft(), "neck": _e(-0.05, 0.0, 0.0), "head": _e(-0.05, 0.0, 0.0)}),
        (24, {**rest_soft(), "neck": _e(0.18, 0.0, 0.0), "head": _e(0.15, 0.0, 0.0)}),
        (32, {**rest_soft(), "neck": _e(0.0, 0.0, 0.0), "head": _e(0.0, 0.0, 0.0)}),
    ],
}

# Optional pelvis location keys (bone location in pose space) per action
LOC_KEYS: dict[str, list[tuple[int, dict[str, tuple]]]] = {
    "sit": [
        (1, {"pelvis": (0.0, 0.0, 0.0)}),
        (20, {"pelvis": (0.0, 0.08, -0.42)}),
        (48, {"pelvis": (0.0, 0.08, -0.42)}),
    ],
}


def log(msg: str) -> None:
    print(f"[body_motion] {msg}", flush=True)


def get_armature():
    arm = bpy.data.objects.get(ARM_NAME)
    if not arm:
        for o in bpy.data.objects:
            if o.type == "ARMATURE":
                return o
    return arm


def clear_pose(arm):
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.location = (0.0, 0.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()


def apply_pose(arm, pose: dict):
    clear_pose(arm)
    for bone_name, eul in pose.items():
        pb = arm.pose.bones.get(bone_name)
        if not pb:
            continue
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = Euler(eul, "XYZ")
    bpy.context.view_layer.update()


def insert_keys(arm, frame: int, pose: dict):
    apply_pose(arm, pose)
    for bone_name in pose:
        pb = arm.pose.bones.get(bone_name)
        if not pb:
            continue
        pb.keyframe_insert(data_path="rotation_euler", frame=frame)
    # also key bones that should stay at rest so they don't float from previous action
    for bone_name in rest_soft():
        if bone_name in pose:
            continue
        pb = arm.pose.bones.get(bone_name)
        if pb:
            pb.keyframe_insert(data_path="rotation_euler", frame=frame)


def make_action(arm, name: str, keys: list[tuple[int, dict]]) -> bpy.types.Action:
    # remove old action with same name
    old = bpy.data.actions.get(name)
    if old:
        bpy.data.actions.remove(old)

    action = bpy.data.actions.new(name=name)
    # Blender 4+/5: assign via animation data
    if not arm.animation_data:
        arm.animation_data_create()
    arm.animation_data.action = action

    clear_pose(arm)
    for frame, pose in keys:
        insert_keys(arm, frame, pose)

    # location keys (e.g. sit pelvis drop)
    for frame, locs in LOC_KEYS.get(name, []):
        for bone_name, loc in locs.items():
            pb = arm.pose.bones.get(bone_name)
            if not pb:
                continue
            pb.location = loc
            pb.keyframe_insert(data_path="location", frame=frame)

    # set linear / bezier
    try:
        for fc in action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"
                kp.handle_left_type = "AUTO_CLAMPED"
                kp.handle_right_type = "AUTO_CLAMPED"
    except Exception:
        pass

    end = max(f for f, _ in keys)
    action.use_frame_range = True
    action.frame_start = 1
    action.frame_end = end
    log(f"action '{name}'  frames 1–{end}  keys={len(keys)}")
    return action


def setup_scene_fps():
    sc = bpy.context.scene
    sc.render.fps = FPS
    sc.frame_start = 1
    sc.frame_end = 48


def setup_camera_preview(arm):
    """Front 3/4 camera looking at torso."""
    cam = bpy.data.objects.get("Camera")
    if not cam:
        bpy.ops.object.camera_add()
        cam = bpy.context.active_object
        cam.name = "Camera"
    # character ~1.7m tall, center torso
    cam.location = (2.2, -2.4, 1.35)
    cam.rotation_euler = Euler((math.radians(75), 0.0, math.radians(40)), "XYZ")
    bpy.context.scene.camera = cam

    # light
    if not bpy.data.objects.get("Light"):
        bpy.ops.object.light_add(type="AREA", location=(1.5, -1.5, 2.5))
        bpy.context.active_object.data.energy = 200


def render_preview(arm, action_name: str, frame: int, out_path: Path):
    sc = bpy.context.scene
    if not arm.animation_data:
        arm.animation_data_create()
    act = bpy.data.actions.get(action_name)
    if not act:
        return
    arm.animation_data.action = act
    sc.frame_set(frame)
    bpy.context.view_layer.update()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sc.render.resolution_x = 960
    sc.render.resolution_y = 1280
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = str(out_path)

    # EEVEE viewport-like if available
    try:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        try:
            sc.render.engine = "BLENDER_EEVEE"
        except Exception:
            sc.render.engine = "BLENDER_WORKBENCH"

    bpy.ops.render.render(write_still=True)
    log(f"preview {action_name} f{frame} → {out_path.name}")


def fix_bodymesh_parent(arm):
    """
    BodyMesh was bone-parented to 'neck' which can break full-body skinning
    display. Re-parent to armature object; Armature modifier already exists.
    """
    bm = bpy.data.objects.get("BodyMesh")
    if not bm:
        return
    if bm.parent_type == "BONE" and bm.parent_bone:
        log(f"fixing BodyMesh parent (was BONE '{bm.parent_bone}') → armature object")
        # keep world matrix
        mw = bm.matrix_world.copy()
        bm.parent = arm
        bm.parent_type = "OBJECT"
        bm.parent_bone = ""
        bm.matrix_world = mw
        # ensure armature modifier
        has_arm = any(m.type == "ARMATURE" for m in bm.modifiers)
        if not has_arm:
            mod = bm.modifiers.new("Armature", "ARMATURE")
            mod.object = arm
        else:
            for m in bm.modifiers:
                if m.type == "ARMATURE":
                    m.object = arm


def write_action_index():
    path = ROOT / "body_motion" / "ACTIONS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Body motion Actions (Blender)",
        "",
        f"File: `{OUT_BLEND.relative_to(ROOT).as_posix()}`",
        f"Armature: `{ARM_NAME}`",
        "",
        "| Action | Frames | Use for |",
        "|--------|--------|---------|",
    ]
    for name, keys in MOTIONS.items():
        end = max(f for f, _ in keys)
        lines.append(f"| `{name}` | 1–{end} | director / polish |")
    lines += [
        "",
        "## How to edit in Blender",
        "",
        "1. Open `body_motion/whole_body_motions.blend`",
        "2. Select **SMPL-X_Armature** → **Pose Mode**",
        "3. **Dope Sheet → Action Editor** → dropdown pick action (wave, shrug, …)",
        "4. Pose bones, **I → Rotation** to keyframe",
        "5. Play timeline (Space) to preview",
        "6. **File → Save**",
        "",
        "Later export (not now): Action → NLA strip → glTF with animation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {path}")


def main():
    log("start")
    arm = get_armature()
    if not arm:
        log("ERROR: no armature")
        sys.exit(1)

    bpy.context.view_layer.objects.active = arm
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    fix_bodymesh_parent(arm)
    setup_scene_fps()
    setup_camera_preview(arm)

    # Create all actions
    bpy.ops.object.mode_set(mode="POSE")
    for name, keys in MOTIONS.items():
        make_action(arm, name, keys)

    # Leave on wave for open convenience
    if arm.animation_data and bpy.data.actions.get("wave"):
        arm.animation_data.action = bpy.data.actions["wave"]

    bpy.ops.object.mode_set(mode="OBJECT")

    # Previews at peak-ish frames
    peaks = {
        "rest": 12,
        "wave": 14,
        "shrug": 10,
        "talk_open": 12,
        "point": 10,
        "think": 14,
        "recoil": 6,
        "celebrate": 18,
        "sit": 24,
        "nod": 8,
    }
    OUT_SS.mkdir(parents=True, exist_ok=True)
    for name, fr in peaks.items():
        if name in MOTIONS:
            render_preview(arm, name, fr, OUT_SS / f"{name}_f{fr:03d}.png")

    write_action_index()

    OUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
    log(f"saved {OUT_BLEND}")
    log("DONE — open blend, Action Editor, polish poses by eye")


if __name__ == "__main__":
    main()
