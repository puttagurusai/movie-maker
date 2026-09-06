"""
MoMask-style contact lock on SMPL-X after retarget.

BVH foot-IK does not survive COPY_ROTATION (different bone lengths).
This re-solves planted effectors on *this* skeleton:

  detect plant (low + slow) → lock world path → IK bake

Same rule for every clip. Swing / jump stay in the air (not planted).
Hands lock only when they are actually on the ground.
"""
from __future__ import annotations

from mathutils import Matrix, Quaternion, Vector

# Same effectors as MoMask remove_fs: planted feet only (not hands).
# SMPL-X has no toe bone — left_foot.tail is the sole / toe tip.
_FOOT_IK = (
    ("left_foot", 3),
    ("right_foot", 3),
)
_IK_BAKE = (
    "left_hip", "left_knee", "left_ankle", "left_foot",
    "right_hip", "right_knee", "right_ankle", "right_foot",
)


def _log(msg: str) -> None:
    print(f"[smplx_contact] {msg}", flush=True)


def _world_head(arm, name: str):
    pb = arm.pose.bones.get(name)
    if pb is None:
        return None
    return (arm.matrix_world @ pb.head).copy()


def _world_tail(arm, name: str):
    pb = arm.pose.bones.get(name)
    if pb is None:
        return None
    return (arm.matrix_world @ pb.tail).copy()


def _sample_track(arm, name: str, f0: int, f1: int, scene, update, *, tail: bool) -> list:
    out = []
    for f in range(int(f0), int(f1) + 1):
        scene.frame_set(f)
        update()
        w = _world_tail(arm, name) if tail else _world_head(arm, name)
        out.append(w.copy() if w is not None else Vector((0.0, 0.0, 0.0)))
    return out


def _detect_plant(track: list, floor_z: float, vel_th: float, height_th: float) -> list:
    n = len(track)
    plant = [False] * n
    if n < 2:
        return plant
    for i in range(n - 1):
        d = track[i + 1] - track[i]
        vel2 = float(d.x * d.x + d.y * d.y + d.z * d.z)
        h = float(track[i].z - floor_z)
        plant[i] = (vel2 < vel_th) and (h < height_th)
    plant[-1] = plant[-2]
    # drop 1-frame flicker
    for i in range(n):
        if not plant[i]:
            continue
        prev_on = i > 0 and plant[i - 1]
        next_on = i + 1 < n and plant[i + 1]
        if not prev_on and not next_on:
            plant[i] = False
    return plant


def _lerp(a: Vector, b: Vector, t: float) -> Vector:
    t = max(0.0, min(1.0, float(t)))
    return a * (1.0 - t) + b * t


def _lock_track(track: list, plant: list, floor_z: float, interp: int = 5) -> list:
    """MoMask remove_fs: average planted runs, slam to floor, lerp edges."""
    n = len(track)
    out = [p.copy() for p in track]
    i = 0
    while i < n:
        while i < n and not plant[i]:
            i += 1
        if i >= n:
            break
        j = i
        acc = track[i].copy()
        while j + 1 < n and plant[j + 1]:
            j += 1
            acc += track[j]
        avg = acc / float(j - i + 1)
        if abs(float(avg.z) - floor_z) < 0.12:
            avg.z = float(floor_z)
        for k in range(i, j + 1):
            out[k] = avg.copy()
        i = j + 1

    for s in range(n):
        if plant[s]:
            continue
        left = None
        right = None
        for k in range(interp):
            if s - k - 1 < 0:
                break
            if plant[s - k - 1]:
                left = s - k - 1
                break
        for k in range(interp):
            if s + k + 1 >= n:
                break
            if plant[s + k + 1]:
                right = s + k + 1
                break
        if left is None and right is None:
            continue
        if left is not None and right is not None:
            t = (s - left) / float(max(1, right - left))
            out[s] = _lerp(out[left], out[right], t)
        elif left is not None:
            t = min(1.0, (s - left) / float(interp + 1))
            out[s] = _lerp(track[s], out[left], 1.0 - t)
        else:
            t = min(1.0, (right - s) / float(interp + 1))
            out[s] = _lerp(track[s], out[right], 1.0 - t)
    return out


def _new_empty(name: str, scene):
    old = scene.objects.get(name)
    if old is not None:
        try:
            scene.collection.objects.unlink(old)
        except Exception:
            pass
        try:
            import bpy
            bpy.data.objects.remove(old, do_unlink=True)
        except Exception:
            pass
    import bpy
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = "SPHERE"
    ob.empty_display_size = 0.03
    scene.collection.objects.link(ob)
    return ob


def _clear_ik(pb) -> None:
    for c in list(pb.constraints):
        if c.type == "IK" or str(getattr(c, "name", "")).startswith("_ContactIK"):
            try:
                pb.constraints.remove(c)
            except Exception:
                pass


_FINGER_TIPS = ("index3", "middle3", "ring3", "pinky3")


def _hand_contact_height(arm, side: str, floor_z: float):
    """Lowest of wrist / knuckles / finger tips vs floor."""
    zs = []
    for n in (
        f"{side}_wrist",
        f"{side}_index1",
        f"{side}_middle1",
        f"{side}_pinky1",
        f"{side}_ring1",
    ):
        w = _world_head(arm, n)
        if w is not None:
            zs.append(float(w.z))
    for tip in _FINGER_TIPS:
        w = _world_tail(arm, f"{side}_{tip}")
        if w is not None:
            zs.append(float(w.z))
    if not zs:
        return None
    return min(zs) - float(floor_z)


def _finger_tip_zs(arm, side: str) -> list:
    zs = []
    for tip in _FINGER_TIPS:
        w = _world_tail(arm, f"{side}_{tip}")
        if w is not None:
            zs.append(float(w.z))
    return zs


def _palm_normal_and_along(arm, side: str):
    """World unsigned palm-or-back normal and finger-along direction."""
    wrist = arm.pose.bones.get(f"{side}_wrist")
    if wrist is None:
        return None, None
    w = arm.matrix_world @ wrist.head
    idx = arm.pose.bones.get(f"{side}_index1")
    pnk = arm.pose.bones.get(f"{side}_pinky1")
    mid = arm.pose.bones.get(f"{side}_middle1")
    if idx is not None and pnk is not None:
        i = arm.matrix_world @ idx.head
        p = arm.matrix_world @ pnk.head
        m = (arm.matrix_world @ mid.head) if mid is not None else (i + p) * 0.5
        across = p - i
        along = m - w
        n = across.cross(along)
        if n.length > 1e-5 and along.length > 1e-5:
            return n.normalized(), along.normalized()
    mw = (arm.matrix_world @ wrist.matrix).to_3x3()
    return (mw @ Vector((0.0, 0.0, 1.0))).normalized(), (mw @ Vector((0.0, 1.0, 0.0))).normalized()


_PALM_SIGN: dict = {}


def _palm_sign(arm, side: str) -> float:
    """+1/-1 so (sign * n) is the rest-pose PALM (faces the floor in rest)."""
    key = f"{getattr(arm, 'name', 'arm')}:{side}"
    if key in _PALM_SIGN:
        return _PALM_SIGN[key]
    import bpy

    prev = arm.data.pose_position
    act = arm.animation_data.action if arm.animation_data else None
    sign = 1.0
    try:
        if arm.animation_data:
            arm.animation_data.action = None
        arm.data.pose_position = "REST"
        bpy.context.view_layer.update()
        n, _al = _palm_normal_and_along(arm, side)
        down = Vector((0.0, 0.0, -1.0))
        if n is not None and n.dot(down) < 0.0:
            sign = -1.0
    except Exception:
        sign = 1.0
    finally:
        try:
            arm.data.pose_position = prev
        except Exception:
            pass
        try:
            if arm.animation_data is not None and act is not None:
                arm.animation_data.action = act
            bpy.context.view_layer.update()
        except Exception:
            pass
    _PALM_SIGN[key] = sign
    return sign


def _palm_facing_floor(arm, side: str):
    n, along = _palm_normal_and_along(arm, side)
    if n is None:
        return None, along
    return (n * _palm_sign(arm, side)).normalized(), along


def _outward_finger_xy(arm, side: str, along) -> Vector:
    """Finger heading in XY that does not point at the pelvis."""
    want = Vector((0.0, 0.0, 0.0))
    if along is not None:
        want = Vector((float(along.x), float(along.y), 0.0))
    wr = _world_head(arm, f"{side}_wrist")
    pel = _world_head(arm, "pelvis")
    if wr is not None and pel is not None:
        inward = Vector((float(pel.x - wr.x), float(pel.y - wr.y), 0.0))
        if inward.length > 1e-4:
            inward.normalize()
            if want.length > 1e-4 and want.normalized().dot(inward) > 0.12:
                want = want.normalized() - inward * want.normalized().dot(inward)
            if want.length < 1e-4:
                want = Vector((1.0, 0.0, 0.0)) if side == "left" else Vector((-1.0, 0.0, 0.0))
    if want.length < 1e-4:
        want = Vector((1.0, 0.0, 0.0)) if side == "left" else Vector((-1.0, 0.0, 0.0))
    return want.normalized()


def _set_bone_world_rot(arm, name: str, world_quat: Quaternion) -> None:
    pb = arm.pose.bones[name]
    wmat = arm.matrix_world @ pb.matrix
    loc, _old, scl = wmat.decompose()
    new_w = Matrix.LocRotScale(loc, world_quat, scl)
    pose_mat = arm.matrix_world.inverted() @ new_w
    local = arm.convert_space(
        pose_bone=pb, matrix=pose_mat, from_space="POSE", to_space="LOCAL"
    )
    _loc, rot, _s = local.decompose()
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = rot


def _orient_wrist_fingers_down(arm, side: str, floor_z: float, weight: float) -> bool:
    """
    Same rule as feet: the *palm skin* faces the floor, not the wrist joint.
    Palm normal is the rest-pose palm (not ±Z guess). Fingers stay outward
    so the hand does not fold toward the body as the arm travels.
    """
    import math

    import bpy

    pb = arm.pose.bones.get(f"{side}_wrist")
    if pb is None:
        return False
    nrm, along = _palm_facing_floor(arm, side)
    if nrm is None:
        return False
    down = Vector((0.0, 0.0, -1.0))
    wmat = arm.matrix_world @ pb.matrix
    _loc, wrot0, _scl = wmat.decompose()
    q0 = pb.rotation_quaternion.copy()
    wr0 = _world_head(arm, f"{side}_wrist")
    pel0 = _world_head(arm, "pelvis")
    tip0 = _world_tail(arm, f"{side}_middle3")
    want = _outward_finger_xy(arm, side, along)

    align = nrm.rotation_difference(down)
    blended = wrot0.slerp(align @ wrot0, float(weight))
    _set_bone_world_rot(arm, f"{side}_wrist", blended)
    bpy.context.view_layer.update()

    _n2, along2 = _palm_facing_floor(arm, side)
    if along2 is not None and want.length > 1e-4:
        got = Vector((along2.x, along2.y, 0.0))
        if got.length > 1e-4:
            a0 = math.atan2(got.y, got.x)
            a1 = math.atan2(want.y, want.x)
            dyaw = a1 - a0
            while dyaw > math.pi:
                dyaw -= 2.0 * math.pi
            while dyaw < -math.pi:
                dyaw += 2.0 * math.pi
            if abs(dyaw) > 0.02:
                wmat = arm.matrix_world @ pb.matrix
                _lc, wrot, _sc = wmat.decompose()
                _set_bone_world_rot(
                    arm, f"{side}_wrist", Quaternion(Vector((0.0, 0.0, 1.0)), dyaw) @ wrot
                )
                bpy.context.view_layer.update()

    # Reject a yaw that pulled fingertips toward the pelvis.
    tip1 = _world_tail(arm, f"{side}_middle3")
    wr1 = _world_head(arm, f"{side}_wrist")
    if tip0 is not None and tip1 is not None and pel0 is not None and wr1 is not None:
        d0 = (Vector((tip0.x, tip0.y, 0.0)) - Vector((pel0.x, pel0.y, 0.0))).length
        d1 = (Vector((tip1.x, tip1.y, 0.0)) - Vector((pel0.x, pel0.y, 0.0))).length
        if d1 + 0.02 < d0:
            pb.rotation_quaternion = q0
            bpy.context.view_layer.update()
            _set_bone_world_rot(arm, f"{side}_wrist", blended)
            bpy.context.view_layer.update()

    tips = []
    for tip in _FINGER_TIPS:
        w = _world_tail(arm, f"{side}_{tip}")
        if w is not None:
            tips.append(w)
    wr = _world_head(arm, f"{side}_wrist")
    if wr is not None and tips:
        tip = min(tips, key=lambda v: v.z)
        along_h = tip - wr
        along_h.z = 0.0
        if along_h.length > 0.02:
            axis = along_h.normalized().cross(Vector((0.0, 0.0, 1.0)))
            if axis.length > 1e-5:
                axis.normalize()
                dz = float(tip.z) - float(floor_z)
                if dz > 0.008:
                    ang = min(0.45, math.atan2(dz, along_h.length)) * float(weight)
                    if ang > 0.01:
                        wmat = arm.matrix_world @ pb.matrix
                        _lc, wrot, _sc = wmat.decompose()
                        _set_bone_world_rot(
                            arm, f"{side}_wrist", Quaternion(axis, -ang) @ wrot
                        )
    return True


def apply_hand_floor_orient(tgt, action, f0: int, f1: int, floor_z: float) -> dict:
    """
    When a hand is near the floor, rotate the wrist so the *palm skin*
    faces the plane — same rule as soles, never the wrist joint.
    Wrist XY is not locked (crawl travel stays). Fingers stay outward.
    """
    import bpy

    if tgt is None or "left_wrist" not in tgt.pose.bones:
        return {"ok": False, "reason": "no_wrist"}
    if not tgt.animation_data:
        tgt.animation_data_create()
    tgt.animation_data.action = action
    scene = bpy.context.scene
    update = bpy.context.view_layer.update
    counts = {"left": 0, "right": 0}
    try:
        bpy.ops.object.mode_set(mode="POSE")
    except Exception:
        pass

    for f in range(int(f0), int(f1) + 1):
        scene.frame_set(f)
        update()
        for side in ("left", "right"):
            if f"{side}_wrist" not in tgt.pose.bones:
                continue
            h = _hand_contact_height(tgt, side, floor_z)
            if h is None or h > 0.14:
                continue
            weight = 1.0 if h <= 0.06 else max(0.0, 1.0 - (h - 0.06) / 0.08)
            if weight < 0.08:
                continue
            if _orient_wrist_fingers_down(tgt, side, floor_z, weight):
                tgt.pose.bones[f"{side}_wrist"].keyframe_insert(
                    "rotation_quaternion", frame=f
                )
                counts[side] += 1

    _log(f"hand floor orient (fingers down) left={counts['left']} right={counts['right']}")
    return {"ok": True, "left": counts["left"], "right": counts["right"]}


def apply_finger_floor_lock(tgt, action, f0: int, f1: int, floor_z: float) -> dict:
    """
    Lock planted fingertips on the floor (MoMask-style, short stance windows).
    IK moves wrist/elbow only — does not glue both hands for the whole clip.
    """
    import bpy

    if tgt is None or "left_wrist" not in tgt.pose.bones:
        return {"ok": False}
    if not tgt.animation_data:
        tgt.animation_data_create()
    tgt.animation_data.action = action
    scene = bpy.context.scene
    update = bpy.context.view_layer.update
    report = []
    empties = []
    bake_extra = []
    try:
        bpy.ops.object.mode_set(mode="POSE")
    except Exception:
        pass

    for side, chain_bone in (("left", "left_wrist"), ("right", "right_wrist")):
        tip_name = f"{side}_middle3"
        if tip_name not in tgt.pose.bones or chain_bone not in tgt.pose.bones:
            continue
        raw = _sample_track(tgt, tip_name, f0, f1, scene, update, tail=True)
        plant = _detect_plant(raw, floor_z, vel_th=0.06, height_th=0.10)
        n_on = sum(1 for p in plant if p)
        n = int(f1) - int(f0) + 1
        # Long "plant" = sliding crawl hands. Averaging the whole path
        # yanks both arms inward/up. Only IK short stance windows.
        if n_on < 3 or n_on > max(8, int(n * 0.45)):
            report.append({"side": side, "planted": int(n_on), "skipped": True})
            continue
        locked = _lock_track(raw, plant, floor_z, interp=4)
        # Wrist target = locked tip minus current (tip - wrist) so fingers stay the effector
        empty = _new_empty(f"_ContactIK_{chain_bone}", scene)
        empties.append(empty)
        for i, f in enumerate(range(int(f0), int(f1) + 1)):
            scene.frame_set(f)
            update()
            wr = _world_head(tgt, chain_bone)
            tp = _world_tail(tgt, tip_name)
            if wr is None or tp is None:
                empty.location = locked[i]
            else:
                off = wr - tp
                tgt_w = locked[i] + off
                empty.location = tgt_w
            empty.keyframe_insert("location", frame=f)
        pb = tgt.pose.bones[chain_bone]
        _clear_ik(pb)
        c = pb.constraints.new("IK")
        c.name = "_ContactIK"
        c.target = empty
        c.chain_count = 2
        try:
            c.use_tail = False
        except Exception:
            pass
        c.influence = 1.0
        bake_extra.extend(
            [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]
        )
        report.append({"side": side, "planted": int(n_on)})

    if empties:
        names = [n for n in bake_extra if n in tgt.pose.bones]
        for f in range(int(f0), int(f1) + 1):
            scene.frame_set(f)
            update()
            dg = bpy.context.evaluated_depsgraph_get()
            arm_e = tgt.evaluated_get(dg)
            for name in names:
                pb = tgt.pose.bones[name]
                pb_e = arm_e.pose.bones[name]
                local = tgt.convert_space(
                    pose_bone=pb, matrix=pb_e.matrix, from_space="POSE", to_space="LOCAL"
                )
                _loc, rot, _ = local.decompose()
                pb.rotation_mode = "QUATERNION"
                pb.rotation_quaternion = rot
                pb.keyframe_insert("rotation_quaternion", frame=f)
        for side, chain_bone in (("left", "left_wrist"), ("right", "right_wrist")):
            if chain_bone in tgt.pose.bones:
                _clear_ik(tgt.pose.bones[chain_bone])
        for ob in empties:
            try:
                bpy.data.objects.remove(ob, do_unlink=True)
            except Exception:
                pass
    _log("finger floor lock " + ", ".join(
        f"{r.get('side')}={r.get('planted', 0)}" for r in report
    ))
    return {"ok": True, "effectors": report}


def replant_from_supports(tgt, action, f0: int, f1: int, floor_z: float) -> dict:
    """
    One constant pelvis Z so the lowest *real* supports (finger tips, knees,
    foot soles) sit on the floor. Drop or tiny lift — never a big upward hop.
    """
    import bpy

    if tgt is None or "pelvis" not in tgt.pose.bones:
        return {"ok": False}
    if not tgt.animation_data:
        tgt.animation_data_create()
    tgt.animation_data.action = action
    scene = bpy.context.scene
    update = bpy.context.view_layer.update

    def _support():
        zs = []
        for n in ("left_foot", "right_foot"):
            w = _world_tail(tgt, n)
            if w is not None:
                zs.append(float(w.z))
        for n in ("left_knee", "right_knee"):
            w = _world_head(tgt, n)
            if w is not None:
                zs.append(float(w.z))
        for side in ("left", "right"):
            for tip in _FINGER_TIPS:
                w = _world_tail(tgt, f"{side}_{tip}")
                if w is not None:
                    zs.append(float(w.z))
        return min(zs) if zs else None

    samples = []
    for f in range(int(f0), int(f1) + 1):
        scene.frame_set(f)
        update()
        s = _support()
        if s is not None:
            samples.append(s)
    if not samples:
        return {"ok": False, "reason": "no_support"}
    ordered = sorted(samples)
    lo_ref = ordered[max(0, int(0.12 * (len(ordered) - 1)))]
    dz = float(floor_z) - lo_ref
    if abs(dz) < 0.003:
        dz = 0.0
    # Do not translate the skeleton. Walks must stay at source standing
    # height; crawl height comes from stamped Hips. Pulling everyone down
    # to the lowest finger/knee puts locomotion through the floor.
    _log(f"replant skipped (keep source hips height) lo_ref={lo_ref:.4f}")
    return {"ok": True, "dz": 0.0, "lo_ref": round(float(lo_ref), 4), "skipped": True}


def apply_contact_ik(tgt, action, f0: int, f1: int, floor_z: float) -> dict:
    """
    Lock planted SMPL-X effectors and IK-bake rotations.
    Requires the Action already bound on tgt.
    """
    import bpy

    if tgt is None or action is None or "pelvis" not in tgt.pose.bones:
        return {"ok": False, "reason": "no_target"}
    if not tgt.animation_data:
        tgt.animation_data_create()
    tgt.animation_data.action = action
    scene = bpy.context.scene
    update = bpy.context.view_layer.update
    f0, f1 = int(f0), int(f1)
    n = f1 - f0 + 1
    if n < 4:
        return {"ok": False, "reason": "short"}

    specs = [(bone, chain) for bone, chain in _FOOT_IK if bone in tgt.pose.bones]

    empties = []
    report = []
    try:
        bpy.ops.object.mode_set(mode="POSE")
    except Exception:
        pass

    # Sole = foot tail. Detect + lock that point; IK use_tail=True.
    vel_th, h_th = 0.05, 0.07
    for bone, chain in specs:
        raw = _sample_track(tgt, bone, f0, f1, scene, update, tail=True)
        plant = _detect_plant(raw, floor_z, vel_th, h_th)
        n_on = sum(1 for p in plant if p)
        locked = _lock_track(raw, plant, floor_z, interp=5)
        empty = _new_empty(f"_ContactIK_{bone}", scene)
        empties.append(empty)
        for i, f in enumerate(range(f0, f1 + 1)):
            empty.location = locked[i]
            empty.keyframe_insert("location", frame=f)
        pb = tgt.pose.bones[bone]
        _clear_ik(pb)
        c = pb.constraints.new("IK")
        c.name = "_ContactIK"
        c.target = empty
        c.chain_count = int(chain)
        try:
            c.use_tail = True
        except Exception:
            pass
        c.influence = 1.0
        report.append({"bone": bone, "planted": int(n_on), "chain": chain})

    if not empties:
        hands = apply_hand_floor_orient(tgt, action, f0, f1, floor_z)
        fingers = apply_finger_floor_lock(tgt, action, f0, f1, floor_z)
        replant = replant_from_supports(tgt, action, f0, f1, floor_z)
        return {
            "ok": True,
            "mode": "no_plants",
            "effectors": report,
            "hands": hands,
            "fingers": fingers,
            "replant": replant,
        }

    # Visual-key the IK result onto the Action (no nla.bake — unreliable in bg)
    bake_names = [b for b in _IK_BAKE if b in tgt.pose.bones]
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        update()
        dg = bpy.context.evaluated_depsgraph_get()
        arm_e = tgt.evaluated_get(dg)
        for name in bake_names:
            pb = tgt.pose.bones[name]
            pb_e = arm_e.pose.bones[name]
            local = tgt.convert_space(
                pose_bone=pb, matrix=pb_e.matrix, from_space="POSE", to_space="LOCAL"
            )
            _loc, rot, _ = local.decompose()
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = rot
            pb.keyframe_insert("rotation_quaternion", frame=f)

    for bone, _chain in specs:
        if bone in tgt.pose.bones:
            _clear_ik(tgt.pose.bones[bone])
    for ob in empties:
        try:
            bpy.data.objects.remove(ob, do_unlink=True)
        except Exception:
            pass

    # Do NOT lift the whole body here — that raised crawls off the floor.
    scene.frame_set(f0)
    update()
    hands = apply_hand_floor_orient(tgt, action, f0, f1, floor_z)
    fingers = apply_finger_floor_lock(tgt, action, f0, f1, floor_z)
    replant = replant_from_supports(tgt, action, f0, f1, floor_z)
    planted_total = sum(int(r.get("planted") or 0) for r in report)
    _log(
        f"contact IK effectors={len(report)} planted_frames_sum={planted_total} "
        + ", ".join(f"{r['bone']}={r.get('planted', 0)}" for r in report)
    )
    return {
        "ok": True,
        "mode": "remove_fs_ik",
        "effectors": report,
        "planted_frames_sum": planted_total,
        "floor_z": round(float(floor_z), 4),
        "hands": hands,
        "fingers": fingers,
        "replant": replant,
    }
