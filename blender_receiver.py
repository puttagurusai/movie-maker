"""
blender_receiver.py  (Updated for emotion + viseme + live MediaPipe + movie camera)

Supports two modes simultaneously:
1. Live MediaPipe face capture (from mediapipe_face_capture.py)
2. Orchestrator-driven emotion + rhubarb lip-sync (from orchestrator.py)
3. Movie camera track (type=camera plan keyframes)

Packet formats supported:
- Legacy / MediaPipe: {"blendshapes": {...}} or {"t": ..., "blendshapes": {...}}
- Emotion:             {"type": "emotion", "blendshapes": {...}}   → upper face only
- Viseme:              {"type": "viseme",  "blendshapes": {...}}   → mouth/jaw only
- Body:                {"type": "body", "action": "..."}
- Camera:              {"type": "camera", "op": "plan"|"rest", "keyframes": [...]}

Run inside Blender:
    1. Open this file in Text Editor → Run Script
    2. In Python Console:
         bpy.ops.face.stream_receiver()

Stop with:
    bpy.ops.face.stream_stop()
"""

import json
import math
import queue
import socket
import threading
import time

import bpy
from mathutils import Vector

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 9001

# Leave empty for AUTO-DETECT + AUTO-SELECT best mesh with shape keys
TARGET_MESH_NAME = ""

# Per-packet-type smoothing (fraction of new sample mixed into target each packet)
# Mouth must track audio tightly or lips look late vs speech.
VISEME_SMOOTHING = 0.92   # nearly snap lips to packet (sync with audio)
EMOTION_SMOOTHING = 0.55  # was 0.28; higher = packets drive target faster, EMOTION_GLIDE still softens mesh
SMOOTHING = 0.30

# Glide rates toward TARGET_VALUES each timer tick (30 Hz). Higher = snappier.
VISEME_GLIDE = 0.80       # lips follow targets immediately
EMOTION_GLIDE = 0.40      # was 0.28; 0.40 reaches 90% of target in ~4.5 frames (~150ms)

# Verbose packet logging kills real-time performance (30 prints/sec)
VERBOSE_PACKETS = False

# Only used for legacy/manual name fixing
NAME_OVERRIDES = {}

# Timeouts for orchestrator / live sender
FIRST_SIGNAL_TIMEOUT = 1200.0   # How long to wait for the VERY FIRST packet after starting the receiver (the "first signal")

# After the first packet has been received ("connected"), we no longer auto-stop on short silence.
# The old 3.5s SENDER_SILENCE_TIMEOUT was causing unwanted stops during pauses between sentences
# or while user is pasting the next JSON. Once connected, the receiver should stay alive
# until the user explicitly calls bpy.ops.face.stream_stop() (or the whole Blender session ends).
# We keep a very long "post-connection" timeout only as a safety net.
POST_CONNECTION_SILENCE_TIMEOUT = 300.0   # 5 minutes of total silence after connection before giving up

# ---------------------------------------------------------------------------
# UPPER vs LOWER FACE SEPARATION
# ---------------------------------------------------------------------------
# Upper face driven by emotion packets. Mouth smiles/frowns stay available
# to lip-sync (wav2arkit needs stretch/smile for speech shapes).
UPPER_FACE_KEYS = {
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekSquintLeft", "cheekSquintRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    # Now sent via emotion packet only (not viseme), so they use slow EMOTION_GLIDE.
    # This lets happy/sad expressions show on the mouth corners during speech.
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthPressLeft", "mouthPressRight",   # sad lip tightening/droop
    # noseSneer driven by Brain upper head — critical for angry/disgusted expressions.
    "noseSneerLeft", "noseSneerRight",
}

# Everything else (jaw, full mouth shapes, etc.) is considered lower/mouth for visemes
# We don't need an explicit list — we just avoid upper keys for viseme packets.

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
_data_queue: "queue.Queue" = queue.Queue()
_smoothed_values = {}
TARGET_VALUES = {}
_tracked_keys = set()
_running = False
_sock = None
_listener_thread = None

_start_time = None
_last_packet_time = None

# Head rotation target — written by background threads, applied by main modal() thread.
# NEVER set obj.rotation_euler from a background thread: Blender's RNA message bus
# (WM_msg_publish_rna) is not thread-safe and causes EXCEPTION_ACCESS_VIOLATION.
_HEAD_ROTATION: dict = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

# Body Action playback (Mixamo retargeted clips on SMPL-X_Armature)
# Packet: {"type":"body","action":"wave","duration":3.2,"intensity":0.8}
# After audio ends (duration) or explicit rest=True → smooth slerp to rest at SAME root location.
_BODY_PLAY = {
    "action": None,       # Action name for live scrub / last clip
    "last_action": None,  # keep after rest so timeline scrub can replay
    "t0": 0.0,            # wall time when action started
    "speed": 1.0,
    "loop": True,
    "f0": 1,
    "f1": 60,
    "arm_name": "SMPL-X_Armature",
    "last_frame": None,   # skip scene.frame_set when unchanged (big FPS win)
    "pending_assign": False,
    "duration": None,     # seconds; stop clip when elapsed >= duration (audio length)
    "live": False,        # True = wall-clock advance; False = user/timeline controls frame
    "resting": False,     # smooth blend to rest in progress
    "rest_t0": 0.0,
    "rest_duration": 0.55,
    "rest_from": {},      # bone → (w,x,y,z) at start of rest blend
    "rest_to": {},        # target IDLE pose (hands down) — never T-pose identity
    "rest_pelvis_loc": None,  # keep standing location (no teleport)
    "idle_pose": {},      # cached hands-down idle for hold after settle
    "idle_hold": False,   # True = keep applying idle pose each tick after rest
}

# Movie camera track — wall-clock during speech; keyframes for timeline scrub.
# Packet: {"type":"camera","op":"plan","duration":4.2,"keyframes":[{t,location,look_at,fov_deg},...]}
# subject_relative: location/look are offsets from live body anchor (head/chest/full_body)
_CAMERA_PLAY = {
    "live": False,
    "t0": 0.0,
    "duration": None,
    "fps": 20.0,
    "keyframes": [],      # list of {t, location, look_at, fov_deg, rel_cam?, rel_look?}
    "camera_name": "MovieCam",
    "look_at_name": "MovieCam_LookAt",
    "track_head": True,
    "last_t": -1.0,
    "subject_relative": True,
    "subject_anchor": "chest",  # head | chest | pelvis | full_body
    "follow_lag": 0.18,
    "hold_after": True,
    "smooth_loc": None,
    "smooth_look": None,
    "min_cam_dist": 1.08,
    "session_clips": [],  # [{frame_start, frame_end, camera_name, look_name}]
    "cam_action": "",
    "look_action": "",
}
SESSION_CAM_PREFIX = "SessionCam_"
MASTER_EVAL_FPS = 20.0

# Session body timeline — LIGHTWEIGHT NLA append (no per-frame bake = no Blender freeze).
# Clip1 ends 133 → clip2 starts 134. User scrubs timeline manually; live play is current clip only.
SESSION_ACTION_NAME = "Session_Timeline"
SESSION_NLA_TRACK = "SessionBody"
_SESSION_BODY = {
    "clips": [],           # [{index, source_action, frame_start, frame_end, span}]
    "frame_cursor": 1,     # next free frame (first empty frame after last clip)
    "continue_root": True,
    "action_name": SESSION_ACTION_NAME,
    "nla_track": SESSION_NLA_TRACK,
    "needs_movie_rerender": False,
}

# Face shape-key timeline bake — scrub/replay lips after live UDP play.
# Packet: {"type":"face_keyframes","path":"temp/face_timeline_….json"} or inline keyframes
_FACE_TIMELINE = {
    "active": False,
    "frame_start": 1,
    "frame_end": 1,
    "path": "",
}

# Mouth / jaw shape keys — owned by face pipeline only (never body Actions).
_MOUTH_SHAPE_PREFIXES = ("jaw", "mouth", "tongue")
_last_viseme_time = 0.0
_BODY_ACTION_ALIASES = {
    "standing": "idle",
    "idle": "idle",
    "walking": "walk",
    "walk": "walk",
    "walk_back": "walk_back",
    "run": "run",
    "sitting": "sit_idle",
    "sit": "sit_idle",
    "sit_idle": "sit_idle",
    "dancing": "celebrate",
    "wave": "wave",
    "shrug": "shrug",
    "talk_open": "talk_open",
    "talk_emphasize": "talk_emphasize",
    "point": "point",
    "point_forward": "point",
    "recoil": "recoil",
    "think": "think",
    "think_chin": "think",
    "nod_yes": "nod_yes",
    "nod": "nod_yes",
    "shake_no": "shake_no",
    "celebrate": "celebrate",
    "hands_reject": "hands_reject",
    "tense": "tense",
    "slump": "slump",
    "bow": "bow",
    "look_around": "look_around",
    "agree": "agree",
    "turn_left": "turn_left",
    "turn_right": "turn_right",
}

# For blink logic (Step B)
MEDIAPIPE_ACTIVE = False
_current_emotion = "neutral"
_last_mediapipe_time = 0.0
_blink_queue: "queue.Queue" = queue.Queue()
_blink_thread = None
_eye_gaze_thread = None
_head_thread = None

# For Step F and G
IS_SPEAKING = False
_current_head_pitch = 0.0
_current_head_yaw = 0.0
_current_head_roll = 0.0


# ---------------------------------------------------------------------------
# AUTO MESH DETECTION (kept from previous version)
# ---------------------------------------------------------------------------
_ARKIT_KEYS = {
    "eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "mouthSmileLeft", "mouthSmileRight",
    "browDownLeft", "browDownRight", "browInnerUp",
}

def _find_best_face_mesh():
    best = None
    best_score = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.data.shape_keys is None:
            continue
        key_names = {kb.name for kb in obj.data.shape_keys.key_blocks}
        score = len(_ARKIT_KEYS & key_names)
        if score > best_score:
            best_score = score
            best = obj
    return best if best_score >= 2 else None

def _get_target_object():
    global TARGET_MESH_NAME
    if TARGET_MESH_NAME:
        obj = bpy.data.objects.get(TARGET_MESH_NAME)
        if obj and obj.data.shape_keys:
            return obj
    obj = _find_best_face_mesh()
    if obj:
        TARGET_MESH_NAME = obj.name
        return obj
    return None

def _select_mesh(obj):
    if obj is None:
        return
    try:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    except Exception:
        pass

# ---------------------------------------------------------------------------
# LISTENER
# ---------------------------------------------------------------------------
def _listener_loop():
    global _sock
    try:
        if _sock:
            try:
                _sock.close()
            except Exception:
                pass
            _sock = None
        _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        _sock.bind((UDP_IP, UDP_PORT))
        _sock.settimeout(0.5)
        print(f"[face_receiver] UDP bound {UDP_IP}:{UDP_PORT}")
    except OSError as e:
        print(f"[face_receiver] UDP bind FAILED {UDP_IP}:{UDP_PORT}: {e}")
        return
    while _running:
        try:
            data, _ = _sock.recvfrom(65536)
            _data_queue.put(json.loads(data.decode("utf-8")))
        except socket.timeout:
            continue
        except OSError:
            break
    if _sock:
        try:
            _sock.close()
        except Exception:
            pass
        _sock = None


def _get_blink_interval():
    """Return random blink interval in seconds based on current emotion."""
    import random
    emotion = _current_emotion.lower()
    if emotion == "fearful":
        return random.uniform(1.5, 3.0)
    elif emotion == "angry":
        return random.uniform(2.0, 3.5)
    elif emotion == "thinking":
        return random.uniform(5.0, 9.0)
    elif emotion == "sad":
        return random.uniform(4.0, 8.0)
    elif emotion == "happy":
        return random.uniform(3.5, 7.0)
    else:  # neutral or others
        return random.uniform(3.0, 6.0)


def _blink_loop():
    """Background blink thread. Ramps eyeBlink keys directly."""
    import random
    import time as _time
    while _running:
        interval = _get_blink_interval()
        _time.sleep(interval)

        if not _running:
            break
        if MEDIAPIPE_ACTIVE and (time.time() - _last_mediapipe_time < 2.0):
            continue  # pause blinks if live capture was active recently

        # Ramp up
        for i in range(5):  # 80ms / 20ms = 4 steps, +1 for 0
            val = (i + 1) / 5.0   # 0.2, 0.4, 0.6, 0.8, 1.0
            _blink_queue.put({"eyeBlinkLeft": val, "eyeBlinkRight": val})
            _time.sleep(0.02)

        _time.sleep(0.04)  # hold at 1.0 for 40ms

        # Ramp down
        for i in range(6):  # 120ms / 20ms = 6 steps
            val = 1.0 - (i / 6.0)   # 1.0 -> 0.0
            _blink_queue.put({"eyeBlinkLeft": val, "eyeBlinkRight": val})
            _time.sleep(0.02)


def eye_gaze_loop():
    """Background thread for eye gaze and micro saccades (Step F). Writes to TARGET_VALUES for gliding."""
    import random
    import time as _time
    gaze_targets = {
        "center": (0.0, 0.0),
        "left": (-0.12, 0.0),
        "right": (0.12, 0.0),
        "up": (0.0, 0.12),
        "down": (0.0, -0.08),
    }
    current_x, current_y = 0.0, 0.0
    last_change = _time.time()
    last_saccade = _time.time()

    while _running:
        now = _time.time()
        is_speaking = IS_SPEAKING

        if is_speaking:
            max_val = 0.20
            change_interval = random.uniform(1.0, 2.5)
        else:
            max_val = 0.15
            change_interval = random.uniform(2.0, 5.0)

        # Saccades - rapid micro flicks
        if now - last_saccade > random.uniform(1.5, 3.0):
            sacc_x = random.uniform(-0.1, 0.1) * max_val
            sacc_y = random.uniform(-0.08, 0.08) * max_val
            # Quick move
            for _ in range(2):
                current_x = current_x * 0.6 + sacc_x * 0.4
                current_y = current_y * 0.6 + sacc_y * 0.4
                # Update TARGET_VALUES (will be glided in main timer)
                TARGET_VALUES["eyeLookInLeft"] = max(0.0, -current_x) if current_x < 0 else 0.0
                TARGET_VALUES["eyeLookOutLeft"] = max(0.0, current_x) if current_x > 0 else 0.0
                TARGET_VALUES["eyeLookUpLeft"] = max(0.0, current_y) if current_y > 0 else 0.0
                TARGET_VALUES["eyeLookDownLeft"] = max(0.0, -current_y) if current_y < 0 else 0.0
                TARGET_VALUES["eyeLookInRight"] = max(0.0, -current_x) if current_x < 0 else 0.0
                TARGET_VALUES["eyeLookOutRight"] = max(0.0, current_x) if current_x > 0 else 0.0
                TARGET_VALUES["eyeLookUpRight"] = max(0.0, current_y) if current_y > 0 else 0.0
                TARGET_VALUES["eyeLookDownRight"] = max(0.0, -current_y) if current_y < 0 else 0.0
                _tracked_keys.update([
                    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
                    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight"
                ])
                _time.sleep(0.02)
            # Return to previous
            for _ in range(3):
                current_x *= 0.7
                current_y *= 0.7
                TARGET_VALUES["eyeLookInLeft"] = max(0.0, -current_x) if current_x < 0 else 0.0
                TARGET_VALUES["eyeLookOutLeft"] = max(0.0, current_x) if current_x > 0 else 0.0
                TARGET_VALUES["eyeLookUpLeft"] = max(0.0, current_y) if current_y > 0 else 0.0
                TARGET_VALUES["eyeLookDownLeft"] = max(0.0, -current_y) if current_y < 0 else 0.0
                TARGET_VALUES["eyeLookInRight"] = max(0.0, -current_x) if current_x < 0 else 0.0
                TARGET_VALUES["eyeLookOutRight"] = max(0.0, current_x) if current_x > 0 else 0.0
                TARGET_VALUES["eyeLookUpRight"] = max(0.0, current_y) if current_y > 0 else 0.0
                TARGET_VALUES["eyeLookDownRight"] = max(0.0, -current_y) if current_y < 0 else 0.0
                _tracked_keys.update([
                    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
                    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight"
                ])
                _time.sleep(0.02)
            last_saccade = now

        # Main gaze drift
        if now - last_change > change_interval:
            target_name = random.choice(list(gaze_targets.keys()))
            target_x, target_y = gaze_targets[target_name]
            target_x *= max_val / 0.15
            target_y *= max_val / 0.15

            steps = random.randint(10, 20)
            step_time = random.uniform(0.2, 0.4) / steps
            for s in range(steps):
                frac = (s + 1) / steps
                current_x = current_x * (1 - frac) + target_x * frac
                current_y = current_y * (1 - frac) + target_y * frac
                TARGET_VALUES["eyeLookInLeft"] = max(0.0, -current_x) if current_x < 0 else 0.0
                TARGET_VALUES["eyeLookOutLeft"] = max(0.0, current_x) if current_x > 0 else 0.0
                TARGET_VALUES["eyeLookUpLeft"] = max(0.0, current_y) if current_y > 0 else 0.0
                TARGET_VALUES["eyeLookDownLeft"] = max(0.0, -current_y) if current_y < 0 else 0.0
                TARGET_VALUES["eyeLookInRight"] = max(0.0, -current_x) if current_x < 0 else 0.0
                TARGET_VALUES["eyeLookOutRight"] = max(0.0, current_x) if current_x > 0 else 0.0
                TARGET_VALUES["eyeLookUpRight"] = max(0.0, current_y) if current_y > 0 else 0.0
                TARGET_VALUES["eyeLookDownRight"] = max(0.0, -current_y) if current_y < 0 else 0.0
                _tracked_keys.update([
                    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
                    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight"
                ])
                _time.sleep(step_time)
            last_change = now

        if int(now) % 5 == 0:
            print(f"[eye_gaze_loop] current_x={current_x:.3f} current_y={current_y:.3f} speaking={is_speaking}")

        _time.sleep(0.1)


def head_movement_loop():
    """Background thread for subtle head movement (Step G). Uses blendshapes for tilt simulation + sends 'head' packet for rotation."""
    import random
    import time as _time
    import math

    last_nod = 0.0
    last_drift = 0.0
    current_pitch = 0.0
    current_yaw = 0.0
    current_roll = 0.0
    prev_speaking = False

    while _running:
        now = _time.time()
        is_speaking = IS_SPEAKING
        emotion = _current_emotion.lower()

        # Breathing always (subtle)
        breath = math.sin(now * 0.8) * 0.02
        target_pitch = breath
        target_yaw = current_yaw
        target_roll = current_roll

        if is_speaking:
            # Nod on sentence start
            if not prev_speaking:
                target_pitch += 0.12
                last_nod = now
            prev_speaking = True
            # Drift
            if now - last_drift > random.uniform(1.0, 2.5):
                target_yaw = random.uniform(-0.08, 0.08)
                last_drift = now
            # Emotion specific
            if emotion == "surprised":
                target_pitch -= 0.10  # pull back
            elif emotion == "angry":
                target_pitch += 0.08  # lean in
            elif emotion == "thinking":
                target_roll = 0.07  # tilt
            elif emotion == "sad":
                target_pitch += 0.10  # drop
        else:
            prev_speaking = False
            # Idle slow drift
            if now - last_drift > random.uniform(4.0, 8.0):
                target_pitch = random.uniform(-0.08, 0.08) + breath
                target_yaw = random.uniform(-0.10, 0.10)
                target_roll = random.uniform(-0.05, 0.05)
                last_drift = now

        # Low-pass filter (weight)
        current_pitch = current_pitch * 0.85 + target_pitch * 0.15
        current_yaw = current_yaw * 0.85 + target_yaw * 0.15
        current_roll = current_roll * 0.85 + target_roll * 0.15

        # Update for blendshape tilt simulation using brow asymmetry only.
        # Skip when orchestrator is actively sending emotion packets — those own browDown.
        # Without this guard, head_movement_loop resets browDown every 50ms and kills
        # the angry (0.95) / sad (0.30) expressions sent by the orchestrator.
        _orch_active = _last_packet_time is not None and (now - _last_packet_time < 3.0)
        if not _orch_active:
            TARGET_VALUES["browDownLeft"] = max(0.0, min(1.0, current_pitch * 0.3 if current_pitch > 0 else 0))
            TARGET_VALUES["browDownRight"] = max(0.0, min(1.0, current_pitch * 0.3 if current_pitch > 0 else 0))
            _tracked_keys.update(["browDownLeft", "browDownRight"])

        # Store target rotation for the main-thread modal() to apply.
        # Direct rotation_euler writes from a background thread crash Blender via
        # WM_msg_publish_rna (RNA message bus is not thread-safe).
        _HEAD_ROTATION["pitch"] = current_pitch * 0.15  # X
        _HEAD_ROTATION["roll"]  = current_roll  * 0.10  # Y
        _HEAD_ROTATION["yaw"]   = current_yaw   * 0.18  # Z
        if int(now) % 5 == 0:  # debug every ~5s
            print(f"[head_loop] breathing pitch={current_pitch:.3f} yaw={current_yaw:.3f} roll={current_roll:.3f} speaking={is_speaking}")

        _time.sleep(0.05)  # ~20fps for head (slow movement)


# ---------------------------------------------------------------------------
# PACKET HANDLING
# ---------------------------------------------------------------------------
def _resolve_shape_key_name(mp_name, key_blocks):
    if mp_name in NAME_OVERRIDES:
        override = NAME_OVERRIDES[mp_name]
        return override if override in key_blocks else None
    if mp_name in key_blocks:
        return mp_name
    return None

def _apply_blendshapes(obj, blendshapes_dict, allowed_keys=None, smoothing=None):
    """Update TARGET_VALUES from incoming packet (don't apply to mesh instantly).
    Use the provided smoothing to blend into target.
    If smoothing is None, uses EMOTION_SMOOTHING.
    """
    if obj is None or obj.data.shape_keys is None:
        return

    if smoothing is None:
        smoothing = EMOTION_SMOOTHING

    key_blocks = obj.data.shape_keys.key_blocks

    for mp_name, raw_value in blendshapes_dict.items():
        if allowed_keys is not None and mp_name not in allowed_keys:
            continue

        key_name = _resolve_shape_key_name(mp_name, key_blocks)
        if key_name is None:
            continue

        old_target = TARGET_VALUES.get(key_name, 0.0)
        new_target = old_target * (1.0 - smoothing) + raw_value * smoothing
        TARGET_VALUES[key_name] = new_target
        _tracked_keys.add(key_name)

def _clear_shape_key_animation(sk) -> None:
    """Remove previous shape-key Action so a new sentence bake is clean."""
    if sk is None:
        return
    try:
        if sk.animation_data and sk.animation_data.action:
            act = sk.animation_data.action
            sk.animation_data.action = None
            # Only remove if unused
            if act.users == 0:
                try:
                    bpy.data.actions.remove(act)
                except Exception:
                    pass
            sk.animation_data_clear()
    except Exception:
        pass


def _keyframe_shape_dict(obj, blendshapes: dict, frame: int, key_blocks) -> int:
    """Insert shape-key value keyframes for one frame. Returns count written."""
    n = 0
    if obj is None or not blendshapes or key_blocks is None:
        return 0
    sk = obj.data.shape_keys
    if sk is None:
        return 0
    for mp_name, raw in blendshapes.items():
        key_name = _resolve_shape_key_name(mp_name, key_blocks)
        if key_name is None or key_name not in key_blocks:
            continue
        try:
            kb = key_blocks[key_name]
            kb.value = float(raw)
            kb.keyframe_insert(data_path="value", frame=int(frame))
            n += 1
        except Exception:
            continue
    return n


def _apply_face_keyframes_packet(packet: dict) -> None:
    """
    Bake face blendshapes (+ head) onto the Blender timeline for scrub/replay.
    Prefer packet['path'] JSON (full track); optional inline keyframes for short clips.
    """
    global _FACE_TIMELINE
    from pathlib import Path
    import json as _json

    track = None
    path = str(packet.get("path") or "").strip()
    if path:
        p = Path(path)
        if not p.is_file():
            print(f"[face_receiver] face_keyframes missing file: {path}")
            return
        try:
            track = _json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[face_receiver] face_keyframes load failed: {e}")
            return
    else:
        track = {
            "fps": packet.get("fps", 30),
            "frame_start": packet.get("frame_start", 1),
            "frame_end": packet.get("frame_end", 1),
            "keyframes": packet.get("keyframes") or [],
        }

    kfs = track.get("keyframes") or []
    if not kfs:
        print("[face_receiver] face_keyframes empty — skip")
        return

    obj = _get_target_object()
    if obj is None or obj.data is None or obj.data.shape_keys is None:
        print("[face_receiver] face_keyframes: no target mesh with shape keys")
        return

    key_blocks = obj.data.shape_keys.key_blocks
    sk = obj.data.shape_keys
    clear_prev = bool(packet.get("clear_previous", True))
    if clear_prev:
        _clear_shape_key_animation(sk)
        # Secondary meshes
        for sec_name in ("eyeLeft_ORIGINAL", "eyeRight_ORIGINAL", "teeth_ORIGINAL"):
            sec = bpy.data.objects.get(sec_name)
            if sec and sec.data and sec.data.shape_keys:
                _clear_shape_key_animation(sec.data.shape_keys)

    f0 = int(track.get("frame_start") or kfs[0].get("frame") or 1)
    f1 = int(track.get("frame_end") or kfs[-1].get("frame") or f0)
    n_keys = 0
    n_frames = 0

    # Head controller (optional)
    head_obj = bpy.data.objects.get("HEAD_CONTROLLER")
    if head_obj and clear_prev and head_obj.animation_data:
        try:
            head_obj.animation_data_clear()
        except Exception:
            pass

    for kf in kfs:
        fr = int(kf.get("frame") or f0)
        bs = kf.get("blendshapes") or {}
        n_keys += _keyframe_shape_dict(obj, bs, fr, key_blocks)
        # Mirror to secondary meshes when same key names exist
        for sec_name in ("eyeLeft_ORIGINAL", "eyeRight_ORIGINAL", "teeth_ORIGINAL"):
            sec = bpy.data.objects.get(sec_name)
            if sec is None or sec.data is None or sec.data.shape_keys is None:
                continue
            _keyframe_shape_dict(sec, bs, fr, sec.data.shape_keys.key_blocks)
        # Head euler (same scale as live head packets)
        hd = kf.get("head") or {}
        if head_obj is not None and hd:
            try:
                head_obj.rotation_euler[0] = float(hd.get("pitch", 0.0)) * 0.15
                head_obj.rotation_euler[1] = float(hd.get("roll", 0.0)) * 0.10
                head_obj.rotation_euler[2] = float(hd.get("yaw", 0.0)) * 0.18
                head_obj.keyframe_insert(data_path="rotation_euler", frame=fr)
            except Exception:
                pass
        n_frames += 1

    # Smooth interpolation on shape-key fcurves
    try:
        if sk.animation_data and sk.animation_data.action:
            for fc in _iter_action_fcurves(sk.animation_data.action):
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
    except Exception:
        pass

    if packet.get("set_frame_range", True):
        try:
            sc = bpy.context.scene
            # Join multi-shot film: expand range, do NOT reset start to this shot only
            if clear_prev:
                sc.frame_start = f0
                sc.frame_end = f1
                sc.frame_set(f0)
            else:
                sc.frame_start = min(int(sc.frame_start or f0), f0)
                sc.frame_end = max(int(sc.frame_end or f1), f1)
        except Exception:
            pass

    prev_f0 = int(_FACE_TIMELINE.get("frame_start") or f0)
    prev_f1 = int(_FACE_TIMELINE.get("frame_end") or f1)
    _FACE_TIMELINE.update({
        "active": True,
        "frame_start": min(prev_f0, f0) if not clear_prev else f0,
        "frame_end": max(prev_f1, f1),
        "path": path or "",
    })
    print(
        f"[face_receiver] FACE TIMELINE keyframed {n_frames} frames "
        f"f{f0}-{f1} on {obj.name} (~{n_keys} key inserts) "
        f"clear={clear_prev} master=f{_FACE_TIMELINE['frame_start']}-{_FACE_TIMELINE['frame_end']} "
        f"— scrub to replay"
    )


def _handle_packet(packet):
    """Route packet to the correct face region based on type."""
    if not isinstance(packet, dict):
        return

    ptype = packet.get("type")

    # Camera / body / look do not need face mesh — handle before shape-key gate
    if ptype == "camera":
        _queue_camera_packet(packet)
        return
    if ptype == "body":
        _queue_body_packet(packet)
        return
    if ptype == "look":
        try:
            _apply_look_packet(packet)
        except Exception as e:
            print(f"[look] apply failed: {e}")
        return
    if ptype in ("bpy", "bpy_exec", "scene_code"):
        try:
            _apply_bpy_packet(packet)
        except Exception as e:
            print(f"[bpy] exec failed: {e}")
        return
    if ptype in ("inspect", "session_inspect", "scene_inspect"):
        try:
            _apply_inspect_packet(packet)
        except Exception as e:
            print(f"[inspect] failed: {e}")
        return
    if ptype in ("light", "lights"):
        try:
            _apply_light_packet(packet)
        except Exception as e:
            print(f"[light] failed: {e}")
        return
    if ptype in ("material", "materials"):
        try:
            _apply_material_packet(packet)
        except Exception as e:
            print(f"[material] failed: {e}")
        return
    if ptype == "wardrobe":
        try:
            _apply_wardrobe_packet(packet)
        except Exception as e:
            print(f"[wardrobe] apply failed: {e}")
        return
    if ptype == "cast":
        try:
            _apply_cast_packet(packet)
        except Exception as e:
            print(f"[cast] apply failed: {e}")
        return

    # Face timeline bake (shape-key keyframes for scrub) — may load JSON path
    if ptype in ("face_keyframes", "face_timeline"):
        _apply_face_keyframes_packet(packet)
        return

    blendshapes = packet.get("blendshapes", packet)  # support legacy flat or wrapped

    obj = _get_target_object()
    if obj is None:
        return
    if getattr(obj, 'data', None) is None or getattr(obj.data, 'shape_keys', None) is None:
        return
    key_blocks = obj.data.shape_keys.key_blocks

    if VERBOSE_PACKETS:
        print(f"[face_receiver] Received packet type={ptype}")

    global IS_SPEAKING

    if ptype == "emotion":
        # Upper face only (brows, eyes, cheeks)
        global _current_emotion
        _current_emotion = packet.get("emotion", _current_emotion)
        if VERBOSE_PACKETS:
            print(f"[face_receiver] EMOTION keys={list(blendshapes.keys())} ({_current_emotion})")
        _apply_blendshapes(obj, blendshapes, allowed_keys=UPPER_FACE_KEYS, smoothing=EMOTION_SMOOTHING)

    elif ptype == "viseme":
        # Mouth / jaw — set targets directly so lips stay in sync with audio
        # (no lag from heavy smoothing; mesh still uses light VISEME_GLIDE)
        global _last_viseme_time
        _last_viseme_time = time.time()
        IS_SPEAKING = True
        lower_only = {k: v for k, v in blendshapes.items() if k not in UPPER_FACE_KEYS}
        if not lower_only:
            return
        for mp_name, raw_value in lower_only.items():
            key_name = _resolve_shape_key_name(mp_name, key_blocks)
            if key_name is None:
                continue
            v = float(raw_value)
            TARGET_VALUES[key_name] = v
            _tracked_keys.add(key_name)
            # Seed smoothed value close to target so first frames aren't stuck at 0
            prev = _smoothed_values.get(key_name, v)
            _smoothed_values[key_name] = prev * (1.0 - VISEME_SMOOTHING) + v * VISEME_SMOOTHING

    elif ptype == "rest_pose":
        # Full-face rest. smooth=True (end of sentence) only updates targets so the
        # existing glide system eases to neutral — no sudden jerk. smooth=False
        # (startup / force) snaps immediately.
        IS_SPEAKING = False
        smooth = bool(packet.get("smooth", False))
        if smooth:
            print(f"[face_receiver] Applying REST_POSE (smooth) — gliding face to neutral rest")
            for k, v in blendshapes.items():
                key_name = _resolve_shape_key_name(k, key_blocks)
                if key_name:
                    TARGET_VALUES[key_name] = v
                    _tracked_keys.add(key_name)
            print("[face_receiver] Neutral rest targets set; glide will settle face smoothly")
        else:
            print(f"[face_receiver] Applying REST_POSE (instant) — forcing full face to neutral rest")
            for k, v in blendshapes.items():
                key_name = _resolve_shape_key_name(k, key_blocks)
                if key_name:
                    TARGET_VALUES[key_name] = v
                    _smoothed_values[key_name] = v
                    _tracked_keys.add(key_name)
                    if key_name in key_blocks:
                        key_blocks[key_name].value = v
            print("[face_receiver] Face forced to NEUTRAL_REST (jaw slightly open, tiny eye squint, etc.)")

    elif ptype == "head":
        # Head rotation packet — store for main-thread application only.
        pitch = packet.get("pitch", 0.0)
        yaw = packet.get("yaw", 0.0)
        roll = packet.get("roll", 0.0)
        _HEAD_ROTATION["pitch"] = pitch * 0.15
        _HEAD_ROTATION["roll"]  = roll  * 0.10
        _HEAD_ROTATION["yaw"]   = yaw   * 0.18
        # cheekSquint NOT set here — emotion packet owns it via cheeks_agent.
        # Setting it from roll was overriding the happy/angry cheek expression.

    else:
        # Legacy / live MediaPipe capture — apply everything (full face)
        global MEDIAPIPE_ACTIVE, _last_mediapipe_time
        _last_mediapipe_time = time.time()
        MEDIAPIPE_ACTIVE = True
        _apply_blendshapes(obj, blendshapes, allowed_keys=None)

# ---------------------------------------------------------------------------
# BODY ACTION PLAYBACK (SMPL-X + retargeted Mixamo Actions)
# ---------------------------------------------------------------------------
# Cache failed library loads so we don't reopen huge .blend every body packet
_LIBRARY_LOAD_TRIED: set = set()
# Catalog actions (idle, talk_open, walk, …) live in these blends — not always in the open scene
_CATALOG_BLEND_REL = (
    "whole_body_retargeted.blend",
    "body_motion/whole_body_with_clips.blend",
    "body_motion/whole_body_retargeted.blend",
)
# Parsed fcurve cache: action_name → list[(bone, prop, index, fcurve)]
_ACTION_FCU_CACHE: dict = {}
# Fast live body: avoid scene.frame_set (main cause of live stutter vs scrub)
_BODY_USE_FAST_EVAL = True


def _project_roots():
    from pathlib import Path
    roots = []
    try:
        if bpy.data.filepath:
            fp = Path(bpy.data.filepath).resolve()
            roots.append(fp.parent)
            roots.append(fp.parent.parent)
    except Exception:
        pass
    # Common workspace (and env override)
    import os
    env = os.environ.get("PROJFACE_ROOT", "").strip()
    if env:
        roots.append(Path(env))
    roots.append(Path(r"C:\me\proj\projface_v1"))
    # de-dupe
    out = []
    seen = set()
    for r in roots:
        try:
            k = str(r.resolve())
        except Exception:
            k = str(r)
        if k not in seen:
            seen.add(k)
            out.append(Path(k))
    return out


def _catalog_blend_paths():
    from pathlib import Path
    paths = []
    for root in _project_roots():
        for rel in _CATALOG_BLEND_REL:
            p = root / rel
            if p.is_file():
                paths.append(p)
    return paths


def _try_load_action_from_library(action_name: str, library_blend: str) -> str | None:
    """Append Action from a momask/cache/catalog blend if not already in this file (once)."""
    if not action_name:
        return None
    from pathlib import Path
    if bpy.data.actions.get(action_name):
        return action_name
    # Build candidate blend list: explicit packet path, per-action cache, catalog
    candidates = []
    if library_blend:
        p = Path(library_blend)
        if p.is_file():
            candidates.append(p)
    want = str(action_name).strip()
    want_l = want.lower()
    # Per-clip slim Action cache (always try — orchestrator may omit library_blend)
    for root in _project_roots():
        for rel in (
            Path("body_motion") / "momask_cache" / f"{want}.blend",
            Path("body_motion") / "momask_actions_library.blend",
            Path("body_motion") / "momask_cache" / "momask_actions_library.blend",
        ):
            p = root / rel
            if p.is_file():
                candidates.append(p)
    candidates.extend(_catalog_blend_paths())
    # de-dupe preserving order
    _seen_p = set()
    _uniq = []
    for p in candidates:
        try:
            k = str(p.resolve())
        except Exception:
            k = str(p)
        if k not in _seen_p:
            _seen_p.add(k)
            _uniq.append(p)
    candidates = _uniq
    for p in candidates:
        key = (want, str(p.resolve()) if p.exists() else str(p))
        if key in _LIBRARY_LOAD_TRIED:
            continue
        _LIBRARY_LOAD_TRIED.add(key)
        try:
            with bpy.data.libraries.load(str(p), link=False) as (data_from, data_to):
                src_acts = list(data_from.actions or [])
                names = [n for n in src_acts if n == want or n.lower() == want_l]
                if not names:
                    names = [n for n in src_acts if want_l in n.lower() or n.lower() in want_l]
                if not names and want_l.startswith("momask_"):
                    names = [n for n in src_acts if n.startswith("momask_")]
                data_to.actions = names[:1] if names else []
            # Find what was appended
            for a in list(bpy.data.actions):
                if a.name == want or a.name.lower() == want_l:
                    a.use_fake_user = True
                    kr = _action_key_range(a)
                    if kr:
                        _expand_layered_action_range(a, kr[0], kr[1])
                    print(
                        f"[body_receiver] loaded Action {a.name!r} from {p.name} "
                        f"keys={kr or 'unknown'}"
                    )
                    return a.name
                if names and a.name == names[0]:
                    a.use_fake_user = True
                    # Rename to expected catalog name when safe
                    try:
                        if not bpy.data.actions.get(want) and a.name != want:
                            a.name = want
                    except Exception:
                        pass
                    kr = _action_key_range(a)
                    if kr:
                        _expand_layered_action_range(a, kr[0], kr[1])
                    print(
                        f"[body_receiver] loaded Action {a.name!r} from {p.name} "
                        f"keys={kr or 'unknown'}"
                    )
                    return a.name
        except Exception as e:
            print(f"[body_receiver] library load failed ({p.name}): {e}")
    return None


def _resolve_body_action_name(raw: str, library_blend: str = "") -> str | None:
    if not raw:
        return None
    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    key = _BODY_ACTION_ALIASES.get(key, key)
    # exact Action data-block
    if bpy.data.actions.get(key):
        return key
    # case-insensitive scan
    for act in bpy.data.actions:
        if act.name.lower() == key:
            return act.name
    # MoMask cache / catalog library blend (idle lives in whole_body_retargeted.blend)
    loaded = _try_load_action_from_library(key, library_blend)
    if loaded:
        return loaded
    # try exact raw name (hash actions are momask_<hex>)
    raw_s = str(raw).strip()
    if bpy.data.actions.get(raw_s):
        return raw_s
    loaded = _try_load_action_from_library(raw_s, library_blend)
    if loaded:
        return loaded
    return None


def _iter_action_fcurves(action):
    """Yield fcurves from classic or layered (Blender 4.4+/5) Actions."""
    if action is None:
        return
    n_yielded = 0
    # Classic (Blender ≤4.1)
    try:
        fcus = getattr(action, "fcurves", None)
        if fcus is not None:
            for fcu in fcus:
                n_yielded += 1
                yield fcu
            if n_yielded:
                return
    except Exception:
        pass
    # Layered (Blender 4.4 / 5.x) — channelbags hold fcurves
    try:
        layers = getattr(action, "layers", None)
        if not layers:
            return
        for layer in layers:
            for strip in getattr(layer, "strips", []) or []:
                bags = None
                try:
                    bags = strip.channelbags
                except Exception:
                    bags = None
                if bags is None or len(bags) == 0:
                    # try slot-bound channelbag()
                    try:
                        slots = getattr(action, "slots", None)
                        if slots and len(slots) > 0 and hasattr(strip, "channelbag"):
                            bag = strip.channelbag(slots[0])
                            if bag is not None:
                                bags = [bag]
                    except Exception:
                        bags = None
                if not bags:
                    continue
                for bag in bags:
                    try:
                        for fcu in bag.fcurves:
                            n_yielded += 1
                            yield fcu
                    except Exception:
                        continue
    except Exception:
        return


def _get_action_fcu_targets(action):
    """Cache parsed bone targets for fast live eval (classic + layered Actions)."""
    if action is None:
        return []
    name = action.name
    cached = _ACTION_FCU_CACHE.get(name)
    if cached is not None:
        return cached
    import re
    pat = re.compile(
        r'^pose\.bones\["([^"]+)"\]\.(location|rotation_quaternion|rotation_euler|scale)$'
    )
    out = []
    for fcu in _iter_action_fcurves(action):
        m = pat.match(fcu.data_path or "")
        if not m:
            continue
        out.append((m.group(1), m.group(2), int(fcu.array_index), fcu))
    # Never cache empty — layered Actions can probe empty once then stay broken
    if out:
        _ACTION_FCU_CACHE[name] = out
        print(f"[body_receiver] fast-eval cache {name!r}: {len(out)} fcurves")
    else:
        print(f"[body_receiver] WARN fast-eval found 0 fcurves on {name!r} (layered?)")
    return out


def _apply_action_frame_fast(arm, action, frame: float) -> bool:
    """
    Apply Action pose at `frame` without scene.frame_set.
    Sets _BODY_PLAY['_wrote_pelvis_loc'] so root continuity can place absolutely.
    """
    global _BODY_PLAY
    if arm is None or action is None:
        return False
    targets = _get_action_fcu_targets(action)
    if not targets:
        _BODY_PLAY["_wrote_pelvis_loc"] = False
        return False
    # Group by bone+prop so we can set rotation mode once
    touched_quat = set()
    touched_eul = set()
    wrote_pelvis_loc = False
    for bname, prop, idx, fcu in targets:
        pb = arm.pose.bones.get(bname)
        if pb is None:
            continue
        try:
            val = float(fcu.evaluate(float(frame)))
        except Exception:
            continue
        if prop == "location":
            if 0 <= idx < 3:
                pb.location[idx] = val
                if bname == "pelvis":
                    wrote_pelvis_loc = True
        elif prop == "rotation_quaternion":
            if bname not in touched_quat:
                pb.rotation_mode = "QUATERNION"
                touched_quat.add(bname)
            if 0 <= idx < 4:
                pb.rotation_quaternion[idx] = val
        elif prop == "rotation_euler":
            if bname not in touched_eul:
                pb.rotation_mode = "XYZ"
                touched_eul.add(bname)
            if 0 <= idx < 3:
                pb.rotation_euler[idx] = val
        elif prop == "scale":
            if 0 <= idx < 3:
                pb.scale[idx] = val
    _BODY_PLAY["_wrote_pelvis_loc"] = bool(wrote_pelvis_loc)
    return True


# Safety only — real MoMask clips are typically 32–196 frames @ 20fps.
MAX_SESSION_CLIP_FRAMES = 400


def _expand_layered_action_range(act, f0: int, f1: int) -> None:
    """Blender 5 layered Actions default to a ~25-frame strip; expand or keys drop."""
    if act is None:
        return
    f0, f1 = int(f0), int(max(int(f0) + 1, int(f1)))
    try:
        layers = getattr(act, "layers", None)
        if not layers:
            return
        for layer in layers:
            for strip in getattr(layer, "strips", []) or []:
                try:
                    cur0 = int(getattr(strip, "frame_start", f0) or f0)
                    cur1 = int(getattr(strip, "frame_end", f1) or f1)
                    strip.frame_start = float(min(cur0, f0))
                    strip.frame_end = float(max(cur1, f1 + 1))
                except Exception:
                    try:
                        strip.frame_end = float(f1 + 1)
                    except Exception:
                        pass
    except Exception:
        pass


def _action_key_range(act) -> tuple | None:
    """True first/last keyframe from classic or layered fcurves. None if empty."""
    if act is None:
        return None
    lo, hi = None, None
    for fcu in _iter_action_fcurves(act):
        try:
            kps = fcu.keyframe_points
            if not kps:
                continue
            a = float(kps[0].co[0])
            b = float(kps[-1].co[0])
            lo = a if lo is None else min(lo, a)
            hi = b if hi is None else max(hi, b)
        except Exception:
            continue
    if lo is None:
        return None
    return int(round(lo)), int(round(hi))


def _action_frame_range(act) -> tuple:
    """Prefer actual keys. Blender 5 Action.frame_range often reports ~25 on layered clips."""
    keys = _action_key_range(act)
    if keys is not None:
        f0, f1 = keys
        if f1 <= f0:
            f1 = f0 + 1
        return f0, f1
    try:
        fr = act.frame_range
        f0, f1 = int(fr[0]), int(fr[1])
        if f1 <= f0:
            f1 = f0 + 1
        return f0, f1
    except Exception:
        return 1, 60


def _clip_span_for_append(act, *, action_frames: int = 0, motion_length: int = 0) -> tuple:
    """
    How many frames to place on the session timeline for this clip.

    Pipeline action_frames wins when Blender reports a bogus ~25-frame range
    (layered-action strip default). Never shrink a 96-frame MoMask clip to 25.
    Returns (src_f0, span, reason).
    """
    src_f0, src_f1 = _action_frame_range(act) if act else (1, 60)
    natural = max(2, int(src_f1) - int(src_f0) + 1)
    try:
        af = int(action_frames or 0)
        ml = int(motion_length or 0)
    except (TypeError, ValueError):
        af, ml = 0, 0

    bogus_short = natural <= 30 and af > natural + 4

    if af > 1:
        # Trust pipeline take length. Do NOT shrink to native Action length —
        # movie Session shots must span full duration (hold last pose after clip ends).
        span = af
        reason = f"action_frames={af}"
        if natural < af:
            src_f0 = int(src_f0) if natural > 30 else 1
            reason += f"|fill_take_native={natural}"
        else:
            reason += f"|keys={natural}"
    elif ml > 0:
        span = ml
        reason = f"motion_length={ml}"
        if not bogus_short:
            span = min(max(span, natural), max(natural, ml))
    else:
        span = natural
        reason = f"action_keys={natural}"

    if span > MAX_SESSION_CLIP_FRAMES:
        reason += f"|capped_{span}→{MAX_SESSION_CLIP_FRAMES}"
        span = MAX_SESSION_CLIP_FRAMES
    span = max(2, int(span))
    return int(src_f0), span, reason


def _is_loop_action(name: str, packet: dict) -> bool:
    """
    Whether live body should *cycle* the Action while speech/body duration runs.

    MoMask one-shots must NEVER loop (wrap = teleport back to start).
    Locomotion catalog (walk/run/jump) also must NOT loop — root motion
    wrapping is the “ran forward then snapped” bug. Idle/talk may loop.
    """
    eng = str(packet.get("engine") or "").lower()
    n = (name or "").strip().lower()
    if eng == "momask" or n.startswith("momask_"):
        return False
    if (name or "") in (SESSION_ACTION_NAME, "Session_Timeline") or n.startswith("session_"):
        return False
    # Root-traveling clips: looping wraps pelvis XY → visible teleport
    if __import__("re").search(r"(walk|run|jump|jog|sprint|leap|hop)", n):
        return False
    if "loop" in packet:
        return bool(packet.get("loop"))
    return bool(
        __import__("re").search(
            r"^(idle|talk|sit|tense|slump)",
            n,
        )
    )


def _session_cursor_load() -> int:
    """Cursor must live on the Scene so script reloads don't reset to 1 (overwrite bug)."""
    global _SESSION_BODY
    try:
        sc = bpy.context.scene
        if "session_body_cursor" in sc:
            cur = max(1, int(sc["session_body_cursor"]))
            _SESSION_BODY["frame_cursor"] = cur
            return cur
    except Exception:
        pass
    return max(1, int(_SESSION_BODY.get("frame_cursor") or 1))


def _session_cursor_save(cur: int) -> None:
    global _SESSION_BODY
    cur = max(1, int(cur))
    _SESSION_BODY["frame_cursor"] = cur
    try:
        bpy.context.scene["session_body_cursor"] = cur
        # Must persist the real Session_<id> name — hardcoding Session_Timeline
        # made live play look up the wrong Action after cursor saves / reload.
        bpy.context.scene["session_body_action"] = _session_action_name()
    except Exception:
        pass


def _clip_label_short(label: str, *, max_len: int = 36) -> str:
    """Compact label safe for Blender timeline markers."""
    s = " ".join(str(label or "").replace("\n", " ").split()).strip()
    if not s:
        s = "clip"
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def _make_clip_display_label(
    *,
    index: int,
    source_action: str = "",
    clip_label: str = "",
    prompt: str = "",
    text: str = "",
    frame_start: int = 1,
    frame_end: int = 1,
) -> str:
    """
    Human-readable clip id for timeline markers / logs.
    Example: #2 wave hand  [115-230]
    """
    base = (
        str(clip_label or "").strip()
        or str(prompt or "").strip()
        or str(text or "").strip()
        or str(source_action or "").strip()
        or f"clip{index}"
    )
    # Drop momask_ prefix noise for display
    if base.lower().startswith("momask_"):
        base = base[7:]
    base = _clip_label_short(base, max_len=28)
    return f"#{int(index)} {base}  [{int(frame_start)}-{int(frame_end)}]"


def _clear_session_timeline_markers() -> None:
    """Remove prior session clip markers (names start with C# or Session|)."""
    try:
        sc = bpy.context.scene
        for m in list(sc.timeline_markers):
            n = str(getattr(m, "name", "") or "")
            if (
                n.startswith("C#")
                or n.startswith("S#")
                or n.startswith("Session|")
                or n.startswith("#")
            ):
                try:
                    sc.timeline_markers.remove(m)
                except Exception:
                    pass
    except Exception:
        pass


def _persist_session_clip_table() -> None:
    """Store clip range table on the Scene + optional JSON for external UI."""
    global _SESSION_BODY
    clips = list(_SESSION_BODY.get("clips") or [])
    try:
        import json as _json
        sc = bpy.context.scene
        sc["session_body_clips"] = len(clips)
        sc["session_body_action"] = _session_action_name()
        sc["session_body_clip_labels"] = _json.dumps(clips, ensure_ascii=False)
        # One-line legend for Properties / Python console
        legend = " | ".join(
            f"#{c.get('index')}:{c.get('label') or c.get('source_action')} "
            f"[{c.get('frame_start')}-{c.get('frame_end')}]"
            for c in clips
        )
        sc["session_body_clip_legend"] = legend[:1020]
    except Exception:
        pass
    try:
        import json as _json
        from pathlib import Path as _Path
        payload = _json.dumps(
            {
                "action": _session_action_name(),
                "frame_cursor": _session_cursor_load(),
                "clips": clips,
            },
            indent=2,
            ensure_ascii=False,
        )
        out = _Path(__file__).resolve().parent / "temp" / "sessions" / "blender_clip_labels.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        snap = _chat_dir() / "session.json"
        snap.write_text(payload, encoding="utf-8")
    except Exception:
        pass


def _set_marker_color(marker, rgb) -> None:
    """Blender 4.4+/5 timeline markers may expose .color (RGBA)."""
    if marker is None:
        return
    try:
        r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
        if hasattr(marker, "color"):
            marker.color = (r, g, b, 1.0)
    except Exception:
        pass


def _vse_editor():
    """Blender 5 uses .strips; older builds use .sequences."""
    sc = bpy.context.scene
    if not sc.sequence_editor:
        sc.sequence_editor_create()
    se = sc.sequence_editor
    coll = getattr(se, "strips", None)
    if coll is None:
        coll = getattr(se, "sequences", None)
    return se, coll


def _vse_iter_all():
    se, _coll = _vse_editor()
    if se is None:
        return []
    bag = getattr(se, "strips_all", None) or getattr(se, "sequences_all", None)
    if bag is not None:
        return list(bag)
    _se, coll = _vse_editor()
    return list(coll) if coll is not None else []


def _vse_remove_named(prefix: str) -> None:
    se, coll = _vse_editor()
    if coll is None:
        return
    for s in _vse_iter_all():
        n = str(getattr(s, "name", "") or "")
        if n.startswith(prefix):
            try:
                coll.remove(s)
            except Exception:
                try:
                    se.sequences.remove(s)
                except Exception:
                    pass


def _vse_color_range(name: str, channel: int, f0: int, f1: int, rgb) -> None:
    """Visible range bar on the Video Sequencer (separate tracks for motion vs speech)."""
    se, coll = _vse_editor()
    if coll is None:
        print("[body_receiver] VSE: no strips/sequences collection")
        return
    _vse_remove_named(name)
    a = int(f0)
    b = max(int(f0) + 1, int(f1) + 1)
    length = max(1, b - a)
    strip = None
    # Blender 5.1: new_effect(name, type, channel, frame_start, length=N)
    # Older: frame_end= or positional end.
    attempts = (
        lambda: coll.new_effect(name, "COLOR", int(channel), a, length=length),
        lambda: coll.new_effect(
            name=name, type="COLOR", channel=int(channel),
            frame_start=a, length=length,
        ),
        lambda: coll.new_effect(name, "COLOR", int(channel), a, frame_end=b),
        lambda: coll.new_effect(name, "COLOR", int(channel), a, b),
    )
    last_err = None
    for fn in attempts:
        try:
            strip = fn()
            break
        except Exception as e:
            last_err = e
            strip = None
    if strip is None:
        print(f"[body_receiver] VSE color {name!r} failed: {last_err}")
        return
    try:
        strip.color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    except Exception:
        pass
    try:
        strip.blend_type = "ALPHA_OVER"
        strip.blend_alpha = 0.9
    except Exception:
        pass
    print(f"[body_receiver] VSE color {name!r} ch{channel} {a}-{b} len={length}")


def _session_action_key_count(act) -> int:
    if act is None:
        return 0
    n = 0
    try:
        for fc in getattr(act, "fcurves", []) or []:
            n += len(fc.keyframe_points)
    except Exception:
        pass
    if n > 0:
        return n
    try:
        for layer in getattr(act, "layers", []) or []:
            for strip in getattr(layer, "strips", []) or []:
                for bag in getattr(strip, "channelbags", []) or []:
                    for fc in getattr(bag, "fcurves", []) or []:
                        n += len(fc.keyframe_points)
    except Exception:
        pass
    return n


def _resolve_existing_session_action_name() -> str:
    """Find the Session Action for THIS session (never steal an older Session_*)."""
    sc = bpy.context.scene
    mem = str(_SESSION_BODY.get("action_name") or "").strip()
    prop = str(sc.get("session_body_action") or "").strip()
    # Memory / scene win if the Action still exists — even with few keys
    for name in (mem, prop):
        if name and not name.startswith(SESSION_CAM_PREFIX) and bpy.data.actions.get(name):
            return name
    # Reload with empty memory: pick Session_* with the most keys (active film)
    ranked = []
    for act in bpy.data.actions:
        n = str(act.name or "")
        if not n.startswith("Session_") or n.startswith(SESSION_CAM_PREFIX):
            continue
        ranked.append((_session_action_key_count(act), n))
    if ranked:
        ranked.sort(reverse=True)
        if ranked[0][0] > 0:
            return ranked[0][1]
    if bpy.data.actions.get(SESSION_ACTION_NAME):
        return SESSION_ACTION_NAME
    return mem or prop or SESSION_ACTION_NAME


def _hydrate_session_clips_from_scene() -> None:
    """Receiver reload wipes _SESSION_BODY; clip table + action name live on the Scene."""
    global _SESSION_BODY
    try:
        import json as _json
        sc = bpy.context.scene
        if not _SESSION_BODY.get("clips"):
            raw = sc.get("session_body_clip_labels")
            if raw:
                clips = _json.loads(raw) if isinstance(raw, str) else list(raw)
                if clips:
                    _SESSION_BODY["clips"] = clips
                    print(f"[body_receiver] hydrated {len(clips)} clips from Scene")
        # Always re-bind the real session Action name (cursor save used to corrupt it)
        act_name = _resolve_existing_session_action_name()
        _SESSION_BODY["action_name"] = act_name
        try:
            sc["session_body_action"] = act_name
        except Exception:
            pass
        if "session_body_cursor" in sc:
            _SESSION_BODY["frame_cursor"] = max(1, int(sc["session_body_cursor"]))
        pel = sc.get("session_pelvis_loc")
        if pel is not None and _BODY_PLAY.get("rest_pelvis_loc") is None:
            try:
                t = (float(pel[0]), float(pel[1]), float(pel[2]))
                _BODY_PLAY["rest_pelvis_loc"] = t
            except Exception:
                pass
        # Restore place SoT only when this session already has clips
        if _SESSION_BODY.get("clips"):
            try:
                sw = sc.get("session_world_xy")
                if sw is not None:
                    _BODY_PLAY["world_root_xy"] = (float(sw[0]), float(sw[1]))
            except Exception:
                pass
        else:
            _place_xy_clear()
    except Exception as e:
        print(f"[body_receiver] clip hydrate skip: {e}")


def _rebuild_session_vse_tracks() -> int:
    """Recreate motion/speech bars from clip table (call when opening VSE)."""
    _hydrate_session_clips_from_scene()
    n = 0
    for c in _SESSION_BODY.get("clips") or []:
        idx = int(c.get("index") or 0)
        f0 = int(c.get("frame_start") or 1)
        f1 = int(c.get("frame_end") or f0)
        _vse_color_range(f"M#{idx}_motion", 3, f0, f1, (0.20, 0.45, 0.95))
        n += 1
        s0 = int(c.get("speech_frame_start") or 0)
        s1 = int(c.get("speech_frame_end") or 0)
        if s1 >= s0 > 0:
            _vse_color_range(f"S#{idx}_zone", 2, s0, s1, (0.95, 0.40, 0.08))
            ap = str(c.get("audio_path") or "")
            if ap:
                _add_speech_sound_strip(ap, s0, s1, f"S#{idx}_speech")
    return n


def _add_speech_sound_strip(audio_path: str, frame_start: int, frame_end: int, name: str) -> None:
    """Put the clip's spoken wav on the VSE so timeline play has speech."""
    if not audio_path:
        return
    from pathlib import Path as _P
    p = _P(audio_path)
    if not p.is_file():
        # also try project-relative
        try:
            alt = _P(__file__).resolve().parent / audio_path
            if alt.is_file():
                p = alt
            else:
                print(f"[body_receiver] speech wav missing: {audio_path}")
                return
        except Exception:
            print(f"[body_receiver] speech wav missing: {audio_path}")
            return
    se, coll = _vse_editor()
    if coll is None:
        return
    _vse_remove_named(name)
    strip = None
    last_err = None
    for fn in (
        lambda: coll.new_sound(name, str(p), 1, int(frame_start)),
        lambda: se.strips.new_sound(name, str(p), 1, int(frame_start)) if hasattr(se, "strips") else None,
        lambda: se.sequences.new_sound(name, str(p), 1, int(frame_start)),
    ):
        try:
            strip = fn()
            if strip is not None:
                break
        except Exception as e:
            last_err = e
            strip = None
    if strip is None:
        print(f"[body_receiver] VSE speech skip: {last_err}")
        return
    try:
        strip.frame_final_end = int(frame_end) + 1
    except Exception:
        pass
    print(f"[body_receiver] VSE speech {name!r} {frame_start}-{frame_end} ← {p.name}")


def _add_session_clip_markers(
    *,
    index: int,
    frame_start: int,
    frame_end: int,
    display_label: str,
    source_action: str = "",
    speech_frame_start: int = 0,
    speech_frame_end: int = 0,
    speech_text: str = "",
    audio_path: str = "",
) -> None:
    """
    Motion markers (#N … [f0-f1]) plus speech markers (S#N … [s0-s1]).
    Speech is the atomic edit unit — do not cut inside S# range.
    """
    f0 = int(frame_start)
    f1 = max(f0, int(frame_end))
    short = _clip_label_short(display_label, max_len=48)
    try:
        sc = bpy.context.scene
        prefix = f"C#{int(index)}"
        sprefix = f"S#{int(index)}"
        for m in list(sc.timeline_markers):
            n = str(getattr(m, "name", "") or "")
            if (
                n.startswith(prefix)
                or n.startswith(sprefix)
                or n.startswith(f"#{int(index)} ")
            ):
                try:
                    sc.timeline_markers.remove(m)
                except Exception:
                    pass
        m0 = sc.timeline_markers.new(short, frame=f0)
        m1 = sc.timeline_markers.new(_clip_label_short(f"C#{int(index)}|end f{f1}", max_len=40), frame=f1)
        # Motion = blue. Speech = orange. Separate VSE tracks beat tiny ticks.
        _set_marker_color(m0, (0.25, 0.55, 1.0))
        _set_marker_color(m1, (0.25, 0.55, 1.0))
        _vse_color_range(f"M#{int(index)}_motion", 3, f0, f1, (0.20, 0.45, 0.95))
        s0, s1 = int(speech_frame_start or 0), int(speech_frame_end or 0)
        if s1 >= s0 > 0:
            bit = (speech_text or "speech").strip().replace("\n", " ")[:18]
            sm0 = sc.timeline_markers.new(
                _clip_label_short(f"S#{int(index)} \"{bit}\" [{s0}-{s1}]", max_len=52),
                frame=s0,
            )
            sm1 = sc.timeline_markers.new(
                _clip_label_short(f"S#{int(index)}|end", max_len=24),
                frame=s1,
            )
            _set_marker_color(sm0, (1.0, 0.45, 0.08))
            _set_marker_color(sm1, (1.0, 0.45, 0.08))
            _vse_color_range(f"S#{int(index)}_zone", 2, s0, s1, (0.95, 0.40, 0.08))
            _add_speech_sound_strip(
                audio_path, s0, s1, f"S#{int(index)}_speech"
            )
    except Exception as e:
        print(f"[body_receiver] timeline markers skip: {e}")

    # Optional NLA block per clip (muted — visual only; Action Editor stays primary)
    try:
        arm = _body_armature()
        session = bpy.data.actions.get(_session_action_name())
        if arm is None or session is None:
            return
        if not arm.animation_data:
            arm.animation_data_create()
        ad = arm.animation_data
        track_name = "SessionClips"
        track = None
        for t in ad.nla_tracks:
            if t.name == track_name:
                track = t
                break
        if track is None:
            track = ad.nla_tracks.new()
            track.name = track_name
        track.mute = True
        # Drop old strip for same clip index
        for s in list(track.strips):
            sn = str(getattr(s, "name", "") or "")
            if sn.startswith(f"C#{int(index)} ") or sn.startswith(f"#{int(index)} "):
                try:
                    track.strips.remove(s)
                except Exception:
                    pass
        strip_name = _clip_label_short(f"C#{int(index)} {source_action or display_label}", max_len=50)
        strip = track.strips.new(strip_name, f0, session)
        try:
            strip.action_frame_start = float(f0)
            strip.action_frame_end = float(f1)
            strip.frame_end = float(f1 + 1)
            strip.extrapolation = "NOTHING"
        except Exception:
            pass
    except Exception as e:
        print(f"[body_receiver] SessionClips NLA skip: {e}")


def _session_action_name() -> str:
    return str(_SESSION_BODY.get("action_name") or SESSION_ACTION_NAME)


def _begin_session_timeline(
    arm=None,
    *,
    reason: str = "begin",
    session_id: str = "",
    session_name: str = "",
    force_new: bool = False,
) -> object:
    """
    Create the session timeline FIRST (empty, named), before any motion.
    Markers + Scene props so the user sees which session this is.
    """
    global _SESSION_BODY
    sid = str(session_id or _SESSION_BODY.get("session_id") or "").strip()
    display = str(session_name or sid or "session").strip()
    if sid:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sid)[:40]
        act_name = f"Session_{safe}" if safe else SESSION_ACTION_NAME
    else:
        act_name = SESSION_ACTION_NAME

    if force_new or not (_SESSION_BODY.get("clips") or []):
        _SESSION_BODY["session_id"] = sid
        _SESSION_BODY["session_name"] = display
        _SESSION_BODY["action_name"] = act_name

    if arm is None:
        arm = _body_armature()
    act = _ensure_session_action()
    _expand_layered_action_range(act, 1, 500)

    if arm is not None:
        if not arm.animation_data:
            arm.animation_data_create()
        try:
            # Never leave master bound during live
            if arm.animation_data.action and arm.animation_data.action.name == _session_action_name():
                if not _SESSION_BODY.get("clips"):
                    arm.animation_data.action = None
        except Exception:
            pass

    try:
        sc = bpy.context.scene
        if force_new or not _SESSION_BODY.get("clips"):
            sc.frame_start = 1
            sc.frame_end = max(int(sc.frame_end or 1), 250)
            sc.frame_current = 1
            sc["session_body_cursor"] = 1
            sc["session_body_clips"] = 0
            sc["session_body_action"] = _session_action_name()
            sc["session_id"] = sid
            sc["session_body_clip_labels"] = "[]"
            sc["session_body_clip_legend"] = f"{display} (empty)"
            _place_xy_clear()
            _clear_session_timeline_markers()
            try:
                sc.timeline_markers.new(_clip_label_short(f"Session|{display}", max_len=48), frame=1)
            except Exception:
                pass
        print(
            f"[body_receiver] SESSION timeline created ({reason}) "
            f"name={display!r} action={_session_action_name()!r} "
            f"clips={len(_SESSION_BODY.get('clips') or [])} cursor={_session_cursor_load()}"
        )
    except Exception as e:
        print(f"[body_receiver] session begin skip: {e}")
    return act


def _ensure_session_action():
    """Permanent master Action shown in Action Editor / Dope Sheet."""
    name = _session_action_name()
    act = bpy.data.actions.get(name)
    if act is None:
        act = bpy.data.actions.new(name)
        print(f"[body_receiver] created master Action {name!r}")
    try:
        act.use_fake_user = True
    except Exception:
        pass
    _expand_layered_action_range(act, 1, 500)
    return act


def _get_or_make_fcurve(action, data_path: str, array_index: int = 0):
    """Get/create an fcurve on Action (legacy). Returns None if layered-only."""
    if action is None:
        return None
    fcurves = getattr(action, "fcurves", None)
    if fcurves is None:
        return None
    try:
        fc = fcurves.find(data_path, index=int(array_index))
        if fc is not None:
            return fc
    except TypeError:
        try:
            fc = fcurves.find(data_path, array_index=int(array_index))
            if fc is not None:
                return fc
        except Exception:
            pass
    except Exception:
        pass
    try:
        return fcurves.new(data_path, index=int(array_index))
    except TypeError:
        try:
            return fcurves.new(data_path, action_group="", index=int(array_index))
        except Exception:
            return None
    except Exception:
        return None


def _pelvis_local_from_world_xy(arm, world_dx: float, world_dy: float):
    """World-XY translation → pelvis pose-location delta (local ≠ world on SMPL-X).

    On this armature pelvis local Y ≈ world Z (height) and local Z ≈ world −Y
    (forward). Never add world_dy to location.y — that lifts the character.
    """
    from mathutils import Vector as _V
    if arm is None:
        return _V((float(world_dx), float(world_dy), 0.0))
    bone = arm.data.bones.get("pelvis")
    if bone is None:
        return _V((float(world_dx), float(world_dy), 0.0))
    r_inv = bone.matrix_local.to_3x3().inverted()
    arm_inv = arm.matrix_world.inverted().to_3x3()
    return r_inv @ (arm_inv @ _V((float(world_dx), float(world_dy), 0.0)))


def _shift_pelvis_world_xy(arm, world_dx: float, world_dy: float) -> None:
    """Add a horizontal world offset onto pelvis.location (all 3 local channels)."""
    if arm is None:
        return
    if abs(float(world_dx or 0.0)) < 1e-9 and abs(float(world_dy or 0.0)) < 1e-9:
        return
    pb = arm.pose.bones.get("pelvis")
    if pb is None:
        return
    ld = _pelvis_local_from_world_xy(arm, float(world_dx or 0.0), float(world_dy or 0.0))
    pb.location.x += float(ld.x)
    pb.location.y += float(ld.y)
    pb.location.z += float(ld.z)


# ---------------------------------------------------------------------------
# Session place SoT (world XY). One owner for MoMask + catalog continuity.
# ---------------------------------------------------------------------------
def _place_xy_clear() -> None:
    """New session / reset — next clip #1 starts at world origin."""
    global _BODY_PLAY
    _BODY_PLAY["world_root_xy"] = (0.0, 0.0)
    _BODY_PLAY["rest_pelvis_loc"] = (0.0, 0.0, 0.0)
    _BODY_PLAY["root_offset"] = None
    try:
        sc = bpy.context.scene
        sc["session_world_xy"] = [0.0, 0.0, 0.0]
        sc["session_pelvis_loc"] = [0.0, 0.0, 0.0]
    except Exception:
        pass


def _place_xy_get() -> tuple[float, float]:
    """Current standing place (world XY). (0,0) if empty session / cleared."""
    if not (_SESSION_BODY.get("clips") or []):
        return (0.0, 0.0)
    wxy = _BODY_PLAY.get("world_root_xy")
    if wxy is not None:
        try:
            return (float(wxy[0]), float(wxy[1]))
        except Exception:
            pass
    try:
        sw = bpy.context.scene.get("session_world_xy")
        if sw is not None:
            return (float(sw[0]), float(sw[1]))
    except Exception:
        pass
    return (0.0, 0.0)


def _place_xy_set(x: float, y: float) -> None:
    """Remember world XY after a clip ends (next clip continues here)."""
    global _BODY_PLAY
    xx, yy = float(x), float(y)
    _BODY_PLAY["world_root_xy"] = (xx, yy)
    try:
        bpy.context.scene["session_world_xy"] = [xx, yy, 0.0]
    except Exception:
        pass


def _pelvis_world_xy(arm) -> tuple[float, float] | None:
    if arm is None:
        return None
    pb = arm.pose.bones.get("pelvis")
    if pb is None:
        return None
    try:
        bpy.context.view_layer.update()
        w = arm.matrix_world @ pb.head
        return (float(w.x), float(w.y))
    except Exception:
        return None


def _is_root_location_channel(data_path: str, array_index: int) -> bool:
    """True for pelvis/root/hips location channels (all 3 — local ≠ world)."""
    if not data_path or "location" not in data_path:
        return False
    try:
        idx = int(array_index)
    except (TypeError, ValueError):
        return False
    if idx not in (0, 1, 2):
        return False
    low = data_path.lower()
    return any(b in low for b in ('"pelvis"', '"root"', '"hips"'))


def _root_channel_delta(data_path: str, array_index: int, root_dx: float, root_dy: float) -> float:
    """World-XY continuity expressed on the matching pelvis local channel."""
    if not _is_root_location_channel(data_path, array_index):
        return 0.0
    try:
        idx = int(array_index)
    except (TypeError, ValueError):
        return 0.0
    arm = bpy.data.objects.get("SMPL-X_Armature")
    loc = _pelvis_local_from_world_xy(arm, root_dx, root_dy)
    if idx == 0:
        return float(loc.x)
    if idx == 1:
        return float(loc.y)
    return float(loc.z)


def _copy_action_keys_offset(
    src_act,
    dst_act,
    *,
    src_f0: int,
    span: int,
    dest_f0: int,
    root_dx: float = 0.0,
    root_dy: float = 0.0,
) -> int:
    """
    Copy keyframes from src→dst with frame offset.
    Also bakes horizontal root continuity into pelvis/root location keys so the
    master Session_Timeline continues from the character's current place (not origin).
    Fast O(keys); does not re-sample every frame.
    Falls back to 0 if Actions are layered without .fcurves.
    """
    if src_act is None or dst_act is None:
        return 0
    n = 0
    src_f1 = src_f0 + span - 1
    wrote_root_loc = False
    # Prefer iterating source fcurves (works for layered via _iter_action_fcurves)
    for sfcu in _iter_action_fcurves(src_act):
        try:
            path = sfcu.data_path
            idx = int(sfcu.array_index)
        except Exception:
            continue
        dfcu = _get_or_make_fcurve(dst_act, path, idx)
        if dfcu is None:
            continue
        try:
            kps = sfcu.keyframe_points
        except Exception:
            continue
        delta = _root_channel_delta(path, idx, root_dx, root_dy)
        for kp in kps:
            try:
                fr = float(kp.co[0])
                val = float(kp.co[1]) + delta
            except Exception:
                continue
            if fr < src_f0 - 0.01 or fr > src_f1 + 0.01:
                continue
            new_fr = float(dest_f0) + (fr - float(src_f0))
            try:
                dfcu.keyframe_points.insert(new_fr, val, options={"FAST"})
                n += 1
                if delta != 0.0:
                    wrote_root_loc = True
            except Exception:
                try:
                    dfcu.keyframe_points.insert(new_fr, val)
                    n += 1
                    if delta != 0.0:
                        wrote_root_loc = True
                except Exception:
                    pass
        try:
            dfcu.update()
        except Exception:
            pass

    # If source has no pelvis location keys but we need continuity, inject hold keys
    # at the session root so scrubbing does not snap back to origin.
    if (abs(float(root_dx or 0.0)) > 1e-6 or abs(float(root_dy or 0.0)) > 1e-6) and not wrote_root_loc:
        n += _inject_root_location_keys(
            dst_act,
            dest_f0=dest_f0,
            span=span,
            root_dx=float(root_dx or 0.0),
            root_dy=float(root_dy or 0.0),
            src_act=src_act,
            src_f0=src_f0,
        )
    return n


def _inject_root_location_keys(
    dst_act,
    *,
    dest_f0: int,
    span: int,
    root_dx: float,
    root_dy: float,
    src_act=None,
    src_f0: int = 1,
) -> int:
    """
    Write pelvis location X/Y keys shifted by root_dx/dy when source has no location keys.
    Samples action start for base X/Y when possible; otherwise uses 0 + offset.
    """
    if dst_act is None or span < 1:
        return 0
    base_x, base_y, base_z = 0.0, 0.0, 0.0
    # Prefer evaluating source start pelvis if fcurves exist
    try:
        targets = _get_action_fcu_targets(src_act) if src_act is not None else []
        for bname, prop, idx, fcu in targets:
            if prop != "location" or bname.lower() not in ("pelvis", "root", "hips"):
                continue
            try:
                val = float(fcu.evaluate(float(src_f0)))
            except Exception:
                continue
            if idx == 0:
                base_x = val
            elif idx == 1:
                base_y = val
            elif idx == 2:
                base_z = val
    except Exception:
        pass
    path = 'pose.bones["pelvis"].location'
    n = 0
    ld = _pelvis_local_from_world_xy(
        bpy.data.objects.get("SMPL-X_Armature"), root_dx, root_dy
    )
    values = (
        (0, float(base_x) + float(ld.x)),
        (1, float(base_y) + float(ld.y)),
        (2, float(base_z) + float(ld.z)),
    )
    for idx, val in values:
        dfcu = _get_or_make_fcurve(dst_act, path, idx)
        if dfcu is None:
            continue
        for i in (0, max(0, span - 1)):
            fr = float(dest_f0 + i)
            try:
                dfcu.keyframe_points.insert(fr, val, options={"FAST"})
                n += 1
            except Exception:
                try:
                    dfcu.keyframe_points.insert(fr, val)
                    n += 1
                except Exception:
                    pass
        try:
            dfcu.update()
        except Exception:
            pass
    return n


def _sample_append_keys(
    arm,
    src_act,
    session,
    *,
    src_f0: int,
    span: int,
    dest_f0: int,
    root_dx: float = 0.0,
    root_dy: float = 0.0,
    fill_mode: str = "hold",
) -> int:
    """Fallback: sample pose each frame and keyframe_insert into session (capped span).

    fill_mode:
      - hold: freeze on last authored frame when take > Action (default)
      - cycle: wrap through Action range (in-place gestures / looped idle)
    """
    n = 0
    if not arm.animation_data:
        arm.animation_data_create()
    rdx = float(root_dx or 0.0)
    rdy = float(root_dy or 0.0)
    try:
        _a0, _a1 = _action_frame_range(src_act)
        src_last = float(_a1)
        src_first = float(_a0)
    except Exception:
        src_first = float(src_f0)
        src_last = float(src_f0 + max(1, span) - 1)
    native = max(1.0, src_last - src_first + 1.0)
    cycle = str(fill_mode or "hold").lower() in ("cycle", "loop", "wrap")
    for i in range(span):
        if cycle:
            src_f = float(src_first + (float(src_f0 - src_first) + i) % native)
        else:
            # Hold last authored pose when take is longer than the Action
            src_f = min(float(src_f0 + i), src_last)
        dest_f = int(dest_f0 + i)
        ok = _apply_action_frame_fast(arm, src_act, src_f)
        if not ok:
            try:
                arm.animation_data.action = src_act
                bpy.context.scene.frame_set(int(round(src_f)))
                bpy.context.view_layer.update()
            except Exception:
                continue
        # Bake horizontal root continuity. Never zero local Z mid-clip:
        # on SMPL-X pelvis local Z ≈ world forward travel; zeroing it teleports.
        # Height (≈ local Y) is left alone here — Look_Ground plant handles it.
        for rname in ("pelvis", "root", "hips", "Pelvis", "Root", "Hips"):
            pb_root = arm.pose.bones.get(rname)
            if pb_root is not None:
                try:
                    if abs(rdx) > 1e-9 or abs(rdy) > 1e-9:
                        ld = _pelvis_local_from_world_xy(arm, rdx, rdy)
                        pb_root.location.x += float(ld.x)
                        pb_root.location.y += float(ld.y)
                        pb_root.location.z += float(ld.z)
                except Exception:
                    pass
                break
        arm.animation_data.action = session
        try:
            slots = getattr(arm.animation_data, "action_suitable_slots", None)
            if slots and len(slots) > 0:
                arm.animation_data.action_slot = slots[0]
        except Exception:
            pass
        for pb in arm.pose.bones:
            try:
                pb.rotation_mode = "QUATERNION"
                pb.keyframe_insert(data_path="rotation_quaternion", frame=dest_f)
                n += 4
            except Exception:
                pass
            try:
                if pb.name.lower() in ("pelvis", "root", "hips") or float(pb.location.length) > 1e-5:
                    pb.keyframe_insert(data_path="location", frame=dest_f)
                    n += 3
            except Exception:
                pass
    # CRITICAL: never leave Session_Timeline or source Action bound after sample-append
    try:
        if arm.animation_data:
            arm.animation_data.action = None
    except Exception:
        pass
    return n


def _assign_session_action_for_scrub(arm, *, allow_during_idle: bool = False) -> bool:
    """
    Bind the session master Action so Space/scrub plays ALL clips (1→N).

    Live UDP play must stay unbound (wrong-frame flash). After rest, idle_hold
    is released by the caller before this runs, or allow_during_idle=True.
    """
    if arm is None:
        return False
    if _BODY_PLAY.get("live") or _BODY_PLAY.get("resting"):
        return False
    if _BODY_PLAY.get("idle_hold") and not allow_during_idle:
        return False
    session = bpy.data.actions.get(_session_action_name()) or _ensure_session_action()
    if session is None:
        return False
    if not arm.animation_data:
        arm.animation_data_create()
    try:
        arm.animation_data.use_tweak_mode = False
    except Exception:
        pass
    arm.animation_data.action = session
    try:
        slots = getattr(arm.animation_data, "action_suitable_slots", None)
        if slots and len(slots) > 0:
            arm.animation_data.action_slot = slots[0]
    except Exception:
        pass
    try:
        arm.animation_data.use_nla = False
    except Exception:
        pass
    print(
        f"[body_receiver] timeline review → {_session_action_name()!r} "
        f"clips={len(_SESSION_BODY.get('clips') or [])} "
        f"frames=1-{int(_SESSION_BODY.get('frame_cursor') or 2) - 1}"
    )
    return True


def _user_reviewing_session(scene) -> bool:
    """True when the user hit Space or scrubbed the timeline after a clip finished."""
    if _BODY_PLAY.get("live") or _BODY_PLAY.get("resting"):
        return False
    try:
        _hydrate_session_clips_from_scene()
    except Exception:
        pass
    has_session = bool(_SESSION_BODY.get("clips") or [])
    if not has_session:
        # Clips table may be empty after reload — Session Action alone is enough
        try:
            act = bpy.data.actions.get(_session_action_name())
            has_session = act is not None
        except Exception:
            has_session = False
    if not has_session:
        return False
    playing = False
    try:
        playing = bool(getattr(bpy.context.screen, "is_animation_playing", False))
    except Exception:
        playing = False
    # After session_bind / Play Session: keep review mode even while paused
    if _BODY_PLAY.get("scrub_user_active"):
        return True
    if _BODY_PLAY.get("scrub_ready") and playing:
        return True
    cur = int(getattr(scene, "frame_current", 1) or 1)
    last = _BODY_PLAY.get("last_frame")
    return bool(playing or (last is not None and int(last) != cur))


def _append_session_master(
    arm,
    src_act,
    *,
    action_frames: int = 0,
    motion_length: int = 0,
    root_dx: float = 0.0,
    root_dy: float = 0.0,
    pose_from: dict | None = None,
    pel_from=None,
    blend_frames: int = 0,
    clip_label: str = "",
    clip_prompt: str = "",
    clip_text: str = "",
    speech_delay_s: float = 0.0,
    speech_duration_s: float = 0.0,
    audio_path: str = "",
    clip_fps: float = 20.0,
    look: dict | None = None,
    clip_policy: str = "",
    engine: str = "",
) -> tuple:
    """
    APPEND into permanent Action ``Session_Timeline``.

    Clip1: frames 1..N  Clip2: N+1..  (133 → next 134). Never replaces prior keys.
    Cursor stored on Scene so reloading the addon does not restart at frame 1.

    root_dx/root_dy: horizontal world continuity — shifts pelvis location keys so the
    new clip continues from the character's current place instead of the origin.
    pose_from/pel_from/blend_frames: smooth first frames from previous pose (no cut jerk).
    clip_label/prompt/text: user-visible labels on the timeline for this clip range.
    """
    global _SESSION_BODY, _ACTION_FCU_CACHE
    if arm is None or src_act is None:
        cur = _session_cursor_load()
        return cur, cur + 1

    if src_act.name == _session_action_name():
        print("[body_receiver] REFUSE: cannot append session timeline into itself")
        cur = _session_cursor_load()
        return cur, cur

    # Ensure master Action exists before first motion
    if not (_SESSION_BODY.get("clips") or []):
        _begin_session_timeline(arm, reason="first-append")

    src_f0, span, reason = _clip_span_for_append(
        src_act, action_frames=action_frames, motion_length=motion_length
    )
    dest_f0 = _session_cursor_load()
    dest_f1 = dest_f0 + span - 1
    session = _ensure_session_action()
    # Expand BEFORE any key insert — Blender 5 default strip is ~25 frames
    _expand_layered_action_range(session, 1, dest_f1 + 8)
    src_act.use_fake_user = True

    # --- Single-owner place: world XY of session end → constant delta on bake ---
    # Clip #1 (no prior clips) always delta (0,0). Never trust caller root_dx if
    # place SoT says origin. MoMask + catalog use the same math.
    n_prior = len(_SESSION_BODY.get("clips") or [])
    place_x, place_y = (0.0, 0.0) if n_prior < 1 else _place_xy_get()

    if not arm.animation_data:
        arm.animation_data_create()
    # Measure source start world XY (then leave Action unbound)
    src_sx, src_sy = 0.0, 0.0
    try:
        arm.animation_data.action = None
        ok0 = _apply_action_frame_fast(arm, src_act, float(src_f0))
        if ok0:
            w0 = _pelvis_world_xy(arm)
            if w0 is not None:
                src_sx, src_sy = w0
        arm.animation_data.action = None
    except Exception:
        pass
    rdx = float(place_x) - float(src_sx)
    rdy = float(place_y) - float(src_sy)
    # Ignore obsolete caller offsets — place SoT is the only input
    print(
        f"[body_receiver] append place=({place_x:.3f},{place_y:.3f}) "
        f"src0=({src_sx:.3f},{src_sy:.3f}) delta=({rdx:.3f},{rdy:.3f}) "
        f"prior_clips={n_prior}"
    )

    t0 = time.time()
    keys = 0
    try:
        # ONE bake path: sample each frame + constant world-XY delta into Session.
        # No fcurve-copy / no pelvis location blend (those caused origin→place grab).
        pol = str(clip_policy or "hold_end").lower()
        src_name = str(getattr(src_act, "name", "") or "").lower()
        # In-place catalog gestures may cycle to fill take; locomotion holds
        # last pose (root wrap = teleport). Explicit loop policy always cycles
        # non-loco clips.
        is_loco = bool(__import__("re").search(r"(walk|run|jump|jog|sprint)", src_name))
        if pol in ("loop", "cycle") and not is_loco:
            fill_mode = "cycle"
        elif (not is_loco) and any(
            k in src_name for k in ("wave", "talk", "idle", "shrug", "nod", "shake", "celebrate")
        ):
            fill_mode = "cycle"
        else:
            fill_mode = "hold"
        keys = _sample_append_keys(
            arm, src_act, session,
            src_f0=src_f0, span=span, dest_f0=dest_f0,
            root_dx=rdx, root_dy=rdy,
            fill_mode=fill_mode,
        )

        # Live must never leave Session bound after append
        try:
            if arm.animation_data:
                arm.animation_data.action = None
        except Exception:
            pass

        try:
            _ACTION_FCU_CACHE.pop(_session_action_name(), None)
        except Exception:
            pass

        # Grow layered strip so later clips are not dropped at ~25 frames
        _expand_layered_action_range(session, 1, dest_f1)

        # Continuity QC + update place SoT from last keyed frame
        try:
            if not arm.animation_data:
                arm.animation_data_create()
            arm.animation_data.action = session
            try:
                slots = getattr(arm.animation_data, "action_suitable_slots", None)
                if slots and len(slots) > 0:
                    arm.animation_data.action_slot = slots[0]
            except Exception:
                pass
            bpy.context.scene.frame_set(int(dest_f0))
            bpy.context.view_layer.update()
            w_start = _pelvis_world_xy(arm)
            bpy.context.scene.frame_set(int(dest_f1))
            bpy.context.view_layer.update()
            w_end = _pelvis_world_xy(arm)
            arm.animation_data.action = None
            if w_start is not None and n_prior >= 1:
                err = (
                    (w_start[0] - place_x) ** 2 + (w_start[1] - place_y) ** 2
                ) ** 0.5
                if err > 0.25:
                    print(
                        f"[body_receiver] ERROR continuity start Δxy={err:.3f}m "
                        f"want=({place_x:.3f},{place_y:.3f}) "
                        f"got=({w_start[0]:.3f},{w_start[1]:.3f})"
                    )
                else:
                    print(
                        f"[body_receiver] continuity start OK Δxy={err:.3f}m"
                    )
            if w_end is not None:
                _place_xy_set(w_end[0], w_end[1])
                print(
                    f"[body_receiver] place SoT ← end ({w_end[0]:.3f},{w_end[1]:.3f})"
                )
        except Exception as e:
            print(f"[body_receiver] place update skip: {e}")
            try:
                if arm.animation_data:
                    arm.animation_data.action = None
            except Exception:
                pass

        # NLA strip of master for NLA Editor (optional mirror)
        try:
            ad = arm.animation_data
            track_name = SESSION_NLA_TRACK
            track = None
            for t in ad.nla_tracks:
                if t.name == track_name:
                    track = t
                    break
            if track is None:
                track = ad.nla_tracks.new()
                track.name = track_name
            track.mute = True  # Action Editor is primary; unmute to use NLA
            for s in list(track.strips):
                try:
                    if getattr(s.action, "name", "") == _session_action_name():
                        track.strips.remove(s)
                except Exception:
                    pass
            strip = track.strips.new("Session_Master", 1, session)
            try:
                strip.action_frame_start = 1.0
                strip.action_frame_end = float(dest_f1)
                strip.frame_end = float(dest_f1 + 1)
            except Exception:
                pass
        except Exception as e:
            print(f"[body_receiver] NLA mirror skip: {e}")

        idx = len(_SESSION_BODY.get("clips") or []) + 1
        display = _make_clip_display_label(
            index=idx,
            source_action=src_act.name,
            clip_label=clip_label,
            prompt=clip_prompt,
            text=clip_text,
            frame_start=dest_f0,
            frame_end=dest_f1,
        )
        fps = float(clip_fps) if float(clip_fps) >= 1.0 else 20.0
        sdelay = max(0.0, float(speech_delay_s or 0.0))
        sdur = max(0.0, float(speech_duration_s or 0.0))
        sf0 = dest_f0 + int(round(sdelay * fps)) if sdur > 0.04 else 0
        sf1 = (sf0 + max(1, int(round(sdur * fps))) - 1) if sf0 > 0 else 0
        if sf1 > dest_f1:
            sf1 = dest_f1
        clip_rec = {
            "index": idx,
            "source_action": src_act.name,
            "label": display,
            "clip_label": str(clip_label or clip_prompt or clip_text or src_act.name),
            "prompt": str(clip_prompt or ""),
            "text": str(clip_text or "")[:200],
            "frame_start": dest_f0,
            "frame_end": dest_f1,
            "span": span,
            "reason": reason,
            "keys": keys,
            "root_dx": rdx,
            "root_dy": rdy,
            "blend_frames": 0,
            "audio_path": str(audio_path or ""),
            "speech_delay_s": sdelay,
            "speech_duration_s": sdur,
            "speech_frame_start": int(sf0),
            "speech_frame_end": int(sf1),
            "clip_fps": fps,
            "look": dict(look or {}),
            "clip_policy": str(clip_policy or "hold_end"),
            "engine": str(engine or ""),
        }
        _SESSION_BODY.setdefault("clips", []).append(clip_rec)
        _session_cursor_save(dest_f1 + 1)

        # Labels on the timeline so the user can see which range is which clip
        _add_session_clip_markers(
            index=idx,
            frame_start=dest_f0,
            frame_end=dest_f1,
            display_label=display,
            source_action=src_act.name,
            speech_frame_start=sf0,
            speech_frame_end=sf1,
            speech_text=clip_text,
            audio_path=audio_path,
        )
        _persist_session_clip_table()
        try:
            _ensure_clip_on_ground(arm, dest_f0, dest_f1)
        except Exception as e:
            print(f"[body_receiver] ground ensure skip: {e}")

        # Clip-boundary continuity QC: prior end vs this start must not teleport
        try:
            clips_now = _SESSION_BODY.get("clips") or []
            if len(clips_now) >= 2 and arm is not None:
                prev = clips_now[-2]
                prev_end = int(prev.get("frame_end") or 0)
                pb = arm.pose.bones.get("pelvis")
                if prev_end >= 1 and pb is not None:
                    if not arm.animation_data:
                        arm.animation_data_create()
                    arm.animation_data.action = session
                    try:
                        slots = getattr(arm.animation_data, "action_suitable_slots", None)
                        if slots and len(slots) > 0:
                            arm.animation_data.action_slot = slots[0]
                    except Exception:
                        pass
                    bpy.context.scene.frame_set(prev_end)
                    bpy.context.view_layer.update()
                    w0 = (arm.matrix_world @ pb.matrix).translation.copy()
                    bpy.context.scene.frame_set(dest_f0)
                    bpy.context.view_layer.update()
                    w1 = (arm.matrix_world @ pb.matrix).translation.copy()
                    arm.animation_data.action = None
                    seam = (
                        (float(w1.x) - float(w0.x)) ** 2
                        + (float(w1.y) - float(w0.y)) ** 2
                    ) ** 0.5
                    if seam > 0.5:
                        print(
                            f"[body_receiver] ERROR continuity seam Δxy={seam:.3f}m "
                            f"f{prev_end}→f{dest_f0} "
                            f"from=({w0.x:.3f},{w0.y:.3f}) to=({w1.x:.3f},{w1.y:.3f}) "
                            f"root=({rdx:.3f},{rdy:.3f}) — origin jump likely"
                        )
                    else:
                        print(
                            f"[body_receiver] continuity seam Δxy={seam:.3f}m OK "
                            f"f{prev_end}→f{dest_f0}"
                        )
        except Exception as e:
            print(f"[body_receiver] continuity QC skip: {e}")

        # Prove the master actually has keys on this clip range (not just a marker)
        kr = _action_key_range(session)
        if kr is None or kr[1] < dest_f0:
            print(
                f"[body_receiver] ERROR session Action has no keys covering "
                f"{dest_f0}-{dest_f1} (key_range={kr}) — review would look empty"
            )
        else:
            print(
                f"[body_receiver] session keys now {kr[0]}-{kr[1]} "
                f"clip#{idx} occupies {dest_f0}-{dest_f1} "
                f"root_shift=({rdx:.3f},{rdy:.3f})"
            )

        sc = bpy.context.scene
        # Update range only — do NOT jump scene.frame_current (evaluates Action → flash)
        try:
            sc.frame_start = 1
            sc.frame_end = max(int(sc.frame_end or 1), int(dest_f1))
            sc["session_body_clips"] = idx
        except Exception:
            pass

        # Never bind Session_Timeline during live append (scrub-only later)
        try:
            if arm.animation_data:
                arm.animation_data.action = None
        except Exception:
            pass

        print(
            f"[body_receiver] MASTER append clip#{idx} label={display!r} "
            f"src={src_act.name!r} → {_session_action_name()} [{dest_f0}..{dest_f1}] "
            f"span={span} ({reason}) keys={keys} root_xy=({rdx:.3f},{rdy:.3f}) "
            f"next={_session_cursor_load()} in {time.time()-t0:.2f}s (unbound live)"
        )
        # Print full session legend after each append
        try:
            legend = sc.get("session_body_clip_legend") or ""
            if legend:
                print(f"[body_receiver] SESSION clips: {legend}")
        except Exception:
            pass
        if keys < 8:
            print(
                "[body_receiver] WARNING: very few keys written — "
                "source Action may be empty or still loading"
            )
        return dest_f0, dest_f1
    except Exception as e:
        print(f"[body_receiver] MASTER append FAILED: {e}")
        import traceback
        traceback.print_exc()
        cur = _session_cursor_load()
        return cur, cur + max(2, span) - 1


def _append_session_nla_strip(arm, src_act, *, action_frames=0, motion_length=0):
    return _append_session_master(
        arm, src_act, action_frames=action_frames, motion_length=motion_length
    )


def _apply_origin_custom_rest(arm=None) -> None:
    """Stand at world origin in pipeline_rest_pose.json. Used by reset / empty timeline."""
    global _BODY_PLAY
    if arm is None:
        arm = _body_armature()
    if arm is None:
        return
    try:
        if arm.animation_data:
            arm.animation_data.action = None
    except Exception:
        pass
    origin = (0.0, 0.0, 0.0)
    pose, src = _resolve_idle_rest_pose(arm)
    _apply_pose_dict(arm, pose, origin)
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    _BODY_PLAY["idle_hold"] = True
    _BODY_PLAY["idle_pose"] = pose
    _BODY_PLAY["resting"] = False
    _BODY_PLAY["live"] = False
    _BODY_PLAY["root_offset"] = None
    _BODY_PLAY["rest_pelvis_loc"] = origin
    _BODY_PLAY["action"] = None
    _BODY_PLAY["scrub_ready"] = False
    _session_root_set(origin)
    _place_xy_clear()
    try:
        sc = bpy.context.scene
        sc.frame_current = 1
    except Exception:
        pass
    print(f"[body_receiver] avatar → origin + custom rest ({src})")


def _reset_session_nla(arm=None, *, session_id: str = "", session_name: str = "") -> None:
    """Wipe prior clips and create a fresh named session timeline."""
    global _SESSION_BODY, _ACTION_FCU_CACHE
    track_name = SESSION_NLA_TRACK
    old_name = _session_action_name()
    _SESSION_BODY = {
        "clips": [],
        "frame_cursor": 1,
        "continue_root": True,
        "action_name": SESSION_ACTION_NAME,
        "nla_track": track_name,
        "session_id": "",
        "session_name": "",
    }
    _session_cursor_save(1)
    if arm is None:
        arm = bpy.data.objects.get("SMPL-X_Armature")
    for drop in (old_name, SESSION_ACTION_NAME):
        act = bpy.data.actions.get(drop)
        if act is None:
            continue
        try:
            _ACTION_FCU_CACHE.pop(drop, None)
        except Exception:
            pass
        try:
            bpy.data.actions.remove(act)
        except Exception:
            try:
                _clear_action_fcurves(act)
            except Exception:
                pass
    if arm is not None and arm.animation_data:
        try:
            for t in list(arm.animation_data.nla_tracks):
                if t.name in (track_name, "SessionClips"):
                    arm.animation_data.nla_tracks.remove(t)
            arm.animation_data.action = None
        except Exception:
            pass
    try:
        _reset_session_cameras()
    except Exception:
        pass
    try:
        sc = bpy.context.scene
        sc.frame_start = 1
        sc.frame_end = 250
        sc.frame_current = 1
        sc["session_body_cursor"] = 1
        sc["session_body_clips"] = 0
        try:
            _vse_remove_named("M#")
            _vse_remove_named("S#")
        except Exception:
            pass
    except Exception:
        pass
    _place_xy_clear()
    _begin_session_timeline(
        arm,
        reason="reset",
        session_id=session_id,
        session_name=session_name,
        force_new=True,
    )
    _apply_origin_custom_rest(arm)
    print(
        f"[body_receiver] SESSION reset — timeline "
        f"{_session_action_name()!r} created, cursor=1, avatar at origin rest"
    )


def _body_armature(name: str | None = None):
    arm = bpy.data.objects.get(name or "SMPL-X_Armature")
    if arm is None:
        for o in bpy.data.objects:
            if o.type == "ARMATURE":
                return o
    return arm


def _capture_body_pose(arm) -> tuple[dict, tuple | None]:
    """Local quats + pelvis location for smooth rest blend."""
    pose = {}
    pel_loc = None
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        q = pb.rotation_quaternion
        pose[pb.name] = (float(q.w), float(q.x), float(q.y), float(q.z))
        if pb.name == "pelvis":
            pel_loc = (float(pb.location.x), float(pb.location.y), float(pb.location.z))
    return pose, pel_loc


def _pelvis_loc_tuple(loc) -> tuple | None:
    """Normalize Vector / tuple / list → (x,y,z) floats."""
    if loc is None:
        return None
    try:
        if hasattr(loc, "x"):
            return (float(loc.x), float(loc.y), float(loc.z))
        if isinstance(loc, (list, tuple)) and len(loc) >= 3:
            return (float(loc[0]), float(loc[1]), float(loc[2]))
    except Exception:
        return None
    return None


def _pelvis_has_place(pel) -> bool:
    """True if pelvis local encodes a real standing place (not bind-pose origin).

    On SMPL-X: local X ≈ side, local Z ≈ forward travel, local Y ≈ height.
    The old check used only |x|+|y| and ignored forward Z — after a walk/run the
    next clip thought we were at origin and jumped home.
    """
    t = _pelvis_loc_tuple(pel)
    if t is None:
        return False
    # Horizontal place (side + forward). Height alone is not travel.
    if abs(t[0]) + abs(t[2]) > 0.02:
        return True
    # Large height offset still counts (sit/crawl / planted root)
    if abs(t[1]) > 0.15:
        return True
    return False


def _session_root_get(arm=None):
    """
    Durable standing position for clip-to-clip continuity.

    Order (must NOT prefer a stale zeroed scene prop over a real live place):
      1) live / rest pelvis if has place
      2) scene session_pelvis_loc if has place
      3) rest / live at origin only when idle_hold (first clip / reset)
    """
    from mathutils import Vector as _V

    # Live / held pose first
    if arm is not None:
        pb = arm.pose.bones.get("pelvis")
        if pb is not None:
            live = _pelvis_loc_tuple(pb.location)
            if _pelvis_has_place(live):
                return _V(live)

    t = _pelvis_loc_tuple(_BODY_PLAY.get("rest_pelvis_loc"))
    if _pelvis_has_place(t):
        return _V(t)

    try:
        t2 = _pelvis_loc_tuple(bpy.context.scene.get("session_pelvis_loc"))
        if _pelvis_has_place(t2):
            return _V(t2)
    except Exception:
        pass

    # Origin is only valid as continuity when we intentionally hold idle/rest there
    if arm is not None:
        pb = arm.pose.bones.get("pelvis")
        if pb is not None:
            live = _pelvis_loc_tuple(pb.location)
            if live is not None and (
                _BODY_PLAY.get("idle_hold") or _BODY_PLAY.get("resting")
            ):
                return _V(live)
    if t is not None and (_BODY_PLAY.get("idle_hold") or _BODY_PLAY.get("resting")):
        return _V(t)
    return None


def _clamp_pelvis_xyz(pel) -> tuple | None:
    """Normalize pelvis tuple. Guard absurd HEIGHT only (local Y ≈ world Z).

    Never touch local Z: on SMPL-X it is forward travel. Zeroing it mid-run
    (or when reading last-frame continuity) teleports the character home.
    """
    t = _pelvis_loc_tuple(pel)
    if t is None:
        return None
    x, y, z = float(t[0]), float(t[1]), float(t[2])
    # Soft height guard only. Crawl/sit need large negative local Y.
    if y > 4.0:
        y = 0.0
    elif y < -3.0:
        y = -2.5
    return (x, y, z)


def _read_last_existing_frame(arm) -> tuple[dict, tuple | None, str]:
    """
    STEP 1 — Position + pose from the last existing motion frame.

    Priority:
      A) Live pose on armature (after detach) if non-origin / idle / rest
      B) Session_Timeline last written frame (end of last appended clip)
      C) scene session_pelvis_loc + empty pose

    Returns (pose_dict, pelvis_xyz_tuple, source_tag).
    Does NOT leave Session_Timeline bound.
    """
    if arm is None:
        return {}, None, "no_arm"

    # Prefer current on-screen pose (where character actually is)
    try:
        if arm.animation_data and arm.animation_data.action is not None:
            arm.animation_data.action = None
    except Exception:
        pass
    try:
        pose, pel = _capture_body_pose(arm)
    except Exception:
        pose, pel = {}, None
    # Trust side+forward (X+Z). Old |x|+|y| missed forward-only walk/run place.
    pel = _clamp_pelvis_xyz(pel)
    if _pelvis_has_place(pel):
        # Trust the on-screen place — character already moved in this session
        _session_root_set(pel)
        return pose or {}, pel, "live_pose"
    if pel is not None and (
        _BODY_PLAY.get("idle_hold") or _BODY_PLAY.get("resting")
    ):
        # Idle/rest at origin is a valid start pose (first clip / reset)
        _session_root_set(pel)
        return pose or {}, pel, "live_pose"
    # Near-origin live pose without idle/rest — prefer Session_Timeline last
    # frame (real end of last clip) over noise at the origin.

    # Session Action last frame (hydrate name — not always Session_Timeline)
    try:
        _hydrate_session_clips_from_scene()
    except Exception:
        pass
    clips = _SESSION_BODY.get("clips") or []
    session = bpy.data.actions.get(_session_action_name())
    if session is None:
        alt = _resolve_existing_session_action_name()
        session = bpy.data.actions.get(alt)
        if session is not None:
            _SESSION_BODY["action_name"] = alt
    last_f = 0
    if clips:
        try:
            last_f = max(int(c.get("frame_end") or 0) for c in clips)
        except Exception:
            last_f = 0
    if last_f < 1:
        try:
            cur = int(_session_cursor_load()) - 1
            last_f = max(0, cur)
        except Exception:
            last_f = 0

    if session is not None and last_f >= 1 and arm.animation_data:
        prev_frame = None
        try:
            prev_frame = int(bpy.context.scene.frame_current)
            arm.animation_data.action = session
            try:
                slots = getattr(arm.animation_data, "action_suitable_slots", None)
                if slots and len(slots) > 0:
                    arm.animation_data.action_slot = slots[0]
            except Exception:
                pass
            bpy.context.scene.frame_set(int(last_f))
            bpy.context.view_layer.update()
            pose2, pel2 = _capture_body_pose(arm)
            pel2 = _clamp_pelvis_xyz(pel2)
            # If the baked end frame is origin, a prior append lost root_dx —
            # do not treat that as the standing place (would chain more jumps).
            if pel2 is not None and _pelvis_has_place(pel2):
                _session_root_set(pel2)
                try:
                    w = arm.matrix_world @ arm.pose.bones["pelvis"].head
                    _BODY_PLAY["world_root_xy"] = (float(w.x), float(w.y))
                    bpy.context.scene["session_world_xy"] = [
                        float(w.x), float(w.y), float(w.z)
                    ]
                except Exception:
                    pass
                return pose2 or {}, pel2, f"session_f{last_f}"
        except Exception as e:
            print(f"[body_receiver] last-frame session read skip: {e}")
        finally:
            try:
                arm.animation_data.action = None
            except Exception:
                pass
            try:
                if prev_frame is not None:
                    bpy.context.scene.frame_set(prev_frame)
            except Exception:
                pass

    # Fallback: stored session root only
    root = _session_root_get(arm)
    if root is not None:
        t = _clamp_pelvis_xyz(root)
        return pose or {}, t, "session_prop"
    return pose or {}, None, "origin"


def _compute_clip_root_offset(arm, act, f0: float, curr_root, restore_pose=None, restore_pel=None):
    """
    Horizontal offset so a new Action (authored near origin) starts at curr_root.
    Samples action start then fully restores prior pose (no origin flash left on bones).
    """
    from mathutils import Vector as _V

    if arm is None or act is None or curr_root is None:
        return None, None
    pb = arm.pose.bones.get("pelvis")
    if pb is None:
        return None, None
    # Ensure no Action is evaluating while we sample via fast path
    prev_act = None
    try:
        if arm.animation_data:
            prev_act = arm.animation_data.action
            arm.animation_data.action = None
    except Exception:
        pass
    try:
        ok = _apply_action_frame_fast(arm, act, float(f0))
        if not ok:
            # Do NOT use scene.frame_set + bind (causes visible origin flash)
            start = _V((0.0, 0.0, float(curr_root.z) if hasattr(curr_root, "z") else 0.0))
            start_w = arm.matrix_world @ pb.head
        else:
            start = pb.location.copy()
            start_w = arm.matrix_world @ pb.head
        # curr_root is pose-local; read its world XY so we never treat local Y as height.
        saved = pb.location.copy()
        try:
            pb.location = curr_root
            bpy.context.view_layer.update()
            curr_w = arm.matrix_world @ pb.head
        except Exception:
            curr_w = start_w
        finally:
            try:
                pb.location = saved
            except Exception:
                pass
        root_offset = _V((
            float(curr_w.x - start_w.x),
            float(curr_w.y - start_w.y),
            0.0,
        ))
        return root_offset, start
    except Exception:
        return None, None
    finally:
        # Always restore previous pose/place — never leave Action pose at origin
        try:
            if restore_pose:
                _apply_pose_dict(arm, restore_pose, restore_pel or curr_root)
            else:
                pb.location.x = float(curr_root.x)
                pb.location.y = float(curr_root.y)
                # keep z from sample or curr
                if hasattr(curr_root, "z"):
                    pb.location.z = float(curr_root.z)
        except Exception:
            pass
        try:
            if arm.animation_data is not None:
                arm.animation_data.action = None  # never leave source Action bound
        except Exception:
            pass


def _session_root_set(loc) -> None:
    """
    Persist session root on _BODY_PLAY and Scene (survives clip resets / script reload).

    Store full pelvis local (x,y,z). On SMPL-X local Z ≈ world forward — must
    keep it so the next clip continues from where the character is. Only soft-
    guard absurd HEIGHT (local Y).
    """
    global _BODY_PLAY
    t = _clamp_pelvis_xyz(loc)
    if t is None:
        return
    _BODY_PLAY["rest_pelvis_loc"] = t
    try:
        bpy.context.scene["session_pelvis_loc"] = [t[0], t[1], t[2]]
    except Exception:
        pass


def _apply_root_offset(arm) -> None:
    """
    Place pelvis in session space after Action eval.

    root_offset is WORLD XY (from _compute_clip_root_offset). Pelvis local Y
    is world height — convert before adding or the avatar floats / sinks.

    Never zero local Z here: that is forward travel and wiping it mid-motion
    is the run/walk teleport bug.

    - If this frame wrote pelvis location from Action fcurves:
        action pose + world-XY continuity (height unchanged)
    - If Action has no root location channels this frame:
        same conversion added onto the authored (usually rest) location
    """
    if arm is None:
        return
    off = _BODY_PLAY.get("root_offset")
    if off is None:
        return
    pb = arm.pose.bones.get("pelvis")
    if pb is None:
        return
    try:
        ox = float(off.x) if hasattr(off, "x") else float(off[0])
        oy = float(off.y) if hasattr(off, "y") else float(off[1])
        # Soft height guard only (local Y). Leave local Z (forward) intact.
        ay = float(pb.location.y)
        if ay > 4.0:
            pb.location.y = 0.0
        elif ay < -3.0:
            pb.location.y = -2.5
        _shift_pelvis_world_xy(arm, ox, oy)
    except Exception:
        pass


def _euler_xyz_to_quat(x: float, y: float, z: float):
    """XYZ euler (radians) → (w,x,y,z) quaternion."""
    try:
        from mathutils import Euler
        q = Euler((float(x), float(y), float(z)), "XYZ").to_quaternion()
        return (float(q.w), float(q.x), float(q.y), float(q.z))
    except Exception:
        # rough fallback: small angles as axis components
        return (1.0, float(x) * 0.5, float(y) * 0.5, float(z) * 0.5)


def _smplx_shoulder_euler(side: str, raise_up: float = 0.0, forward: float = 0.0, twist: float = 0.0):
    """
    Same axis map as face_agents.body_agent (rest ~T-pose, arms horizontal).
    raise_up>0 lifts arm; raise_up<0 lowers arm toward body (hands down).
    """
    if side == "right":
        return (
            0.25 * raise_up + 0.15 * forward,
            -0.35 * forward + twist,
            -0.95 * raise_up,
        )
    return (
        0.25 * raise_up + 0.15 * forward,
        0.35 * forward + twist,
        0.95 * raise_up,
    )


def _smplx_elbow_euler(side: str, flex: float = 0.0, twist: float = 0.0):
    if side == "right":
        return (flex, twist, -0.15 * flex)
    return (flex, -twist, 0.15 * flex)


def _pipeline_rest_pose_path():
    """body_motion/pipeline_rest_pose.json — live-captured rest from Blender."""
    import os
    from pathlib import Path
    roots = []
    # 1) Explicit env
    env = os.environ.get("PROJFACE_ROOT", "").strip()
    if env:
        roots.append(Path(env))
    # 2) Fixed project root (always try)
    roots.append(Path(r"C:\me\proj\projface_v1"))
    # 3) This script’s project root (…/projface_v1/blender_receiver.py)
    try:
        roots.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    # 4) Open blend file’s folder and parents
    try:
        if bpy.data.filepath:
            fp = Path(bpy.data.filepath).resolve()
            roots.insert(0, fp.parent)
            roots.insert(0, fp.parent.parent)
    except Exception:
        pass
    # 5) CWD
    try:
        roots.append(Path.cwd())
    except Exception:
        pass
    seen = set()
    for r in roots:
        try:
            key = str(r.resolve())
        except Exception:
            key = str(r)
        if key in seen:
            continue
        seen.add(key)
        for rel in (
            Path("body_motion") / "pipeline_rest_pose.json",
            Path("pipeline_rest_pose.json"),
        ):
            p = r / rel
            if p.is_file():
                return p
    return None


def _load_pipeline_rest_pose(arm) -> dict | None:
    """
    Load captured rest pose (wxyz quats per bone). Authoritative custom rest
    for the whole pipeline — not A-pose and not T-pose.
    """
    import json
    path = _pipeline_rest_pose_path()
    if path is None:
        print(
            "[body_receiver] WARN: pipeline_rest_pose.json NOT FOUND "
            "(expected body_motion/pipeline_rest_pose.json) — custom rest required"
        )
        return None
    if arm is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        bones = data.get("bones_wxyz") or data.get("pose") or {}
        if not bones:
            return None
        pose = {}
        for pb in arm.pose.bones:
            pose[pb.name] = (1.0, 0.0, 0.0, 0.0)
        n = 0
        for name, q in bones.items():
            if arm.pose.bones.get(name) is None:
                continue
            if isinstance(q, dict):
                q = q.get("quat") or q.get("wxyz") or [1, 0, 0, 0]
            if not isinstance(q, (list, tuple)) or len(q) < 4:
                continue
            pose[name] = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            n += 1
        if n < 4:
            print(f"[body_receiver] rest file {path} has only {n} matching bones")
            return None
        print(f"[body_receiver] REST LOADED from {path} ({n} bones)")
        return pose
    except Exception as e:
        print(f"[body_receiver] pipeline_rest_pose load failed: {e}")
        return None


def _procedural_idle_pose(arm) -> dict:
    """
    Last-resort identity hold if custom rest file is missing.
    Does NOT invent A-pose or T-pose arm drops — pipeline must use
    body_motion/pipeline_rest_pose.json.
    """
    captured = _load_pipeline_rest_pose(arm)
    if captured:
        return captured

    print(
        "[body_receiver] ERROR: no custom rest — using identity bind. "
        "Capture rest to body_motion/pipeline_rest_pose.json"
    )
    pose = {}
    for pb in arm.pose.bones:
        pose[pb.name] = (1.0, 0.0, 0.0, 0.0)
    return pose


def _sample_action_pose(arm, act_name: str, frame: int | None = None) -> dict | None:
    """Evaluate an Action on the armature and capture pose (restored after)."""
    act = bpy.data.actions.get(act_name) if act_name else None
    if arm is None or act is None:
        return None
    prev_act = arm.animation_data.action if arm.animation_data else None
    prev_frame = None
    try:
        prev_frame = int(bpy.context.scene.frame_current)
    except Exception:
        pass
    try:
        if not arm.animation_data:
            arm.animation_data_create()
        arm.animation_data.action = act
        f0, f1 = _action_frame_range(act)
        fr = int(frame) if frame is not None else int(f0)
        fr = max(f0, min(f1, fr))
        bpy.context.scene.frame_set(fr)
        bpy.context.view_layer.update()
        pose, _ = _capture_body_pose(arm)
        return pose
    except Exception as e:
        print(f"[body_receiver] sample action pose failed ({act_name}): {e}")
        return None
    finally:
        try:
            if arm.animation_data is not None:
                arm.animation_data.action = prev_act
            if prev_frame is not None:
                bpy.context.scene.frame_set(prev_frame)
        except Exception:
            pass


def _resolve_idle_rest_pose(arm) -> tuple[dict, str]:
    """
    Rest priority — custom rest only:
      1) pipeline_rest_pose.json (captured live Blender pose — authoritative)
      2) identity bind (last resort, logs error) — never geometric A-pose
    """
    captured = _load_pipeline_rest_pose(arm)
    if captured:
        return captured, "pipeline_rest_pose.json"

    return _procedural_idle_pose(arm), "identity_fallback_no_custom_rest"


def _assign_action_on_arm(arm, act_name: str) -> bool:
    act = bpy.data.actions.get(act_name) if act_name else None
    if arm is None or act is None:
        return False
    if not arm.animation_data:
        arm.animation_data_create()
    arm.animation_data.action = act
    try:
        slots = getattr(arm.animation_data, "action_suitable_slots", None)
        if slots and len(slots) > 0:
            arm.animation_data.action_slot = slots[0]
    except Exception:
        pass
    try:
        arm.data.pose_position = "POSE"
    except Exception:
        pass
    return True


def _apply_pose_dict(arm, pose: dict, pel_loc=None) -> None:
    """Write a full pose dict onto the armature (quaternions)."""
    if arm is None or not pose:
        return
    for bname, src in pose.items():
        pb = arm.pose.bones.get(bname)
        if pb is None or not src:
            continue
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = src
        if bname == "pelvis" and pel_loc is not None:
            pb.location = pel_loc
    if pel_loc is not None and "pelvis" in arm.pose.bones:
        arm.pose.bones["pelvis"].location = pel_loc


def _ease_smoothstep(t: float) -> float:
    """Hermite smoothstep 0→1 (zero derivative at ends — no blend jerk)."""
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _clip_transition_active() -> bool:
    """True while live clip-to-clip pose blend is still running."""
    if not _BODY_PLAY.get("trans_from"):
        return False
    dur = float(_BODY_PLAY.get("trans_duration") or 0.0)
    if dur <= 0.01:
        return False
    t0 = float(_BODY_PLAY.get("trans_t0") or 0.0)
    if t0 <= 0.0:
        return False
    return (time.time() - t0) < dur


def _blend_current_pose_from(arm, pose_from: dict, pel_from, ease: float) -> None:
    """
    In-place blend: pose_from (ease=0) → current armature pose (ease=1).
    Call AFTER evaluating the target clip frame (+ root offset). Smooths clip cuts.
    """
    if arm is None or not pose_from:
        return
    ease = _ease_smoothstep(ease)
    if ease >= 0.999:
        return
    if ease <= 0.001:
        _apply_pose_dict(arm, pose_from, pel_from)
        return

    try:
        from mathutils import Quaternion
        use_mu = True
    except Exception:
        use_mu = False

    identity = (1.0, 0.0, 0.0, 0.0)
    names = set(pose_from.keys())
    # Include major bones currently on the armature (clip may introduce others)
    try:
        for pb in arm.pose.bones:
            if pb.name in pose_from or pb.name.lower() in (
                "pelvis", "spine1", "spine2", "spine3", "neck", "head",
                "left_hip", "right_hip", "left_knee", "right_knee",
                "left_ankle", "right_ankle", "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                "left_collar", "right_collar",
            ):
                names.add(pb.name)
    except Exception:
        pass
    if len(names) > 90:
        names = {
            n for n in names
            if not any(x in n for x in (
                "index2", "index3", "middle2", "middle3",
                "ring2", "ring3", "pinky2", "pinky3",
                "thumb2", "thumb3",
            ))
        }

    pel_to = None
    pb_pel = arm.pose.bones.get("pelvis")
    if pb_pel is not None:
        pel_to = (
            float(pb_pel.location.x),
            float(pb_pel.location.y),
            float(pb_pel.location.z),
        )
    pf = _pelvis_loc_tuple(pel_from)

    for bname in names:
        pb = arm.pose.bones.get(bname)
        if pb is None:
            continue
        pb.rotation_mode = "QUATERNION"
        # Target = current evaluated clip pose
        try:
            q = pb.rotation_quaternion
            dst = (float(q.w), float(q.x), float(q.y), float(q.z))
        except Exception:
            dst = identity
        src = pose_from.get(bname) or dst
        if use_mu:
            pb.rotation_quaternion = Quaternion(src).slerp(Quaternion(dst), ease)
        else:
            w0, x0, y0, z0 = src
            w1, x1, y1, z1 = dst
            w = w0 * (1.0 - ease) + w1 * ease
            x = x0 * (1.0 - ease) + x1 * ease
            y = y0 * (1.0 - ease) + y1 * ease
            z = z0 * (1.0 - ease) + z1 * ease
            nrm = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
            pb.rotation_quaternion = (w / nrm, x / nrm, y / nrm, z / nrm)

    # Pelvis location: ALWAYS the new-clip place (already root-offset).
    # Never lerp from pel_from → pel_to — that looks like "grab from origin
    # (or last place) and slide to the clip start" before the motion plays.
    if pb_pel is not None and pel_to is not None:
        pb_pel.location = (
            float(pel_to[0]),
            float(pel_to[1]),
            float(pel_to[2]),
        )


def _apply_clip_transition_blend(arm) -> bool:
    """
    Soften clip1→clip2 cut: slerp last pose into current evaluated frame.
    Returns True if blend still active.
    """
    global _BODY_PLAY
    pose_from = _BODY_PLAY.get("trans_from") or {}
    if not pose_from or arm is None:
        return False
    dur = float(_BODY_PLAY.get("trans_duration") or 0.0)
    if dur <= 0.01:
        _BODY_PLAY["trans_from"] = {}
        return False
    t0 = float(_BODY_PLAY.get("trans_t0") or 0.0)
    elapsed = max(0.0, time.time() - t0)
    if elapsed >= dur:
        _BODY_PLAY["trans_from"] = {}
        _BODY_PLAY["trans_pel"] = None
        return False
    ease = elapsed / max(1e-4, dur)
    pel_from = _BODY_PLAY.get("trans_pel")
    _blend_current_pose_from(arm, pose_from, pel_from, ease)
    return True


def _bake_session_clip_transition(
    arm,
    session,
    src_act,
    *,
    src_f0: int,
    dest_f0: int,
    span: int,
    root_dx: float,
    root_dy: float,
    pose_from: dict,
    pel_from,
    blend_frames: int,
) -> int:
    """
    Overwrite the first blend_frames of an appended clip with pose-slerp keys
    from the previous clip's end pose → new clip (root-offset applied).
    Prevents jerks when scrubbing Session_Timeline across clip boundaries.
    """
    if arm is None or session is None or src_act is None or not pose_from:
        return 0
    bf = max(0, min(int(blend_frames), int(span)))
    if bf < 2:
        return 0
    rdx = float(root_dx or 0.0)
    rdy = float(root_dy or 0.0)
    n = 0
    if not arm.animation_data:
        arm.animation_data_create()
    # Detach evaluation while we sample
    prev_act = None
    try:
        prev_act = arm.animation_data.action
        arm.animation_data.action = None
    except Exception:
        pass

    for i in range(bf):
        ease = _ease_smoothstep(i / max(1, bf - 1))
        src_f = float(src_f0 + i)
        dest_f = int(dest_f0 + i)
        ok = _apply_action_frame_fast(arm, src_act, src_f)
        if not ok:
            try:
                arm.animation_data.action = src_act
                bpy.context.scene.frame_set(int(round(src_f)))
                arm.animation_data.action = None
            except Exception:
                continue
        # World XY place for new clip. Must convert — local Y is height.
        if abs(rdx) > 1e-9 or abs(rdy) > 1e-9:
            try:
                _shift_pelvis_world_xy(arm, rdx, rdy)
            except Exception:
                pass
        _blend_current_pose_from(arm, pose_from, pel_from, ease)
        try:
            arm.animation_data.action = session
            slots = getattr(arm.animation_data, "action_suitable_slots", None)
            if slots and len(slots) > 0:
                arm.animation_data.action_slot = slots[0]
        except Exception:
            pass
        for pb in arm.pose.bones:
            try:
                pb.rotation_mode = "QUATERNION"
                pb.keyframe_insert(data_path="rotation_quaternion", frame=dest_f)
                n += 4
            except Exception:
                pass
            try:
                if pb.name.lower() in ("pelvis", "root", "hips") or float(pb.location.length) > 1e-5:
                    pb.keyframe_insert(data_path="location", frame=dest_f)
                    n += 3
            except Exception:
                pass
        try:
            arm.animation_data.action = None
        except Exception:
            pass

    try:
        if prev_act is not None:
            arm.animation_data.action = prev_act
    except Exception:
        pass
    try:
        _ACTION_FCU_CACHE.pop(_session_action_name(), None)
    except Exception:
        pass
    print(
        f"[body_receiver] clip transition bake frames {dest_f0}..{dest_f0 + bf - 1} "
        f"({bf}f) keys≈{n}"
    )
    return n


def _hold_idle_pose(arm) -> None:
    """Keep hands-down idle applied every tick while idle_hold is set."""
    global _BODY_PLAY
    if not _BODY_PLAY.get("idle_hold"):
        return
    if _BODY_PLAY.get("live") or _BODY_PLAY.get("resting"):
        return
    pose = _BODY_PLAY.get("idle_pose") or {}
    pel = _BODY_PLAY.get("rest_pelvis_loc")
    if not pose:
        return
    # Ensure no action overrides the hold
    if arm.animation_data and arm.animation_data.action is not None:
        # Keep last_action name for scrub; clear evaluation for hold
        if not _BODY_PLAY.get("scrub_user_active"):
            arm.animation_data.action = None
    _apply_pose_dict(arm, pose, pel)


def _hold_live_session_end(arm, sess_act=None, frame=None) -> None:
    """End of take: freeze last session pose. Do not rest-blend to catalog idle."""
    global _BODY_PLAY
    if arm is not None and sess_act is not None and frame is not None:
        try:
            _apply_action_frame_fast(arm, sess_act, float(frame))
        except Exception:
            pass
    pose, pel = (None, None)
    try:
        pose, pel = _capture_body_pose(arm)
    except Exception:
        pass
    _BODY_PLAY["live"] = False
    _BODY_PLAY["resting"] = False
    _BODY_PLAY["idle_hold"] = True
    if pose:
        _BODY_PLAY["idle_pose"] = pose
    if pel is not None:
        pel = _clamp_pelvis_xyz(pel)
        _BODY_PLAY["rest_pelvis_loc"] = pel
        _session_root_set(pel)
    try:
        wxy = _pelvis_world_xy(arm)
        if wxy is not None:
            _place_xy_set(wxy[0], wxy[1])
    except Exception:
        pass
    _BODY_PLAY["scrub_ready"] = True
    print(
        f"[body_receiver] hold last session pose frame={frame} "
        f"pel={pel} place={_place_xy_get()} (no idle rest)"
    )


def _start_body_rest_blend(arm=None, rest_duration: float = 0.55) -> None:
    """
    Smooth slerp from current pose → hands-down IDLE (not T-pose bind).
    Keeps pelvis.location so the character stays where they stood.
    After settle, holds idle pose; last Action kept for timeline scrub only.
    """
    global _BODY_PLAY
    arm = arm or _body_armature(_BODY_PLAY.get("arm_name"))
    if arm is None:
        _BODY_PLAY["live"] = False
        _BODY_PLAY["resting"] = False
        return
    # Remember clip for scrub/replay
    if _BODY_PLAY.get("action"):
        _BODY_PLAY["last_action"] = _BODY_PLAY.get("action")
    # Evaluate current action pose one more time before capturing
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    pose, pel_loc = _capture_body_pose(arm)
    idle_to, idle_src = _resolve_idle_rest_pose(arm)
    # Temporarily clear action so we can drive bones for the rest blend
    if arm.animation_data:
        arm.animation_data.action = None
    try:
        arm.data.pose_position = "POSE"
    except Exception:
        pass
    _BODY_PLAY["resting"] = True
    _BODY_PLAY["live"] = False  # stop wall-clock scrub
    _BODY_PLAY["idle_hold"] = False
    _BODY_PLAY["rest_t0"] = time.time()
    _BODY_PLAY["rest_duration"] = max(0.25, float(rest_duration))
    _BODY_PLAY["rest_from"] = pose
    _BODY_PLAY["rest_to"] = idle_to  # hands-down idle target (NOT identity T-pose)
    _BODY_PLAY["idle_pose"] = idle_to
    _BODY_PLAY["rest_pelvis_loc"] = pel_loc
    _session_root_set(pel_loc)  # durable for next clip (scene + body play)
    _BODY_PLAY["action"] = None
    _BODY_PLAY["pending_assign"] = False
    _BODY_PLAY["scrub_ready"] = False
    print(
        f"[body_receiver] smooth IDLE rest ({_BODY_PLAY['rest_duration']:.2f}s) "
        f"src={idle_src} pelvis_loc={pel_loc}  (hands down, not T-pose)"
    )


def _apply_body_rest_blend(arm) -> bool:
    """Return True if still blending; False when finished. Throttled ~30Hz."""
    global _BODY_PLAY
    if not _BODY_PLAY.get("resting"):
        return False
    # throttle: body rest does not need every modal tick
    now = time.time()
    last = float(_BODY_PLAY.get("rest_last_tick") or 0.0)
    if (now - last) < 0.03 and (_BODY_PLAY.get("rest_from")):
        # still in progress; skip work this frame
        dur_chk = float(_BODY_PLAY.get("rest_duration") or 0.55)
        if (now - float(_BODY_PLAY.get("rest_t0") or 0.0)) < dur_chk:
            return True
    _BODY_PLAY["rest_last_tick"] = now

    dur = float(_BODY_PLAY.get("rest_duration") or 0.55)
    elapsed = max(0.0, now - float(_BODY_PLAY.get("rest_t0") or 0.0))
    t = min(1.0, elapsed / max(1e-4, dur))
    ease = t * t * (3.0 - 2.0 * t)
    rest_from = _BODY_PLAY.get("rest_from") or {}
    rest_to = _BODY_PLAY.get("rest_to") or _BODY_PLAY.get("idle_pose") or {}
    if not rest_to and arm is not None:
        rest_to, _src = _resolve_idle_rest_pose(arm)
        _BODY_PLAY["rest_to"] = rest_to
        _BODY_PLAY["idle_pose"] = rest_to
    pel_loc = _BODY_PLAY.get("rest_pelvis_loc")

    # Blend all bones that appear in either pose (major body + hands)
    names = set(rest_from.keys()) | set(rest_to.keys())
    if len(names) > 80:
        names = {
            n for n in names
            if not any(x in n for x in ("index2", "index3", "middle2", "middle3",
                                        "ring2", "ring3", "pinky2", "pinky3",
                                        "thumb2", "thumb3"))
        }

    try:
        from mathutils import Quaternion
        use_mathutils = True
    except Exception:
        use_mathutils = False

    identity = (1.0, 0.0, 0.0, 0.0)
    for bname in names:
        pb = arm.pose.bones.get(bname)
        if pb is None:
            continue
        pb.rotation_mode = "QUATERNION"
        src = rest_from.get(bname) or identity
        dst = rest_to.get(bname) or identity
        if use_mathutils:
            pb.rotation_quaternion = Quaternion(src).slerp(Quaternion(dst), ease)
        else:
            w0, x0, y0, z0 = src
            w1, x1, y1, z1 = dst
            w = w0 * (1.0 - ease) + w1 * ease
            x = x0 * (1.0 - ease) + x1 * ease
            y = y0 * (1.0 - ease) + y1 * ease
            z = z0 * (1.0 - ease) + z1 * ease
            n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
            pb.rotation_quaternion = (w / n, x / n, y / n, z / n)
        if bname == "pelvis" and pel_loc is not None:
            pb.location = pel_loc

    if pel_loc is not None and "pelvis" in arm.pose.bones:
        arm.pose.bones["pelvis"].location = pel_loc

    if t >= 1.0:
        # Snap to custom pipeline rest and HOLD it (do NOT rebind Session_Timeline —
        # that jumped to last motion frame and undid rest + position).
        final_pose = rest_to or _procedural_idle_pose(arm)
        if arm.animation_data:
            arm.animation_data.action = None
        _apply_pose_dict(arm, final_pose, pel_loc)
        _BODY_PLAY["resting"] = False
        _BODY_PLAY["rest_from"] = {}
        _BODY_PLAY["rest_to"] = final_pose
        _BODY_PLAY["idle_pose"] = final_pose
        _BODY_PLAY["idle_hold"] = True  # keep custom rest until next body packet
        _BODY_PLAY["live"] = False
        _BODY_PLAY["root_offset"] = None
        # Remember end root for next clip continuity (must survive next _BODY_PLAY rebuild)
        _session_root_set(pel_loc)

        # NEVER leave Session_Timeline (or any Action) bound after rest — causes flash
        try:
            if arm.animation_data:
                arm.animation_data.action = None
        except Exception:
            pass

        clips = _SESSION_BODY.get("clips") or []
        # Mark scrub range for Action Editor only — do not evaluate Session_Timeline now
        if clips:
            f1 = max(1, int(_SESSION_BODY.get("frame_cursor") or 2) - 1)
            try:
                sc = bpy.context.scene
                sc.frame_start = 1
                sc.frame_end = f1
            except Exception:
                pass
            # Review MUST play the master session Action (all clips), never last source clip
            _BODY_PLAY["last_action"] = _session_action_name()
            _BODY_PLAY["f0"] = 1
            _BODY_PLAY["f1"] = f1
            _BODY_PLAY["scrub_ready"] = True
            _BODY_PLAY["action"] = None
            print(
                f"[body_receiver] IDLE settled (custom rest HELD, pelvis={pel_loc}) "
                f"session frames 1-{f1} ({len(clips)} clips). "
                f"Space/scrub will play {_session_action_name()!r}."
            )
        else:
            last = _BODY_PLAY.get("last_action")
            _BODY_PLAY["scrub_ready"] = bool(last)
            _BODY_PLAY["action"] = None
            print(
                f"[body_receiver] IDLE rest settled + held. last_action={last!r} pelvis={pel_loc}."
            )
        return False
    return True


def _queue_body_packet(packet: dict) -> None:
    """Store body play intent; modal applies on main thread."""
    global _BODY_PLAY, _SESSION_BODY
    # Script reload clears memory; Scene still has clip table + Session_<id> Action
    try:
        _hydrate_session_clips_from_scene()
    except Exception:
        pass

    op = str(packet.get("op") or "").lower()
    if op == "session_delete_range":
        a = int(packet.get("frame_start") or 1)
        b = int(packet.get("frame_end") or a)
        removed, n_keys = _delete_session_clips_overlapping(a, b)
        print(f"[session] delete range {a}-{b} clips={len(removed)} keys={n_keys}")
        return

    # After movie install: bind Session master so Space/scrub shows body motion
    if op in ("session_bind", "timeline_review", "play_ready"):
        arm = _body_armature(packet.get("armature") or _BODY_PLAY.get("arm_name"))
        try:
            _hydrate_session_clips_from_scene()
            _BODY_PLAY["idle_hold"] = False
            _BODY_PLAY["live"] = False
            _BODY_PLAY["resting"] = False
            _BODY_PLAY["scrub_ready"] = True
            _BODY_PLAY["scrub_user_active"] = True
            _BODY_PLAY["pending_assign"] = False
            _BODY_PLAY["action"] = None
            _BODY_PLAY["last_action"] = _session_action_name()
            _CAMERA_PLAY["live"] = False
            _CAMERA_PLAY["hold_after"] = False
            _CAMERA_PLAY["hold_world_loc"] = None
            _CAMERA_PLAY["hold_world_look"] = None
            ok = _assign_session_action_for_scrub(arm, allow_during_idle=True)
            _bind_camera_for_timeline_review()
            sc = bpy.context.scene
            cur = max(1, int(_SESSION_BODY.get("frame_cursor") or 2) - 1)
            sc.frame_start = 1
            if cur > 1:
                sc.frame_end = cur
            sc.frame_current = 1
            # Keep Session bound even if a later idle tick runs
            if arm and arm.animation_data:
                sess = bpy.data.actions.get(_session_action_name())
                if sess is not None:
                    arm.animation_data.action = sess
                    try:
                        slots = getattr(arm.animation_data, "action_suitable_slots", None)
                        if slots and len(slots) > 0:
                            arm.animation_data.action_slot = slots[0]
                    except Exception:
                        pass
                    try:
                        arm.animation_data.use_nla = False
                    except Exception:
                        pass
            print(
                f"[body_receiver] session_bind ok={ok} action={_session_action_name()!r} "
                f"clips={len(_SESSION_BODY.get('clips') or [])} end={sc.frame_end} "
                f"bound={arm.animation_data.action.name if arm and arm.animation_data and arm.animation_data.action else None}"
            )
        except Exception as e:
            print(f"[body_receiver] session_bind failed: {e}")
        return

    # Session create / reset — timeline first, named, empty
    if (
        packet.get("session_reset")
        or packet.get("session_begin")
        or op in ("session_reset", "session_begin")
    ):
        arm = _body_armature(packet.get("armature") or _BODY_PLAY.get("arm_name"))
        sid = str(packet.get("session_id") or packet.get("session_name") or "").strip()
        sname = str(packet.get("session_name") or sid).strip()
        force = bool(packet.get("session_reset")) or op == "session_reset"
        if force or not (_SESSION_BODY.get("clips") or []):
            if force:
                _reset_session_nla(arm, session_id=sid, session_name=sname)
            else:
                _begin_session_timeline(
                    arm, reason="session_begin", session_id=sid, session_name=sname, force_new=True
                )
        if force:
            _start_body_rest_blend(arm, rest_duration=0.35)
        return

    # Explicit end-of-speech / rest request (smooth like face)
    act_raw = str(packet.get("action") or packet.get("clip_id") or "").strip().lower()
    want_rest = bool(packet.get("rest") or packet.get("stop")) or act_raw in (
        "rest",
        "stop",
        "none",
    )
    if want_rest and not packet.get("force_play"):
        rd = float(packet.get("rest_duration") or packet.get("smooth_s") or 0.45)
        arm = _body_armature(packet.get("armature") or _BODY_PLAY.get("arm_name"))
        _start_body_rest_blend(arm, rest_duration=rd)
        return

    raw = (
        packet.get("clip_id")
        or packet.get("action")
        or packet.get("state")
        or packet.get("base_state")
        or "idle"
    )
    lib = str(packet.get("library_blend") or "")
    act_name = _resolve_body_action_name(str(raw), library_blend=lib)
    if not act_name:
        # try state fallback
        st = packet.get("state") or packet.get("base_state") or "standing"
        act_name = (
            _resolve_body_action_name(str(st), library_blend=lib)
            or _resolve_body_action_name("idle", library_blend=lib)
        )
    if not act_name:
        if VERBOSE_PACKETS:
            print(f"[body_receiver] no Action for {raw!r} lib={lib!r}")
        return

    intensity = float(packet.get("intensity") or 0.75)
    speed = float(packet.get("speed") or (0.85 + 0.4 * intensity))
    speed = max(0.5, min(1.6, speed))
    eng = str(packet.get("engine") or "").lower()
    # MoMask / momask_* Action names: never loop (prevents two walk cycles per sentence)
    if eng == "momask" or str(act_name or "").lower().startswith("momask_"):
        loop = False
        packet = dict(packet)
        packet["loop"] = False
        packet["engine"] = "momask"
    else:
        loop = _is_loop_action(act_name, packet)
    # Audio-tied stop: prefer packet duration (speech length)
    duration = packet.get("duration")
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None
    if duration is not None and duration <= 0:
        duration = None

    # Shared audio clock from orchestrator (same wall time as speech start)
    play_t0 = packet.get("play_t0") or packet.get("sync_t0") or packet.get("audio_t0")
    try:
        play_t0 = float(play_t0) if play_t0 is not None else None
    except (TypeError, ValueError):
        play_t0 = None
    clip_fps = packet.get("clip_fps") or packet.get("fps") or 30.0
    try:
        clip_fps = float(clip_fps)
    except (TypeError, ValueError):
        clip_fps = 30.0
    if clip_fps < 1.0:
        clip_fps = 30.0

    # Session append bake DISABLED — it looped/bloat to 1000+ frames.
    # Play the generated Action as-is (MoMask/catalog real frame range only).

    # Restart only when the Action actually changes (or we were idle/resting).
    # force=True alone must NOT re-fire the same live clip (origin flash + double play).
    force = bool(packet.get("force") or packet.get("restart"))
    hard_restart = bool(packet.get("restart"))
    same_action_live = (
        act_name == _BODY_PLAY.get("action")
        and bool(_BODY_PLAY.get("live"))
        and not hard_restart
    )
    if same_action_live:
        # mid-stream refine only — never re-append / re-sample origin
        _BODY_PLAY["speed"] = speed
        if duration is not None:
            _BODY_PLAY["duration"] = duration
        if eng == "momask" or str(act_name or "").lower().startswith("momask_"):
            _BODY_PLAY["loop"] = False
            _BODY_PLAY["engine"] = "momask"
            if clip_fps > 22.0 or clip_fps < 8.0:
                clip_fps = 20.0
        else:
            _BODY_PLAY["loop"] = loop
        _BODY_PLAY["clip_fps"] = clip_fps
        if play_t0 is not None and abs(float(_BODY_PLAY.get("t0") or 0.0) - play_t0) > 0.08:
            _BODY_PLAY["t0"] = play_t0
        return

    need_new_clip = (
        act_name != _BODY_PLAY.get("action")
        or hard_restart
        or _BODY_PLAY.get("resting")
        or _BODY_PLAY.get("idle_hold")
        or not _BODY_PLAY.get("live")
    )
    if not need_new_clip:
        # force with same name but not live flag edge-case: refine
        _BODY_PLAY["speed"] = speed
        if duration is not None:
            _BODY_PLAY["duration"] = duration
        _BODY_PLAY["clip_fps"] = clip_fps
        return

    if need_new_clip:
        arm = _body_armature(packet.get("armature") or "SMPL-X_Armature")
        act = bpy.data.actions.get(act_name)

        # --- CRITICAL ORDER ---
        # 1) Last existing frame: live pose → Session_Timeline end → session prop
        # 2) Root offset so new Action (near origin) starts at that place
        # 3) Hold prior pose until first live tick
        # 4) Master-append with root_dx/dy + boundary blend (no Session bind for live)
        from mathutils import Vector as _V

        continue_root = bool(packet.get("continue_root", True))
        prev_idle = _BODY_PLAY.get("idle_pose") or _BODY_PLAY.get("rest_to") or {}

        last_pose, last_pel, last_src = _read_last_existing_frame(arm)
        print(
            f"[body_receiver] last-frame src={last_src!r} "
            f"pel={last_pel} bones={len(last_pose or {})}"
        )

        trans_from: dict = dict(last_pose) if last_pose else {}
        trans_pel = last_pel
        if not trans_from and prev_idle:
            trans_from = dict(prev_idle)
            if trans_pel is None:
                trans_pel = _pelvis_loc_tuple(_BODY_PLAY.get("rest_pelvis_loc"))

        curr_root = None
        if last_pel is not None:
            curr_root = _V(last_pel)
        if curr_root is None:
            curr_root = _session_root_get(arm)
        if curr_root is None and arm is not None:
            pb0 = arm.pose.bones.get("pelvis")
            if pb0 is not None:
                curr_root = pb0.location.copy()
        if curr_root is not None:
            _session_root_set(curr_root)
        if trans_pel is None:
            trans_pel = _pelvis_loc_tuple(curr_root)

        had_prior = bool(
            trans_from
            and (
                last_src in ("live_pose", "session_prop")
                or str(last_src).startswith("session_f")
                or _BODY_PLAY.get("live")
                or _BODY_PLAY.get("idle_hold")
                or _BODY_PLAY.get("resting")
                or (_SESSION_BODY.get("clips") or [])
            )
        )

        # Cache fcurves without binding Action; drop empty cache so layered Actions re-probe
        if act is not None:
            try:
                _ACTION_FCU_CACHE.pop(act.name, None)
                nfc = len(_get_action_fcu_targets(act) or [])
                print(f"[body_receiver] action {act.name!r} fast-eval channels={nfc}")
            except Exception as e:
                print(f"[body_receiver] fcurve cache skip: {e}")
        f0, f1 = _action_frame_range(act) if act else (1, 60)
        t0 = play_t0 if play_t0 is not None else time.time()

        # Place continuity is owned by _append_session_master via _place_xy_*.
        # Live path never applies a second root_offset (that caused origin→place grab).
        root_offset = None
        root_dx, root_dy = 0.0, 0.0
        blend_s = 0.0
        blend_frames = 0
        if eng == "momask" or str(act_name or "").lower().startswith("momask_"):
            loop = False
            if clip_fps > 22.0 or clip_fps < 8.0:
                clip_fps = 20.0
        # Hold prior pose visually while append runs (no origin flash)
        if arm is not None and curr_root is not None and had_prior:
            if trans_from:
                _apply_pose_dict(arm, trans_from, curr_root)
            elif prev_idle:
                _apply_pose_dict(arm, prev_idle, curr_root)
            _session_root_set(curr_root)
        print(
            f"[body_receiver] place SoT before append={_place_xy_get()} "
            f"prior_clips={len(_SESSION_BODY.get('clips') or [])}"
        )

        # Session append: ANY new Action must enter Session (place SoT).
        # Coordinator mid-stream spam sets append_timeline=False — that must NOT
        # skip append when the Action changes (catalog wave after walk → origin bug).
        same_live = (
            act_name == _BODY_PLAY.get("action")
            and bool(_BODY_PLAY.get("live"))
            and not hard_restart
        )
        if same_live:
            append_timeline = False
        else:
            append_timeline = True
        try:
            pkt_af = int(packet.get("action_frames") or 0)
        except (TypeError, ValueError):
            pkt_af = 0
        try:
            pkt_ml = int(packet.get("motion_length") or 0)
        except (TypeError, ValueError):
            pkt_ml = 0
        if pkt_af <= 0 and act is not None:
            try:
                a0, a1 = _action_frame_range(act)
                pkt_af = max(0, int(a1) - int(a0) + 1)
                if pkt_af > MAX_SESSION_CLIP_FRAMES:
                    pkt_af = 0
            except Exception:
                pkt_af = 0

        sess_f0, sess_f1 = f0, f1
        n_clips_before = len(_SESSION_BODY.get("clips") or [])
        if append_timeline and act is not None and arm is not None:
            try:
                clip_label = str(
                    packet.get("clip_label")
                    or packet.get("prompt")
                    or packet.get("text")
                    or packet.get("clip_id")
                    or act_name
                    or ""
                )
                sess_f0, sess_f1 = _append_session_master(
                    arm, act,
                    action_frames=pkt_af,
                    motion_length=pkt_ml,
                    root_dx=0.0,
                    root_dy=0.0,
                    pose_from=None,
                    pel_from=None,
                    blend_frames=0,
                    clip_label=clip_label,
                    clip_prompt=str(packet.get("prompt") or packet.get("humanml_prompt") or ""),
                    clip_text=str(packet.get("text") or ""),
                    speech_delay_s=float(packet.get("speech_delay_s") or 0.0),
                    speech_duration_s=float(packet.get("speech_duration_s") or packet.get("duration") or 0.0),
                    audio_path=str(packet.get("audio_path") or ""),
                    clip_fps=float(clip_fps or 20.0),
                    look=packet.get("look") if isinstance(packet.get("look"), dict) else {},
                    clip_policy=str(packet.get("clip_policy") or ""),
                    engine=str(packet.get("engine") or ""),
                )
            except Exception as e:
                print(f"[body_receiver] master append skip: {e}")
            # NEVER leave Session_Timeline bound for live — causes wrong-frame flash
            try:
                if arm.animation_data:
                    arm.animation_data.action = None
            except Exception:
                pass
            if curr_root is not None and trans_from:
                _apply_pose_dict(arm, trans_from, curr_root)
            elif curr_root is not None and prev_idle:
                _apply_pose_dict(arm, prev_idle, curr_root)

        keep_root = _pelvis_loc_tuple(curr_root)
        session_ok = (
            append_timeline
            and len(_SESSION_BODY.get("clips") or []) > n_clips_before
            and int(sess_f1) >= int(sess_f0) > 0
            and bpy.data.actions.get(_session_action_name()) is not None
        )
        # Single owner: live ALWAYS plays Session when append succeeded.
        # Never live-blend / never root_offset on top of Session keys.
        live_root_offset = None
        if session_ok:
            loop = False
            print(
                f"[body_receiver] LIVE from Session [{sess_f0}-{sess_f1}] "
                f"place={_place_xy_get()} (src={act_name!r})"
            )
        else:
            # Fallback: still place source Action via world-XY so catalog wave
            # cannot snap to Action origin if append was skipped.
            try:
                from mathutils import Vector as _V
                px, py = _place_xy_get()
                if abs(px) + abs(py) > 0.02 and act is not None and arm is not None:
                    sx, sy = 0.0, 0.0
                    ok0 = _apply_action_frame_fast(arm, act, float(f0))
                    if ok0:
                        w0 = _pelvis_world_xy(arm)
                        if w0 is not None:
                            sx, sy = w0
                    if arm.animation_data:
                        arm.animation_data.action = None
                    live_root_offset = _V((px - sx, py - sy, 0.0))
                    print(
                        f"[body_receiver] WARN append skipped — live source "
                        f"{act_name!r} with place offset "
                        f"({float(live_root_offset.x):.3f},{float(live_root_offset.y):.3f})"
                    )
                else:
                    print(
                        f"[body_receiver] WARN append failed — live fallback "
                        f"{act_name!r} at origin"
                    )
            except Exception as e:
                print(f"[body_receiver] WARN append failed fallback: {e}")

        # Movie Session install: append only — do not start wall-clock live
        # (live would race session_bind and leave the armature unbound).
        append_only = bool(
            packet.get("append_only")
            or packet.get("session_install")
            or (packet.get("live") is False)
        )
        _BODY_PLAY = {
            "action": act_name,
            "last_action": act_name,
            "t0": t0,
            "speed": speed,
            "loop": loop,
            "engine": eng or (
                "momask" if str(act_name).lower().startswith("momask_") else "clip_catalog"
            ),
            "f0": f0,
            "f1": f1,
            "clip_fps": clip_fps,
            "arm_name": packet.get("armature") or "SMPL-X_Armature",
            "pending_assign": (not append_only),
            "last_frame": None,
            "duration": duration,
            "root_offset": live_root_offset if not append_only else None,
            "continue_root": continue_root,
            "live": (not append_only),
            "idle_hold": False,
            "resting": False,
            "rest_t0": 0.0,
            "rest_duration": 0.55,
            "rest_from": {},
            "rest_to": prev_idle or {},
            "rest_pelvis_loc": keep_root,
            "idle_pose": prev_idle or {},
            "scrub_ready": bool(append_only and session_ok),
            "scrub_user_active": False,
            "fast_eval": True,
            "session_frame_start": sess_f0,
            "session_frame_end": sess_f1,
            "session_play": bool(session_ok),
            "trans_from": {},
            "trans_pel": None,
            "trans_t0": 0.0,
            "trans_duration": 0.0,
        }
        if keep_root is not None:
            _session_root_set(keep_root)
        print(
            f"[body_receiver] LIVE={act_name!r} session_play={bool(session_ok)} "
            f"MASTER=[{sess_f0}-{sess_f1}] place={_place_xy_get()} "
            f"clips={len(_SESSION_BODY.get('clips') or [])}"
        )


def _zero_mouth_shapes_if_idle() -> None:
    """If no recent viseme, force mouth/jaw shapes to 0 (body must not drive them)."""
    global _last_viseme_time
    if (time.time() - float(_last_viseme_time or 0.0)) < 0.35:
        return
    for obj_name in ("head_lod0_ORIGINAL", "teeth_ORIGINAL"):
        obj = bpy.data.objects.get(obj_name)
        if obj is None or obj.data is None or obj.data.shape_keys is None:
            continue
        for kb in obj.data.shape_keys.key_blocks:
            if kb.name == "Basis":
                continue
            ln = kb.name.lower()
            if ln.startswith(_MOUTH_SHAPE_PREFIXES):
                if kb.value != 0.0:
                    kb.value = 0.0
    arm = bpy.data.objects.get("SMPL-X_Armature")
    if arm and "jaw" in arm.pose.bones:
        j = arm.pose.bones["jaw"]
        if j.rotation_mode == "QUATERNION":
            j.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        else:
            j.rotation_euler = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# LOOK / SCENE (ground + HDRI or three-point + simple set)
# ---------------------------------------------------------------------------

LOOK_COLLECTION = "Look_Set"
LOOK_GROUND = "Look_Ground"


def _project_root_for_assets():
    """Best-effort project root so assets/looks/hdri resolves from Blender."""
    import os
    from pathlib import Path
    env = os.environ.get("PROJFACE_ROOT") or ""
    if env and (Path(env) / "assets" / "looks").is_dir():
        return Path(env)
    here = Path(__file__).resolve().parent
    if (here / "assets" / "looks").is_dir():
        return here
    for root in _project_roots():
        if (Path(root) / "assets" / "looks").is_dir():
            return Path(root)
    return here


def _look_collection(create: bool = True):
    col = bpy.data.collections.get(LOOK_COLLECTION)
    if col is None and create:
        col = bpy.data.collections.new(LOOK_COLLECTION)
        try:
            bpy.context.scene.collection.children.link(col)
        except Exception:
            pass
    return col


def _clear_look_set():
    col = bpy.data.collections.get(LOOK_COLLECTION)
    if col is None:
        return
    for obj in list(col.objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass


def _look_tex_dir(kind: str):
    """assets/looks/textures/<kind>/{diff,nor,rough}.jpg"""
    from pathlib import Path
    root = _project_root_for_assets()
    d = root / "assets" / "looks" / "textures" / str(kind)
    return d if d.is_dir() else None


def _look_mat(name: str, color, roughness: float = 0.85, *, tex: str = ""):
    """Solid or PBR textured material (Poly Haven maps when tex= grass|asphalt|brick|…)."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    c = tuple(color) + (1.0,) if len(color) == 3 else tuple(color)
    bsdf.inputs["Base Color"].default_value = c
    bsdf.inputs["Roughness"].default_value = float(roughness)
    tex_dir = _look_tex_dir(tex) if tex else None
    if tex_dir is not None:
        diff = tex_dir / "diff.jpg"
        nor = tex_dir / "nor.jpg"
        rough = tex_dir / "rough.jpg"
        uv = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (4.0, 4.0, 4.0)
        nt.links.new(uv.outputs["UV"], mapping.inputs["Vector"])
        if diff.is_file():
            img = bpy.data.images.load(str(diff), check_existing=True)
            img.colorspace_settings.name = "sRGB"
            texn = nt.nodes.new("ShaderNodeTexImage")
            texn.image = img
            nt.links.new(mapping.outputs["Vector"], texn.inputs["Vector"])
            nt.links.new(texn.outputs["Color"], bsdf.inputs["Base Color"])
        if rough.is_file():
            img = bpy.data.images.load(str(rough), check_existing=True)
            try:
                img.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
            texn = nt.nodes.new("ShaderNodeTexImage")
            texn.image = img
            nt.links.new(mapping.outputs["Vector"], texn.inputs["Vector"])
            nt.links.new(texn.outputs["Color"], bsdf.inputs["Roughness"])
        if nor.is_file():
            img = bpy.data.images.load(str(nor), check_existing=True)
            try:
                img.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
            texn = nt.nodes.new("ShaderNodeTexImage")
            texn.image = img
            nt.links.new(mapping.outputs["Vector"], texn.inputs["Vector"])
            nrm = nt.nodes.new("ShaderNodeNormalMap")
            nt.links.new(texn.outputs["Color"], nrm.inputs["Color"])
            nt.links.new(nrm.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _look_link_gltf(folder: str, name_prefix: str, loc, scale=1.0, rot_z: float = 0.0):
    """
    Import a Poly Haven glTF prop once, then duplicate instances into Look_Set.
    folder e.g. pine_tree → assets/looks/models/pine_tree/*.gltf
    """
    from pathlib import Path
    root = _project_root_for_assets()
    model_dir = root / "assets" / "looks" / "models" / folder
    if not model_dir.is_dir():
        return None
    gltfs = list(model_dir.glob("*.gltf")) + list(model_dir.glob("*.glb"))
    if not gltfs:
        return None
    src_path = gltfs[0]
    # Hidden template collection
    tmpl_name = f"_LookTemplate_{folder}"
    tmpl = bpy.data.objects.get(tmpl_name)
    if tmpl is None:
        before = set(bpy.data.objects.keys())
        try:
            bpy.ops.import_scene.gltf(filepath=str(src_path))
        except Exception as e:
            print(f"[look] gltf import fail {folder}: {e}")
            return None
        after = [n for n in bpy.data.objects.keys() if n not in before]
        if not after:
            return None
        # Parent all imported into an empty template
        tmpl = bpy.data.objects.new(tmpl_name, None)
        bpy.context.scene.collection.objects.link(tmpl)
        for n in after:
            ob = bpy.data.objects.get(n)
            if ob is None:
                continue
            try:
                ob.parent = tmpl
            except Exception:
                pass
        tmpl.hide_viewport = True
        tmpl.hide_render = True
        # Move template out of way
        tmpl.location = (0.0, 0.0, -50.0)
    # Instance
    col = _look_collection(True)
    inst_name = name_prefix
    # Remove prior instance with same name
    old = bpy.data.objects.get(inst_name)
    if old is not None:
        try:
            bpy.data.objects.remove(old, do_unlink=True)
        except Exception:
            pass
    inst = tmpl.copy()
    inst.name = inst_name
    inst.hide_viewport = False
    inst.hide_render = False
    col.objects.link(inst)
    # Also duplicate children hierarchy
    for child in list(tmpl.children):
        ch = child.copy()
        if child.data:
            ch.data = child.data  # share mesh data
        ch.parent = inst
        try:
            col.objects.link(ch)
        except Exception:
            try:
                bpy.context.scene.collection.objects.link(ch)
            except Exception:
                pass
    inst.location = loc
    try:
        s = float(scale)
        inst.scale = (s, s, s)
    except Exception:
        pass
    try:
        inst.rotation_euler[2] = float(rot_z)
    except Exception:
        pass
    return inst


def _look_cube(name: str, loc, scale, mat):
    col = _look_collection(True)
    obj = bpy.data.objects.get(name)
    if obj is None:
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        col.objects.link(obj)
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=2.0)
        bm.to_mesh(mesh)
        bm.free()
    obj.location = loc
    obj.scale = scale
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj


def _look_cylinder(name: str, loc, scale, mat):
    col = _look_collection(True)
    obj = bpy.data.objects.get(name)
    if obj is None:
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        col.objects.link(obj)
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=1.0, radius2=1.0, depth=2.0)
        bm.to_mesh(mesh)
        bm.free()
    obj.location = loc
    obj.scale = scale
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj


def _ensure_look_ground(size: float = 12.0, *, color=(0.12, 0.12, 0.13), mat_name="Look_Ground_Mat"):
    """Ground at z=0 — same plane as mesh soles. Color depends on location kit."""
    col = _look_collection(True)
    ground = bpy.data.objects.get(LOOK_GROUND)
    if ground is None:
        mesh = bpy.data.meshes.new(LOOK_GROUND)
        ground = bpy.data.objects.new(LOOK_GROUND, mesh)
        col.objects.link(ground)
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=max(1.0, float(size) * 0.5))
        bm.to_mesh(mesh)
        bm.free()
    ground.location = (0.0, 0.0, 0.0)
    mat = _look_mat(mat_name, color, 0.9)
    if ground.data.materials:
        ground.data.materials[0] = mat
    else:
        ground.data.materials.append(mat)
    try:
        ground.is_shadow_catcher = False
    except Exception:
        pass
    s = max(1.0, float(size) / 2.0)
    ground.scale = (s, s, 1.0)
    return ground


def _ensure_studio_cyc(size: float = 12.0):
    """Simple backdrop wall behind the character (−Y is camera)."""
    col = _look_collection(True)
    name = "Look_Cyc"
    cyc = bpy.data.objects.get(name)
    if cyc is None:
        mesh = bpy.data.meshes.new(name)
        cyc = bpy.data.objects.new(name, mesh)
        col.objects.link(cyc)
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.0)
        bm.to_mesh(mesh)
        bm.free()
    cyc.rotation_euler = (1.57079632679, 0.0, 0.0)  # stand upright
    cyc.location = (0.0, 4.0, float(size) * 0.35)
    cyc.scale = (float(size) * 0.6, float(size) * 0.45, 1.0)
    mat = bpy.data.materials.get("Look_Cyc_Mat")
    if mat is None:
        mat = bpy.data.materials.new("Look_Cyc_Mat")
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (0.22, 0.23, 0.26, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.95
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if cyc.data.materials:
        cyc.data.materials[0] = mat
    else:
        cyc.data.materials.append(mat)
    return cyc


def _ensure_interior_walls(size: float = 10.0):
    col = _look_collection(True)
    walls = []
    specs = [
        ("Look_Wall_Back", (0.0, 3.5, 1.5), (size * 0.5, 0.08, 1.5)),
        ("Look_Wall_L", (-size * 0.45, 0.5, 1.5), (0.08, size * 0.35, 1.5)),
        ("Look_Wall_R", (size * 0.45, 0.5, 1.5), (0.08, size * 0.35, 1.5)),
    ]
    mat = bpy.data.materials.get("Look_Wall_Mat")
    if mat is None:
        mat = bpy.data.materials.new("Look_Wall_Mat")
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (0.55, 0.52, 0.48, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.9
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    for name, loc, scale in specs:
        obj = bpy.data.objects.get(name)
        if obj is None:
            mesh = bpy.data.meshes.new(name)
            obj = bpy.data.objects.new(name, mesh)
            col.objects.link(obj)
            import bmesh
            bm = bmesh.new()
            bmesh.ops.create_cube(bm, size=2.0)
            bm.to_mesh(mesh)
            bm.free()
        obj.location = loc
        obj.scale = scale
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        walls.append(obj)
    return walls


def _look_ico(name: str, loc, scale, mat):
    col = _look_collection(True)
    obj = bpy.data.objects.get(name)
    if obj is None:
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        col.objects.link(obj)
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
        bm.to_mesh(mesh)
        bm.free()
    obj.location = loc
    obj.scale = scale
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj


def _build_set_exterior_park(size: float = 24.0):
    """Realistic park: PBR grass/path + real glTF trees/benches when assets exist."""
    _ensure_look_ground(size, color=(0.22, 0.48, 0.18), mat_name="Look_Grass_Mat")
    # Re-apply ground with grass texture if available
    g = bpy.data.objects.get(LOOK_GROUND)
    if g is not None:
        g.data.materials[0] = _look_mat("Look_Grass_Mat", (0.22, 0.48, 0.18), 0.9, tex="grass")
    path_mat = _look_mat("Look_Path_Mat", (0.42, 0.36, 0.28), 0.92, tex="concrete")
    _look_cube("Look_Park_Path", (0.0, -1.2, 0.015), (1.35, size * 0.38, 0.02), path_mat)

    tree_spots = (
        (-7.5, 5.5), (-5.5, 8.0), (6.5, 6.0), (8.0, 3.5), (-8.0, -5.0), (7.0, -6.5),
        (-4.5, -8.0), (4.5, 9.0), (-9.5, 1.5), (9.0, -2.0),
    )
    linked = 0
    for i, (x, y) in enumerate(tree_spots):
        folder = "pine_tree" if i % 2 == 0 else "island_tree"
        sc = 0.85 + (i % 3) * 0.12
        if _look_link_gltf(folder, f"Look_Tree_{i}", (x, y, 0.0), scale=sc, rot_z=i * 0.4):
            linked += 1
        else:
            bark = _look_mat("Look_Bark_Mat", (0.32, 0.2, 0.11), 0.88, tex="wood")
            leaf = _look_mat("Look_Leaf_A", (0.14, 0.42, 0.16), 0.75, tex="grass")
            _look_cylinder(f"Look_TreeTrunk_{i}", (x, y, 1.2), (0.25, 0.25, 1.2), bark)
            _look_ico(f"Look_TreeCrown_{i}", (x, y, 2.8), (1.5, 1.5, 1.35), leaf)
    print(f"[look] park trees gltf={linked}/{len(tree_spots)}")

    hedge = _look_mat("Look_Hedge_Mat", (0.12, 0.36, 0.14), 0.85, tex="grass")
    _look_cube("Look_Hedge_Back", (0.0, 10.5, 0.7), (10.0, 0.45, 0.7), hedge)
    _look_cube("Look_Hedge_L", (-10.5, 2.0, 0.55), (0.4, 6.0, 0.55), hedge)
    _look_cube("Look_Hedge_R", (10.5, 2.0, 0.55), (0.4, 6.0, 0.55), hedge)

    for bi, (bx, by) in enumerate(((-2.8, 3.2), (3.2, -3.8), (-3.5, -2.5))):
        if not _look_link_gltf("bench", f"Look_Bench_{bi}", (bx, by, 0.0), scale=1.0, rot_z=bi):
            bench = _look_mat("Look_Bench_Mat", (0.45, 0.28, 0.14), 0.65, tex="wood")
            _look_cube(f"Look_BenchSeat_{bi}", (bx, by, 0.42), (1.1, 0.32, 0.06), bench)
    return True


def _build_set_exterior_forest(size: float = 28.0):
    """Forest with real tree GLBs when present + dirt path."""
    _ensure_look_ground(size, color=(0.22, 0.28, 0.14), mat_name="Look_Dirt_Mat")
    g = bpy.data.objects.get(LOOK_GROUND)
    if g is not None:
        g.data.materials[0] = _look_mat("Look_Dirt_Mat", (0.22, 0.28, 0.14), 0.9, tex="concrete")
    path_mat = _look_mat("Look_Path_Mat", (0.3, 0.25, 0.18), 0.95, tex="concrete")
    _look_cube("Look_Forest_Path", (0.0, -0.5, 0.01), (1.0, size * 0.4, 0.02), path_mat)
    spots = [
        (-7, 3), (-5, 6), (-8, -2), (-6, -7), (6, 4), (8, 7), (5, -5), (7, -8),
        (-3, 8), (3, 9), (-9, 5), (9, -3), (-4, -9), (4, -9),
    ]
    for i, (x, y) in enumerate(spots):
        folder = "pine_tree" if i % 2 == 0 else "island_tree"
        if not _look_link_gltf(folder, f"Look_Tree_{i}", (x, y, 0.0), scale=1.0 + (i % 3) * 0.1, rot_z=i * 0.5):
            bark = _look_mat("Look_Bark_Mat", (0.25, 0.16, 0.08), 0.9, tex="wood")
            leaf = _look_mat("Look_Leaf_Mat", (0.08, 0.28, 0.1), 0.85, tex="grass")
            h = 1.1 + (i % 3) * 0.25
            _look_cylinder(f"Look_TreeTrunk_{i}", (x, y, h), (0.22, 0.22, h), bark)
            _look_ico(f"Look_TreeCrown_{i}", (x, y, h * 2.1), (1.4, 1.4, 1.2), leaf)
    return True


def _build_set_exterior_street(size: float = 22.0):
    """City street: PBR asphalt/sidewalk + real facade modules + lamps."""
    _ensure_look_ground(size, color=(0.25, 0.25, 0.26), mat_name="Look_Asphalt_Mat")
    g = bpy.data.objects.get(LOOK_GROUND)
    if g is not None:
        g.data.materials[0] = _look_mat("Look_Asphalt_Mat", (0.25, 0.25, 0.26), 0.85, tex="asphalt")
    walk = _look_mat("Look_Sidewalk_Mat", (0.45, 0.44, 0.42), 0.9, tex="concrete")
    _look_cube("Look_Sidewalk", (0.0, -2.0, 0.03), (size * 0.35, 1.5, 0.03), walk)
    # Real apartment facade modules when downloaded
    for i, x in enumerate((-7.0, -2.5, 2.5, 7.0)):
        if not _look_link_gltf("urban_facade", f"Look_Building_{i}", (x, 6.5, 0.0), scale=1.0, rot_z=3.1416):
            facade = _look_mat("Look_Facade_Mat", (0.55, 0.5, 0.45), 0.85, tex="brick")
            _look_cube(f"Look_Shop_{i}", (x, 5.5, 2.2), (1.8, 0.5, 2.2), facade)
    curb = _look_mat("Look_Curb_Mat", (0.5, 0.5, 0.48), 0.8, tex="concrete")
    _look_cube("Look_Curb", (0.0, -0.4, 0.08), (size * 0.35, 0.15, 0.08), curb)
    for i, x in enumerate((-5.0, 0.0, 5.0)):
        _look_link_gltf("street_lamp", f"Look_Lamp_{i}", (x, -3.2, 0.0), scale=1.0)
    return True


def _build_set_exterior_playground(size: float = 22.0):
    """Playground: textured grass + rubber pad + poles + real bench if present."""
    _ensure_look_ground(size, color=(0.2, 0.45, 0.22), mat_name="Look_Grass_Mat")
    g = bpy.data.objects.get(LOOK_GROUND)
    if g is not None:
        g.data.materials[0] = _look_mat("Look_Grass_Mat", (0.2, 0.45, 0.22), 0.9, tex="grass")
    rubber = _look_mat("Look_Play_Mat", (0.55, 0.2, 0.15), 0.7, tex="asphalt")
    _look_cube("Look_Play_Pad", (0.0, 0.5, 0.02), (4.0, 4.0, 0.02), rubber)
    metal = _look_mat("Look_Metal_Mat", (0.6, 0.62, 0.65), 0.4)
    _look_cylinder("Look_Swing_Pole_L", (-1.5, 1.0, 1.2), (0.08, 0.08, 1.2), metal)
    _look_cylinder("Look_Swing_Pole_R", (1.5, 1.0, 1.2), (0.08, 0.08, 1.2), metal)
    _look_cube("Look_Swing_Top", (0.0, 1.0, 2.4), (1.7, 0.08, 0.08), metal)
    if not _look_link_gltf("bench", "Look_Play_Bench", (-3.5, -2.0, 0.0), scale=1.0):
        wood = _look_mat("Look_Bench_Mat", (0.4, 0.25, 0.12), 0.7, tex="wood")
        _look_cube("Look_Play_Bench", (-3.5, -2.0, 0.28), (1.0, 0.28, 0.28), wood)
    return True


def _build_set_exterior_station(size: float = 26.0):
    """Station platform with PBR concrete + benches/lamps."""
    _ensure_look_ground(size, color=(0.3, 0.3, 0.32), mat_name="Look_Platform_Mat")
    g = bpy.data.objects.get(LOOK_GROUND)
    if g is not None:
        g.data.materials[0] = _look_mat("Look_Platform_Mat", (0.3, 0.3, 0.32), 0.8, tex="concrete")
    plat = _look_mat("Look_PlatTop_Mat", (0.4, 0.4, 0.42), 0.75, tex="concrete")
    _look_cube("Look_Platform", (0.0, 0.0, 0.15), (8.0, 2.5, 0.15), plat)
    roof = _look_mat("Look_Canopy_Mat", (0.35, 0.38, 0.4), 0.6, tex="metal" if False else "plaster")
    _look_cube("Look_Canopy", (0.0, 0.0, 3.2), (7.0, 2.2, 0.08), roof)
    for i, x in enumerate((-4.0, 0.0, 4.0)):
        _look_cylinder(f"Look_Canopy_Pole_{i}", (x, -1.8, 1.6), (0.12, 0.12, 1.6), roof)
    if not _look_link_gltf("bench", "Look_Station_Bench", (2.5, 1.2, 0.15), scale=1.0):
        bench = _look_mat("Look_Bench_Mat", (0.35, 0.35, 0.38), 0.5, tex="metal" if False else "wood")
        _look_cube("Look_Station_Bench", (2.5, 1.2, 0.55), (1.2, 0.35, 0.2), bench)
    _look_link_gltf("street_lamp", "Look_Station_Lamp", (-3.0, -2.5, 0.0), scale=1.0)
    return True


def _build_set_exterior_beach(size: float = 30.0):
    """Beach sand (PBR) + dunes."""
    _ensure_look_ground(size, color=(0.76, 0.68, 0.48), mat_name="Look_Sand_Mat")
    g = bpy.data.objects.get(LOOK_GROUND)
    if g is not None:
        g.data.materials[0] = _look_mat("Look_Sand_Mat", (0.76, 0.68, 0.48), 0.95, tex="sand")
    dune = _look_mat("Look_Dune_Mat", (0.7, 0.6, 0.4), 0.95, tex="sand")
    _look_cube("Look_Dune_1", (-8.0, 6.0, 0.4), (3.0, 1.5, 0.4), dune)
    _look_cube("Look_Dune_2", (7.0, 5.0, 0.35), (2.5, 1.2, 0.35), dune)
    return True


def _build_look_set_geometry(set_style: str, size: float) -> str:
    """Dispatch procedural set for ANY location kit (park/forest/street/…)."""
    style = str(set_style or "studio_cyc")
    if style in ("studio_cyc", "empty_floor"):
        _ensure_look_ground(size, color=(0.12, 0.12, 0.13))
        if style == "studio_cyc":
            _ensure_studio_cyc(size)
        return style
    if style == "interior_simple":
        _ensure_look_ground(size, color=(0.35, 0.32, 0.28), mat_name="Look_Floor_Mat")
        _ensure_interior_walls(size)
        return style
    if style == "exterior_park":
        _build_set_exterior_park(size)
        return style
    if style == "exterior_forest":
        _build_set_exterior_forest(size)
        return style
    if style == "exterior_street":
        _build_set_exterior_street(size)
        return style
    if style == "exterior_playground":
        _build_set_exterior_playground(size)
        return style
    if style == "exterior_station":
        _build_set_exterior_station(size)
        return style
    if style == "exterior_beach":
        _build_set_exterior_beach(size)
        return style
    # Generic outdoor fallback
    _ensure_look_ground(size, color=(0.2, 0.35, 0.18), mat_name="Look_Grass_Mat")
    _ensure_studio_cyc(size)
    return "exterior_ground"


def _set_world_hdri(hdri_path: str, strength: float = 0.85) -> bool:
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("LookWorld")
        bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = float(strength)
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    try:
        img = bpy.data.images.load(hdri_path, check_existing=True)
        env.image = img
    except Exception as e:
        print(f"[look] HDRI load failed ({e}) — solid world")
        bg.inputs["Color"].default_value = (0.04, 0.045, 0.055, 1.0)
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
        return False
    mapn = nt.nodes.new("ShaderNodeMapping")
    texcoord = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(texcoord.outputs["Generated"], mapn.inputs["Vector"])
    nt.links.new(mapn.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return True


def _set_world_solid(strength: float = 0.2, color=(0.04, 0.045, 0.055)):
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("LookWorld")
        bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = float(strength)
    bg.inputs["Color"].default_value = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def _ensure_area_light(name: str, loc, rot, energy: float, color, size: float = 2.5):
    col = _look_collection(True)
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "LIGHT":
        if obj is not None:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass
        data = bpy.data.lights.new(name, type="AREA")
        obj = bpy.data.objects.new(name, data)
        col.objects.link(obj)
    obj.data.type = "AREA"
    try:
        obj.data.shape = "RECTANGLE"
        obj.data.size = float(size)
        obj.data.size_y = float(size) * 0.7
    except Exception:
        pass
    obj.data.energy = float(energy)
    obj.data.color = (float(color[0]), float(color[1]), float(color[2]))
    obj.location = loc
    obj.rotation_euler = rot
    return obj


def _ensure_sun_light(name: str, energy: float, color, rot):
    col = _look_collection(True)
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "LIGHT":
        if obj is not None:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass
        data = bpy.data.lights.new(name, type="SUN")
        obj = bpy.data.objects.new(name, data)
        col.objects.link(obj)
    obj.data.type = "SUN"
    obj.data.energy = float(energy)
    obj.data.color = (float(color[0]), float(color[1]), float(color[2]))
    obj.rotation_euler = rot
    return obj


def _hide_retarget_helpers():
    for obj in bpy.data.objects:
        n = obj.name or ""
        if n.endswith("_RSL_H") or "_RSL_H" in n or n.startswith("_BVH"):
            try:
                obj.hide_render = True
                obj.hide_viewport = True
            except Exception:
                pass


def _look_ground_world_z() -> float:
    g = bpy.data.objects.get(LOOK_GROUND)
    if g is None:
        return 0.0
    try:
        bpy.context.view_layer.update()
        return float(g.matrix_world.translation.z)
    except Exception:
        return 0.0


def _body_mesh_object(arm=None):
    for n in ("BodyMesh", "SMPLX", "smplx", "body"):
        o = bpy.data.objects.get(n)
        if o is not None and getattr(o, "type", "") == "MESH":
            return o
    if arm is None:
        arm = _body_armature()
    if arm is not None:
        for c in getattr(arm, "children", []) or []:
            if getattr(c, "type", "") == "MESH":
                return c
    return None


def _mesh_min_world_z(obj) -> float | None:
    if obj is None:
        return None
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        if not me.vertices:
            return None
        zs = [(ev.matrix_world @ v.co).z for v in me.vertices]
        return float(min(zs)) if zs else None
    except Exception:
        return None
    finally:
        try:
            ev.to_mesh_clear()
        except Exception:
            pass


def _ensure_clip_on_ground(arm, f0: int, f1: int) -> None:
    """Constant world-Z lift so mesh sole clears Look_Ground — no per-frame re-plant.

    Retarget already ran apply_mesh_sole_clearance on the source Action. Session
    append only shifts XY; a stepped second plant caused mid-clip pops. If the
    plane is ~0 and the lowest sample is already clear, this is a no-op.
    """
    if arm is None or "pelvis" not in arm.pose.bones:
        return
    mesh = _body_mesh_object(arm)
    if mesh is None:
        return
    act = bpy.data.actions.get(_session_action_name())
    if act is None:
        return
    if not arm.animation_data:
        arm.animation_data_create()
    prev_act = arm.animation_data.action
    arm.animation_data.action = act
    try:
        slots = getattr(arm.animation_data, "action_suitable_slots", None)
        if slots and len(slots) > 0:
            arm.animation_data.action_slot = slots[0]
    except Exception:
        pass
    plane = float(_look_ground_world_z())
    pb = arm.pose.bones["pelvis"]
    f0, f1 = int(f0), int(f1)
    # Sample a few frames for the worst sink (constant lift fixes all equally)
    skins = []
    for f in (f0, (f0 + f1) // 2, f1):
        bpy.context.scene.frame_set(int(f))
        bpy.context.view_layer.update()
        skin = _mesh_min_world_z(mesh)
        if skin is not None:
            skins.append(float(skin))
    if not skins:
        arm.animation_data.action = prev_act
        return
    lo = min(skins)
    err = plane - lo
    if err <= 0.002:
        print(
            f"[body_receiver] ground OK plane={plane:.4f} skin_lo={lo:.4f} "
            f"frames={f0}-{f1} (skip constant lift)"
        )
        arm.animation_data.action = None
        return
    lift = min(float(err), 0.25)  # one constant; retarget should need little
    # Apply same world-Z lift to pelvis location keys across the clip
    for f in range(f0, f1 + 1):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        world = arm.matrix_world @ pb.matrix
        world.translation.z += lift
        pose_mat = arm.matrix_world.inverted() @ world
        local = arm.convert_space(
            pose_bone=pb, matrix=pose_mat, from_space="POSE", to_space="LOCAL"
        )
        loc, _, _ = local.decompose()
        pb.location = loc
        pb.keyframe_insert("location", frame=f)
    bpy.context.scene.frame_set(f0)
    bpy.context.view_layer.update()
    arm.animation_data.action = None
    print(
        f"[body_receiver] ground constant lift={lift:.4f} plane={plane:.4f} "
        f"skin_lo_was={lo:.4f} frames={f0}-{f1}"
    )


def _plant_clip_to_look_ground(arm, f0: int, f1: int) -> None:
    """Back-compat alias — prefer constant lift (no stepped mid-clip plant)."""
    _ensure_clip_on_ground(arm, f0, f1)


# ---------------------------------------------------------------------------
# Wardrobe (outfit presets) + Cast extras (hero + body-only NPCs)
# ---------------------------------------------------------------------------

_WARDROBE_PRESETS = (
    "hero_default",
    "casual_01",
    "formal_01",
    "hoodie_01",
    "worker_01",
)

# Preferred real mesh files under assets/cast/clothes/<outfit_id>/
_WARDROBE_ASSET_FILES = {
    "casual_01": ("casual_character.glb", "Casual_Male.fbx", "Casual2_Male.fbx"),
    "formal_01": ("business_man.glb", "Suit_Male.fbx"),
    "hoodie_01": ("hoodie_character.glb",),
    "worker_01": ("worker_character.glb", "Worker_Male.fbx"),
}

_WARDROBE_COLORS = {
    # shirt, pants RGB (fallback proxies only)
    "hero_default": ((0.55, 0.55, 0.58), (0.22, 0.22, 0.25)),
    "casual_01": ((0.25, 0.45, 0.75), (0.18, 0.28, 0.48)),
    "formal_01": ((0.08, 0.08, 0.10), (0.06, 0.06, 0.08)),
    "hoodie_01": ((0.35, 0.35, 0.40), (0.18, 0.18, 0.22)),
    "worker_01": ((0.55, 0.40, 0.20), (0.22, 0.22, 0.25)),
}


def _wardrobe_root_collection(create: bool = True):
    name = "Wardrobe"
    col = bpy.data.collections.get(name)
    if col is None and create:
        col = bpy.data.collections.new(name)
        try:
            bpy.context.scene.collection.children.link(col)
        except Exception:
            pass
        char = bpy.data.collections.get("Character_FullBody")
        if char is not None:
            try:
                char.children.link(col)
            except Exception:
                pass
    return col


def _wardrobe_preset_collection(outfit_id: str, create: bool = True):
    oid = str(outfit_id or "hero_default").strip() or "hero_default"
    name = f"Wardrobe_{oid}"
    col = bpy.data.collections.get(name)
    if col is None and create:
        col = bpy.data.collections.new(name)
        root = _wardrobe_root_collection(True)
        if root is not None:
            try:
                root.children.link(col)
            except Exception:
                try:
                    bpy.context.scene.collection.children.link(col)
                except Exception:
                    pass
    return col


def _cast_fabric_dir(kind: str):
    """assets/cast/fabrics/<kind>/{diff,nor,rough}.jpg from download_cast_assets.py"""
    for root in _project_roots():
        d = root / "assets" / "cast" / "fabrics" / str(kind)
        if (d / "diff.jpg").is_file():
            return d
    return None


def _wardrobe_fabric_for(outfit_id: str, part: str) -> str:
    oid = str(outfit_id or "hero_default")
    if oid == "formal_01":
        return "suit_wool"
    if part == "pants":
        return "denim"
    return "jersey"


def _wardrobe_mat(name: str, color, *, fabric_kind: str = ""):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
    try:
        nt = mat.node_tree
        bsdf = nt.nodes.get("Principled BSDF")
        if not bsdf:
            return mat
        bsdf.inputs["Base Color"].default_value = (
            float(color[0]), float(color[1]), float(color[2]), 1.0,
        )
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.55
        fdir = _cast_fabric_dir(fabric_kind) if fabric_kind else None
        if fdir is not None:
            diff = fdir / "diff.jpg"
            if diff.is_file():
                tex = nt.nodes.get("WardrobeDiff") or nt.nodes.new("ShaderNodeTexImage")
                tex.name = "WardrobeDiff"
                tex.location = (-400, 200)
                try:
                    img = bpy.data.images.load(str(diff), check_existing=True)
                    tex.image = img
                except Exception:
                    pass
                if tex.image:
                    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
            rough = fdir / "rough.jpg"
            if rough.is_file() and "Roughness" in bsdf.inputs:
                rtex = nt.nodes.get("WardrobeRough") or nt.nodes.new("ShaderNodeTexImage")
                rtex.name = "WardrobeRough"
                rtex.location = (-400, -50)
                try:
                    rtex.image = bpy.data.images.load(str(rough), check_existing=True)
                    rtex.image.colorspace_settings.name = "Non-Color"
                except Exception:
                    pass
                if rtex.image:
                    nt.links.new(rtex.outputs["Color"], bsdf.inputs["Roughness"])
            nor = fdir / "nor.jpg"
            if nor.is_file() and "Normal" in bsdf.inputs:
                ntex = nt.nodes.get("WardrobeNor") or nt.nodes.new("ShaderNodeTexImage")
                ntex.name = "WardrobeNor"
                ntex.location = (-400, -280)
                nmap = nt.nodes.get("WardrobeNMap") or nt.nodes.new("ShaderNodeNormalMap")
                nmap.name = "WardrobeNMap"
                nmap.location = (-180, -280)
                try:
                    ntex.image = bpy.data.images.load(str(nor), check_existing=True)
                    ntex.image.colorspace_settings.name = "Non-Color"
                except Exception:
                    pass
                if ntex.image:
                    nt.links.new(ntex.outputs["Color"], nmap.inputs["Color"])
                    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    except Exception:
        pass
    return mat


def _find_pose_bone_name(arm, candidates):
    if arm is None or arm.type != "ARMATURE":
        return None
    names = {b.name for b in arm.data.bones}
    for c in candidates:
        if c in names:
            return c
    low = {b.name.lower(): b.name for b in arm.data.bones}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _ensure_wardrobe_proxy(outfit_id: str, part: str, size, loc_local, bone_name: str | None, color):
    """Create a simple clothing proxy mesh parented to a bone (demo outfits)."""
    oid = str(outfit_id or "hero_default")
    obj_name = f"Cloth_{oid}_{part}"
    obj = bpy.data.objects.get(obj_name)
    arm = _body_armature()
    if obj is None:
        mesh = bpy.data.meshes.new(obj_name + "_Mesh")
        obj = bpy.data.objects.new(obj_name, mesh)
        # Unit cube then scale
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(mesh)
        bm.free()
        col = _wardrobe_preset_collection(oid, True)
        if col is not None:
            try:
                col.objects.link(obj)
            except Exception:
                bpy.context.scene.collection.objects.link(obj)
        else:
            bpy.context.scene.collection.objects.link(obj)
    obj.scale = (float(size[0]), float(size[1]), float(size[2]))
    obj.location = (float(loc_local[0]), float(loc_local[1]), float(loc_local[2]))
    mat = _wardrobe_mat(f"Mat_{oid}_{part}", color)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    if arm is not None:
        try:
            obj.parent = arm
            if bone_name and bone_name in arm.pose.bones:
                obj.parent_type = "BONE"
                obj.parent_bone = bone_name
            else:
                obj.parent_type = "OBJECT"
        except Exception as e:
            print(f"[wardrobe] parent {obj_name}: {e}")
    obj["wardrobe_id"] = oid
    obj["wardrobe_part"] = part
    return obj


def _wardrobe_clothes_dir(outfit_id: str):
    oid = str(outfit_id or "").strip()
    for root in _project_roots():
        d = root / "assets" / "cast" / "clothes" / oid
        if d.is_dir():
            return d
    return None


def _wardrobe_find_asset(outfit_id: str):
    """Return path to preferred GLB/FBX clothing mesh, or None."""
    d = _wardrobe_clothes_dir(outfit_id)
    if d is None:
        return None
    preferred = _WARDROBE_ASSET_FILES.get(str(outfit_id), ())
    for name in preferred:
        p = d / name
        if p.is_file() and p.stat().st_size > 1000:
            return p
    # Any glb/fbx in folder
    for pat in ("*.glb", "*.gltf", "*.fbx"):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    return None


def _hide_box_proxies(outfit_id: str = "") -> None:
    """Always hide cube Cloth_* proxies — they are fallback only."""
    for obj in list(bpy.data.objects):
        if not obj.name.startswith("Cloth_"):
            continue
        if outfit_id and str(obj.get("wardrobe_id") or "") not in ("", outfit_id):
            pass
        try:
            obj.hide_set(True)
            obj.hide_render = True
            obj.hide_viewport = True
        except Exception:
            pass


def _hero_height_m(arm) -> float:
    if arm is None:
        return 1.70
    try:
        # Prefer mesh body bounds
        body = bpy.data.objects.get("BodyMesh")
        if body is not None and body.type == "MESH":
            zs = [(body.matrix_world @ v.co).z for v in body.data.vertices]
            if zs:
                return max(0.8, float(max(zs) - min(zs)))
    except Exception:
        pass
    try:
        return max(0.8, float(arm.dimensions.z) or 1.70)
    except Exception:
        return 1.70


def _ensure_wardrobe_asset(outfit_id: str):
    """
    Import real clothing GLB/FBX from assets/cast/clothes/<id>/ into Wardrobe_<id>.
    Returns root empty name, or None if no asset on disk.
    """
    from pathlib import Path

    oid = str(outfit_id or "casual_01")
    if oid == "hero_default":
        return None
    root_name = f"Outfit_{oid}"
    existing = bpy.data.objects.get(root_name)
    if existing is not None:
        return root_name

    asset = _wardrobe_find_asset(oid)
    if asset is None:
        print(f"[wardrobe] no mesh asset for {oid} under assets/cast/clothes/")
        return None

    col = _wardrobe_preset_collection(oid, True)
    before = set(bpy.data.objects.keys())
    path = Path(asset)
    try:
        if path.suffix.lower() in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=str(path))
        elif path.suffix.lower() == ".fbx":
            bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=True)
        else:
            print(f"[wardrobe] unsupported format {path.suffix}")
            return None
    except Exception as e:
        print(f"[wardrobe] import failed {path.name}: {e}")
        return None

    imported = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]
    if not imported:
        print(f"[wardrobe] import produced no objects: {path.name}")
        return None

    # Root empty holds the outfit
    root = bpy.data.objects.new(root_name, None)
    root.empty_display_type = "PLAIN_AXES"
    root["wardrobe_id"] = oid
    root["wardrobe_asset"] = path.name
    root["wardrobe_kind"] = "mesh_asset"
    if col is not None:
        try:
            col.objects.link(root)
        except Exception:
            bpy.context.scene.collection.objects.link(root)
    else:
        bpy.context.scene.collection.objects.link(root)

    # Parent top-level imported objects (usually CharacterArmature) under Outfit_*
    for obj in imported:
        try:
            if obj.parent is None or obj.parent not in imported:
                # Keep relative transform when re-parenting
                mw = obj.matrix_world.copy()
                obj.parent = root
                obj.matrix_world = mw
        except Exception:
            pass
        try:
            if col is not None and obj.name not in col.objects:
                for c in list(obj.users_collection):
                    try:
                        c.objects.unlink(obj)
                    except Exception:
                        pass
                col.objects.link(obj)
        except Exception:
            pass
        try:
            obj["wardrobe_id"] = oid
            obj["wardrobe_kind"] = "mesh_asset"
        except Exception:
            pass
        # Tag whole hierarchy so show/hide works even if nesting is deep
        try:
            stack = list(getattr(obj, "children_recursive", None) or obj.children)
            for ch in stack:
                ch["wardrobe_id"] = oid
                ch["wardrobe_kind"] = "mesh_asset"
        except Exception:
            pass

    # Scale to hero height and snap to armature
    arm = _body_armature()
    try:
        # Compute imported height from mesh bounds
        zs = []
        for obj in imported:
            if obj.type != "MESH":
                continue
            for v in obj.bound_box:
                ws = obj.matrix_world @ Vector((v[0], v[1], v[2]))
                zs.append(float(ws.z))
        src_h = (max(zs) - min(zs)) if zs else 1.7
        tgt_h = _hero_height_m(arm)
        scale = (tgt_h / src_h) if src_h > 0.05 else 1.0
        # Quaternius packs are often ~1 unit; clamp sane range
        scale = max(0.4, min(2.5, float(scale)))
        root.scale = (scale, scale, scale)
        if arm is not None:
            root.location = arm.matrix_world.translation.copy()
            # Keep feet near ground: shift so min z ≈ 0 after scale
            bpy.context.view_layer.update()
            zs2 = []
            for obj in imported:
                if obj.type != "MESH":
                    continue
                for v in obj.bound_box:
                    ws = obj.matrix_world @ Vector((v[0], v[1], v[2]))
                    zs2.append(float(ws.z))
            if zs2:
                root.location.z -= min(zs2)
            root.parent = arm
            root.parent_type = "OBJECT"
    except Exception as e:
        print(f"[wardrobe] align {oid}: {e}")

    print(f"[wardrobe] loaded mesh asset {path.name} → {root_name}")
    return root_name


def _set_body_mesh_visible(visible: bool) -> None:
    """Hide/show BodyMesh when a full-character outfit mesh is worn."""
    body = bpy.data.objects.get("BodyMesh")
    if body is None:
        return
    try:
        body.hide_set(not visible)
        body.hide_render = not visible
        body.hide_viewport = not visible
    except Exception:
        pass


def _ensure_wardrobe_presets() -> list:
    """
    Prefer real clothes from assets/cast/clothes/.
    Only create box proxies if NO mesh asset exists for that outfit.
    """
    created = []
    for oid in _WARDROBE_PRESETS:
        if oid == "hero_default":
            created.append(oid)
            continue
        root = _ensure_wardrobe_asset(oid)
        if root:
            created.append(oid)
            continue
        # Fallback proxies only when download_cast_assets clothes are missing
        arm = _body_armature()
        spine = _find_pose_bone_name(arm, ("spine3", "spine2", "Spine3", "Spine2", "chest", "torso"))
        pelvis = _find_pose_bone_name(arm, ("pelvis", "Pelvis", "hips", "root"))
        colors = _WARDROBE_COLORS.get(oid, _WARDROBE_COLORS["hero_default"])
        shirt_c, pants_c = colors
        shirt_size, pants_size = (0.40, 0.26, 0.50), (0.36, 0.24, 0.68)
        _ensure_wardrobe_proxy(oid, "shirt", shirt_size, (0.0, 0.02, 0.05), spine, shirt_c)
        _ensure_wardrobe_proxy(oid, "pants", pants_size, (0.0, 0.0, -0.15), pelvis, pants_c)
        print(f"[wardrobe] WARN {oid}: using box proxies (no GLB/FBX in assets/cast/clothes/{oid})")
        created.append(oid)
    return created


def _wardrobe_is_full_character(outfit_id: str) -> bool:
    """True if asset is a whole Quaternius-style body (not a garment on our hero)."""
    oid = str(outfit_id or "")
    root = bpy.data.objects.get(f"Outfit_{oid}")
    if root is None:
        return False
    names = []
    try:
        stack = [root] + list(getattr(root, "children_recursive", []) or list(root.children))
    except Exception:
        stack = [root]
    for obj in bpy.data.objects:
        if str(obj.get("wardrobe_id") or "") != oid:
            continue
        names.append(obj.name.lower())
        if obj.type == "ARMATURE" and "character" in obj.name.lower():
            return True
    joined = " ".join(names)
    # Full packs ship Body+Head+Legs+Feet as a second person
    hits = sum(1 for tok in ("_body", "_head", "_legs", "_feet", "suit_body", "casual2_body") if tok in joined)
    return hits >= 2


def _verify_wardrobe_worn(outfit_id: str) -> dict:
    """
    Gate for MP4 / demo: only pass if clothes look like garments on OUR hero.
    Full-character swaps (second body/head) FAIL — those belong on NPCs, not hero.
    """
    from face_agents.cast_schema import normalize_wardrobe_id

    oid = normalize_wardrobe_id(outfit_id)
    out = {"ok": False, "outfit_id": oid, "reason": "", "kind": "none"}
    if oid in ("", "hero_default"):
        out["ok"] = True
        out["kind"] = "none"
        out["reason"] = "hero_default (no extra clothes)"
        return out
    root = bpy.data.objects.get(f"Outfit_{oid}")
    if root is None:
        out["reason"] = "outfit root missing"
        return out
    if _wardrobe_is_full_character(oid):
        out["kind"] = "full_character"
        out["reason"] = (
            "asset is a full second character (body+head), not wearable clothes on SMPL-X — "
            "skip for hero; use as NPC instead"
        )
        return out
    # Garment path (future): must be visible and near hero
    arm = _body_armature()
    if arm is None:
        out["reason"] = "no hero armature"
        return out
    try:
        dist = (root.matrix_world.translation - arm.matrix_world.translation).length
    except Exception:
        dist = 999.0
    if dist > 2.5:
        out["reason"] = f"outfit too far from hero ({dist:.2f}m)"
        return out
    visible = False
    for obj in bpy.data.objects:
        if str(obj.get("wardrobe_id") or "") != oid:
            continue
        if obj.type == "MESH" and not obj.hide_get() and not obj.hide_render:
            visible = True
            break
    if not visible:
        out["reason"] = "no visible garment meshes"
        return out
    out["ok"] = True
    out["kind"] = "garment"
    out["reason"] = "garment visible and near hero"
    return out


def _apply_wardrobe(outfit_id: str, *, actor: str = "hero", require_fit: bool = True) -> str:
    """
    Apply outfit: real mesh from assets/cast/clothes when available.
    If require_fit and asset is a full second character (not clothes), revert to hero_default.
    Does not touch face blendshape meshes.
    """
    from face_agents.cast_schema import normalize_wardrobe_id

    oid = normalize_wardrobe_id(outfit_id)
    _ensure_wardrobe_presets()

    # Detect whether selected outfit has a real mesh root
    has_mesh = bpy.data.objects.get(f"Outfit_{oid}") is not None and oid != "hero_default"

    # Hide all wardrobe collections / outfit roots, then show target
    for col in list(bpy.data.collections):
        if not col.name.startswith("Wardrobe_"):
            continue
        show = col.name == f"Wardrobe_{oid}"
        try:
            col.hide_viewport = not show
            col.hide_render = not show
        except Exception:
            pass
        for obj in col.objects:
            try:
                # Never show cube proxies if this outfit has a real mesh
                if obj.name.startswith("Cloth_") and has_mesh:
                    obj.hide_set(True)
                    obj.hide_render = True
                    obj.hide_viewport = True
                    continue
                obj.hide_set(not show)
                obj.hide_render = not show
                obj.hide_viewport = not show
            except Exception:
                pass

    # Toggle every object tagged with wardrobe_id (meshes under CharacterArmature too)
    for obj in bpy.data.objects:
        wid = str(obj.get("wardrobe_id") or "")
        if not wid:
            if obj.name.startswith("Outfit_"):
                show = obj.name == f"Outfit_{oid}"
                try:
                    obj.hide_set(not show)
                    obj.hide_render = not show
                    obj.hide_viewport = not show
                except Exception:
                    pass
            continue
        if obj.name.startswith("Cloth_"):
            continue  # handled / always hidden when mesh exists
        show = (wid == oid) and (oid != "hero_default")
        try:
            obj.hide_set(not show)
            obj.hide_render = not show
            obj.hide_viewport = not show
        except Exception:
            pass

    # Prefer mesh clothes → hide boxes always when any Outfit_* exists for this id
    if has_mesh:
        _hide_box_proxies(oid)
        _set_body_mesh_visible(False)  # full-character outfit covers body; keep face head
    else:
        _set_body_mesh_visible(True)
        # hero_default: hide all outfits + boxes
        if oid == "hero_default":
            _hide_box_proxies()
            for obj in bpy.data.objects:
                if obj.name.startswith("Outfit_") or str(obj.get("wardrobe_kind") or "") == "mesh_asset":
                    try:
                        obj.hide_set(True)
                        obj.hide_render = True
                    except Exception:
                        pass

    src = "mesh" if has_mesh else ("body" if oid == "hero_default" else "proxy")
    print(f"[wardrobe] actor={actor} outfit={oid} source={src}")

    if require_fit and oid != "hero_default":
        ver = _verify_wardrobe_worn(oid)
        if not ver.get("ok"):
            print(f"[wardrobe] FIT FAIL → hero_default ({ver.get('reason')})")
            # Hide the rejected outfit and restore body
            return _apply_wardrobe("hero_default", actor=actor, require_fit=False)
        print(f"[wardrobe] FIT OK kind={ver.get('kind')}")
    return oid


def _apply_wardrobe_packet(packet: dict) -> None:
    op = str(packet.get("op") or "apply").lower()
    if op == "list":
        print(f"[wardrobe] presets={list(_WARDROBE_PRESETS)}")
        return
    if op == "verify":
        oid = packet.get("outfit_id") or packet.get("wardrobe_id") or "hero_default"
        print("[wardrobe] verify", _verify_wardrobe_worn(str(oid)))
        return
    oid = packet.get("outfit_id") or packet.get("wardrobe_id") or "hero_default"
    actor = str(packet.get("actor") or "hero")
    _apply_wardrobe(str(oid), actor=actor, require_fit=True)


def _cast_extras_collection(create: bool = True):
    name = "Cast_Extras"
    col = bpy.data.collections.get(name)
    if col is None and create:
        col = bpy.data.collections.new(name)
        try:
            bpy.context.scene.collection.children.link(col)
        except Exception:
            pass
    return col


def _clear_cast_extras() -> int:
    col = bpy.data.collections.get("Cast_Extras")
    removed = 0
    if col is None:
        return 0
    for obj in list(col.objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
        except Exception:
            pass
    # Also remove orphan NPC armatures by naming convention
    for obj in list(bpy.data.objects):
        if obj.name.startswith("NPC_") or obj.name.startswith("Extra_"):
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
            except Exception:
                pass
    print(f"[cast] cleared extras ({removed} objects)")
    return removed


def _extras_spawn_slots(preset: str, count: int):
    """World XY slots around the set (v1: fixed offsets, kit-agnostic)."""
    preset = str(preset or "sidewalk").lower()
    n = max(0, min(8, int(count or 0)))
    if n <= 0:
        return []
    if preset == "park_path":
        base = [(-2.5, 3.0, 0.0), (2.2, 4.0, 1.2), (-1.0, 5.5, -0.4),
                (3.5, 2.5, 2.0), (-3.5, 4.5, 0.8), (0.5, 6.5, -1.0)]
    elif preset == "plaza":
        base = [(-3.0, -2.0, 0.5), (3.0, -2.5, -0.5), (-2.5, 2.5, 2.5),
                (2.8, 3.0, -2.2), (0.0, -3.5, 3.1), (-4.0, 0.5, 1.0)]
    else:  # sidewalk / default
        base = [(-3.5, 1.5, 1.6), (3.5, 1.2, -1.6), (-4.0, 3.0, 0.2),
                (4.0, 2.8, 3.0), (-2.8, 4.5, -2.5), (2.5, 5.0, 1.0)]
    return base[:n]


def _npc_glb_candidates() -> list:
    """Prefer Kenney blocky GLBs, then single kenney_character.glb."""
    out = []
    for root in _project_roots():
        blocky = root / "assets" / "cast" / "npcs" / "kenney_blocky" / "Models" / "GLB format"
        if blocky.is_dir():
            for p in sorted(blocky.glob("character-*.glb")):
                out.append(p)
        single = root / "assets" / "cast" / "npcs" / "kenney_character.glb"
        if single.is_file():
            out.append(single)
        if out:
            break
    return out


def _import_npc_instance(glb_path, name: str, loc, yaw: float, col):
    """Import a CC0 NPC glTF/GLB once and place a copy in Cast_Extras."""
    from pathlib import Path
    path = Path(glb_path)
    if not path.is_file():
        return None
    # Template empty holds linked imports; instance is duplicated into Cast_Extras
    tpl_name = f"_NpcTemplate_{path.stem}"
    tpl = bpy.data.objects.get(tpl_name)
    if tpl is None:
        before = set(bpy.data.objects.keys())
        try:
            bpy.ops.import_scene.gltf(filepath=str(path))
        except Exception as e:
            print(f"[cast] gltf import fail {path.name}: {e}")
            return None
        after = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]
        if not after:
            return None
        # Parent roots under a template empty
        tpl = bpy.data.objects.new(tpl_name, None)
        bpy.context.scene.collection.objects.link(tpl)
        for obj in after:
            try:
                if obj.parent is None:
                    obj.parent = tpl
            except Exception:
                pass
            try:
                obj.hide_set(True)
                obj.hide_render = True
            except Exception:
                pass
        tpl.hide_set(True)
        tpl.hide_render = True
        tpl["npc_source"] = str(path)
    # Duplicate template hierarchy
    roots = [o for o in bpy.data.objects if o.parent == tpl]
    if not roots:
        roots = [tpl]
    dup_roots = []
    for src in roots:
        try:
            new = src.copy()
            if src.data:
                new.data = src.data.copy() if hasattr(src.data, "copy") else src.data
            col.objects.link(new) if col is not None else bpy.context.scene.collection.objects.link(new)
            new.hide_set(False)
            new.hide_render = False
            new.hide_viewport = False
            new.name = f"{name}_{src.name}"[:60]
            new["cast_role"] = "extra"
            new["cast_id"] = name.lower()
            dup_roots.append(new)
        except Exception as e:
            print(f"[cast] dup {src.name}: {e}")
    if not dup_roots:
        return None
    # Place primary root
    primary = dup_roots[0]
    primary.location = (float(loc[0]), float(loc[1]), float(loc[2]))
    primary.rotation_euler = (0.0, 0.0, float(yaw))
    # Kenney characters are often ~1 unit; scale up toward human size if tiny
    try:
        dim = max(primary.dimensions) if hasattr(primary, "dimensions") else 0.0
        if dim < 0.5:
            primary.scale = (1.8, 1.8, 1.8)
            primary.location = (float(loc[0]), float(loc[1]), 0.0)
    except Exception:
        pass
    return primary


def _spawn_cast_extras(count: int, preset: str = "sidewalk", outfit_id: str = "casual_01") -> int:
    """
    Spawn body-only extras for crowd.
    Prefers CC0 Kenney GLBs under assets/cast/npcs/; falls back to capsule proxies.
    """
    from face_agents.cast_schema import clamp_extras_count, normalize_extras_preset, normalize_wardrobe_id

    n = clamp_extras_count(count)
    preset = normalize_extras_preset(preset)
    if preset == "none":
        n = 0
    _clear_cast_extras()
    if n <= 0:
        return 0
    oid = normalize_wardrobe_id(outfit_id or "casual_01")
    if oid == "hero_default":
        oid = "casual_01"
    colors = _WARDROBE_COLORS.get(oid, _WARDROBE_COLORS["casual_01"])
    col = _cast_extras_collection(True)
    slots = _extras_spawn_slots(preset if preset != "none" else "sidewalk", n)
    glbs = _npc_glb_candidates()
    spawned = 0
    for i, (x, y, yaw) in enumerate(slots):
        name = f"NPC_{i + 1:02d}"
        obj = None
        if glbs:
            glb = glbs[i % len(glbs)]
            obj = _import_npc_instance(glb, name, (x, y, 0.0), yaw, col)
        if obj is None:
            # Capsule fallback
            import bmesh
            mesh = bpy.data.meshes.new(name + "_Mesh")
            obj = bpy.data.objects.new(name, mesh)
            bm = bmesh.new()
            bmesh.ops.create_cube(bm, size=1.0)
            bm.to_mesh(mesh)
            bm.free()
            obj.scale = (0.45, 0.28, 1.55)
            obj.location = (float(x), float(y), 0.85)
            obj.rotation_euler = (0.0, 0.0, float(yaw))
            mat = _wardrobe_mat(
                f"Mat_NPC_{oid}", colors[0], fabric_kind=_wardrobe_fabric_for(oid, "shirt"),
            )
            obj.data.materials.append(mat)
            obj["cast_role"] = "extra"
            obj["cast_id"] = f"npc_{i + 1}"
            obj["wardrobe_id"] = oid
            if col is not None:
                try:
                    col.objects.link(obj)
                except Exception:
                    bpy.context.scene.collection.objects.link(obj)
            else:
                bpy.context.scene.collection.objects.link(obj)
        spawned += 1
    print(
        f"[cast] spawned {spawned} extras preset={preset} outfit={oid} "
        f"source={'kenney_glb' if glbs else 'capsule'}"
    )
    return spawned


def _apply_cast_packet(packet: dict) -> None:
    op = str(packet.get("op") or "spawn_extras").lower()
    if op in ("clear", "clear_extras"):
        _clear_cast_extras()
        return
    if op in ("spawn", "spawn_extras", "apply"):
        count = packet.get("count", packet.get("extras_count", 0))
        preset = packet.get("preset", packet.get("extras_preset", "sidewalk"))
        outfit = packet.get("outfit_id", packet.get("wardrobe_id", "casual_01"))
        _spawn_cast_extras(int(count or 0), str(preset), str(outfit))
        return
    print(f"[cast] unknown op={op}")


_BPY_DENY = (
    "os.system", "subprocess", "socket", "shutil.rmtree", "__import__('os')",
    "eval(", "exec(", "compile(", "builtins",
)


def _inspect_dump_path() -> str:
    """Fixed path Python tools poll after type=inspect."""
    try:
        root = _project_root_for_assets()
        out = root / "temp" / "movie_os_inspect.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        return str(out)
    except Exception:
        return ""


def _apply_inspect_packet(packet: dict) -> None:
    """UDP type=inspect — dump Session + Look_Set state for agentic loops."""
    import json as _json

    sc = bpy.context.scene
    arm = _body_armature()
    clips = []
    try:
        raw = sc.get("session_body_clip_labels")
        if isinstance(raw, str) and raw.strip():
            clips = _json.loads(raw)
        elif isinstance(raw, (list, tuple)):
            clips = list(raw)
    except Exception:
        clips = []
    look_objs = []
    col = bpy.data.collections.get(LOOK_COLLECTION)
    if col is not None:
        look_objs = [o.name for o in col.objects][:40]
    npcs = [o.name for o in bpy.data.objects if str(o.get("cast_role") or "") == "extra"][:20]
    cams = [
        o.name for o in bpy.data.objects
        if o.type == "CAMERA" or o.name.startswith("MovieCam_")
    ][:12]
    bound = None
    if arm and arm.animation_data and arm.animation_data.action:
        bound = arm.animation_data.action.name
    payload = {
        "ok": True,
        "session_id": str(sc.get("session_id") or ""),
        "session_action": str(sc.get("session_body_action") or ""),
        "bound_action": bound,
        "armature": arm.name if arm else None,
        "frame_current": int(sc.frame_current),
        "frame_start": int(sc.frame_start),
        "frame_end": int(sc.frame_end),
        "clip_count": len(clips) if isinstance(clips, list) else 0,
        "clips": [
            {
                "source_action": c.get("source_action"),
                "clip_label": c.get("clip_label"),
                "frame_start": c.get("frame_start"),
                "frame_end": c.get("frame_end"),
            }
            for c in (clips or [])[:24]
            if isinstance(c, dict)
        ],
        "look_objects": look_objs,
        "npcs": npcs,
        "cameras": cams,
        "scene_camera": sc.camera.name if sc.camera else None,
    }
    path = str(packet.get("path") or "").strip() or _inspect_dump_path()
    if path:
        try:
            with open(path, "w", encoding="utf-8") as f:
                _json.dump(payload, f, indent=2)
            print(f"[inspect] wrote {path} clips={payload['clip_count']} bound={bound}")
        except Exception as e:
            print(f"[inspect] write failed: {e}")
    else:
        print(f"[inspect] {payload}")


def _apply_light_packet(packet: dict) -> None:
    """UDP type=light — adjust Look_Set three-point / world strength."""
    op = str(packet.get("op") or "mood").lower()
    mood = str(packet.get("mood") or packet.get("light_mood") or "soft").lower()
    strength = packet.get("world_strength")
    key_e = packet.get("key_energy")
    fill_e = packet.get("fill_energy")
    rim_e = packet.get("rim_energy")
    # Mood presets
    presets = {
        "soft": (0.35, 350.0, 140.0, 70.0),
        "bright": (0.7, 550.0, 200.0, 100.0),
        "dramatic": (0.2, 700.0, 80.0, 220.0),
        "night": (0.12, 180.0, 60.0, 40.0),
        "golden": (0.55, 480.0, 160.0, 90.0),
    }
    ws, ke, fe, re = presets.get(mood, presets["soft"])
    if strength is not None:
        ws = float(strength)
    if key_e is not None:
        ke = float(key_e)
    if fill_e is not None:
        fe = float(fill_e)
    if rim_e is not None:
        re = float(rim_e)
    # Apply to known Look lights if present
    for name, energy in (
        ("Look_Key", ke),
        ("Look_Fill", fe),
        ("Look_Rim", re),
        ("Key", ke),
        ("Fill", fe),
        ("Rim", re),
    ):
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "LIGHT":
            continue
        try:
            obj.data.energy = float(energy)
        except Exception:
            pass
    try:
        world = bpy.context.scene.world
        if world and world.use_nodes:
            for n in world.node_tree.nodes:
                if n.type == "BACKGROUND":
                    n.inputs[1].default_value = float(ws)
                    break
    except Exception:
        pass
    print(f"[light] op={op} mood={mood} world={ws} key={ke}")


def _apply_material_packet(packet: dict) -> None:
    """UDP type=material — simple PBR tint on Look_Ground / named mesh."""
    target = str(packet.get("target") or "Look_Ground").strip()
    color = packet.get("color") or packet.get("base_color") or (0.35, 0.35, 0.38, 1.0)
    rough = float(packet.get("roughness") if packet.get("roughness") is not None else 0.65)
    metal = float(packet.get("metallic") if packet.get("metallic") is not None else 0.0)
    obj = bpy.data.objects.get(target)
    if obj is None or obj.type != "MESH":
        print(f"[material] missing mesh {target!r}")
        return
    mat_name = f"Mat_{target}"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = None
    for n in nt.nodes:
        if n.type == "BSDF_PRINCIPLED":
            bsdf = n
            break
    if bsdf is None:
        print(f"[material] no Principled on {mat_name}")
        return
    try:
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            rgba = (float(color[0]), float(color[1]), float(color[2]), float(color[3]) if len(color) > 3 else 1.0)
        else:
            rgba = (0.35, 0.35, 0.38, 1.0)
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = max(0.0, min(1.0, rough))
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = max(0.0, min(1.0, metal))
    except Exception as e:
        print(f"[material] set failed: {e}")
        return
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    print(f"[material] {target} rough={rough} metal={metal}")


def _apply_bpy_packet(packet: dict) -> None:
    """
    UDP type=bpy — freeform scene mediation for agentic LLMs.

    Runs small bpy snippets against Look_Set / Cast_Extras. Denies OS/network.
    Prefer typed look/cast/wardrobe tools; use this when kits cannot express the set.
    """
    code = str(packet.get("code") or packet.get("script") or "").strip()
    if not code:
        print("[bpy] empty code")
        return
    low = code.lower()
    for bad in _BPY_DENY:
        if bad.lower() in low:
            print(f"[bpy] blocked pattern {bad!r}")
            return
    # Allow open/write only under project temp/ (agentic inspect helpers)
    if "open(" in low:
        if "temp" not in low and "movie_os" not in low:
            print("[bpy] blocked open() outside temp/movie_os")
            return
    if len(code) > 12000:
        print("[bpy] code too long")
        return
    # Ensure Look_Set exists for set dressing
    col = bpy.data.collections.get(LOOK_COLLECTION)
    if col is None:
        col = bpy.data.collections.new(LOOK_COLLECTION)
        try:
            bpy.context.scene.collection.children.link(col)
        except Exception:
            pass
    ns = {
        "bpy": bpy,
        "look_collection": col,
        "LOOK_COLLECTION": LOOK_COLLECTION,
        "result": {},
    }
    try:
        import mathutils as _mu
        ns["mathutils"] = _mu
        ns["Vector"] = _mu.Vector
    except Exception:
        pass
    try:
        exec(code, ns, ns)  # noqa: S102 — intentional agentic bpy mediation
        print(f"[bpy] ok result={ns.get('result')!r}")
    except Exception as e:
        print(f"[bpy] error: {e}")


def _apply_look_packet(packet: dict) -> None:
    """
    UDP type=look — build Look_Set (ground + lights + optional HDRI + simple set).
    Never moves the avatar armature; set sits around z=0 sole plane.
    """
    import sys
    from pathlib import Path

    op = str(packet.get("op") or "apply").lower()
    if op == "clear":
        _clear_look_set()
        print("[look] cleared Look_Set")
        return

    look = packet.get("look") if isinstance(packet.get("look"), dict) else {}
    plan = None
    root = _project_root_for_assets()
    # Allow importing face_agents from project when Blender runs the addon
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    try:
        from face_agents.look_schema import LookPlan
        from face_agents.scene_presets import resolve_look
        plan = LookPlan.from_dict(look)
        look = plan.to_dict()
        recipe = resolve_look(plan)
    except Exception as e:
        print(f"[look] resolve failed ({e}) — studio three-point fallback")
        recipe = {
            "recipe_name": "studio|day",
            "hdri_path": "",
            "use_three_point": True,
            "world_strength": 0.25,
            "key_energy": 400.0,
            "fill_energy": 140.0,
            "rim_energy": 80.0,
            "sun_energy": 0.0,
            "ground_size": 12.0,
            "set_style": "studio_cyc",
            "key_color": (1.0, 0.98, 0.95),
            "fill_color": (0.75, 0.82, 1.0),
            "rim_color": (1.0, 1.0, 1.0),
            "fallback": True,
        }
        # Prefer project HDRI if present even without import
        hdri = root / "assets" / "looks" / "hdri" / "studio_soft.exr"
        if hdri.is_file():
            recipe["hdri_path"] = str(hdri)
            recipe["use_three_point"] = False
            recipe["world_strength"] = 0.85
            recipe["fallback"] = False

    _clear_look_set()
    gsize = float(recipe.get("ground_size") or 12.0)
    style = str(recipe.get("set_style") or "studio_cyc")
    built = _build_look_set_geometry(style, gsize)

    hdri_path = str(recipe.get("hdri_path") or "")
    used_hdri = False
    if hdri_path and Path(hdri_path).is_file():
        used_hdri = _set_world_hdri(hdri_path, float(recipe.get("world_strength") or 0.85))
    if not used_hdri:
        _set_world_solid(0.18)
        recipe["use_three_point"] = True
        recipe["fallback"] = True
    print(
        f"[look] set built style={built} loc={recipe.get('location')} "
        f"hdri={Path(hdri_path).name if hdri_path else 'none'} "
        f"ok={used_hdri}"
    )

    # Three-point (also when HDRI present but soft fill helps character)
    key_e = float(recipe.get("key_energy") or 300)
    fill_e = float(recipe.get("fill_energy") or 100)
    rim_e = float(recipe.get("rim_energy") or 50)
    if recipe.get("use_three_point") or used_hdri:
        # Scale down area lights when HDRI is already lighting
        scale = 0.35 if used_hdri and not recipe.get("use_three_point") else 1.0
        if used_hdri and not recipe.get("fallback"):
            scale = 0.45
        _ensure_area_light(
            "Look_Key",
            loc=(2.2, -2.4, 2.6),
            rot=(0.9, 0.0, 0.7),
            energy=key_e * scale,
            color=recipe.get("key_color") or (1, 1, 1),
            size=2.8,
        )
        _ensure_area_light(
            "Look_Fill",
            loc=(-2.4, -1.6, 1.9),
            rot=(1.1, 0.0, -0.6),
            energy=fill_e * scale,
            color=recipe.get("fill_color") or (0.8, 0.85, 1),
            size=3.2,
        )
        _ensure_area_light(
            "Look_Rim",
            loc=(0.4, 2.8, 2.4),
            rot=(0.7, 0.0, 3.14),
            energy=rim_e * scale,
            color=recipe.get("rim_color") or (1, 1, 1),
            size=2.0,
        )
    sun_e = float(recipe.get("sun_energy") or 0.0)
    if sun_e > 0.05:
        _ensure_sun_light(
            "Look_Sun",
            energy=sun_e,
            color=recipe.get("key_color") or (1.0, 0.9, 0.7),
            rot=(0.85, 0.2, 0.4),
        )

    _hide_retarget_helpers()
    gz = _look_ground_world_z()
    try:
        from tools.blender_movie_render import apply_compositor_grade
        apply_compositor_grade(bpy.context.scene, str(recipe.get("grade") or look.get("grade") or "neutral"))
    except Exception as e:
        print(f"[look] compositor grade skip: {e}")

    # Wardrobe + extras (typed LookPlan fields — not BodyDirector)
    try:
        wid = str(look.get("wardrobe_id") or getattr(plan, "wardrobe_id", "") or "hero_default")
        # Only keep clothes if they pass fit verification (real garments on hero).
        applied = _apply_wardrobe(wid, actor="hero", require_fit=True)
        if applied == "hero_default" and wid not in ("", "hero_default"):
            look["wardrobe_id"] = "hero_default"
    except Exception as e:
        print(f"[look] wardrobe skip: {e}")
    try:
        ec = int(look.get("extras_count", getattr(plan, "extras_count", 0)) or 0)
        ep = str(look.get("extras_preset", getattr(plan, "extras_preset", "none")) or "none")
        if ec > 0 and ep != "none":
            _spawn_cast_extras(ec, ep, str(look.get("wardrobe_id") or "casual_01"))
        elif ec == 0 or ep == "none":
            # Only clear when explicitly zero/none in this packet (keep prior crowd otherwise)
            if "extras_count" in look or "extras_preset" in look:
                _clear_cast_extras()
    except Exception as e:
        print(f"[look] cast extras skip: {e}")

    print(
        f"[look] apply recipe={recipe.get('recipe_name')} "
        f"hdri={'yes' if used_hdri else 'no'} "
        f"fallback={bool(recipe.get('fallback'))} "
        f"set={style} ground={gsize:.1f}m z={gz:.4f} "
        f"wardrobe={look.get('wardrobe_id', '')} extras={look.get('extras_count', 0)}"
    )


# ---------------------------------------------------------------------------
# MOVIE CAMERA (cinematic path / look-at)
# ---------------------------------------------------------------------------

def _ensure_movie_camera(cam_name: str = "MovieCam", look_name: str = "MovieCam_LookAt"):
    """
    Create movie camera + look-at empty if missing; set as scene camera.
    Supports multi-cam rig: MovieCam_A / _B / _C / _Env created on demand.
    """
    scene = bpy.context.scene
    # Legacy alias: plain MovieCam → A
    if cam_name in ("MovieCam", ""):
        cam_name = "MovieCam_A"
    if look_name in ("MovieCam_LookAt", ""):
        look_name = cam_name + "_LookAt" if not look_name.endswith("_LookAt") else look_name
        if look_name == "MovieCam_LookAt":
            look_name = "MovieCam_A_LookAt"

    look = bpy.data.objects.get(look_name)
    if look is None:
        look = bpy.data.objects.new(look_name, None)
        look.empty_display_type = "PLAIN_AXES"
        look.empty_display_size = 0.08
        scene.collection.objects.link(look)
        look.location = (0.0, 0.0, 1.55)

    cam_obj = bpy.data.objects.get(cam_name)
    if cam_obj is None or cam_obj.type != "CAMERA":
        # Remove wrong-type object if name collision
        if cam_obj is not None and cam_obj.type != "CAMERA":
            try:
                bpy.data.objects.remove(cam_obj, do_unlink=True)
            except Exception:
                pass
        cam_data = bpy.data.cameras.new(cam_name)
        cam_data.lens = 45.0
        cam_obj = bpy.data.objects.new(cam_name, cam_data)
        scene.collection.objects.link(cam_obj)
        # Safe default outside head (never -0.55 ECU)
        cam_obj.location = (0.0, -2.9, 1.48)
        print(f"[camera_receiver] CREATED camera {cam_name!r} + look {look_name!r}")

    scene.camera = cam_obj
    return cam_obj, look


def _clamp_cam_look(loc, look, min_dist: float = 1.08, soft: bool = True):
    """Push camera away from look_at if too close (prevent head penetration)."""
    try:
        lv = Vector(loc)
        tv = Vector(look)
        d = (lv - tv).length
        if d < 1e-6:
            return [float(tv.x), float(tv.y) - max(min_dist, 1.15), float(tv.z)], list(look)
        if d < min_dist:
            dirn = (lv - tv).normalized()
            lv = tv + dirn * min_dist
        # Soft front bias only — hard Y snap was jumping the cam every frame
        # when subject-relative offsets crossed the threshold.
        if soft:
            if lv.y > -0.85 and abs(lv.x) < 1.8:
                # ease toward front plane, never teleport
                lv.y = lv.y * 0.65 + (-1.2) * 0.35
        else:
            if lv.y > -1.15 and abs(lv.x) < 1.6:
                lv.y = -1.15
        return [float(lv.x), float(lv.y), float(lv.z)], [float(tv.x), float(tv.y), float(tv.z)]
    except Exception:
        return list(loc), list(look)


def _head_world_location():
    """Best-effort head / neck world position for track_head."""
    return _subject_world_location("head")


def _subject_world_location(anchor: str = "chest"):
    """
    FilmAgent-style subject point for camera framing.
    Not always hip: head / chest / pelvis / full_body (pelvis + height bias applied by look).
    """
    arm = bpy.data.objects.get("SMPL-X_Armature")
    if arm is None:
        return None
    a = (anchor or "chest").lower().strip()
    name_lists = {
        "head": ("head", "Head", "neck", "Neck"),
        "chest": ("spine3", "Spine3", "spine2", "Spine2", "chest", "Chest", "neck", "Neck"),
        "pelvis": ("pelvis", "Pelvis", "hips", "Hips", "root", "Root"),
        "full_body": ("pelvis", "Pelvis", "hips", "Hips", "root", "Root"),
    }
    names = name_lists.get(a, name_lists["chest"])
    try:
        for bname in names:
            if bname in arm.pose.bones:
                pb = arm.pose.bones[bname]
                return arm.matrix_world @ pb.head
        # fallback any bone
        for bname in ("pelvis", "spine3", "head"):
            if bname in arm.pose.bones:
                return arm.matrix_world @ arm.pose.bones[bname].head
    except Exception:
        pass
    try:
        return arm.matrix_world.translation.copy()
    except Exception:
        return None


def _pelvis_world_location():
    return _subject_world_location("pelvis")


def _align_action_root_to_current(arm, act_name: str, f0: int, prev_world=None) -> None:
    """
    Session continuity: after assigning a new Action, shift armature.location so the
    character starts where they currently stand (no teleport to origin).
    prev_world: Vector captured BEFORE switching Action (required for correct Δ).
    """
    if arm is None:
        return
    try:
        prev = prev_world
        if prev is None:
            return
        # Evaluate new action at start frame to measure new root
        scene = bpy.context.scene
        scene.frame_set(int(f0))
        bpy.context.view_layer.update()
        new = _pelvis_world_location()
        if new is None:
            return
        dx = float(prev.x - new.x)
        dy = float(prev.y - new.y)
        # Horizontal only. Armature origin is the character center (~chest);
        # never accumulate Z onto the object or the avatar floats off the floor.
        if abs(dx) + abs(dy) < 1e-4:
            return
        arm.location.x += dx
        arm.location.y += dy
        bpy.context.view_layer.update()
        print(
            f"[body_receiver] root-align Δ=({dx:.3f},{dy:.3f}) "
            f"action={act_name!r} arm.loc={tuple(round(c, 3) for c in arm.location)}"
        )
    except Exception as e:
        print(f"[body_receiver] root-align failed: {e}")


def _clear_action_fcurves(act) -> None:
    """Wipe keys on an Action without deleting the datablock (session reset)."""
    if act is None:
        return
    try:
        # Blender 4.4+ layered actions
        if hasattr(act, "layers") and act.layers:
            for layer in act.layers:
                for strip in getattr(layer, "strips", []) or []:
                    chans = getattr(strip, "channelbags", None) or []
                    for bag in chans:
                        fcs = getattr(bag, "fcurves", None)
                        if fcs is None:
                            continue
                        while len(fcs):
                            fcs.remove(fcs[0])
            return
    except Exception:
        pass
    try:
        fcs = getattr(act, "fcurves", None)
        if fcs is not None:
            while len(fcs):
                fcs.remove(fcs[0])
    except Exception:
        pass


def _keyframe_armature_pose(arm, frame: int, bones_filter=None) -> int:
    """Insert pose keys at `frame` into the currently assigned Action. Returns key count."""
    n = 0
    fr = int(frame)
    for pb in arm.pose.bones:
        if bones_filter is not None and pb.name not in bones_filter:
            continue
        try:
            pb.rotation_mode = "QUATERNION"
            pb.keyframe_insert(data_path="rotation_quaternion", frame=fr)
            n += 4
        except Exception:
            try:
                pb.keyframe_insert(data_path="rotation_euler", frame=fr)
                n += 3
            except Exception:
                pass
        # Location: root / free bones (pelvis always; others if non-zero)
        try:
            if pb.name.lower() in ("pelvis", "root", "hips") or pb.location.length > 1e-5:
                pb.keyframe_insert(data_path="location", frame=fr)
                n += 3
        except Exception:
            pass
    return n


def _sample_src_pelvis_loc(arm, src_act, frame: float):
    """Pelvis local location at a source Action frame (via fast eval)."""
    if arm is None or src_act is None:
        return None
    ok = _apply_action_frame_fast(arm, src_act, float(frame))
    if not ok:
        try:
            if not arm.animation_data:
                arm.animation_data_create()
            arm.animation_data.action = src_act
            bpy.context.scene.frame_set(int(round(frame)))
            bpy.context.view_layer.update()
        except Exception:
            return None
    pb = arm.pose.bones.get("pelvis") or arm.pose.bones.get("Pelvis")
    if pb is None:
        return None
    return pb.location.copy()


def _append_clip_to_session_action(
    arm,
    src_act,
    frame_start: int = 0,
    *,
    continue_root: bool = True,
    action_frames: int = 0,
    motion_length: int = 0,
) -> tuple:
    """Deprecated alias → _append_session_master (keeps old call sites safe)."""
    return _append_session_master(
        arm, src_act, action_frames=action_frames, motion_length=motion_length
    )


def _reset_session_body_timeline(arm=None) -> None:
    """Alias: clear session NLA timeline (cursor back to 1)."""
    _reset_session_nla(arm)


def _look_at_rotation(cam_loc, target_loc):
    """Euler rotation so Blender camera -Z looks at target, Y up."""
    direction = Vector(target_loc) - Vector(cam_loc)
    if direction.length < 1e-6:
        return (math.radians(90.0), 0.0, 0.0)
    rot_quat = direction.to_track_quat("-Z", "Y")
    return rot_quat.to_euler()


def _fov_to_lens_mm(fov_deg: float, sensor_width: float = 36.0) -> float:
    fov = max(5.0, min(120.0, float(fov_deg)))
    # lens = (sensor/2) / tan(fov/2)
    return float(sensor_width) / (2.0 * math.tan(math.radians(fov) * 0.5))


def _interp_camera_at_t(keyframes, t: float):
    """Linear interpolate location / look_at / fov at time t (seconds)."""
    if not keyframes:
        return None
    kfs = sorted(keyframes, key=lambda k: float(k.get("t", 0.0)))
    if t <= float(kfs[0].get("t", 0.0)):
        k = kfs[0]
        return (
            list(k.get("location") or [0, -2.4, 1.5]),
            list(k.get("look_at") or [0, 0, 1.55]),
            float(k.get("fov_deg") or 45.0),
        )
    if t >= float(kfs[-1].get("t", 0.0)):
        k = kfs[-1]
        return (
            list(k.get("location") or [0, -2.4, 1.5]),
            list(k.get("look_at") or [0, 0, 1.55]),
            float(k.get("fov_deg") or 45.0),
        )
    for i in range(len(kfs) - 1):
        t0 = float(kfs[i].get("t", 0.0))
        t1 = float(kfs[i + 1].get("t", 0.0))
        if t0 <= t <= t1:
            span = max(1e-6, t1 - t0)
            u = (t - t0) / span
            # smoothstep
            u = u * u * (3.0 - 2.0 * u)
            a, b = kfs[i], kfs[i + 1]
            la = list(a.get("location") or [0, -2.4, 1.5])
            lb = list(b.get("location") or [0, -2.4, 1.5])
            ta = list(a.get("look_at") or [0, 0, 1.55])
            tb = list(b.get("look_at") or [0, 0, 1.55])
            fa = float(a.get("fov_deg") or 45.0)
            fb = float(b.get("fov_deg") or 45.0)
            loc = [la[j] + (lb[j] - la[j]) * u for j in range(3)]
            look = [ta[j] + (tb[j] - ta[j]) * u for j in range(3)]
            fov = fa + (fb - fa) * u
            return loc, look, fov
    k = kfs[-1]
    return (
        list(k.get("location") or [0, -2.4, 1.5]),
        list(k.get("look_at") or [0, 0, 1.55]),
        float(k.get("fov_deg") or 45.0),
    )


def _session_cam_action_name(obj) -> str:
    return f"{SESSION_CAM_PREFIX}{obj.name}" if obj is not None else ""


def _ensure_session_cam_action(obj):
    """
    Persistent Action per MovieCam / LookAt. Never create a new Action after
    live detach — that orphaned clip-1 keys and left only the latest take.
    """
    if obj is None:
        return None
    name = _session_cam_action_name(obj)
    act = bpy.data.actions.get(name)
    if act is None:
        act = bpy.data.actions.new(name)
        act.use_fake_user = True
    else:
        act.use_fake_user = True
    if obj.animation_data is None:
        obj.animation_data_create()
    obj.animation_data.action = act
    return act


def _camera_clip_covering(frame: int):
    clips = _CAMERA_PLAY.get("session_clips") or []
    f = int(frame)
    for c in reversed(clips):
        a = int(c.get("frame_start") or 1)
        b = int(c.get("frame_end") or a)
        if a <= f <= b:
            return c
    if clips:
        last = clips[-1]
        if f >= int(last.get("frame_end") or 1):
            return last
        return clips[0]
    return None


def _reset_session_cameras() -> None:
    """Drop SessionCam_* Actions on reset / new scene."""
    global _CAMERA_PLAY
    drop = []
    for act in list(bpy.data.actions):
        try:
            if str(act.name or "").startswith(SESSION_CAM_PREFIX):
                drop.append(act)
        except Exception:
            pass
    for act in drop:
        try:
            bpy.data.actions.remove(act)
        except Exception:
            pass
    _CAMERA_PLAY["session_clips"] = []
    _CAMERA_PLAY["cam_action"] = ""
    _CAMERA_PLAY["look_action"] = ""
    _CAMERA_PLAY["keyframes"] = []
    _CAMERA_PLAY["live"] = False


def _bake_camera_keyframes(
    cam_obj,
    look_obj,
    keyframes,
    fps: float,
    *,
    clear_previous: bool = False,
) -> None:
    """Insert Blender keyframes for scrub / review. Prefer absolute frame_abs/t_abs for multi-shot join."""
    if not keyframes:
        return
    cam_act = _ensure_session_cam_action(cam_obj)
    look_act = _ensure_session_cam_action(look_obj)
    if cam_obj is not None and cam_obj.data is not None:
        if cam_obj.data.animation_data is None:
            try:
                cam_obj.data.animation_data_create()
            except Exception:
                pass
        if cam_obj.data.animation_data is not None and cam_act is not None:
            try:
                cam_obj.data.animation_data.action = cam_act
            except Exception:
                pass
    # Clear only when starting a new film; multi-shot append keeps prior keys
    if clear_previous:
        for act in (cam_act, look_act):
            if act is None:
                continue
            try:
                _clear_action_fcurves(act)
            except Exception:
                try:
                    for fc in list(act.fcurves):
                        act.fcurves.remove(fc)
                except Exception:
                    pass

    scene = bpy.context.scene
    f0 = 10**9
    f1 = 1
    for k in keyframes:
        # Absolute master-timeline times (movie package) take priority
        if k.get("frame_abs") is not None:
            fr = int(k.get("frame_abs"))
        elif k.get("t_abs") is not None:
            fr = max(1, int(round(float(k.get("t_abs")) * fps)) + 1)
        else:
            t = float(k.get("t", 0.0))
            fr = int(k.get("frame") or (max(1, int(round(t * fps)) + 1)))
        f0 = min(f0, fr)
        f1 = max(f1, fr)
        loc = list(k.get("location") or [0, -2.4, 1.5])
        look = list(k.get("look_at") or [0, 0, 1.55])
        fov = float(k.get("fov_deg") or 45.0)

        look_obj.location = Vector(look)
        look_obj.keyframe_insert(data_path="location", frame=fr)

        cam_obj.location = Vector(loc)
        eul = _look_at_rotation(loc, look)
        cam_obj.rotation_euler = eul
        cam_obj.keyframe_insert(data_path="location", frame=fr)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=fr)

        if cam_obj.data:
            cam_obj.data.lens = _fov_to_lens_mm(fov)
            cam_obj.data.keyframe_insert(data_path="lens", frame=fr)

    # LINEAR — Bezier between takes arcs through space and looks like a mid-clip jump.
    # Blender 5 layered Actions often have empty .fcurves — iterate channelbags too.
    def _linearize_act(act):
        if act is None:
            return
        try:
            for fc in _iter_action_fcurves(act):
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
        except Exception:
            try:
                for fc in getattr(act, "fcurves", []) or []:
                    for kp in fc.keyframe_points:
                        kp.interpolation = "LINEAR"
            except Exception:
                pass

    for obj in (cam_obj, look_obj):
        if not obj.animation_data or not obj.animation_data.action:
            continue
        _linearize_act(obj.animation_data.action)
    if cam_obj.data and cam_obj.data.animation_data and cam_obj.data.animation_data.action:
        _linearize_act(cam_obj.data.animation_data.action)

    if f0 > 10**8:
        f0 = 1
    if clear_previous:
        scene.frame_start = f0
        scene.frame_end = max(int(scene.frame_end or 1), f1)
    else:
        scene.frame_start = min(int(scene.frame_start or f0), f0)
        scene.frame_end = max(int(scene.frame_end or 1), f1)
    # Keep the same SessionCam_* Action for every take on this object
    try:
        if cam_act is not None:
            cam_act.use_fake_user = True
            _CAMERA_PLAY["cam_action"] = cam_act.name
        if look_act is not None:
            look_act.use_fake_user = True
            _CAMERA_PLAY["look_action"] = look_act.name
    except Exception:
        pass
    print(
        f"[camera_receiver] baked keyframes frames {f0}-{f1} on {cam_obj.name} "
        f"clear={clear_previous} scene=f{scene.frame_start}-{scene.frame_end} "
        f"action={getattr(cam_act, 'name', '')}"
    )


def _detach_cam_action(cam_obj, look_obj) -> None:
    """
    While live wall-clock drives the camera, Blender Action FCurves on the same
    objects fight every frame_set (body scrub) → cam jumps to baked frame then
    back. Detach actions for live/hold; re-bake only for offline timeline scrub.
    """
    for obj in (cam_obj, look_obj):
        if obj is None:
            continue
        try:
            if obj.animation_data and obj.animation_data.action:
                # Keep action datablock for later scrub; just unbind evaluation
                obj.animation_data.action = None
        except Exception:
            pass
        try:
            if obj.data is not None and getattr(obj.data, "animation_data", None):
                if obj.data.animation_data and obj.data.animation_data.action:
                    obj.data.animation_data.action = None
        except Exception:
            pass


def _queue_camera_packet(packet: dict) -> None:
    """Store camera plan; modal applies live pose each tick."""
    global _CAMERA_PLAY
    op = str(packet.get("op") or "plan").lower()
    if op in ("rest", "stop", "end"):
        # End live follow. Prefer SessionCam history over a frozen hold so
        # Play/scrub/export still evaluate multi-take camera keys.
        _CAMERA_PLAY["live"] = False
        if _CAMERA_PLAY.get("session_clips"):
            _CAMERA_PLAY["hold_after"] = False
            _CAMERA_PLAY["hold_world_loc"] = None
            _CAMERA_PLAY["hold_world_look"] = None
            try:
                cam_n = _CAMERA_PLAY.get("camera_name") or "MovieCam_A"
                look_n = _CAMERA_PLAY.get("look_at_name") or (cam_n + "_LookAt")
                _ensure_session_cam_action(bpy.data.objects.get(cam_n))
                _ensure_session_cam_action(bpy.data.objects.get(look_n))
                _bind_camera_for_timeline_review()
            except Exception:
                pass
            print("[camera_receiver] rest → SessionCam history (no freeze)")
        else:
            _CAMERA_PLAY["hold_after"] = True
            if _CAMERA_PLAY.get("smooth_loc") is not None:
                _CAMERA_PLAY["hold_world_loc"] = list(_CAMERA_PLAY["smooth_loc"])
                _CAMERA_PLAY["hold_world_look"] = list(
                    _CAMERA_PLAY.get("smooth_look") or _CAMERA_PLAY["smooth_loc"]
                )
            print("[camera_receiver] hold last framing (no SessionCam history yet)")
        return

    kfs = packet.get("keyframes") or []
    if not kfs:
        print("[camera_receiver] plan ignored — no keyframes")
        return

    cam_name = str(packet.get("camera_name") or "MovieCam_A")
    look_name = str(packet.get("look_at_name") or (cam_name + "_LookAt"))
    role = str(packet.get("camera_role") or "")
    fps = float(packet.get("fps") or MASTER_EVAL_FPS)
    if fps > 22.0 or fps < 8.0:
        fps = MASTER_EVAL_FPS
    min_dist = float(packet.get("min_cam_dist") or 1.08)
    duration = packet.get("duration")
    if duration is None and kfs:
        duration = max(float(k.get("t", 0.0)) for k in kfs)

    subject_relative = bool(packet.get("subject_relative", True))
    subject_anchor = str(packet.get("subject_anchor") or "chest")
    # Higher lag = less whip when subject/plan updates (continuity)
    follow_lag = float(packet.get("follow_lag") or 0.55)
    hold_after = bool(packet.get("hold_after", True))
    # Wipe camera history only on an empty session / explicit reset
    session_has_clips = bool(_SESSION_BODY.get("clips"))
    clear_previous = bool(packet.get("clear_previous")) and not session_has_clips
    bake_keyframes = bool(packet.get("bake_keyframes", True))
    # Optional absolute timeline offset so cam keys land next to body session frames
    try:
        frame_offset = int(packet.get("session_frame_start") or packet.get("frame_offset") or 0)
    except (TypeError, ValueError):
        frame_offset = 0

    # Soft clamp keyframes before bake/live
    safe_kfs = []
    for k in kfs:
        kk = dict(k)
        loc = list(kk.get("rel_cam") or kk.get("location") or [0, -2.9, 1.5])
        look = list(kk.get("rel_look") or kk.get("look_at") or [0, 0, 1.55])
        if not subject_relative:
            loc, look = _clamp_cam_look(loc, look, min_dist=min_dist, soft=True)
        kk["location"] = loc
        kk["look_at"] = look
        if subject_relative:
            kk["rel_cam"] = list(loc)
            kk["rel_look"] = list(look)
        # Shift bake frames onto session timeline when offset provided
        if frame_offset >= 1:
            t = float(kk.get("t") or 0.0)
            fr = int(kk.get("frame") or max(1, int(round(t * fps)) + 1))
            kk["frame_abs"] = frame_offset + (fr - 1)
        safe_kfs.append(kk)
    kfs = safe_kfs

    # Keep previous smooth state so we don't hard-snap (flicker) on every plan
    prev_smooth_loc = _CAMERA_PLAY.get("smooth_loc") or _CAMERA_PLAY.get("hold_world_loc")
    prev_smooth_look = _CAMERA_PLAY.get("smooth_look") or _CAMERA_PLAY.get("hold_world_look")
    prev_cam_name = _CAMERA_PLAY.get("camera_name")

    try:
        cam_obj, look_obj = _ensure_movie_camera(cam_name, look_name)
        if packet.get("set_scene_camera", True):
            bpy.context.scene.camera = cam_obj
        # Bind the persistent SessionCam Action BEFORE bake so keys append
        _ensure_session_cam_action(cam_obj)
        _ensure_session_cam_action(look_obj)
        if bake_keyframes:
            bake_kfs = kfs
            if subject_relative:
                subj = _subject_world_location(subject_anchor)
                if subj is not None:
                    bake_kfs = []
                    for k in kfs:
                        kk = dict(k)
                        rc = list(kk.get("location") or [0, -2.9, 0])
                        rl = list(kk.get("look_at") or [0, 0, 0])
                        kk["location"] = [
                            float(subj.x) + rc[0],
                            float(subj.y) + rc[1],
                            float(subj.z) + rc[2],
                        ]
                        kk["look_at"] = [
                            float(subj.x) + rl[0],
                            float(subj.y) + rl[1],
                            float(subj.z) + rl[2],
                        ]
                        bake_kfs.append(kk)
            _bake_camera_keyframes(
                cam_obj,
                look_obj,
                bake_kfs,
                fps,
                clear_previous=clear_previous,
            )
            # Record this take on the camera session so later clips cannot wipe it
            rec_f0, rec_f1 = 10**9, 1
            for k in bake_kfs:
                if k.get("frame_abs") is not None:
                    fr = int(k["frame_abs"])
                else:
                    t = float(k.get("t") or 0.0)
                    fr = int(k.get("frame") or (max(1, int(round(t * fps)) + 1)))
                    if frame_offset >= 1:
                        fr = frame_offset + (fr - 1)
                rec_f0 = min(rec_f0, fr)
                rec_f1 = max(rec_f1, fr)
            if rec_f0 > 10**8:
                rec_f0, rec_f1 = max(1, frame_offset or 1), max(1, frame_offset or 1)
            if clear_previous:
                _CAMERA_PLAY["session_clips"] = []
            _CAMERA_PLAY.setdefault("session_clips", []).append({
                "frame_start": int(rec_f0),
                "frame_end": int(rec_f1),
                "camera_name": cam_name,
                "look_name": look_name,
            })
            # Persist cam take table so replay works after script reload
            try:
                import json as _json
                bpy.context.scene["session_cam_clips"] = _json.dumps(
                    _CAMERA_PLAY.get("session_clips") or [], ensure_ascii=False
                )
            except Exception:
                pass
            # Keep SessionCam_* bound with keys — history must survive for Play session.
            # Live wall-clock path detaches only while actively driving (below).
            _ensure_session_cam_action(cam_obj)
            _ensure_session_cam_action(look_obj)
            print(
                f"[camera_receiver] history clips={len(_CAMERA_PLAY.get('session_clips') or [])} "
                f"cam={cam_name} bound={getattr(cam_obj.animation_data.action, 'name', None)}"
            )
        # Soft handoff: seed from previous camera world even if role changed
        if cam_obj is not None:
            if prev_smooth_loc is None:
                seed_obj = bpy.data.objects.get(str(prev_cam_name or "")) if prev_cam_name else None
                if seed_obj is not None:
                    prev_smooth_loc = list(seed_obj.location)
                else:
                    prev_smooth_loc = list(cam_obj.location)
            if prev_smooth_look is None:
                prev_look_obj = None
                if prev_cam_name:
                    prev_look_obj = bpy.data.objects.get(str(prev_cam_name) + "_LookAt")
                if prev_look_obj is not None:
                    prev_smooth_look = list(prev_look_obj.location)
                elif look_obj is not None:
                    prev_smooth_look = list(look_obj.location)
                else:
                    prev_smooth_look = list(cam_obj.location)
    except Exception as e:
        print(f"[camera_receiver] setup error: {e}")

    _CAMERA_PLAY.update({
        "live": True,
        "t0": time.time(),
        "duration": float(duration) if duration is not None else None,
        "fps": fps,
        "keyframes": list(kfs),
        "camera_name": cam_name,
        "look_at_name": look_name,
        "track_head": bool(packet.get("track_head", False)),
        "min_cam_dist": min_dist,
        "last_t": -1.0,
        "subject_relative": subject_relative,
        "subject_anchor": subject_anchor,
        # Stronger lag = continuous follow (less per-frame jump)
        "follow_lag": max(0.40, min(0.88, follow_lag)),
        "hold_after": hold_after,
        # Continuity: keep previous smooth to avoid flicker
        "smooth_loc": prev_smooth_loc,
        "smooth_look": prev_smooth_look,
        "hold_world_loc": None,
        "hold_world_look": None,
    })
    shot = packet.get("shot") or ""
    move = packet.get("move_type") or ""
    print(
        f"[camera_receiver] PLAN continuous cam={cam_name} shot={shot} move={move} "
        f"keys={len(kfs)} duration={duration} clear_prev={clear_previous} "
        f"bake={bake_keyframes} anchor={subject_anchor} lag={follow_lag:.2f} "
        f"frame_off={frame_offset}"
    )


def _camera_world_from_sample(sample, subject_relative: bool, anchor: str):
    """
    Convert keyframe sample to world cam/look.
    subject_relative: sample offsets are added to live subject bone.
    """
    loc, look, fov = sample
    if subject_relative:
        subj = _subject_world_location(anchor)
        if subj is not None:
            # rel_cam / location stored as offset from subject
            loc = [
                float(subj.x) + float(loc[0]),
                float(subj.y) + float(loc[1]),
                float(subj.z) + float(loc[2]),
            ]
            look = [
                float(subj.x) + float(look[0]),
                float(subj.y) + float(look[1]),
                float(subj.z) + float(look[2]),
            ]
            # full_body: raise look toward mid-torso so whole figure is framed
            if (anchor or "").lower() == "full_body":
                look[2] = float(look[2]) + 0.15
    return loc, look, fov


def _smooth_camera(loc, look, lag: float):
    """Exponential lag so follow doesn't whip on jumps."""
    global _CAMERA_PLAY
    lag = max(0.0, min(0.9, float(lag)))
    prev_l = _CAMERA_PLAY.get("smooth_loc")
    prev_k = _CAMERA_PLAY.get("smooth_look")
    if prev_l is None or lag <= 1e-4:
        _CAMERA_PLAY["smooth_loc"] = list(loc)
        _CAMERA_PLAY["smooth_look"] = list(look)
        return list(loc), list(look)
    a = 1.0 - lag
    sl = [
        prev_l[0] + (loc[0] - prev_l[0]) * a,
        prev_l[1] + (loc[1] - prev_l[1]) * a,
        prev_l[2] + (loc[2] - prev_l[2]) * a,
    ]
    sk = [
        prev_k[0] + (look[0] - prev_k[0]) * a,
        prev_k[1] + (look[1] - prev_k[1]) * a,
        prev_k[2] + (look[2] - prev_k[2]) * a,
    ]
    _CAMERA_PLAY["smooth_loc"] = sl
    _CAMERA_PLAY["smooth_look"] = sk
    return sl, sk


def _bind_camera_for_timeline_review() -> bool:
    """Re-attach baked camera Actions so Space/scrub moves the cam with the body."""
    sc = bpy.context.scene
    # Hydrate cam take table after reload (mirrors body clip hydrate)
    if not (_CAMERA_PLAY.get("session_clips") or []):
        try:
            import json as _json
            raw = sc.get("session_cam_clips")
            if raw:
                clips = _json.loads(raw) if isinstance(raw, str) else list(raw)
                if clips:
                    _CAMERA_PLAY["session_clips"] = clips
        except Exception:
            pass
    fr = int(getattr(sc, "frame_current", 1) or 1)
    rec = _camera_clip_covering(fr)
    cam_name = (rec or {}).get("camera_name") or _CAMERA_PLAY.get("camera_name") or "MovieCam_A"
    look_name = (rec or {}).get("look_name") or _CAMERA_PLAY.get("look_at_name") or (cam_name + "_LookAt")
    # Bind every SessionCam_* so past takes stay on their objects
    bound = False
    for act in list(bpy.data.actions):
        n = str(getattr(act, "name", "") or "")
        if not n.startswith(SESSION_CAM_PREFIX):
            continue
        obj_name = n[len(SESSION_CAM_PREFIX):]
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        if obj.animation_data is None:
            obj.animation_data_create()
        obj.animation_data.action = act
        try:
            slots = getattr(obj.animation_data, "action_suitable_slots", None)
            if slots and len(slots) > 0:
                obj.animation_data.action_slot = slots[0]
        except Exception:
            pass
        bound = True
    cam_obj = bpy.data.objects.get(cam_name)
    look_obj = bpy.data.objects.get(look_name)
    if cam_obj is None:
        return bound
    _ensure_session_cam_action(cam_obj)
    if look_obj is not None:
        _ensure_session_cam_action(look_obj)
    try:
        sc.camera = cam_obj
    except Exception:
        pass
    return True


def _apply_camera_play(scene) -> None:
    """Main-thread: fly MovieCam along planned path (wall clock), continuous follow."""
    global _CAMERA_PLAY
    # Timeline review / Space play: evaluate baked SessionCam_* keys.
    # Must NOT freeze hold_world here — that made replay look static after take 1.
    try:
        playing = bool(getattr(bpy.context.screen, "is_animation_playing", False))
    except Exception:
        playing = False
    try:
        reviewing = (
            bool(_BODY_PLAY.get("scrub_user_active"))
            or playing
            or _user_reviewing_session(scene)
        )
    except Exception:
        reviewing = bool(playing)
    if reviewing and not _BODY_PLAY.get("live") and not _CAMERA_PLAY.get("live"):
        _CAMERA_PLAY["hold_after"] = False
        _CAMERA_PLAY["hold_world_loc"] = None
        _CAMERA_PLAY["hold_world_look"] = None
        _bind_camera_for_timeline_review()
        return

    if not _CAMERA_PLAY.get("live") and not _CAMERA_PLAY.get("hold_after"):
        return
    kfs = _CAMERA_PLAY.get("keyframes") or []
    if not kfs and not _CAMERA_PLAY.get("hold_world_loc"):
        return

    cam_name = _CAMERA_PLAY.get("camera_name") or "MovieCam_A"
    look_name = _CAMERA_PLAY.get("look_at_name") or "MovieCam_A_LookAt"
    min_dist = float(_CAMERA_PLAY.get("min_cam_dist") or 1.08)
    subject_relative = bool(_CAMERA_PLAY.get("subject_relative", True))
    anchor = str(_CAMERA_PLAY.get("subject_anchor") or "chest")
    lag = float(_CAMERA_PLAY.get("follow_lag") or 0.55)
    cam_obj = bpy.data.objects.get(cam_name)
    look_obj = bpy.data.objects.get(look_name)
    if cam_obj is None or look_obj is None:
        try:
            cam_obj, look_obj = _ensure_movie_camera(cam_name, look_name)
        except Exception:
            return

    # While LIVE wall-clock drives the cam, detach Action so body frame_set
    # does not yank the cam. When live ends, re-bind SessionCam for history.
    if _CAMERA_PLAY.get("live"):
        if cam_obj.animation_data and cam_obj.animation_data.action:
            _detach_cam_action(cam_obj, look_obj)
    else:
        # After live take: bind SessionCam history. Do NOT freeze-hold over it
        # or Play session / scrub only shows the last framing once.
        if (_CAMERA_PLAY.get("session_clips") or []):
            _CAMERA_PLAY["hold_after"] = False
            _CAMERA_PLAY["hold_world_loc"] = None
            _CAMERA_PLAY["hold_world_look"] = None
            if not (cam_obj.animation_data and cam_obj.animation_data.action):
                _bind_camera_for_timeline_review()

    # Frozen hold only when there is no SessionCam history yet (first empty take).
    if (
        (not _CAMERA_PLAY.get("live"))
        and _CAMERA_PLAY.get("hold_after")
        and not (_CAMERA_PLAY.get("session_clips") or [])
        and not playing
        and not _BODY_PLAY.get("scrub_user_active")
    ):
        hw = _CAMERA_PLAY.get("hold_world_loc")
        hk = _CAMERA_PLAY.get("hold_world_look")
        if hw is not None and hk is not None:
            look_obj.location = Vector(hk)
            cam_obj.location = Vector(hw)
            cam_obj.rotation_euler = _look_at_rotation(hw, hk)
            return

    elapsed = max(0.0, time.time() - float(_CAMERA_PLAY.get("t0") or time.time()))
    duration = _CAMERA_PLAY.get("duration")
    finished = duration is not None and elapsed >= float(duration)

    if finished and _CAMERA_PLAY.get("live"):
        _CAMERA_PLAY["live"] = False
        # With SessionCam history: re-bind Actions and stop freezing the view.
        # Freeze-hold over baked keys made export/Play look like a dead camera.
        if _CAMERA_PLAY.get("session_clips"):
            _CAMERA_PLAY["hold_after"] = False
            _CAMERA_PLAY["hold_world_loc"] = None
            _CAMERA_PLAY["hold_world_look"] = None
            _ensure_session_cam_action(cam_obj)
            _ensure_session_cam_action(look_obj)
            _bind_camera_for_timeline_review()
            print(
                f"[camera_receiver] plan finished — SessionCam rebound "
                f"(anchor={anchor} rel={subject_relative})"
            )
            return
        if _CAMERA_PLAY.get("smooth_loc") is not None:
            _CAMERA_PLAY["hold_world_loc"] = list(_CAMERA_PLAY["smooth_loc"])
            _CAMERA_PLAY["hold_world_look"] = list(
                _CAMERA_PLAY.get("smooth_look") or _CAMERA_PLAY["smooth_loc"]
            )
        print(
            f"[camera_receiver] plan finished — freeze framing "
            f"(anchor={anchor} rel={subject_relative})"
        )

    # After take: frozen framing only when no SessionCam history
    if finished and not _CAMERA_PLAY.get("hold_after"):
        return
    if finished and _CAMERA_PLAY.get("session_clips"):
        _CAMERA_PLAY["hold_after"] = False
        _bind_camera_for_timeline_review()
        return
    if finished:
        hw = _CAMERA_PLAY.get("hold_world_loc")
        hk = _CAMERA_PLAY.get("hold_world_look")
        if hw is not None and hk is not None:
            look_obj.location = Vector(hk)
            cam_obj.location = Vector(hw)
            cam_obj.rotation_euler = _look_at_rotation(hw, hk)
            return
        sample = _interp_camera_at_t(kfs, float(duration)) if kfs else None
    else:
        # Throttle updates slightly for stability
        if abs(elapsed - float(_CAMERA_PLAY.get("last_t") or -1.0)) < 0.016:
            return
        _CAMERA_PLAY["last_t"] = elapsed
        sample = _interp_camera_at_t(kfs, elapsed) if kfs else None

    if not sample:
        return

    loc, look, fov = _camera_world_from_sample(sample, subject_relative, anchor)

    # Soft head blend only when NOT subject_relative (legacy) and track_head
    if (not subject_relative) and _CAMERA_PLAY.get("track_head"):
        hw = _head_world_location()
        if hw is not None:
            look = [
                float(hw.x) * 0.22 + look[0] * 0.78,
                float(hw.y) * 0.12 + look[1] * 0.88,
                float(hw.z) * 0.50 + look[2] * 0.50,
            ]

    # Always smooth (even non-relative) so plan handoffs don't jump
    loc, look = _smooth_camera(loc, look, max(0.35, lag))
    loc, look = _clamp_cam_look(loc, look, min_dist=min_dist, soft=True)

    look_obj.location = Vector(look)
    cam_obj.location = Vector(loc)
    cam_obj.rotation_euler = _look_at_rotation(loc, look)
    if cam_obj.data:
        cam_obj.data.lens = _fov_to_lens_mm(fov)


def _apply_body_play(scene) -> None:
    """Main-thread: live wall-clock scrub OR leave timeline free for user scrub/replay."""
    global _BODY_PLAY
    arm = _body_armature(_BODY_PLAY.get("arm_name"))
    if arm is None:
        return

    # Smooth rest blend takes priority (after audio stop)
    if _BODY_PLAY.get("resting"):
        _apply_body_rest_blend(arm)
        try:
            _zero_mouth_shapes_if_idle()
        except Exception:
            pass
        return

    # After settle: hold idle until the user plays/scrubs the SESSION timeline
    if _BODY_PLAY.get("idle_hold") and not _BODY_PLAY.get("live"):
        if _BODY_PLAY.get("scrub_ready") and _user_reviewing_session(scene):
            _BODY_PLAY["idle_hold"] = False
            _BODY_PLAY["scrub_user_active"] = True
            _assign_session_action_for_scrub(arm, allow_during_idle=True)
            _BODY_PLAY["last_frame"] = int(scene.frame_current)
            try:
                _zero_mouth_shapes_if_idle()
            except Exception:
                pass
            return
        if _BODY_PLAY.get("last_frame") is None:
            _BODY_PLAY["last_frame"] = int(scene.frame_current)
        _hold_idle_pose(arm)
        try:
            _zero_mouth_shapes_if_idle()
        except Exception:
            pass
        return

    # Review mode: keep session master bound (not last received clip)
    if (
        (not _BODY_PLAY.get("live"))
        and _BODY_PLAY.get("scrub_user_active")
        and (_SESSION_BODY.get("clips") or [])
    ):
        sess_name = _session_action_name()
        sess_act = bpy.data.actions.get(sess_name)
        if sess_act is not None:
            if not arm.animation_data or arm.animation_data.action != sess_act:
                _assign_session_action_for_scrub(arm, allow_during_idle=True)
            try:
                _zero_mouth_shapes_if_idle()
            except Exception:
                pass
            return

    name = _BODY_PLAY.get("action") or _BODY_PLAY.get("last_action")
    if not name:
        # No clip — still hold idle if we have one
        if _BODY_PLAY.get("idle_pose"):
            _BODY_PLAY["idle_hold"] = True
            _hold_idle_pose(arm)
        return
    act = bpy.data.actions.get(name)
    if act is None:
        if _BODY_PLAY.get("idle_pose") or True:
            # Clip missing (e.g. not loaded) — fall back to hands-down idle
            if not _BODY_PLAY.get("idle_pose"):
                idle, src = _resolve_idle_rest_pose(arm)
                _BODY_PLAY["idle_pose"] = idle
                print(f"[body_receiver] missing Action {name!r} → idle hold ({src})")
            _BODY_PLAY["idle_hold"] = True
            _hold_idle_pose(arm)
        return

    # After rest without idle_hold: user owns the timeline — bind SESSION master
    if not _BODY_PLAY.get("live"):
        if _BODY_PLAY.get("scrub_ready") and (_SESSION_BODY.get("clips") or []):
            if _user_reviewing_session(scene):
                _assign_session_action_for_scrub(arm, allow_during_idle=True)
                _BODY_PLAY["last_frame"] = int(scene.frame_current)
        try:
            _zero_mouth_shapes_if_idle()
        except Exception:
            pass
        return

    # --- live streaming: wall clock locked to audio play_t0 ---
    if not arm.animation_data:
        arm.animation_data_create()

    # CRITICAL: never leave an Action bound during live — Blender will re-evaluate
    # it every redraw at scene.frame_current (origin / fly-Z / teleport).
    try:
        if arm.animation_data and arm.animation_data.action is not None:
            arm.animation_data.action = None
    except Exception:
        pass

    if _BODY_PLAY.get("pending_assign"):
        # Live: ONLY fast_eval on source Action — never bind source or Session_Timeline
        if act is not None:
            try:
                _get_action_fcu_targets(act)
            except Exception:
                pass
        _BODY_PLAY["pending_assign"] = False
        _BODY_PLAY["last_frame"] = None
        try:
            arm.hide_set(False)
            arm.hide_viewport = False
        except Exception:
            pass

    f0 = int(_BODY_PLAY.get("f0") or 1)
    f1 = int(_BODY_PLAY.get("f1") or 60)
    speed = float(_BODY_PLAY.get("speed") or 1.0)
    # Re-assert: MoMask never loops mid-play (state walking must not force 2nd cycle)
    act_name_live = str(_BODY_PLAY.get("action") or name or "")
    eng_live = str(_BODY_PLAY.get("engine") or "").lower()
    if eng_live == "momask" or act_name_live.lower().startswith("momask_"):
        loop = False
        _BODY_PLAY["loop"] = False
    else:
        loop = bool(_BODY_PLAY.get("loop", True))
    elapsed = max(0.0, time.time() - float(_BODY_PLAY.get("t0") or time.time()))
    duration = _BODY_PLAY.get("duration")

    try:
        _hydrate_session_clips_from_scene()
    except Exception:
        pass
    sess_name = _session_action_name()
    sess_act = bpy.data.actions.get(sess_name)
    if sess_act is None:
        # Fall back if cursor-save once wrote the wrong name
        alt = _resolve_existing_session_action_name()
        sess_act = bpy.data.actions.get(alt)
        if sess_act is not None:
            _SESSION_BODY["action_name"] = alt
            sess_name = alt
    sf0 = int(_BODY_PLAY.get("session_frame_start") or 0)
    sf1 = int(_BODY_PLAY.get("session_frame_end") or 0)
    use_session = bool(_BODY_PLAY.get("session_play")) and sess_act is not None and sf1 >= sf0 > 0
    if use_session:
        loop = False
        _BODY_PLAY["loop"] = False

    fps = float(_BODY_PLAY.get("clip_fps") or MASTER_EVAL_FPS)
    if use_session or eng_live == "momask" or act_name_live.lower().startswith("momask_"):
        if fps > 22.0 or fps < 1.0:
            fps = MASTER_EVAL_FPS
            _BODY_PLAY["clip_fps"] = MASTER_EVAL_FPS
    if fps < 1:
        fps = MASTER_EVAL_FPS

    play_act = sess_act if use_session else act
    play_f0 = sf0 if use_session else f0
    play_f1 = sf1 if use_session else f1

    # End of take: hold last pose on the session (no catalog idle, no origin)
    if duration is not None and elapsed >= float(duration):
        if use_session:
            _hold_live_session_end(arm, play_act, play_f1)
        else:
            _start_body_rest_blend(arm, rest_duration=0.55)
        try:
            _zero_mouth_shapes_if_idle()
        except Exception:
            pass
        return

    span = max(1, play_f1 - play_f0)
    if loop and not use_session:
        raw = play_f0 + elapsed * fps * speed
        frame = play_f0 + (raw - play_f0) % span
    else:
        raw = play_f0 + elapsed * fps * max(0.35, min(2.5, speed if speed > 0.01 else 1.0))
        frame = min(float(play_f1), max(float(play_f0), raw))
        if raw >= play_f1:
            _hold_live_session_end(arm, play_act, play_f1)
            try:
                _zero_mouth_shapes_if_idle()
            except Exception:
                pass
            return

    frame_f = float(frame)
    frame_i = int(round(frame_f))
    in_clip_trans = (not use_session) and _clip_transition_active()
    if (
        frame_i == _BODY_PLAY.get("last_frame")
        and _BODY_PLAY.get("fast_eval")
        and not in_clip_trans
    ):
        try:
            _zero_mouth_shapes_if_idle()
        except Exception:
            pass
        return

    use_fast = bool(_BODY_PLAY.get("fast_eval", True)) and _BODY_USE_FAST_EVAL
    applied = False
    if use_fast:
        if arm.animation_data and arm.animation_data.action is not None:
            arm.animation_data.action = None
        applied = _apply_action_frame_fast(arm, play_act, frame_f)

    if not applied:
        try:
            if arm.animation_data is None:
                arm.animation_data_create()
            arm.animation_data.action = play_act
            scene.frame_set(frame_i)
            bpy.context.view_layer.update()
            _BODY_PLAY["_wrote_pelvis_loc"] = True
            applied = True
            arm.animation_data.action = None
        except Exception as e:
            if VERBOSE_PACKETS:
                print(f"[body_receiver] frame apply failed: {e}")

    if applied:
        # Session keys already include root_dx/dy — do not add origin offset again
        if not use_session:
            _apply_root_offset(arm)
        if in_clip_trans or _BODY_PLAY.get("trans_from"):
            _apply_clip_transition_blend(arm)
        try:
            pb_now = arm.pose.bones.get("pelvis")
            if pb_now is not None:
                _session_root_set(pb_now.location)
                # World XY for next-clip continuity (catalog in-place still keeps place)
                w = arm.matrix_world @ pb_now.head
                _BODY_PLAY["world_root_xy"] = (float(w.x), float(w.y))
                try:
                    bpy.context.scene["session_world_xy"] = [
                        float(w.x), float(w.y), float(w.z)
                    ]
                except Exception:
                    pass
        except Exception:
            pass
        _BODY_PLAY["last_frame"] = frame_i
    try:
        _zero_mouth_shapes_if_idle()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# PRINT SHAPE KEYS
# ---------------------------------------------------------------------------
def _print_shape_keys():
    configured = TARGET_MESH_NAME
    obj = _get_target_object()
    if obj is None:
        print("[face_receiver] No suitable mesh found automatically.")
        print("[face_receiver] Set TARGET_MESH_NAME manually.")
        return
    used_auto = (not configured) or (configured != obj.name)
    tag = "(auto-detected + selected)" if used_auto else ""
    print(f"[face_receiver] Target mesh: {obj.name} {tag}".strip())

    if obj.data.shape_keys is None:
        print("[face_receiver]   Mesh has no shape keys.")
        return
    print("[face_receiver] Available shape keys:")
    for kb in obj.data.shape_keys.key_blocks:
        print("   -", kb.name)

# ---------------------------------------------------------------------------
# OPERATORS
# ---------------------------------------------------------------------------
_last_app_tick = 0.0


def _receiver_app_timer():
    """Drain UDP + tick body even when the modal operator did not stay active.

    MCP / script-invoked stream_receiver() often never keeps a modal handler.
    A persistent app timer is what actually applies orchestrator body packets.
    """
    global _last_app_tick, _last_packet_time
    if not _running:
        return None
    now = time.time()
    if now - float(_last_app_tick or 0.0) < 0.012:
        return 1.0 / 30.0
    _last_app_tick = now
    try:
        while not _data_queue.empty():
            try:
                pkt = _data_queue.get_nowait()
            except Exception:
                break
            if _last_packet_time is None:
                print("[face_receiver] *** CONNECTED to orchestrator! ***")
            _last_packet_time = now
            try:
                _handle_packet(pkt)
            except Exception as e:
                print(f"[face_receiver] timer handle: {e}")
        try:
            _apply_body_play(bpy.context.scene)
        except Exception as e:
            print(f"[body_receiver] timer play: {e}")
        try:
            _apply_camera_play(bpy.context.scene)
        except Exception:
            pass
    except Exception as e:
        print(f"[face_receiver] app timer: {e}")
    return 1.0 / 30.0


def _ensure_receiver_app_timer() -> None:
    try:
        if not bpy.app.timers.is_registered(_receiver_app_timer):
            bpy.app.timers.register(_receiver_app_timer, first_interval=0.03, persistent=True)
            print("[face_receiver] app timer registered (body/face UDP)")
    except Exception as e:
        print(f"[face_receiver] app timer register failed: {e}")


class FACE_OT_stream_receiver(bpy.types.Operator):
    bl_idname = "face.stream_receiver"
    bl_label = "Start MediaPipe / Orchestrator Face Stream"

    _timer = None

    def modal(self, context, event):
        global _last_packet_time

        if not _running:
            return self.cancel(context)

        if event.type == "TIMER":
            now = time.time()

            # Drain queue — process EVERY packet (do not keep only the last).
            # Multi-agent orchestrator sends viseme + emotion + head each frame;
            # dropping all but the last would leave only "head" and lips never move.
            got_any = False
            while not _data_queue.empty():
                try:
                    pkt = _data_queue.get_nowait()
                except Exception:
                    break
                if _last_packet_time is None:
                    print("[face_receiver] *** CONNECTED to orchestrator! ***")
                    print("[face_receiver] Receiving emotion/viseme/head packets (all types applied).")
                _last_packet_time = now
                got_any = True
                try:
                    _handle_packet(pkt)
                except Exception as e:
                    print(f"[face_receiver] Error handling packet: {e}")
            try:
                _drain_studio_commands()
            except Exception:
                pass

            # Live face glide vs timeline scrub:
            # - While speaking (UDP live): drive shape keys from TARGET_VALUES
            # - After speech, if face timeline is baked: do NOT overwrite shape keys
            #   so scrubbing the timeline evaluates keyframes (lips move on replay)
            _live_face = bool(IS_SPEAKING) or (
                _last_packet_time is not None and (now - _last_packet_time) < 0.35
            )
            _use_timeline_face = (
                _FACE_TIMELINE.get("active")
                and not _live_face
            )

            if not _use_timeline_face:
                # Glide mesh toward targets. Mouth uses faster VISEME_GLIDE so lips open fully.
                try:
                    obj = _get_target_object()
                    if obj and obj.data.shape_keys:
                        key_blocks = obj.data.shape_keys.key_blocks
                        for key_name in list(_tracked_keys):
                            target = TARGET_VALUES.get(key_name, 0.0)
                            current = _smoothed_values.get(key_name, 0.0)
                            is_upper = key_name in UPPER_FACE_KEYS or key_name.startswith("eye") or key_name.startswith("brow")
                            rate = EMOTION_GLIDE if is_upper else VISEME_GLIDE
                            new_val = current * (1.0 - rate) + target * rate
                            _smoothed_values[key_name] = new_val
                            if key_name in key_blocks:
                                key_blocks[key_name].value = new_val
                except Exception as e:
                    print(f"[face_receiver] Error in gliding: {e}")

                # Drive secondary meshes (eye balls + teeth) with smoothed values.
                try:
                    for _sec_name in ("eyeLeft_ORIGINAL", "eyeRight_ORIGINAL", "teeth_ORIGINAL"):
                        _sec_obj = bpy.data.objects.get(_sec_name)
                        if _sec_obj is None or _sec_obj.data is None or _sec_obj.data.shape_keys is None:
                            continue
                        _sec_kb = _sec_obj.data.shape_keys.key_blocks
                        for _kn, _v in _smoothed_values.items():
                            if _kn in _sec_kb:
                                _sec_kb[_kn].value = float(_v)
                except Exception as _e:
                    print(f"[face_receiver] Secondary mesh error: {_e}")

                # Apply head rotation from main thread (live only)
                try:
                    _head_obj = bpy.data.objects.get("HEAD_CONTROLLER")
                    if _head_obj:
                        _head_obj.rotation_euler[0] = _HEAD_ROTATION["pitch"]
                        _head_obj.rotation_euler[1] = _HEAD_ROTATION["roll"]
                        _head_obj.rotation_euler[2] = _HEAD_ROTATION["yaw"]
                except Exception:
                    pass
            # else: timeline owns face — scrub Space / timeline to evaluate shape-key Action

            # Body: play retargeted Mixamo Actions on SMPL-X_Armature
            try:
                _apply_body_play(context.scene)
            except Exception as e:
                if VERBOSE_PACKETS:
                    print(f"[body_receiver] play error: {e}")

            # Movie camera path (independent of body Action frames)
            try:
                _apply_camera_play(context.scene)
            except Exception as e:
                if VERBOSE_PACKETS:
                    print(f"[camera_receiver] play error: {e}")

            # Reset mediapipe active flag if no recent legacy packets
            global MEDIAPIPE_ACTIVE
            if time.time() - _last_mediapipe_time > 2.0:
                MEDIAPIPE_ACTIVE = False

            # Apply any pending blink values directly (bypass normal queue for procedural blinks)
            obj = _get_target_object()
            if obj and obj.data.shape_keys:
                key_blocks = obj.data.shape_keys.key_blocks
                while not _blink_queue.empty():
                    try:
                        blink_dict = _blink_queue.get_nowait()
                        for k, v in blink_dict.items():
                            if k in key_blocks:
                                key_blocks[k].value = v
                    except Exception:
                        break

            # Auto-timeout logic explained:
            #
            # === FIRST SIGNAL ===
            # _last_packet_time starts as None.
            # The very first packet the receiver ever sees is the "first signal".
            # We only care about how long since we *started* the operator (_start_time).
            # If no packet arrives within FIRST_SIGNAL_TIMEOUT (120s), we stop.
            # This protects you if you start the receiver but forget to start the sender.
            #
            # When the first packet arrives:
            #   - We print "*** CONNECTED to orchestrator! ***"
            #   - We set _last_packet_time = now
            #   - From now on we are in "connected" mode.
            #
            # === ONGOING / SECOND AND LATER SIGNALS ===
            # After the first signal, every new packet is a "subsequent signal".
            # Each one updates _last_packet_time.
            # If more than POST_CONNECTION_SILENCE_TIMEOUT passes with *no packets at all*,
            # we assume the sender has permanently stopped and we auto-cancel.
            #
            # Important: once connected, we do NOT stop after short gaps anymore.
            # We wait for a very long silence (or explicit bpy.ops.face.stream_stop()).
            # This is what you wanted: "wait till exit of sender once connected".
            if _last_packet_time is None:
                # Still waiting for the very first packet ("first signal")
                if _start_time is not None and (now - _start_time) > FIRST_SIGNAL_TIMEOUT:
                    self.report({"WARNING"}, f"No signal received in {FIRST_SIGNAL_TIMEOUT:.0f}s. Stopping.")
                    return self.cancel(context)
            else:
                # Connected — only stop after very long total silence since last packet
                if (now - _last_packet_time) > POST_CONNECTION_SILENCE_TIMEOUT:
                    self.report({"INFO"}, "No data for a very long time after connection. Auto-stopping for safety.")
                    return self.cancel(context)

        return {"PASS_THROUGH"}

    def execute(self, context):
        global _running, _listener_thread, _start_time, _last_packet_time, _smoothed_values

        obj = _get_target_object()
        _print_shape_keys()

        if obj is None:
            self.report({"ERROR"}, "No target mesh with shape keys found.")
            return {"CANCELLED"}

        _select_mesh(obj)

        # Report body Actions available (Mixamo retarget pack)
        _body_acts = [
            a.name for a in bpy.data.actions
            if a.name in _BODY_ACTION_ALIASES.values() or a.name in set(_BODY_ACTION_ALIASES.values())
        ]
        _named = sorted({v for v in _BODY_ACTION_ALIASES.values()})
        _have = [n for n in _named if bpy.data.actions.get(n)]
        print(f"[body_receiver] SMPL-X Actions ready: {len(_have)}/{len(_named)} → {_have[:12]}{'…' if len(_have) > 12 else ''}")
        if not _have:
            print("[body_receiver] WARN: no retargeted Actions. Run tools/retarget_mixamo_to_smplx.py first.")

        # Restore session memory; only snap to origin rest when there is NO session yet.
        # Re-Start used to always apply pipeline rest at origin → wiped walk position.
        try:
            _hydrate_session_clips_from_scene()
            arm0 = _body_armature("SMPL-X_Armature")
            has_clips = bool(_SESSION_BODY.get("clips"))
            if arm0 is not None and not has_clips:
                idle, src = _resolve_idle_rest_pose(arm0)
                if arm0.animation_data:
                    arm0.animation_data.action = None
                pel0 = _session_root_get(arm0)
                _apply_pose_dict(arm0, idle, pel0)
                _BODY_PLAY["idle_pose"] = idle
                _BODY_PLAY["rest_to"] = idle
                _BODY_PLAY["idle_hold"] = True
                _BODY_PLAY["live"] = False
                print(f"[body_receiver] startup REST applied ({src}) — empty session")
            elif has_clips:
                print(
                    f"[body_receiver] startup keep session "
                    f"action={_session_action_name()!r} "
                    f"clips={len(_SESSION_BODY.get('clips') or [])} "
                    f"(no origin rest)"
                )
                _BODY_PLAY["idle_hold"] = False
                _BODY_PLAY["scrub_ready"] = True
        except Exception as e:
            print(f"[body_receiver] startup rest/hydrate failed: {e}")

        _smoothed_values.clear()
        _start_time = time.time()
        _last_packet_time = None

        _running = True
        _listener_thread = threading.Thread(target=_listener_loop, daemon=True)
        _listener_thread.start()

        global _blink_thread
        _blink_thread = threading.Thread(target=_blink_loop, daemon=True)
        _blink_thread.start()

        global _eye_gaze_thread, _head_thread
        _eye_gaze_thread = threading.Thread(target=eye_gaze_loop, daemon=True)
        _eye_gaze_thread.start()
        print("[face_receiver] eye_gaze_loop thread started")

        _head_thread = threading.Thread(target=head_movement_loop, daemon=True)
        _head_thread.start()
        print("[face_receiver] head_movement_loop thread started (breathing + head movement active)")

        print("[face_receiver] All background threads (blink, gaze, head) started. Check console for periodic [head_loop] and [eye_gaze_loop] messages.")

        wm = context.window_manager
        self._timer = wm.event_timer_add(1.0 / 30.0, window=context.window)
        wm.modal_handler_add(self)
        _ensure_receiver_app_timer()
        self.report({"INFO"}, f"Listening on {UDP_IP}:{UDP_PORT} (will stay connected after first packet; use stream_stop() to exit)")
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        global _running, _sock
        _running = False
        try:
            if _sock:
                _sock.close()
                _sock = None
        except Exception:
            pass
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None
        return {"CANCELLED"}


class FACE_OT_stream_stop(bpy.types.Operator):
    bl_idname = "face.stream_stop"
    bl_label = "Stop Face Stream"

    def execute(self, context):
        global _running, _sock
        _running = False
        try:
            if _sock:
                _sock.close()
                _sock = None
        except Exception:
            pass
        self.report({"INFO"}, "Stopped listening.")
        return {"FINISHED"}


def _clip_at_frame(frame: int):
    for c in _SESSION_BODY.get("clips") or []:
        if int(c.get("frame_start") or 0) <= int(frame) <= int(c.get("frame_end") or 0):
            return c
    return None


def _speech_block_at(frame: int):
    """Return (clip, s0, s1) if frame is inside spoken audio."""
    f = int(frame)
    for c in _SESSION_BODY.get("clips") or []:
        s0 = int(c.get("speech_frame_start") or 0)
        s1 = int(c.get("speech_frame_end") or 0)
        if s1 >= s0 > 0 and s0 <= f <= s1:
            return c, s0, s1
    return None


def _range_overlaps_speech_interior(a: int, b: int):
    """True if [a,b] cuts through the middle of a spoken line (not just touching an edge)."""
    a, b = int(min(a, b)), int(max(a, b))
    for c in _SESSION_BODY.get("clips") or []:
        s0 = int(c.get("speech_frame_start") or 0)
        s1 = int(c.get("speech_frame_end") or 0)
        if s1 < s0 or s0 <= 0:
            continue
        if a < s1 and b > s0 and (a > s0 or b < s1):
            return c, s0, s1
    return None


def _mark_movie_dirty(reason: str = "") -> None:
    _SESSION_BODY["needs_movie_rerender"] = True
    if reason:
        print(f"[session] needs_movie_rerender=1 ({reason})")


def _snap_range_speech_safe(a: int, b: int):
    """
    Best speech rule: never split a spoken line.
    If the edit hits speech interior, snap to the FULL speech block (or whole clip).
    """
    try:
        from face_agents.speech_safe import snap_range_speech_safe
        return snap_range_speech_safe(_SESSION_BODY.get("clips") or [], a, b)
    except Exception:
        hit = _range_overlaps_speech_interior(a, b)
        if not hit:
            return int(a), int(b), ""
        c, s0, s1 = hit
        return (
            min(int(a), s0),
            max(int(b), s1),
            f"snapped to speech S#{c.get('index')} [{s0}-{s1}] (do not cut mid-line)",
        )


def _clear_session_keys_span(frame_start: int, frame_end: int) -> int:
    """Remove master keys in [frame_start, frame_end]."""
    act = bpy.data.actions.get(_session_action_name())
    if act is None:
        return 0
    a = float(min(frame_start, frame_end)) - 0.01
    b = float(max(frame_start, frame_end)) + 0.01
    n = 0
    for fcu in _iter_action_fcurves(act):
        try:
            kps = list(fcu.keyframe_points)
        except Exception:
            continue
        for kp in reversed(kps):
            try:
                t = float(kp.co[0])
                if a <= t <= b:
                    fcu.keyframe_points.remove(kp)
                    n += 1
            except Exception:
                pass
        try:
            fcu.update()
        except Exception:
            pass
    return n


def _delete_session_clips_overlapping(frame_start: int, frame_end: int):
    """Speech-safe: drop whole clips whose body or speech overlaps the range."""
    global _SESSION_BODY
    a, b = int(min(frame_start, frame_end)), int(max(frame_start, frame_end))
    clips = list(_SESSION_BODY.get("clips") or [])
    keep = []
    removed = []
    for c in clips:
        s0 = int(c.get("speech_frame_start") or 0)
        s1 = int(c.get("speech_frame_end") or 0)
        f0 = int(c.get("frame_start") or 1)
        f1 = int(c.get("frame_end") or f0)
        hit_speech = s1 >= s0 > 0 and a < s1 and b > s0
        hit_body = f1 >= a and f0 <= b
        if hit_speech or hit_body:
            removed.append(c)
        else:
            keep.append(c)
    n_keys = 0
    for c in removed:
        f0 = int(c.get("frame_start") or 1)
        f1 = int(c.get("frame_end") or f0)
        n_keys += _clear_session_keys_span(f0, f1)
        idx = int(c.get("index") or 0)
        try:
            _vse_remove_named(f"M#{idx}_")
            _vse_remove_named(f"S#{idx}_")
        except Exception:
            pass
    _SESSION_BODY["clips"] = keep
    _mark_movie_dirty("delete_range")
    if keep:
        last = keep[-1]
        _session_cursor_save(int(last.get("frame_end") or 1) + 1)
        sc = bpy.context.scene
        sc.frame_end = max(1, int(last.get("frame_end") or 1))
    else:
        _session_cursor_save(1)
        arm = _body_armature()
        _apply_origin_custom_rest(arm)
        sc = bpy.context.scene
        sc.frame_start = 1
        sc.frame_end = 250
        sc.frame_current = 1
    _refresh_markers_from_clips()
    try:
        _persist_session_clip_table()
    except Exception:
        pass
    return removed, n_keys


def _clear_session_keys_after(frame_keep: int) -> int:
    """Remove master keys after frame_keep (truncate last clip)."""
    act = bpy.data.actions.get(_session_action_name())
    if act is None:
        return 0
    n = 0
    cut = float(frame_keep) + 0.01
    for fcu in _iter_action_fcurves(act):
        try:
            kps = list(fcu.keyframe_points)
        except Exception:
            continue
        for kp in reversed(kps):
            try:
                if float(kp.co[0]) > cut:
                    fcu.keyframe_points.remove(kp)
                    n += 1
            except Exception:
                pass
        try:
            fcu.update()
        except Exception:
            pass
    return n


def _refresh_markers_from_clips() -> None:
    _clear_session_timeline_markers()
    try:
        sc = bpy.context.scene
        sc.timeline_markers.new("Session|start", frame=1)
    except Exception:
        pass
    for c in _SESSION_BODY.get("clips") or []:
        _add_session_clip_markers(
            index=int(c.get("index") or 0),
            frame_start=int(c.get("frame_start") or 1),
            frame_end=int(c.get("frame_end") or 1),
            display_label=str(c.get("label") or f"#{c.get('index')}"),
            source_action=str(c.get("source_action") or ""),
            speech_frame_start=int(c.get("speech_frame_start") or 0),
            speech_frame_end=int(c.get("speech_frame_end") or 0),
            speech_text=str(c.get("text") or ""),
            audio_path=str(c.get("audio_path") or ""),
        )
    _persist_session_clip_table()


# ---------------------------------------------------------------------------
# SESSION SIDEBAR (N-panel chat + user buttons)
# ---------------------------------------------------------------------------
def _chat_dir():
    from pathlib import Path
    for root in _project_roots():
        d = root / "temp" / "blender_chat"
        try:
            d.mkdir(parents=True, exist_ok=True)
            return d
        except Exception:
            continue
    return Path(r"C:\me\proj\projface_v1\temp\blender_chat")


def _chat_history_path():
    return _chat_dir() / "history.jsonl"


def _text_looks_like_json(text: str) -> bool:
    s = str(text or "").strip()
    return (s.startswith("{") and s.endswith("}")) or (
        s.startswith("[") and s.endswith("]")
    )


def _chat_history_append(role: str, text: str, *, kind: str = "", payload: str = "") -> dict:
    import json as _json
    raw = str(text or "")
    role_s = str(role or "sys")
    kind_s = str(kind or "").strip().lower()
    if not kind_s:
        if role_s == "json" or _text_looks_like_json(raw):
            kind_s = "json"
        elif role_s == "user":
            kind_s = "chat"
        elif role_s == "sys":
            kind_s = "sys"
        else:
            kind_s = "chat"
    rec = {
        "t": time.time(),
        "role": role_s,
        "kind": kind_s,
        "text": raw,
        "payload": str(payload or ""),
    }
    try:
        p = _chat_history_path()
        with p.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return rec


def _drain_studio_commands() -> None:
    """Studio window writes temp/blender_chat/commands.jsonl; run here on the main thread."""
    import json as _json
    p = _chat_dir() / "commands.jsonl"
    if not p.is_file():
        return
    try:
        rows = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return
    if not rows:
        return
    try:
        p.write_text("", encoding="utf-8")
    except Exception:
        return
    for ln in rows:
        try:
            cmd = _json.loads(ln)
        except Exception:
            continue
        op = str((cmd or {}).get("op") or "").lower().strip()
        if not op:
            continue
        try:
            if op in ("export_preview", "preview"):
                bpy.ops.session.export_preview()
            elif op in ("export_movie", "movie"):
                try:
                    bpy.ops.session.export_movie()
                except Exception:
                    bpy.ops.session.export_preview()
            elif op == "play":
                bpy.ops.session.play_timeline()
            elif op == "reset":
                bpy.ops.session.reset_session()
            elif op == "trim":
                sc = bpy.context.scene
                sc.session_inpaint_start = int(cmd.get("start") or sc.session_inpaint_start)
                sc.session_inpaint_end = int(cmd.get("end") or sc.session_inpaint_end)
                bpy.ops.session.trim_clip()
            elif op == "inpaint":
                sc = bpy.context.scene
                sc.session_inpaint_start = int(cmd.get("start") or 1)
                sc.session_inpaint_end = int(cmd.get("end") or 40)
                sc.session_inpaint_prompt = str(cmd.get("prompt") or "")
                bpy.ops.session.inpaint()
            elif op in ("delete", "delete_range"):
                sc = bpy.context.scene
                sc.session_inpaint_start = int(cmd.get("start") or 1)
                sc.session_inpaint_end = int(cmd.get("end") or 40)
                bpy.ops.session.delete_range()
            elif op == "delete_last":
                bpy.ops.session.delete_last()
            print(f"[studio] command {op}")
        except Exception as e:
            print(f"[studio] command {op} failed: {e}")


def _pipeline_mode_path():
    return _chat_dir() / "pipeline_mode.txt"


def _read_pipeline_mode() -> str:
    """Sticky chat mode shared with orchestrator: 'full' | 'scene'."""
    try:
        p = _pipeline_mode_path()
        if p.is_file():
            m = p.read_text(encoding="utf-8").strip().lower()
            if m in ("scene", "look", "set"):
                return "scene"
            if m in ("full", "chat", "performance"):
                return "full"
    except Exception:
        pass
    return "full"


def _write_pipeline_mode(mode: str) -> str:
    m = str(mode or "full").strip().lower()
    if m in ("scene", "look", "set"):
        m = "scene"
    else:
        m = "full"
    try:
        _chat_dir().mkdir(parents=True, exist_ok=True)
        _pipeline_mode_path().write_text(m, encoding="utf-8")
    except Exception as e:
        print(f"[chat] pipeline_mode write: {e}")
    return m


def _on_pipeline_mode_update(self, context):
    """N-panel toggle → sticky file so orchestrator stays in scene/full."""
    mode = str(getattr(self, "session_pipeline_mode", "FULL") or "FULL").upper()
    written = _write_pipeline_mode("scene" if mode == "SCENE" else "full")
    label = "SCENE (Look_Set only)" if written == "scene" else "FULL (scene+motion+speech)"
    _chat_history_append("sys", f"Mode → {label}", kind="sys")


def _chat_push_inbox(text: str, mode: str = "", *, persist_mode: bool = True) -> None:
    import json as _json
    raw = str(text or "").strip()
    p = _chat_dir() / "inbox.jsonl"
    mode_s = str(mode or "").strip().lower()
    if mode_s in ("look", "set"):
        mode_s = "scene"
    if mode_s not in ("full", "scene"):
        mode_s = _read_pipeline_mode()
    if persist_mode:
        _write_pipeline_mode(mode_s)
    rec = {"t": time.time(), "text": raw, "src": "blender", "mode": mode_s}
    with p.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    flag = _chat_dir() / "ui_active"
    try:
        flag.write_text(str(time.time()), encoding="utf-8")
    except Exception:
        pass
    # Show the user bubble immediately (right side).
    kind = "json" if _text_looks_like_json(raw) else "chat"
    prefix = "[scene] " if mode_s == "scene" and kind == "chat" else ""
    _chat_history_append("user", f"{prefix}{raw}", kind=kind)


def _chat_read_outbox(limit: int = 16):
    import json as _json
    p = _chat_dir() / "outbox.jsonl"
    if not p.is_file():
        return []
    try:
        rows = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return []
    out = []
    for ln in rows[-limit:]:
        try:
            out.append(_json.loads(ln))
        except Exception:
            out.append({"role": "sys", "kind": "sys", "text": ln})
    return out


def _chat_read_history(limit: int = 80):
    import json as _json
    merged = []
    seen = set()

    def _eat(path, default_role="sys"):
        if not path.is_file():
            return
        try:
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception:
            return
        for ln in lines:
            try:
                rec = _json.loads(ln)
            except Exception:
                rec = {"role": default_role, "kind": "sys", "text": ln, "t": 0.0}
            if not isinstance(rec, dict):
                continue
            role = str(rec.get("role") or "")
            text = str(rec.get("text") or "")
            kind = str(rec.get("kind") or "")
            if role == "user":
                key = ("user", text[:240])
            else:
                key = (
                    round(float(rec.get("t") or 0.0), 3),
                    role,
                    kind,
                    text[:160],
                )
            if key in seen:
                continue
            seen.add(key)
            merged.append(rec)

    _eat(_chat_history_path())
    _eat(_chat_dir() / "outbox.jsonl")
    merged.sort(key=lambda r: float(r.get("t") or 0.0))
    return merged[-int(limit) :]


def _orch_pid_path():
    return _chat_dir().parent / "orchestrator.pid"


def _orch_python_exe():
    """Host Python for orchestrator_agents.py — never Blender's bundled interpreter."""
    import os
    import shutil
    from pathlib import Path
    env = (os.environ.get("PROJFACE_PYTHON") or "").strip()
    if env and Path(env).is_file():
        return env
    which = shutil.which("python") or shutil.which("python.exe")
    if which:
        return which
    for root in _project_roots():
        for rel in (
            Path("third_party") / "momask_venv" / "Scripts" / "python.exe",
            Path(".venv") / "Scripts" / "python.exe",
        ):
            p = root / rel
            if p.is_file():
                return str(p)
    home = Path(os.environ.get("LOCALAPPDATA") or "") / "Programs" / "Python"
    for name in ("Python313", "Python312", "Python311"):
        p = home / name / "python.exe"
        if p.is_file():
            return str(p)
    return None


def _orch_is_running():
    import os
    p = _orch_pid_path()
    if not p.is_file():
        return False, 0
    try:
        pid = int((p.read_text(encoding="utf-8") or "0").strip() or "0")
    except Exception:
        return False, 0
    if pid <= 0:
        return False, 0
    try:
        os.kill(pid, 0)
        return True, pid
    except OSError:
        return False, pid


class SESSION_OT_start_orchestrator(bpy.types.Operator):
    bl_idname = "session.start_orchestrator"
    bl_label = "Start orchestrator"
    bl_description = "Launch orchestrator_agents.py (MoMask + director chat) and the face stream"

    def execute(self, context):
        import os
        import subprocess
        from pathlib import Path

        running, pid = _orch_is_running()
        if running:
            # Still ensure the UDP receiver is up so packets land.
            if not _running:
                try:
                    bpy.ops.face.stream_receiver()
                except Exception as e:
                    print(f"[session_ui] stream start: {e}")
            _ensure_receiver_app_timer()
            try:
                (_chat_dir() / "ui_active").write_text(str(time.time()), encoding="utf-8")
            except Exception:
                pass
            self.report({"INFO"}, f"Orchestrator already running (pid {pid})")
            return {"FINISHED"}

        root = None
        for r in _project_roots():
            if (r / "orchestrator_agents.py").is_file():
                root = r
                break
        if root is None:
            self.report({"ERROR"}, "orchestrator_agents.py not found")
            return {"CANCELLED"}

        exe = _orch_python_exe()
        if not exe:
            self.report({"ERROR"}, "No host Python found (set PROJFACE_PYTHON)")
            return {"CANCELLED"}

        env = os.environ.copy()
        env["LLM_CHAT"] = "1"
        env["USE_MOMASK"] = "1"
        env["MOMASK_ALL"] = "0"
        env["MOMASK_SYNC"] = "1"
        env["USE_BRAIN"] = env.get("USE_BRAIN") or "0"
        env["USE_MOVIE_CAMERA"] = "1"
        env["USE_MOVIE_LOOK"] = env.get("USE_MOVIE_LOOK") or "1"
        env["PROJFACE_ROOT"] = str(root)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["BLENDER_SIDEBAR"] = "1"
        log_dir = root / "temp"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        log_path = log_dir / "orchestrator_addon.log"
        try:
            log_f = open(log_path, "a", encoding="utf-8")
        except Exception:
            log_f = subprocess.DEVNULL

        creation = 0
        if os.name == "nt":
            # CREATE_NO_WINDOW keeps stdout on our log file. DETACHED_PROCESS
            # drops the handle and the child can exit silently.
            creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creation |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        try:
            proc = subprocess.Popen(
                [exe, "-u", str(root / "orchestrator_agents.py")],
                cwd=str(root),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=creation,
            )
        except Exception as e:
            self.report({"ERROR"}, f"Launch failed: {e}")
            return {"CANCELLED"}

        try:
            _orch_pid_path().write_text(str(proc.pid), encoding="utf-8")
        except Exception:
            pass
        try:
            (_chat_dir() / "ui_active").write_text(str(time.time()), encoding="utf-8")
        except Exception:
            pass

        if not _running:
            try:
                bpy.ops.face.stream_receiver()
            except Exception as e:
                print(f"[session_ui] stream start: {e}")
        _ensure_receiver_app_timer()

        self.report({"INFO"}, f"Orchestrator started pid={proc.pid}")
        print(f"[session_ui] orchestrator pid={proc.pid} exe={exe} cwd={root}")
        _chat_history_append("sys", f"Orchestrator started  pid {proc.pid}", kind="sys")
        return {"FINISHED"}


def _orch_stop_process() -> tuple:
    import os
    import subprocess
    running, pid = _orch_is_running()
    if not running:
        try:
            p = _orch_pid_path()
            if p.is_file():
                p.unlink()
        except Exception:
            pass
        return False, int(pid or 0)
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=12,
            )
        except Exception as e:
            print(f"[session_ui] taskkill: {e}")
            try:
                os.kill(pid, 9)
            except Exception:
                pass
    else:
        try:
            os.kill(pid, 15)
        except Exception:
            try:
                os.kill(pid, 9)
            except Exception:
                pass
    try:
        p = _orch_pid_path()
        if p.is_file():
            p.unlink()
    except Exception:
        pass
    return True, int(pid)


class SESSION_OT_stop_orchestrator(bpy.types.Operator):
    bl_idname = "session.stop_orchestrator"
    bl_label = "Stop orchestrator"
    bl_description = "Stop orchestrator_agents.py started from this panel"

    def execute(self, context):
        stopped, pid = _orch_stop_process()
        if stopped:
            _chat_history_append("sys", f"Orchestrator stopped  pid {pid}", kind="sys")
            self.report({"INFO"}, f"Orchestrator stopped (pid {pid})")
        else:
            self.report({"WARNING"}, "Orchestrator is not running")
        return {"FINISHED"}


class SESSION_OT_open_studio(bpy.types.Operator):
    bl_idname = "session.open_studio"
    bl_label = "Open Studio window"
    bl_description = "Open the separate Studio chat (Start/Stop, history, JSON, inpaint, export)"

    def execute(self, context):
        import os
        import subprocess
        import webbrowser
        from pathlib import Path

        root = None
        for r in _project_roots():
            if (r / "studio" / "server.py").is_file():
                root = r
                break
        if root is None:
            self.report({"ERROR"}, "studio/server.py not found")
            return {"CANCELLED"}
        exe = _orch_python_exe() or "python"
        flag = _chat_dir() / "studio.pid"
        already = False
        if flag.is_file():
            try:
                pid = int(flag.read_text(encoding="utf-8").strip() or "0")
                if pid:
                    os.kill(pid, 0)
                    already = True
            except Exception:
                already = False
        if not already:
            creation = 0
            if os.name == "nt":
                creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                creation |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            try:
                proc = subprocess.Popen(
                    [exe, str(root / "studio" / "server.py")],
                    cwd=str(root),
                    creationflags=creation,
                )
                flag.write_text(str(proc.pid), encoding="utf-8")
            except Exception as e:
                self.report({"ERROR"}, f"Studio launch failed: {e}")
                return {"CANCELLED"}
        try:
            webbrowser.open("http://127.0.0.1:8765/")
        except Exception:
            pass
        if not _running:
            try:
                bpy.ops.face.stream_receiver()
            except Exception:
                pass
        self.report({"INFO"}, "Studio window: http://127.0.0.1:8765/")
        return {"FINISHED"}


class SESSION_OT_chat_send(bpy.types.Operator):
    bl_idname = "session.chat_send"
    bl_label = "Send"
    bl_description = "Send this prompt (chat or JSON) to the orchestrator"

    def execute(self, context):
        sc = context.scene
        text = (getattr(sc, "session_chat_input", "") or "").strip()
        if not text:
            self.report({"WARNING"}, "Type a message first")
            return {"CANCELLED"}
        enum_m = str(getattr(sc, "session_pipeline_mode", "FULL") or "FULL").upper()
        mode = "scene" if enum_m == "SCENE" else "full"
        _chat_push_inbox(text, mode=mode)
        sc.session_chat_input = ""
        try:
            sc.session_chat_index = max(0, len(sc.session_chat_log) - 1)
        except Exception:
            pass
        kind = "JSON" if _text_looks_like_json(text) else "chat"
        tag = "scene" if mode == "scene" else kind
        self.report({"INFO"}, f"Sent [{tag}]: {text[:50]}")
        return {"FINISHED"}


class SESSION_OT_chat_clear(bpy.types.Operator):
    bl_idname = "session.chat_clear"
    bl_label = "Clear history"
    bl_description = "Clear the sidebar chat/JSON history (does not stop the orchestrator)"

    def execute(self, context):
        try:
            p = _chat_history_path()
            if p.is_file():
                p.write_text("", encoding="utf-8")
        except Exception:
            pass
        items = getattr(context.scene, "session_chat_log", None)
        if items is not None:
            items.clear()
        _chat_history_append("sys", "History cleared", kind="sys")
        self.report({"INFO"}, "Chat history cleared")
        return {"FINISHED"}


class SESSION_OT_show_sequencer(bpy.types.Operator):
    bl_idname = "session.show_sequencer"
    bl_label = "Show motion/speech tracks"
    bl_description = "Open the Video Sequencer (blue M# motion + orange S# speech). Timeline editor is Action only."

    def execute(self, context):
        n = 0
        try:
            n = _rebuild_session_vse_tracks()
        except Exception as e:
            self.report({"WARNING"}, f"Rebuild tracks failed: {e}")
        # Prefer an existing sequencer; else convert Timeline / Dopesheet
        target = None
        for area in context.screen.areas:
            if area.type == "SEQUENCE_EDITOR":
                target = area
                break
        if target is None:
            for area in context.screen.areas:
                if area.type in {"DOPESHEET_EDITOR", "NLA_EDITOR", "GRAPH_EDITOR"}:
                    target = area
                    break
        if target is None:
            for area in context.screen.areas:
                if area.type == "VIEW_3D" and area != context.area:
                    target = area
                    break
        if target is None:
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    target = area
                    break
        if target is None:
            self.report({"ERROR"}, "No editor area to convert")
            return {"CANCELLED"}
        if target.type != "SEQUENCE_EDITOR":
            target.type = "SEQUENCE_EDITOR"
        try:
            for sp in target.spaces:
                if sp.type == "SEQUENCE_EDITOR":
                    sp.view_type = "SEQUENCER"
                    # 0 hides ch1–3 (motion/speech/audio). Show through ch4.
                    try:
                        sp.display_channel = 4
                    except Exception:
                        pass
        except Exception:
            pass
        self.report(
            {"INFO"},
            f"Video Sequencer: {n} motion strips — ch3 blue=motion, "
            f"ch2 orange=speech, ch1=audio.",
        )
        return {"FINISHED"}


class SESSION_OT_play_timeline(bpy.types.Operator):
    bl_idname = "session.play_timeline"
    bl_label = "Play session"
    bl_description = "Jump to frame 1 and play the full session timeline + camera"

    def execute(self, context):
        arm = _body_armature()
        try:
            _BODY_PLAY["idle_hold"] = False
            _BODY_PLAY["live"] = False
            _BODY_PLAY["scrub_ready"] = True
            _BODY_PLAY["scrub_user_active"] = True
            # Release live cam freeze so SessionCam_* Actions can evaluate on replay
            _CAMERA_PLAY["live"] = False
            _CAMERA_PLAY["hold_after"] = False
            _CAMERA_PLAY["hold_world_loc"] = None
            _CAMERA_PLAY["hold_world_look"] = None
            _assign_session_action_for_scrub(arm, allow_during_idle=True)
            _bind_camera_for_timeline_review()
        except Exception as e:
            self.report({"WARNING"}, f"Bind failed: {e}")
        sc = context.scene
        sc.frame_current = 1
        try:
            bpy.ops.screen.animation_cancel(restore_frame=False)
            bpy.ops.screen.animation_play()
        except Exception:
            pass
        self.report({"INFO"}, "Playing session from frame 1")
        return {"FINISHED"}


class SESSION_OT_reset_session(bpy.types.Operator):
    bl_idname = "session.reset_session"
    bl_label = "New scene"
    bl_description = "Clear timeline, stand at origin in custom rest, tell orchestrator to reset"

    def execute(self, context):
        _chat_push_inbox("reset", mode="full", persist_mode=False)
        arm = _body_armature()
        try:
            _reset_session_nla(arm)
        except Exception as e:
            print(f"[session_ui] reset local: {e}")
        self.report({"INFO"}, "Reset sent")
        return {"FINISHED"}


class SESSION_OT_inpaint(bpy.types.Operator):
    bl_idname = "session.inpaint"
    bl_label = "Inpaint range"
    bl_description = "Regenerate this range. Mid-speech ranges snap to the full spoken line."

    def execute(self, context):
        sc = context.scene
        a = int(getattr(sc, "session_inpaint_start", 1) or 1)
        b = int(getattr(sc, "session_inpaint_end", 40) or 40)
        prompt = (getattr(sc, "session_inpaint_prompt", "") or "").strip()
        if not prompt:
            self.report({"WARNING"}, "Enter an inpaint prompt")
            return {"CANCELLED"}
        if b <= a:
            self.report({"WARNING"}, "End frame must be after start")
            return {"CANCELLED"}
        a2, b2, note = _snap_range_speech_safe(a, b)
        if note:
            sc.session_inpaint_start = a2
            sc.session_inpaint_end = b2
            self.report({"WARNING"}, note)
        _chat_push_inbox(f"inpaint {a2}-{b2} {prompt}", mode="full", persist_mode=False)
        self.report({"INFO"}, f"Inpaint {a2}-{b2} queued")
        return {"FINISHED"}


class SESSION_OT_delete_range(bpy.types.Operator):
    bl_idname = "session.delete_range"
    bl_label = "Delete range"
    bl_description = "Remove takes overlapping From–To. Speech-safe: whole spoken line, never a mid-line cut."

    def execute(self, context):
        sc = context.scene
        a = int(getattr(sc, "session_inpaint_start", 1) or 1)
        b = int(getattr(sc, "session_inpaint_end", 40) or 40)
        if b <= a:
            self.report({"WARNING"}, "End frame must be after start")
            return {"CANCELLED"}
        a2, b2, note = _snap_range_speech_safe(a, b)
        if note:
            sc.session_inpaint_start = a2
            sc.session_inpaint_end = b2
        _chat_push_inbox(f"delete {a2}-{b2}", mode="full", persist_mode=False)
        removed, n_keys = _delete_session_clips_overlapping(a2, b2)
        if not removed:
            self.report({"WARNING"}, "No takes overlap that range")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Deleted {len(removed)} take(s) {a2}-{b2} (keys={n_keys})"
            + (f" — {note}" if note else ""),
        )
        return {"FINISHED"}


class SESSION_OT_delete_last(bpy.types.Operator):
    bl_idname = "session.delete_last"
    bl_label = "Delete last clip"
    bl_description = "Remove the last take (motion + speech together). Never splits a spoken line."

    def execute(self, context):
        clips = _SESSION_BODY.get("clips") or []
        if not clips:
            self.report({"WARNING"}, "No clips")
            return {"CANCELLED"}
        last = clips.pop()
        keep = int(last.get("frame_start") or 2) - 1
        n = _clear_session_keys_after(max(1, keep))
        _session_cursor_save(max(1, keep + 1))
        idx = int(last.get("index") or 0)
        try:
            _vse_remove_named(f"M#{idx}_")
            _vse_remove_named(f"S#{idx}_")
        except Exception:
            pass
        _refresh_markers_from_clips()
        sc = context.scene
        sc.frame_end = max(1, keep)
        if not clips:
            _apply_origin_custom_rest(_body_armature())
            sc.frame_start = 1
            sc.frame_end = 250
            sc.frame_current = 1
            self.report({"INFO"}, "Deleted last clip — avatar reset to origin rest")
        else:
            self.report({"INFO"}, f"Deleted clip #{last.get('index')} (keys removed={n})")
        return {"FINISHED"}


class SESSION_OT_trim_clip(bpy.types.Operator):
    bl_idname = "session.trim_clip"
    bl_label = "Trim last clip to From–To"
    bl_description = "Trim last clip. If range hits speech, it snaps to the full spoken line."

    def execute(self, context):
        clips = _SESSION_BODY.get("clips") or []
        if not clips:
            self.report({"WARNING"}, "No clips")
            return {"CANCELLED"}
        last = clips[-1]
        f0 = int(last.get("frame_start") or 1)
        f1 = int(last.get("frame_end") or 1)
        sc = context.scene
        a = int(getattr(sc, "session_inpaint_start", f0) or f0)
        b = int(getattr(sc, "session_inpaint_end", f1) or f1)
        a, b = max(f0, min(a, b)), min(f1, max(a, b))
        a, b, note = _snap_range_speech_safe(a, b)
        a, b = max(f0, a), min(f1, b)
        if note:
            self.report({"WARNING"}, note)
        if b <= a:
            self.report({"ERROR"}, "Nothing left to keep")
            return {"CANCELLED"}
        # Truncate keys after new end; start trim only allowed if we keep from f0
        # (shrinking the start would leave a hole — snap start to clip start)
        if a > f0:
            self.report(
                {"WARNING"},
                "Trim start kept at clip start (use Delete last or Split at speech edge)",
            )
            a = f0
        n = _clear_session_keys_after(b)
        last["frame_end"] = b
        last["span"] = b - int(last.get("frame_start") or b) + 1
        last["label"] = f"#{last.get('index')} trimmed [{last.get('frame_start')}-{b}]"
        s1 = int(last.get("speech_frame_end") or 0)
        if s1 > b:
            last["speech_frame_end"] = b
        _session_cursor_save(b + 1)
        _mark_movie_dirty("trim")
        context.scene.frame_end = max(int(context.scene.frame_end or 1), b)
        _refresh_markers_from_clips()
        self.report({"INFO"}, f"Trimmed clip #{last.get('index')} → {last.get('frame_start')}-{b} (cut {n} keys)")
        return {"FINISHED"}


class SESSION_OT_split_safe(bpy.types.Operator):
    bl_idname = "session.split_safe"
    bl_label = "Split at playhead (speech-safe)"
    bl_description = "Split the clip at the current frame. Refuses to split inside speech."

    def execute(self, context):
        cur = int(context.scene.frame_current)
        hit = _speech_block_at(cur)
        if hit:
            _c, s0, s1 = hit
            self.report(
                {"ERROR"},
                f"Playhead is inside speech S#{_c.get('index')} [{s0}-{s1}]. "
                f"Move to {s0} or {s1} (speech edge), then split.",
            )
            return {"CANCELLED"}
        c = _clip_at_frame(cur)
        if not c:
            self.report({"WARNING"}, "Playhead is not on a clip")
            return {"CANCELLED"}
        f0, f1 = int(c.get("frame_start") or 1), int(c.get("frame_end") or 1)
        if cur <= f0 or cur >= f1:
            self.report({"WARNING"}, "Playhead is at a clip edge — nothing to split")
            return {"CANCELLED"}
        # Metadata split only (keys stay; two labeled ranges)
        idx = int(c.get("index") or 0)
        left_end = cur - 1
        c["frame_end"] = left_end
        c["span"] = left_end - f0 + 1
        c["label"] = f"#{idx}a [{f0}-{left_end}]"
        s0 = int(c.get("speech_frame_start") or 0)
        s1 = int(c.get("speech_frame_end") or 0)
        if s1 >= s0 > 0 and s1 > left_end:
            # speech entirely after split → move to right clip
            right_speech = (s0, s1)
            c["speech_frame_start"] = 0
            c["speech_frame_end"] = 0
        else:
            right_speech = (0, 0)
        right = dict(c)
        right["index"] = idx
        right["frame_start"] = cur
        right["frame_end"] = f1
        right["span"] = f1 - cur + 1
        right["label"] = f"#{idx}b [{cur}-{f1}]"
        right["speech_frame_start"] = right_speech[0]
        right["speech_frame_end"] = right_speech[1]
        # insert after c
        clips = _SESSION_BODY.get("clips") or []
        i = clips.index(c)
        clips.insert(i + 1, right)
        _refresh_markers_from_clips()
        self.report({"INFO"}, f"Split #{idx} at {cur} (speech kept intact)")
        return {"FINISHED"}


class SESSION_OT_goto_speech(bpy.types.Operator):
    bl_idname = "session.goto_speech"
    bl_label = "Jump to speech start"
    bl_description = "Move playhead to the spoken-line start of the clip under the playhead"

    def execute(self, context):
        cur = int(context.scene.frame_current)
        c = _clip_at_frame(cur) or (
            (_SESSION_BODY.get("clips") or [None])[-1]
        )
        if not c:
            self.report({"WARNING"}, "No clip")
            return {"CANCELLED"}
        s0 = int(c.get("speech_frame_start") or 0)
        context.scene.frame_current = s0 if s0 > 0 else int(c.get("frame_start") or 1)
        return {"FINISHED"}


def _export_session_preview_mp4(context) -> str:
    """Playblast the live session (body + MovieCam) and mix clip WAVs."""
    import subprocess
    from pathlib import Path
    from datetime import datetime

    sc = context.scene
    arm = _body_armature()
    try:
        _BODY_PLAY["idle_hold"] = False
        _BODY_PLAY["live"] = False
        _BODY_PLAY["scrub_ready"] = True
        _assign_session_action_for_scrub(arm, allow_during_idle=True)
        _bind_camera_for_timeline_review()
    except Exception as e:
        print(f"[session_export] bind skip: {e}")

    clips = (_SESSION_BODY.get("clips") or []) if isinstance(_SESSION_BODY, dict) else []
    f1 = int(sc.frame_end or 1)
    if clips:
        try:
            f1 = max(f1, max(int(c.get("frame_end") or 1) for c in clips))
        except Exception:
            pass
    sc.frame_start = 1
    sc.frame_end = max(2, f1)
    sc.frame_current = 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _chat_dir().parent / "movies" / f"session_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / "session_cam.mp4"
    final = out_dir / "session_preview.mp4"

    sc.render.filepath = str(video)
    sc.render.fps = 20
    sc.render.fps_base = 1.0
    sc.render.image_settings.file_format = "FFMPEG"
    ff = sc.render.ffmpeg
    ff.format = "MPEG4"
    try:
        ff.codec = "H264"
    except Exception:
        pass
    sc.render.resolution_percentage = min(100, int(sc.render.resolution_percentage or 100))
    if sc.render.resolution_percentage > 80:
        sc.render.resolution_percentage = 70
    try:
        bpy.ops.render.opengl(animation=True, view_context=False)
    except TypeError:
        bpy.ops.render.opengl(animation=True)

    if not video.is_file():
        raise RuntimeError("OpenGL playblast did not write " + str(video))

    pic_s = max(0.1, float(sc.frame_end) / 20.0)
    try:
        from tools.blender_movie_render import mix_session_audio
        mixed = mix_session_audio(clips, video, final, master_fps=20.0, picture_seconds=pic_s)
        return str(mixed)
    except Exception as e:
        print(f"[session_export] mix helper: {e}")
        return str(video)


def _qc_blender_session(context) -> dict:
    """
    Blender gates before Export movie: identity, floor contact, FACE heading, frustum.
    Returns {ok, errors, warnings, where}.
    """
    errors = []
    warnings = []
    sc = context.scene
    arm = _body_armature()
    mesh = bpy.data.objects.get("BodyMesh")
    if arm is None:
        errors.append("QC-ID: missing SMPL-X_Armature — reload character")
    clips = (_SESSION_BODY.get("clips") or []) if isinstance(_SESSION_BODY, dict) else []
    f_end = int(sc.frame_end or 1)
    samples = [1]
    if f_end > 4:
        for i in range(1, 5):
            samples.append(max(1, int(round(1 + (f_end - 1) * i / 4.0))))

    def _eval_mesh_zmin():
        if mesh is None:
            return None
        try:
            deps = bpy.context.evaluated_depsgraph_get()
            ev = mesh.evaluated_get(deps)
            tmp = ev.to_mesh()
            try:
                mat = ev.matrix_world
                zs = [(mat @ v.co).z for v in tmp.vertices]
                return min(zs) if zs else None
            finally:
                ev.to_mesh_clear()
        except Exception:
            return None

    zmins = []
    if arm is not None and mesh is not None:
        prev = int(sc.frame_current or 1)
        try:
            for fr in samples:
                sc.frame_set(int(fr))
                try:
                    bpy.context.view_layer.update()
                except Exception:
                    pass
                z = _eval_mesh_zmin()
                if z is not None:
                    zmins.append((fr, z))
        finally:
            try:
                sc.frame_set(prev)
            except Exception:
                pass
        if zmins:
            worst = min(zmins, key=lambda t: t[1])
            if worst[1] < -0.002:
                errors.append(
                    f"QC-FLR: BodyMesh zmin={worst[1]:.4f}m at frame {worst[0]} "
                    "(skin through floor — re-generate body)"
                )
        else:
            warnings.append("QC-FLR: could not sample BodyMesh zmin")

        # QC-HDG: FACE vs -Y on first sample
        try:
            sc.frame_set(int(samples[0]))
            bpy.context.view_layer.update()
            lh = arm.pose.bones.get("left_hip") or arm.pose.bones.get("hip_l")
            rh = arm.pose.bones.get("right_hip") or arm.pose.bones.get("hip_r")
            if lh is not None and rh is not None:
                mw = arm.matrix_world
                lp = mw @ lh.head
                rp = mw @ rh.head
                across = lp - rp
                dx, dy = float(across.y), float(-across.x)
                import math as _m
                ang = _m.degrees(_m.atan2(dx, -dy)) if (dx * dx + dy * dy) > 1e-8 else 0.0
                if abs(abs(ang) - 180.0) <= 30.0:
                    errors.append(
                        f"QC-HDG: FACE yaw {ang:.0f}° vs camera -Y (180±30 — re-generate body)"
                    )
        except Exception as e:
            warnings.append(f"QC-HDG skip: {e}")

    # QC-CAM: locomotion clips — BodyMesh AABB in camera frustum
    cam = getattr(sc, "camera", None)
    loco = any(
        str((c or {}).get("camera_shot") or "").upper() in ("WS", "MLS")
        for c in clips
    )
    if loco and cam is not None and mesh is not None:
        try:
            sc.frame_set(int(samples[min(2, len(samples) - 1)]))
            bpy.context.view_layer.update()
            deps = bpy.context.evaluated_depsgraph_get()
            ev = mesh.evaluated_get(deps)
            tmp = ev.to_mesh()
            try:
                mat = ev.matrix_world
                xs = [(mat @ v.co) for v in tmp.vertices]
            finally:
                ev.to_mesh_clear()
            if xs:
                from mathutils import Vector as _V
                mn = _V((min(p.x for p in xs), min(p.y for p in xs), min(p.z for p in xs)))
                mx = _V((max(p.x for p in xs), max(p.y for p in xs), max(p.z for p in xs)))
                corners = [
                    _V((x, y, z))
                    for x in (mn.x, mx.x)
                    for y in (mn.y, mx.y)
                    for z in (mn.z, mx.z)
                ]
                inv = cam.matrix_world.inverted()
                inside = 0
                for p in corners:
                    cam_p = inv @ p
                    if cam_p.z >= 0:
                        continue
                    # Blender camera looks down -Z
                    if -cam_p.z > 1e-4:
                        inside += 1
                if inside < 4:
                    errors.append(
                        "QC-CAM: BodyMesh mostly behind/outside MovieCam "
                        "(change coverage / re-plan camera)"
                    )
        except Exception as e:
            warnings.append(f"QC-CAM skip: {e}")

    return {"ok": not errors, "errors": errors, "warnings": warnings, "where": "blender"}


def _export_session_movie_mp4(context) -> str:
    """EEVEE movie plate @ 20 fps, ffmpeg 24, mix speech on the 20 fps clock."""
    from pathlib import Path
    from datetime import datetime

    sc = context.scene
    arm = _body_armature()
    try:
        _BODY_PLAY["idle_hold"] = False
        _BODY_PLAY["live"] = False
        _BODY_PLAY["scrub_ready"] = True
        _assign_session_action_for_scrub(arm, allow_during_idle=True)
        _bind_camera_for_timeline_review()
    except Exception as e:
        print(f"[movie_export] bind skip: {e}")

    if bpy.data.objects.get("Look_Ground") is None:
        try:
            _apply_look_packet({"op": "apply", "look": {}})
        except Exception as e:
            print(f"[movie_export] default look skip: {e}")

    clips = (_SESSION_BODY.get("clips") or []) if isinstance(_SESSION_BODY, dict) else []
    try:
        import sys as _sys
        root = _project_root_for_assets()
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        from face_agents.qc_gates import run_cpu_gates
        cpu = run_cpu_gates(clips=clips)
        bqc = _qc_blender_session(context)
        errs = list(cpu.get("errors") or []) + list(bqc.get("errors") or [])
        warns = list(cpu.get("warnings") or []) + list(bqc.get("warnings") or [])
        for w in warns:
            print(f"[movie_export] WARN {w}")
        if errs:
            msg = "Export movie blocked:\n" + "\n".join(errs[:12])
            print(f"[movie_export] QC FAIL {msg}")
            raise RuntimeError(msg)
    except RuntimeError:
        raise
    except Exception as e:
        print(f"[movie_export] QC skip: {e}")
    f1 = int(sc.frame_end or 1)
    if clips:
        try:
            f1 = max(f1, max(int(c.get("frame_end") or 1) for c in clips))
        except Exception:
            pass
    sc.frame_start = 1
    sc.frame_end = max(2, f1)
    sc.frame_current = 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _chat_dir().parent / "movies" / f"session_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plate20 = out_dir / "session_plate_20fps.mp4"
    plate24 = out_dir / "session_plate.mp4"
    final = out_dir / "session_movie.mp4"

    import sys
    root = _project_root_for_assets()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools.blender_movie_render import (
        configure_movie_render,
        enable_camera_dof,
        mix_session_audio,
        conform_fps,
        MASTER_FPS,
    )

    grade = "neutral"
    for c in reversed(clips or []):
        lk = (c or {}).get("look") or {}
        if isinstance(lk, dict) and lk.get("grade"):
            grade = str(lk.get("grade") or "neutral")
            break
    sc.render.filepath = str(plate20)
    eng = configure_movie_render(sc, evaluate_fps=MASTER_FPS, grade=grade)
    cam = sc.camera
    shot = ""
    if clips:
        shot = str((clips[-1] or {}).get("camera_shot") or "")
    try:
        enable_camera_dof(cam, shot or "MS")
    except Exception as e:
        print(f"[movie_export] dof skip: {e}")

    print(f"[movie_export] EEVEE animation {sc.frame_start}-{sc.frame_end} engine={eng}")
    bpy.ops.render.render(animation=True)
    if not plate20.is_file():
        raise RuntimeError("EEVEE did not write " + str(plate20))

    conform_fps(plate20, plate24, fps=24)
    pic_s = max(0.1, float(sc.frame_end) / float(MASTER_FPS))
    mixed = mix_session_audio(
        clips, plate24 if plate24.is_file() else plate20, final,
        master_fps=MASTER_FPS, picture_seconds=pic_s,
    )
    print(f"[movie_export] wrote {mixed}")
    _SESSION_BODY["needs_movie_rerender"] = False
    return str(mixed)


class SESSION_OT_export_preview(bpy.types.Operator):
    bl_idname = "session.export_preview"
    bl_label = "Export preview MP4"
    bl_description = "Playblast this session (motion + MovieCam) and mix speech WAVs into an MP4"

    def execute(self, context):
        try:
            path = _export_session_preview_mp4(context)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            print(f"[session_export] {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {path}")
        print(f"[session_export] wrote {path}")
        return {"FINISHED"}


class SESSION_OT_export_movie(bpy.types.Operator):
    bl_idname = "session.export_movie"
    bl_label = "Export movie MP4"
    bl_description = "EEVEE 1080p @ 20 fps master, ffmpeg 24, mix speech. Not OpenGL."

    @classmethod
    def poll(cls, context):
        try:
            if bpy.app.is_job_running("RENDER"):
                return False
        except Exception:
            pass
        return True

    def execute(self, context):
        try:
            path = _export_session_movie_mp4(context)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            print(f"[movie_export] {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Movie {path}")
        print(f"[movie_export] wrote {path}")
        return {"FINISHED"}


class SESSION_ChatItem(bpy.types.PropertyGroup):
    role: bpy.props.StringProperty(name="Role", default="")
    kind: bpy.props.StringProperty(name="Kind", default="chat")
    text: bpy.props.StringProperty(name="Text", default="")
    payload: bpy.props.StringProperty(name="Payload", default="")


def _wrap_chat_line(text: str, width: int = 40, max_lines: int = 3):
    raw = " ".join(str(text or "").split())
    if not raw:
        return [""]
    lines = []
    while raw and len(lines) < max_lines:
        if len(raw) <= width:
            lines.append(raw)
            break
        cut = raw.rfind(" ", 0, width)
        if cut < 10:
            cut = width
        lines.append(raw[:cut])
        raw = raw[cut:].lstrip()
    if raw and len(lines) == max_lines:
        lines[-1] = lines[-1][: max(1, width - 1)] + "…"
    return lines


class SESSION_UL_chat(bpy.types.UIList):
    bl_idname = "SESSION_UL_chat"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        role = str(getattr(item, "role", "") or "").lower()
        kind = str(getattr(item, "kind", "") or "").lower()
        text = str(getattr(item, "text", "") or "")
        is_user = role == "user"
        is_json = kind == "json" or role == "json" or _text_looks_like_json(text)
        if is_user:
            split = layout.split(factor=0.20)
            split.separator()
            col = split.column(align=True)
            tag = col.row()
            tag.alignment = "RIGHT"
            tag.label(text="You · JSON" if is_json else "You", icon="USER")
            for ln in _wrap_chat_line(text, 34, 3):
                body = col.row()
                body.alignment = "RIGHT"
                body.label(text=ln)
        else:
            split = layout.split(factor=0.82)
            col = split.column(align=True)
            tag = col.row(align=True)
            if is_json:
                tag.label(text="JSON", icon="TEXT")
            elif role == "sys":
                tag.label(text="System", icon="INFO")
            else:
                tag.label(text="Agent · chat", icon="OUTLINER_OB_ARMATURE")
            show = text
            if is_json:
                import json as _json
                try:
                    show = _json.dumps(_json.loads(text), ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    show = text
            for ln in _wrap_chat_line(show, 42, 4):
                col.label(text=ln)

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        flt = str(getattr(context.scene, "session_chat_filter", "ALL") or "ALL")
        flags = []
        for it in items:
            kind = str(getattr(it, "kind", "") or "").lower()
            role = str(getattr(it, "role", "") or "").lower()
            is_json = kind == "json" or role == "json"
            if flt == "JSON":
                ok = is_json
            elif flt == "CHAT":
                ok = (not is_json) and role != "sys"
            else:
                ok = True
            flags.append(self.bitflag_filter_item if ok else 0)
        return flags, []


def _sync_chat_ui_list(sc):
    rows = _chat_read_history(80)
    items = getattr(sc, "session_chat_log", None)
    if items is None:
        return
    last_ui = items[-1].text if len(items) else ""
    last_file = str((rows[-1] or {}).get("text") or "") if rows else ""
    if len(items) == len(rows) and last_ui == last_file:
        return
    items.clear()
    for rec in rows:
        it = items.add()
        it.role = str(rec.get("role") or "")
        kind = str(rec.get("kind") or "")
        text = str(rec.get("text") or "")
        if not kind:
            if it.role == "json" or _text_looks_like_json(text):
                kind = "json"
            elif it.role == "sys":
                kind = "sys"
            else:
                kind = "chat"
        it.kind = kind
        it.text = text[:800]
        it.payload = str(rec.get("payload") or "")[:2000]
    try:
        sc.session_chat_index = max(0, len(items) - 1)
    except Exception:
        pass


class SESSION_PT_chat(bpy.types.Panel):
    bl_label = "Agent"
    bl_idname = "SESSION_PT_chat"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Avatar"

    def _draw_msg(self, layout, rec):
        role = str((rec or {}).get("role") or "").lower()
        kind = str((rec or {}).get("kind") or "").lower()
        text = str((rec or {}).get("text") or "")
        is_json = kind == "json" or role == "json" or _text_looks_like_json(text)
        is_user = role == "user"
        box = layout.box()
        if is_user:
            tag = box.row()
            tag.alignment = "RIGHT"
            tag.label(text="You · JSON" if is_json else "You")
            for ln in _wrap_chat_line(text, 36, 5):
                row = box.row()
                row.alignment = "RIGHT"
                row.label(text=ln)
            return
        tag = box.row()
        if is_json:
            tag.label(text="JSON")
        elif role == "sys":
            tag.label(text="System")
        else:
            tag.label(text="Agent")
        show = text
        if is_json:
            import json as _json
            try:
                show = _json.dumps(_json.loads(text), ensure_ascii=False, separators=(",", ":"))
            except Exception:
                show = text
        for ln in _wrap_chat_line(show, 40, 6):
            box.label(text=ln)

    def draw(self, context):
        sc = context.scene
        layout = self.layout
        try:
            (_chat_dir() / "ui_active").write_text(str(time.time()), encoding="utf-8")
        except Exception:
            pass

        running, pid = _orch_is_running()
        head = layout.box()
        status = head.row()
        if running:
            status.label(text=f"Running   pid {pid}", icon="CHECKMARK")
        else:
            status.label(text="Stopped", icon="PAUSE")
        btns = head.row(align=True)
        btns.scale_y = 1.25
        st = btns.row(align=True)
        st.enabled = not running
        st.operator("session.start_orchestrator", text="Start", icon="PLAY")
        sp = btns.row(align=True)
        sp.enabled = running
        sp.operator("session.stop_orchestrator", text="Stop", icon="CANCEL")
        head.operator("session.open_studio", text="Open Studio window", icon="WINDOW")
        if not _running:
            head.operator("face.stream_receiver", text="Start face stream", icon="UV_SYNC_SELECT")

        flt = str(getattr(sc, "session_chat_filter", "ALL") or "ALL")
        try:
            rows = _chat_read_history(48)
        except Exception:
            rows = []
        shown = []
        for rec in rows:
            role = str((rec or {}).get("role") or "").lower()
            kind = str((rec or {}).get("kind") or "").lower()
            is_json = kind == "json" or role == "json"
            if flt == "JSON" and not is_json:
                continue
            if flt == "CHAT" and (is_json or role == "sys"):
                continue
            shown.append(rec)

        hist = layout.box()
        title = hist.row()
        title.label(text=f"Chat  ({len(shown)})")
        title.operator("session.chat_clear", text="", icon="X")
        if hasattr(sc, "session_chat_filter"):
            hist.prop(sc, "session_chat_filter", text="")
        if not shown:
            hist.label(text="No messages yet.")
            hist.label(text="Type a prompt below and press Send.")
        else:
            for rec in shown[-20:]:
                self._draw_msg(hist, rec)
        hist.label(text="You = right    Agent / JSON = left")

        compose = layout.box()
        compose.label(text="Prompt  (chat or JSON)")
        if hasattr(sc, "session_pipeline_mode"):
            mode_row = compose.row(align=True)
            mode_row.prop(sc, "session_pipeline_mode", expand=True)
            if str(getattr(sc, "session_pipeline_mode", "FULL")) == "SCENE":
                compose.label(text="Scene mode: updates Look_Set only", icon="WORLD")
            else:
                compose.label(text="Full mode: scene + motion + speech", icon="ARMATURE_DATA")
        if hasattr(sc, "session_chat_input"):
            compose.prop(sc, "session_chat_input", text="")
        send = compose.row()
        send.scale_y = 1.2
        send.operator("session.chat_send", text="Send", icon="EXPORT")


class SESSION_PT_timeline(bpy.types.Panel):
    bl_label = "Avatar Timeline"
    bl_idname = "SESSION_PT_timeline"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Avatar"

    def draw(self, context):
        sc = context.scene
        layout = self.layout
        col = layout.column(align=True)
        col.operator("session.play_timeline", icon="PLAY")
        col.operator("session.show_sequencer", icon="SEQ_SEQUENCER")
        col.operator("session.goto_speech", icon="SOUND")
        col.operator("session.reset_session", icon="FILE_REFRESH")

        clips = (_SESSION_BODY.get("clips") or []) if isinstance(_SESSION_BODY, dict) else []
        box = layout.box()
        box.label(text=f"Takes: {len(clips)}   (motion #  /  speech S#)", icon="TIME")
        for c in clips[-8:]:
            lab = c.get("label") or "?"
            s0 = int(c.get("speech_frame_start") or 0)
            s1 = int(c.get("speech_frame_end") or 0)
            sd = float(c.get("speech_duration_s") or 0.0)
            box.label(text=str(lab)[:42])
            if s1 >= s0 > 0:
                box.label(text=f"   speech f{s0}-{s1}  {sd:.2f}s")
            else:
                box.label(text="   speech: (none / body only)")

        box = layout.box()
        box.label(text="Edit (speech-safe)", icon="MOD_MASK")
        box.label(text="BLUE M# = motion (trim/split OK here)")
        box.label(text="ORANGE S# = speech (whole line only)")
        box.label(text="Open Video Sequencer to see both ranges.")
        row = box.row(align=True)
        row.prop(sc, "session_inpaint_start", text="From")
        row.prop(sc, "session_inpaint_end", text="To")
        box.prop(sc, "session_inpaint_prompt", text="Inpaint")
        col = box.column(align=True)
        col.operator("session.inpaint", icon="SHADERFX")
        col.operator("session.trim_clip", icon="FULLSCREEN_EXIT")
        col.operator("session.split_safe", icon="SPLIT_HORIZONTAL")
        col.operator("session.delete_range", icon="X")
        col.operator("session.delete_last", icon="TRASH")

        layout.separator()
        layout.operator("session.export_preview", icon="RENDER_ANIMATION")
        layout.operator("session.export_movie", icon="RENDER_STILL")
        layout.label(text="MP4 speech: movie-package mix, or VSE S# strips")


# ---------------------------------------------------------------------------
# REGISTRATION
# ---------------------------------------------------------------------------
_classes = (
    FACE_OT_stream_receiver,
    FACE_OT_stream_stop,
    SESSION_ChatItem,
    SESSION_UL_chat,
    SESSION_OT_start_orchestrator,
    SESSION_OT_stop_orchestrator,
    SESSION_OT_open_studio,
    SESSION_OT_chat_send,
    SESSION_OT_chat_clear,
    SESSION_OT_show_sequencer,
    SESSION_OT_play_timeline,
    SESSION_OT_reset_session,
    SESSION_OT_inpaint,
    SESSION_OT_delete_range,
    SESSION_OT_delete_last,
    SESSION_OT_trim_clip,
    SESSION_OT_split_safe,
    SESSION_OT_goto_speech,
    SESSION_OT_export_preview,
    SESSION_OT_export_movie,
    SESSION_PT_chat,
    SESSION_PT_timeline,
)

def register():
    for cls in _classes:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
        bpy.utils.register_class(cls)
    sc = bpy.types.Scene
    # Recreate chat RNA so new fields (kind/payload) and the filter enum exist.
    for attr in (
        "session_chat_log",
        "session_chat_index",
        "session_chat_filter",
        "session_chat_input",
        "session_pipeline_mode",
    ):
        if hasattr(sc, attr):
            try:
                delattr(sc, attr)
            except Exception:
                pass
    sc.session_chat_input = bpy.props.StringProperty(
        name="Prompt",
        default="",
        description="Chat or JSON sent to the orchestrator",
    )
    sc.session_chat_filter = bpy.props.EnumProperty(
        name="Show",
        items=(
            ("ALL", "All", "Chat, JSON, and system"),
            ("CHAT", "Chat", "User + agent chat only"),
            ("JSON", "JSON", "JSON payloads only"),
        ),
        default="ALL",
    )
    _mode_default = "SCENE" if _read_pipeline_mode() == "scene" else "FULL"
    sc.session_pipeline_mode = bpy.props.EnumProperty(
        name="Pipeline",
        items=(
            ("FULL", "Full", "Scene + motion (T2M) + speech"),
            ("SCENE", "Scene", "Only update Look_Set / background (no T2M/speech)"),
        ),
        default=_mode_default,
        description="Sticky chat mode: Scene only builds the set; Full runs the whole pipeline",
        update=_on_pipeline_mode_update,
    )
    sc.session_chat_log = bpy.props.CollectionProperty(type=SESSION_ChatItem)
    sc.session_chat_index = bpy.props.IntProperty(name="Chat index", default=0)
    # Keep sticky file aligned with the panel default on reload.
    try:
        _write_pipeline_mode("scene" if _mode_default == "SCENE" else "full")
    except Exception:
        pass
    if not hasattr(sc, "session_inpaint_start"):
        sc.session_inpaint_start = bpy.props.IntProperty(name="From", default=1, min=1)
    if not hasattr(sc, "session_inpaint_end"):
        sc.session_inpaint_end = bpy.props.IntProperty(name="To", default=40, min=2)
    if not hasattr(sc, "session_inpaint_prompt"):
        sc.session_inpaint_prompt = bpy.props.StringProperty(
            name="Inpaint prompt", default="a person waves with the right hand"
        )
    try:
        flag = _chat_dir() / "ui_active"
        flag.write_text(str(time.time()), encoding="utf-8")
    except Exception:
        pass

def unregister():
    global _running
    _running = False
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

if __name__ == "__main__":
    register()
    _print_shape_keys()
    print("[face_receiver] Ready.")
    print("Start: bpy.ops.face.stream_receiver()")
    print("Stop:  bpy.ops.face.stream_stop()")
    print("Sidebar: 3D View → N → Avatar  (chat + play / reset / inpaint / export)")
    print("Supports: live MediaPipe + orchestrator emotion/viseme packets")
