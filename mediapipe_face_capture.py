"""
mediapipe_face_capture.py

Captures from webcam OR a video file, runs MediaPipe's Face Landmarker
(with blendshapes enabled), and streams the 52 ARKit-compatible blendshape
scores to Blender over UDP as JSON, once per frame.

Run this with your normal system Python (NOT Blender's bundled Python).

Setup:
    pip install mediapipe opencv-python

    Download the model file once into models/:
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
    Path: models/face_landmarker.task

Usage:
    python mediapipe_face_capture.py

    Press "q" in the preview window to quit.

Input source:
    Set INPUT_SOURCE below:
      - Integer for webcam (e.g. 0, 1)
      - String path for video file (e.g. "testing.mp4")

Workflow tip:
    1. In Blender: run the receiver script, then execute:
         bpy.ops.face.stream_receiver()
       (it will wait up to 15 seconds for this script to start sending)
    2. Then run this file.
    3. When you quit here (press q), the Blender receiver will auto-stop after a few seconds of silence.
"""

import json
import os
import socket
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join("models", "face_landmarker.task")
UDP_IP = "127.0.0.1"
UDP_PORT = 9001

# INPUT_SOURCE:
#   int  -> webcam index (0 = default camera)
#   str  -> path to video file, e.g. "testing.mp4"
INPUT_SOURCE = 0  # default camera

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
SHOW_PREVIEW = True

# Small delay (in seconds) after opening the source and before streaming.
# Gives you time to switch to Blender and start the receiver operator.
STARTUP_DELAY = 2
# ---------------------------------------------------------------------------


def build_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def main():
    # Normalize INPUT_SOURCE: if it's a string that looks like an integer, convert it
    input_source = int(INPUT_SOURCE) if isinstance(INPUT_SOURCE, str) and INPUT_SOURCE.isdigit() else INPUT_SOURCE

    landmarker = build_landmarker()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    cap = cv2.VideoCapture(input_source)

    # Resolution settings only apply to cameras
    if isinstance(input_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        src_desc = f"camera index {input_source}" if isinstance(input_source, int) else f"video file '{input_source}'"
        raise RuntimeError(f"Could not open {src_desc}.")

    if STARTUP_DELAY > 0:
        print(f"Waiting {STARTUP_DELAY}s for you to start the Blender receiver (bpy.ops.face.stream_receiver())...")
        time.sleep(STARTUP_DELAY)

    start_time = time.time()
    frame_count = 0
    last_fps_print = start_time

    src = f"video '{input_source}'" if isinstance(input_source, str) else f"camera {input_source}"
    print(f"Streaming blendshapes from {src} to {UDP_IP}:{UDP_PORT}. Press 'q' to quit.")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                if isinstance(input_source, str):
                    print("End of video file reached.")
                else:
                    print("Frame grab failed, stopping.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_blendshapes:
                categories = result.face_blendshapes[0]
                # "_neutral" is an extra category MediaPipe includes alongside
                # the 52 ARKit ones -- it's not a real ARKit shape key, so drop it.
                blendshapes = {
                    c.category_name: round(float(c.score), 4)
                    for c in categories
                    if c.category_name != "_neutral"
                }
                payload = json.dumps(
                    {"t": timestamp_ms, "blendshapes": blendshapes}
                ).encode("utf-8")
                try:
                    sock.sendto(payload, (UDP_IP, UDP_PORT))
                except OSError as e:
                    print(f"UDP send failed: {e}")

            if SHOW_PREVIEW:
                cv2.imshow("MediaPipe Face Capture (press 'q' to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            now = time.time()
            if now - last_fps_print >= 2.0:
                fps = frame_count / (now - start_time)
                print(f"  ~{fps:.1f} fps")
                last_fps_print = now

    finally:
        cap.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        sock.close()
        landmarker.close()


if __name__ == "__main__":
    main()
