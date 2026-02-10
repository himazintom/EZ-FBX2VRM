"""
Motion capture module using MediaPipe + OpenCV.

Captures webcam video, runs MediaPipe Pose estimation,
and provides landmark data for the pose solver.

Uses the MediaPipe Tasks API (0.10.x+) with PoseLandmarker.
"""

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options as base_options_module
except ImportError:
    mp = None

logger = logging.getLogger(__name__)

# Model file path - look in models/ directory next to the project root
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_MODEL_FILE = _MODEL_DIR / "pose_landmarker_lite.task"

# Landmark connection pairs for skeleton drawing
_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),   # left eye
    (0, 4), (4, 5), (5, 6), (6, 8),   # right eye
    (9, 10),                            # mouth
    (11, 12),                           # shoulders
    (11, 13), (13, 15),                # left arm
    (12, 14), (14, 16),                # right arm
    (11, 23), (12, 24),                # torso sides
    (23, 24),                           # hips
    (23, 25), (25, 27),                # left leg
    (24, 26), (26, 28),                # right leg
    (27, 29), (29, 31),                # left foot
    (28, 30), (30, 32),                # right foot
    (15, 17), (15, 19), (15, 21),      # left hand
    (16, 18), (16, 20), (16, 22),      # right hand
]


class _LandmarkCompat:
    """Adapter to make new API landmarks compatible with pose_solver expectations."""
    __slots__ = ('x', 'y', 'z', 'visibility')

    def __init__(self, x, y, z, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class MotionCapture:
    """
    Webcam-based motion capture using MediaPipe Pose.

    Usage:
        mc = MotionCapture()
        mc.start(camera_index=0)
        frame, landmarks, world_landmarks = mc.get_latest()
        mc.stop()
    """

    def __init__(self):
        if cv2 is None:
            raise ImportError("opencv-python is required: pip install opencv-python")
        if mp is None:
            raise ImportError("mediapipe is required: pip install mediapipe")
        if not _MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Pose landmarker model not found: {_MODEL_FILE}\n"
                "Run: python _download_pose_model.py"
            )

        self._capture = None
        self._running = False
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

        # Latest frame and landmarks
        self._frame: PILImage.Image | None = None
        self._landmarks = None
        self._world_landmarks = None
        self._fps = 0.0
        self._mirror = True

        self._landmarker = None
        self._frame_count = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def mirror(self) -> bool:
        return self._mirror

    @mirror.setter
    def mirror(self, value: bool):
        self._mirror = value

    def list_cameras(self, max_check: int = 5) -> list[int]:
        """List available camera indices."""
        available = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def start(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """Start capturing from the specified camera."""
        if self._running:
            self.stop()

        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Create PoseLandmarker with VIDEO mode
        options = vision.PoseLandmarkerOptions(
            base_options=base_options_module.BaseOptions(
                model_asset_path=str(_MODEL_FILE),
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._frame_count = 0

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Motion capture started on camera {camera_index}")

    def stop(self):
        """Stop capturing."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        # Only release resources after thread has fully stopped
        if self._capture:
            self._capture.release()
            self._capture = None
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
        logger.info("Motion capture stopped")

    def get_latest(self) -> tuple[PILImage.Image | None, list | None, list | None]:
        """
        Get the latest frame and landmarks.

        Returns:
            (frame_image, pose_landmarks, world_landmarks)
            - frame_image: PIL Image of the annotated webcam frame
            - pose_landmarks: normalized landmarks (0-1 range, screen coords)
            - world_landmarks: real-world 3D landmarks (meters)
        """
        with self._lock:
            return self._frame, self._landmarks, self._world_landmarks

    def _capture_loop(self):
        """Main capture loop running in a background thread."""
        frame_times = []

        while self._running and not self._stop_event.is_set():
            t0 = time.perf_counter()

            capture = self._capture
            landmarker = self._landmarker
            if capture is None or landmarker is None:
                break

            ret, frame = capture.read()
            if not ret:
                if self._stop_event.wait(timeout=0.01):
                    break
                continue

            if self._mirror:
                frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image and detect
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._frame_count += 1
            timestamp_ms = int(self._frame_count * (1000 / 30))

            try:
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                logger.debug(f"Pose detection error: {e}")
                # Still show camera frame even if detection fails
                pil_image = PILImage.fromarray(rgb)
                with self._lock:
                    self._frame = pil_image
                    self._landmarks = None
                    self._world_landmarks = None
                continue

            landmarks = None
            world_landmarks = None

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                # Convert to compatible format
                raw = result.pose_landmarks[0]
                landmarks = [
                    _LandmarkCompat(lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0)
                    for lm in raw
                ]

                # Draw skeleton overlay
                self._draw_skeleton(frame, raw)

            if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
                raw_w = result.pose_world_landmarks[0]
                world_landmarks = [
                    _LandmarkCompat(lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0)
                    for lm in raw_w
                ]

            # Convert annotated frame to PIL
            annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(annotated_rgb)

            with self._lock:
                self._frame = pil_image
                self._landmarks = landmarks
                self._world_landmarks = world_landmarks

            # FPS calculation
            t1 = time.perf_counter()
            frame_times.append(t1 - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg = sum(frame_times) / len(frame_times)
            self._fps = 1.0 / avg if avg > 0 else 0

            # Cap at ~30fps to save CPU
            elapsed = t1 - t0
            remaining = 1.0 / 30 - elapsed
            if remaining > 0:
                self._stop_event.wait(timeout=remaining)

    def _draw_skeleton(self, frame, landmarks):
        """Draw pose skeleton on the BGR frame."""
        h, w = frame.shape[:2]
        for start, end in _POSE_CONNECTIONS:
            if start >= len(landmarks) or end >= len(landmarks):
                continue
            lm1 = landmarks[start]
            lm2 = landmarks[end]
            x1, y1 = int(lm1.x * w), int(lm1.y * h)
            x2, y2 = int(lm2.x * w), int(lm2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
