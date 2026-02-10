"""
Motion capture module using MediaPipe + OpenCV.

Captures webcam video, runs MediaPipe Pose estimation,
and provides landmark data for the pose solver.
"""

import logging
import threading
import time

import numpy as np
from PIL import Image as PILImage

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger(__name__)


class MotionCapture:
    """
    Webcam-based motion capture using MediaPipe Pose.

    Usage:
        mc = MotionCapture()
        mc.start(camera_index=0)
        frame, landmarks = mc.get_latest()
        mc.stop()
    """

    def __init__(self):
        if cv2 is None:
            raise ImportError("opencv-python is required: pip install opencv-python")
        if mp is None:
            raise ImportError("mediapipe is required: pip install mediapipe")

        self._capture = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Latest frame and landmarks
        self._frame: PILImage.Image | None = None
        self._landmarks = None
        self._world_landmarks = None
        self._fps = 0.0
        self._mirror = True

        # MediaPipe
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._pose = None

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

        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Motion capture started on camera {camera_index}")

    def stop(self):
        """Stop capturing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture:
            self._capture.release()
            self._capture = None
        if self._pose:
            self._pose.close()
            self._pose = None
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

        while self._running:
            t0 = time.perf_counter()

            ret, frame = self._capture.read()
            if not ret:
                time.sleep(0.01)
                continue

            if self._mirror:
                frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)

            landmarks = None
            world_landmarks = None

            if results.pose_landmarks:
                landmarks = list(results.pose_landmarks.landmark)

                # Draw skeleton overlay on the frame
                self._mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self._mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            if results.pose_world_landmarks:
                world_landmarks = list(results.pose_world_landmarks.landmark)

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
            if elapsed < 1.0 / 30:
                time.sleep(1.0 / 30 - elapsed)
