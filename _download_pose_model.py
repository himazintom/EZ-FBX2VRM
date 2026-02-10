"""Download MediaPipe pose landmarker model."""
import urllib.request
import os

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pose_landmarker_lite.task")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if os.path.exists(MODEL_PATH):
    print(f"Model already exists: {MODEL_PATH}")
else:
    print(f"Downloading pose landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Downloaded to: {MODEL_PATH} ({size_mb:.1f} MB)")
