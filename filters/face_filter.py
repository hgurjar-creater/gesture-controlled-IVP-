"""Face detection and privacy blur via MediaPipe."""

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from config import settings

MODEL_FILENAME = "blaze_face_short_range.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

def ensure_model(directory=None, url=MODEL_URL):
    directory = directory or settings.MODEL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILENAME

    if not path.exists():
        print(f"Downloading face detector model to {path} ...")
        temporary = path.with_suffix(".partial")
        try:
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"could not download the model from {url} ({error}). "
            ) from error
        print("Model ready.")

    return path

_global_detector = None

def get_detector():
    global _global_detector
    if _global_detector is None:
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.IMAGE,
        )
        _global_detector = vision.FaceDetector.create_from_options(options)
    return _global_detector

def anonymous_filter(frame):
    """Blurs all detected faces for privacy."""
    detector = get_detector()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    result = detector.detect(image)
    output = frame.copy()
    
    for detection in result.detections:
        bbox = detection.bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        
        # Ensure within frame
        x, y = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        
        if x2 > x and y2 > y:
            roi = output[y:y2, x:x2]
            roi = cv2.GaussianBlur(roi, (51, 51), 30)
            output[y:y2, x:x2] = roi
            
    return output
