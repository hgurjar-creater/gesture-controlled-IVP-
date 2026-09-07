"""Background segmentation via MediaPipe."""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from config import settings

MODEL_FILENAME = "selfie_segmenter.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"

def ensure_model(directory=None, url=MODEL_URL):
    directory = directory or settings.MODEL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILENAME

    if not path.exists():
        print(f"Downloading selfie segmenter model to {path} ...")
        temporary = path.with_suffix(".partial")
        try:
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"could not download the model from {url} ({error}). "
                f"Download it manually and save it as {path}."
            ) from error
        print("Model ready.")

    return path

# Global segmenter so we don't reload the model on every frame
_global_segmenter = None

def get_segmenter():
    global _global_segmenter
    if _global_segmenter is None:
        options = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        _global_segmenter = vision.ImageSegmenter.create_from_options(options)
    return _global_segmenter

def background_blur(frame):
    """Blurs the background, keeping the person in focus."""
    segmenter = get_segmenter()
    
    # Needs to be a bit smaller to run fast enough for realtime? 
    # ImageSegmenter handles resizing internally, so we just pass the frame.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    result = segmenter.segment(image)
    mask = result.category_mask.numpy_view() == 0
    mask = np.squeeze(mask)
    mask = np.stack((mask,) * 3, axis=-1)
    
    blurred = cv2.GaussianBlur(frame, (51, 51), 0)
    return np.where(mask, frame, blurred)

