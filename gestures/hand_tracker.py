"""Hand detection via MediaPipe, wrapped so the rest of the app sees numpy.

MediaPipe 1.0 dropped the old `mp.solutions.hands` helper, so this uses
the Tasks API. That needs a model file on disk, which is fetched on first
run and cached under `models/`.

Landmarks come back as a plain (21, 3) float array of normalised
coordinates -- x and y in 0..1 across the frame, z relative to the wrist.
Keeping the MediaPipe result types confined to this module means the
gesture and UI code never has to know which backend produced them.
"""

import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from config import settings

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Bone topology, for drawing the skeleton.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm edge
)

FINGERTIPS = (4, 8, 12, 16, 20)


def ensure_model(directory=None, url=MODEL_URL):
    """Return the model path, downloading it on first run if needed."""
    directory = directory or settings.MODEL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILENAME

    if not path.exists():
        print(f"Downloading hand landmark model to {path} ...")
        temporary = path.with_suffix(".partial")
        try:
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"could not download the hand model from {url} ({error}). "
                f"Download it manually and save it as {path}."
            ) from error
        print("Model ready.")

    return path


class HandTracker:
    """Detects one hand per frame and returns its landmarks as an array."""

    def __init__(self, max_hands=None, model_path=None):
        self.max_hands = settings.MAX_HANDS if max_hands is None else max_hands
        self.model_path = model_path or ensure_model()
        self.landmarker = None
        self._start = None
        self._last_timestamp_ms = -1

    # -- lifecycle ---------------------------------------------------------
    def open(self):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            # VIDEO mode carries tracking state between frames, so the hand
            # stays locked on instead of being re-detected from scratch each
            # time. It is synchronous, which keeps the main loop simple.
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=settings.DETECTION_CONFIDENCE,
            min_hand_presence_confidence=settings.DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.TRACKING_CONFIDENCE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._start = time.monotonic()
        return self

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc_info):
        self.close()

    # -- detection ---------------------------------------------------------
    def detect(self, frame):
        """Find the most confident hand in a BGR frame.

        Returns an (n, 21, 3) array of normalised landmarks, one entry per
        hand, or None when no hand is visible. Coordinates are relative to
        the frame, so they stay valid whatever resolution was fed in.
        """
        if self.landmarker is None:
            raise RuntimeError("hand tracker is not open")

        # Detection runs on a reduced copy: at 480px wide the landmarks are
        # just as usable and the model runs appreciably faster. Normalised
        # output means the caller never sees the difference.
        if frame.shape[1] > settings.TRACKING_WIDTH:
            scale = settings.TRACKING_WIDTH / frame.shape[1]
            frame = cv2.resize(
                frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode rejects any timestamp that is not strictly increasing.
        timestamp_ms = max(
            int((time.monotonic() - self._start) * 1000), self._last_timestamp_ms + 1
        )
        self._last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None

        return np.array(
            [[(p.x, p.y, p.z) for p in hand] for hand in result.hand_landmarks],
            dtype=np.float32,
        )


def to_pixels(landmarks, width, height):
    """Convert normalised landmarks to integer pixel coordinates."""
    points = landmarks[:, :2] * np.array([width, height], dtype=np.float32)
    return points.astype(np.int32)


def palm_center(landmarks):
    """Normalised centre of the palm -- steady as the fingers articulate."""
    return landmarks[[0, 5, 9, 13, 17], :2].mean(axis=0)
