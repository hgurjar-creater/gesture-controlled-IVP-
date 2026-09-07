"""Webcam access -- opening the device, pulling frames, shutting down.

This module deliberately knows nothing about filters, gestures or the UI.
Its one job is handing out frames.
"""

import cv2

from config import settings


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or has stopped delivering."""


class CameraManager:
    """Wraps a single webcam and hands out full-resolution frames.

    The device is held open at capture resolution for the whole session and
    every frame is downscaled for the preview. Switching camera modes at
    shutter time would cost hundreds of milliseconds and drop frames, so we
    pay the downscale on every frame instead and get an instant shutter.
    """

    def __init__(self, index=None):
        self.index = settings.CAMERA_INDEX if index is None else index
        self.capture = None
        self.width = 0
        self.height = 0

    # -- lifecycle ---------------------------------------------------------
    def open(self):
        """Open the device and negotiate the requested capture mode."""
        # DirectShow starts up far faster than the Media Foundation default
        # on Windows; fall back to whatever the platform picks elsewhere.
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            capture = cv2.VideoCapture(self.index, backend)
            if capture.isOpened():
                self.capture = capture
                break
            capture.release()
        else:
            raise CameraError(
                f"could not open camera {self.index}. Close any other app "
                f"using the webcam, or set CAMERA_INDEX in config/settings.py."
            )

        # MJPG is what lets most webcams actually deliver 30fps at 1080p;
        # without it the driver falls back to raw YUY2 and throttles to ~5fps.
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAPTURE_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAPTURE_HEIGHT)
        self.capture.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Drivers silently clamp to the nearest supported mode, so read back
        # what we actually got rather than trusting the request.
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.release()
            raise CameraError(f"camera {self.index} opened but delivered no frames")
        self.height, self.width = frame.shape[:2]
        return self

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc_info):
        self.release()

    # -- frames ------------------------------------------------------------
    def read(self):
        """Return the next full-resolution frame, mirrored if configured.

        Mirroring happens here so that everything downstream -- preview,
        landmarks, swipe direction and the saved photo -- agrees on which
        way round the image is.
        """
        if self.capture is None:
            raise CameraError("camera is not open")

        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        if settings.MIRROR_PREVIEW:
            frame = cv2.flip(frame, 1)
        return frame

    def to_preview(self, frame):
        """Downscale a full-resolution frame to preview size.

        INTER_AREA is the right choice for shrinking; it averages the pixels
        being discarded instead of point-sampling them.
        """
        width = settings.PREVIEW_WIDTH
        if frame.shape[1] <= width:
            return frame
        height = round(frame.shape[0] * width / frame.shape[1])
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def get_resolution(self):
        """The real capture resolution, as (width, height)."""
        return self.width, self.height
