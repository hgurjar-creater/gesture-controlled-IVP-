"""Taking the photo: full-resolution filtering, naming and writing to disk.

The shutter has to feel instant, so `capture()` only copies the frame and
hands it to a worker thread. Filtering a 1920x1080 frame and encoding a
quality-95 JPEG costs on the order of 100ms; doing that on the main loop
would show up as a visible stutter every single time you take a photo.
"""

import queue
import threading
from datetime import datetime

import cv2

from config import settings


class PhotoCapture:
    """Queues full-resolution frames and writes them out off the main loop."""

    def __init__(self, filter_engine, directory=None):
        self.filter_engine = filter_engine
        self.directory = directory or settings.PHOTO_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

        self._queue = queue.Queue()
        self._worker = None
        self._lock = threading.Lock()

        self.saved_count = 0
        self.last_saved_path = None
        self.last_error = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self._worker = threading.Thread(target=self._run, name="photo-writer", daemon=True)
        self._worker.start()
        return self

    def stop(self, timeout=5.0):
        """Let any queued photos finish writing, then shut the worker down."""
        if self._worker is None:
            return
        self._queue.put(None)
        self._worker.join(timeout=timeout)
        self._worker = None

    @property
    def pending(self):
        return self._queue.qsize()

    # -- capture -----------------------------------------------------------
    def capture(self, frame, filter_name):
        """Queue a frame for filtering and saving. Returns immediately.

        The frame is copied because the camera loop will overwrite the
        buffer it came from long before the worker gets to it, and the
        filter name is captured now so a filter change made while the
        write is in flight does not retroactively alter the photo.
        """
        self._queue.put((frame.copy(), filter_name, datetime.now()))

    # -- worker ------------------------------------------------------------
    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            frame, filter_name, timestamp = item
            try:
                self._write(frame, filter_name, timestamp)
            except Exception as error:  # a bad write must not kill the thread
                self.last_error = str(error)

    def _write(self, frame, filter_name, timestamp):
        filtered = self.filter_engine.apply_named(frame, filter_name)
        path = self._next_path(timestamp)

        ok = cv2.imwrite(
            str(path), filtered, [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY]
        )
        if not ok:
            raise IOError(f"could not write {path}")

        with self._lock:
            self.saved_count += 1
            self.last_saved_path = path

    def _next_path(self, timestamp):
        """photos/photo_20260818_204755.jpg, suffixed if that name is taken.

        Two captures inside the same second would otherwise collide -- the
        cooldown makes that unlikely, but the keyboard shutter has no
        cooldown at all.
        """
        stem = f"photo_{timestamp:%Y%m%d_%H%M%S}"
        path = self.directory / f"{stem}.jpg"
        counter = 2
        while path.exists():
            path = self.directory / f"{stem}_{counter}.jpg"
            counter += 1
        return path
