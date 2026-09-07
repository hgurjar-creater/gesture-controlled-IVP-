"""Recording video: full-resolution filtering and writing to disk.

Unlike photos, video frames need to be written continuously. We use a thread
to encode frames to avoid stalling the main loop.
"""

import queue
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import settings


class VideoCapture:
    def __init__(self, filter_engine, directory=None):
        self.filter_engine = filter_engine
        self.directory = directory or settings.PHOTO_DIR
        self.directory.mkdir(parents=True, exist_ok=True)
        
        self._queue = queue.Queue()
        self._worker = None
        self._lock = threading.Lock()
        
        self.is_recording = False
        self.recorded_count = 0
        self.last_saved_path = None
        self.last_error = None
        self._writer = None

    def start(self):
        self._worker = threading.Thread(target=self._run, name="video-writer", daemon=True)
        self._worker.start()
        return self

    def stop(self, timeout=5.0):
        if self._worker is None:
            return
        self._queue.put(("QUIT", None, None))
        self._worker.join(timeout=timeout)
        self._worker = None

    def toggle_recording(self):
        with self._lock:
            self.is_recording = not self.is_recording
            if self.is_recording:
                path = self._next_path(datetime.now())
                self._queue.put(("START", path, None))
            else:
                self._queue.put(("STOP", None, None))
            return self.is_recording

    def capture(self, frame, filter_name):
        if self.is_recording:
            self._queue.put(("FRAME", frame.copy(), filter_name))

    def _run(self):
        while True:
            cmd, arg1, arg2 = self._queue.get()
            if cmd == "QUIT":
                if self._writer is not None:
                    self._writer.release()
                break
            elif cmd == "START":
                path = arg1
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = settings.CAMERA_FPS
                size = (settings.CAPTURE_WIDTH, settings.CAPTURE_HEIGHT)
                self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)
                if not self._writer.isOpened():
                    self.last_error = f"Failed to open video writer for {path}"
                    self._writer = None
                else:
                    self.last_error = None
                    self.last_saved_path = path
            elif cmd == "STOP":
                if self._writer is not None:
                    self._writer.release()
                    self._writer = None
                    with self._lock:
                        self.recorded_count += 1
            elif cmd == "FRAME":
                if self._writer is not None:
                    frame, filter_name = arg1, arg2
                    try:
                        filtered = self.filter_engine.apply_named(frame, filter_name)
                        self._writer.write(filtered)
                    except Exception as error:
                        self.last_error = str(error)

    def _next_path(self, timestamp):
        stem = f"video_{timestamp:%Y%m%d_%H%M%S}"
        path = self.directory / f"{stem}.mp4"
        counter = 2
        while path.exists():
            path = self.directory / f"{stem}_{counter}.mp4"
            counter += 1
        return path
