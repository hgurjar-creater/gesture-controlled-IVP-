"""Gesture Camera -- webcam, hand gestures, live filters, instant capture.

    python main.py

Peace sign held for a moment takes a photo; swiping left or right changes
the filter. Everything is also on the keyboard -- press H for the list.
"""

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import cv2

from camera.camera import CameraError, CameraManager
from capture.photo_capture import PhotoCapture
from capture.video_capture import VideoCapture
from config import settings
from filters.filter_engine import FilterEngine
from gestures import gesture_detector as gestures
from gestures.gesture_detector import GestureDetector, GestureStateMachine
from gestures.hand_tracker import HandTracker, palm_center
from ui import overlay


@dataclass
class ApplicationState:
    """One place to read the whole app from, which makes it debuggable."""

    # Filter selection.
    filter_index: int = 0
    filter_name: str = "Original"
    filter_names: list = field(default_factory=list)

    # Hand and gesture.
    hands: object = None
    hand_present: bool = False
    hand_center: object = None
    gesture: str = gestures.NONE
    last_action: str = None
    hold_progress: float = 0.0
    in_cooldown: bool = False

    # Capture.
    photo_count: int = 0
    pending_writes: int = 0
    flash_alpha: float = 0.0
    toast_text: str = ""
    toast_alpha: float = 0.0
    is_recording: bool = False
    recorded_count: int = 0

    # Display.
    fps: float = 0.0
    show_landmarks: bool = settings.SHOW_LANDMARKS
    show_fps: bool = settings.SHOW_FPS
    show_help: bool = False
    running: bool = True


class GestureCamera:
    """Wires the pieces together and runs the main loop."""

    def __init__(self, camera_index=None, start_filter=0):
        self.camera = CameraManager(camera_index)
        self.tracker = HandTracker()
        self.detector = GestureDetector()
        self.state_machine = GestureStateMachine()
        self.filters = FilterEngine(start_filter)
        self.photos = PhotoCapture(self.filters)
        self.videos = VideoCapture(self.filters)

        self.state = ApplicationState(filter_names=self.filters.names)
        self._frame_times = deque(maxlen=30)
        self._flash_until = 0.0
        self._toast_until = 0.0

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self):
        self.camera.open()
        self.tracker.open()
        self.photos.start()
        self.videos.start()

        width, height = self.camera.get_resolution()
        print(f"Camera open at {width}x{height}. Press H for controls, Q to quit.")

        cv2.namedWindow(settings.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            settings.WINDOW_NAME,
            settings.PREVIEW_WIDTH,
            round(settings.PREVIEW_WIDTH * height / width),
        )
        return self

    def __exit__(self, *exc_info):
        # Drain queued photos before tearing anything down, so quitting
        # right after a capture still writes the file.
        self.photos.stop()
        self.videos.stop()
        self.tracker.close()
        self.camera.release()
        cv2.destroyAllWindows()
        print(f"Saved {self.photos.saved_count} photo(s) to {self.photos.directory}")

    # -- per-frame steps ---------------------------------------------------
    def _track_fps(self):
        self._frame_times.append(time.monotonic())
        if len(self._frame_times) > 1:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                self.state.fps = (len(self._frame_times) - 1) / span

    def _detect(self, preview):
        """Run hand tracking and gesture classification on the preview."""
        state = self.state
        hands = self.tracker.detect(preview)

        state.hands = hands
        state.hand_present = hands is not None

        landmarks = hands[0] if hands is not None else None
        state.hand_center = palm_center(landmarks) if landmarks is not None else None

        pose, swipe = self.detector.update(landmarks)
        state.gesture = swipe or pose

        action = self.state_machine.update(pose, swipe)
        state.hold_progress = self.state_machine.progress
        state.in_cooldown = self.state_machine.in_cooldown()
        return action

    def _handle_action(self, action, full_frame):
        """Apply a confirmed gesture. Full frame is passed for capture."""
        if action in (gestures.SWIPE_RIGHT, gestures.POINT_RIGHT):
            self.filters.next()
        elif action in (gestures.SWIPE_LEFT, gestures.POINT_LEFT):
            self.filters.previous()
        elif action == gestures.PEACE:
            self._capture(full_frame)
        elif action == gestures.OPEN_PALM:
            self._toggle_recording()
        elif action == gestures.THUMBS_DOWN:
            self._delete_last_photo()
        elif action == gestures.THUMBS_UP:
            self.state.toast_text = "FAVORITED!"
            self._toast_until = time.monotonic() + settings.TOAST_SECONDS

        if action is not None:
            self.state.last_action = action

    def _delete_last_photo(self):
        if self.photos.last_saved_path and self.photos.last_saved_path.exists():
            try:
                self.photos.last_saved_path.unlink()
                self.photos.saved_count = max(0, self.photos.saved_count - 1)
                self.state.toast_text = "DELETED!"
                self._toast_until = time.monotonic() + settings.TOAST_SECONDS
                self.photos.last_saved_path = None
            except Exception:
                pass

    def _toggle_recording(self):
        self.state.is_recording = self.videos.toggle_recording()

    def _capture(self, full_frame):
        """Fire the shutter on the frame that is on screen right now."""
        # The full-resolution frame is already in hand, so the photo is the
        # exact moment the gesture was confirmed -- no re-read, no lag.
        self.photos.capture(full_frame, self.filters.name)
        now = time.monotonic()
        self._flash_until = now + settings.FLASH_SECONDS
        self._toast_until = now + settings.TOAST_SECONDS

    def _update_feedback(self):
        """Advance the flash and toast animations, and read the writer."""
        state = self.state
        now = time.monotonic()

        remaining = self._flash_until - now
        state.flash_alpha = max(0.0, min(1.0, remaining / settings.FLASH_SECONDS))

        remaining = self._toast_until - now
        # Hold the toast at full strength, then fade the last half second.
        state.toast_alpha = max(0.0, min(1.0, remaining / 0.5))

        state.photo_count = self.photos.saved_count
        state.recorded_count = self.videos.recorded_count
        state.pending_writes = self.photos.pending
        if self.photos.last_saved_path is not None:
            state.toast_text = self.photos.last_saved_path.name

        state.filter_index = self.filters.index
        state.filter_name = self.filters.name

    def _handle_keys(self, full_frame):
        """Keyboard control -- a full alternative to every gesture."""
        key = cv2.waitKey(1) & 0xFF
        if key == 255:  # nothing pressed
            return

        state = self.state
        if key in (ord("q"), 27):
            state.running = False
        elif key == ord(" "):
            self._capture(full_frame)
        elif key in (ord("n"), 83):  # 83 = right arrow
            self.filters.next()
        elif key in (ord("p"), 81):  # 81 = left arrow
            self.filters.previous()
        elif ord("1") <= key <= ord("9"):
            self.filters.select(key - ord("1"))
        elif key == ord("0"):
            self.filters.select(9)
        elif key == ord("-"):
            self.filters.select(10)
        elif key == ord("l"):
            state.show_landmarks = not state.show_landmarks
        elif key == ord("f"):
            state.show_fps = not state.show_fps
        elif key == ord("h"):
            state.show_help = not state.show_help

    def _window_closed(self):
        try:
            return cv2.getWindowProperty(settings.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    # -- main loop ---------------------------------------------------------
    def run(self):
        dropped = 0

        while self.state.running:
            full_frame = self.camera.read()
            if full_frame is None:
                dropped += 1
                if dropped > 30:
                    print("Camera stopped delivering frames.", file=sys.stderr)
                    break
                continue
            dropped = 0

            preview = self.camera.to_preview(full_frame)

            action = self._detect(preview)
            self._handle_action(action, full_frame)
            
            if self.state.is_recording:
                self.videos.capture(full_frame, self.filters.name)

            # Filter the preview copy, never the frame the photo comes from.
            display = self.filters.apply(preview)
            if display is preview:
                display = preview.copy()  # Original returns its input

            self._track_fps()
            self._update_feedback()
            overlay.draw(display, self.state)

            cv2.imshow(settings.WINDOW_NAME, display)
            self._handle_keys(full_frame)

            if self._window_closed():
                break


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Gesture-controlled camera with live filters.")
    parser.add_argument(
        "--camera", type=int, default=None,
        help=f"webcam index (default {settings.CAMERA_INDEX})",
    )
    parser.add_argument(
        "--filter", type=int, default=0, metavar="N",
        help="filter to start on, 0-7 (default 0)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        with GestureCamera(args.camera, args.filter) as app:
            app.run()
    except CameraError as error:
        print(f"Camera error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
