"""Verify the install and the processing pipeline without needing a webcam.

    python selftest.py

Checks that every filter runs at both preview and capture resolution and
reports how long each takes, that the gesture classifier and its state
machine behave, and that a photo can actually be written. Run this first
when something is not working -- it isolates everything except the camera.
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from config import settings
from filters.filter_engine import FilterEngine
from gestures import gesture_detector as g
from gestures.gesture_detector import GestureDetector, GestureStateMachine

PASS, FAIL = "  ok  ", " FAIL "
failures = []


def check(condition, label):
    print(f"[{PASS if condition else FAIL}] {label}")
    if not condition:
        failures.append(label)
    return condition


# -- synthetic hands -------------------------------------------------------
FINGER_X = {"index": 0.44, "middle": 0.50, "ring": 0.56, "pinky": 0.62}
FINGER_BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def make_hand(thumb=False, index=False, middle=False, ring=False, pinky=False):
    """Build a plausible 21-landmark hand with the given fingers extended."""
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = (0.50, 0.90, 0.0)  # wrist

    # Thumb, which folds sideways across the palm rather than curling in.
    points[1] = (0.44, 0.82, 0.0)
    points[2] = (0.42, 0.72, 0.0)
    if thumb:
        points[3] = (0.35, 0.66, 0.0)
        points[4] = (0.30, 0.60, 0.0)
    else:
        points[3] = (0.46, 0.68, 0.0)
        points[4] = (0.50, 0.66, 0.0)

    for name, extended in (
        ("index", index), ("middle", middle), ("ring", ring), ("pinky", pinky)
    ):
        base = FINGER_BASE[name]
        x = FINGER_X[name]
        points[base] = (x, 0.65, 0.0)
        if extended:
            points[base + 1] = (x, 0.50, 0.0)
            points[base + 2] = (x, 0.42, 0.0)
            points[base + 3] = (x, 0.35, 0.0)
        else:
            points[base + 1] = (x, 0.60, 0.0)
            points[base + 2] = (x, 0.66, 0.0)
            points[base + 3] = (x, 0.68, 0.0)
    return points


def shift(hand, dx):
    moved = hand.copy()
    moved[:, 0] += dx
    return moved


# -- the checks ------------------------------------------------------------
def test_filters():
    print("\nFilters")
    engine = FilterEngine()
    check(len(engine) == 11, f"11 filters registered (got {len(engine)})")

    preview_size = (settings.PREVIEW_WIDTH * 9 // 16, settings.PREVIEW_WIDTH)
    capture_size = (settings.CAPTURE_HEIGHT, settings.CAPTURE_WIDTH)

    rng = np.random.default_rng(7)
    for label, (height, width) in (
        (f"preview {preview_size[1]}x{preview_size[0]}", preview_size),
        (f"capture {capture_size[1]}x{capture_size[0]}", capture_size),
    ):
        frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        print(f"\n  {label}")
        budget_ms = 1000.0 / 30.0
        for index, name in enumerate(engine.names):
            engine.select(index)
            engine.apply(frame)  # warm caches so the timing is steady

            start = time.perf_counter()
            out = engine.apply(frame)
            elapsed = (time.perf_counter() - start) * 1000

            valid = out.shape == frame.shape and out.dtype == np.uint8
            realtime = "" if elapsed < budget_ms else "   <- over 33ms budget"
            print(f"    [{PASS if valid else FAIL}] {name:<14} {elapsed:6.1f} ms{realtime}")
            if not valid:
                failures.append(f"{name} at {label}: shape/dtype wrong")


def test_gestures():
    print("\nGesture classification")
    detector = GestureDetector()

    cases = (
        ("OPEN_PALM", make_hand(True, True, True, True, True), g.OPEN_PALM),
        ("FIST", make_hand(), g.FIST),
        ("PEACE", make_hand(index=True, middle=True), g.PEACE),
        ("THUMBS_UP", make_hand(thumb=True), g.THUMBS_UP),
        ("POINTING", make_hand(index=True), g.POINTING),
    )
    for label, hand, expected in cases:
        actual = detector.classify_pose(hand)
        check(actual == expected, f"{label} classified as {actual}")

    check(detector.classify_pose(None) == g.NONE, "no hand -> NONE")


def test_swipe():
    print("\nSwipe detection")
    for direction, sign, expected in (("right", +1, g.SWIPE_RIGHT), ("left", -1, g.SWIPE_LEFT)):
        detector = GestureDetector()
        now = 100.0
        result = None
        for step in range(8):
            hand = shift(make_hand(index=True), sign * 0.045 * step)
            _, swipe = detector.update(hand, now=now + step * 0.04)
            result = result or swipe
        check(result == expected, f"swipe {direction} -> {result}")

    # A hand that drifts back and forth without a net sweep must not fire.
    detector = GestureDetector()
    now, fired = 200.0, None
    for step in range(12):
        hand = shift(make_hand(index=True), 0.05 * np.sin(step))
        _, swipe = detector.update(hand, now=now + step * 0.04)
        fired = fired or swipe
    check(fired is None, f"jitter does not fire a swipe (got {fired})")


def test_state_machine():
    print("\nGesture state machine")
    machine = GestureStateMachine()
    peace = g.PEACE
    now = 500.0

    fires = []
    # Hold the peace sign for a full second at 30fps.
    for step in range(30):
        action = machine.update(peace, None, now=now + step / 30.0)
        if action:
            fires.append(round(step / 30.0, 3))

    check(len(fires) == 1, f"one second of PEACE fires once (fired at {fires})")
    check(
        fires and fires[0] >= settings.GESTURE_HOLD_SECONDS - 0.05,
        f"fired after the {settings.GESTURE_HOLD_SECONDS}s hold, not instantly",
    )

    # Keep holding through the cooldown; it must fire again after it lapses.
    for step in range(30, 60):
        action = machine.update(peace, None, now=now + step / 30.0)
        if action:
            fires.append(round(step / 30.0, 3))
    print(f"DEBUG fires = {fires}")
    gap = fires[1] - fires[0] if len(fires) > 1 else 0
    check(
        len(fires) == 2 and gap >= settings.GESTURE_COOLDOWN_SECONDS - 0.001,
        f"second fire waited out the cooldown (gap {gap:.2f}s)",
    )

    # A brief flicker must not count as a held pose.
    machine = GestureStateMachine()
    flicker = [machine.update(peace if step % 2 else g.FIST, None, now=900.0 + step / 30.0)
               for step in range(30)]
    check(not any(flicker), "alternating poses never fire")


def test_capture():
    print("\nPhoto capture")
    from capture.photo_capture import PhotoCapture

    engine = FilterEngine()
    with tempfile.TemporaryDirectory() as directory:
        photos = PhotoCapture(engine, Path(directory)).start()
        frame = np.full((settings.CAPTURE_HEIGHT, settings.CAPTURE_WIDTH, 3), 90, np.uint8)

        start = time.perf_counter()
        photos.capture(frame, "Vintage")
        photos.capture(frame, "Dreamy")
        enqueue_ms = (time.perf_counter() - start) * 1000
        check(enqueue_ms < 60, f"shutter returns in {enqueue_ms:.1f} ms without blocking")

        photos.stop()
        check(photos.saved_count == 2, f"both photos written ({photos.saved_count})")
        check(photos.last_error is None, f"no write errors ({photos.last_error})")

        written = sorted(Path(directory).glob("*.jpg"))
        if written:
            image = cv2.imread(str(written[-1]))
            check(
                image is not None and image.shape[:2] == frame.shape[:2],
                f"saved at full resolution ({image.shape[1]}x{image.shape[0]})",
            )
            print(f"         filename pattern: {written[-1].name}")


def test_model():
    print("\nHand tracking model")
    try:
        from gestures.hand_tracker import HandTracker, ensure_model

        path = ensure_model()
        check(path.exists(), f"model present ({path.stat().st_size / 1e6:.1f} MB)")

        tracker = HandTracker().open()
        blank = np.zeros((360, 480, 3), np.uint8)
        start = time.perf_counter()
        result = tracker.detect(blank)
        elapsed = (time.perf_counter() - start) * 1000
        check(result is None, "blank frame yields no hand")
        print(f"         detection pass: {elapsed:.1f} ms")
        tracker.close()
    except Exception as error:
        check(False, f"hand tracker failed: {type(error).__name__}: {error}")


def test_cameras():
    print("\nCameras (index probe)")
    found = []
    for index in range(3):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            ok, frame = capture.read()
            if ok and frame is not None:
                found.append(index)
                print(f"  [{PASS}] index {index}: {frame.shape[1]}x{frame.shape[0]}")
        capture.release()
    if not found:
        print(f"  [ note ] no webcam found. Everything else above still applies;")
        print(f"           run main.py --camera N once a camera is connected.")
    else:
        print(f"         start with: python main.py --camera {found[0]}")


def main():
    print(f"Gesture Camera self-test")
    print(f"python {sys.version.split()[0]} | opencv {cv2.__version__} | numpy {np.__version__}")

    test_filters()
    test_gestures()
    test_swipe()
    test_state_machine()
    test_capture()
    test_model()
    test_cameras()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
