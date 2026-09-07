"""Everything drawn on top of the preview.

None of this touches the saved photo -- the overlay goes onto the preview
copy only, while the photo is filtered separately from the full-resolution
frame. That separation is the whole reason the preview can carry as much
on-screen information as it likes.
"""

import cv2
import numpy as np

from gestures.hand_tracker import FINGERTIPS, HAND_CONNECTIONS, to_pixels

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR, because that is what OpenCV works in.
ACCENT = (64, 196, 255)
ACCENT_DIM = (40, 120, 158)
TEXT = (242, 242, 242)
TEXT_DIM = (156, 156, 160)
PANEL = (18, 18, 22)
READY = (120, 230, 140)
BUSY = (90, 160, 250)

# Gestures that do something, and the label to show for each.
ACTION_LABELS = {
    "PEACE": "CAPTURE",
    "OPEN_PALM": "RECORD VIDEO",
    "THUMBS_UP": "FAVORITE",
    "THUMBS_DOWN": "DELETE LAST",
    "SWIPE_LEFT": "PREVIOUS FILTER",
    "SWIPE_RIGHT": "NEXT FILTER",
    "POINT_LEFT": "PREVIOUS FILTER",
    "POINT_RIGHT": "NEXT FILTER",
}


# -- primitives ------------------------------------------------------------
def _panel(frame, x1, y1, x2, y2, color=PANEL, alpha=0.58):
    """Blend a translucent rectangle into the frame, clipped to its bounds."""
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    tint = np.full(region.shape, color, dtype=np.uint8)
    cv2.addWeighted(tint, alpha, region, 1.0 - alpha, 0, dst=region)


def _text(frame, message, origin, scale=0.5, color=TEXT, thickness=1, center=False):
    """Draw anti-aliased text, optionally centred on the origin's x."""
    x, y = origin
    if center:
        (text_width, _), _ = cv2.getTextSize(message, FONT, scale, thickness)
        x -= text_width // 2
    cv2.putText(frame, message, (int(x), int(y)), FONT, scale, color, thickness, cv2.LINE_AA)


def _text_width(message, scale=0.5, thickness=1):
    (width, _), _ = cv2.getTextSize(message, FONT, scale, thickness)
    return width


# -- landmarks -------------------------------------------------------------
def draw_landmarks(frame, hands):
    """Draw the hand skeleton for every detected hand."""
    height, width = frame.shape[:2]
    for landmarks in hands:
        points = to_pixels(landmarks, width, height)
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, tuple(points[start]), tuple(points[end]), ACCENT_DIM, 2, cv2.LINE_AA)
        for index, point in enumerate(points):
            radius, color = (5, ACCENT) if index in FINGERTIPS else (3, TEXT_DIM)
            cv2.circle(frame, tuple(point), radius, color, -1, cv2.LINE_AA)


# -- panels ----------------------------------------------------------------
def _draw_top_bar(frame, state):
    width = frame.shape[1]
    _panel(frame, 0, 0, width, 44)
    _text(frame, "GESTURE CAMERA", (16, 28), scale=0.58, color=TEXT, thickness=1)

    right = width - 16
    if state.show_fps:
        label = f"FPS {state.fps:4.1f}"
        right -= _text_width(label, 0.5)
        _text(frame, label, (right, 28), scale=0.5, color=TEXT_DIM)
        right -= 22

    photos = f"PHOTOS {state.photo_count}   VIDEOS {state.recorded_count}"
    right -= _text_width(photos, 0.5)
    _text(frame, photos, (right, 28), scale=0.5, color=TEXT_DIM)

    if state.is_recording:
        label = "REC"
        right -= _text_width(label, 0.5) + 22
        _text(frame, label, (right, 28), scale=0.5, color=(60, 60, 255), thickness=2)

    if state.pending_writes:
        label = "SAVING"
        right -= _text_width(label, 0.5) + 22
        _text(frame, label, (right, 28), scale=0.5, color=BUSY)


def _draw_gesture_badge(frame, state):
    """Current pose, and whether the engine is ready to act on it."""
    height = frame.shape[0]
    top = height - 132

    if not state.hand_present:
        _text(frame, "SHOW YOUR HAND", (16, top), scale=0.52, color=TEXT_DIM)
        return

    gesture = state.gesture if state.gesture != "NONE" else "-"
    _text(frame, f"GESTURE  {gesture}", (16, top), scale=0.52, color=TEXT)

    action = ACTION_LABELS.get(state.gesture)
    if state.in_cooldown:
        _text(frame, "COOLDOWN", (16, top + 22), scale=0.46, color=BUSY)
    elif action:
        _text(frame, f"HOLD TO {action}", (16, top + 22), scale=0.46, color=READY)


def _draw_hold_ring(frame, state):
    """Progress ring at the palm, so the hold time reads as feedback."""
    if state.hold_progress <= 0.0 or state.hand_center is None:
        return

    height, width = frame.shape[:2]
    center = (int(state.hand_center[0] * width), int(state.hand_center[1] * height))
    radius = 34

    cv2.circle(frame, center, radius, ACCENT_DIM, 2, cv2.LINE_AA)
    cv2.ellipse(
        frame, center, (radius, radius), -90, 0, 360 * state.hold_progress, ACCENT, 4, cv2.LINE_AA
    )


def _draw_filter_strip(frame, state):
    """Previous / current / next names, plus a dot per filter."""
    height, width = frame.shape[:2]
    _panel(frame, 0, height - 88, width, height)

    centre_x = width // 2
    count = len(state.filter_names)
    previous = state.filter_names[(state.filter_index - 1) % count]
    following = state.filter_names[(state.filter_index + 1) % count]

    _text(frame, state.filter_name.upper(), (centre_x, height - 50), scale=0.86,
          color=ACCENT, thickness=2, center=True)

    current_half = _text_width(state.filter_name.upper(), 0.86, 2) // 2
    _text(frame, f"< {previous}", (centre_x - current_half - 24, height - 54),
          scale=0.44, color=TEXT_DIM)
    _text(frame, f"{following} >", (centre_x + current_half + 24, height - 54),
          scale=0.44, color=TEXT_DIM)

    # Position dots.
    spacing = 18
    start_x = centre_x - (count - 1) * spacing // 2
    for index in range(count):
        selected = index == state.filter_index
        cv2.circle(
            frame,
            (start_x + index * spacing, height - 28),
            4 if selected else 3,
            ACCENT if selected else (70, 70, 78),
            -1,
            cv2.LINE_AA,
        )

    _text(frame, "PEACE = CAPTURE   SWIPE = CHANGE FILTER   H = HELP",
          (centre_x, height - 8), scale=0.4, color=TEXT_DIM, center=True)


def _draw_toast(frame, state):
    """Brief confirmation after a photo lands on disk."""
    if state.toast_alpha <= 0.0 or not state.toast_text:
        return

    width = frame.shape[1]
    box_width = max(220, _text_width(state.toast_text, 0.46) + 48)
    x1 = (width - box_width) // 2
    _panel(frame, x1, 56, x1 + box_width, 122, color=(16, 16, 20), alpha=0.62 * state.toast_alpha)

    intensity = state.toast_alpha
    _text(frame, "PHOTO SAVED", (width // 2, 84), scale=0.6,
          color=tuple(int(c * intensity) for c in READY), thickness=2, center=True)
    _text(frame, state.toast_text, (width // 2, 108), scale=0.44,
          color=tuple(int(c * intensity) for c in TEXT_DIM), center=True)


def _draw_flash(frame, state):
    """White wash on the shutter -- the cue that the moment was taken."""
    if state.flash_alpha <= 0.0:
        return
    white = np.full(frame.shape, 255, dtype=np.uint8)
    cv2.addWeighted(white, state.flash_alpha, frame, 1.0 - state.flash_alpha, 0, dst=frame)


HELP_LINES = (
    ("GESTURES", None),
    ("Peace sign, held", "take a photo"),
    ("Swipe right", "next filter"),
    ("Swipe left", "previous filter"),
    ("", None),
    ("KEYS", None),
    ("1 - 8", "pick a filter directly"),
    ("N / P", "next / previous filter"),
    ("Space", "shutter"),
    ("L", "toggle hand landmarks"),
    ("F", "toggle FPS"),
    ("H", "close this help"),
    ("Q or Esc", "quit"),
)


def _draw_help(frame):
    height, width = frame.shape[:2]
    box_width, box_height = 380, 24 * len(HELP_LINES) + 40
    x1 = (width - box_width) // 2
    y1 = (height - box_height) // 2
    _panel(frame, x1, y1, x1 + box_width, y1 + box_height, color=(12, 12, 16), alpha=0.86)

    y = y1 + 34
    for left, right in HELP_LINES:
        if right is None:
            _text(frame, left, (x1 + 24, y), scale=0.5, color=ACCENT, thickness=1)
        else:
            _text(frame, left, (x1 + 24, y), scale=0.45, color=TEXT)
            _text(frame, right, (x1 + 186, y), scale=0.45, color=TEXT_DIM)
        y += 24


# -- entry point -----------------------------------------------------------
def draw(frame, state):
    """Composite the whole overlay onto a preview frame, in place."""
    if state.show_landmarks and state.hands is not None:
        draw_landmarks(frame, state.hands)

    _draw_hold_ring(frame, state)
    _draw_top_bar(frame, state)
    _draw_filter_strip(frame, state)
    _draw_gesture_badge(frame, state)

    if state.show_help:
        _draw_help(frame)

    _draw_toast(frame, state)
    # Flash goes last so it washes over everything else, like a real one.
    _draw_flash(frame, state)
    return frame
