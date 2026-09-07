"""The cheap filters: colour matrices, lookup tables, brightness curves.

Every filter has the same signature -- it takes a BGR uint8 frame and
returns a BGR uint8 frame of the same shape. Nothing mutates its input,
so the caller can keep the clean frame around.

The recurring trick here is folding point operations together. Contrast,
gamma, per-channel offsets and black lift are all "new value depends only
on old value", so any stack of them collapses into one 256-entry table
built once at import. Applying that table is a single memory-bound pass,
which is what keeps these usable at 1080p.
"""

import cv2
import numpy as np


def _channel_lut(blue, green, red):
    """Pack three 256-entry curves into a LUT that cv2.LUT can apply to BGR."""
    return np.dstack((blue, green, red)).astype(np.uint8)


def _curve(points_in, points_out):
    """Interpolate a 256-entry curve through the given control points."""
    return np.interp(np.arange(256), points_in, points_out).clip(0, 255)


def tone_curve(gamma=1.0, contrast=1.0, offset=0.0, lift_to=None, lift_amount=0.0):
    """Build one 256-entry curve from a stack of point operations.

    Applied in the order gamma, contrast (around mid-grey), black lift,
    offset. Defaults are all no-ops, so pass only what you need.
    """
    values = np.arange(256, dtype=np.float64)

    if gamma != 1.0:
        values = ((values / 255.0) ** gamma) * 255.0
    if contrast != 1.0:
        values = (values - 128.0) * contrast + 128.0
    if lift_to is not None and lift_amount:
        values = values * (1.0 - lift_amount) + lift_to * lift_amount
    if offset:
        values = values + offset

    return values.clip(0, 255)


# Colour-grading curves ----------------------------------------------------
_SHADOWS = [0, 64, 128, 192, 255]

_LIFT = _curve(_SHADOWS, [0, 82, 150, 212, 255])
_DROP = _curve(_SHADOWS, [0, 50, 106, 172, 255])
_NEUTRAL = np.arange(256, dtype=np.float64)
_SLIGHT_LIFT = _curve(_SHADOWS, [0, 70, 133, 198, 255])

# Warm lifts red and nudges green with it, so the cast lands on yellow
# rather than magenta. Cool lifts blue alone, which reads cleaner.
WARM_LUT = _channel_lut(_DROP, _SLIGHT_LIFT, _LIFT)
COOL_LUT = _channel_lut(_LIFT, _NEUTRAL, _DROP)

# Brightness lift plus a slight contrast reduction, as one table.
_BRIGHT = tone_curve(gamma=0.72, contrast=0.92)
BRIGHT_LUT = _channel_lut(_BRIGHT, _BRIGHT, _BRIGHT)

# Sepia matrix, rows/columns ordered for BGR in and BGR out.
SEPIA_MATRIX = np.array(
    [
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.189, 0.769, 0.393],
    ]
)


# Reusable adjustments ----------------------------------------------------
def adjust_contrast(frame, amount):
    """Scale contrast around mid-grey. 1.0 leaves the frame alone."""
    return cv2.convertScaleAbs(frame, alpha=amount, beta=128.0 * (1.0 - amount))


def adjust_saturation(frame, amount):
    """Push colour away from or toward its own brightness.

    Blending against the greyscale version costs about half of an HSV
    round trip and is the same operation colour graders call vibrance.
    Above 1.0 saturates, below 1.0 mutes, 0.0 is fully grey.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, amount, grey, 1.0 - amount, 0)


def apply_gamma(frame, gamma):
    """Gamma-correct via LUT. Below 1.0 brightens, above 1.0 darkens."""
    return cv2.LUT(frame, tone_curve(gamma=gamma).astype(np.uint8))


# The filters themselves --------------------------------------------------
def original(frame):
    """Filter 0 -- untouched."""
    return frame


def grayscale(frame):
    """Filter 1 -- black and white, returned as 3-channel BGR."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def sepia(frame):
    """Filter 2 -- classic sepia colour matrix."""
    return cv2.transform(frame, SEPIA_MATRIX)


def warm(frame):
    """Filter 3 -- pushes reds and yellows, pulls blues back."""
    return cv2.LUT(frame, WARM_LUT)


def cool(frame):
    """Filter 4 -- pushes blues, pulls reds back."""
    return cv2.LUT(frame, COOL_LUT)


def bright(frame):
    """Filter 6 -- lifts brightness and softens contrast slightly."""
    return cv2.LUT(frame, BRIGHT_LUT)
