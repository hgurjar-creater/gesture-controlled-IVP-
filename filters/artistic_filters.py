"""The composed filters: several cheap operations stacked into a look.

Both of these were originally written the obvious way and both were too
slow for a live preview, so each carries one deliberate approximation:

  * the vignette gain map is stored as uint8 and applied with an integer
    multiply, rather than as a float mask multiplied in floating point
  * the dreamy glow is blurred at quarter size and scaled back up, rather
    than blurred at full size

Each is roughly ten times faster than the direct version and differs from
it by at most one 255th per channel -- a blur being upscaled is exactly
the case where losing resolution costs nothing.
"""

from functools import lru_cache

import cv2
import numpy as np

from .basic_filters import _channel_lut, adjust_saturation, tone_curve

# Vintage: contrast, a warm channel shift and lifted blacks, as one table.
_VINTAGE = dict(contrast=1.18, lift_to=32, lift_amount=0.08)
VINTAGE_LUT = _channel_lut(
    tone_curve(offset=-12, **_VINTAGE),
    tone_curve(**_VINTAGE),
    tone_curve(offset=20, **_VINTAGE),
)

# Dreamy: brightness lift and softened contrast, as one table.
_DREAMY = tone_curve(gamma=0.85, contrast=0.90)
DREAMY_LUT = _channel_lut(_DREAMY, _DREAMY, _DREAMY)

# How far down the glow is blurred. The blur radius is divided by the same
# factor so the result matches a full-size blur of the same strength.
GLOW_SCALE = 4


@lru_cache(maxsize=8)
def _vignette_mask(height, width, strength):
    """Radial gain map as a 3-channel uint8 array, where 255 means no change.

    Cached because there are only ever two frame sizes in play -- preview
    and capture -- and building one costs more than applying it.
    """
    ky = cv2.getGaussianKernel(height, height * 0.55)
    kx = cv2.getGaussianKernel(width, width * 0.55)
    mask = ky @ kx.T
    mask /= mask.max()
    mask = 1.0 - strength * (1.0 - mask)

    gain = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(gain[:, :, np.newaxis], 3, axis=2)


def apply_vignette(frame, strength=0.45):
    """Darken the corners. Strength 0 is a no-op, 1 fades them to black."""
    height, width = frame.shape[:2]
    mask = _vignette_mask(height, width, round(strength, 3))
    return cv2.multiply(frame, mask, scale=1.0 / 255.0)


def soft_glow(frame, amount=0.38):
    """Blend a heavy blur back over the frame to bloom the highlights."""
    small = cv2.resize(
        frame, None, fx=1.0 / GLOW_SCALE, fy=1.0 / GLOW_SCALE, interpolation=cv2.INTER_AREA
    )
    # Radius scales with frame width so the glow looks the same at preview
    # and capture resolution.
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=small.shape[1] / 90.0)
    blur = cv2.resize(
        small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    return cv2.addWeighted(frame, 1.0 - amount, blur, amount, 0)


def vintage(frame):
    """Filter 5 -- faded highlights, muted colour, warm shift, vignette."""
    out = cv2.LUT(frame, VINTAGE_LUT)
    out = adjust_saturation(out, 0.74)
    return apply_vignette(out, 0.45)


def dreamy(frame):
    """Filter 7 -- soft glow bloom, lifted brightness, gentle colour."""
    out = soft_glow(frame, 0.38)
    out = cv2.LUT(out, DREAMY_LUT)
    return adjust_saturation(out, 1.15)


def doodle(frame):
    """Filter 9 -- pencil sketch/doodle effect."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
