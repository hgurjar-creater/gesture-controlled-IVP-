"""Central configuration for the gesture camera.

Every tunable number lives here so behaviour can be adjusted without
touching the modules that use it.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHOTO_DIR = PROJECT_ROOT / "photos"
MODEL_DIR = PROJECT_ROOT / "models"

# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
CAMERA_INDEX = 0

# The camera runs at capture resolution the whole time and every frame is
# downscaled for the preview. That way a full-resolution frame is always
# available and capturing never has to renegotiate the camera mode.
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAMERA_FPS = 30

# Preview / hand-tracking resolution. Width is authoritative; the height is
# derived from the real camera aspect ratio so nothing gets stretched.
PREVIEW_WIDTH = 960

# Mirror the image so the preview behaves like a mirror.
MIRROR_PREVIEW = True

# --------------------------------------------------------------------------
# Hand tracking
# --------------------------------------------------------------------------
MAX_HANDS = 1
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5

# Run MediaPipe on a further-reduced copy of the preview frame. 1 = every
# frame at full preview size.
TRACKING_WIDTH = 480

# --------------------------------------------------------------------------
# Gesture state machine
# --------------------------------------------------------------------------
# A static pose must be held this long before it counts as an action.
GESTURE_HOLD_SECONDS = 0.30

# After an action fires, ignore everything for this long.
GESTURE_COOLDOWN_SECONDS = 1.00
SWIPE_COOLDOWN_SECONDS = 0.60

# A pose has to survive this many consecutive frames before the state
# machine even considers it a candidate. Filters out single-frame noise.
GESTURE_MIN_FRAMES = 2

# Swipe detection: horizontal travel of the palm, in fractions of frame
# width, that must happen inside the time window.
SWIPE_WINDOW_SECONDS = 0.45
SWIPE_MIN_DISTANCE = 0.10
SWIPE_MAX_VERTICAL_DRIFT = 0.25

# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------
JPEG_QUALITY = 95
FLASH_SECONDS = 0.12
TOAST_SECONDS = 1.40

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
WINDOW_NAME = "Gesture Camera"
SHOW_LANDMARKS = True
SHOW_FPS = True
