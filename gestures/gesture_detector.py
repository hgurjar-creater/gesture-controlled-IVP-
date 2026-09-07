"""Turning 21 hand landmarks into named gestures, then into actions.

Two things live here, and the split matters:

`GestureDetector` is stateless per frame for poses and keeps a short
motion history for swipes. It answers "what is the hand doing right now".

`GestureStateMachine` answers "should something happen". Raw detections
arrive dozens of times a second, so firing directly off them would take
dozens of photos from one peace sign. The state machine requires a pose
to be held, fires once, then goes into cooldown.
"""

import time
from collections import deque

import numpy as np

from config import settings

# Landmark indices, as MediaPipe orders them.
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# Gesture names.
NONE = "NONE"
OPEN_PALM = "OPEN_PALM"
FIST = "FIST"
PEACE = "PEACE"
THUMBS_UP = "THUMBS_UP"
THUMBS_DOWN = "THUMBS_DOWN"
POINTING = "POINTING"
POINT_LEFT = "POINT_LEFT"
POINT_RIGHT = "POINT_RIGHT"
SWIPE_LEFT = "SWIPE_LEFT"
SWIPE_RIGHT = "SWIPE_RIGHT"

# SWIPE_POSES defines which gestures are allowed to be swept across the screen.
SWIPE_POSES = (OPEN_PALM, POINTING, POINT_LEFT, POINT_RIGHT, PEACE)

# Actions the state machine will emit, and how long each must be held.
# Swipes are motion, not a pose -- by the time one is recognised it has
# already happened, so there is nothing left to hold.
ACTION_HOLD_SECONDS = {
    PEACE: settings.GESTURE_HOLD_SECONDS,
    OPEN_PALM: settings.GESTURE_HOLD_SECONDS,
    THUMBS_UP: settings.GESTURE_HOLD_SECONDS,
    THUMBS_DOWN: settings.GESTURE_HOLD_SECONDS,
    POINT_LEFT: settings.GESTURE_HOLD_SECONDS,
    POINT_RIGHT: settings.GESTURE_HOLD_SECONDS,
    SWIPE_LEFT: 0.0,
    SWIPE_RIGHT: 0.0,
}

ACTION_COOLDOWN_SECONDS = {
    PEACE: settings.GESTURE_COOLDOWN_SECONDS,
    OPEN_PALM: settings.GESTURE_COOLDOWN_SECONDS,
    THUMBS_UP: settings.GESTURE_COOLDOWN_SECONDS,
    THUMBS_DOWN: settings.GESTURE_COOLDOWN_SECONDS,
    POINT_LEFT: settings.GESTURE_COOLDOWN_SECONDS,
    POINT_RIGHT: settings.GESTURE_COOLDOWN_SECONDS,
    SWIPE_LEFT: settings.SWIPE_COOLDOWN_SECONDS,
    SWIPE_RIGHT: settings.SWIPE_COOLDOWN_SECONDS,
}


def _distance(landmarks, a, b):
    return float(np.linalg.norm(landmarks[a, :2] - landmarks[b, :2]))


class GestureDetector:
    """Classifies a hand pose and watches for horizontal swipes."""

    def __init__(self):
        # (timestamp, x, y) of the palm centre, newest last.
        self._track = deque(maxlen=60)

    # -- finger state ------------------------------------------------------
    @staticmethod
    def fingers_extended(landmarks):
        """Which of the five fingers are extended, thumb first.

        Curling a finger pulls its tip back toward the wrist, so comparing
        tip-to-wrist against knuckle-to-wrist works at any hand rotation --
        unlike the usual "is the tip higher than the knuckle" test, which
        breaks the moment the hand tilts.
        """
        extended = []

        # The thumb folds sideways across the palm rather than curling in,
        # so it gets measured against the far side of the hand instead.
        extended.append(
            _distance(landmarks, THUMB_TIP, PINKY_MCP)
            > _distance(landmarks, THUMB_MCP, PINKY_MCP)
        )

        for tip, pip in (
            (INDEX_TIP, INDEX_PIP),
            (MIDDLE_TIP, MIDDLE_PIP),
            (RING_TIP, RING_PIP),
            (PINKY_TIP, PINKY_PIP),
        ):
            extended.append(
                _distance(landmarks, tip, WRIST) > _distance(landmarks, pip, WRIST) * 1.05
            )

        return extended

    # -- pose classification ----------------------------------------------
    def classify_pose(self, landmarks):
        """Name the static pose the hand is currently making."""
        if landmarks is None:
            return NONE

        thumb, index, middle, ring, pinky = self.fingers_extended(landmarks)

        if index and middle and not ring and not pinky:
            return PEACE
        if index and not middle and not ring and not pinky:
            dx = landmarks[INDEX_TIP, 0] - landmarks[WRIST, 0]
            dy = landmarks[INDEX_TIP, 1] - landmarks[WRIST, 1]
            if abs(dx) > abs(dy) * 1.5:
                return POINT_RIGHT if dx > 0 else POINT_LEFT
            return POINTING
        if thumb and index and middle and ring and pinky:
            return OPEN_PALM
        if thumb and not index and not middle and not ring and not pinky:
            # Only a thumbs *up* -- pointing the thumb sideways or down is a
            # different intent, and would otherwise read the same.
            if landmarks[THUMB_TIP, 1] < landmarks[WRIST, 1] - 0.10:
                return THUMBS_UP
            if landmarks[THUMB_TIP, 1] > landmarks[WRIST, 1] + 0.10:
                return THUMBS_DOWN
            return NONE
        if not any((index, middle, ring, pinky)):
            return FIST
        return NONE

    # -- swipe detection ---------------------------------------------------
    def detect_swipe(self, landmarks, pose, now=None):
        """Look for a horizontal sweep of the palm across the frame.

        Returns SWIPE_LEFT, SWIPE_RIGHT or None. On a hit the history is
        cleared, so one physical sweep can only ever produce one swipe.
        """
        now = time.monotonic() if now is None else now

        if landmarks is None or pose not in SWIPE_POSES:
            self._track.clear()
            return None

        # Palm centre rather than a fingertip: it barely moves as the fingers
        # articulate, so the trace reflects the arm sweep and not the pose.
        centre = landmarks[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], :2].mean(axis=0)
        self._track.append((now, float(centre[0]), float(centre[1])))

        window = [p for p in self._track if now - p[0] <= settings.SWIPE_WINDOW_SECONDS]
        if len(window) < 4:
            return None

        xs = [p[1] for p in window]
        ys = [p[2] for p in window]
        travel = xs[-1] - xs[0]

        if abs(travel) < settings.SWIPE_MIN_DISTANCE:
            return None
        if max(ys) - min(ys) > settings.SWIPE_MAX_VERTICAL_DRIFT:
            return None

        # Insist the motion was mostly one-directional. Net displacement
        # alone would accept a hand that wandered back and forth and happened
        # to end up displaced.
        steps = np.diff(xs)
        forward = steps[np.sign(steps) == np.sign(travel)].sum()
        if abs(forward) < abs(travel) * 0.9:
            return None

        self._track.clear()
        # The frame is mirrored, so screen-right is the user's right.
        return SWIPE_RIGHT if travel > 0 else SWIPE_LEFT

    def update(self, landmarks, now=None):
        """Classify one frame. Returns (pose, swipe_or_None)."""
        pose = self.classify_pose(landmarks)
        return pose, self.detect_swipe(landmarks, pose, now)


class GestureStateMachine:
    """Debounces raw detections into at-most-one action per intent.

        NONE -> detected -> stable for hold time -> ACTION -> cooldown -> ready

    Swipes pass straight through to ACTION because their hold time is zero,
    but they still take the cooldown on the way out.
    """

    def __init__(self):
        self.candidate = NONE
        self.candidate_since = 0.0
        self.candidate_frames = 0
        self.cooldown_until = 0.0
        self.last_action = None
        self.last_action_at = 0.0

    @property
    def progress(self):
        """How far the held pose is toward firing, in 0..1.

        The UI draws this as a ring so the hold time is visible instead of
        feeling like lag.
        """
        hold = ACTION_HOLD_SECONDS.get(self.candidate, 0.0)
        if not hold or self.candidate_frames < settings.GESTURE_MIN_FRAMES:
            return 0.0
        elapsed = time.monotonic() - self.candidate_since
        return min(1.0, max(0.0, elapsed / hold))

    def in_cooldown(self, now=None):
        now = time.monotonic() if now is None else now
        return now < self.cooldown_until

    def update(self, pose, swipe, now=None):
        """Feed one frame of detection in; get an action back, or None."""
        now = time.monotonic() if now is None else now

        # A swipe is a one-off event, so it is handled before the pose
        # tracking and never becomes a held candidate.
        if swipe is not None:
            if self.in_cooldown(now):
                return None
            return self._fire(swipe, now)

        if pose != self.candidate:
            self.candidate = pose
            self.candidate_since = now
            self.candidate_frames = 1
            return None

        self.candidate_frames += 1

        if self.in_cooldown(now):
            return None

        hold = ACTION_HOLD_SECONDS.get(pose)
        if hold is None:
            return None
        if self.candidate_frames < settings.GESTURE_MIN_FRAMES:
            return None
        if now - self.candidate_since < hold:
            return None

        return self._fire(pose, now)

    def _fire(self, action, now):
        self.last_action = action
        self.last_action_at = now
        self.cooldown_until = now + ACTION_COOLDOWN_SECONDS.get(action, 0.5)
        # Force a fresh hold before the same pose can fire again.
        self.candidate = NONE
        self.candidate_frames = 0
        return action
