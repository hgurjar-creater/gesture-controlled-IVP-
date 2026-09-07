"""The filter registry and the current-selection state.

Filters are held in a fixed order so the index maps to the same look
every run: gesture navigation, the number keys and the on-screen strip
all agree because they all read this one list.
"""

from . import artistic_filters as artistic
from . import basic_filters as basic
from . import segmenter
from . import face_filter


class FilterEngine:
    """Owns the ordered filter list and which one is currently selected."""

    FILTERS = (
        ("Original", basic.original),
        ("Blur BG", segmenter.background_blur),
        ("Anonymous", face_filter.anonymous_filter),
        ("Black & White", basic.grayscale),
        ("Sepia", basic.sepia),
        ("Warm", basic.warm),
        ("Cool", basic.cool),
        ("Vintage", artistic.vintage),
        ("Bright", basic.bright),
        ("Dreamy", artistic.dreamy),
        ("Doodle", artistic.doodle),
    )

    def __init__(self, index=0):
        self.index = index % len(self.FILTERS)

    # -- selection ---------------------------------------------------------
    @property
    def name(self):
        return self.FILTERS[self.index][0]

    @property
    def names(self):
        return [name for name, _ in self.FILTERS]

    def next(self):
        self.index = (self.index + 1) % len(self.FILTERS)
        return self.name

    def previous(self):
        self.index = (self.index - 1) % len(self.FILTERS)
        return self.name

    def select(self, index):
        if 0 <= index < len(self.FILTERS):
            self.index = index
        return self.name

    # -- application -------------------------------------------------------
    def apply(self, frame):
        """Run the selected filter over a frame of any resolution."""
        return self.FILTERS[self.index][1](frame)

    def apply_named(self, frame, name):
        """Run a filter by name -- used so a queued capture keeps the look
        that was selected when the shutter fired, not whatever is selected
        by the time the worker thread gets to it."""
        for filter_name, function in self.FILTERS:
            if filter_name == name:
                return function(frame)
        raise KeyError(f"unknown filter: {name!r}")

    def __len__(self):
        return len(self.FILTERS)
