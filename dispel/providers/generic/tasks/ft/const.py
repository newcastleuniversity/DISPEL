"""A utils module for the finger tapping assessment."""
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import AVEnum

TASK_NAME = AV("Finger Tapping Assessment", "FT")

#: The attributes of the class `processing.generic.touch.Touch` to be extracted
TOUCH_ATTRIBUTES = ["begin", "end", "duration", "first_position"]

#: Enriched touch attributes calculated from the base touch attributes
ENRICHED_TOUCH_ATTRIBUTES = ["tap_duration", "location"]

#: Flags threshold
MAX_TAPPING_INTERVAL = 5
MAX_ONE_SIDED_TAPS = 7
MIN_NB_VALID_TAPS = 20


class TappingTarget(AVEnum):
    """Enumerated constant representing the finger tapping events."""

    LEFT = ("leftzone", "left")
    RIGHT = ("rightzone", "right")
    OUTSIDE = ("outsidezone", "none")


class AllHandsModalities(AVEnum):
    """Combine DA and Konectom level modalities."""

    RIGHT_HAND = ("right hand", "right")
    LEFT_HAND = ("left hand", "left")
    DOMINANT_HAND = ("dominant hand", "domhand")
    NON_DOMINANT_HAND = ("non dominant hand", "nondomhand")
