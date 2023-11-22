"""Constants used in SBT module."""

from dispel.data.values import AbbreviatedValue as AV

KERNEL_WINDOW_SEGMENT = 151
"""The window length for the median filter smoothing during segmentation."""
FIXED_SIGN_AMP_THRESHOLD = 0.1
"""The maximum amplitude allowed to perform the statistical thresholding
during segmentation."""

MIN_MOTION_DUR = 1
"""The minimum excessive motion duration (in seconds)."""
MIN_COVERAGE_THRESHOLD = 80.0
"""The minimum % of signal left after segmentation for a flag."""

TASK_NAME_SBT = AV("Static Balance Test", "sbt")
