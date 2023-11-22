"""A module to store the technical flags."""

from dispel.data.flags import FlagSeverity, FlagType
from dispel.data.levels import Level
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.flags import flag
from dispel.processing.level import FlagLevelStep

MAX_TIME_INCR = 2
r"""Maximum time increment allowed in data in seconds."""
MIN_FREQ_GAIT = 40
r"""Minimum sampling frequency allowed in gait data in Hz."""
MIN_FREQ_SBT = 40
r"""Minimum sampling frequency allowed in SBT data in Hz."""


def detect_fs_below_threshold(level: Level, min_freq: int, **kwargs) -> bool:
    """Detect a sampling frequency violation."""
    acc = level.get_raw_data_set("accelerometer").data
    fs = 1 / acc.ts.diff().dt.total_seconds().median()
    return fs > min_freq


class FrequencyLowerThanThres(FlagLevelStep):
    """Flag record with sampling rate less than a threshold."""

    flag_name = AV("Median frequency lower than {min_freq}Hz", "freq_low_{min_freq}Hz")
    flag_type = FlagType.TECHNICAL
    flag_severity = FlagSeverity.INVALIDATION
    reason = "The median frequency is lower than {min_freq}Hz."


class FrequencyLowerThanGaitThres(FrequencyLowerThanThres):
    """Flag gait record with sampling rate less than a threshold."""

    @flag(min_freq=MIN_FREQ_GAIT)
    def _median_instant_freq_lower_than_thres(
        self, level: Level, **kwargs  # pylint: disable=all
    ) -> bool:
        self.set_flag_kwargs(min_freq=MIN_FREQ_GAIT)
        return detect_fs_below_threshold(level, min_freq=MIN_FREQ_GAIT)


class FrequencyLowerThanSBTThres(FrequencyLowerThanThres):
    """Flag SBT record with sampling rate less than a threshold."""

    @flag(min_freq=MIN_FREQ_SBT)
    def _median_instant_freq_lower_than_thres(
        self, level: Level, **kwargs  # pylint: disable=all
    ) -> bool:
        self.set_flag_kwargs(min_freq=MIN_FREQ_SBT)
        return detect_fs_below_threshold(level, min_freq=MIN_FREQ_SBT)


class MaxTimeIncrement(FlagLevelStep):
    """Flag record with maximum time increment larger than threshold."""

    flag_name = AV(
        f"Maximum time increment larger than {MAX_TIME_INCR} seconds",
        f"max_time_inc_greater_than_{MAX_TIME_INCR}_sec",
    )
    flag_type = FlagType.TECHNICAL
    flag_severity = FlagSeverity.INVALIDATION
    reason = f"The maximum time increment is higher than " f"{MAX_TIME_INCR} seconds."

    @flag
    def _max_time_increment_larger_than(
        self, level: Level, **kwargs  # pylint: disable=all
    ) -> bool:
        acc = level.get_raw_data_set("accelerometer").data
        time_inc = acc.ts.diff().dt.total_seconds().max()
        return time_inc < MAX_TIME_INCR
