"""A module containing all the finger tapping invalidations."""
from typing import List

from dispel.data.flags import FlagSeverity
from dispel.data.levels import Level
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing import ProcessingStep
from dispel.processing.flags import flag
from dispel.processing.level import FlagLevelStep, ProcessingStepGroup
from dispel.providers.generic.tasks.ft.const import (
    MAX_ONE_SIDED_TAPS,
    MAX_TAPPING_INTERVAL,
    MIN_NB_VALID_TAPS,
    TASK_NAME,
    AllHandsModalities,
)


class FlagTappingInterval(FlagLevelStep):
    """A Level Validation step to detect invalid tapping pause."""

    flag_name = AV("tapping pause", "tapping_pause")
    flag_type = "behavioral"
    reason = (
        "A tapping interval greater than 3 seconds has been detected for {level_id}"
    )
    flag_severity = FlagSeverity.DEVIATION

    @flag
    def _validate_max_tapping_pause(self, level: Level, **kwargs) -> bool:
        self.set_flag_kwargs(level_id=level.id)
        taps = level.get_raw_data_set("enriched_tap_events_ts").data.copy()
        return (
            MAX_TAPPING_INTERVAL
            > taps.index.to_series().diff().dt.total_seconds().max()
        )


class FlagOneSidedTaps(FlagLevelStep):
    """A Level Validation step to flag one-sided tapping."""

    flag_name = AV("one sided tapping", "one_sided_tapping")
    flag_type = "behavioral"
    reason = (
        f"More than {MAX_ONE_SIDED_TAPS} consecutive taps in the same zone"
        + " for {level_id}"
    )
    flag_severity = FlagSeverity.DEVIATION

    @flag
    def _validate_max_tapping_pause(self, level: Level, **kwargs) -> bool:
        self.set_flag_kwargs(level_id=level.id)
        taps = level.get_raw_data_set("enriched_tap_events_ts").data.copy()["location"]
        max_one_sided_tap = (
            taps.groupby((taps != taps.shift()).cumsum()).cumcount() + 1
        ).max()
        return MAX_ONE_SIDED_TAPS > max_one_sided_tap


class FlagMinNumberOfTaps(FlagLevelStep):
    """A Level Validation step to flag the miniumum number of taps."""

    task_name = TASK_NAME
    flag_name = AV("Invalid number of valid taps", "invalid_min_n_taps")
    flag_type = "behavioral"
    reason = f"Less than {MIN_NB_VALID_TAPS} valid taps" + " for {level_id}"
    flag_severity = FlagSeverity.DEVIATION

    @flag
    def _validate_min_numbers_of_taps(self, level: Level, **kwargs) -> bool:
        self.set_flag_kwargs(level_id=level.id)
        valid_taps = level.get_raw_data_set("valid_enriched_tap_events_ts").data.copy()
        return valid_taps.shape[0] > MIN_NB_VALID_TAPS


class GenericFlagsStepGroup(ProcessingStepGroup):
    """Generic step group for finger tapping flags."""

    steps: List[ProcessingStep] = [
        ProcessingStepGroup(
            [
                FlagOneSidedTaps(),
                FlagTappingInterval(),
                FlagMinNumberOfTaps(),
            ],
            level_filter=hand.abbr,
            modalites=[hand.abbr],
        )
        for hand in AllHandsModalities
    ]
