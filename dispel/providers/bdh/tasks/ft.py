"""Processing functionality for Finger Tapping (FT) task."""
from typing import List, Union

import pandas as pd

from dispel.data.raw import RawDataValueDefinition
from dispel.processing.core import ProcessingStep
from dispel.processing.data_set import transformation
from dispel.processing.level import ProcessingStepGroup
from dispel.processing.transform import TransformStep
from dispel.providers.bdh.data import BDHReading
from dispel.providers.generic.sensor import SetTimestampIndex
from dispel.providers.generic.tasks.ft.const import TOUCH_ATTRIBUTES
from dispel.providers.generic.tasks.ft.steps import TASK_NAME, GenericFingerTappingSteps
from dispel.providers.generic.touch import TOUCH_COLUMNS, Gesture, split_touches
from dispel.providers.registry import process_factory


class TransformRawTouchEvents(TransformStep):
    """Preprocess the io raw touch events.

    The touch path ids that we receive with the io format are not increasing
    over time, this is just a counter of the number of fingers on the screen.
    In order to use the :class: `~dispel.providers.generic.touch.Gesture`
    module we need to transform those ids into an increasing sequence with one
    touch path id per tap.
    """

    data_set_ids = "screen"
    new_data_set_id = "processed_screen"
    definitions = [RawDataValueDefinition(column, column) for column in TOUCH_COLUMNS]

    @staticmethod
    @transformation
    def preprocess_raw_touch_events(data):
        """Pre-process the raw touch events."""
        return split_touches(
            data, begin=data["tsTouch"].min(), end=data["tsTouch"].max()
        )


class SetTimestampIndexBDHonly(SetTimestampIndex):
    """BDH specific set timestamp processing step."""


class TransformGesture(TransformStep):
    """Generate a gesture dataset from the preprocessed raw touch events."""

    data_set_ids = "processed_screen"
    new_data_set_id = "gestures"
    definitions = [RawDataValueDefinition("gestures", "gestures")]

    @staticmethod
    @transformation
    def _generate_gesture(data):
        return pd.Series(Gesture.from_data_frame(data))


class TransformTapsFromRaw(TransformStep):
    """Retrieve the tap events from the gesture dataset."""

    data_set_ids = "gestures"
    new_data_set_id = "taps_from_raw"
    definitions = [
        RawDataValueDefinition(column, column)
        for column in TOUCH_ATTRIBUTES + ["tap_duration"]
    ]

    @staticmethod
    @transformation
    def _generate_taps_from_raw(data):
        new_data = {}
        # Explode the gesture dataset to have individual taps
        exploded_touch = (
            data["gestures"].apply(lambda x: x.touches).explode().reset_index(drop=True)
        )

        # Extract the touch attributes
        for attribute in TOUCH_ATTRIBUTES:
            new_data[attribute] = exploded_touch.apply(lambda x: getattr(x, attribute))

        df = pd.DataFrame(new_data).sort_values(by="begin")
        df["tap_duration"] = (df["end"] - df["begin"]).dt.total_seconds() * 1e3

        return df


class TransformTapEvents(TransformStep):
    """Generate a gesture dataset from the preprocessed raw touch events."""

    data_set_ids = ["taps_from_raw", "tap_events_ts"]
    new_data_set_id = "enriched_tap_events"
    definitions = [
        RawDataValueDefinition(column, column)
        for column in TOUCH_ATTRIBUTES + ["tap_duration", "location"]
    ]

    @staticmethod
    @transformation
    def enrich(raw_ts, taps):
        """Enrich the tap events with the information provided by the raw events."""
        raw_ts_copy = raw_ts.copy().sort_values(by="begin")
        taps_copy = taps.copy().sort_index()
        if len(raw_ts) == len(taps):
            return pd.concat([taps_copy.reset_index(), raw_ts_copy], axis=1).drop(
                columns="timestamp"
            )
        return pd.merge_asof(
            raw_ts_copy,
            taps_copy,
            left_on="begin",
            right_index=True,
            direction="nearest",
        )


class BDHPreprocessingStepGroup(ProcessingStepGroup):
    """BDH preprocessing step group for finger tapping."""

    steps: List[Union[ProcessingStep, ProcessingStepGroup]] = [
        TransformRawTouchEvents(),
        # Generate gesture from the raw  touch events
        TransformGesture(),
        # Generate the tap events from the gesture dataset
        TransformTapsFromRaw(),
        SetTimestampIndexBDHonly(
            data_set_id="tap_events",
            columns=["location"],
            time_stamp_column="timestamp",
            duplicates="first",
        ),
        TransformTapEvents(),
        SetTimestampIndexBDHonly(
            data_set_id="enriched_tap_events",
            columns=["location", "end", "tap_duration", "first_position"],
            time_stamp_column="begin",
            duplicates="first",
        ),
    ]


class BDHSteps(ProcessingStepGroup):
    """BDH steps used to process finger tapping records."""

    steps: List[ProcessingStep] = [
        BDHPreprocessingStepGroup(),
        GenericFingerTappingSteps(),
    ]


process_ft = process_factory(
    task_name=TASK_NAME,
    steps=BDHSteps(),
    codes="fingertap-activity",
    supported_type=BDHReading,
)
