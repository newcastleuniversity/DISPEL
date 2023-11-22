"""ADS exclusive data model for the analysis library."""
from typing import Any, Iterable, List, Optional

import pandas as pd
from packaging import version

from dispel.data.collections import MeasureSet
from dispel.data.core import Evaluation, Reading, ReadingSchema, Session
from dispel.data.devices import Device
from dispel.data.levels import Context, Level, Modalities
from dispel.data.values import ValueDefinition
from dispel.utils import camel_to_snake_case


class ADSReading(Reading):
    """A data capture from an ADS experiment.

    Parameters
    ----------
    evaluation
        The evaluation information
    session
        The session information
    levels
        An iterable of levels
    measure_set
        A list of measures already processed on the device
    schema
        The schema of the reading
    date
        The time the reading was recorded
    device
        The device that captured the reading
    """

    def __init__(
        self,
        evaluation: Evaluation,
        session: Session,
        levels: Optional[Iterable[Level]] = None,
        measure_set: Optional[MeasureSet] = None,
        schema: Optional[ReadingSchema] = None,
        date: Any = None,
        device: Optional[Device] = None,
    ):
        date_ = pd.Timestamp(date, unit="ms")
        super().__init__(
            evaluation, session, levels, measure_set, schema, date_, device
        )


def get_snake_case_context_value(context: Context, key: str) -> str:
    """Find the context with its name into the list, converted in snake case.

    Parameters
    ----------
    context
        A :class:`dispel.data.core.Context`.
    key
        The desire levelType

    Returns
    -------
    str
        The context value into snake case
    """
    value = context.get_raw_value(key)
    return camel_to_snake_case(value)


class ADSModalities(Modalities):
    """An entity to match level context modalities with a ``Level.id``.

    Specific to ADS Readings.
    """

    def __init__(self, app_version: Optional[str]):
        mapping = {
            "6mwt": self.get_gait_level_from_context,
            "cps": self.get_cps_level_from_context,
            "drawing": self.get_drawing_level_from_context,
            "gripForce": self.get_grip_level_from_context,
            "mood": self.get_mood_level_from_context,
            "msis29": self.get_msis29_level_from_context,
            "passive": self.get_passive_level_from_context,
            "pinch": self.get_pinch_level_from_context,
            "sbtUtt": self.get_sbt_utt_level_from_context,
        }
        super().__init__(mapping, app_version)

    @property
    def is_default_mode(self):
        """Return whether the app version is the default mode."""
        return self.app_version > version.parse("0.2.8")

    @staticmethod
    def get_cps_level_from_context(context: Context) -> str:
        """Find the level from context enumerating through the modalities.

        This function is specific to the ADS CPS test.

        Parameters
        ----------
        context
            The context of the level in question.

        Returns
        -------
        str
            The snake case level name.
        """
        res = get_snake_case_context_value(context, "levelType")
        # overwrite old CPS identifier < Konectom 0.2.8
        if res == "digit_to_symbol":
            return "symbol_to_digit"

        return res

    @staticmethod
    def get_gait_level_from_context(_: List) -> str:
        """Find the only level of the ADS 6MW test."""
        return "6mwt"

    @staticmethod
    def get_passive_level_from_context(_: List) -> str:
        """Find the only level of the ADS Passive test."""
        return "passive"

    @staticmethod
    def get_sbt_utt_level_from_context(context: Context) -> str:
        """Find the level from context enumerating through the modalities.

        This function is specific to the ADS SBT-UTT test.

        Parameters
        ----------
        context
            The context of the level in question.

        Returns
        -------
        str
            The snake case level name.
        """
        return get_snake_case_context_value(context, "levelType")

    @staticmethod
    def get_drawing_level_from_context(context: Context) -> str:
        """Find the level from context enumerating through the modalities.

        This function is specific to the ADS CPS test.

        Parameters
        ----------
        context
            The context of the level in question.

        Returns
        -------
        str
            The snake case level name.
        """
        name = get_snake_case_context_value(context, "levelType")
        hand = get_snake_case_context_value(context, "usedHand")

        return name + "-" + hand

    def get_pinch_level_from_context(self, context: Context) -> List[str]:
        """Find the level from context enumerating through the modalities.

        This function is specific to the ADS Pinch test.

        Parameters
        ----------
        context
            The context of the level in question.

        Returns
        -------
        str
            The snake case level name.

        Raises
        ------
        ValueError
            If the radius of pinch bubble does not correspond to any known radius
            interval.
        """

        def _target_radius_to_modality(radius: float) -> str:
            if 50 <= radius < 65:
                return "small"
            if 65 <= radius < 80:
                return "medium"
            if 80 <= radius < 95:
                return "large"
            if 95 <= radius <= 105:
                return "extra_large"

            raise ValueError(f"target radius {radius} does not match any size")

        # For non default mode replace diameter with radius
        if not self.is_default_mode:
            context.set(
                value=context["targetRadius"].value / 2,
                definition=context.get_definition("targetRadius"),
                overwrite=True,
            )

        # enumerate through the context to find usedHand and bubbleSize
        hand_modality = context["usedHand"].value
        target_radius = context["targetRadius"].value

        # check we have enough data
        if hand_modality != "":
            target_radius_modality = _target_radius_to_modality(target_radius)
            context.set(
                value=target_radius_modality,
                definition=ValueDefinition("bubbleSize", "bubble size modality"),
            )
            return [hand_modality, target_radius_modality]

        raise ValueError("userHand has not been found in context")

    @staticmethod
    def get_grip_level_from_context(context: Context) -> str:
        """Find the level from context enumerating through the modalities.

        This function is specific to the ADS GRIP test.

        Parameters
        ----------
        context
            The context of the level in question.

        Returns
        -------
        str
            The snake case level name.
        """
        return get_snake_case_context_value(context, "usedHand")

    @staticmethod
    def get_mood_level_from_context(context: Context) -> str:
        """Find the level from context enumerating through the modalities.

        This function is specific to the ADS MOOD survey.

        Parameters
        ----------
        context
            The context of the level in question.

        Returns
        -------
        str
            The snake case level name.
        """
        return get_snake_case_context_value(context, "idMoodscale")

    @staticmethod
    def get_msis29_level_from_context(context: Context) -> str:
        """Find the level from context enumerating through the modalities.

        This function is specific to the ADS MSIS-29 test.

        Parameters
        ----------
        context
            The context of the level in question.

        Returns
        -------
        str
            The snake case level name.
        """
        return get_snake_case_context_value(context, "idMsis29")
