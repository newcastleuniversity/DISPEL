"""Specific functionality for signal processing of the tremor detection."""
from typing import List, Optional

from dispel.processing import ProcessingStep
from dispel.processing.level import LevelFilterType, ProcessingStepGroup
from dispel.processing.modalities import SensorModality
from dispel.processing.transform import Add, Apply
from dispel.providers.generic.sensor import (
    ExtractAverageSignalEnergy,
    ExtractPowerSpectrumMeasures,
)
from dispel.signal.core import euclidean_norm, uniform_power_spectrum


class PowerSpectrumDensity(Apply):
    """A transform step to compute the power spectrum of a signal.

    Parameters
    ----------
    data_set_id
        The data set id on which the transformation is to be performed.
    columns
        The columns onto which the signal's tremor filter is applied.
    new_data_set_id
        The new ``id`` used for the :class:`~dispel.data.raw.RawDataSetDefinition`
        .
    level_filter
        An optional :class:`~dispel.processing.level.LevelFilter` to determine
        the levels to be transformed.
    """

    def __init__(
        self,
        data_set_id: str,
        columns: Optional[List[str]] = None,
        new_data_set_id: Optional[str] = None,
        level_filter: Optional[LevelFilterType] = None,
    ):
        columns = columns or list("xyz")
        super().__init__(
            data_set_id=data_set_id,
            method=uniform_power_spectrum,
            columns=columns,
            new_data_set_id=new_data_set_id,
            level_filter=level_filter,
        )


class TremorMeasures(ProcessingStepGroup):
    r"""A group of tremor processing steps according a given data set.

    Parameters
    ----------
    sensor
        The type of sensor on which the extraction is to be performed.
    data_set_id
        The data set id on which the transformation is to be performed ('accelerometer',
        'gyroscope').
    columns
        The columns onto which the signal's tremor measures are to be extracted.
    lower_bound
        The lower bound of frequencies below which the signal is filtered.
    upper_bound
        The upper bound of frequencies above which the signal is filtered.
    level_filter
        An optional :class:`~dispel.processing.level.LevelFilter` to determine the levels
        to be transformed. If no filter is provided, all levels will be transformed. The
        ``level_filter`` also accepts :class:`str`, :class:`~dispel.data.core.LevelId`\ s
        and lists of either and passes them to a
        :class:`~dispel.processing.level.LevelIdFilter` for convenience.
    add_norm
        An optional boolean to determine if the norm should be added to the columns.

    Notes
    -----
    The lower and upper bound of the band filter values are set by default to 2.0 and
    5.0.
    """

    def __init__(
        self,
        sensor: SensorModality,
        data_set_id: str,
        lower_bound: float = 2.0,
        upper_bound: float = 5.0,
        add_norm: bool = True,
        add_average_signal: bool = True,
        columns: Optional[List[str]] = None,
        level_filter: Optional[LevelFilterType] = None,
    ):
        # initialize the columns if they are not specified
        columns = columns or list("xyz")

        # initialize processing steps
        steps: List[ProcessingStep] = []

        # Define a new data set id specific to the power spectrum measures
        power_spectrum_id = data_set_id

        # Optional addition of the norm
        if add_norm:
            add_euclidean_norm = Add(
                data_set_id=data_set_id, method=euclidean_norm, columns=columns
            )
            steps.append(add_euclidean_norm)
            power_spectrum_id = add_euclidean_norm.new_data_set_id
            all_columns = [*columns, "".join(columns)]
        else:
            all_columns = columns

        # PSD Transformation
        psd_step = PowerSpectrumDensity(
            data_set_id=power_spectrum_id,
            columns=all_columns,
        )
        steps.append(psd_step)
        power_spectrum_id = psd_step.new_data_set_id

        # Extraction
        if add_average_signal:
            steps.append(
                ExtractAverageSignalEnergy(
                    sensor=sensor,
                    data_set_id=data_set_id,
                    columns=columns,
                )
            )
        steps.append(
            ExtractPowerSpectrumMeasures(
                sensor=sensor,
                data_set_id=power_spectrum_id,
                columns=all_columns,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        )

        super().__init__(steps, level_filter=level_filter)
