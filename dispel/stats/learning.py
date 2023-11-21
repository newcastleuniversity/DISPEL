"""Inter-session Learning analysis.

A module where functions are provided to compute and extract learning parameters from
measure collections containing processed measures. The module provides class and
functions to compute and extract parameters from fitted model by curve fit and compute
relevant learning related measures.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm, zscore
from sklearn.metrics import r2_score

from dispel.data.collections import MeasureCollection

NumericType = Union[float, int, np.ndarray, pd.Series]


@dataclass(frozen=True)
class LearningCurve:
    """Class ensemble of learning curve parameters."""

    #: The asymptote of the fitted learning curve.
    asymptote: float

    #: The slope of the fitted learning curve.
    slope: float

    def to_dict(self):
        """Convert learning curve information to dictionary format."""
        return dict(
            optimal_performance=self.asymptote,
            slope_coefficient=self.slope,
            learning_rate=self.learning_rate,
        )

    @staticmethod
    def compute_learning(x: NumericType, a: NumericType, b: NumericType) -> NumericType:
        """Compute learning curve function."""
        return a - b / x

    def __call__(self, data: NumericType) -> NumericType:
        """Compute the learning curve for a given input (trial number)."""
        return self.compute_learning(data, self.asymptote, self.slope)

    @property
    def learning_rate(self) -> float:
        """Get the learning rate related to curve."""
        return self.slope / self.asymptote

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "LearningCurve":
        """Fit learning curve using :func:`scipy.optimize.curve_fit`.

        See :meth:`dispel.stats.learning.LearningCurve.compute_learning`.

        Parameters
        ----------
        x
            The trial numbers associated with data points.
        y
            The measure data points.

        Returns
        -------
        LearningCurve
            The fitted learning curve.
        """
        (asymptote, slope), *_ = curve_fit(cls.compute_learning, x, y)
        return cls(asymptote=asymptote, slope=slope)

    def get_warm_up(self, data: Union[pd.Series, np.ndarray]) -> int:
        """Compute the warm-up argmax for measure values and fitted parameters.

        The ``warm_up`` here is actually the minimum number of trials the user has to
        perform in order to reach 90% of the optimal performance (asymptote value) given
        by the model.

        Parameters
        ----------
        data
            A numpy array series containing ordered measure values.

        Returns
        -------
        int
            The argmax of the first occurrence of the measure values that reaches 90% of
            the optimal performance given by the model.

        Raises
        ------
        TypeError
            If the given data is not a pandas series nor a numpy array.
        """
        if data.size == 0:
            return cast(int, np.nan)

        if self.slope < 0:
            threshold_90 = 0.9 * (self.asymptote - (max_ := data.max())) + max_
            argmax_index = np.argmax(data < threshold_90)
        else:
            threshold_90 = 0.9 * (self.asymptote - (min_ := data.min())) + min_
            argmax_index = np.argmax(data > threshold_90)

        if isinstance(data, np.ndarray):
            return int(argmax_index)
        if isinstance(data, pd.Series):
            return data.index[argmax_index]
        raise TypeError(
            f"Unsupported data type {type(data)}. Only a ``pandas.Series`` or "
            "``numpy.ndarray`` is allowed."
        )

    @classmethod
    def empty(cls):
        """Return empty learning curve."""
        return cls(asymptote=np.nan, slope=np.nan)


@dataclass(frozen=True)
class LearningModel:
    """Class ensemble of learning model."""

    #: The fitted learning curve.
    curve: LearningCurve

    #: The data points without outliers
    new_data: pd.Series

    # Thr R squared score of the fitted learning model.
    r2_score: Optional[float]

    #: The number of outliers rejected during the model fitting.
    nb_outliers: Optional[int] = 0

    def to_dict(self):
        """Convert learning model information to dictionary format."""
        return dict(
            **self.curve.to_dict(),
            warm_up=self.curve.get_warm_up(self.new_data),
            r2_score=self.r2_score,
            nb_outliers=self.nb_outliers,
        )

    @classmethod
    def empty(cls) -> "LearningModel":
        """Return empty learning model."""
        return cls(
            curve=LearningCurve.empty(),
            new_data=pd.Series(dtype="float64"),
            r2_score=None,
            nb_outliers=None,
        )


@dataclass(frozen=True)
class DelayParameters:
    """Class ensemble of delay parameters."""

    #: The mean delay between sessions.
    mean: Optional[float]

    #: The median delay between sessions.
    median: Optional[float]

    #: The maximum delay between sessions.
    max: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert learning parameters to dictionary format."""
        return dict(delay_mean=self.mean, delay_median=self.median, delay_max=self.max)

    @classmethod
    def empty(cls) -> "DelayParameters":
        """Return empty delay parameters."""
        return cls(mean=None, median=None, max=None)


@dataclass(frozen=True)
class LearningParameters:
    """Class ensemble of learning parameters."""

    #: The subject's id
    subject_id: str

    #: The measure id
    measure_id: str

    #: The learning model
    model: LearningModel

    #: The delay parameters in days
    delay_parameters: DelayParameters

    def to_dict(self) -> Dict[str, Optional[Union[float, int, str]]]:
        """Convert learning parameters to dictionary format."""
        return dict(
            subject_id=self.subject_id,
            measure_id=self.measure_id,
            **self.model.to_dict(),
            **self.delay_parameters.to_dict(),
        )


def compute_delay(data: pd.Series) -> DelayParameters:
    """Extract mean, median and maximum delay between consecutive sessions.

    Parameters
    ----------
    data
        A pandas series containing timestamps.

    Returns
    -------
    DelayParameters
        A :class:`dispel.stats.learning.DelayParameters` with the values of the mean,
        median and maximum delay between consecutive trials for a given measure and
        subject in days.
    """
    day_diff = data.diff().dt.total_seconds() / 86_400
    delay_mean, delay_median, delay_max = day_diff.agg(["mean", "median", "max"])
    return DelayParameters(mean=delay_mean, median=delay_median, max=delay_max)


def reject_outliers(data: pd.Series, sigma: float) -> pd.Series:
    """Reject outliers with Z-score outside the tolerated bounds.

    Parameters
    ----------
    data
        A pandas series composed of measure values for only one measure and only one
        user and trials numbers as index.
    sigma
        The standard deviation threshold above which the data points are to be
        considered as outliers and therefore rejected.

    Returns
    -------
    pandas.Series
        The data without the detected outlier (if detected) with the same structure as
        the entry.
    """
    # Fit of the model by curve fit.
    baseline_curve = LearningCurve.fit(data.index.values, data.values)

    # Compute predictions and residuals of fitted model curve
    predictions = baseline_curve(data.index.values)
    residuals = (data - predictions).abs()

    if residuals.std() == 0.0 or len(data) <= 3:
        return data.copy()

    # Compute Z-scores of residuals
    zscores = pd.Series(zscore(residuals), index=residuals.index)

    if len(data[zscores >= sigma]) == 0:
        return data.copy()
    return reject_outliers(data.drop(zscores.idxmax()), sigma)


def compute_learning_model(
    data: pd.Series, tolerance: float = 0.99, reset_trials: bool = True
) -> Tuple[LearningModel, DelayParameters]:
    """Compute the learning model.

    Parameters
    ----------
    data
        A pandas series composed of measure values for only one measure and only one
        user and trials numbers as index.
    tolerance
        The tolerance threshold above which the data points are to be considered
        outliers and therefore rejected. Should be between ``0`` and ``1``.
    reset_trials
        ``True`` if the trial numbers are to be reset for the new data (without
        outliers). ``False`` otherwise.

    Returns
    -------
    Tuple[LearningModel, DelayParameters]
        The output contains the following information:

            - The fitted learning model.
            - The delay parameters.

    Raises
    ------
    ValueError
        If the threshold tolerance is outside the legal bounds i.e. [0, 1].
    """
    if tolerance < 0 or tolerance > 1:
        raise ValueError(
            f"Unsupported tolerance threshold value: {tolerance}. Must be between 0 "
            f"and 1."
        )

    if data.size < 2:
        return LearningModel.empty(), DelayParameters.empty()

    # Reject outliers
    trial_data = data.copy()
    trial_data.index = trial_data.index.get_level_values("trial")
    new_data = reject_outliers(trial_data, sigma=norm.ppf(tolerance))

    # Retrieve trial start dates and compute delay parameters
    all_start_dates = data.index.to_frame().set_index("trial")
    start_dates = all_start_dates[all_start_dates.index.isin(new_data.index)][
        "start_date"
    ]
    delay_parameters = compute_delay(start_dates)

    if reset_trials:
        new_data.index = pd.Series(np.arange(1, len(new_data) + 1), name="trial")

    model_curve = LearningCurve.fit(new_data.index.values, new_data.values)

    predictions = model_curve(new_data.index)

    model = LearningModel(
        curve=model_curve,
        new_data=new_data,
        nb_outliers=len(trial_data) - len(new_data),
        r2_score=r2_score(new_data, predictions),
    )
    return model, delay_parameters


class LearningResult:
    """The learning results for one measure and one or multiple subjects."""

    _COLUMNS = [
        "subject_id",
        "optimal_performance",
        "slope_coefficient",
        "learning_rate",
        "warm_up",
        "r2_score",
        "nb_outliers",
        "delay_mean",
        "delay_median",
        "delay_max",
    ]

    def __init__(self):
        self.measure_id = None
        self._parameters = pd.DataFrame(columns=self._COLUMNS)
        self._new_data: Dict[str, pd.Series] = {}

    def _add_learning_result(self, other: "LearningResult"):
        # pylint: disable=protected-access
        if other.measure_id:
            if self.measure_id and self.measure_id != other.measure_id:
                raise ValueError(
                    "Cannot append learning results for different measures."
                )
            self.measure_id = other.measure_id
            self._parameters = self._parameters.append(
                other._parameters, ignore_index=True
            )
            self._new_data = {
                **self._new_data,
                **other._new_data,
            }

    def __add__(self, other):
        if isinstance(other, LearningResult):
            (res := LearningResult())._add_learning_result(self)
            res._add_learning_result(other)
            return res
        raise TypeError("Can only add LearningResults.")

    def __iadd__(self, other):
        if isinstance(other, LearningResult):
            self._add_learning_result(other)
            return self

        raise TypeError("Can only add LearningResults.")

    @classmethod
    def from_parameters(cls, learning_parameters: LearningParameters):
        """Initialize learning result from parameters.

        Parameters
        ----------
        learning_parameters
            The learning parameters for the measure and subject in question.

        Returns
        -------
        LearningResult
            The learning result regrouping the given information.
        """
        (res := cls()).append(learning_parameters)
        return res

    def append(self, learning_parameters: LearningParameters):
        """Append new learning results for one subject to learning results.

        Parameters
        ----------
        learning_parameters
            The learning parameters for the measure and subject in question.

        Raises
        ------
        ValueError
            If the learning parameters are for a different measure than the one
            concerning the learning result.
        """
        self.measure_id = self.measure_id or learning_parameters.measure_id
        if self.measure_id != learning_parameters.measure_id:
            raise ValueError("Cannot append learning results for different measures.")

        self._parameters = self._parameters.append(
            learning_parameters.to_dict(), ignore_index=True
        )
        self._new_data[
            learning_parameters.subject_id
        ] = learning_parameters.model.new_data

    def get_parameters(
        self, subject_id: Optional[str] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """Get learning results for one or all subjects.

        Parameters
        ----------
        subject_id
            The subject identifier for which the learning is to be retrieved. If
            ``None`` is provided all learning results will be given.

        Returns
        -------
        Union[pandas.Series, pandas.DataFrame]
            If a valid subject id is given, the output is a pandas series summarizing
            learning results. If ``None`` is given the output will be a pandas data
            frame summarizing all learning results.

        Raises
        ------
        ValueError
            If the subject identifier is not found in the learning analysis results.
        """
        if subject_id is None:
            return self._parameters.copy()
        if subject_id not in (ids := set(self._parameters["subject_id"])):
            raise ValueError(
                f"The given subject id {subject_id} is not found. Must be within the "
                f"following values: {ids}."
            )
        return self._parameters[self._parameters["subject_id"] == subject_id]

    def get_new_data(self, subject_id: str) -> pd.Series:
        """Get the new data points without outliers.

        Parameters
        ----------
        subject_id
            The identifier of the subject for which the new data is to be retrieved.

        Returns
        -------
        pandas.Series
            A pandas series containing the new data points for the measure in question
            (without outliers).

        Raises
        ------
        ValueError
            If the subject identifier is not found in the learning analysis results.
        """
        if subject_id not in self._new_data:
            raise ValueError(
                f"Subject not found {subject_id}. Must be withing following values: "
                f"{self._new_data.keys()}"
            )
        return self._new_data[subject_id]


def extract_learning_for_one_subject(
    measure_collection: MeasureCollection,
    subject_id: str,
    measure_id: str,
    tolerance: float = 0.99,
    reset_trials: bool = True,
) -> LearningResult:
    """Compute learning for a unique subject and a unique measure.

    Parameters
    ----------
    measure_collection
        A measure collection containing any measures and any subjects.
    subject_id
        The identifier of the subject for which the delay is to be computed.
    measure_id
        The identifier of the measure for which the delay is to be computed.
    tolerance
        The tolerance threshold above which the data points are to be considered
        outliers and therefore rejected. Should be between ``0`` and ``1``.
    reset_trials
        ``True`` if the trial numbers are to be reset for the new data (without
        outliers). ``False`` otherwise.

    Returns
    -------
    LearningResult
        The learning result for one subject of the measure in question. See:
        :class:`dispel.stats.learning.LearningResult`.
    """
    # Retrieve measure values
    measure_values = measure_collection.get_measure_values_over_time(
        subject_id=subject_id, measure_id=measure_id, index=["start_date", "trial"]
    ).dropna()

    # Compute learning model and delay parameters
    model, delay_parameters = compute_learning_model(
        measure_values, tolerance, reset_trials
    )

    return LearningResult.from_parameters(
        LearningParameters(
            subject_id=subject_id,
            measure_id=measure_id,
            model=model,
            delay_parameters=delay_parameters,
        )
    )


def extract_learning_for_all_subjects(
    measure_collection: MeasureCollection,
    measure_id: str,
    tolerance: float = 0.99,
    reset_trials: bool = True,
) -> LearningResult:
    """Compute learning parameters for all subjects in a measure collection.

    Parameters
    ----------
    measure_collection
        A measure collection containing any measures and any subjects.
    measure_id
        The measure id on which the learning parameters are to be computed.
    tolerance
        The tolerance threshold above which the data points are to be considered
        outliers and therefore rejected. Should be between ``0`` and ``1``.
    reset_trials
        ``True`` if the trial numbers are to be reset for the new data (without
        outliers). ``False`` otherwise.

    Returns
    -------
    LearningResult
        The learning result for all subjects of the measure in question. See:
        :class:`dispel.stats.learning.LearningResult`.
    """
    learning_results = (
        extract_learning_for_one_subject(
            measure_collection,
            subject_id=subject_id,
            measure_id=measure_id,
            tolerance=tolerance,
            reset_trials=reset_trials,
        )
        for subject_id in measure_collection.subject_ids
    )

    return sum(learning_results, LearningResult())
