"""A module for collections of measure values."""
import warnings
from heapq import nsmallest
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    ValuesView,
    cast,
)

import numpy as np
import pandas as pd
from jellyfish import damerau_levenshtein_distance  # pylint: disable=E0611

from dispel import __version__
from dispel.data.core import Evaluation, Reading, Session
from dispel.data.measures import MeasureSet, MeasureValue, row_to_definition
from dispel.data.raw import MissingColumnError
from dispel.data.values import DefinitionIdType, ValueDefinition
from dispel.utils import convert_column_types, plural


class SubjectNotFound(Exception):
    """Class exception for not found subjects in measure collections."""

    def __init__(self, subject_id: str):
        message = f"{subject_id=} not found in measure collection."
        super().__init__(message)


class MeasureNotFound(Exception):
    """Class exception for not found measures in measure collections."""

    def __init__(self, measure_id: str, measures: Iterable[str]):
        top_3_closest_measures = nsmallest(
            3, measures, key=lambda x: damerau_levenshtein_distance(x, measure_id)
        )
        message = (
            f"{measure_id=} not found in measure collection. Did you mean any "
            f'of these: "{top_3_closest_measures}" ?'
        )
        super().__init__(message)


class MeasureCollection:
    """A measure collection from one or multiple readings.

    The measure collection structure provides a common object to handle basic
    transformations needed to perform analyses across multiple subjects and measures.
    The data is stored in a pandas data frame and can be retrieved by calling
    :attr:`data`. The returned data frame contains the measure values as well as some
    automatically computed properties, such as the *trail number*, reflecting the number
    of times a test was performed. A comprehensive list of properties can be found in
    the table below.

    +---------------------+------------------------------------------------------------+
    | Column              | Description                                                |
    +=====================+============================================================+
    | subject_id          | A unique identifier of the subject                         |
    +---------------------+------------------------------------------------------------+
    | evaluation_uuid     | A unique identifier of the evaluation                      |
    +---------------------+------------------------------------------------------------+
    | evaluation_code     | The code identifying the type of evaluation                |
    +---------------------+------------------------------------------------------------+
    | session_uuid        | A unique identifier of a session of multiple evaluations   |
    +---------------------+------------------------------------------------------------+
    | session_code        | The code identifying the type of session                   |
    +---------------------+------------------------------------------------------------+
    | start_date          | The start date and time of the evaluation                  |
    +---------------------+------------------------------------------------------------+
    | end_date            | The end date and time of the evaluation                    |
    +---------------------+------------------------------------------------------------+
    | is_finished         | If the evaluation was completed or not                     |
    +---------------------+------------------------------------------------------------+
    | algo_version        | Version of the analysis library                            |
    +---------------------+------------------------------------------------------------+
    | measure_id          | The id of the measure                                      |
    +---------------------+------------------------------------------------------------+
    | measure_name        | The human readable name of the measure                     |
    +---------------------+------------------------------------------------------------+
    | measure_value       | The actual measure value                                   |
    +---------------------+------------------------------------------------------------+
    | measure_unit        | The unit of the measure, if applicable                     |
    +---------------------+------------------------------------------------------------+
    | measure_type        | The numpy type of the value                                |
    +---------------------+------------------------------------------------------------+
    | trial               | The number of times the evaluation was performed by the    |
    |                     | subject                                                    |
    +---------------------+------------------------------------------------------------+
    | relative_start_date | The relative start date based on the first evaluation for  |
    |                     | each subject                                               |
    +---------------------+------------------------------------------------------------+

    The data frame might contain additional columns if the collection was constructed
    using :meth:`from_data_frame` and ``only_required_columns`` set to ``False``.
    """

    _REQUIRED_COLUMN_TYPES = {
        "subject_id": "U",
        "evaluation_uuid": "U",
        "evaluation_code": "U",
        "session_uuid": "U",
        "session_code": "U",
        "start_date": "datetime64[ms]",
        "end_date": "datetime64[ms]",
        "is_finished": "bool",
        "measure_id": "U",
        "measure_name": "U",
        "measure_value": "float64",
        "measure_unit": "U",
        "measure_type": "U",
    }

    _COLUMN_TYPES = {**_REQUIRED_COLUMN_TYPES, "trial": "int16"}

    def __init__(self):
        self._data = pd.DataFrame(columns=self._COLUMN_TYPES)
        self._measure_definitions: Dict[str, ValueDefinition] = {}

    def __repr__(self):
        return (
            f'<MeasureCollection: {plural("subject", self.subject_count)}, '
            f'{plural("evaluation", self.evaluation_count)}>'
        )

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            raise TypeError("Can only use operators between two MeasureCollections.")
        return self._data.equals(other.data)

    @staticmethod
    def _assert_add_type(other):
        if not isinstance(other, MeasureCollection):
            raise TypeError("Can only add measures from MeasureCollection")

    def __add__(self, other) -> "MeasureCollection":
        self._assert_add_type(other)
        fc = self.__class__()
        fc.extend(self)
        fc.extend(other)
        return fc

    def __iadd__(self, other) -> "MeasureCollection":
        self._assert_add_type(other)
        self.extend(other)
        return self

    @property
    def data(self) -> pd.DataFrame:
        """Get measure collection data frame."""
        return self._data.copy()

    @property
    def measure_definitions(self) -> ValuesView[ValueDefinition]:
        """Get measure definitions from measure collection."""
        return self._measure_definitions.values()

    @property
    def size(self) -> int:
        """Get size of measure collection data frame."""
        return self._data.size

    @property
    def evaluation_count(self) -> int:
        """Get the number of different evaluations."""
        return self._data.evaluation_uuid.nunique()

    @property
    def evaluation_ids(self) -> np.ndarray:
        """Get the evaluation ids in the measure collection."""
        return self._data.evaluation_uuid.unique()

    def get_evaluation_ids_for_subject(self, subject_id: str) -> List[str]:
        """Get evaluations related to a subject.

        Parameters
        ----------
        subject_id
            The subject identifier.

        Returns
        -------
        List[str]
            The list of evaluation ids.
        """
        mask = self._data["subject_id"] == subject_id
        return list(self._data[mask].evaluation_uuid.unique())

    @property
    def subject_count(self) -> int:
        """Get the number of different subjects."""
        return self._data.subject_id.nunique()

    @property
    def subject_ids(self) -> np.ndarray:
        """Get the subject ids in the measure collection."""
        return self._data.subject_id.unique()

    @property
    def session_count(self) -> int:
        """Get the number of different session."""
        return self._data.session_uuid.nunique()

    @property
    def session_ids(self) -> np.ndarray:
        """Get the session ids in the measure collection."""
        return self._data.session_uuid.unique()

    @property
    def measure_count(self) -> int:
        """Get the number of different measures."""
        return self._data.measure_id.nunique()

    @property
    def measure_ids(self) -> np.ndarray:
        """Get the measure ids in the measure collection."""
        return self._data.measure_id.unique()

    def get_measure_definition(self, measure_id: DefinitionIdType) -> ValueDefinition:
        """Get the measure definition for a specific measure id.

        Parameters
        ----------
        measure_id
            The measure identifier.

        Returns
        -------
        ValueDefinition
            The corresponding measure definition.

        Raises
        ------
        MeasureNotFound
            If the measure id does not correspond to any known measure
            definition.
        """
        if (id_ := str(measure_id)) not in self._measure_definitions:
            raise MeasureNotFound(id_, self._measure_definitions.keys())

        return self._measure_definitions[id_]

    @classmethod
    def from_measure_set(
        cls,
        measure_set: MeasureSet,
        evaluation: Evaluation,
        session: Session,
        _ignore_consistency: bool = False,
    ) -> "MeasureCollection":
        """Create a class instance from measure set.

        Parameters
        ----------
        measure_set
            The measure set whose measures are to be collected.
        evaluation
            The evaluation corresponding to the given measure set.
        session
            The session corresponding to the given evaluation.

        Returns
        -------
        MeasureCollection
            A measure collection containing all measures from the ``measure_set`` using
            the ``evaluation`` and ``session`` to complement the necessary information.
        """
        # pylint: disable=protected-access
        fc = cls()
        for value in measure_set.values():
            fc.append(
                cast(MeasureValue, value), evaluation, session, _ignore_consistency=True
            )
        if not _ignore_consistency:
            fc._ensure_consistency()
        return fc

    @classmethod
    def from_reading(
        cls, reading: Reading, _ignore_consistency: bool = False
    ) -> "MeasureCollection":
        """Create a class instance from reading.

        Parameters
        ----------
        reading
            The reading from which the measure collection is to be initialized.

        Returns
        -------
        MeasureCollection
            A measure collection containing all measures from the ``reading`` measure
            sets of each level. See also :meth:`from_measure_set`.

        Raises
        ------
        ValueError
            If the reading session information is not provided.
        """
        if reading.session is None:
            raise ValueError("Reading has no session information")
        return cls.from_measure_set(
            reading.get_merged_measure_set(),
            reading.evaluation,
            reading.session,
            _ignore_consistency=_ignore_consistency,
        )

    @classmethod
    def from_readings(cls, readings: Iterable[Reading]) -> "MeasureCollection":
        """Create a class instance from readings.

        Parameters
        ----------
        readings
            The readings from which the measure collection is to be initialized.

        Returns
        -------
        MeasureCollection
            A measure collection from all measure sets of all readings. See also
            :meth:`from_reading`.
        """
        # pylint: disable=protected-access
        fc = cls()
        for reading in readings:
            fc.extend(cls.from_reading(reading, _ignore_consistency=True))

        fc._ensure_consistency()
        return fc

    @classmethod
    def from_data_frame(
        cls, data: pd.DataFrame, only_required_columns: bool = False
    ) -> "MeasureCollection":
        """Create a class instance from a DataFrame.

        Parameters
        ----------
        data
            A data frame containing the information relative to measures. The data frame
            should contain the following columns (``subject_id`` or ``user_id``,
            ``evaluation_uuid``, ``evaluation_code``, ``session_uuid``,
            ``session_code``, ``start_date``, ``end_date``, ``is_finished``,
            ``measure_id``, ``measure_name``, ``measure_value``, ``measure_unit``,
            ``measure_type``).
        only_required_columns
            ``True`` if only the required columns are to be preserved in the measure
            collection. ``False`` otherwise.

        Returns
        -------
        MeasureCollection
            A measure collection from a pandas data frame.

        Raises
        ------
        ValueError
            If duplicate measures for same evaluations exist in the initializing data
            frame.
        MissingColumnError
            If required columns are missing from the data frame.
        """
        # pylint: disable=protected-access
        fc = cls()
        data_ = data.rename(
            {"user_id": "subject_id", "uuid_session": "session_uuid"}, axis=1
        )
        if data_.duplicated(["evaluation_uuid", "measure_id"]).any():
            raise ValueError("Duplicate measures exist for same evaluations.")

        input_columns = set(data_.columns)
        required_columns = set(fc._REQUIRED_COLUMN_TYPES)

        if not required_columns <= input_columns:
            raise MissingColumnError(required_columns - input_columns)

        data_ = convert_column_types(data_, lambda x: fc._COLUMN_TYPES[x])
        if only_required_columns:
            data_ = data_[fc._REQUIRED_COLUMN_TYPES]
        fc._data = data_

        definition_data = fc.data.drop_duplicates("measure_id")
        fc._merge_measure_definitions(definition_data.apply(row_to_definition, axis=1))

        fc._ensure_consistency()
        return fc

    @classmethod
    def from_csv(cls, path: str) -> "MeasureCollection":
        """Create a class instance from a csv file.

        Parameters
        ----------
        path
            The path to a csv file from which measures are to be collected.

        Returns
        -------
        MeasureCollection
            A measure collection from the CSV file specified in ``path``. See also
            :meth:`from_data_frame`.
        """
        return cls.from_data_frame(pd.read_csv(path))

    def _drop_nans(self):
        """Ensure NaN measure values are dropped and the user is warned."""
        if self._data["measure_value"].isnull().any():
            warnings.warn("Collection pruned of NaN measure values", UserWarning)

            self._data.dropna(subset=["measure_value"], inplace=True)

    def _sort_values(self):
        """Sort data frame by start date."""
        self._data.sort_values("start_date", inplace=True)

    def _drop_duplicates(self, overwrite: bool):
        """Drop measure collection duplicates.

        Parameters
        ----------
        overwrite
            ``True`` if recent measure information is to be replaced with existing one.
            ``False`` otherwise.
        """
        self._data.drop_duplicates(
            subset=["evaluation_uuid", "measure_id"],
            keep="last" if overwrite else "first",
            inplace=True,
            ignore_index=True,
        )

    def _update_trials(self):
        """Update trial count values for all subjects."""
        grouped = self._data.groupby(["subject_id", "measure_id"], sort=False)
        trial = (grouped.cumcount() + 1).astype(self._COLUMN_TYPES["trial"])
        self._data["trial"] = trial

    def _update_relative_start(self):
        """Update relative start date by measure for all subjects."""
        grp_idx = ["subject_id", "measure_id"]
        grouped = self._data.groupby(grp_idx, sort=False)
        first_start_date = grouped.start_date.min().rename("first_start_date")
        joined = self._data.join(first_start_date, on=grp_idx)
        relative = joined.start_date - joined.first_start_date
        self._data["relative_start_date"] = relative.dt.total_seconds() / 86400

    def _ensure_consistency(self, overwrite: bool = True):
        """Ensure consistency of measure collection data frame."""
        self._drop_nans()
        self._sort_values()
        self._drop_duplicates(overwrite=overwrite)
        self._update_trials()
        self._update_relative_start()

    def _add_measure_definition(self, definition: ValueDefinition):
        """Add a measure definition to the measure collection.

        Parameters
        ----------
        definition
            The measure value definition to be added.
        """
        self._measure_definitions[str(definition.id)] = definition

    def _merge_measure_definitions(
        self, definitions: Iterable[ValueDefinition], overwrite: bool = True
    ):
        """Merge measure definitions.

        Parameters
        ----------
        definitions
            The measure value definitions to be merged.
        overwrite
            ``True`` If the measure value definitions are to be overwritten.
            ``False`` otherwise.
        """
        if overwrite:
            for definition in definitions:
                self._add_measure_definition(definition)

    def append(
        self,
        measure_value: MeasureValue,
        evaluation: Evaluation,
        session: Session,
        _ignore_consistency: bool = False,
    ):
        """Adding measure value to the measure collection.

        Parameters
        ----------
        measure_value
            The measure value to be added to the collection.
        evaluation
            The evaluation corresponding to the given measure value.
        session
            The session corresponding to the given evaluation.
        _ignore_consistency
            If ``True``, methods for ensuring consistency of the data will be skipped.
        """
        meta_data = dict(
            subject_id=evaluation.user_id,
            evaluation_uuid=evaluation.uuid,
            evaluation_code=evaluation.id,
            session_code=session.id,
            session_uuid=session.uuid,
            trial=0,
            start_date=evaluation.start,
            end_date=evaluation.end,
            is_finished=evaluation.finished,
            algo_version=__version__,
        )

        # Add measure value information to the pandas data frame
        self._data = self._data.append(
            {**meta_data, **measure_value.to_dict()}, ignore_index=True
        )
        # Add measure definition
        measure_id = str(measure_value.id)
        self._measure_definitions[measure_id] = measure_value.definition

        if not _ignore_consistency:
            self._ensure_consistency(overwrite=True)

    def extend(
        self,
        other: "MeasureCollection",
        overwrite: bool = True,
        _ignore_consistency: bool = False,
    ):
        """Extend measure collection by another.

        Parameters
        ----------
        other
            The object with which the measure collection is to be expanded.
        overwrite
            ``True`` if new measure information is to be replaced with existing one.
            ``False`` otherwise.
        _ignore_consistency
            If ``True``, methods for ensuring consistency of the data will be skipped.

        Raises
        ------
        TypeError
            If the type of the object to be extended is not a measure
            collection.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Unsupported extender type: {type(other)}. Measure collection "
                "expansion can only support another MeasureCollection."
            )

        # Adding measure collection data frame.
        self._data = self._data.append(other.data)

        # Merging measure definitions
        self._merge_measure_definitions(other.measure_definitions)

        if not _ignore_consistency:
            self._ensure_consistency(overwrite=overwrite)

    def get_data(
        self, measure_id: Optional[str] = None, subject_id: Optional[str] = None
    ) -> Union[pd.DataFrame, pd.Series]:
        """Retrieve data from measure collection.

        Parameters
        ----------
        measure_id
            The identifier of the measure for which the data is being retrieved.
        subject_id
            The identifier of the subject for which the data is being retrieved.

        Returns
        -------
        pandas.DataFrame
            A pandas data frame filtered w.r.t. the given arguments.
        """
        measure_id_dic = {"measure_id": measure_id}
        subject_id_dic = {"subject_id": subject_id}

        def _assert_existence(args):
            for name, value in args.items():
                if name == "subject_id":
                    if value not in self.subject_ids:
                        raise SubjectNotFound(subject_id)
                elif name == "measure_id":
                    if value not in self.measure_ids:
                        raise MeasureNotFound(measure_id, self.measure_ids)
                else:
                    raise ValueError("Unsupported type value.")

        if subject_id is None and measure_id is None:
            return self.data
        if subject_id is None:
            _assert_existence(measure_id_dic)
            mask = self._data["measure_id"] == measure_id
        elif measure_id is None:
            _assert_existence(subject_id_dic)
            mask = self._data["subject_id"] == subject_id
        else:
            _assert_existence({**measure_id_dic, **subject_id_dic})
            mask = (self._data["subject_id"] == subject_id) & (
                self._data["measure_id"] == measure_id
            )
        return self._data[mask].copy()

    def get_measure_values_over_time(
        self,
        measure_id: str,
        subject_id: str,
        index: Union[str, List[str]] = "start_date",
    ) -> pd.Series:
        """Retrieve data as time indexed measure value series.

        Parameters
        ----------
        measure_id
            The identifier of the measure for which the data is being retrieved.
        subject_id
            The identifier of the subject for which the data is being retrieved.
        index
            The index of the measure values pandas series.

        Returns
        -------
        pandas.Series
            A pandas series with start date as index and measure values as values.
        """
        data = self.get_data(subject_id=subject_id, measure_id=measure_id)

        return data.set_index(index)["measure_value"].rename(measure_id)

    def get_measure_values_by_trials(self, measure_id: str) -> pd.DataFrame:
        """Retrieve measure values over all trials by subject.

        Parameters
        ----------
        measure_id
            The identifier of the measure for which the data is being retrieved.

        Returns
        -------
        pandas.DataFrame
            A pandas data frame with subjects as indexes, trials as columns and
            measure values as values.
        """
        data = self.get_data(measure_id=measure_id)
        return data.pivot("subject_id", "trial", "measure_value")

    def get_aggregated_measures_over_period(
        self, measure_id: str, period: str, aggregation: Union[str, Callable]
    ) -> pd.DataFrame:
        """Get aggregated measure values over a given period.

        Parameters
        ----------
        measure_id
            The identifier of the measure for which the data is being computed.
        period
            The period on which the measure is to be aggregated.
        aggregation
            The aggregation method to be used.

        Returns
        -------
        pandas.DataFrame
            A pandas data frame regrouping aggregated measure values over a given
            period. The resulting data frame contains subjects as rows, aggregation
            periods as columns, and values based on the provided aggregation method.
        """
        data = self.get_data(measure_id=measure_id)
        grp = ["subject_id", pd.Grouper(key="start_date", freq=period)]
        return data.groupby(grp).measure_value.agg(aggregation).unstack()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the measure collection to a dictionary."""
        return self._data.to_dict()

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """Convert the measure collection to a JSON string.

        Parameters
        ----------
        path
            File path or object. If not specified, the result is returned as a string.

        Returns
        -------
        Optional[str]
            If ``path`` is ``None``, returns the resulting json format as a string.
            Otherwise, returns ``None``.
        """
        return self._data.to_json(path)

    def to_csv(self, path: Optional[str] = None):
        """Write object to a comma-separated values (csv) file.

        Parameters
        ----------
        path
            File path or object, if ``None`` is provided the result is returned as a
            string. If a file object is passed it should be opened with newline=’’,
            disabling universal newlines.

        Returns
        -------
        Optional[str]
            If ``path`` is ``None``, returns the resulting csv format as a string.
            Otherwise, returns ``None``.
        """
        return self._data.to_csv(path)
