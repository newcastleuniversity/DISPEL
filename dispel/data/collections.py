"""A module for collections of feature values."""
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
from dispel.data.features import FeatureSet, FeatureValue, row_to_definition
from dispel.data.raw import MissingColumnError
from dispel.data.values import DefinitionIdType, ValueDefinition
from dispel.utils import convert_column_types, plural


class SubjectNotFound(Exception):
    """Class exception for not found subjects in feature collections."""

    def __init__(self, subject_id: str):
        message = f"{subject_id=} not found in feature collection."
        super().__init__(message)


class FeatureNotFound(Exception):
    """Class exception for not found features in feature collections."""

    def __init__(self, feature_id: str, features: Iterable[str]):
        top_3_closest_features = nsmallest(
            3, features, key=lambda x: damerau_levenshtein_distance(x, feature_id)
        )
        message = (
            f"{feature_id=} not found in feature collection. Did you mean any "
            f'of these: "{top_3_closest_features}" ?'
        )
        super().__init__(message)


class FeatureCollection:
    """A feature collection from one or multiple readings.

    The feature collection structure provides a common object to handle basic
    transformations needed to perform analyses across multiple subjects and features.
    The data is stored in a pandas data frame and can be retrieved by calling
    :attr:`data`. The returned data frame contains the feature values as well as some
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
    | feature_id          | The id of the feature                                      |
    +---------------------+------------------------------------------------------------+
    | feature_name        | The human readable name of the feature                     |
    +---------------------+------------------------------------------------------------+
    | feature_value       | The actual feature value                                   |
    +---------------------+------------------------------------------------------------+
    | feature_unit        | The unit of the feature, if applicable                     |
    +---------------------+------------------------------------------------------------+
    | feature_type        | The numpy type of the value                                |
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
        "feature_id": "U",
        "feature_name": "U",
        "feature_value": "float64",
        "feature_unit": "U",
        "feature_type": "U",
    }

    _COLUMN_TYPES = {**_REQUIRED_COLUMN_TYPES, "trial": "int16"}

    def __init__(self):
        self._data = pd.DataFrame(columns=self._COLUMN_TYPES)
        self._feature_definitions: Dict[str, ValueDefinition] = {}

    def __repr__(self):
        return (
            f'<FeatureCollection: {plural("subject", self.subject_count)}, '
            f'{plural("evaluation", self.evaluation_count)}>'
        )

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            raise TypeError("Can only use operators between two FeatureCollections.")
        return self._data.equals(other.data)

    @staticmethod
    def _assert_add_type(other):
        if not isinstance(other, FeatureCollection):
            raise TypeError("Can only add features from FeatureCollection")

    def __add__(self, other) -> "FeatureCollection":
        self._assert_add_type(other)
        fc = self.__class__()
        fc.extend(self)
        fc.extend(other)
        return fc

    def __iadd__(self, other) -> "FeatureCollection":
        self._assert_add_type(other)
        self.extend(other)
        return self

    @property
    def data(self) -> pd.DataFrame:
        """Get feature collection data frame."""
        return self._data.copy()

    @property
    def feature_definitions(self) -> ValuesView[ValueDefinition]:
        """Get feature definitions from feature collection."""
        return self._feature_definitions.values()

    @property
    def size(self) -> int:
        """Get size of feature collection data frame."""
        return self._data.size

    @property
    def evaluation_count(self) -> int:
        """Get the number of different evaluations."""
        return self._data.evaluation_uuid.nunique()

    @property
    def evaluation_ids(self) -> np.ndarray:
        """Get the evaluation ids in the feature collection."""
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
        """Get the subject ids in the feature collection."""
        return self._data.subject_id.unique()

    @property
    def session_count(self) -> int:
        """Get the number of different session."""
        return self._data.session_uuid.nunique()

    @property
    def session_ids(self) -> np.ndarray:
        """Get the session ids in the feature collection."""
        return self._data.session_uuid.unique()

    @property
    def feature_count(self) -> int:
        """Get the number of different features."""
        return self._data.feature_id.nunique()

    @property
    def feature_ids(self) -> np.ndarray:
        """Get the feature ids in the feature collection."""
        return self._data.feature_id.unique()

    def get_feature_definition(self, feature_id: DefinitionIdType) -> ValueDefinition:
        """Get the feature definition for a specific feature id.

        Parameters
        ----------
        feature_id
            The feature identifier.

        Returns
        -------
        ValueDefinition
            The corresponding feature definition.

        Raises
        ------
        FeatureNotFound
            If the feature id does not correspond to any known feature
            definition.
        """
        if (id_ := str(feature_id)) not in self._feature_definitions:
            raise FeatureNotFound(id_, self._feature_definitions.keys())

        return self._feature_definitions[id_]

    @classmethod
    def from_feature_set(
        cls,
        feature_set: FeatureSet,
        evaluation: Evaluation,
        session: Session,
        _ignore_consistency: bool = False,
    ) -> "FeatureCollection":
        """Create a class instance from feature set.

        Parameters
        ----------
        feature_set
            The feature set whose features are to be collected.
        evaluation
            The evaluation corresponding to the given feature set.
        session
            The session corresponding to the given evaluation.

        Returns
        -------
        FeatureCollection
            A feature collection containing all features from the ``feature_set`` using
            the ``evaluation`` and ``session`` to complement the necessary information.
        """
        # pylint: disable=protected-access
        fc = cls()
        for value in feature_set.values():
            fc.append(
                cast(FeatureValue, value), evaluation, session, _ignore_consistency=True
            )
        if not _ignore_consistency:
            fc._ensure_consistency()
        return fc

    @classmethod
    def from_reading(
        cls, reading: Reading, _ignore_consistency: bool = False
    ) -> "FeatureCollection":
        """Create a class instance from reading.

        Parameters
        ----------
        reading
            The reading from which the feature collection is to be initialized.

        Returns
        -------
        FeatureCollection
            A feature collection containing all features from the ``reading`` feature
            sets of each level. See also :meth:`from_feature_set`.

        Raises
        ------
        ValueError
            If the reading session information is not provided.
        """
        if reading.session is None:
            raise ValueError("Reading has no session information")
        return cls.from_feature_set(
            reading.get_merged_feature_set(),
            reading.evaluation,
            reading.session,
            _ignore_consistency=_ignore_consistency,
        )

    @classmethod
    def from_readings(cls, readings: Iterable[Reading]) -> "FeatureCollection":
        """Create a class instance from readings.

        Parameters
        ----------
        readings
            The readings from which the feature collection is to be initialized.

        Returns
        -------
        FeatureCollection
            A feature collection from all feature sets of all readings. See also
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
    ) -> "FeatureCollection":
        """Create a class instance from a DataFrame.

        Parameters
        ----------
        data
            A data frame containing the information relative to features. The data frame
            should contain the following columns (``subject_id`` or ``user_id``,
            ``evaluation_uuid``, ``evaluation_code``, ``session_uuid``,
            ``session_code``, ``start_date``, ``end_date``, ``is_finished``,
            ``feature_id``, ``feature_name``, ``feature_value``, ``feature_unit``,
            ``feature_type``).
        only_required_columns
            ``True`` if only the required columns are to be preserved in the feature
            collection. ``False`` otherwise.

        Returns
        -------
        FeatureCollection
            A feature collection from a pandas data frame.

        Raises
        ------
        ValueError
            If duplicate features for same evaluations exist in the initializing data
            frame.
        MissingColumnError
            If required columns are missing from the data frame.
        """
        # pylint: disable=protected-access
        fc = cls()
        data_ = data.rename(
            {"user_id": "subject_id", "uuid_session": "session_uuid"}, axis=1
        )
        if data_.duplicated(["evaluation_uuid", "feature_id"]).any():
            raise ValueError("Duplicate features exist for same evaluations.")

        input_columns = set(data_.columns)
        required_columns = set(fc._REQUIRED_COLUMN_TYPES)

        if not required_columns <= input_columns:
            raise MissingColumnError(required_columns - input_columns)

        data_ = convert_column_types(data_, lambda x: fc._COLUMN_TYPES[x])
        if only_required_columns:
            data_ = data_[fc._REQUIRED_COLUMN_TYPES]
        fc._data = data_

        definition_data = fc.data.drop_duplicates("feature_id")
        fc._merge_feature_definitions(definition_data.apply(row_to_definition, axis=1))

        fc._ensure_consistency()
        return fc

    @classmethod
    def from_csv(cls, path: str) -> "FeatureCollection":
        """Create a class instance from a csv file.

        Parameters
        ----------
        path
            The path to a csv file from which features are to be collected.

        Returns
        -------
        FeatureCollection
            A feature collection from the CSV file specified in ``path``. See also
            :meth:`from_data_frame`.
        """
        return cls.from_data_frame(pd.read_csv(path))

    def _drop_nans(self):
        """Ensure NaN feature values are dropped and the user is warned."""
        if self._data["feature_value"].isnull().any():
            warnings.warn("Collection pruned of NaN feature values", UserWarning)

            self._data.dropna(subset=["feature_value"], inplace=True)

    def _sort_values(self):
        """Sort data frame by start date."""
        self._data.sort_values("start_date", inplace=True)

    def _drop_duplicates(self, overwrite: bool):
        """Drop feature collection duplicates.

        Parameters
        ----------
        overwrite
            ``True`` if recent feature information is to be replaced with existing one.
            ``False`` otherwise.
        """
        self._data.drop_duplicates(
            subset=["evaluation_uuid", "feature_id"],
            keep="last" if overwrite else "first",
            inplace=True,
            ignore_index=True,
        )

    def _update_trials(self):
        """Update trial count values for all subjects."""
        grouped = self._data.groupby(["subject_id", "feature_id"], sort=False)
        trial = (grouped.cumcount() + 1).astype(self._COLUMN_TYPES["trial"])
        self._data["trial"] = trial

    def _update_relative_start(self):
        """Update relative start date by feature for all subjects."""
        grp_idx = ["subject_id", "feature_id"]
        grouped = self._data.groupby(grp_idx, sort=False)
        first_start_date = grouped.start_date.min().rename("first_start_date")
        joined = self._data.join(first_start_date, on=grp_idx)
        relative = joined.start_date - joined.first_start_date
        self._data["relative_start_date"] = relative.dt.total_seconds() / 86400

    def _ensure_consistency(self, overwrite: bool = True):
        """Ensure consistency of feature collection data frame."""
        self._drop_nans()
        self._sort_values()
        self._drop_duplicates(overwrite=overwrite)
        self._update_trials()
        self._update_relative_start()

    def _add_feature_definition(self, definition: ValueDefinition):
        """Add a feature definition to the feature collection.

        Parameters
        ----------
        definition
            The feature value definition to be added.
        """
        self._feature_definitions[str(definition.id)] = definition

    def _merge_feature_definitions(
        self, definitions: Iterable[ValueDefinition], overwrite: bool = True
    ):
        """Merge feature definitions.

        Parameters
        ----------
        definitions
            The feature value definitions to be merged.
        overwrite
            ``True`` If the feature value definitions are to be overwritten.
            ``False`` otherwise.
        """
        if overwrite:
            for definition in definitions:
                self._add_feature_definition(definition)

    def append(
        self,
        feature_value: FeatureValue,
        evaluation: Evaluation,
        session: Session,
        _ignore_consistency: bool = False,
    ):
        """Adding feature value to the feature collection.

        Parameters
        ----------
        feature_value
            The feature value to be added to the collection.
        evaluation
            The evaluation corresponding to the given feature value.
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

        # Add feature value information to the pandas data frame
        self._data = self._data.append(
            {**meta_data, **feature_value.to_dict()}, ignore_index=True
        )
        # Add feature definition
        feature_id = str(feature_value.id)
        self._feature_definitions[feature_id] = feature_value.definition

        if not _ignore_consistency:
            self._ensure_consistency(overwrite=True)

    def extend(
        self,
        other: "FeatureCollection",
        overwrite: bool = True,
        _ignore_consistency: bool = False,
    ):
        """Extend feature collection by another.

        Parameters
        ----------
        other
            The object with which the feature collection is to be expanded.
        overwrite
            ``True`` if new feature information is to be replaced with existing one.
            ``False`` otherwise.
        _ignore_consistency
            If ``True``, methods for ensuring consistency of the data will be skipped.

        Raises
        ------
        TypeError
            If the type of the object to be extended is not a feature
            collection.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Unsupported extender type: {type(other)}. Feature collection "
                "expansion can only support another FeatureCollection."
            )

        # Adding feature collection data frame.
        self._data = self._data.append(other.data)

        # Merging feature definitions
        self._merge_feature_definitions(other.feature_definitions)

        if not _ignore_consistency:
            self._ensure_consistency(overwrite=overwrite)

    def get_data(
        self, feature_id: Optional[str] = None, subject_id: Optional[str] = None
    ) -> Union[pd.DataFrame, pd.Series]:
        """Retrieve data from feature collection.

        Parameters
        ----------
        feature_id
            The identifier of the feature for which the data is being retrieved.
        subject_id
            The identifier of the subject for which the data is being retrieved.

        Returns
        -------
        pandas.DataFrame
            A pandas data frame filtered w.r.t. the given arguments.
        """
        feature_id_dic = {"feature_id": feature_id}
        subject_id_dic = {"subject_id": subject_id}

        def _assert_existence(args):
            for name, value in args.items():
                if name == "subject_id":
                    if value not in self.subject_ids:
                        raise SubjectNotFound(subject_id)
                elif name == "feature_id":
                    if value not in self.feature_ids:
                        raise FeatureNotFound(feature_id, self.feature_ids)
                else:
                    raise ValueError("Unsupported type value.")

        if subject_id is None and feature_id is None:
            return self.data
        if subject_id is None:
            _assert_existence(feature_id_dic)
            mask = self._data["feature_id"] == feature_id
        elif feature_id is None:
            _assert_existence(subject_id_dic)
            mask = self._data["subject_id"] == subject_id
        else:
            _assert_existence({**feature_id_dic, **subject_id_dic})
            mask = (self._data["subject_id"] == subject_id) & (
                self._data["feature_id"] == feature_id
            )
        return self._data[mask].copy()

    def get_feature_values_over_time(
        self,
        feature_id: str,
        subject_id: str,
        index: Union[str, List[str]] = "start_date",
    ) -> pd.Series:
        """Retrieve data as time indexed feature value series.

        Parameters
        ----------
        feature_id
            The identifier of the feature for which the data is being retrieved.
        subject_id
            The identifier of the subject for which the data is being retrieved.
        index
            The index of the feature values pandas series.

        Returns
        -------
        pandas.Series
            A pandas series with start date as index and feature values as values.
        """
        data = self.get_data(subject_id=subject_id, feature_id=feature_id)

        return data.set_index(index)["feature_value"].rename(feature_id)

    def get_feature_values_by_trials(self, feature_id: str) -> pd.DataFrame:
        """Retrieve feature values over all trials by subject.

        Parameters
        ----------
        feature_id
            The identifier of the feature for which the data is being retrieved.

        Returns
        -------
        pandas.DataFrame
            A pandas data frame with subjects as indexes, trials as columns and
            feature values as values.
        """
        data = self.get_data(feature_id=feature_id)
        return data.pivot("subject_id", "trial", "feature_value")

    def get_aggregated_features_over_period(
        self, feature_id: str, period: str, aggregation: Union[str, Callable]
    ) -> pd.DataFrame:
        """Get aggregated feature values over a given period.

        Parameters
        ----------
        feature_id
            The identifier of the feature for which the data is being computed.
        period
            The period on which the feature is to be aggregated.
        aggregation
            The aggregation method to be used.

        Returns
        -------
        pandas.DataFrame
            A pandas data frame regrouping aggregated feature values over a given
            period. The resulting data frame contains subjects as rows, aggregation
            periods as columns, and values based on the provided aggregation method.
        """
        data = self.get_data(feature_id=feature_id)
        grp = ["subject_id", pd.Grouper(key="start_date", freq=period)]
        return data.groupby(grp).feature_value.agg(aggregation).unstack()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the feature collection to a dictionary."""
        return self._data.to_dict()

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """Convert the feature collection to a JSON string.

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
