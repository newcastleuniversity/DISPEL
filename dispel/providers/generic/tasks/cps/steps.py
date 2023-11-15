"""Cognitive Processing Speed test related functionality.

This module contains functionality to extract features for the *Cognitive
Processing Speed* test.
"""
# pylint: disable=no-member
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from dispel.data.features import FeatureValueDefinition, FeatureValueDefinitionPrototype
from dispel.data.flags import WrappedResult
from dispel.data.levels import Level
from dispel.data.raw import RawDataValueDefinition
from dispel.data.validators import GREATER_THAN_ZERO, RangeValidator
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import ValueDefinition
from dispel.processing.assertions import NotEmptyDataSetAssertionMixin
from dispel.processing.data_set import (
    TransformationFunctionGeneratorType,
    transformation,
)
from dispel.processing.extract import (
    DEFAULT_AGGREGATIONS,
    EXTENDED_AGGREGATIONS,
    AggregateFeatures,
    AggregateModalities,
    AggregateRawDataSetColumn,
    ExtractStep,
)
from dispel.processing.level import LevelFilter, ProcessingStepGroup
from dispel.processing.transform import TransformStep
from dispel.providers.generic.tasks.cps.modalities import (
    INTERACT_DURATION_VALIDATOR,
    NBACKS,
    TOTAL_RESPONSES_VALIDATOR,
    CPSKeySet,
    CPSModalityFilter,
    CPSMode,
    CPSSequence,
    DigitConfusionPairModality,
    DigitEnum,
    KeyType,
    PairType,
    RegressionMode,
    ResponsesModality,
    SymbolConfusionPairModality,
    SymbolEnum,
    SymmetryModality,
    ThirdsModality,
    ThirdsPairModality,
)
from dispel.providers.generic.tasks.cps.utils import (
    AV_REACTION_TIME,
    CPS_AGGREGATION_LIST,
    CPS_BASIC_AGGREGATION,
    CPS_EXTENDED_AGGREGATION,
    CPS_SYMBOL_SPECIFIC_AGGREGATION,
    DTD_RT_MEAN,
    EXTRA_MODALITY_LIST,
    STD_KEY_FIXED1_RT_MEAN,
    STD_KEY_FIXED2_RT_MEAN,
    STD_KEY_RANDOM_RT_MEAN,
    TASK_NAME,
    _compute_substitution_time,
    agg_reaction_time,
    compute_confusion_error_rate,
    compute_confusion_matrix,
    compute_correct_third_from_paired,
    compute_response_time_linear_regression,
    compute_streak,
    get_third_data,
    study2and3back,
    transform_user_input,
)
from dispel.utils import drop_none


class SequenceTransformMixin(metaclass=ABCMeta):
    """A Mix in Class for Sequence parameters.

    Parameters
    ----------
    Sequence
        The sequence type
    """

    def __init__(self, *args, **kwargs):
        self.sequence: CPSSequence = kwargs.pop("sequence")
        self.definition = self.get_prototype_definition()
        super().__init__(*args, **kwargs)  # type: ignore

    @abstractmethod
    def get_prototype_definition(self) -> FeatureValueDefinitionPrototype:
        """Get the feature value definition."""
        raise NotImplementedError


class SequenceModeKeyModalityDefinitionMixIn:
    """A Mix in Class for Sequence, mode and key parameters.

    Parameters
    ----------
    mode
        The desired mode to compute the transformation
        (``digit-to-symbol``, ``digit-to-digit``).
    key_set
        The specific key set filter on which you desire to extract features.
    sequence
        The sequence type
    """

    feature_name: AV
    description: str
    unit: Optional[str] = None
    data_type: Optional[str] = None
    validator: Optional[Callable[[Any], None]] = None

    def __init__(self, *args, **kwargs):
        self.mode: CPSMode = kwargs.pop("mode")
        self.sequence: CPSSequence = kwargs.pop("sequence")
        self.key_set: Optional[CPSKeySet] = None
        self.key_set: Optional[CPSKeySet] = kwargs.pop("key_set", None)

        self.modality = drop_none(
            [self.mode.av, self.sequence.av, self.key_set.av if self.key_set else None]
        )

        self.definition = self.get_prototype_definition()
        self.transform_functions = self.get_function()

        super().__init__(*args, **kwargs)  # type: ignore

    def get_prototype_definition(self) -> FeatureValueDefinitionPrototype:
        """Get the feature value definition."""
        return FeatureValueDefinitionPrototype(
            feature_name=self.feature_name,
            data_type=self.data_type,
            unit=self.unit,
            validator=self.validator,
            description=f"{self.description} for the {{mode}} with "
            "{sequence}"
            f'{" using {key_set}" if self.key_set else ""}.',
        )

    def add_modality(self, modality: AV):
        """Add additional modality."""
        return self.modality + [modality]

    def get_function(self) -> Iterable[Dict[str, Any]]:
        """Get iterable of transform function."""
        return []


class TransformUserInputBase(TransformStep):
    """Transformation step based on keys-analysis."""

    data_set_ids = "userInput"


class TransformStreakData(NotEmptyDataSetAssertionMixin, TransformUserInputBase):
    """Extract the longest streak of incorrect/correct responses."""

    new_data_set_id = "streak-data"
    definitions = [
        RawDataValueDefinition("correct", "longest streak of correct responses."),
        RawDataValueDefinition("incorrect", "longest streak of incorrect responses."),
    ]

    @transformation
    def apply(self, data):
        """Get transform function."""
        correct_streak, incorrect_streak = compute_streak(data)
        return pd.DataFrame(
            {"correct": correct_streak, "incorrect": incorrect_streak}, index=["streak"]
        )


class TransformKeysAnalysisData(NotEmptyDataSetAssertionMixin, TransformUserInputBase):
    """
    Create a data frame of reaction time per symbol or digit association.

    The data frame has three columns, the reaction time to press a key when
    one is displayed, a column expected with the displayed key, and a column
    actual with the key pressed by the user.
    """

    new_data_set_id = "keys-analysis"
    transform_function = transform_user_input
    definitions = [
        RawDataValueDefinition("expected", "expected item."),
        RawDataValueDefinition("actual", "actual item given."),
        RawDataValueDefinition(
            "mismatching", "The error between expected and actual items."
        ),
        RawDataValueDefinition("reactionTime", "reaction time.", unit="s"),
        RawDataValueDefinition("tsAnswer", "timestamp answer"),
    ]


class TransformKeyAnalysisBase(TransformStep):
    """Transformation step based on keys-analysis."""

    data_set_ids = "keys-analysis"


class TransformConfusion(TransformKeyAnalysisBase):
    """
    Create a confusion matrix between pressed and displayed symbols or digits.

    The confusion matrix is either on the symbols or digits CPS keys depending
    on the current processed level.
    """

    new_data_set_id = "confusion-matrix"
    transform_function = compute_confusion_matrix
    definitions = [
        RawDataValueDefinition(
            symbol.value, f"confusion for {symbol.value}", "float64"  # type: ignore
        )
        for symbol in SymbolEnum
    ]


class TransformNBacks(TransformKeyAnalysisBase):
    """
    A transform step to extract data frame containing n-backs information.

    Extract 1Back, 2Back and 3Back reaction time for correct responses
    to capture the working memory capacity of participants.
    """

    new_data_set_id = "n-backs"
    transform_function = study2and3back
    definitions = [
        RawDataValueDefinition(
            f"rt{pos.title()}{back}", f"reaction time for {back}-back {pos}", "float64"
        )
        for back in NBACKS
        for pos in ("back", "current")
    ]


class ExtractCPSSteps(ExtractStep):
    """CPS multiple steps extraction.

    Attributes
    ----------
    transform_functions
            An optional list of dictionaries containing at least the processing
            function under the key ``func``
    """

    transform_functions: Iterable[Dict[str, Any]]

    def get_transform_functions(self) -> TransformationFunctionGeneratorType:
        """Get the transform functions applied to the data sets."""
        yield from super().get_transform_functions()

        for function_spec in self.transform_functions:
            spec = function_spec.copy()
            yield spec.pop("func"), spec


class ConfusionBase(
    SequenceModeKeyModalityDefinitionMixIn, ExtractCPSSteps, metaclass=ABCMeta
):
    """Confusion extract multiple steps Base.

    Attributes
    ----------
    confusion_pair
        List of the most likely to be confused pair.
    """

    data_set_ids = "confusion-matrix"
    data_type = "float64"

    confusion_pair: List[PairType]

    @abstractmethod
    def apply_pair(self, data: pd.DataFrame, pair: PairType) -> float:
        """Get the feature value definition."""
        raise NotImplementedError

    def function_factory(self, pair: PairType) -> dict:
        """Get function factory."""
        return dict(
            func=lambda data: self.apply_pair(data, pair),
            modalities=self.add_modality(pair.av),
            subset=(pair.left, pair.right),
        )

    def get_function(self) -> List[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [self.function_factory(pair) for pair in self.confusion_pair]


class ExtractConfusionBase(ConfusionBase):
    """Confusion features extraction mix in."""

    feature_name = AV("confusion", "conf")
    validator = GREATER_THAN_ZERO
    description = (
        "The total number of time {subset[0]} was provided instead of {subset[1]} "
    )

    def apply_pair(self, data, pair: PairType):
        """Get apply function for ExtractSteps."""
        return data[pair.left.value][pair.right.value]


class ExtractConfusionRateBase(ConfusionBase):
    """Confusion rate features extraction mix in."""

    feature_name = AV("confusion rate", "confr")
    validator = GREATER_THAN_ZERO
    description = "The confusion rate between {subset} "

    def apply_pair(self, data, pair: PairType):
        """Get apply function for ExtractSteps."""
        return compute_confusion_error_rate(data, pair.left.value, pair.right.value)


class ExtractConfusionDTD(ExtractConfusionBase):
    """Confusion features extraction."""

    confusion_pair = list(DigitConfusionPairModality)


class ExtractConfusionDTS(ExtractConfusionBase):
    """Confusion features extraction."""

    confusion_pair = list(SymbolConfusionPairModality)


class ExtractConfusionRateDTD(ExtractConfusionRateBase):
    """Confusion features extraction for digit."""

    confusion_pair = list(DigitConfusionPairModality)


class ExtractConfusionRateDTS(ExtractConfusionRateBase):
    """Confusion features extraction for digit."""

    confusion_pair = list(SymbolConfusionPairModality)


class KeyAnalysisBase(
    SequenceModeKeyModalityDefinitionMixIn, ExtractCPSSteps, metaclass=ABCMeta
):
    """Keys Analysis Multiple Extract Steps mix in.

    Attributes
    ----------
    key_list
        List of symbol or digits on which to apply analysis
    """

    data_set_ids = "keys-analysis"
    data_type = "float64"
    validator = INTERACT_DURATION_VALIDATOR

    key_list: List[KeyType]
    aggregation_list: List[Tuple[str, str]] = CPS_BASIC_AGGREGATION

    @abstractmethod
    def apply_key(self, data: pd.DataFrame, key: Any, agg: AV) -> WrappedResult[float]:
        """Get the feature value definition."""
        raise NotImplementedError

    def function_factory(self, key: KeyType, agg: AV) -> dict:
        """Get function factory."""
        return dict(
            func=lambda data: self.apply_key(data, key, agg),
            aggregation=agg,
            modalities=self.add_modality(key.av),
            key=key,
            unit="s",
        )

    def get_function(self) -> List[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [
            self.function_factory(key, AV(agg[1], agg[0]))
            for agg in self.aggregation_list
            for key in self.key_list
        ]


class ExtractDifferencesKeyReactionTimeBase(KeyAnalysisBase):
    """Difference Reaction Time ExtractStep mix in."""

    feature_name = AV("reaction time difference", "rt_diff")
    description = "The {aggregation} absolute reaction time difference between {key} "

    def apply_key(self, data: pd.DataFrame, key: Any, agg: AV) -> WrappedResult[float]:
        """Get apply function for ExtractSteps."""
        left_result: WrappedResult = agg_reaction_time(data, agg, key.left)
        right_result: WrappedResult = agg_reaction_time(data, agg, key.right)
        return abs(left_result - right_result)


class KeyReactionTimeBase(KeyAnalysisBase):
    """Reaction Time ExtractStep mix in."""

    feature_name = AV_REACTION_TIME
    description = "The {aggregation} reaction time for {key} "

    def apply_key(self, data: pd.DataFrame, key: Any, agg: AV) -> WrappedResult[float]:
        """Get apply function for ExtractSteps."""
        return agg_reaction_time(data, agg, key)


class ExtractDigitSpecificReactionTimesDTD(KeyReactionTimeBase):
    """Digit to Digit Reaction Time ExtractStep.

    The digits 1, 6 and 9 are the most likely to be confused, and they
    will be used to compute reaction time features.
    """

    key_list = [DigitEnum.DIGIT_1, DigitEnum.DIGIT_6, DigitEnum.DIGIT_9]


class ExtractDigitSpecificReactionTimesSTD(KeyReactionTimeBase):
    """Symbol to Digit Reaction Time ExtractStep."""

    aggregation_list = CPS_SYMBOL_SPECIFIC_AGGREGATION
    key_list = list(SymbolEnum)


class ExtractDigitSymmetryPairedReactionTimes(KeyReactionTimeBase):
    """Pair symmetry Reaction Time ExtractStep."""

    key_list = [SymmetryModality.PAIRED]

    def apply_key(
        self, data: pd.DataFrame, key: SymmetryModality, agg: AV
    ) -> WrappedResult[float]:
        """Get apply function for ExtractSteps."""
        return agg_reaction_time(data, agg, key.get_pair_modality(self.mode))


class ExtractDigitSymmetryUniqueReactionTimes(KeyReactionTimeBase):
    """Unique symmetry Reaction Time ExtractStep."""

    key_list = [SymmetryModality.UNIQUE]

    def apply_key(
        self, data: pd.DataFrame, key: SymmetryModality, agg: AV
    ) -> WrappedResult[float]:
        """Get apply function for ExtractSteps."""
        return agg_reaction_time(data, agg, key.get_unique_modality(self.mode))


class ExtractKeySpecificReactionTimeDifferencesDTD(
    ExtractDifferencesKeyReactionTimeBase
):
    """Key specific difference Reaction Time ExtractStep."""

    key_list = [DigitConfusionPairModality.DIGIT_6_9]


class ExtractKeySpecificReactionTimeDifferencesSTD(
    ExtractDifferencesKeyReactionTimeBase
):
    """Key specific difference Reaction Time ExtractStep."""

    aggregation_list = CPS_SYMBOL_SPECIFIC_AGGREGATION
    key_list = [
        SymbolConfusionPairModality.SYMBOL_2_7,
        SymbolConfusionPairModality.SYMBOL_3_8,
        SymbolConfusionPairModality.SYMBOL_4_6,
    ]


class ExtractDigit1Error(SequenceTransformMixin, ExtractStep):
    """Extract how many times an incorrect response was given for digit one."""

    data_set_ids = "keys-analysis"

    @transformation
    def _error_digit_1(self, data: pd.DataFrame) -> int:
        return len(data[data.expected == 1].loc[data.actual != 1])

    def get_prototype_definition(self) -> FeatureValueDefinitionPrototype:
        """Get the feature value definition."""
        return FeatureValueDefinitionPrototype(
            feature_name=AV("number of errors", "err"),
            data_type="int16",
            validator=GREATER_THAN_ZERO,
            description=f"The number of errors when digit 1 is displayed for "
            f"{CPSMode.DIGIT_TO_DIGIT} with {self.sequence.av}.",
            modalities=[
                CPSMode.DIGIT_TO_DIGIT.av,
                self.sequence.av,
                DigitEnum.DIGIT_1.av,
            ],
        )

    def get_level_filter(self) -> LevelFilter:
        """Get the level filter based on the bubble size."""
        return CPSModalityFilter(CPSMode.DIGIT_TO_DIGIT, self.sequence)


class ExtractTotalAnswersBase(SequenceModeKeyModalityDefinitionMixIn, ExtractCPSSteps):
    """Mismatching Multiple Extract Steps mix in."""

    data_set_ids = "keys-analysis"
    data_type = "int64"
    validator = GREATER_THAN_ZERO


class ExtractErrorInThird(ExtractTotalAnswersBase):
    """Number of errors in a specific subset ExtractStep."""

    feature_name = AV("number of errors", "err")
    description = "The number of errors within the {subset} "
    subset_list = list(ThirdsModality)

    @staticmethod
    def apply(data: pd.DataFrame, subset: ThirdsModality, level: Level):
        """Compute the number of errors in the selected third."""
        third_data = get_third_data(data, subset, level)
        return third_data["mismatching"].sum()

    def _function_factory(self, subset):
        return dict(
            func=lambda data, level: self.apply(data, subset, level),
            modalities=self.add_modality(subset.av),
            subset=subset,
        )

    def get_function(self) -> List[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [self._function_factory(subset) for subset in self.subset_list]


class ExtractTotalNumError(ExtractTotalAnswersBase):
    """Total of errors ExtractStep."""

    feature_name = AV("total number of errors", "err")
    description = "The total number of errors of the user "

    def get_function(self) -> Iterable[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [{"func": lambda data: data["mismatching"].sum().item()}]


class ExtractCorrectByThird(ExtractTotalAnswersBase):
    """Number of errors in a specific subset ExtractStep."""

    feature_name = AV("number of correct answers", "corr")
    description = "The number of correct answer within the {subset} subset."
    subset_list = list(ThirdsModality)

    @staticmethod
    def apply(data: pd.DataFrame, subset: ThirdsModality, level: Level):
        """Compute the number of errors in the selected third."""
        filtered_data = get_third_data(data, subset, level)
        return (~filtered_data["mismatching"]).sum()

    def _function_factory(self, subset):
        return dict(
            func=lambda data, level: self.apply(data, subset, level),
            modalities=self.add_modality(subset.av),
            subset=subset,
        )

    def get_function(self) -> List[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [self._function_factory(subset) for subset in self.subset_list]


class ExtractCorrectDiffBetweenThird(ExtractTotalAnswersBase):
    """Compute the difference of correct answers between two thirds."""

    feature_name = AV("correct responses difference", "corr_diff")
    data_type = "float64"
    validator = RangeValidator(-30, 30)
    description = (
        "The difference between the number of correct responses of "
        "the {left} of keys and the number of correct responses of "
        "the {right} of keys."
    )
    third_list: List[ThirdsPairModality] = list(ThirdsPairModality)

    def get_function(self) -> Iterable[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [self.function_factory(pair) for pair in self.third_list]

    def _apply_correct_diff(
        self, data: pd.DataFrame, subset: ThirdsPairModality, level: Level
    ) -> float:
        """Difference of correct responses between thirds."""
        left_answers = compute_correct_third_from_paired(
            data, subset, level, is_left=True
        )
        right_answers = compute_correct_third_from_paired(
            data, subset, level, is_left=False
        )

        return left_answers - right_answers

    def function_factory(self, subset: ThirdsPairModality) -> dict:
        """Get function dictionary."""
        return dict(
            func=lambda data, level: self._apply_correct_diff(data, subset, level),
            modalities=self.add_modality(subset.av),
            left=subset.left,
            right=subset.right,
        )


class ExtractTotalErrorPercentage(ExtractTotalAnswersBase):
    """Percentage Total of errors ExtractStep."""

    feature_name = AV("percentage of errors", "err_per")
    description = "The percentage of errors of the user "

    def get_function(self) -> Iterable[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [
            {
                "func": lambda data: (
                    (data["mismatching"].sum() / len(data["mismatching"])) * 100
                ).item()
            }
        ]


class ExtractTotalAnswerLen(ExtractTotalAnswersBase):
    """Total of answers ExtractStep."""

    feature_name = AV("total number of responses", "tot")
    description = "The total number of responses of the user "

    def get_function(self) -> Iterable[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [{"func": lambda data: len(data["mismatching"])}]


class ExtractTotalValidAnswerLen(ExtractTotalAnswersBase):
    """Total of correct answers ExtractStep."""

    feature_name = AV("number of correct responses", "corr")
    description = "The number of correct responses of the user "

    def get_function(self) -> Iterable[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [
            {
                "func": lambda data: len(data["mismatching"])
                - data["mismatching"].sum().item()
            }
        ]


class ExtractMaxStreaksBase(
    NotEmptyDataSetAssertionMixin,
    SequenceModeKeyModalityDefinitionMixIn,
    ExtractCPSSteps,
    metaclass=ABCMeta,
):
    """A feature extraction processing step."""

    data_set_ids = "streak-data"
    data_type = "int64"
    validator = GREATER_THAN_ZERO

    @abstractmethod
    def apply_streak(self, data: pd.DataFrame) -> int:
        """Get the feature value definition."""
        raise NotImplementedError

    def get_function(self) -> Iterable[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [{"func": self.apply_streak}]


class ExtractMaxStreaksCorrectAnswers(ExtractMaxStreaksBase):
    """A feature extraction processing step."""

    feature_name = AV("maximum streak of correct responses", "stk_corr")
    description = "The maximum streak of correct responses of the user "

    def apply_streak(self, data):
        """Get iterable of transform function."""
        return data["correct"][0]


class ExtractMaxStreaksIncorrectAnswers(ExtractMaxStreaksBase):
    """A feature extraction processing step."""

    feature_name = AV("maximum streak of incorrect responses", "stk_incorr")
    description = "The maximum streak of incorrect responses of the user "

    def apply_streak(self, data):
        """Get iterable of transform function."""
        return data["incorrect"][0]


class ExtractPressures(
    SequenceModeKeyModalityDefinitionMixIn, AggregateRawDataSetColumn
):
    """Extract descriptive statistics of applied pressure."""

    data_set_ids = "screen"
    column_id = "pressure"
    feature_name = AV("pressure", "press")
    data_type = "float64"
    description = "The {aggregation} pressure exerted on the screen "
    aggregations = EXTENDED_AGGREGATIONS


class ExtractReactionTimesBase(
    SequenceModeKeyModalityDefinitionMixIn,
    ExtractCPSSteps,
    NotEmptyDataSetAssertionMixin,
):
    """An extraction processing step mix in for reaction time."""

    data_set_ids = "keys-analysis"
    feature_name = AV_REACTION_TIME
    unit = "s"
    data_type = "float64"
    description = "The {aggregation} reaction time for {subset} of the user "
    aggregation: List[Tuple[str, str]]


class ExtractSubsetReactionTimesBase(ExtractReactionTimesBase):
    """An extraction processing step mix in for subset reaction time.

    Attributes
    ----------
    subset
        Enumerated constant representing the specific selection modalities
    lower
        The lower index to select of the data frame.
    upper
        The upper index to select of the data frame.
    """

    aggregation = CPS_EXTENDED_AGGREGATION
    subset: Union[AV, ResponsesModality]
    lower: Optional[int] = None
    upper: Optional[int] = None

    def get_function(self) -> Iterable[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [
            self.function_factory(agg, agg_label) for agg, agg_label in self.aggregation
        ]

    def function_factory(self, agg, agg_label):
        """Get apply function for ExtractSteps."""
        pair = (agg, agg_label)
        validator = (
            INTERACT_DURATION_VALIDATOR if pair in DEFAULT_AGGREGATIONS else None
        )
        modality = (
            self.add_modality(self.subset.av)
            if self.subset != AV("all keys", "all")
            else self.modality
        )
        return dict(
            func=lambda data: agg_reaction_time(
                data, agg, None, self.lower, self.upper
            ),
            aggregation=AV(agg_label, agg),
            modalities=modality,
            subset=self.subset,
            validator=validator,
        )


class ExtractAllReactionTime(ExtractSubsetReactionTimesBase):
    """A reaction time extraction processing step for all keys."""

    subset = AV("all keys", "all")
    lower = 0


class ExtractReactionTimeFiveFirst(ExtractSubsetReactionTimesBase):
    """A reaction time extraction processing step for five first keys."""

    subset = ResponsesModality.FIVE_FIRST
    lower = 0
    upper = 5
    aggregation = CPS_EXTENDED_AGGREGATION


class ExtractReactionTimeFiveLast(ExtractSubsetReactionTimesBase):
    """A reaction time extraction processing step for five last keys."""

    subset = ResponsesModality.FIVE_LAST
    lower = -5
    aggregation = CPS_EXTENDED_AGGREGATION


class ExtractReactionThirdFactory(ExtractReactionTimesBase):
    """Extract reaction time  related features."""

    subset_list = list(ThirdsModality)
    aggregation: List[Tuple[str, str]] = [
        *CPS_BASIC_AGGREGATION,
        ("cv", "coefficient of variation of"),
    ]

    def function_factory(
        self, subset: ThirdsModality, agg: str, agg_label: str
    ) -> Dict:
        """Get apply function for ExtractSteps."""
        pair = (agg, agg_label)
        validator = (
            INTERACT_DURATION_VALIDATOR if pair in DEFAULT_AGGREGATIONS else None
        )
        return dict(
            func=lambda data, level: self.apply_third(data, subset, agg, level),
            aggregation=AV(agg_label, agg),
            modalities=self.add_modality(subset.av),
            subset=subset,
            validator=validator,
        )

    def get_function(self) -> Iterable[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [
            self.function_factory(subset, agg, agg)
            for subset in self.subset_list
            for agg, agg_label in self.aggregation
        ]

    def apply_third(
        self, data: pd.DataFrame, subset: ThirdsModality, agg: str, level: Level
    ) -> WrappedResult[float]:
        """Get Apply function."""
        duration = level.context.get("levelDuration").value
        return agg_reaction_time(
            data, agg, None, *subset.get_lower_upper(data, duration)
        )


class ExtractDifferencesReactionTimesBase(ExtractReactionTimesBase):
    """A reaction time extraction processing step for five last keys."""

    aggregation = DEFAULT_AGGREGATIONS
    data_set_ids = "keys-analysis"
    feature_name = AV("reaction time difference", "rt_diff")
    unit = "s"
    data_type = "float64"
    validator = RangeValidator(-10, 10)
    description = (
        "The difference between the mean reaction time of the"
        " {left} keys and the mean reaction time of the"
        " {right} keys answered by user "
    )


class ExtractReactionTimeDifferencesLastFirst(ExtractDifferencesReactionTimesBase):
    """Extract reaction time difference related features."""

    def get_function(self) -> Iterable[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [
            self.function_factory(agg_tuple[0], agg_tuple[1])
            for agg_tuple in CPS_BASIC_AGGREGATION
        ]

    @staticmethod
    def apply_last_first_reaction_time(
        data: pd.DataFrame, agg: Union[str, AV]
    ) -> WrappedResult[float]:
        """Difference of reaction time between set beginning and end."""
        last: WrappedResult[float] = agg_reaction_time(data, agg, None, -5, None)
        first: WrappedResult[float] = agg_reaction_time(data, agg, None, 0, 5)
        return last - first

    def function_factory(self, agg: str, agg_label: str) -> Dict:
        """Get function dictionary."""
        return dict(
            func=lambda data: self.apply_last_first_reaction_time(data, agg),
            aggregation=AV(agg_label, agg),
            modalities=self.add_modality(
                AV("five last vs five first responses", "5lvs5f")
            ),
            left="five last",
            right="five first",
        )


class ExtractReactionTimeDifferencesThirdDiff(ExtractDifferencesReactionTimesBase):
    """Extract reaction time difference related features."""

    third_list: List[ThirdsPairModality] = list(ThirdsPairModality)

    def get_function(self) -> Iterable[Dict[str, Any]]:
        """Get iterable of transform function."""
        agg_rt_diff = ["mean", "std", "iqr", "min", "q05", "q95", "median"]
        return [
            self.function_factory(pair, agg)
            for pair in self.third_list
            for agg in agg_rt_diff
        ]

    def apply_last_first_diff_reaction_time(
        self, data: pd.DataFrame, subset: ThirdsPairModality, level: Level, agg: str
    ) -> WrappedResult[float]:
        """Difference of reaction time between set beginning and end."""
        duration = level.context.get("levelDuration").value
        return agg_reaction_time(
            data,
            agg,
            None,
            *subset.left.get_lower_upper(data, duration),  # type: ignore
        ) - agg_reaction_time(
            data,
            agg,
            None,
            *subset.right.get_lower_upper(data, duration),  # type: ignore
        )

    def function_factory(self, subset: ThirdsPairModality, agg: str) -> dict:
        """Get function dictionary."""
        return dict(
            func=lambda data, level: self.apply_last_first_diff_reaction_time(
                data, subset, level, agg
            ),
            aggregation=AV(agg, agg),
            modalities=self.add_modality(subset.av),
            left=subset.left,
            right=subset.right,
        )


class ExtractNBacks(SequenceModeKeyModalityDefinitionMixIn, ExtractCPSSteps):
    """Extract multiple strike pattern features."""

    data_set_ids = "n-backs"
    feature_name = AV(
        "reaction time difference over {n_back}-backs occurrences",
        "{n_back}back",
    )
    data_type = "float64"
    unit = "s"
    description = (
        "The {aggregation} reaction time difference between "
        "{n_back}-backs occurrences (e.g. the "
        "{aggregation} reaction time difference between "
        "identical keys encountered in an interval of "
        "{n_back} keys) of the user "
    )

    @staticmethod
    def function_factory(agg: str, agg_label: str, nback: int) -> dict:
        """Get function dictionary."""
        return dict(
            func=lambda data: (data[f"rtCurrent{nback}"] - data[f"rtBack{nback}"]).agg(
                agg
            ),
            aggregation=AV(agg_label, agg),
            n_back=nback,
        )

    def get_function(self) -> Iterable[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [
            self.function_factory(agg, agg_label, nback)
            for agg, agg_label in EXTENDED_AGGREGATIONS
            for nback in [2, 3]
        ]


class ExtractFatigabilityMixin(SequenceModeKeyModalityDefinitionMixIn, ExtractCPSSteps):
    """Extract slope coefficients for capturing fatigability.

    Attributes
    ----------
    feat
        AV to define the slope coefficient or the r2 score of the model.
    """

    feat: AV
    regression_results: str
    validator: Optional[Callable[[Any], None]] = None

    data_set_ids = "keys-analysis"
    feature_name = AV("fatigability", "fat")
    data_type = "float64"
    description = (
        "The {feat} of the linear regression on correct"
        "response times with {regression_modality} capturing "
        "the fatigability of the user "
    )

    def get_function(self) -> Iterable[Dict[str, Any]]:
        """Get iterable of transform function."""
        return [self.function_factory(reg_mod) for reg_mod in RegressionMode]

    def function_factory(self, reg_mod: RegressionMode) -> dict:
        """Get function dictionary."""
        return dict(
            func=lambda data: self.apply_fatigability(data, reg_mod),
            modalities=self.modality
            if reg_mod == RegressionMode.ALL_RESPONSES
            else self.add_modality(reg_mod.av),
            aggregation=self.feat.abbr,
            regression_modality=reg_mod,
            validator=self.validator,
            feat=self.feat,
        )

    def apply_fatigability(self, data: pd.DataFrame, reg_mod: RegressionMode) -> float:
        """Get fatigability apply function."""
        if self.regression_results == "Slope":
            return compute_response_time_linear_regression(data, reg_mod.to_drop)[0]
        if self.regression_results == "R2":
            return compute_response_time_linear_regression(data, reg_mod.to_drop)[1]
        raise ValueError(f"Unknown regression result {self.regression_results}")


class ExtractSlopeFatigability(ExtractFatigabilityMixin):
    """Extract slope coefficients for capturing fatigability."""

    feat = AV("slope coefficient", "slope")
    regression_results = "Slope"
    validator = RangeValidator(-10.0, 10.0)


class ExtractR2ScoreFatigability(ExtractFatigabilityMixin):
    """Extract r2 scores to assess the quality of the slope coefficients."""

    feat = AV("r2 score", "r2")
    regression_results = "R2"
    validator = RangeValidator(0.0, 1.0)


class SummarizeCorrectResponses(AggregateModalities):
    """Summarize correct responses irrespective of key set.

    Parameters
    ----------
    sequence
        The CPS sequence for which to aggregate the features.
    """

    def __init__(self, sequence: CPSSequence):
        super().__init__()
        self.sequence = sequence

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get the feature value definition."""
        return FeatureValueDefinition(
            task_name=TASK_NAME,
            feature_name=AV("correct responses", "corr"),
            unit=None,
            description=f"Total number of correct responses for symbol-to-"
            f"digit with{self.sequence.av} irrespective of the used key set",
            data_type="int64",
            validator=TOTAL_RESPONSES_VALIDATOR,
            modalities=[CPSMode.SYMBOL_TO_DIGIT.av, self.sequence.av],
            aggregation=None,
        )

    def get_modalities(self) -> List[List[Union[str, AV]]]:
        """Get the modalities to aggregate."""
        return [
            [CPSMode.SYMBOL_TO_DIGIT.av, self.sequence.av, key.av] for key in CPSKeySet
        ]


class SummarizeKeySetOneTwoCorrectResponses(SummarizeCorrectResponses):
    """Summarize correct responses of key set one and two."""

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get the feature value definition."""
        return FeatureValueDefinition(
            task_name=TASK_NAME,
            feature_name=AV("correct responses", "corr"),
            description=f"Total number of correct responses for symbol-to-digit with"
            f" {self.sequence.av} and key set one or two.",
            unit="int64",
            validator=TOTAL_RESPONSES_VALIDATOR,
            modalities=[
                CPSMode.SYMBOL_TO_DIGIT.av,
                self.sequence.av,
                AV("predefined key set one and two", "key1n2"),
            ],
        )

    def get_modalities(self) -> List[List[Union[str, AV]]]:
        """Get the modalities to aggregate."""
        return [
            [CPSMode.SYMBOL_TO_DIGIT.av, self.sequence.av, key.av]
            for key in (CPSKeySet.KEY1, CPSKeySet.KEY2)
        ]


class SummarizeResponseTimes(AggregateModalities):
    """Summarize response times irrespective of key set."""

    def __init__(self, mode: CPSMode):
        super().__init__()
        self.mode = mode

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get the feature value definition."""
        return FeatureValueDefinition(
            task_name=TASK_NAME,
            feature_name=AV_REACTION_TIME,
            unit="s",
            description=f"The average reaction time for {self.mode} test "
            "irrespective of sequence.",
            data_type="float64",
            validator=INTERACT_DURATION_VALIDATOR,
            modalities=[self.mode.av],
            aggregation=AV("mean", "mean"),
        )

    def get_modalities(self) -> List[List[Union[str, AV]]]:
        """Get the modalities to aggregate."""
        if self.mode not in CPSMode:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode == CPSMode.DIGIT_TO_DIGIT:
            return [[self.mode.av, sequence.av] for sequence in CPSSequence]
        return [
            [self.mode.av, sequence.av, key.av]
            for sequence in CPSSequence
            for key in CPSKeySet
        ]


class SummarizeKeySetOneTwoReactionTimeExtraModality(AggregateModalities):
    """Summarize reaction time for key set one and two."""

    def __init__(self, sequence: CPSSequence, aggregation: str, **kwargs):
        self.sequence = sequence
        self.aggregation = aggregation
        self.extra_modality = kwargs.pop("extra_modality", None)
        super().__init__()

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get the feature value definition."""
        modalities = [
            CPSMode.SYMBOL_TO_DIGIT.av,
            self.sequence.av,
            AV("predefined key set one and two", "key1n2"),
        ]
        if self.extra_modality is not None:
            modalities.append(self.extra_modality)

        return FeatureValueDefinition(
            task_name=TASK_NAME,
            feature_name=AV_REACTION_TIME,
            unit="s",
            description=f"Reaction time {self.aggregation} for symbol-to-digit"
            f" with {self.sequence.av} and key set one or two "
            f'{self.extra_modality or ""}.',
            data_type="int64",
            modalities=modalities,
            aggregation=self.aggregation,
        )

    def get_modalities(self) -> List[List[Union[str, AV]]]:
        """Get the modalities to aggregate."""
        if self.extra_modality is not None:
            return [
                [
                    CPSMode.SYMBOL_TO_DIGIT.av,
                    self.sequence.av,
                    key.av,
                    self.extra_modality,
                ]
                for key in (CPSKeySet.KEY1, CPSKeySet.KEY2)
            ]
        return [
            [CPSMode.SYMBOL_TO_DIGIT.av, self.sequence.av, key.av]
            for key in (CPSKeySet.KEY1, CPSKeySet.KEY2)
        ]


class SummarizeKeySetOneTwoReactionTimeDiff(AggregateModalities):
    """Summarize reaction time difference of key set one and two."""

    def __init__(self, sequence: CPSSequence, aggregation: str, extra_modality):
        self.sequence = sequence
        self.aggregation = aggregation
        self.extra_modality = extra_modality
        super().__init__()

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get the feature value definition."""
        return FeatureValueDefinition(
            task_name=TASK_NAME,
            feature_name=AV("reaction time difference", "rt_diff"),
            unit="s",
            description=f"Reaction time difference {self.aggregation} for "
            f"symbol-to-digit with {self.sequence.av} and key set one or two "
            f"{self.extra_modality}.",
            data_type="int64",
            validator=RangeValidator(-10, 10),
            modalities=[
                CPSMode.SYMBOL_TO_DIGIT.av,
                self.sequence.av,
                AV("predefined key set one and two", "key1n2"),
                self.extra_modality,
            ],
            aggregation=self.aggregation,
        )

    def get_modalities(self) -> List[List[Union[str, AV]]]:
        """Get the modalities to aggregate."""
        return [
            [CPSMode.SYMBOL_TO_DIGIT.av, self.sequence.av, key.av, self.extra_modality]
            for key in (CPSKeySet.KEY1, CPSKeySet.KEY2)
        ]


class AggregateFixedSubstitutionTime(AggregateFeatures):
    """Extract the substitution time for the fixed keys."""

    feature_ids = [DTD_RT_MEAN, STD_KEY_FIXED1_RT_MEAN, STD_KEY_FIXED2_RT_MEAN]
    fail_if_missing = False
    aggregation_method = _compute_substitution_time
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("substitution time", "substitution_time"),
        description="The substitution time for fixed keys is "
        "defined as the difference between the symbol to digit "
        "reaction time (the time required to associate a symbol "
        "with a number) for the fixed keys 1 or 2 and the "
        "digit to digit reaction time (the time required to "
        "associate a number with a number).",
        unit="s",
        data_type="float",
        modalities=[AV("keyf", "keyf")],
    )


class AggregateRandomSubstitutionTime(AggregateFeatures):
    """Extract the substitution time for the random keys."""

    feature_ids = [DTD_RT_MEAN, STD_KEY_RANDOM_RT_MEAN]
    fail_if_missing = False
    aggregation_method = _compute_substitution_time
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("substitution time", "substitution_time"),
        description="The substitution time for random keys is "
        "defined as the difference between the symbol to digit "
        "reaction time (the time required to associate a symbol "
        "with a number) for the random keys and the digit to "
        "digit reaction time (the time required to associate a "
        "number with a number).",
        unit="s",
        data_type="float",
        modalities=[AV("keyr", "keyr")],
    )


class ExtractCPSFeatures(ProcessingStepGroup):
    """Extract all features for a given mode, sequence and key set.

    Parameters
    ----------
    mode
        The desired mode to compute the transformation
        (``digit-to-symbol``, ``digit-to-digit``).
    key_set
        The specific key set filter on which you desire to extract features.
    sequence
        The sequence type
    """

    def __init__(
        self, mode: CPSMode, sequence: CPSSequence, key_set: Optional[CPSKeySet] = None
    ):
        steps = [
            ExtractAllReactionTime(mode=mode, sequence=sequence, key_set=key_set),
            ExtractReactionTimeFiveFirst(mode=mode, sequence=sequence, key_set=key_set),
            ExtractReactionTimeFiveLast(mode=mode, sequence=sequence, key_set=key_set),
            ExtractReactionThirdFactory(mode=mode, sequence=sequence, key_set=key_set),
            ExtractReactionTimeDifferencesLastFirst(
                mode=mode, sequence=sequence, key_set=key_set
            ),
            ExtractReactionTimeDifferencesThirdDiff(
                mode=mode, sequence=sequence, key_set=key_set
            ),
            ExtractTotalValidAnswerLen(mode=mode, sequence=sequence, key_set=key_set),
            ExtractTotalAnswerLen(mode=mode, sequence=sequence, key_set=key_set),
            ExtractErrorInThird(mode=mode, sequence=sequence, key_set=key_set),
            ExtractCorrectByThird(mode=mode, sequence=sequence, key_set=key_set),
            ExtractCorrectDiffBetweenThird(
                mode=mode, sequence=sequence, key_set=key_set
            ),
            ExtractTotalNumError(mode=mode, sequence=sequence, key_set=key_set),
            ExtractTotalErrorPercentage(mode=mode, sequence=sequence, key_set=key_set),
            ExtractMaxStreaksCorrectAnswers(
                mode=mode, sequence=sequence, key_set=key_set
            ),
            ExtractMaxStreaksIncorrectAnswers(
                mode=mode, sequence=sequence, key_set=key_set
            ),
            ExtractPressures(mode=mode, sequence=sequence, key_set=key_set),
            ExtractNBacks(mode=mode, sequence=sequence, key_set=key_set),
            ExtractDigitSymmetryUniqueReactionTimes(
                mode=mode, sequence=sequence, key_set=key_set
            ),
            ExtractDigitSymmetryPairedReactionTimes(
                mode=mode, sequence=sequence, key_set=key_set
            ),
            ExtractSlopeFatigability(mode=mode, sequence=sequence, key_set=key_set),
            ExtractR2ScoreFatigability(mode=mode, sequence=sequence, key_set=key_set),
        ]

        super().__init__(
            steps,
            task_name=TASK_NAME,
            mode=mode.av,
            sequence=sequence.av,
            key_set=key_set.av if key_set else None,
            modalities=drop_none(
                [mode.av, sequence.av, key_set.av if key_set else None]
            ),
            level_filter=CPSModalityFilter(mode, sequence, key_set),
        )


class ExtractCPSFeaturesDTD(ProcessingStepGroup):
    """Extract all features for DTD mode, sequence and key set.

    Parameters
    ----------
    sequence
        The sequence type
    """

    def __init__(self, sequence: CPSSequence):
        mode = CPSMode.DIGIT_TO_DIGIT
        steps = [
            ExtractConfusionDTD(mode=mode, sequence=sequence),
            ExtractConfusionRateDTD(mode=mode, sequence=sequence),
            ExtractDigitSpecificReactionTimesDTD(mode=mode, sequence=sequence),
            ExtractKeySpecificReactionTimeDifferencesDTD(mode=mode, sequence=sequence),
            ExtractDigit1Error(sequence=sequence),
        ]  # type: ignore

        super().__init__(
            steps,
            task_name=TASK_NAME,
            mode=mode.av,
            sequence=sequence.av,
            modalities=[mode.av, sequence.av],
            level_filter=CPSModalityFilter(mode, sequence),
        )


class ExtractCPSFeaturesSTD(ProcessingStepGroup):
    """Extract all features for STD mode, sequence and key set.

    Parameters
    ----------
    key_set
        The specific key set filter on which you desire to extract features.
    sequence
        The sequence type
    """

    def __init__(self, sequence: CPSSequence, key_set: CPSKeySet):
        mode = CPSMode.SYMBOL_TO_DIGIT
        steps = [
            ExtractConfusionDTS(mode=mode, sequence=sequence, key_set=key_set),
            ExtractConfusionRateDTS(mode=mode, sequence=sequence, key_set=key_set),
            ExtractDigitSpecificReactionTimesSTD(
                mode=mode, sequence=sequence, key_set=key_set
            ),
            ExtractKeySpecificReactionTimeDifferencesSTD(
                mode=mode, sequence=sequence, key_set=key_set
            ),
        ]  # type: ignore

        super().__init__(
            steps,
            task_name=TASK_NAME,
            mode=mode.av,
            sequence=sequence.av,
            key_set=key_set.av,
            modalities=[mode.av, sequence.av, key_set.av],
            level_filter=CPSModalityFilter(mode, sequence, key_set),
        )


class SummarizeFeatures(ProcessingStepGroup):
    """A processing step group containing all the feature aggregation steps."""

    def __init__(self):
        steps = [
            # Summarize correct responses global
            *(SummarizeCorrectResponses(sequence) for sequence in CPSSequence),
            # Summarize reaction time global
            *(SummarizeResponseTimes(mode) for mode in CPSMode),
            # Summarize correct responses key1 and key2
            *(
                SummarizeKeySetOneTwoCorrectResponses(sequence)
                for sequence in CPSSequence
            ),
            # Summarize reaction time key1 key2
            *(
                SummarizeKeySetOneTwoReactionTimeExtraModality(sequence, agg)
                for sequence in CPSSequence
                for agg in CPS_AGGREGATION_LIST
            ),
            # Summarize reaction time key1 key2 for all the extra modalities
            *(
                SummarizeKeySetOneTwoReactionTimeExtraModality(
                    sequence, agg, extra_modality=modality
                )
                for sequence in CPSSequence
                for agg in CPS_AGGREGATION_LIST
                for modality in EXTRA_MODALITY_LIST
            ),
            # Summarize reaction time difference key1 key2
            *(
                SummarizeKeySetOneTwoReactionTimeDiff(
                    sequence, agg, extra_modality=modality
                )
                for sequence in CPSSequence
                for agg in CPS_AGGREGATION_LIST
                for modality in EXTRA_MODALITY_LIST
            ),
        ]
        super().__init__(steps, task_name=TASK_NAME)


class CPSProcessingStepGroup(ProcessingStepGroup):
    """A group of all cps processing steps for features extraction."""

    def __init__(self):
        steps = [
            TransformKeysAnalysisData(),
            TransformStreakData(),
            TransformConfusion(),
            TransformNBacks(),
            *(
                ExtractCPSFeatures(CPSMode.SYMBOL_TO_DIGIT, sequence, key_set)
                for sequence in CPSSequence
                for key_set in CPSKeySet
            ),
            *(
                ExtractCPSFeatures(CPSMode.DIGIT_TO_DIGIT, sequence)
                for sequence in CPSSequence
            ),
            *(ExtractCPSFeaturesDTD(sequence) for sequence in CPSSequence),
            *(
                ExtractCPSFeaturesSTD(sequence, key_set)
                for sequence in CPSSequence
                for key_set in CPSKeySet
            ),
            SummarizeFeatures(),
            AggregateFixedSubstitutionTime(),
            AggregateRandomSubstitutionTime(),
        ]

        super().__init__(steps, task_name=TASK_NAME)
