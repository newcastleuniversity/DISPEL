"""A module containing functionality to process/filter specific modalities."""
from enum import Enum
from typing import Iterable, List, Optional, Set, Tuple, Union

import pandas as pd

from dispel.data.levels import Level
from dispel.data.validators import RangeValidator
from dispel.data.values import AVEnum
from dispel.processing.level import LevelFilter

#: range of symbols/digits
_DIGIT_RANGE = range(1, 10)

#: range of n-backs studied
NBACKS = range(1, 4)

#: Range validator for maximum total responses
TOTAL_RESPONSES_VALIDATOR = RangeValidator(0, 120)

#: The maximum time for the user to interact during an attempt of an upper task
INTERACT_DURATION_VALIDATOR = RangeValidator(lower_bound=0, upper_bound=10)


class CPSMode(AVEnum):
    """Enumerated constant representing the level modalities."""

    SYMBOL_TO_DIGIT = ("symbol-to-digit", "std")
    DIGIT_TO_DIGIT = ("digit-to-digit", "dtd")


class CPSKeySet(AVEnum):
    """Enumerated constant representing the key set modalities."""

    KEY1 = ("predefined key set one", "key1")
    KEY2 = ("predefined key set two", "key2")
    KEY3 = ("predefined key set three", "key3")
    RANDOM = ("random key set", "keyr")


class CPSSequence(AVEnum):
    """Enumerated constant representing the sequence type."""

    PRE_DEFINED = ("predefined sequence", "pre")
    RANDOM = ("random sequence", "rand")


class ResponsesModality(AVEnum):
    """Enumerated constant representing the specific selection modalities."""

    FIVE_FIRST = ("five first responses", "f5")
    FIVE_LAST = ("five last responses", "l5")
    CORRECT = ("correct responses only", "corr")


class SymbolEnum(AVEnum):
    """Enumerated constant representing symbol modalities."""

    def __init__(self, number: int):
        super().__init__(f"symbol {number}", f"sym{number}")

    SYMBOL_1 = 1
    SYMBOL_2 = 2
    SYMBOL_3 = 3
    SYMBOL_4 = 4
    SYMBOL_5 = 5
    SYMBOL_6 = 6
    SYMBOL_7 = 7
    SYMBOL_8 = 8
    SYMBOL_9 = 9


class DigitEnum(AVEnum):
    """Enumerated constant of the digits for the d2d phase."""

    def __init__(self, number: int):
        super().__init__(f"digit {number}", f"dig{number}")

    DIGIT_1 = 1
    DIGIT_2 = 2
    DIGIT_3 = 3
    DIGIT_4 = 4
    DIGIT_5 = 5
    DIGIT_6 = 6
    DIGIT_7 = 7
    DIGIT_8 = 8
    DIGIT_9 = 9


class ThirdsModality(AVEnum):
    """Enumerated constant representing the specific set third modalities.

    The Konectom symbol-to-digit cognitive processing speed test is decomposed
    into three sets (first, second, third). The sets are computed based on the
    total duration of the level obtained from the context. The length of a set
    is defined as the total_duration / 3.
    """

    def get_lower_upper(self, data, duration) -> Tuple[int, int]:
        """Get third boundaries from data set."""
        third_duration = pd.Timedelta(duration / 3, unit="seconds")
        n_third = int(self.value)
        ts_start = data["tsAnswer"].iloc[0]
        if n_third == 1:
            lower = ts_start
            upper = ts_start + third_duration
        elif n_third == 2:
            lower = ts_start + third_duration
            upper = ts_start + (third_duration * 2)
        elif n_third == 3:
            lower = ts_start + (third_duration * 2)
            upper = ts_start + pd.Timedelta(duration, unit="seconds")
        elif n_third == 4:
            lower = ts_start + third_duration
            upper = ts_start + pd.Timedelta(duration, unit="seconds")
        else:
            raise ValueError("num_third has to be comprise in [1, 2, 3, 4]")

        return lower, upper

    FIRST_THIRD = ("1st third of keys", "third1")
    SECOND_THIRD = ("2nd third of keys", "third2")
    THIRD_THIRD = ("3rd third of keys", "third3")
    SECOND_LAST_THIRD = ("2nd and 3rd third of keys", "third2third3")


class PairModalityBase(AVEnum):
    """Enumerated pair modalities base."""

    def __init__(self, left: AVEnum, right: AVEnum):
        self.left = left
        self.right = right
        super().__init__(f"{left.av} and {right.av}", f"{left.abbr}{right.abbr}")

    @classmethod
    def from_pair(cls, left: AVEnum, right: AVEnum) -> "PairModalityBase":
        """Extract PairModality for a given pair of keys."""
        for member in cls:
            if member.left == left and member.right == right:
                return member
        raise KeyError("Unknown pair: {symbol} ~ {other}")


class ThirdsPairModality(PairModalityBase):
    """Enumerated comparisons between the different parts of the data set."""

    THIRD_2_1 = (ThirdsModality.SECOND_THIRD, ThirdsModality.FIRST_THIRD)
    THIRD_3_1 = (ThirdsModality.THIRD_THIRD, ThirdsModality.FIRST_THIRD)
    THIRD_3_2 = (ThirdsModality.THIRD_THIRD, ThirdsModality.SECOND_THIRD)


class SymbolConfusionPairModality(PairModalityBase):
    """Enumerated symbol  confusions modalities."""

    SYMBOL_2_7 = (SymbolEnum.SYMBOL_2, SymbolEnum.SYMBOL_7)
    SYMBOL_7_2 = (SymbolEnum.SYMBOL_7, SymbolEnum.SYMBOL_2)
    SYMBOL_4_6 = (SymbolEnum.SYMBOL_4, SymbolEnum.SYMBOL_6)
    SYMBOL_6_4 = (SymbolEnum.SYMBOL_6, SymbolEnum.SYMBOL_4)
    SYMBOL_3_8 = (SymbolEnum.SYMBOL_3, SymbolEnum.SYMBOL_8)
    SYMBOL_8_3 = (SymbolEnum.SYMBOL_8, SymbolEnum.SYMBOL_3)


class DigitConfusionPairModality(PairModalityBase):
    """Enumerated digit confusions modalities."""

    DIGIT_6_9 = (DigitEnum.DIGIT_6, DigitEnum.DIGIT_9)
    DIGIT_9_6 = (DigitEnum.DIGIT_9, DigitEnum.DIGIT_6)


PairType = Union[
    DigitConfusionPairModality, SymbolConfusionPairModality, ThirdsPairModality
]


class SymmetryModality(AVEnum):
    """Enumerated constant representing the coupling modalities."""

    @staticmethod
    def get_pair_modality(mode: CPSMode) -> List[SymbolEnum]:
        """Get pair values from modality."""
        if mode == CPSMode.SYMBOL_TO_DIGIT:
            return [SymbolEnum(i) for i in [2, 7, 3, 8, 4, 6]]
        return [SymbolEnum(i) for i in [6, 9]]

    @staticmethod
    def get_unique_modality(mode: CPSMode) -> List[SymbolEnum]:
        """Get Unique values from modality."""
        if mode == CPSMode.SYMBOL_TO_DIGIT:
            return [SymbolEnum(i) for i in [1, 5, 9]]
        return [SymbolEnum(i) for i in [1, 2, 3, 4, 5, 7, 8]]

    PAIRED = ("paired keys", "pair")
    UNIQUE = ("unique keys", "unique")


KeyType = Union[DigitEnum, SymmetryModality, PairType, SymbolEnum]


class CPSLevel(str, Enum):
    """An enum for CPS levels."""

    SYMBOL_TO_DIGIT = "symbol_to_digit"
    DIGIT_TO_DIGIT = "digit_to_digit"


class CPSModalityFilter(LevelFilter):
    """Filter over levels, sequences, and key sets."""

    _MODE_TO_LEVEL = {
        CPSMode.SYMBOL_TO_DIGIT: CPSLevel.SYMBOL_TO_DIGIT,
        CPSMode.DIGIT_TO_DIGIT: CPSLevel.DIGIT_TO_DIGIT,
    }

    _KEY_SET_TO_CONTEXT = {
        CPSKeySet.KEY1: "predefinedKey1",
        CPSKeySet.KEY2: "predefinedKey2",
        CPSKeySet.KEY3: "predefinedKey3",
        CPSKeySet.RANDOM: "randomKey",
    }

    _SEQUENCE_TO_CONTEXT = {
        CPSSequence.PRE_DEFINED: "predefinedSequence",
        CPSSequence.RANDOM: "randomSequence",
    }

    def __init__(
        self, mode: CPSMode, sequence: CPSSequence, key_set: Optional[CPSKeySet] = None
    ):
        self.mode = mode
        self.sequence = sequence
        self.key_set = key_set

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter for levels."""

        def _filter(level: Level):
            return (
                level.id == self.level_id
                and level.context[self.sequence_context_id].value
                and (
                    not self.key_set
                    or (
                        self.key_set_context_id in level.context
                        and level.context[self.key_set_context_id].value
                    )
                )
            )

        return set(filter(_filter, levels))

    def repr(self) -> str:
        """Get representation of the filter."""
        res = f"{self.mode} with {self.sequence}"
        if self.key_set:
            res += f" using {self.key_set}"
        return res

    @property
    def level_id(self) -> CPSLevel:
        """Return the proper level id."""
        return self._MODE_TO_LEVEL[self.mode]

    @property
    def sequence_context_id(self) -> str:
        """Return the proper sequence context id."""
        return self._SEQUENCE_TO_CONTEXT[self.sequence]

    @property
    def key_set_context_id(self) -> str:
        """Return the proper key set context id."""
        assert self.key_set is not None, "key set must be provided"
        return self._KEY_SET_TO_CONTEXT[self.key_set]


class RegressionMode(AVEnum):
    """Enumerated constant representing the linear regression measures."""

    ALL_RESPONSES = ("all keys", "all")
    ONE_ANS_REM = ("one answer removed", "1rem")
    TWO_ANS_REM = ("two answers removed", "2rem")
    THREE_ANS_REM = ("three answers removed", "3rem")

    @property
    def to_drop(self) -> int:
        """Return the number of data points to be dropped in the regression."""
        return int(self) - 1
