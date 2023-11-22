"""A module to store the upper limbs related flags."""

from dispel.data.core import Reading
from dispel.data.flags import FlagSeverity, FlagType
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.core import FlagReadingStep
from dispel.processing.flags import flag


class OnlyOneHandPerformed(FlagReadingStep):
    """Flag record with only one hand performed."""

    flag_name = AV("only one hand", "1hand")
    flag_type = FlagType.TECHNICAL
    flag_severity = FlagSeverity.DEVIATION
    reason = "The user is not using {missing_hand} hand."

    @flag
    def _check_single_hand(self, reading: Reading, **kwargs) -> bool:
        levels = reading.levels
        missing_hand = None
        if not any(["left" in str(lvl.id) for lvl in levels]):
            missing_hand = "left"
        if not any(["right" in str(lvl.id) for lvl in levels]):
            missing_hand = "right"
        if missing_hand:
            self.set_flag_kwargs(missing_hand=missing_hand, **kwargs)
            return False
        return True
