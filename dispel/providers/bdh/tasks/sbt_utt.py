"""Processing functionality for SBT/UTT task."""

from dispel.processing.assertions import AssertEvaluationFinished
from dispel.processing.level import ProcessingStepGroup
from dispel.providers.bdh.data import BDHReading
from dispel.providers.bdh.transform import TRUNCATE_SENSORS, TruncateSensorsSBT
from dispel.providers.generic.tasks.sbt_utt import TASK_NAME
from dispel.providers.generic.tasks.sbt_utt.sbt import (
    SBTProcessingSteps as GenericSBTProcessingSteps,
)
from dispel.providers.generic.tasks.sbt_utt.utt import (
    UTTProcessingSteps as GenericUTTProcessingSteps,
)
from dispel.providers.registry import process_factory


class SBTProcessingSteps(ProcessingStepGroup):
    """SBT processing steps."""

    steps = [
        TruncateSensorsSBT(),
        GenericSBTProcessingSteps(),
    ]


class UTTProcessingSteps(ProcessingStepGroup):
    """UTT processing steps."""

    steps = [
        *TRUNCATE_SENSORS,
        GenericUTTProcessingSteps(),
    ]


class SBTUTTProcessingSteps(ProcessingStepGroup):
    """Combined SBT and UTT processing steps."""

    steps = [
        AssertEvaluationFinished(),
        SBTProcessingSteps(),
        UTTProcessingSteps(),
    ]


process_sbt_utt = process_factory(
    task_name=TASK_NAME,
    steps=SBTUTTProcessingSteps(),
    codes=("sbtUtt-activity", "sbut-activity"),
    supported_type=BDHReading,
)
