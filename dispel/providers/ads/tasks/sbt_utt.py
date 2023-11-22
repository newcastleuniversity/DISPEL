"""AdS specific SBT and UTT processing."""

from dispel.processing.assertions import AssertEvaluationFinished
from dispel.providers.ads.data import ADSReading
from dispel.providers.generic.tasks.sbt_utt import TASK_NAME
from dispel.providers.generic.tasks.sbt_utt.sbt import SBTProcessingSteps
from dispel.providers.generic.tasks.sbt_utt.utt import UTTProcessingSteps
from dispel.providers.registry import process_factory

process_sbt_utt = process_factory(
    task_name=TASK_NAME,
    steps=[
        AssertEvaluationFinished(),
        SBTProcessingSteps(),
        UTTProcessingSteps(),
    ],
    codes="sbtUtt",
    supported_type=ADSReading,
)
