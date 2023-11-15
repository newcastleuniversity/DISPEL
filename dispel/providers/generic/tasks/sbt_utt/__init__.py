"""Static Balance and U-turn Test module.

A module containing the functionality to process the *Static Balance* test
(SBT) and *U-turn* test (UTT).

As the two tests SBT and UTT are performed consecutively, they are always
processed together. The respective functionality can be found in their modules:

- :mod:`~dispel.providers.generic.tasks.sbt_utt.sbt` for the SBT
- :mod:`~dispel.providers.generic.tasks.sbt_utt.utt` for the UTT
"""

from dispel.data.values import AbbreviatedValue as AV
from dispel.providers.generic.tasks.sbt_utt.sbt import SBTProcessingSteps
from dispel.providers.registry import process_factory

TASK_NAME = AV("Static Balance and U-turn test", "SBT-UTT")

process_sbt = process_factory(
    task_name=TASK_NAME,
    steps=SBTProcessingSteps(),
    codes=("sbt", "sbt-test", "sbt-activity"),
)
