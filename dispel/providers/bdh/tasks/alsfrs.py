"""Functionalities to process ALSFRS data."""

from typing import List

import pandas as pd

from dispel.data.features import FeatureValueDefinitionPrototype
from dispel.data.validators import RangeValidator
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import ValueDefinition
from dispel.processing import ProcessingStep
from dispel.processing.data_set import transformation
from dispel.processing.extract import ExtractStep
from dispel.processing.level import ProcessingStepGroup
from dispel.providers.registry import process_factory

TASK_NAME = AV("Amyotrophic Lateral Sclerosis Functional Rating Scale", "ALSFRS")


DOMAINS = {
    "speech": 0,
    "salivation": 1,
    "swallowing": 2,
    "handwritting": 3,
    "cutting_food": 5,
    "dressing_and_hygiene": 6,
    "turning_in_bed": 7,
    "walking": 8,
    "climbing_stairs": 9,
    "dyspnea": 10,
    "orthopnea": 11,
    "respiratory_insufficiency": 12,
}
r"""The twelve functional domains evaluated through the survey."""


class ExtractAnswer(ExtractStep):
    """Extract alsfrs score for individual questions."""

    data_set_ids = "userInput"

    def __init__(self, domain: str, *args, **kwargs):
        self.domain = domain
        super().__init__(*args, **kwargs)

    @transformation
    def read_als_score(self, data: pd.DataFrame) -> float:
        """Read als_frs score corresponding to the domain."""
        return 5 - data.iloc[DOMAINS[self.domain]]["answer"]

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Overwrite get_definition."""
        description = f"ALS FRS score for {self.domain} between 0 and 4."
        return FeatureValueDefinitionPrototype(
            feature_name=AV(f"{self.domain} score", f"{self.domain}_score"),
            data_type="float",
            validator=RangeValidator(lower_bound=0, upper_bound=4),
            description=description,
        ).create_definition(**kwargs)


class ExtractFullScore(ExtractStep):
    """Extract alsfrs total score."""

    data_set_ids = "userInput"

    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("Total ALSFRS score", "total_score"),
        data_type="float",
        validator=RangeValidator(lower_bound=0, upper_bound=48),
        description="The total ALS FRS score is the sum of the twelve "
        "scores (one for each function evaluated) and has a value"
        "between 0 and 48.",
    )

    @transformation
    def read_als_score(self, data: pd.DataFrame) -> float:
        """Read als_frs score corresponding to the domain."""
        return sum(5 - data.iloc[list(DOMAINS.values())]["answer"])


STEPS: List[ProcessingStep] = [ExtractFullScore()]
STEPS += [ExtractAnswer(domain) for domain in DOMAINS]

STEPS = [ProcessingStepGroup(STEPS, task_name=TASK_NAME)]

process_als_frs = process_factory(
    task_name=TASK_NAME,
    steps=STEPS,
    codes="alsfrs-activity",
)
