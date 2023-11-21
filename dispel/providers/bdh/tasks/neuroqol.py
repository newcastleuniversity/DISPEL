"""Functionalities to process neuroqol data."""

from typing import List, Tuple, Type

import pandas as pd

from dispel.data.levels import Level
from dispel.data.measures import MeasureValue, MeasureValueDefinitionPrototype
from dispel.data.raw import RawDataSet, RawDataSetDefinition, RawDataValueDefinition
from dispel.data.validators import GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import ValueDefinition
from dispel.processing.core import ProcessingResult, ProcessingStep
from dispel.processing.data_set import transformation
from dispel.processing.extract import ExtractStep
from dispel.processing.level import ProcessingStepGroup
from dispel.providers.registry import process_factory

TASK_NAME = AV("NeuroQol Assessment", "neuroqol")

NEUROQOL_SUBTEST_NAME = [
    "upper_extremity",
    "lower_extremity",
    "sleep",
    "fatigue",
    "anxiety",
    "depression",
    "stigma",
    "cognitive_function",
    "ability_participate_social_roles",
    "satisfaction_social_roles",
]
r"""The name of the neuroqol sub tests."""


def get_theta_scores(
    levels: List[Level],
) -> Tuple[pd.DataFrame, List[Type[MeasureValue]]]:
    """Create a data frame with standard error theta and t scores."""
    len_levels = len(levels)
    theta_scores = [float] * len_levels
    standard_errors = [float] * len_levels
    level_names = [""] * len_levels
    measure_values = [MeasureValue] * len_levels

    # Enumerate through the levels to read the mobile_computed information
    for it, level in enumerate(levels):
        _id = str(level.id)
        measure_id = f"mobile_computed_theta_score_{_id}"
        level_names[it] = _id
        measure_value = level.measure_set.get(measure_id)
        measure_values[it] = measure_value  # type: ignore
        theta_scores[it] = measure_value.value
        standard_errors[it] = level.measure_set.get_raw_value(
            f"mobile_computed_standard_error_{_id}"
        )
    # Create a single dataframe with all the information
    df = pd.DataFrame(
        {
            "level_name": level_names,
            "theta_score": theta_scores,
            "standard_error": standard_errors,
        }
    )
    df["t_score"] = df["theta_score"] * 10 + 50
    return df, measure_values


class ListNeuroqolThetaScores(ProcessingStep):
    """Gather individual scores in the same data frame."""

    def process_reading(self, reading, **kwargs):
        """Gather theta scores."""
        definitions = [
            RawDataValueDefinition("level_name", "level name"),
            RawDataValueDefinition("theta_score", "theta score"),
            RawDataValueDefinition("standard_error", "standard error"),
            RawDataValueDefinition("t_score", "T score"),
        ]
        new_data_set_id = "theta_scores"

        data, measure_values = get_theta_scores(reading.levels)

        raw_data_sets = [
            RawDataSet(
                definition=RawDataSetDefinition(
                    id=new_data_set_id,
                    source=None,
                    value_definitions_list=definitions,
                    is_computed=True,
                ),
                data=data,
            )
        ]

        yield ProcessingResult(
            step=self,
            sources=measure_values,
            result=Level(
                id_="all_levels",
                start=reading.evaluation.start,
                end=reading.evaluation.end,
                raw_data_sets=raw_data_sets,
            ),
        )


class ExtractNeuroqol(ExtractStep):
    """Extract subset information."""

    data_set_ids = "theta_scores"

    def __init__(self, level_name: str, *args, **kwargs):
        self.level_name = level_name
        super().__init__(*args, **kwargs)


class ExtractNeuroqolTScore(ExtractNeuroqol):
    """Extract subset t scores."""

    @transformation
    def get_t_score(self, data: pd.DataFrame) -> float:
        """Get t_score."""
        return float(data.loc[data.level_name == self.level_name].t_score)

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Overwrite get_definition."""
        description = f"T-score for {self.level_name} rescaled between 0 and 100."
        return MeasureValueDefinitionPrototype(
            measure_name=AV("t score", "t_score"),
            data_type="float",
            validator=GREATER_THAN_ZERO,
            description=description,
            modalities=[AV(self.level_name)],
        ).create_definition(**kwargs)


class ExtractNeuroqolStandardError(ExtractNeuroqol):
    """Extract subset standard error."""

    @transformation
    def get_standard_error(self, data: pd.DataFrame) -> float:
        """Get t_score."""
        return float(data.loc[data.level_name == self.level_name].standard_error)

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Overwrite get_definition."""
        description = f"Standard Error for {self.level_name}."
        return MeasureValueDefinitionPrototype(
            measure_name=AV("Standard error", "standard_error"),
            data_type="float",
            validator=GREATER_THAN_ZERO,
            description=description,
            modalities=[AV(self.level_name)],
        ).create_definition(**kwargs)


EXTRACTION_STEPS: List[ProcessingStep] = []
for name in NEUROQOL_SUBTEST_NAME:
    EXTRACTION_STEPS.append(ExtractNeuroqolTScore(name))
    EXTRACTION_STEPS.append(ExtractNeuroqolStandardError(name))


STEPS: List[ProcessingStep] = [
    # We filter on only one level to create the data set a single time
    ListNeuroqolThetaScores(),
    ProcessingStepGroup(
        EXTRACTION_STEPS, level_filter="all_levels", task_name=TASK_NAME
    ),
]

process_neuroqol = process_factory(
    task_name=TASK_NAME,
    steps=STEPS,
    codes="neuroqol-activity",
)
