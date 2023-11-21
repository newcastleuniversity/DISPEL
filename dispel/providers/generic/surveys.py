"""Core functionality to process surveys."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from dispel.data.core import Reading
from dispel.data.levels import Level
from dispel.data.measures import MeasureValueDefinitionPrototype
from dispel.data.raw import (
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.core import ProcessingResult, ProcessingStep, ProcessResultType
from dispel.processing.extract import (
    DEFAULT_AGGREGATIONS,
    ExtractMultipleStep,
    ExtractStep,
)
from dispel.processing.level import (
    LevelFilterProcessingStepMixin,
    LevelFilterType,
    ProcessingStepGroup,
)

SURVEY_RESPONSES_LEVEL_ID = "survey_responses"
RAW_DATA_SET_DEFINITION = RawDataSetDefinition(
    "responses",
    RawDataSetSource("processing"),
    value_definitions_list=[
        RawDataValueDefinition("question_id", "Question index"),
        RawDataValueDefinition(
            "ts_question_displayed", "Time the question was shown to the subject"
        ),
        RawDataValueDefinition(
            "ts_question_hidden",
            "Time the question was hidden",
            description="The question is either hidden by going back to a "
            "previous question or providing a response.",
        ),
        RawDataValueDefinition("ts_response", "The time the response was provided"),
        RawDataValueDefinition("response", "The response provided by the subject"),
        RawDataValueDefinition(
            "response_time", "The time it took the subject to respond"
        ),
    ],
    is_computed=True,
)


def _level_to_question_response(level: Level, context_id: str) -> Dict[str, Any]:
    """Convert a level into a question dictionary."""
    response = {"ts_response": None, "response": None, "response_time": None}

    if level.has_raw_data_set("userInput"):
        data = level.get_raw_data_set("userInput").data.iloc[0]
        response["ts_response"] = data["tsAnswer"]
        response["response"] = data["answer"]
        response["response_time"] = data["tsAnswer"] - level.start

    return {
        "question_id": level.context.get_raw_value(context_id),
        "ts_question_displayed": level.start,
        "ts_question_hidden": level.end,
        **response,
    }


class ConcatenateSurveyLevels(LevelFilterProcessingStepMixin, ProcessingStep):
    """Concatenate individual survey question and response levels.

    This step creates a single raw data set out of individual levels to
    simplify the analysis of responses.

    Parameters
    ----------
    context_id
        The context id that identifies which question was posed to the user.
    level_filter
        An optional filter to only apply the concatenation to specific levels.
    """

    def __init__(self, context_id: str, level_filter: Optional[LevelFilterType] = None):
        self.context_id = context_id
        super().__init__(level_filter=level_filter)

    def get_levels(self, reading: Reading) -> Iterable[Level]:
        """Get the levels used for the concatenation."""
        return self.get_level_filter()(reading.levels)

    def get_raw_data_sets(self, reading: Reading) -> List[RawDataSet]:
        """Get the raw data sets used for the concatenation."""
        return [
            level.get_raw_data_set("userInput")
            for level in filter(
                lambda lvl: lvl.has_raw_data_set("userInput"), self.get_levels(reading)
            )
        ]

    def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
        """Concatenate individual levels."""
        entries = []
        for level in self.get_levels(reading):
            entries.append(_level_to_question_response(level, self.context_id))

        if entries:
            data_set = RawDataSet(
                RAW_DATA_SET_DEFINITION,
                pd.DataFrame(entries).sort_values("ts_question_hidden"),
            )
            new_level = Level(
                id_=SURVEY_RESPONSES_LEVEL_ID,
                start=reading.evaluation.start,
                end=reading.evaluation.end,
                raw_data_sets=[data_set],
            )
            yield ProcessingResult(
                step=self, sources=self.get_raw_data_sets(reading), result=new_level
            )


@dataclass
class SurveyQuestion:
    """A data class for survey questions."""

    id: str
    """The id that identifies the question.

    This is the value of the context variable specified for
    :class:`ConcatenateSurveyLevels`."""

    label: str
    """An abbreviated form of the question presented to the user.

    This is used as part of the measure name."""

    abbr: str
    """The abbreviation of the question.

    This is used as the part of the measure name"""

    full: str
    """The full description of the question posed to the user."""

    responses: Dict[Any, str]
    """A dictionary of responses and their labels."""

    @property
    def av(self):
        """Get the abbreviated value representation for the question."""
        return AV(f"{self.label} question", f"q_{self.abbr}")


class ExtractSurveyResponse(ExtractStep):
    """Extract the actual value of the survey question."""

    def __init__(self, question: SurveyQuestion):
        self.question = question

        def _get_last_response(data: pd.DataFrame) -> int:
            mask = data["question_id"] == question.id
            return data[mask]["response"].iloc[-1]

        super().__init__(
            RAW_DATA_SET_DEFINITION.id,
            _get_last_response,
            MeasureValueDefinitionPrototype(
                measure_name=AV("response", "res"),
                description="The final response to the question " f'"{question.full}".',
                data_type="int",
                # TODO: Fix the hashing crashing networkx including the validator lead
                #  to unstable hashing of the measure validator=SetValidator(
                #  question.responses)
            ),
            level_filter=SURVEY_RESPONSES_LEVEL_ID,
        )


class ExtractSurveyResponseTimesSummary(ExtractMultipleStep):
    """Extract measures on response times to questions."""

    def __init__(self, question: SurveyQuestion):
        self.question = question

        def _aggregate_response_time_factory(agg):
            def _summarize_response(data):
                mask = data["question_id"] == question.id
                return data[mask]["response_time"].dt.total_seconds().agg(agg)

            return _summarize_response

        functions = [
            {
                "func": _aggregate_response_time_factory(agg),
                "measure_name": AV("response time", "rt"),
                "aggregation": AV(agg_label, agg),
                "description": f"The {agg_label} response time for question "
                f'"{question.full}".',
            }
            for agg, agg_label in (("sum", "total"), *DEFAULT_AGGREGATIONS)
        ]

        def _total_time(data):
            mask = data["question_id"] == question.id
            return (
                data[mask]["ts_question_displayed"].min()
                - data[mask]["ts_response"].max()
            ).total_seconds()

        functions.append(
            {
                "func": _total_time,
                "measure_name": AV("time until response", "tur"),
                "description": f"The total time between first presenting "
                f'the question "{question.full}" until the final '
                f"response.",
            }
        )

        definition = MeasureValueDefinitionPrototype(
            unit="s",
        )

        super().__init__(
            RAW_DATA_SET_DEFINITION.id,
            functions,
            definition,
            level_filter=SURVEY_RESPONSES_LEVEL_ID,
        )


class ExtractSurveyResponseSummary(ExtractMultipleStep):
    """Extract measures on response changes."""

    def __init__(self, question: SurveyQuestion):
        self.question = question

        def _aggregate_response_factory(agg):
            def _summarize_response(data):
                mask = data["question_id"] == question.id
                return data[mask]["response"].diff().dropna().agg(agg)

            return _summarize_response

        functions = [
            {
                "func": _aggregate_response_factory(agg),
                "measure_name": AV("response change", "rc"),
                "aggregation": AV(agg_label, agg),
                "description": f"The {agg_label} response change for the "
                f'question "{question.full}".',
                "data_type": "float64",
            }
            for agg, agg_label in DEFAULT_AGGREGATIONS
        ]

        def _total_responses(data):
            mask = data["question_id"] == question.id
            return len(data[mask])

        functions.append(
            {
                "func": _total_responses,
                "measure_name": AV("response count", "count"),
                "description": "The number of responses that were given to "
                f'the question "{question.full}".',
                "data_type": "int",
            }
        )

        super().__init__(
            RAW_DATA_SET_DEFINITION.id,
            functions,
            MeasureValueDefinitionPrototype(),
            level_filter=SURVEY_RESPONSES_LEVEL_ID,
        )


class ExtractSurveyResponseMeasures(ProcessingStepGroup):
    """Extract measures for individual questions."""

    def __init__(self, question: SurveyQuestion):
        self.question = question

        super().__init__(
            [
                ExtractSurveyResponse(question),
                ExtractSurveyResponseTimesSummary(question),
                ExtractSurveyResponseSummary(question),
            ],
            question=question,
            modalities=[question.av],
        )
