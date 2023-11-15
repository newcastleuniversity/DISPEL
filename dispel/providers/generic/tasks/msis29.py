"""Multiple Sclerosis Impact Scale 29 (MSIS-29) processing module."""

from typing import List

import pandas as pd

from dispel.data.features import FeatureValueDefinitionPrototype
from dispel.data.validators import RangeValidator
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing import ProcessingStep
from dispel.processing.extract import ExtractStep
from dispel.processing.level import ProcessingStepGroup
from dispel.providers.generic.surveys import (
    RAW_DATA_SET_DEFINITION,
    SURVEY_RESPONSES_LEVEL_ID,
    ConcatenateSurveyLevels,
    ExtractSurveyResponseFeatures,
    SurveyQuestion,
)
from dispel.providers.registry import process_factory

MSIS29_TASK_NAME = AV("Multiple Sclerosis Impact Scale 29", "MSIS29")


class ExtractMsis29Scale(ExtractStep):
    """Extract the summary score for the MSIS-29 for a specific range."""

    def __init__(self, questions, definition_kwargs):
        self.questions = questions
        self.definition_kwargs = definition_kwargs

        def _summary(data: pd.DataFrame) -> int:
            final_response = data.groupby("question_id").last()
            mask = final_response.index.isin(self.questions)
            return final_response[mask].response.sum()

        super().__init__(
            RAW_DATA_SET_DEFINITION.id,
            _summary,
            FeatureValueDefinitionPrototype(**self.definition_kwargs, data_type="int"),
            level_filter=SURVEY_RESPONSES_LEVEL_ID,
        )


_MSIS29_PREAMBLE_MS = (
    "In the past two weeks how much has your MS limited your ability to"
)
_MSIS29_PREAMBLE_BOTHER = "In the past two weeks, how much have you been bothered by"
_RAW_QUESTIONS_MSIS_29 = [
    (
        "limits for physically demanding tasks",
        f"{_MSIS29_PREAMBLE_MS} do physically demanding tasks",
    ),
    (
        "grip things tightly",
        f"{_MSIS29_PREAMBLE_MS} grip things tightly (e.g. turning on taps)",
    ),
    ("ability carry things", f"{_MSIS29_PREAMBLE_MS} carry things"),
    ("balance problems", f"{_MSIS29_PREAMBLE_BOTHER} problems with your balance"),
    (
        "difficulties moving about indoors",
        f"{_MSIS29_PREAMBLE_BOTHER} difficulties moving about indoors",
    ),
    ("being clumsy", f"{_MSIS29_PREAMBLE_BOTHER} being clumsy"),
    ("stiffness", f"{_MSIS29_PREAMBLE_BOTHER} stiffness"),
    ("heavy arms and/or legs", f"{_MSIS29_PREAMBLE_BOTHER} heavy arms and/or legs"),
    (
        "tremor of arms or legs",
        f"{_MSIS29_PREAMBLE_BOTHER} tremors of your arms or legs",
    ),
    ("spasms in limbs", f"{_MSIS29_PREAMBLE_BOTHER} spasms in your limbs"),
    (
        "body not doing what wanted",
        f"{_MSIS29_PREAMBLE_BOTHER} your body not doing what you want it to do",
    ),
    (
        "depending on others to do things",
        f"{_MSIS29_PREAMBLE_BOTHER} having to depend on others to do things for "
        f"you",
    ),
    (
        "limitations in social and leisure activities at home",
        f"{_MSIS29_PREAMBLE_BOTHER} limitations in your social and leisure "
        f"activities at home",
    ),
    (
        "being stuck at home more than desired",
        f"{_MSIS29_PREAMBLE_BOTHER} being stuck at home more than you would like "
        f"to be",
    ),
    (
        "difficulties using hands in everyday tasks",
        f"{_MSIS29_PREAMBLE_BOTHER} difficulties using your hands in everyday "
        f"tasks",
    ),
    (
        "cut down the amount of time spent on work or other",
        f"{_MSIS29_PREAMBLE_BOTHER} having to cut down the amount of time you "
        f"spent on work or other daily activities",
    ),
    (
        "problems using transport ",
        f"{_MSIS29_PREAMBLE_BOTHER} problems using transport (e.g. car, bus, "
        f"train, taxi, etc.)",
    ),
    (
        "taking longer to do things",
        f"{_MSIS29_PREAMBLE_BOTHER} taking longer to do things",
    ),
    (
        "difficulty doing this spontaneously",
        f"{_MSIS29_PREAMBLE_BOTHER} difficulty doing things spontaneously (e.g. "
        f"going out on the spur of the moment)",
    ),
    (
        "needing to go to the toilet urgently",
        f"{_MSIS29_PREAMBLE_BOTHER} needing to go to the toilet urgently",
    ),
    ("feeling unwell", f"{_MSIS29_PREAMBLE_BOTHER} feeling unwell"),
    ("problems sleeping", f"{_MSIS29_PREAMBLE_BOTHER} problems sleeping"),
    (
        "feeling mentally fatigued",
        f"{_MSIS29_PREAMBLE_BOTHER} feeling mentally fatigued",
    ),
    ("worries related to MS", f"{_MSIS29_PREAMBLE_BOTHER} worries related to your MS"),
    ("feeling anxious or tense", f"{_MSIS29_PREAMBLE_BOTHER} feeling anxious or tense"),
    (
        "feeling irritable, impatient, short tempered",
        f"{_MSIS29_PREAMBLE_BOTHER} feeling irritable, impatient, or short "
        f"tempered",
    ),
    ("problems concentrating", f"{_MSIS29_PREAMBLE_BOTHER} problems concentrating"),
    ("lack of confidence", f"{_MSIS29_PREAMBLE_BOTHER} lack of confidence"),
    ("feeling depressed", f"{_MSIS29_PREAMBLE_BOTHER} feeling depressed"),
]

RESPONSES_MSIS_29 = {1: "Not at all", 2: "A little", 3: "Moderately", 4: "Extremely"}

QUESTIONS_MSIS_29 = [
    SurveyQuestion(str(i), label, f"{i + 1}", full, RESPONSES_MSIS_29)
    for i, (label, full) in enumerate(_RAW_QUESTIONS_MSIS_29)
]

_RAW_SCORES_MSIS_29 = [
    (
        range(0, 29),
        "total scale",
        "all",
        "Sum of all answers to the MSIS-29 questionnaire",
    ),
    (
        range(0, 20),
        "physical impact sub-scale",
        "phys",
        "Sum of answers to the MSIS-29 physical impact section including "
        "questions 1 to 20",
    ),
    (
        range(20, 29),
        "psychological impact sub-scale",
        "psy",
        "Sum of answers to the MSIS-29 psychological impact section including "
        "questions 21 to 29",
    ),
]

MSIS29_STEPS_SCORES: List[ProcessingStep] = [
    ExtractMsis29Scale(
        list(map(str, questions)),
        {
            "feature_name": AV(name, f"ans_{abbr}"),
            "description": description,
            "validator": RangeValidator(len(questions), len(questions) * 4),
        },
    )
    for questions, name, abbr, description in _RAW_SCORES_MSIS_29
]

MSIS29_STEPS_ANSWERS: List[ProcessingStep] = [
    ExtractSurveyResponseFeatures(question) for question in QUESTIONS_MSIS_29
]

MSIS29_STEPS: List[ProcessingStep] = [
    ConcatenateSurveyLevels("idMsis29"),
    ProcessingStepGroup(
        steps=MSIS29_STEPS_SCORES + MSIS29_STEPS_ANSWERS, task_name=MSIS29_TASK_NAME
    ),
]

process_msis29 = process_factory(
    task_name=MSIS29_TASK_NAME,
    steps=MSIS29_STEPS,
    codes=("msis29", "msis29-test", "msis29-activity"),
)
