"""Konectom mood scale processing module."""
from typing import List

from dispel.data.values import AbbreviatedValue as AV
from dispel.processing import ProcessingStep
from dispel.processing.level import ProcessingStepGroup
from dispel.providers.generic.surveys import (
    ConcatenateSurveyLevels,
    ExtractSurveyResponseMeasures,
    SurveyQuestion,
)
from dispel.providers.registry import process_factory

MOOD_TASK_NAME = AV("Mood scale", "MOOD")

RESPONSES_MOOD = {1: "Excellent", 2: "Good", 3: "Average", 4: "Poor", 5: "Very Poor"}

QUESTIONS_MOOD = [
    SurveyQuestion("mood", "mood", "psy", "How is your mood today?", RESPONSES_MOOD),
    SurveyQuestion(
        "physicalState",
        "physical state",
        "phys",
        "In general, how is your physical state?",
        RESPONSES_MOOD,
    ),
]

MOOD_STEPS: List[ProcessingStep] = [
    ConcatenateSurveyLevels("idMoodscale"),
    ProcessingStepGroup(
        [ExtractSurveyResponseMeasures(question) for question in QUESTIONS_MOOD],
        task_name=MOOD_TASK_NAME,
    ),
]

process_mood = process_factory(
    task_name=MOOD_TASK_NAME,
    steps=MOOD_STEPS,
    codes=("mood", "mood-test", "mood-activity"),
)
