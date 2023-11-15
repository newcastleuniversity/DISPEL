"""Processing functionality for Finger Tapping (FT) task."""

from dispel.providers.digital_artefacts.data import DigitalArtefactsReading
from dispel.providers.generic.tasks.ft.steps import TASK_NAME, GenericFingerTappingSteps
from dispel.providers.registry import process_factory

process_ft = process_factory(
    task_name=TASK_NAME,
    steps=GenericFingerTappingSteps(),
    codes="fingertap-da",
    supported_type=DigitalArtefactsReading,
)
