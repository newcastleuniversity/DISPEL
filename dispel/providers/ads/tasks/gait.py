"""AdS specific gait processing."""

from dispel.providers.ads.data import ADSReading
from dispel.providers.generic.tasks.gait.steps import TASK_NAME, GaitSteps
from dispel.providers.registry import process_factory

process_6mwt = process_factory(
    task_name=TASK_NAME,
    steps=GaitSteps(),
    codes="6mwt",
    supported_type=ADSReading,
)
