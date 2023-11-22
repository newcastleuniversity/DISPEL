"""Processing package for the pinch test."""
from dispel.providers.generic.tasks.pinch.steps import STEPS, TASK_NAME
from dispel.providers.registry import process_factory

process_pinch = process_factory(
    task_name=TASK_NAME,
    steps=STEPS,
    codes=("pinch", "pinch-activity"),
)
