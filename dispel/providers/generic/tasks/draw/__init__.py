"""Processing package for the drawing test."""
from dispel.providers.generic.tasks.draw.steps import STEPS, TASK_NAME
from dispel.providers.registry import process_factory

process_draw = process_factory(
    task_name=TASK_NAME,
    steps=STEPS,
    codes=("drawing", "drawing-activity"),
)
