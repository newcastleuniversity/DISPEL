"""Processing package for the cps."""
from dispel.data.values import AbbreviatedValue as AV
from dispel.providers.generic.tasks.cps.steps import CPSProcessingStepGroup
from dispel.providers.registry import process_factory

TASK_NAME = AV("Cognitive Processing Speed test", "CPS")

STEPS = [CPSProcessingStepGroup()]

process_cps = process_factory(
    task_name=TASK_NAME,
    steps=STEPS,
    codes=("cps", "cps-test", "cps-activity"),
)
