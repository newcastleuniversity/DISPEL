"""Processing functionality."""
from typing import Iterable, List, Union

from dispel.data.core import Reading
from dispel.processing.core import ProcessingStep
from dispel.processing.data_trace import DataTrace
from dispel.utils import raise_multiple_errors

ProcessingStepsType = Union[ProcessingStep, Iterable[ProcessingStep]]


def process(reading: Reading, steps: ProcessingStepsType, **kwargs) -> DataTrace:
    r"""Perform processing steps on a reading.

    Parameters
    ----------
    reading
        The reading to be processed.
    steps
        An iterable of :class:`~dispel.processing.core.ProcessingStep`\ s to be applied to
        the ``reading`` or a single processing step to be applied.
    kwargs
        Additional named arguments are passed to the
        :meth:`~dispel.processing.core.ProcessingStep.process` function.

    Returns
    -------
    DataTrace
        The data trace corresponding to the provided reading after applying the
        processing steps.
    """
    if isinstance(steps, ProcessingStep):
        steps = [steps]

    accumulated_errors: List[Exception] = []
    trace = DataTrace.from_reading(reading)

    for step in steps:
        for output in step.process(reading, **kwargs):
            error = trace.populate(output)
            if error:
                accumulated_errors.append(error)
    raise_multiple_errors(accumulated_errors)
    return trace
