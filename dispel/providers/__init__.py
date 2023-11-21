"""A module containing functionality specific to providers."""
from dispel.data.core import Reading
from dispel.processing.data_trace import DataTrace
from dispel.providers.registry import get_processing_function


def auto_process(reading: Reading, **kwargs) -> DataTrace:
    """Process measures automatically for readings."""
    code = str(reading.evaluation.id)
    return get_processing_function(code, type(reading))(reading, **kwargs)
