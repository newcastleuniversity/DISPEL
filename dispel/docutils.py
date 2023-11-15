"""Utility functions for feature definition documentation."""
from itertools import starmap
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from dispel.data.features import FeatureValueDefinition
from dispel.data.validators import RangeValidator, SetValidator
from dispel.data.values import ValueDefinition
from dispel.processing.core import ProcessingStep
from dispel.processing.extract import FeatureDefinitionMixin
from dispel.processing.trace import collect_feature_value_definitions
from dispel.providers.registry import PROCESSING_STEPS


def convert_feature_value_definition_to_dict(
    step: ProcessingStep, definition: ValueDefinition
) -> Dict[str, Optional[str]]:
    """Convert a feature value definition into a dictionary representation.

    Parameters
    ----------
    step
        The processing step that produced the definition.
    definition
        The definition to be converted

    Returns
    -------
    Dict[str, Optional[str]]
        A dictionary representation of the definition. Note that if the validator is a
        range validator it will populate the ``values_min`` and ``values_max``
        attributes based on the lower and upper bound, respectively. All other
        attributes of the object are mapped one-to-one.

    """
    # expand on validator
    values_min = None
    values_max = None
    values_in = None
    validator = definition.validator
    if isinstance(validator, RangeValidator):
        values_min = validator.lower_bound
        values_max = validator.upper_bound
    elif isinstance(validator, SetValidator):
        if validator.labels:
            values_in = validator.labels
        else:
            values_in = validator.allowed_values

    # Additional parameters
    feature_definition_kwargs = {}
    if isinstance(definition, FeatureValueDefinition):
        feature_definition_kwargs = {
            "task_name": definition.task_name,
            "feature_name": definition.feature_name,
            **{
                f"modality_{i}": str(m)
                for i, m in enumerate(definition.modalities or [])
            },
            "aggregation": definition.aggregation,
        }

    return {
        "id": str(definition.id),
        "name": str(definition.name),
        "description": definition.description,
        "unit": definition.unit,
        "data_type": definition.data_type,
        "values_min": values_min,
        "values_max": values_max,
        "values_in": values_in,
        "produced_by": repr(step),
        **feature_definition_kwargs,
    }


def feature_value_definitions_to_data_frame(
    definitions: Iterable[Tuple[FeatureDefinitionMixin, ValueDefinition]]
) -> pd.DataFrame:
    """Convert a list of feature value definitions into a data frame.

    Parameters
    ----------
    definitions
        A list of tuples of processing steps and the produced feature value definitions
        to be converted into a data frame representation.

    Returns
    -------
    pandas.DataFrame
        The definitions as data frame. See also
        :func:`convert_feature_value_definition_to_dict` on the specific columns
        provided.

    """
    return pd.DataFrame(starmap(convert_feature_value_definition_to_dict, definitions))


def get_feature_value_definitions_data_frame() -> pd.DataFrame:
    """Get a data frame of all available features in the library.

    Returns
    -------
    pandas.DataFrame
        A data frame containing all feature value definitions. See also
        :func:`feature_value_definitions_to_data_frame`.
    """
    instrument_data = []

    for steps in PROCESSING_STEPS.values():
        if isinstance(steps, ProcessingStep):
            steps = [steps]
        definitions = collect_feature_value_definitions(steps)
        data = feature_value_definitions_to_data_frame(definitions)
        instrument_data.append(data)

    return pd.concat(instrument_data, ignore_index=True)
