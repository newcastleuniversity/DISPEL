"""Tests for :mod:`dispel.data.levels`."""

from copy import deepcopy
from functools import partial

import pandas as pd
import pytest

from dispel.data.levels import Level, LevelId, Modalities
from dispel.data.raw import (
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.io.raw import generate_raw_data_set

generate_raw_data_set_ = partial(generate_raw_data_set, columns=["a", "b"])

EXAMPLE_LEVEL_1 = Level(
    id_="level_1",
    start="now",
    end="now",
    raw_data_sets=[
        generate_raw_data_set_("data-set-1"),
        generate_raw_data_set_("data-set-2"),
    ],
)

EXAMPLE_LEVEL_2 = Level(
    id_="level_2", start=1, end=3, raw_data_sets=[generate_raw_data_set_("data-set-3")]
)

EXAMPLE_LEVELS = [EXAMPLE_LEVEL_1, EXAMPLE_LEVEL_2]


def test_level_id():
    """Test the class :class:`~dispel.data.core.LevelId`."""
    level_std = LevelId("digit_to_symbol")
    assert isinstance(level_std, LevelId)
    assert level_std.id == "digit_to_symbol"

    level_pinch = LevelId(["right", "small"])
    assert isinstance(level_pinch, LevelId)
    assert level_pinch.id == "right-small"


def test_modalities():
    """Test the class :class:`~dispel.data.core.Modalities`."""
    # TODO: create actual useful test case
    modalities = Modalities()
    assert isinstance(modalities, Modalities)


@pytest.fixture
def example_level():
    """Get a fixture for an example level."""
    return deepcopy(EXAMPLE_LEVEL_1)


def test_level(example_level):
    """Test the class :class:`~dispel.data.core.Level`."""
    level = example_level
    assert isinstance(level, Level)

    # initialize a RawDataSet
    data_set = RawDataSet(
        RawDataSetDefinition(
            "data-set",
            RawDataSetSource("example"),
            [RawDataValueDefinition("bar", "baz")],
            True,
        ),
        pd.DataFrame(dict(bar=[0, 1, 2, 3])),
    )
    # set RawDataSet into level
    level.set(data_set)

    # get data set from level
    res = level.get_raw_data_set("data-set")

    assert res == data_set
    assert res.data.equals(pd.DataFrame(dict(bar=[0, 1, 2, 3])))

    with pytest.raises(ValueError):
        level.set(
            RawDataSet(
                RawDataSetDefinition(
                    "data-set",
                    RawDataSetSource("example"),
                    [RawDataValueDefinition("baro", "baz")],
                    True,
                ),
                pd.DataFrame(dict(bar=[0, 1, 2, 3])),
            )
        )

    level.set(data_set, concatenate=True)
    res = level.get_raw_data_set("data-set")

    assert res.definition == data_set.definition
    assert res.data.equals(
        pd.DataFrame(dict(bar=[0, 1, 2, 3, 0, 1, 2, 3]), index=[0, 1, 2, 3, 0, 1, 2, 3])
    )
