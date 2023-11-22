"""Tests for :mod:`dispel.data.values`."""
from unittest.mock import Mock

import pytest

from dispel.data.values import (
    DefinitionId,
    Value,
    ValueDefinition,
    ValueDefinitionPrototype,
    ValueSet,
)


def test_definition_id_eq():
    """Test definition ids equal both for other definitions and strings."""
    definition_id1 = DefinitionId("some-id")
    definition_id2 = DefinitionId("some-id")
    definition_id3 = DefinitionId("other-id")

    assert definition_id1 == definition_id2
    assert definition_id1 != definition_id3
    assert definition_id1 == "some-id"


def test_definition_id_from_str():
    """Test creation of definitions from strings."""
    definition_id = DefinitionId.from_str("some-id")
    assert isinstance(definition_id, DefinitionId)
    assert definition_id == "some-id"


def test_value_definition_non_callable_validator():
    """Test that the validator is callable for value definitions."""
    with pytest.raises(TypeError):
        ValueDefinition("sample", "Sample", validator=5)


def test_value_definition_wrap_definition_id():
    """Test that value definitions wrap the passed id into a definition id."""
    definition1 = ValueDefinition("some-id", "Some name")
    assert isinstance(definition1.id, DefinitionId)

    definition2 = ValueDefinition(DefinitionId("other-id"), "Other name")
    assert isinstance(definition2.id, DefinitionId)


@pytest.fixture
def value_definition_parameter_example():
    """Get a fixture for value definition parameters."""

    def _validator(_value):
        pass

    parameters = dict(
        id_="a",
        name="a",
        unit="s",
        description="some description",
        data_type="int64",
        validator=_validator,
    )

    return parameters


def test_value_definition_eq(value_definition_parameter_example):
    """Test that eq method works correctly for valued definitions."""
    definition1 = ValueDefinition(**value_definition_parameter_example)
    definition2 = ValueDefinition(**value_definition_parameter_example)

    assert definition1 == definition2

    for key in value_definition_parameter_example:
        # test alteration of any value
        mod_params = value_definition_parameter_example.copy()
        if key != "validator":
            mod_params[key] += "-mod"
        else:
            mod_params[key] = lambda x: None

        assert definition1 != ValueDefinition(**mod_params)

        if key in ("id_", "name"):
            continue

        # test that any missing parameter causes difference
        sub_params = value_definition_parameter_example.copy()
        sub_params.pop(key)

        assert definition1 != ValueDefinition(**sub_params)


@pytest.mark.parametrize(
    "update_params",
    [
        {"id_": "b"},
        {"name": "b"},
        {"unit": "ms"},
        {"description": "another"},
        {"name": "b", "unit": "ms"},
        {"data_type": "int32"},
    ],
)
def test_value_definition_derive(value_definition_parameter_example, update_params):
    """Test deriving value definitions."""
    original = ValueDefinition(**value_definition_parameter_example)
    updated_params = value_definition_parameter_example.copy()
    updated_params.update(update_params)

    res = original.derive(**updated_params)

    assert isinstance(res, ValueDefinition)
    assert res == ValueDefinition(**updated_params)


def test_value_definition_derive_unknown(value_definition_parameter_example):
    """Test that only known parameters can be updated."""
    definition = ValueDefinition(**value_definition_parameter_example)

    with pytest.raises(ValueError) as exc_info:
        definition.derive(id_="b", ref="foo", desc="bar")

    assert exc_info.type is ValueError
    assert (
        exc_info.value.args[0] == "The following parameters are unknown "
        "to the constructor: desc, ref"
    )


def test_value_definition_prototype_create_definition_fill_arguments():
    """Test filling arguments of definition class from prototype."""
    prototype = ValueDefinitionPrototype(name="Some name")
    definition = prototype.create_definition(id_="some-id")

    assert isinstance(definition, ValueDefinition)
    assert definition.id == "some-id"
    assert definition.name == "Some name"


def test_value_definition_prototype_create_definition_fill_placeholders():
    """Test filling placeholders in arguments of prototypes."""
    prototype = ValueDefinitionPrototype(id_="{ph}-id", name="{ph} name")
    definition = prototype.create_definition(ph="some")

    assert isinstance(definition, ValueDefinition)
    assert definition.id == "some-id"
    assert definition.name == "some name"


def test_value_definition_prototype_create_definition_custom_class():
    """Test creation of custom definition class from prototype."""

    class _MyValueDefinition(ValueDefinition):
        pass

    prototype = ValueDefinitionPrototype(
        cls=_MyValueDefinition, id_="{ph}", name="{ph}"
    )
    definition = prototype.create_definition(ph="placeholder")

    assert isinstance(definition, _MyValueDefinition)
    assert definition.id == "placeholder"
    assert definition.name == "placeholder"


def test_value_definition_prototype_create_definitions():
    """Test creation of multiple definitions from prototype."""
    prototype = ValueDefinitionPrototype(id_="id-{enum}", name="Name {enum}")
    definitions = prototype.create_definitions({"enum": i} for i in range(5))

    assert definitions is not None
    assert isinstance(definitions, list)
    assert len(definitions) == 5
    assert all(map(lambda s: isinstance(s, ValueDefinition), definitions))

    for i in range(5):
        assert definitions[i].id == f"id-{i}"
        assert definitions[i].name == f"Name {i}"


def test_value_flag_upon_creation():
    """Test value flag upon value creation."""
    validator = Mock()
    definition = ValueDefinition("some-id", "Some name", validator=validator)
    Value(definition, 5)

    validator.assert_called_once_with(5)


def test_value_id():
    """Test that the definition id is accessible from values."""
    value = Value(ValueDefinition("some-id", "Some name"), 5)
    assert isinstance(value.id, DefinitionId)
    assert value.id == "some-id"


def test_value_set_set():
    """Test setting values to value sets."""
    value_set = ValueSet()
    definition1 = ValueDefinition("some-id", "Some name")
    value = Value(definition1, 5)

    # set value as value
    value_set.set(value)
    assert definition1 in value_set
    assert value_set.get_raw_value(definition1) == 5

    # set value as definition and value
    definition2 = ValueDefinition("other-id", "Other name")
    value_set.set(6, definition2)
    assert definition2 in value_set
    assert value_set.get_raw_value(definition2) == 6

    # check missing definition
    with pytest.raises(ValueError):
        value_set.set(5)

    # check overwrite exception
    with pytest.raises(ValueError):
        value_set.set(value)

    # check overwrite
    value_set.set(value, overwrite=True)
    value_set.set(7, definition1, overwrite=True)
    assert value_set.get_raw_value(definition1) == 7


def test_value_set_set_values():
    """Test setting multiple values to value sets."""
    values = [Value(ValueDefinition(f"id-{i}", f"name {i}"), i) for i in range(5)]

    # via setter and value classes
    value_set1 = ValueSet()
    value_set1.set_values(values)

    assert len(value_set1) == 5
    assert all(map(lambda v: v.id in value_set1, values))

    # via setter and value and definitions
    value_set2 = ValueSet()
    value_set2.set_values([v.value for v in values], [v.definition for v in values])

    assert len(value_set2) == 5
    assert all(map(lambda v: v.id in value_set2, values))

    # via constructor and values
    value_set3 = ValueSet(values)

    assert len(value_set3) == 5
    assert all(map(lambda v: v.id in value_set3, values))

    # via constructor and value and definition
    value_set4 = ValueSet([v.value for v in values], [v.definition for v in values])

    assert len(value_set4) == 5
    assert all(map(lambda v: v.id in value_set4, values))


def test_value_set_set_values_mismatching_definitions():
    """Test that non-equal count of values and definitions raises error."""
    with pytest.raises(ValueError):
        ValueSet([1], [ValueDefinition("a", "a"), ValueDefinition("b", "b")])


def test_value_set_has_value():
    """Test has_value for value sets."""
    definition1 = ValueDefinition("a", "a")
    definition2 = ValueDefinition("b", "b")
    definition3 = ValueDefinition("c", "c")
    value_set = ValueSet(
        [
            Value(definition1, 5),
            Value(definition2, 6),
        ]
    )

    # test has_value function
    assert value_set.has_value(definition1)
    assert value_set.has_value(definition1.id)
    assert value_set.has_value("a")

    assert not value_set.has_value("c")
    assert not value_set.has_value(definition3.id)
    assert not value_set.has_value(definition3)

    # test __in__ magic method
    assert definition1 in value_set
    assert definition1.id in value_set
    assert "a" in value_set

    assert "c" not in value_set
    assert definition3.id not in value_set
    assert definition3 not in value_set


def test_value_set_get_value():
    """Test getting values from a value set."""
    value1 = Value(ValueDefinition("a", "a"), 5)
    value2 = Value(ValueDefinition("b", "b"), 6)
    value_set = ValueSet([value1, value2])

    # test get function
    assert value_set.get(value1.definition) == value1
    assert value_set.get(value2.id) == value2
    assert value_set.get("b") == value2

    # test __getitem__ magic method
    assert value_set[value2.definition] == value2
    assert value_set[value1.id] == value1
    assert value_set["a"] == value1

    # test raising errors for non-existing keys
    with pytest.raises(KeyError):
        value_set.get("c")
    with pytest.raises(KeyError):
        value_set.get(ValueDefinition("c", "c"))
    with pytest.raises(KeyError):
        value_set.get(DefinitionId("c"))

    # get raw values directly
    assert value_set.get_raw_value("a") == 5
    assert value_set.get_raw_value(value2.id) == 6
    assert value_set.get_raw_value(value2.definition) == 6


def test_value_set_get_definition():
    """Test getting the definition of a value from a value set."""
    definition = ValueDefinition("a", "a")
    value_set = ValueSet([Value(definition, 7)])

    assert value_set.get_definition("a") == definition
    assert value_set.get_definition(DefinitionId("a")) == definition


@pytest.fixture
def example_values():
    """Get example values list for test cases."""
    return [Value(ValueDefinition(f"id-{i}", f"name {i}"), i) for i in range(3)]


@pytest.fixture
def example_value_set(example_values):
    """Get example value set for test cases."""
    return ValueSet(example_values)


def test_value_set_values(example_values, example_value_set):
    """Test that values can be retrieved from value sets."""
    assert list(example_value_set.values()) == example_values


def test_value_set_ids(example_values, example_value_set):
    """Test that ids can be retrieved from value sets."""
    expected_ids = [x.id for x in example_values]
    assert list(example_value_set.ids()) == expected_ids


def test_value_set_definitions(example_values, example_value_set):
    """Test that definitions can be retrieved from value sets."""
    expected_definitions = [x.definition for x in example_values]
    assert list(example_value_set.definitions()) == expected_definitions


def test_value_set_len(example_value_set):
    """Test that right number of values for value sets is returned."""
    assert len(ValueSet()) == 0
    assert len(example_value_set) == 3


def test_value_set_combine():
    """Test combining different value sets."""
    value_set1 = ValueSet([Value(ValueDefinition("a", "a"), 1)])
    value_set2 = ValueSet([Value(ValueDefinition("b", "b"), 5)])
    value_set3 = ValueSet([Value(ValueDefinition("a", "a"), 6)])

    # combine via addition
    add_value_set = value_set1 + value_set2
    assert len(add_value_set) == 2
    assert "a" in add_value_set
    assert "b" in add_value_set

    with pytest.raises(ValueError):
        value_set1 + value_set3  # pylint: disable=pointless-statement

    # combine via union
    union_value_set1 = value_set1 | value_set2
    assert len(union_value_set1) == 2
    assert "a" in union_value_set1
    assert "b" in union_value_set1

    # ensure union overwrites values
    union_value_set2 = value_set1 | value_set3
    assert len(union_value_set2) == 1
    assert "a" in union_value_set2
    assert union_value_set2["a"].value == 6

    # test update via addition
    value_set1 += value_set2

    assert len(value_set1) == 2
    assert "b" in value_set1

    # test update via union
    value_set3 |= value_set1
    assert len(value_set3) == 2
    assert "a" in value_set1
    assert value_set3["a"].value == 1


def test_precision():
    """Test precision of a value set."""
    # test scenario where various precision values are set
    values = [
        Value(ValueDefinition(f"id-{i}", f"name {i}", precision=i), i + 0.333333)
        for i in range(5)
    ]

    assert values[0].value == 0
    assert values[1].value == 1.3
    assert values[2].value == 2.33
    assert values[3].value == 3.333
    assert values[4].value == 4.3333

    assert values[0].raw_value == 0.333333
    assert values[1].raw_value == 1.333333
    assert values[2].raw_value == 2.333333
    assert values[3].raw_value == 3.333333
    assert values[4].raw_value == 4.333333

    # test scenario where no precision is set
    val = 1.5555555
    no_precision_value = Value(ValueDefinition("id1", "name 1"), val)
    assert no_precision_value.raw_value == val
    assert no_precision_value.value == val

    # test scenario where negative precision is set
    vals = [1333.33, 2555.55]
    values = [
        Value(ValueDefinition(f"id {i}", f"name {i}", precision=-2), val)
        for i, val in enumerate(vals)
    ]

    assert values[0].raw_value == vals[0]
    assert values[1].raw_value == vals[1]

    assert values[0].value == 1300
    assert values[1].value == 2600
