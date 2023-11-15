"""Tests for :mod:`dispel.data.validators`."""

import pytest

from dispel.data.validators import RangeValidator, SetValidator, ValidationException


@pytest.mark.parametrize("lower,upper", [(6, 3), (10, 10), (3, -2), (1, 1.0)])
def test_range_validator_input_order(lower, upper):
    """Test that errors are risen when input length is not increasing."""
    with pytest.raises(ValueError):
        RangeValidator(lower_bound=lower, upper_bound=upper)


def test_range_validator_at_least_one_bound():
    """Test that a range validator has to have at least one bound."""
    with pytest.raises(ValueError):
        RangeValidator()


def test_range_validator():
    """Test the range validator call."""
    validator1 = RangeValidator(lower_bound=2, upper_bound=3)
    validator2 = RangeValidator(lower_bound=-5, upper_bound=8)
    validator3 = RangeValidator(lower_bound=10)
    validator4 = RangeValidator(upper_bound=8)

    # Verify that no errors are raised
    validator1(2.5)
    validator2(0)
    validator3(12)
    validator4(-4)

    # Verify that value errors are raised
    with pytest.raises(ValidationException):
        validator1(1)

    with pytest.raises(ValidationException):
        validator2(9)

    with pytest.raises(ValidationException):
        validator3(0)

    with pytest.raises(ValidationException):
        validator4(23)

    # verify hashing
    assert hash(validator1) == hash(validator1)
    assert hash(validator1) != hash(validator2)


def test_range_validator_boundaries():
    """Test the range validator at boundaries."""
    validator1 = RangeValidator(lower_bound=0, include_lower=False)
    with pytest.raises(ValidationException):
        validator1(0)

    validator2 = RangeValidator(upper_bound=1, include_upper=False)
    with pytest.raises(ValidationException):
        validator2(1)

    validator3 = RangeValidator(lower_bound=0)
    validator3(0)

    validator4 = RangeValidator(upper_bound=1)
    validator4(1)


def test_set_validator_list():
    """Test the set validator with list as values."""
    validator = SetValidator([1, 2, 3])

    validator(1)
    validator(2)
    validator(3)

    with pytest.raises(ValidationException):
        validator(4)

    with pytest.raises(ValidationException):
        validator("some value")

    # verify hashing
    assert hash(validator) == hash(validator)


def test_set_validator_dict():
    """Test the set validator with labels and values."""
    validator = SetValidator({1: "a", 2: "b", 3: "c"})

    validator(1)
    validator(2)
    validator(3)

    with pytest.raises(ValidationException):
        validator(0)

    with pytest.raises(ValidationException):
        validator("a")

    # verify hashing
    assert hash(validator) == hash(validator)


def test_set_validator_at_least_list_or_dict():
    """Test that the set validator gets at least a list or dictionary."""
    with pytest.raises(ValueError):
        SetValidator(5)


def test_set_validator_list_unique():
    """Test that the list values have to be unique."""
    with pytest.raises(ValueError):
        SetValidator([1, 1, 2])


def test_set_validator_repr():
    """Test the set validator representation."""
    validator1 = SetValidator([1, 2, 3])
    assert repr(validator1) == "<SetValidator: {1, 2, 3}>"

    validator2 = SetValidator({1: "a", 2: "b", 3: "c"})
    assert repr(validator2) == "<SetValidator: {1: 'a', 2: 'b', 3: 'c'}>"


def test_set_validator_get_label_no_labels():
    """Test that the set validator returns None if no labels exist."""
    validator = SetValidator([1, 2, 3])
    assert validator.get_label(1) is None


def test_set_validator_get_label():
    """Test that the set validator returns the correct label."""
    validator = SetValidator({1: "one", 2: "two"})
    assert validator.get_label(1) == "one"


def test_set_validator_get_label_not_in_allowed_values():
    """Test that the set validator raises an error if asked for non-allowed."""
    validator = SetValidator({1: "one", 2: "two", 3: "three"})
    with pytest.raises(KeyError):
        validator.get_label(0)
