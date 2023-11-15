"""A module containing value validators to ensure conformity of values.

The following validators are implemented:

- :class:`RangeValidator` to ensure values are within a given range
- :class:`SetValidator` to ensure values are one of a set of allowed values

As some validators are very common, hence there are constants for convenience:

- :attr:`GREATER_THAN_ZERO` to allow values greater or equal than zero

"""
import operator
from typing import Any, Dict, List, Optional, Union


class ValidationException(Exception):
    """An exception risen if a value didn't match the validator's expectations.

    Parameters
    ----------
    message
        Explanation of the error
    value
        The observed value
    validator
        The validator that rose the exception
    """

    def __init__(self, message, value, validator):
        super().__init__(message)
        self.value = value
        self.validator = validator


class RangeValidator:
    r"""A range validator.

    The range validator can be used to ensure a feature value is within a given range.
    It is specified via the ``validator`` argument when creating
    :class:`~dispel.data.values.ValueDefinition`\ s.

    Examples
    --------
    To create a range validator that allows all values between ``-4`` and ``7`` one can
    use it the following way:

    .. testsetup:: validator

        >>> from dispel.data.validators import RangeValidator, SetValidator, \
        ...     ValidationException

    .. doctest:: validator

        >>> validator = RangeValidator(-4, 7)
        >>> validator
        <RangeValidator: [-4, 7]>
        >>> validator(5)

    When called the range validator will raise an except only if the value is outside
    its range:

    .. doctest:: validator

        >>> validator(10)
        Traceback (most recent call last):
         ...
        dispel.data.validators.ValidationException: Value violated upper bound
        [-4, 7]: 10

    The range validator can also be used with just one side by specifying only the lower
    or upper bound:

    .. doctest:: validator

        >>> RangeValidator(lower_bound=-4)
        <RangeValidator: [-4, ∞]>
        >>> RangeValidator(upper_bound=7)
        <RangeValidator: [-∞, 7]>

    To exclude the boundaries in the range, one can set ``include_lower`` or
    ``include_upper`` to ``False``:

    .. doctest:: validator

        >>> validator = RangeValidator(lower_bound=0, include_lower=False)
        >>> validator
        <RangeValidator: (0, ∞]>
        >>> validator(0)
        Traceback (most recent call last):
         ...
        dispel.data.validators.ValidationException: Value violated lower bound
        (0, ∞]: 0

    Attributes
    ----------
    lower_bound
        The lower bound of the range validator.
    upper_bound
        The upper bound of the range validator.
    include_lower
        Include the lower boundary in the range check.
    include_upper
        Include the upper boundary in the range check.
    """

    def __init__(
        self,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        include_lower: bool = True,
        include_upper: bool = True,
    ):
        # Check that bounds are valid
        if lower_bound and upper_bound and lower_bound >= upper_bound:
            raise ValueError(
                "The validator range values have to be strictly increasing."
            )

        if lower_bound is None and upper_bound is None:
            raise ValueError("At least one bound needs to be specified.")

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.include_lower = include_lower
        self.include_upper = include_upper

    def _raise_exception(self, boundary: str, value: Any):
        msg = f"Value violated {boundary} bound {self.repr_boundaries()}: " f"{value}"
        raise ValidationException(msg, value, self)

    def __call__(self, value: Any):
        """Validate a value if it complies with bounds.

        Parameters
        ----------
        value
            The value to be validated
        """
        lower_op = operator.lt if self.include_lower else operator.le
        if self.lower_bound is not None and lower_op(value, self.lower_bound):
            self._raise_exception("lower", value)
        upper_op = operator.gt if self.include_upper else operator.ge
        if self.upper_bound is not None and upper_op(value, self.upper_bound):
            self._raise_exception("upper", value)

    def repr_boundaries(self):
        """Get a representation of the boundaries."""
        lower = "[" if self.include_lower else "("
        lower += "-∞" if self.lower_bound is None else str(self.lower_bound)
        upper = "∞" if self.upper_bound is None else str(self.upper_bound)
        upper += "]" if self.include_upper else ")"
        return f"{lower}, {upper}"

    def __repr__(self):
        return f"<RangeValidator: {self.repr_boundaries()}>"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.lower_bound == other.lower_bound
                and self.upper_bound == other.upper_bound
                and self.include_lower == other.include_lower
                and self.include_upper == other.include_upper
            )
        return False

    def __hash__(self) -> int:
        return hash(
            (
                self.lower_bound,
                self.upper_bound,
                self.include_lower,
                self.include_upper,
            )
        )


#: A validator that ensures values are greater or equal than zero
GREATER_THAN_ZERO = RangeValidator(lower_bound=0)

#: A validator that ensures values are between zero and one
BETWEEN_ZERO_AND_ONE = RangeValidator(lower_bound=0, upper_bound=1)

#: A validator that ensures values are between minus one and one
BETWEEN_MINUS_ONE_AND_ONE = RangeValidator(lower_bound=-1, upper_bound=1)


class SetValidator:
    r"""A set validator.

    The set validator can be used to ensure that a value is within a particular set of
    values. It is specified via the ``validator`` argument when creating
    :class:`~dispel.data.values.ValueDefinition`\ s.

    Examples
    --------
    The most common application of the :class:`SetValidator` is to validate survey
    responses and to provide additional labels for numerical responses. Assuming you
    have a survey that has possible responses ranging from ``1`` to ``4`` and you want
    to provide the respective labels to those responses you can achieve this in the
    following way:

    .. doctest:: validators

        >>> validator = SetValidator({
        ...    1: 'Not at all',
        ...    2: 'A little',
        ...    3: 'Moderately',
        ...    4: 'Extremely'
        ... })
        >>> validator
        <SetValidator: {1: 'Not at all', 2: 'A little', 3: 'Moderately', ...}>
        >>> validator(1)

    When calling the validator with a value not being part of the ``allowed_values`` the
    validator will raise an exception:

    .. doctest:: validator

        >>> validator(0)
        Traceback (most recent call last):
          ...
        dispel.data.validators.ValidationException: Value must be one of: {1, ...}

    If there are no labels for values one can simply pass a list of unique values to the
    validator:

    .. doctest:: validator

        >>> validator = SetValidator([1, 2, 3, 4])
        >>> validator
        <SetValidator: {1, 2, 3, 4}>

    Attributes
    ----------
    allowed_values
        The allowed values by the validator. Any value within this set will pass the
        validator.
    labels
        The labels for the allowed values. To get a label for a specific value consider
        using :meth:`~SetValidator.get_label`. Labels for values are specified by
        providing a dictionary with allowed values as keys and labels as values, e.g.

        >>> from dispel.data.validators import SetValidator
        >>> validator = SetValidator({1: 'label for 1', 2: 'label for two'})
    """

    def __init__(
        self,
        values: Union[List[Any], Dict[Any, str]],
    ):
        if isinstance(values, list):
            self.allowed_values = set(values)
            if len(self.allowed_values) != len(values):
                raise ValueError("Values must be unique")
            self.labels = None
        elif isinstance(values, dict):
            self.allowed_values = set(values.keys())
            self.labels = values.copy()
        else:
            raise ValueError(
                f"values must be a list of allowed values or dictionary of allowed "
                f"values as keys and values as labels. Got: {values}"
            )

    def __call__(self, value: Any):
        """Validate a value if it is within a set.

        Parameters
        ----------
        value
            The value to be validated

        Raises
        ------
        ValidationException
            If the value is not present in the set.
        """
        if value not in self.allowed_values:
            raise ValidationException(
                f"Value must be one of: {self.allowed_values}", value, self
            )

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.allowed_values == other.allowed_values
                and self.labels == other.labels
            )
        return False

    def __hash__(self) -> int:
        return hash(self.__dict__.values())

    def get_label(self, value: Any) -> Optional[str]:
        """Get the label for the specified value.

        Parameters
        ----------
        value
            The value for which to get the label for

        Returns
        -------
        str
            The label for the specified ``value``.

        Raises
        ------
        KeyError
            If the value is not part of the allowed values.
        """
        if not self.labels:
            return None

        if value not in self.allowed_values:
            raise KeyError(
                f"Value is not part of allowed values: {self.allowed_values}"
            )

        return self.labels[value]

    def __repr__(self):
        if self.labels:
            res = repr(self.labels)
        else:
            res = repr(self.allowed_values)
        return f"<SetValidator: {res}>"
