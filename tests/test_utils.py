"""Test functions for :mod:`dispel.utils`."""
import pytest

from dispel.utils import camel_to_snake_case, plural, to_camel_case


def test_camel_to_snake_case():
    """Test camel to snake case conversion."""
    assert camel_to_snake_case("someCamel") == "some_camel"
    assert camel_to_snake_case("someCAMel") == "some_camel"
    assert camel_to_snake_case("some") == "some"
    assert camel_to_snake_case("SomeCamel") == "some_camel"


def test_to_camel_case():
    """Test the formatting of a string to a camel case string."""
    assert to_camel_case("not_a_camel") == "notACamel"
    assert to_camel_case("I want_to Be_a_camel") == "iWantToBeACamel"


@pytest.mark.parametrize(
    "single, count, multiple, expected",
    [
        ("foo", 0, None, "0 foos"),
        ("bar", 4, None, "4 bars"),
        ("baz", 1, None, "1 baz"),
        ("tooth", 3, "teeth", "3 teeth"),
        ("tooth", 1, "teeth", "1 tooth"),
    ],
)
def test_plural(single, count, multiple, expected):
    """Test wrapping word in plural with count."""
    assert plural(single, count, multiple) == expected
