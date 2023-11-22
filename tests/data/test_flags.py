"""Unit tests for :mod:`dispel.data.flags`."""
import operator
from copy import deepcopy

import pytest

from dispel.data.flags import (
    Flag,
    FlagAlreadyExists,
    FlagId,
    FlagMixIn,
    FlagNotFound,
    FlagType,
    WrappedResult,
)


@pytest.mark.parametrize("flag_type", [FlagType.TECHNICAL, FlagType.BEHAVIORAL])
def test_flag_id_creation(flag_type):
    """Test creation of flag id."""
    flag_id = FlagId(
        task_name="x", flag_name="y", flag_type=flag_type, flag_severity="deviation"
    )
    assert str(flag_id) == f"x-{flag_type}-deviation-y"


def test_flag_id_creation_invalid_type():
    """Test creation of flag id with invalid type."""
    with pytest.raises(KeyError):
        _ = FlagId(
            task_name="x", flag_name="y", flag_type="z", flag_severity="deviation"
        )


def test_flag_id_from_str():
    """Test flag id creation from string."""
    id_ = "x-technical-deviation-y"
    flag_id = FlagId.from_str(id_)

    assert isinstance(flag_id, FlagId)
    assert str(flag_id) == id_


@pytest.mark.parametrize(
    "id_, error",
    (
        ["", ValueError],
        ["x-y", ValueError],
        ["x-y-z", ValueError],
        ["x-y-deviation-z", KeyError],
        ["x-technical-y-z", KeyError],
    ),
)
def test_flag_id_from_str_invalid_format(id_, error):
    """Test flag id creation from string with invalid format."""
    with pytest.raises(error):
        _ = FlagId.from_str(id_)


def test_flag_creation():
    """Test :class:`dispel.data.flags.Flag`."""
    id_str = "x-technical-deviation-y"
    id_ = FlagId.from_str("x-technical-deviation-y")
    reason = "reason"

    flag1 = Flag(id_=id_, reason=reason)

    assert isinstance(flag1, Flag)
    assert flag1.id == id_
    assert flag1.reason == reason

    flag2 = Flag(id_=id_str, reason=reason)
    assert flag2 == flag1


def test_flag_creation_wrong_id():
    """Test flag creation with wrong id."""
    with pytest.raises(TypeError):
        _ = Flag(id_=1, reason="reason")


@pytest.fixture(scope="module")
def flags():
    """Create a fixture for multiple flags."""
    return [
        Flag("pinch-technical-deviation-ta", "reason 1"),
        Flag("pinch-behavioral-deviation-sp", "reason 2"),
        Flag("cps-behavioral-deviation-sp", "reason 3"),
        Flag("cps-technical-deviation-fc", "reason 4"),
        Flag("grip-technical-deviation-al", "reason 5"),
        Flag("grip-technical-deviation-al", "reason 6"),
    ]


class _TestEntity(FlagMixIn):
    """Test class for flag Mix-Ins."""


def test_flag_mix_in_creation(flags):
    """Test flag Mix in creation."""
    imi1 = _TestEntity()
    assert isinstance(imi1, FlagMixIn)
    assert imi1.is_valid

    (imi2 := _TestEntity()).add_flags(flags)
    assert isinstance(imi2, FlagMixIn)
    assert imi2.flag_count == 6


@pytest.fixture(scope="module")
def flag_mixin(flags):
    """Create fixture for flag mix in."""
    (imi := _TestEntity()).add_flags(flags)
    return imi


def test_get_flags(flag_mixin):
    """Test get flags from mix in."""
    id1 = "cps-behavioral-deviation-sp"
    inv1 = flag_mixin.get_flags(id1)[0]

    assert isinstance(inv1, Flag)
    assert str(inv1.id) == id1
    assert inv1.reason == "reason 3"

    id2 = FlagId.from_str("pinch-technical-deviation-ta")
    inv2 = flag_mixin.get_flags(id2)[0]

    invs = flag_mixin.get_flags()
    assert len(invs) == 6

    assert isinstance(inv2, Flag)
    assert inv2.id == id2
    assert inv2.reason == "reason 1"

    with pytest.raises(FlagNotFound):
        flag_mixin.get_flags("grip-technical-deviation-ta")


def test_has_flag(flag_mixin):
    """Test check if an flag is contained inside mix in."""
    id1 = "cps-behavioral-deviation-sp"
    assert flag_mixin.has_flag(id1)

    id2 = FlagId.from_str("cps-technical-deviation-ta")
    assert not flag_mixin.has_flag(id2)

    inv = Flag("pinch-technical-deviation-ta", "reason 1")
    assert flag_mixin.has_flag(inv)

    with pytest.raises(TypeError):
        flag_mixin.has_flag(None)


def test_add_flag(flag_mixin):
    """Test flag addition to the mix in."""
    imi = deepcopy(flag_mixin)
    flag = Flag("cps-technical-deviation-ta", "reason 6")

    assert not imi.has_flag(flag)
    imi.add_flag(flag)
    assert imi.has_flag(flag)
    assert imi.flag_count == 7

    with pytest.raises(FlagAlreadyExists):
        imi.add_flag(flag)


def test_add_flags(flag_mixin):
    """Test flags addition to the mix in."""
    imi = deepcopy(flag_mixin)
    inv_reason7 = Flag("cps-technical-deviation-ta", "reason 7")
    flags = (
        inv_reason7,
        Flag("cps-behavioral-deviation-aa", "reason 8"),
    )

    imi.add_flags(flags)
    assert imi.flag_count == 8
    imi.add_flags([inv_reason7], ignore_duplicates=True)
    assert imi.flag_count == 8


@pytest.mark.parametrize(
    "left,right,func,expected",
    [
        (WrappedResult(42), 5, operator.add, 47),
        (WrappedResult(42), WrappedResult(5), operator.add, 47),
        (WrappedResult(42), WrappedResult(5), operator.sub, 37),
        (WrappedResult(5), WrappedResult(5), operator.mul, 25),
    ],
)
def test_binary_operations_flag(left, right, func, expected):
    """Test flag inheritance for binary operation of WrappedResult."""
    left.add_flag(Flag("test-behavioral-deviation-id", reason="test"))
    res = func(left, right)
    assert res.measure_value == expected
    assert isinstance(res, WrappedResult)
    assert {"test-behavioral-deviation-id"} <= res.flag_ids


@pytest.mark.parametrize(
    "obj,func,expected",
    [
        (WrappedResult(-42.0), operator.abs, 42),
    ],
)
def test_unary_operations_flag(obj, func, expected):
    """Test flag inheritance for unary op. of WrappedResult objects."""
    obj.add_flag(Flag("test-behavioral-deviation-id", reason="test"))
    res = func(obj)
    assert isinstance(res, WrappedResult)
    assert res.measure_value == expected
    assert {"test-behavioral-deviation-id"} <= res.flag_ids
