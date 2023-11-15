"""Tests for :mod:`dispel.providers.generic.tasks.draw.intersections`."""
from dispel.providers.generic.tasks.draw.intersections import (
    Point,
    Segment,
    get_intersection_data,
    get_intersection_features,
    get_ratio,
)


def test_get_point_objects(example_paths):
    """Test the good completion of Point objects creation."""
    point = Point(example_paths["x"][0], example_paths["y"][0])
    assert isinstance(point, Point)
    assert point.coordinates == (0, 0)


def test_get_segment(example_segments):
    """Test the good computation of segment objects."""
    assert isinstance(example_segments["seg"][1], Segment)
    assert example_segments["seg"][1].segment == ((0, 1), (0, 0))
    assert example_segments["seg"][2].segment == ((0, 2), (0, 1))
    assert example_segments["seg"][1].distance == 1
    assert example_segments["dist"][2] == 1


def test_get_ratio(example_segments):
    """Test the good computation of distance ratios."""
    data = get_ratio(example_segments)
    assert data[2] == 1.0


def test_get_intersection_data(intersection_data):
    """Test the good transformation of intersection data."""
    data, model = intersection_data
    user, ref = get_intersection_data(data, model)
    assert list(user.columns) == ["tsTouch", "seg", "norm", "ratio"]
    assert list(ref.columns) == ["seg", "norm", "ratio"]
    assert isinstance(ref["seg"][0], Segment)
    assert isinstance(user["seg"][0], Segment)


def test_get_intersection_features(intersection_data_formatted):
    """Test the good computation of intersection detection."""
    user, ref = intersection_data_formatted
    result = get_intersection_features(user, ref)
    assert len(result) == 2
    assert result.tsDiff[1] == 3
    assert result["cross_per_sec"][0] == 0.3333333333333333
