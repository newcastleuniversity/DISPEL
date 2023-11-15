"""Tests for :mod:`dispel.providers.ads.data`."""

from dispel.providers.ads.data import ADSModalities
from dispel.providers.ads.io import parse_context
from dispel.providers.generic.tasks.cps.modalities import CPSLevel


def test_ads_modalities():
    """Test the class :class:`~dispel.data.ads.ADSModalities`."""
    modalities = ADSModalities("0.2.9")

    assert isinstance(modalities, ADSModalities)

    cps_context_raw = [
        {"name": "levelType", "value": "digitToSymbol", "unit": "n/a"},
        {"name": "predefinedKey1", "value": "false", "unit": "n/a"},
        {"name": "predefinedKey2", "value": "true", "unit": "n/a"},
        {"name": "randomKey", "value": "false", "unit": "n/a"},
        {"name": "predefinedSequence", "value": "true", "unit": "n/a"},
        {"name": "randomSequence", "value": "false", "unit": "n/a"},
    ]
    cps_context = parse_context(cps_context_raw)

    pinch_context_raw = [
        {"name": "targetRadius", "value": "102.0", "unit": "point"},
        {"name": "xTargetBall", "value": "187.5", "unit": "point"},
        {"name": "yTargetBall", "value": "427.9770250160713", "unit": "point"},
        {"name": "usedHand", "value": "right", "unit": "n/a"},
    ]
    pinch_context = parse_context(pinch_context_raw)

    assert modalities.get_modalities_from_context("cps", cps_context) == [
        CPSLevel.SYMBOL_TO_DIGIT.value
    ]
    assert modalities.get_modalities_from_context("pinch", pinch_context) == [
        "right",
        "extra_large",
    ]
