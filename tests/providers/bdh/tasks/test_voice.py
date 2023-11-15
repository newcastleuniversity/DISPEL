"""Tests for :mod:`dispel.providers.bdh.tasks.voice`."""

import pandas as pd

from dispel.data.core import Reading


def test_process_voice(example_reading_processed_voice):
    """Test end-to-end processing of a Voice Assessment."""
    pataka_sample = 661248
    assert isinstance(example_reading_processed_voice, Reading)
    audio = (
        example_reading_processed_voice.get_level("pataka.1")
        .get_raw_data_set("amplitude")
        .data
    )
    assert isinstance(audio, pd.DataFrame)
    assert audio.shape[0] == pataka_sample
