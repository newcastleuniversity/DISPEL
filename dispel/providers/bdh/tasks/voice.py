"""Voice assessment related functionality.

This module contains functionality to extract features for the
*Voice* assessment.
"""
import base64
import io
from typing import List

import pandas as pd
import soundfile as sf

from dispel.data.raw import RawDataValueDefinition
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing import ProcessingStep
from dispel.processing.data_set import transformation
from dispel.processing.transform import TransformStep
from dispel.providers.registry import process_factory

TASK_NAME = AV("Voice Assessment", "VOICE")


class DecodeRawAudio(TransformStep):
    """Transform the raw audio string to a pd.Series object.

    This class transforms a base64 encoded audio string to a pd.Series object.
    The code is optimized so that the transformation is done using
    a bytes stream buffer in order to avoid writing audio files in memory.
    """

    data_set_ids = "audio"
    new_data_set_id = "amplitude"
    definitions = [
        RawDataValueDefinition(
            "raw_amplitude",
            "raw amplitude",
            data_type="float64",
        )
    ]

    @staticmethod
    @transformation
    def decode_raw_audio_base64(data: pd.DataFrame) -> pd.Series:
        """Return a filtered version of the tap dataset."""
        # Decode the raw base 64 audio file
        base64_encoded = data["base64"].iloc[0]
        decoded_b64_raw_audio = base64.b64decode(base64_encoded)
        # Create a buffer of bytes from the decoded string
        raw_bytes = io.BytesIO(decoded_b64_raw_audio)
        # Convert the data to a tabular format using sf
        decoded_audio, _ = sf.read(raw_bytes)

        return decoded_audio


STEPS: List[ProcessingStep] = [
    # FIXME : Add a flag
    #  step https://app.asana.com/0/1201445953640941/1201698241256099
    # Transform the raw audio
    DecodeRawAudio(),
]

process_voice = process_factory(
    task_name=TASK_NAME,
    steps=STEPS,
    codes="voice-activity",
)
