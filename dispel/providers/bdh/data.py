"""Additional models for BDH data sets."""
from typing import Any, Dict, Optional

from dispel.data.core import Evaluation, Reading
from dispel.data.raw import RawDataSetSource


class BDHReading(Reading):
    """BDH reading."""


class BDHRawDataSetSource(RawDataSetSource):
    """BDH raw data source model.

    Parameters
    ----------
    manufacturer
        The manufacturer of the raw data source
    chipset
        The chipset of the source should it be a sensor
    reference
        The reference of the data source
    """

    def __init__(
        self, manufacturer: str, chipset: Optional[str], reference: Optional[str]
    ):
        super().__init__(manufacturer)
        self.chipset = chipset
        self.reference = reference


class BDHEvaluation(Evaluation):
    """BDH specific evaluation class capturing header meta information."""

    def __init__(self, *args, header_meta: Dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.header_meta = header_meta

    def to_dict(self):
        """Retrieve values of evaluation as dictionary."""
        res = super().to_dict()
        res.update(self.header_meta)
        return res
