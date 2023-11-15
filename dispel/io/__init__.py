"""IO functionality."""
from typing import Any, Optional

from dispel.data.core import Reading
from dispel.providers.registry import READERS, ReaderRegistryT


def read(source: Any, registry: Optional[ReaderRegistryT] = None) -> Reading:
    """Read a source into a Reading data model.

    TODO: write documentation
    """
    for reader in (registry or READERS).values():
        if reader["readable"](source):
            return reader["func"](source)

    raise ValueError(f"Provided {source} is not automatically readable")
