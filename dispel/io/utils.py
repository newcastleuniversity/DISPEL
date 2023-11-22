"""A collection of IO utility functions."""

import json
from json import JSONDecodeError
from os.path import exists
from typing import Any, Dict, List, Tuple

import pandas as pd


def flatten(data: Dict, parent_key: str = "") -> Dict:
    """Return a flattened version of a nested dictionary."""
    items: List[Tuple[str, Any]] = []
    for key, item in data.items():
        try:
            items.extend(flatten(item, "%s%s_" % (parent_key, key)).items())
        except AttributeError:
            items.append(("%s%s" % (parent_key, key), item))
    return dict(items)


def readable_file(path: str) -> bool:
    """FIXME: documentation."""
    return exists(path)


def load_json(path: str, encoding=None) -> dict:
    """FIXME: documentation."""
    with open(path, "r", encoding=encoding) as fh:
        return json.load(fh)


def readable_json(path: str) -> bool:
    """FIXME: documentation."""
    try:
        load_json(path)
        return True
    except (JSONDecodeError, FileNotFoundError):
        return False


def readable_dict(value: Any) -> bool:
    """FIXME: documentation."""
    return isinstance(value, dict)


def readable_data_frame(value: Any) -> bool:
    """FIXME: documentation."""
    return isinstance(value, pd.DataFrame)
