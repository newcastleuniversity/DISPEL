"""Functions to manage flags when generating measure export."""

from typing import Any, Dict, Iterable, List

from dispel.data.core import Reading
from dispel.data.flags import Flag, FlagMixIn


def format_flag(flag_object: FlagMixIn) -> List:
    """Format flags objects into a string format to add to the json."""
    if not isinstance(flag_object, FlagMixIn):
        raise TypeError(
            "flag_object should be an FlagMixin." f"but is of type: {type(flag_object)}"
        )
    return [
        {
            "id": flag.id.id,
            "reason": flag.reason,
            "stop_processing": flag.stop_processing,
        }
        for flag in flag_object.get_flags()
    ]


def format_all_flags(reading: Reading) -> Dict[str, Any]:
    """Format all flag inside a reading into a usable format."""
    flags: Dict[str, Any] = {
        "reading": format_flag(reading),
        "level": {},
    }

    for level in reading.levels:
        lvl_id = str(level.id)
        flags["level"][lvl_id] = {}
        level_flag = format_flag(level)
        if len(level_flag) > 0:
            flags["level"][lvl_id][lvl_id] = level_flag
        for data_set in level.raw_data_sets:
            ds_id = str(data_set.id)
            data_set_flag = format_flag(data_set)
            if len(data_set_flag) > 0:
                flags["level"][lvl_id][ds_id] = data_set_flag
        measure_set = level.measure_set
        for f_id in measure_set.ids():
            feat_flag = format_flag(measure_set.get(f_id))  # type: ignore
            if len(feat_flag) > 0:
                flags["level"][lvl_id][str(f_id)] = feat_flag
    return flags


def merge_flags(flags: Iterable[Flag]) -> Dict[str, str]:
    """Merge multiple flags in a dictionary.

    Parameters
    ----------
    flags
        An iterable of flags.

    Returns
    -------
    Dict[str, str]
        A dictionary regrouping the provided flags by id and reason.
    """

    def merge(flags: Iterable[Flag], attr: str) -> str:
        return ";".join(map(lambda i: str(getattr(i, attr)), flags))

    return dict(
        flag_ids=merge(flags, "id"),
        flag_reasons=merge(flags, "reason"),
    )
