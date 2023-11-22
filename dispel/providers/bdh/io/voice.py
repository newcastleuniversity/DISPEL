"""Function for converting voice BDH JSON files into a reading."""
from dispel.data.levels import LevelId


def get_level_id(config: dict) -> LevelId:
    """Parse level id from level type and configuration.

    Parameters
    ----------
    config
        The level configuration

    Returns
    -------
    LevelId
        Level id for the level.

    Raises
    ------
    NotImplementedError
        If the given mode parsing has not been implemented.
    """
    attempt_number = config["attempt_number"]

    if config["exercise_name"] == "pataka":
        return LevelId(f"pataka.{attempt_number}")
    if config["exercise_name"] == "aah":
        return LevelId(f"aah.{attempt_number}")

    raise NotImplementedError(
        f"Level Id is not implemented for mode: {config['mode']} "
        f"and attempt_number {attempt_number}"
    )
