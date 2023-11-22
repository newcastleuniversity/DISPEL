"""Functionality to read ADS files."""
import warnings
from typing import Any, Dict, Iterable, List, Optional, Union, cast

import pandas as pd

from dispel.data.core import Device, Evaluation, ReadingSchema, Session
from dispel.data.devices import (
    AndroidPlatform,
    AndroidScreen,
    IOSPlatform,
    IOSScreen,
    Screen,
)
from dispel.data.epochs import EpochDefinition
from dispel.data.levels import Context, Level
from dispel.data.raw import (
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.data.values import Value, ValueDefinition
from dispel.io.core import (
    convert_data_frame_type,
    convert_literal_type,
    get_data_type_mapping,
)
from dispel.io.utils import load_json
from dispel.providers.ads.data import ADSModalities, ADSReading
from dispel.providers.generic.tasks.cps.utils import (
    EXPECTED_DURATION_D2D,
    EXPECTED_DURATION_S2D,
    LEVEL_DURATION_DEF,
)
from dispel.providers.registry import register_reader


def parse_screen(platform: str, screen_dict: dict) -> Screen:
    """Parse a screen dictionary into a Screen class.

    Parameters
    ----------
    platform
        The device platform.
    screen_dict
        The screen information dictionary.

    Returns
    -------
    Screen
        The screen information.

    Raises
    ------
    ValueError
        If given an unsupported platform.
    """
    width_dp_pt = screen_dict["widthPixels"]
    height_dp_pt = screen_dict["heightPixels"]
    if platform == "iOS":
        scale_factor = screen_dict["scaleFactor"]
        return IOSScreen(
            width_pixels=width_dp_pt * scale_factor,
            height_pixels=height_dp_pt * scale_factor,
            density_dpi=screen_dict["densityDpi"],
            width_dp_pt=width_dp_pt,
            height_dp_pt=height_dp_pt,
            scale_factor=scale_factor,
        )
    if platform == "Android":
        return AndroidScreen(
            width_dp_pt,
            height_dp_pt,
            screen_dict["densityDpi"],
            screen_dict["xDpi"],
            screen_dict["yDpi"],
        )

    raise ValueError(
        "Platform only supports the following values: `iOS` and `Android`."
    )


def parse_device(device_dict: dict) -> Device:
    """Parse a device dictionary into a Device class.

    Parameters
    ----------
    device_dict
        The device information dictionary.

    Returns
    -------
    Device
        The device information.
    """
    platform_str = device_dict["platform"]
    platform = IOSPlatform() if platform_str == "iOS" else AndroidPlatform()
    screen = parse_screen(platform_str, device_dict["screen"])
    version = (
        device_dict["osVersion"]
        if platform_str == "iOS"
        else device_dict["kernelVersion"]
    )

    return Device(
        None,
        platform,
        device_dict.get("model"),
        device_dict["modelCode"],
        version,
        device_dict["versionNumber"],
        device_dict["buildNumber"],
        screen,
    )


def parse_session(session_dict: dict) -> Session:
    """Parse the session information into a Session class.

    Parameters
    ----------
    session_dict
        The session information dictionary.

    Returns
    -------
    Session
        The session related information.
    """
    start = pd.Timestamp(session_dict["startDate"]).tz_convert(None)
    end = pd.Timestamp(session_dict["endDate"]).tz_convert(None)

    return Session(
        start=start,
        end=end,
        definition=EpochDefinition(id_=session_dict["sessionCode"]),
        uuid=session_dict["uuidSession"],
        evaluation_codes=session_dict.get("evaluationCodes"),
    )


def parse_evaluation(
    id_: str,
    evaluation: Dict[str, Union[str, bool, int]],
    user_id: Optional[str] = None,
) -> Evaluation:
    """Parse the evaluation information into an Evaluation class.

    Parameters
    ----------
    id_
        The evaluation identifier
    evaluation
        The evaluation dictionary information
    user_id
        The identifier of the user

    Returns
    -------
    Evaluation
        The evaluation related information.
    """
    start = pd.Timestamp(evaluation["beginTimestamp"], unit="ms")
    end = pd.Timestamp(evaluation["endTimestamp"], unit="ms")

    evaluation_code = cast(str, evaluation["code"])
    return Evaluation(
        start=start,
        end=end,
        definition=EpochDefinition(id_=evaluation_code),
        uuid=id_,
        finished=cast(bool, evaluation["finished"]),
        exit_reason=cast(str, evaluation["exitReason"]),
        user_id=user_id,
    )


def create_ads_value_definitions(
    value_sample: List[dict], raw_data_set_id: str
) -> List[RawDataValueDefinition]:
    """Create ADS related value definition classes.

    Parameters
    ----------
    value_sample
        A list of an ads raw data sensor value level
    raw_data_set_id
        The raw data set id

    Returns
    -------
    List[RawDataValueDefinition]
    """
    value_definitions = []
    for value_dict in value_sample:
        unit = value_dict["unit"] if (value_dict["unit"] != "n/a") else None
        value_definitions.append(
            RawDataValueDefinition(
                value_dict["name"],
                ".".join([raw_data_set_id, value_dict["name"]]),
                unit=unit,
                data_type=get_data_type_mapping(value_dict["name"]),
            )
        )
    return value_definitions


def create_ads_raw_data_set_definition(
    value_definitions: Iterable[RawDataValueDefinition], raw_data_set_id: str
) -> RawDataSetDefinition:
    """Create ADS raw data set definition.

    Parameters
    ----------
    value_definitions
        An iterable of raw data value definitions
    raw_data_set_id
        The raw data set id

    Returns
    -------
    RawDataSetDefinition
        The definition of the raw data set
    """
    # check if value is computed or measured
    is_computed = raw_data_set_id not in ["accelerometer", "gyroscope", "gps"]

    return RawDataSetDefinition(
        raw_data_set_id, RawDataSetSource("ADS"), value_definitions, is_computed
    )


def _frame_measurement_data(measurements: List[dict]) -> pd.DataFrame:
    """Convert a list of raw ADS measurements to a pandas data frame format.

    Parameters
    ----------
    measurements
        A list of dictionary value measurements

    Returns
    -------
    pandas.DataFrame
    """
    columns = [value["name"] for value in measurements[0]["values"]]
    data = pd.DataFrame(
        [
            [value["value"] for value in measurement["values"]]
            for measurement in measurements
        ],
        columns=columns,
    )
    return convert_data_frame_type(data)


def get_ads_raw_data_set(level: dict, raw_data_set_id: str) -> pd.DataFrame:
    """Read the ads raw data sets.

    Parameters
    ----------
    level
        An evaluation level as in ADS json format, e.g.
        ``data['mobileEvaluationTest']['levels'][level_num]`` with
        ``level_num in {0,1,2,...}``
    raw_data_set_id
        The raw data set id

    Returns
    -------
    pandas.DataFrame
        The raw data set data frame.

    Raises
    ------
    ValueError
        If the raw data set id is not found in level data.
    """
    # go through sensors (e.g. level['sensors'] = {'user_input', 'screen'})
    for sensor in filter(lambda s: s["name"] == raw_data_set_id, level["sensors"]):
        raw_data_set = _frame_measurement_data(sensor["measurements"])
        return raw_data_set

    raise ValueError(f"Unknown raw_data_set_id {raw_data_set_id}")


def parse_raw_data_set_value_definitions(
    level: dict, raw_data_set_id: str
) -> RawDataSetDefinition:
    """Parse ADS raw data set value definitions.

    Parameters
    ----------
    level : dict
        An evaluation level as in ADS json format, e.g.
        ``level = data['mobileEvaluationTest']['levels'][level_id]`` with
        ``level_id`` in ``{0, 1, 2, ...}``
    raw_data_set_id
        The raw data set id

    Returns
    -------
    RawDataSetDefinition
    """
    # initialize value_definition
    value_definitions = []

    # go through sensors (e.g. level['sensors'] = {'user_input', 'screen'})
    for sensor in level["sensors"]:
        # check if id matches the sensor
        if sensor["name"] == raw_data_set_id:
            # get measurements values and create definition
            value_definitions = create_ads_value_definitions(
                sensor["measurements"][0]["values"], raw_data_set_id
            )

    # create definition
    definition = create_ads_raw_data_set_definition(value_definitions, raw_data_set_id)
    return definition


def create_ads_raw_data_set(data: dict, raw_data_set_id: str) -> RawDataSet:
    """Create ADS raw data set.

    Parameters
    ----------
    data : dict
        An evaluation level as in ADS json format, e.g.
        ``data['mobileEvaluationTest']['levels'][level_num]`` with
        ``level_num`` in ``{0, 1, 2, ...}``
    raw_data_set_id
        The raw data set id

    Returns
    -------
    RawDataSet
    """
    # parse raw dataset and set definitions
    definition = parse_raw_data_set_value_definitions(data, raw_data_set_id)

    # create dataframe and update types
    data_frame = convert_data_frame_type(get_ads_raw_data_set(data, raw_data_set_id))

    return RawDataSet(definition, data_frame)


def get_ads_raw_data_set_ids(data: dict) -> Iterable[str]:
    """Get the ads raw data sets ids.

    Parameters
    ----------
    data
        A sample of an ads raw data sensor level

    Returns
    -------
    Generator[str, None, None]
        The list of raw data set ids
    """
    return (sensor["name"] for sensor in data if len(sensor["measurements"]) > 0)


def parse_context(data: List) -> Context:
    """Parse the context information available for each level.

    Parameters
    ----------
    data
        A dictionary extracted from a json corresponding to the context related
        information

    Returns
    -------
    Context
        The context representation of the passed ``data``.
    """

    def _values_for_context(value_data: Dict) -> Value:
        name = value_data["name"]
        return Value(
            ValueDefinition(name, name, value_data["unit"]),
            convert_literal_type(name, value_data["value"]),
        )

    values = [_values_for_context(item) for item in data]

    return Context(values)


def enrich_context(context, evaluation_code, level_modalities):
    """Enrich the context information with test specific information."""
    # CPS
    if evaluation_code == "cps":
        assert isinstance(level_modalities, list)
        if level_modalities[0] == "symbol_to_digit":
            duration = EXPECTED_DURATION_S2D
        elif level_modalities[0] == "digit_to_digit":
            duration = EXPECTED_DURATION_D2D
        else:
            raise ValueError(f"unexpected modality {level_modalities}")
        context.set(duration, LEVEL_DURATION_DEF)


def parse_level(
    data: Dict[str, Any], evaluation_code: str, ads_modalities: ADSModalities
) -> Level:
    """Parse a specific level.

    Parameters
    ----------
    data
        A dictionary extracted from a json corresponding to a level.
    evaluation_code
        The evaluation code, e.g. ``CPS``.
    ads_modalities
        The AdS modalities object.

    Returns
    -------
    Level
        The level representation of ``data``.

    """
    context = parse_context(data["contexts"])

    start = pd.Timestamp(int(data["beginTimestamp"]), unit="ms")
    end = pd.Timestamp(int(data["endTimestamp"]), unit="ms")

    # level_id
    level_modalities = ads_modalities.get_modalities_from_context(
        evaluation_code=evaluation_code, context=context
    )

    raw_data_set_ids = get_ads_raw_data_set_ids(data["sensors"])

    # Remove duplicate UserInput from Mood
    if evaluation_code == "mood":
        raw_data_set_ids = list(raw_data_set_ids)
        if raw_data_set_ids.count("userInput") > 1:
            warnings.warn(
                "Several answers to the same question have been detected. The "
                "last answer is kept.",
                UserWarning,
            )
            raw_data_set_ids = ["userInput"]
            data["sensors"] = [data["sensors"][-1]]

    # init raw_data_sets data structure
    raw_data_sets = []
    for raw_data_set_id in raw_data_set_ids:
        raw_data_sets.append(create_ads_raw_data_set(data, raw_data_set_id))

    enrich_context(context, evaluation_code, level_modalities)

    # fill levels
    return Level(
        id_=level_modalities,
        start=start,
        end=end,
        context=context,
        raw_data_sets=raw_data_sets,
        measure_set=None,
    )


def parse_levels(
    data: dict, evaluation_code: str, ads_modalities: ADSModalities
) -> Optional[List[Level]]:
    r"""Extract a list of Level from ``levels_data``.

    Here ``levels_data`` refers to ``data['mobileEvaluationTest']['levels']``.

    Parameters
    ----------
    data
        A dictionary extracted from a json corresponding to every level
        related information.
    evaluation_code
        The evaluation code, e.g. ``CPS``
    ads_modalities
        The AdS modalities object.

    Returns
    -------
    List[Level]
        A list of :class:`~dispel.data.core.Level`\ s.
    """
    if not data:
        return None

    levels = []
    for level_data in data:
        levels.append(parse_level(level_data, evaluation_code, ads_modalities))
    return levels


def parsable_ads_json(value: Any) -> bool:
    """Infer if a value can be automatically read with :func:`parse_ads_raw_json`."""
    if not isinstance(value, dict):
        return False
    return "mobileEvaluationTest" in value.keys()


@register_reader(parsable_ads_json, ADSReading)
def parse_ads_raw_json(data: dict) -> ADSReading:
    """Parse data from ADS JSON file.

    Parameters
    ----------
    data
        The ADS raw data.

    Returns
    -------
    ADSReading
        The :class:`~dispel.data.ads.ADSReading` representation of the ADS JSON
        raw data.
    """
    evaluation = parse_evaluation(
        data["uuid"], data["mobileEvaluationTest"], data.get("userId")
    )

    # sessions
    session_data = data.get("mobileSession")
    if session_data is not None:
        session = parse_session(session_data)
    else:
        # `data['mobileSession']` is `None` in the passive test.
        session = Session(
            start=evaluation.start,
            end=evaluation.end,
            definition=EpochDefinition(id_="n/a"),
            uuid="n/a",
        )

    schema = ReadingSchema("ADS", "konectom", "1.0")
    device = parse_device(data["mobileDevice"])

    # parse levels
    levels_data = data["mobileEvaluationTest"]["levels"]
    levels = parse_levels(
        levels_data, str(evaluation.id), ADSModalities(device.app_version_number)
    )

    return ADSReading(
        evaluation=evaluation,
        session=session,
        levels=levels,
        schema=schema,
        date=data.get("receptionDate"),
        device=device,
    )


def read_ads(path: str) -> ADSReading:
    """Read raw ADS JSON file.

    Parameters
    ----------
    path
        The path to the JSON file containing the data to be read.

    Returns
    -------
    ADSReading
    """
    return parse_ads_raw_json(load_json(path, "utf-8"))
