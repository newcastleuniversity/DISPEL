"""Functionality to read files from BDH apps.

Examples
--------
To read a BDH json file and work with the contained data one can read the file:

.. testsetup:: bdh

    >>> import pkg_resources
    >>> path = pkg_resources.resource_filename('tests.providers.bdh',
    ...                                        '_resources/DRAW/uat_drawing.json')

.. doctest:: bdh

    >>> from dispel.providers.bdh.io import read_bdh
    >>> reading = read_bdh(path)

And access raw sensor data for the first level

.. doctest:: bdh
    :options: +NORMALIZE_WHITESPACE

    >>> id_ = reading.level_ids[0]
    >>> reading.get_level(id_).get_raw_data_set('screen').data.head().tsTouch
    0   2021-03-21 10:33:26.666
    1   2021-03-21 10:33:26.674
    2   2021-03-21 10:33:26.690
    3   2021-03-21 10:33:26.698
    4   2021-03-21 10:33:26.716
    Name: tsTouch, dtype: datetime64[ns]

For further details on :class:`~dispel.data.core.Reading` and the data model take a look at
:mod:`dispel.data.core`.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional

from dispel.data.collections import FeatureSet
from dispel.data.core import Device, Reading, ReadingSchema, Session
from dispel.data.devices import AndroidPlatform, IOSPlatform, PlatformType, Screen
from dispel.data.epochs import EpochDefinition
from dispel.data.features import FeatureValue
from dispel.data.levels import Context, Level, LevelId
from dispel.data.raw import RawDataSet, RawDataSetDefinition, RawDataValueDefinition
from dispel.data.values import Value, ValueDefinition
from dispel.io.utils import flatten, load_json
from dispel.providers.bdh.data import BDHEvaluation, BDHRawDataSetSource, BDHReading
from dispel.providers.bdh.io.core import (
    KEYS,
    get_level_id_two_hands,
    parse_epoch,
    parse_raw_data_sets,
)
from dispel.providers.bdh.io.cps import (
    convert_activity_sequence as convert_activity_sequence_cps,
)
from dispel.providers.bdh.io.cps import get_level_id as get_level_id_cps
from dispel.providers.bdh.io.cps import (
    translate_reference_table_type,
    translate_sequence_type,
)
from dispel.providers.bdh.io.drawing import convert_touch_events
from dispel.providers.bdh.io.drawing import get_level_id as get_level_id_drawing
from dispel.providers.bdh.io.gait import convert_gps
from dispel.providers.bdh.io.gait import get_level_id as get_level_id_gait
from dispel.providers.bdh.io.msis29 import (
    convert_activity_sequence as convert_activity_sequence_msis,
)
from dispel.providers.bdh.io.msis29 import create_levels as create_levels_msis
from dispel.providers.bdh.io.msis29 import get_level_id as get_level_id_msis
from dispel.providers.bdh.io.neuroqol import (
    convert_activity_sequence as convert_activity_sequence_neuroqol,
)
from dispel.providers.bdh.io.pinch import create_levels as create_levels_pinch
from dispel.providers.bdh.io.pinch import get_level_id as get_level_id_pinch
from dispel.providers.bdh.io.pinch import (
    update_raw_data_definition as update_raw_data_definition_pinch,
)
from dispel.providers.bdh.io.sbt_utt import convert_timestamp
from dispel.providers.bdh.io.sbt_utt import get_level_id as get_level_id_sbt
from dispel.providers.bdh.io.survey import convert_flagged_answers
from dispel.providers.bdh.io.survey import create_levels as create_levels_mood
from dispel.providers.bdh.io.survey import get_level_id as get_level_id_mood
from dispel.providers.bdh.io.voice import get_level_id as get_level_id_voice
from dispel.providers.generic.tasks.cps.utils import (
    EXPECTED_DURATION_D2D,
    EXPECTED_DURATION_S2D,
    LEVEL_DURATION_DEF,
)
from dispel.providers.registry import register_reader

DATE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
MAPPING_SHAPE_LEVEL_TYPE = {
    "infinity_loop": "infinity",
    "rectangle_counter_clockwise": "squareCounterClock",
    "rectangle_clockwise": "squareClock",
    "spiral": "spiral",
}


def _convert_activity_sequence(
    data: Dict[str, Any], definition: RawDataSetDefinition
) -> RawDataSet:
    """Convert activity sequence dataset to userInput."""
    # CPS case
    if "presented_symbol_timestamp" in data:
        return convert_activity_sequence_cps(data, definition)
    # NEUROQOL Case
    if "form_item_id" in data:
        return convert_activity_sequence_neuroqol(data, definition)
    # MSIS29 Case
    return convert_activity_sequence_msis(data, definition)


TO_CONVERT: Dict[str, Callable] = {
    "touch_events": convert_touch_events,
    "activity_sequence": _convert_activity_sequence,
    "validated_answers": convert_flagged_answers,
    "accelerometer": lambda x, y: convert_timestamp("accelerometer", x, y),
    "raw_accelerometer": lambda x, y: convert_timestamp("raw_accelerometer", x, y),
    "calibrated_accelerometer": lambda x, y: convert_timestamp("accelerometer", x, y),
    "gyroscope": lambda x, y: convert_timestamp("gyroscope", x, y),
    "raw_gyroscope": lambda x, y: convert_timestamp("raw_gyroscope", x, y),
    "calibrated_gyroscope": lambda x, y: convert_timestamp("gyroscope", x, y),
    "gravity": lambda x, y: convert_timestamp("gravity", x, y),
    "gps": convert_gps,
}


def parse_schema(data: Dict) -> ReadingSchema:
    """Parse the schema of a reading.

    Parameters
    ----------
    data
        A dictionary containing the schema attributes

    Returns
    -------
    ReadingSchema
        The created reading schema.
    """
    return ReadingSchema(**data)


def update_raw_data_definition(
    data: Dict[str, Any], schema: ReadingSchema
) -> Dict[str, Any]:
    """Update raw data definitions.

    Parameters
    ----------
    data
        Raw data definitions
    schema
        Data scheme

    Returns
    -------
    Dict[str, Any]
        Level data

    """
    if "pinch" in schema.name:
        return update_raw_data_definition_pinch(data)

    return data


def create_levels(data: Dict[str, Any], schema: ReadingSchema) -> Dict[str, Any]:
    """Create levels from uni-level activity data.

    The BDH format does not split data in levels in the same way the ADS format does.
    For example, in the mood test, the two questions are lumped in the same level. This
    function splits the data in the way `dispel` expects.

    Parameters
    ----------
    data
        Level data
    schema
        Data scheme

    Returns
    -------
    Dict[str, Any]
        Level data
    """
    if "mood" in schema.name:
        return create_levels_mood(data)
    if "msis" in schema.name:
        return create_levels_msis(data)
    if "pinch" in schema.name:
        return create_levels_pinch(data)

    return data


def get_level_id(config: dict, schema: ReadingSchema) -> LevelId:
    """Parse level id from level type and configuration.

    Parameters
    ----------
    config
        The level configuration
    schema
        The schema from data header

    Returns
    -------
    LevelId
        Level id for the level.

    Raises
    ------
    NotImplementedError
        If the schema name parsing has not been implemented.
    """
    id_functions = {
        "drawing": get_level_id_drawing,
        "cps": get_level_id_cps,
        "sb": get_level_id_sbt,
        "pinch": get_level_id_pinch,
        "6mw": get_level_id_gait,
        "mood": get_level_id_mood,
        "msis": get_level_id_msis,
        "sp-activity": get_level_id_two_hands,
        "fingertap-activity": get_level_id_two_hands,
        "voice-activity": get_level_id_voice,
    }

    for k, v in id_functions.items():
        if k in schema.name:
            return v(config)

    raise NotImplementedError(f"{schema.name} is not implemented yet.")


def _parse_value_definition(id_: str, data: Dict, cls):
    return cls(
        id_=id_,
        name=id_,
        description=data.get(KEYS.description, None),
        unit=data.get(KEYS.unit, None),
    )


def parse_feature_definition(id_: str, data: Dict) -> ValueDefinition:
    """Parse a feature definition.

    Parameters
    ----------
    id_
        The id of the feature value definition
    data
        The feature definition in BDH json format

    Returns
    -------
    ValueDefinition
        The created value definition.
    """
    return _parse_value_definition(id_, data, ValueDefinition)


def parse_raw_data_source(data: Dict) -> BDHRawDataSetSource:
    """Parse a raw data source definition.

    Parameters
    ----------
    data
        The BDH raw data source definition dictionary in BDH json format.

    Returns
    -------
    BDHRawDataSetSource
        The created BDH format raw data set source.
    """
    return BDHRawDataSetSource(
        data[KEYS.manufacturer],
        data.get(KEYS.chipset, None),
        data.get(KEYS.reference, None),
    )


def parse_raw_data_value_definition(id_: str, data: Dict) -> RawDataValueDefinition:
    """Parse a raw data value definition.

    Parameters
    ----------
    id_
        The id of the raw data value definition.
    data
        The raw data value definition in BDH json format.

    Returns
    -------
    RawDataValueDefinition
        The created raw data value definition.
    """
    return _parse_value_definition(id_, data, RawDataValueDefinition)


def parse_screen(device: Dict[str, Any]) -> Screen:
    """Parse a screen dictionary into a Screen class.

    Parameters
    ----------
    device
        Device information.

    Returns
    -------
    Screen
        The screen information.
    """
    width_pixels = int(device["screen_width_pixels"])
    height_pixels = int(device["screen_height_pixels"])
    try:
        return Screen(
            width_pixels=width_pixels,
            height_pixels=height_pixels,
            density_dpi=(width_pixels / device["screen_width_mm"]) * 25.4,
            width_dp_pt=int(device["screen_width_pt_dp"]),
            height_dp_pt=int(device["screen_height_pt_dp"]),
        )
    except KeyError:
        return Screen(width_pixels=width_pixels, height_pixels=height_pixels)


def _parse_platform(model_name: str) -> PlatformType:
    """Get the platform from the model name."""
    if model_name is None:
        return AndroidPlatform()

    if "iPhone" in model_name:
        return IOSPlatform()

    return AndroidPlatform()


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
    platform = _parse_platform(device_dict.get(KEYS.model_name, None))
    screen = parse_screen(device_dict)
    version = device_dict.get(KEYS.os_version, None)

    return Device(
        None,
        platform,
        device_dict.get(KEYS.model_name, None),
        device_dict.get(KEYS.model_number, None),
        version,
        screen=screen,
    )


def parse_session(header: dict) -> Session:
    """Parse the header into a Session class.

    Parameters
    ----------
    header
        The header dictionary.

    Returns
    -------
    Session
        Session data.
    """
    epoch = parse_epoch(header["effective_time_frame"])
    session_cluster_id = None
    cluster_activity_codes = None
    if "session_cluster_id" in header:
        session_cluster_id = header["session_cluster_id"]
    if "cluster_activity_codes" in header:
        cluster_activity_codes = header["cluster_activity_codes"]

    code = "InClinic" if header["inclinic_mode"] else "Daily"
    return Session(
        start=epoch.start,
        end=epoch.end,
        definition=EpochDefinition(id_=code),
        uuid=session_cluster_id,
        evaluation_codes=cluster_activity_codes,
    )


def parse_raw_data_set_definition(id_: str, data: Dict) -> RawDataSetDefinition:
    """Parse a raw data set definition for a reading.

    Parameters
    ----------
    id_
        The id of the raw data set
    data
        The definition of the raw data set in BDH json format.

    Returns
    -------
    RawDataSetDefinition
        The created raw data set definition.

    Raises
    ------
    ValueError
        If no source is defined for the given raw data source.
    ValueError
        If no values are defined for the given raw data source.
    """
    if KEYS.source not in data or not data[KEYS.source]:
        raise ValueError(f"No source defined for raw data source {id_}")

    source = parse_raw_data_source(data[KEYS.source])

    if KEYS.values not in data:
        raise ValueError(f"No values defined for raw data source {id_}")

    definitions = []
    for data_name, data_def in data[KEYS.values].items():
        # FIXME once values have ids too then they can be replaced
        data_def[KEYS.name] = data_name
        definitions.append(parse_raw_data_value_definition(data_name, data_def))

    return RawDataSetDefinition(
        id=id_,
        source=source,
        value_definitions_list=definitions,
        is_computed=data[KEYS.computed],
    )


def parse_features(
    data: Dict[str, Any], definitions: Iterable[ValueDefinition]
) -> FeatureSet:
    """Parse features from a reading.

    Parameters
    ----------
    data
        The data dictionary for features in BDH json format
    definitions
        The definitions for the features

    Returns
    -------
    FeatureSet
        The created feature set.
    """
    # create dictionary of definitions to match
    def_dict = {str(x.id): x for x in definitions}
    return FeatureSet([FeatureValue(def_dict[k], v) for k, v in data.items()])


def parse_evaluation(data: Dict) -> BDHEvaluation:
    """Parse the evaluation information for a reading.

    Parameters
    ----------
    data
        The header of the BDH json file

    Returns
    -------
    BDHEvaluation
        The evaluation information for the reading

    Raises
    ------
    ValueError
        If the evaluation id is missing from the data.
    """
    # TODO support interruptions
    if KEYS.id not in data:
        raise ValueError("Missing evaluation id")
    evaluation_id = data[KEYS.id]

    if KEYS.user_id not in data:
        raise ValueError("Missing user id")
    user_id = data[KEYS.user_id]

    if KEYS.effective_time_frame not in data:
        raise ValueError("Missing effective time frame")

    epoch = parse_epoch(data[KEYS.effective_time_frame])

    if KEYS.schema_id not in data or KEYS.name not in data[KEYS.schema_id]:
        raise ValueError("Missing task or schema information")

    task = data[KEYS.schema_id][KEYS.name]

    finished = data[KEYS.completion] == "completed"

    try:
        exit_reason = data[KEYS.interruption_reason]
    except KeyError:
        exit_reason = None

    # flatten header information
    header_meta = data.copy()
    header_meta.pop("configuration", None)
    header_meta.pop("raw_data", None)
    header_meta_flat = flatten(header_meta)

    if (key := "cluster_activity_codes") in header_meta_flat:
        if len(header_meta_flat[key]) > 0:
            header_meta_flat[key] = header_meta_flat[key][0]
    if (key := "effective_time_frame_idle_times") in header_meta_flat:
        header_meta_flat[key] = str(header_meta_flat[key])

    return BDHEvaluation(
        start=epoch.start,
        end=epoch.end,
        definition=EpochDefinition(id_=task),
        uuid=evaluation_id,
        finished=finished,
        user_id=user_id,
        exit_reason=exit_reason,
        header_meta=header_meta_flat,
    )


def get_context(config: Optional[Dict[str, Any]], schema_name: str) -> Context:
    """
    Create a context from a config dictionary.

    Parameters
    ----------
    config
        An optional dict that contains the raw information about the context

    schema_name
        The name of the schema

    Returns
    -------
    Context
        The parsed context

    """
    if not config:
        return Context()

    schema_name.split("-")[0]
    _context = [
        Value(ValueDefinition(key, key), value) for key, value in config.items()
    ]

    ref_table = 2
    if "reference_table_type" in config:
        ref_table = config["reference_table_type"]

    _context += translate_reference_table_type(ref_table)
    _context += translate_sequence_type("random")

    if "drawing_hand" in config:
        _context.append(
            Value(ValueDefinition("usedHand", "usedHand"), config["drawing_hand"])
        )
        _context.append(
            Value(
                ValueDefinition("levelType", "levelType"),
                MAPPING_SHAPE_LEVEL_TYPE[config["drawing_figure_name"]],
            )
        )

    return Context(_context)


def enrich_context(schema, level_id, context):
    """Enrich the context with level specific information."""
    if "cps" in schema.name:
        if level_id == "digit_to_digit":
            duration = EXPECTED_DURATION_D2D
        elif level_id == "symbol_to_digit":
            duration = EXPECTED_DURATION_S2D
        else:
            raise ValueError(f"Unknown level id: {level_id}")
        context.set(duration, LEVEL_DURATION_DEF)


def parse_level(
    data: dict,
    schema: ReadingSchema,
    feature_definitions: List[ValueDefinition],
    raw_data_definitions: Dict[Any, RawDataSetDefinition],
) -> Level:
    """Parse level id from dictionary containing level info.

    Parameters
    ----------
    data
        Dictionary containing level info
    schema
        The schema from data header
    feature_definitions
        list of feature definitions
    raw_data_definitions
        Dict of raw data definitions

    Returns
    -------
    Level
         The parsed level
    """
    # pylint: disable=unused-argument
    # TODO investigate how to use feature_definitions
    raw_data_sets = []
    config = data.get(KEYS.configuration)
    if not config:
        id_ = data["name"]
        if id_ == "6mw":
            id_ = "6mwt"
    else:
        id_ = get_level_id(config, schema)
    # Initialize feature_set to None
    feature_set = None

    # Specific case of NeuroQol where we already have computed features.
    if schema.name == "neuroqol-activity":
        feature_set = FeatureSet(
            values=[
                float(data["mobile_computed_features"]["t_score"]),
                float(data["mobile_computed_features"]["standard_error"]),
            ],
            definitions=[
                ValueDefinition(
                    f"mobile_computed_theta_score_{id_}",
                    f"The mobile computed theta score of the subtest {id_}.",
                ),
                ValueDefinition(
                    f"mobile_computed_standard_error_{id_}",
                    f"The mobile computed standard error of the subtest {id_}.",
                ),
            ],
        )

    if KEYS.raw_data in data:
        raw_data_sets = parse_raw_data_sets(data[KEYS.raw_data], raw_data_definitions)
        for key, function in filter(
            lambda item: item[0] in raw_data_definitions, TO_CONVERT.items()
        ):
            new_data_set = function(data[KEYS.raw_data][key], raw_data_definitions[key])

            # Drop key
            raw_data_sets = [x for x in raw_data_sets if x.id != key]
            raw_data_sets.append(new_data_set)

    epoch = parse_epoch(data[KEYS.effective_time_frame])

    context = get_context(config, schema.name)

    enrich_context(schema, id_, context)

    return Level(
        id_=id_,
        start=epoch.start,
        end=epoch.end,
        context=context,
        raw_data_sets=raw_data_sets,
        feature_set=feature_set,
    )


def parse_levels(
    data: dict,
    schema: ReadingSchema,
    feature_definitions: List[ValueDefinition],
    raw_data_definitions: Dict[Any, RawDataSetDefinition],
) -> List[Level]:
    """Parse levels from dict.

    Parameters
    ----------
    data
        Dict containing data body
    schema
        The schema from data header
    feature_definitions
        list of feature definitions
    raw_data_definitions
        Dict of raw data definitions

    Returns
    -------
    List[Level]
        list of levels

    Raises
    ------
    ValueError
        If a level property is missing in ``data``.
    """
    if KEYS.levels not in data:
        raise ValueError(f"Missing {KEYS.levels} property")

    data = create_levels(data, schema)
    raw_data_definitions = update_raw_data_definition(raw_data_definitions, schema)
    return [
        parse_level(level_data, schema, feature_definitions, raw_data_definitions)
        for level_data in data[KEYS.levels]
    ]


def parsable_bdh_json(value: Any) -> bool:
    """Test if a value is a dictionary and contains BDH specific keys."""
    if not isinstance(value, dict):
        return False

    return ("header" in value.keys()) & ("body" in value.keys())


@register_reader(parsable_bdh_json, BDHReading)
def parse_bdh_reading(data: Dict) -> BDHReading:
    """Get class representation of dictionary representation.

    Parameters
    ----------
    data
        The dictionary containing the information about the reading

    Returns
    -------
    Reading
        The class representation of the record passed.

    Raises
    ------
    ValueError
        If header, schema or body information is missing in ``data``.
    """
    if KEYS.header not in data:
        raise ValueError("Missing header information")

    data_header = data[KEYS.header]

    if KEYS.schema_id not in data_header:
        raise ValueError("Missing schema information")

    schema = parse_schema(data_header[KEYS.schema_id])
    evaluation = parse_evaluation(data_header)

    feature_definitions = []

    if KEYS.features in data_header:
        header_features = data_header[KEYS.features]
        if "mobile_computed_features" in header_features:
            mcf = header_features["mobile_computed_features"]
            if "activity_features" in mcf:
                for id_, data_def in mcf["activity_features"].items():
                    feature_definitions.append(
                        parse_feature_definition("mobile_" + id_, data_def)
                    )
        if "activity_features" in header_features:
            for id_, data_def in header_features["activity_features"].items():
                feature_definitions.append(
                    parse_feature_definition("pre-existing_" + id_, data_def)
                )

    raw_data_definitions = {}
    if KEYS.raw_data in data_header:
        for id_, data_def in data_header[KEYS.raw_data].items():
            raw_data_definitions[id_] = parse_raw_data_set_definition(id_, data_def)

    if KEYS.body not in data:
        raise ValueError("Missing body")

    data_body = data[KEYS.body]

    device = parse_device(data_header[KEYS.acquisition_provenance][KEYS.source_device])

    parsed_levels = parse_levels(
        data_body, schema, feature_definitions, raw_data_definitions
    )

    session = parse_session(data_header)

    res = BDHReading(
        evaluation=evaluation,
        schema=schema,
        levels=parsed_levels,
        feature_set=None,
        date=data_header.get(KEYS.creation_date_time, None),
        device=device,
        session=session,
    )

    return res


def read_bdh(path: str) -> Reading:
    """Read a *BDH* data record.

    Parameters
    ----------
    path
        The path to the reading to be parsed

    Returns
    -------
    Reading
        The class representation of the record parsed.
    """
    data = load_json(path, encoding="utf-8")
    return parse_bdh_reading(data)
