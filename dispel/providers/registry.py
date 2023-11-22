"""TODO: Write documentation about registry."""
import importlib
import inspect
import pkgutil
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union

from dispel import providers
from dispel.data.core import Reading
from dispel.data.values import AbbreviatedValue
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing import ProcessingStepsType, process
from dispel.processing.data_trace import DataTrace

ReaderRegistryT = Dict[Tuple[Callable[[Any], bool], Type[Reading]], Dict]
READERS: ReaderRegistryT = {}

PROCESSING_FUNCTIONS: Dict[str, Dict[type, Callable[[Reading], DataTrace]]] = {}
PROCESSING_STEPS: Dict[Tuple[Iterable[str], Type[Reading]], ProcessingStepsType] = {}

PROVIDERS = {}


def register_reader(
    readable: Callable[[Any], bool],
    reading_type: Type[Reading],
    registry: Optional[ReaderRegistryT] = None,
) -> Callable:
    """Decorate a function to serve as a reader.

    TODO: write documentation
    """
    registry = registry or READERS

    def wrap(func):
        if (idx := (readable, reading_type)) in registry:
            raise ValueError("Reader have to be unique per readable and reading_type")

        # TODO: consider replacing dictionary with data class for type safety
        registry[idx] = {
            "func": func,
            "readable": readable,
            "reading_type": reading_type,
        }

        return func

    return wrap


def register_processing(
    task_name: Union[str, AV],
    steps: ProcessingStepsType,
    codes: Union[str, Tuple[str, ...]],
    supported_type: Type[Reading] = Reading,
) -> Callable:
    """Register a processing function and steps for automatic processing.

    Parameters
    ----------
    task_name
        The standard task name.
    steps
        The processing steps to be registered.
    codes
        The possible task code acronyms.
    supported_type
        The supported reading type by the processing function.

    Returns
    -------
    Callable
        The decorated function.
    """

    def func(reading: Reading) -> DataTrace:
        if not isinstance(reading, supported_type):
            raise ValueError(
                f"Unsupported reading type: {type(reading)}. "
                f"Expected {supported_type}."
            )
        return process(reading, steps, task_name=task_name)

    if isinstance(codes, str):
        codes = (codes,)

    for code in codes:
        if code not in PROCESSING_FUNCTIONS:
            PROCESSING_FUNCTIONS[code] = {}

        if supported_type in PROCESSING_FUNCTIONS[code]:
            raise ValueError(
                f"Already registered processing function for {code} and "
                f"{supported_type}"
            )
        PROCESSING_FUNCTIONS[code][supported_type] = func

    PROCESSING_STEPS[(codes, supported_type)] = steps

    return func


def get_processing_function(
    code: str, reading_type: type
) -> Callable[[Reading], DataTrace]:
    """FIXME: documentation."""
    if code not in PROCESSING_FUNCTIONS:
        raise ValueError(f"{code} missing in {PROCESSING_FUNCTIONS.keys()} keys.")

    hierarchy = inspect.getmro(reading_type)
    for cls in hierarchy:
        if cls in PROCESSING_FUNCTIONS[code]:
            return PROCESSING_FUNCTIONS[code][cls]

    raise ValueError(
        f"{reading_type} not supported for {code}. Supported types: "
        f"{PROCESSING_FUNCTIONS[code].keys()}"
    )


def process_factory(
    task_name: Union[str, AbbreviatedValue],
    steps: ProcessingStepsType,
    codes: Union[str, Tuple[str, ...]],
    supported_type: Type[Reading] = Reading,
) -> Callable[[Reading], DataTrace]:
    """Register and return the corresponding processing function.

    Parameters
    ----------
    task_name
        The standard task name.
    steps
        The processing step(s) to be registered.
    codes
        The possible task code acronyms.
    supported_type
        The type of reading supported for the processing

    Returns
    -------
    Callable[[Reading], DataTrace]
        The decorated processing function.
    """
    return register_processing(task_name, steps, codes, supported_type)


def _load_provider_io(provider_module):
    mod_io = None
    try:
        mod_io = importlib.import_module(f"{provider_module.__name__}.io")
    except ModuleNotFoundError:
        pass
    return mod_io


def _load_provider_tasks(provider_module):
    test_mod_iter = pkgutil.iter_modules(
        [provider_module.__path__[0] + "/tasks"],
        provider_module.__name__ + ".tasks.",
    )
    tests = {}
    for finder, name, _ in test_mod_iter:
        tests[name] = importlib.import_module(name)

    return tests


def _load_provider(provider_name: str):
    """Load provider modules and trigger registration of functionality."""
    provider_module = importlib.import_module(provider_name)
    mod_io = _load_provider_io(provider_module)
    tests = _load_provider_tasks(provider_module)

    # TODO replace dictionary entries with data class for type safety
    return {"provider": provider_module, "io": mod_io, "tests": tests}


def discover_providers():
    """Discover providers from the provider package.

    TODO: describe the discovery process
    """
    provider_iter = pkgutil.iter_modules(providers.__path__, providers.__name__ + ".")
    for finder, name, ispkg in provider_iter:
        if ispkg and name not in PROVIDERS:
            PROVIDERS[name] = _load_provider(name)


discover_providers()
