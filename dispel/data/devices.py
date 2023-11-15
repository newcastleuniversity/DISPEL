"""Data structures to support provenance on data acquisition devices."""

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class PlatformType(Protocol):
    """A type specifying the platform used to record a reading."""

    def repr(self) -> str:
        """Get a string representation of the platform."""
        ...


class IOSPlatform(PlatformType):
    """iOS device used to record a reading."""

    def repr(self) -> str:
        """Get a string representation of the platform."""
        return "iOS"


class AndroidPlatform(PlatformType):
    """Android device used to record a reading."""

    def repr(self) -> str:
        """Get a string representation of the platform."""
        return "Android"


@dataclass
class Screen:
    """Screen properties of a device."""

    #: The width of the screen in pixels
    width_pixels: int
    #: The height of the screen in pixels
    height_pixels: int
    #: The width of the screen in physical units
    density_dpi: Optional[int] = None
    #: The height of the screen in physical units
    width_dp_pt: Optional[int] = None
    #: The number of pixels along the diagonal of the screen per inch.
    height_dp_pt: Optional[int] = None

    def __post_init__(self):
        if self.width_pixels < 1:
            raise ValueError("screen width cannot be smaller than 1")
        if self.height_pixels < 1:
            raise ValueError("screen height cannot be smaller than 1")


class AndroidScreen(Screen):
    """Screen properties of an Android device.

    Attributes
    ----------
    x_dpi
        The exact physical pixels per inch of the screen in the X dimension.
    y_dpi
        The exact physical pixels per inch of the screen in the Y dimension.
    """

    def __init__(
        self,
        width_pixels: int,
        height_pixels: int,
        density_dpi: int,
        x_dpi: int,
        y_dpi: int,
    ):
        super().__init__(width_pixels, height_pixels, density_dpi)
        if x_dpi < 1:
            raise ValueError("X-dpi cannot be smaller than 1")
        if y_dpi < 1:
            raise ValueError("Y-dpi cannot be smaller than 1")

        self.x_dpi = x_dpi
        self.y_dpi = y_dpi


class IOSScreen(Screen):
    """Screen properties of an Android device.

    Attributes
    ----------
    scale_factor
        The screen scaling factor to convert logical pixels into physical pixels.
    """

    def __init__(
        self,
        width_pixels: int,
        height_pixels: int,
        density_dpi: int,
        scale_factor: int,
        width_dp_pt: int,
        height_dp_pt: int,
    ):
        super().__init__(
            width_pixels, height_pixels, density_dpi, width_dp_pt, height_dp_pt
        )
        self.scale_factor = scale_factor


@dataclass(frozen=True)
class Device:
    """A device used to capture the :class:`Reading`."""

    #: A unique device identifier. Can be a hardware or vendor identifier
    uuid: Optional[str] = None
    #: The platform on which the device runs (e.g., iOS, Android, ...).
    platform: Optional[PlatformType] = None
    #: The model of the phone.
    model: Optional[str] = None
    #: The code used by the manufacturer to identify the model
    model_code: Optional[str] = None
    #: The OS or kernel version depending on the platform.
    os_version: Optional[str] = None
    #: The version of the application running on the device
    app_version_number: Optional[str] = None
    #: The build number of the application running on the device
    app_build_number: Optional[str] = None
    #: The screen properties of the device
    screen: Optional[Screen] = None
