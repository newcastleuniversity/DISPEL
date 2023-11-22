"""Tests for the :mod:`dispel.providers` module."""
import pkg_resources


def resource_path(provider: str, path: str) -> str:
    """Get the absolute path to a resource for a provider.

    It assumes that the resources for a provider are located in a ``_resources`` folder
    at the top level of the test folder for each provider
    (``tests/providers/[provider-name]/_resources/...``).

    Parameters
    ----------
    provider
        The name of the provider
    path
        The relative path of the resource

    Returns
    -------
    str
        The absolute path to the resource provided in ``path`` for the given
        ``provider``.
    """
    return pkg_resources.resource_filename(
        f"tests.providers.{provider}", f"_resources/{path}"
    )
