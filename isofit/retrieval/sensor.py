"""
Sensor-specific config helpers.

Each supported sensor names its radiance files by its own convention, from which
the flightline ID (``fid``) is parsed. :func:`flightline_id` does the parsing,
:class:`~isofit.configs.sections.io.InputConfig` uses it to derive
``input.fid``, and :func:`derive_fid` applies it to a raw config dict ahead of
``${...}`` interpolation so that references like ``${input.fid}`` resolve.

The config models and loader import this module, so it must not import
:mod:`isofit.configs`.
"""

from pathlib import Path
from typing import Callable, Union

# Flightline ID parsers keyed by sensor, applied to the radiance file name
FID_PARSERS: dict[str, Callable[[str], str]] = {
    "ang": lambda name: name[:18],
    "av3": lambda name: name[:18],
    "av4": lambda name: name[:-8],
    "av5": lambda name: name[:18],
    "avcl": lambda name: name[:16],
    "emit": lambda name: name[:19],
    "enmap": lambda name: name.split("_")[5],
    "gao": lambda name: name[:23],
    "hyp": lambda name: name[:22],
    "neon": lambda name: name[:21],
    "oci": lambda name: name[:24],
    "prism": lambda name: name[:18],
    "prisma": lambda name: name.split("_")[1],
    "tanager": lambda name: name[:23],
}


def is_custom_sensor(sensor: str) -> bool:
    """Whether ``sensor`` is a custom sensor ("custom" or "NA-*")."""
    return sensor == "custom" or sensor.startswith("NA")


def is_supported(sensor: str) -> bool:
    """Whether ``sensor`` is one of :data:`FID_PARSERS` or a custom sensor."""
    return sensor in FID_PARSERS or is_custom_sensor(sensor)


def flightline_id(sensor: str, radiance: Union[str, Path]) -> str:
    """
    Derive the flightline ID from a radiance file name per the sensor's naming
    convention.

    Custom sensors ("custom" or "NA-*") keep the full file name minus its
    extension and any trailing ``_rdn``.

    Parameters
    ----------
    sensor : str
        One of :data:`FID_PARSERS`, "custom", or "NA-*".
    radiance : str or pathlib.Path
        Path to the radiance file.

    Returns
    -------
    str
        The flightline ID.

    Raises
    ------
    ValueError
        If ``sensor`` is not supported.

    Examples
    --------
    >>> from isofit.retrieval.sensor import flightline_id
    >>> flightline_id("emit", "emit20220820t102407_o23207_s000_l1b_rdn_b0106_v01")
    'emit20220820t102407'
    >>> flightline_id("NA-20220820", "data/scene_rdn.hdr")
    'scene'
    """
    name = Path(radiance).name

    if sensor in FID_PARSERS:
        return FID_PARSERS[sensor](name)

    if is_custom_sensor(sensor):
        return Path(name).stem.removesuffix("_rdn")

    raise ValueError(
        f"Unsupported sensor {sensor!r}, must be one of {list(FID_PARSERS)}, "
        "'custom', or 'NA-*'"
    )


def derive_fid(data: dict) -> None:
    """
    Fill ``input.fid`` from ``input.sensor`` and the radiance file, in place.

    Skipped when ``fid`` is already set, there is no sensor, or the radiance
    path is itself an unresolved ``${...}`` reference (e.g. the ``subs`` inputs,
    which are built from ``${input.fid}``).

    Parameters
    ----------
    data : dict
        A full, uninterpolated config.

    Examples
    --------
    >>> from isofit.retrieval.sensor import derive_fid
    >>> data = {"input": {"sensor": "emit", "rdn": "emit20220820t102407_rdn"}}
    >>> derive_fid(data)
    >>> data["input"]["fid"]
    'emit20220820t102407'
    """
    inputs = data.get("input") or {}
    if inputs.get("fid") is not None or not inputs.get("sensor"):
        return

    # The alias takes precedence over the field name, matching Pydantic
    rdn = inputs.get("rdn") or inputs.get("measured_radiance_file")
    if rdn is None or "${" in str(rdn):
        return

    inputs["fid"] = flightline_id(inputs["sensor"], rdn)
