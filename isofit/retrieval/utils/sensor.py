"""
Sensor-specific config helpers.

Each supported sensor names its radiance files by its own convention, from which
the flightline ID (``fid``) is parsed. :func:`flightline_id` does the parsing,
:class:`~isofit.configs.sections.io.InputConfig` uses it to derive
``input.fid``, and :func:`derive_fid` applies it to a raw config dict ahead of
``${...}`` interpolation so that references like ``${input.fid}`` resolve.

A sensor is one of:

- a supported sensor, see :data:`FID_PARSERS`;
- a strptime format (contains ``%``) matching the start of the radiance file
  name, e.g. ``"emit%Y%m%dt%H%M%S"``, for custom sensors;
- ``"NA-*"``, for custom sensors whose file names carry no parsable date.

The config models and loader import this module, so it must not import
:mod:`isofit.configs`.
"""

from datetime import datetime
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

SENSOR_CHOICES = (
    f"one of {list(FID_PARSERS)}, a strptime format matching the start of the "
    "radiance file name (e.g. 'emit%Y%m%dt%H%M%S'), or 'NA-*'"
)


def is_datestr(sensor: str) -> bool:
    """Whether ``sensor`` is a strptime format rather than a sensor name."""
    return "%" in sensor


def is_supported(sensor: str) -> bool:
    """Whether ``sensor`` is any of the forms in :data:`SENSOR_CHOICES`."""
    return sensor in FID_PARSERS or is_datestr(sensor) or sensor.startswith("NA")


def match_datestr(name: str, datestr: str) -> tuple[str, datetime]:
    """
    Find the longest leading portion of ``name`` that matches a strptime format.

    Parameters
    ----------
    name : str
        The string to match, e.g. a radiance file name or flightline ID.
    datestr : str
        strptime format matching the start of ``name``.

    Returns
    -------
    tuple[str, datetime.datetime]
        The matched leading portion of ``name`` and the datetime parsed from it.

    Raises
    ------
    ValueError
        If no leading portion of ``name`` matches ``datestr``.

    Examples
    --------
    >>> from isofit.retrieval.utils.sensor import match_datestr
    >>> match_datestr("FAKE20190830-26_tile08", "FAKE%Y%m%d")
    ('FAKE20190830', datetime.datetime(2019, 8, 30, 0, 0))
    """
    for end in range(len(name), 0, -1):
        try:
            return name[:end], datetime.strptime(name[:end], datestr)
        except ValueError:
            continue

    raise ValueError(
        f'"{name}" does not start with a date matching datestr "{datestr}"'
    )


def flightline_id(sensor: str, radiance: Union[str, Path]) -> str:
    """
    Derive the flightline ID from a radiance file name per the sensor's naming
    convention.

    - Supported sensors use their parser in :data:`FID_PARSERS`.
    - A strptime format keeps the leading portion of the file name it matches.
    - ``"NA-*"`` keeps the full file name minus its extension and any trailing
      ``_rdn``.

    The strptime format is checked before ``"NA-*"`` so that a format which
    happens to start with ``NA`` is still treated as one.

    Parameters
    ----------
    sensor : str
        Any of the forms in :data:`SENSOR_CHOICES`.
    radiance : str or pathlib.Path
        Path to the radiance file.

    Returns
    -------
    str
        The flightline ID.

    Raises
    ------
    ValueError
        If ``sensor`` is not supported, or is a strptime format that does not
        match the start of the radiance file name.

    Examples
    --------
    >>> from isofit.retrieval.utils.sensor import flightline_id
    >>> flightline_id("emit", "emit20220820t102407_o23207_s000_l1b_rdn_b0106_v01")
    'emit20220820t102407'
    >>> flightline_id("emit%Y%m%dt%H%M%S", "data/emit20220820t102407_o23207_rdn")
    'emit20220820t102407'
    >>> flightline_id("NA-20220820", "data/scene_rdn.hdr")
    'scene'
    """
    name = Path(radiance).name

    if sensor in FID_PARSERS:
        return FID_PARSERS[sensor](name)

    if is_datestr(sensor):
        return match_datestr(name, sensor)[0]

    if sensor.startswith("NA"):
        return Path(name).stem.removesuffix("_rdn")

    raise ValueError(f"Unsupported sensor {sensor!r}, must be {SENSOR_CHOICES}")


def derive_fid(data: dict) -> None:
    """
    Fill ``input.fid`` from ``input.sensor`` and the radiance file, in place.

    Skipped when ``fid`` is already set, there is no sensor, or the radiance
    path is itself an unresolved ``${...}`` reference (e.g. the ``subs`` inputs,
    which are built from ``${input.fid}``). An unsupported sensor or a datestr
    that does not match is also skipped, leaving the error for
    :class:`~isofit.configs.sections.io.InputConfig` validation to report.

    Parameters
    ----------
    data : dict
        A full, uninterpolated config.

    Examples
    --------
    >>> from isofit.retrieval.utils.sensor import derive_fid
    >>> data = {"input": {"sensor": "emit", "rdn": "emit20220820t102407_rdn"}}
    >>> derive_fid(data)
    >>> data["input"]["fid"]
    'emit20220820t102407'
    """
    inputs = data.get("input") or {}
    if inputs.get("fid") is not None or not inputs.get("sensor"):
        return

    # The alias takes precedence over the field name, matching Pydantic
    # (see isofit.configs.loader.unalias, which normally runs first)
    rdn = inputs.get("rdn") or inputs.get("measured_radiance_file")
    if rdn is None or "${" in str(rdn):
        return

    try:
        inputs["fid"] = flightline_id(inputs["sensor"], rdn)
    except ValueError:
        pass
