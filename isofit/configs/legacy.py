"""
Upgrades configuration dicts written for older versions of ISOFIT.

The config models in :mod:`isofit.configs.sections` only accept the current
schema. Configs written for older versions of ISOFIT are converted here first,
before they are overlaid onto a base config and validated (see
:func:`isofit.configs.loader.resolve_config_dict`). Keeping every legacy fix in
this module keeps the models free of backwards-compatibility aliases.

The legacy layouts handled are:

- ``forward_model.radiative_transfer``, holding the engine settings under
  ``radiative_transfer_engines`` alongside the ``statevector``, ``lut_grid``,
  and ``unknowns``::

      forward_model:
        radiative_transfer:
          radiative_transfer_engines:
            vswir:
              engine_name: modtran
              lut_path: ...
          statevector: {...}

- ``forward_model.atmosphere`` with engine and atmosphere settings mixed
  together, either flat at the atmosphere level or all nested under
  ``engine`` (as written by older versions of ``apply_oe``)::

      forward_model:
        atmosphere:
          engine:
            engine_name: sRTMnet
            engine_base_dir: ...
            lut_path: ...

Both are converted to the current layout, where ``engine`` holds only the fields
of the selected engine's model::

    forward_model:
      atmosphere:
        engine:
          name: srtmnet
          sixs_dir: ...
        lut_path: ...

Renamed keys are also updated (e.g. ``engine_name`` -> ``name``, ``lut_names``
-> ``lut_subset``). Configs already in the current schema pass through unchanged.
"""

from __future__ import annotations

import logging
import typing
from copy import deepcopy

from isofit.configs.sections.atmosphere.engines import Engines, standardName

Logger = logging.getLogger(__name__)

# Legacy engine names that `standardName` does not map to a current engine
EngineNames = {
    "kernelflowsgp": "kernelflows",
}

# Renamed keys, {legacy: current}, for keys of the atmosphere and its engine
RenamedKeys = {
    "engine_name": "name",
    "engine_base_dir": "base_dir",
    "lut_names": "lut_subset",
    "obs_file": "obs",
}

# Per-engine overrides of RenamedKeys
RenamedEngineKeys = {
    # sRTMnet's engine_base_dir held the path to 6S, not to sRTMnet
    "srtmnet": {"engine_base_dir": "sixs_dir"},
}

# Keys that are fields of both the atmosphere and (some) engines
SharedKeys = {"multipart_transmittance"}


def engine_fields() -> dict[str, set[str]]:
    """
    Map each engine name to the set of fields its config model accepts.

    Introspects the ``Engines`` discriminated union so this stays in sync as
    engines are added or removed.

    Returns
    -------
    dict[str, set[str]]
        Mapping of standardized engine name to its model's field names.
    """
    union, _discriminator = typing.get_args(Engines)

    fields = {}
    for member in typing.get_args(union):
        model, tag = typing.get_args(member)
        fields[tag.tag] = set(model.model_fields)

    return fields


def engine_name(name: str) -> str:
    """
    Convert an engine name, possibly a legacy one, to its current name.

    Parameters
    ----------
    name : str
        Engine name, e.g. ``"6s"``, ``"sRTMnet"``, or ``"KernelFlowsGP"``.

    Returns
    -------
    str
        The standardized engine name, e.g. ``"sixs"``.
    """
    name = standardName(name)
    return EngineNames.get(name, name)


def rename(data: dict, renames: dict) -> dict:
    """
    Rename legacy keys of a dict.

    Parameters
    ----------
    data : dict
        Dict to rename keys of. Not mutated.
    renames : dict
        Mapping of legacy key to current key.

    Returns
    -------
    dict
        Copy of ``data`` with legacy keys renamed. If both a legacy key and its
        replacement are set, the replacement wins.
    """
    data = dict(data)

    for old, new in renames.items():
        if old not in data:
            continue

        value = data.pop(old)
        if new in data:
            Logger.warning(
                f"Both the legacy key {old!r} and its replacement {new!r} are set, ignoring {old!r}"
            )
        else:
            Logger.debug(f"Renamed legacy key {old!r} to {new!r}")
            data[new] = value

    return data


def convert_radiative_transfer(rt: dict) -> dict:
    """
    Convert a legacy ``forward_model.radiative_transfer`` section to a (still
    possibly legacy) ``forward_model.atmosphere`` section.

    Parameters
    ----------
    rt : dict
        The legacy ``radiative_transfer`` section.

    Returns
    -------
    dict
        The settings of the first engine merged with the section's other keys
        (``statevector``, ``lut_grid``, ...). Pass the result through
        :func:`convert_atmosphere` to split it into atmosphere/engine fields.
    """
    engines = rt.get("radiative_transfer_engines") or {}
    if isinstance(engines, dict):
        engines = list(engines.values())

    if len(engines) > 1:
        Logger.warning(
            "Legacy config defines multiple radiative_transfer_engines, only the first will be used"
        )

    atmosphere = dict(engines[0]) if engines else {}
    for key, value in rt.items():
        if key != "radiative_transfer_engines":
            atmosphere.setdefault(key, value)

    return atmosphere


def convert_atmosphere(atmosphere: dict) -> dict:
    """
    Split a ``forward_model.atmosphere`` section into atmosphere and engine
    fields, renaming legacy keys.

    Engine settings may be at the atmosphere level, nested under ``engine``, or
    both; keys set at the atmosphere level win on conflict. Keys that are fields
    of the selected engine are moved under ``engine``, and all others to the
    atmosphere level.

    Parameters
    ----------
    atmosphere : dict
        The ``atmosphere`` section. Not mutated.

    Returns
    -------
    dict
        The section in the current schema. If no engine is named the engine
        can't be selected, so the keys are renamed but not moved.
    """
    atmosphere = dict(atmosphere)

    engine = atmosphere.pop("engine", None) or {}
    if isinstance(engine, str):
        engine = {"name": engine}

    # Engine and atmosphere settings may be mixed in either dict
    flat = {**engine, **atmosphere}

    name = flat.get("name", flat.get("engine_name"))
    if not isinstance(name, str):
        # The engine can't be selected, so only rename
        atmosphere = rename(atmosphere, RenamedKeys)
        if engine:
            atmosphere["engine"] = rename(engine, RenamedKeys)
        return atmosphere

    name = engine_name(name)
    flat = rename(flat, {**RenamedKeys, **RenamedEngineKeys.get(name, {})})
    fields = engine_fields().get(name, set())

    engine = {"name": name}
    atmosphere = {}
    for key, value in flat.items():
        if key == "name":
            continue
        if key in fields:
            engine[key] = value
        if key not in fields or key in SharedKeys:
            atmosphere[key] = value

    atmosphere["engine"] = engine
    return atmosphere


def convert(config: dict) -> dict:
    """
    Convert a config dict written for an older version of ISOFIT to the
    current schema.

    Parameters
    ----------
    config : dict
        Full configuration dict. Not mutated.

    Returns
    -------
    dict
        The converted config. Configs already in the current schema are
        returned unchanged (as a copy).
    """
    config = deepcopy(config)

    forward_model = config.get("forward_model")
    if not isinstance(forward_model, dict):
        return config

    if "radiative_transfer" in forward_model:
        rt = forward_model.pop("radiative_transfer")
        if "atmosphere" in forward_model:
            Logger.warning(
                "Config defines both forward_model.atmosphere and the legacy "
                "forward_model.radiative_transfer, ignoring radiative_transfer"
            )
        elif isinstance(rt, dict):
            Logger.info(
                "Converting legacy forward_model.radiative_transfer to forward_model.atmosphere"
            )
            forward_model["atmosphere"] = convert_radiative_transfer(rt)

    if isinstance(forward_model.get("atmosphere"), dict):
        forward_model["atmosphere"] = convert_atmosphere(forward_model["atmosphere"])

    return config
