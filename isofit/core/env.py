"""
Constructs and stores environment options for ISOFIT
"""

import logging
from copy import copy
from pathlib import Path

import toml

Logger = logging.getLogger(__file__)

BASE = Path.home() / ".isofit"
DATA = {
    "env": BASE / "env.toml",
    "data": BASE / "data",
    "examples": BASE / "examples",
    "srtmnet": BASE / "sRTMnet",
    "sixs": BASE / "sixs",
    "modtran": BASE / "modtran",
}


def __getattr__(key):
    """
    Retrieves a key from DATA if it exists, otherwise raises an exception
    """
    if key in DATA:
        return DATA[key]


def setEnv(key, value):
    """ """
    if key in DATA:
        DATA[key] = Path(value).absolute()
    elif key == "base":
        updateBase(value)
    else:
        Logger.error(f"Key is not a valid option: {key}")


def updateBase(value):
    """ """
    global BASE

    new = Path(value)

    # Update the value if it was the default
    # If this value is already different, do not change
    for key, value in DATA.items():
        if value == BASE / value.name:
            DATA[key] = new / value.name

    # Update the base
    BASE = new


def keys():
    """
    Passthrough
    """
    return DATA.keys()


def items():
    """
    Passthrough
    """
    return DATA.items()


def dump():
    """ """
    # TOML doesn't support PosixPath, cast back to str
    data = copy(DATA)
    for key, val in data.items():
        if isinstance(val, Path):
            data[key] = str(val)

    return data


def mkdir(path=None):
    """ """
    path = Path(path)
    if not (base := path.parent).exists():
        base.mkdir(parents=True, exist_ok=True)


def loadEnv(base=None, env=None):
    """
    Loads an environment file

    Parameters
    ----------
    base: str
        Override the base directory
    env: str
        Override the environment file path
    """
    base = Path(base or BASE)
    if not (env := env or DATA["env"]):
        env = base / "env.toml"

    load = DATA
    if Path(env).exists():
        load.update(toml.load(env))
    else:
        env = "defaults"

    for key in keys():
        val = Path(load.get(key))
        setEnv(key, val)

    Logger.debug(f"Loaded env from: {env}")


def dumpEnv(base=None, env=None):
    """
    Dumps to an environment file

    Parameters
    ----------
    base: str
        Override the base directory
    env: str
        Override the environment file path
    """
    base = Path(base or BASE)
    if not (env := env or DATA["env"]):
        env = base / "env.toml"

    # Make sure the parent directories exist
    mkdir(env)
    with open(env, "w") as file:
        toml.dump(dump(), file)

    Logger.debug(f"Writing env to: {env}")
