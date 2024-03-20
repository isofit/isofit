from pathlib import Path
from typing import Optional

import toml

HOME: Path = Path.home()
TOML: Path = HOME / ".isofit/isofit.toml"
KEYS: list[str] = ["data", "examples", "srtmnet", "sixs", "modtran"]
DATA: dict[str, str] = {key: str(HOME / f".isofit/{key}") for key in KEYS}


def __getattr__(key: str) -> Optional[str]:
    """
    Retrieves a value from DATA if the key doesn't exist on the module already.

    Parameters
    ----------
    key : str
        The key to retrieve the value for.

    Returns
    -------
    str or None
        The value associated with the key if it exists in DATA, otherwise None.
    """
    return DATA.get(key)


def changePath(key: str, value: str) -> None:
    """
    Change the path associated with the specified key in the DATA dictionary.

    Parameters
    ----------
    key : str
        The key whose path needs to be changed.
    value : str or Path
        The new path to associate with the key.
    """
    DATA[key] = str(Path(value).absolute())


def loadEnv(env: Optional[str] = None) -> None:
    """
    Load environment variables from a TOML file.

    Parameters
    ----------
    env : str or Path, optional
        The path to the TOML file containing environment variables. If None, the default TOML file path is used.
        If provided, sets the global TOML for the remainder of the session.
    """
    global TOML
    if env:
        TOML = Path(env)

    if TOML.exists():
        DATA.update(toml.load(TOML))
        print(f"Loaded env from: {TOML}")
    else:
        print(f"Env does not exist, falling back to defaults: {TOML}")


def saveEnv(env: Optional[str] = None, mkdir: bool = True) -> None:
    """
    Save DATA variables to the TOML file.

    Parameters
    ----------
    env : str or Path, optional
        The path to save the environment variables to. If None, the default TOML file path is used.
        If provided, sets the global TOML for the remainder of the session.
    mkdir : bool, optional
        Whether to create directories in the path if they do not exist. Default is True.
    """
    global TOML
    if env:
        TOML = Path(env)

    if mkdir:
        TOML.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(TOML, "w") as file:
            toml.dump(DATA, file)
    except Exception as e:
        print(f"Failed to dump env to file: {TOML}")
