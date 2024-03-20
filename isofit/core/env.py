import logging
from configparser import ConfigParser
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

Logger = logging.getLogger(__file__)

INI: Path = Path.home() / ".isofit/isofit.ini"
KEYS: List[str] = ["data", "examples", "srtmnet", "sixs", "modtran"]
CONFIG: ConfigParser = ConfigParser()
SECTION: str = "DEFAULT"

# Initialize ConfigParser with default values
for key in KEYS:
    CONFIG[SECTION][key] = str(Path.home() / f".isofit/{key}")


def __getattr__(key: str) -> Optional[str]:
    """
    Retrieves a value from CONFIG[SECTION] if the key doesn't exist on the module already.

    Parameters
    ----------
    key : str
        The key to retrieve the value for.

    Returns
    -------
    str or None
        The value associated with the key if it exists in CONFIG[SECTION], otherwise None.
    """
    return CONFIG[SECTION].get(key)


def mkdir(path: str, isdir: bool = False) -> None:
    """
    Create a directory at the given path.

    Parameters
    ----------
    path : str
        The path where the directory should be created.
    isdir : bool, optional
        Flag indicating if the provided path points to a directory.
        If False, the directory will be created at the parent of the provided path.
        Defaults to False.
    """
    path = Path(path)
    if not isdir:
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)


def items() -> Iterable[Tuple[str, str]]:
    """
    Passthrough to the items() function on the working section of the config.
    """
    return CONFIG[SECTION].items()


def changeSection(section: str) -> None:
    """
    Changes the working section of the config.

    Parameters
    ----------
    section : str
        The section of the config to reference for lookups.
    """
    global SECTION
    SECTION = section


def changePath(key: str, value: str) -> None:
    """
    Change the path associated with the specified key in the CONFIG[SECTION].

    Parameters
    ----------
    key : str
        The key whose path needs to be changed.
    value : str or Path
        The new path to associate with the key.
    """
    CONFIG[SECTION][key] = str(Path(value).absolute())


def load(ini: Optional[str] = None, section: Optional[str] = None) -> None:
    """
    Load environment variables from an ini file.

    Parameters
    ----------
    ini : str or Path, optional
        The path to the INI file containing config variables. If None, the default INI file path is used.
        If provided, sets the global INI for the remainder of the session.
    section : str, optional
        Sets the working section for the session. Key lookups will use this section.
    """
    global INI
    if ini:
        INI = Path(ini)

    if section:
        changeSection(section)

    if INI.exists():
        CONFIG.read(INI)
        Logger.info(f"Loaded ini from: {INI}")
    else:
        Logger.info(f"ini does not exist, falling back to defaults: {INI}")


def save(ini: Optional[str] = None) -> None:
    """
    Save CONFIG variables to the INI (ini) file.

    Parameters
    ----------
    ini : str or Path, optional
        The path to save the config variables to. If None, the default INI file path is used.
        If provided, sets the global INI for the remainder of the session.
    """
    global INI
    if ini:
        INI = Path(ini)

    mkdir(INI)

    try:
        with open(INI, "w") as file:
            CONFIG.write(file)
    except:
        Logger.exception(f"Failed to dump ini to file: {INI}")
