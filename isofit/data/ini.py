"""
ISOFIT environment module
"""

import logging
from configparser import ConfigParser
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

Logger = logging.getLogger(__file__)


class Ini:
    base: Path = Path.home() / ".isofit/"
    dirs: List[str] = ["data", "examples", "srtmnet", "sixs", "modtran", "hypertrace"]
    config: ConfigParser = ConfigParser()
    section: str = "DEFAULT"

    def __init__(self):
        self.ini = self.base / "isofit.ini"

        # Initialize ConfigParser with default values
        for key in self.dirs:
            self.changePath(key, self.base / key)

        self.load()

    def __getattr__(self, key: str) -> Optional[str]:
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
        return self.config[self.section].get(key)

    def __getitem__(self, key: str) -> Optional[str]:
        """
        Simple passthrough function to __getattr__

        Parameters
        ----------
        key : str
            The key to retrieve the value for.

        Returns
        -------
        str or None
            The value associated with the key if it exists in CONFIG[SECTION], otherwise None.
        """
        return getattr(self, key)

    def __iter__(self) -> Iterable:
        return iter(self.config[self.section])

    def keys(self) -> Iterable[str]:
        return iter(self.config[self.section])

    def items(self) -> Iterable[Tuple[str, str]]:
        """
        Passthrough to the items() function on the working section of the config.
        """
        return self.config[self.section].items()

    def changeSection(self, section: str) -> None:
        """
        Changes the working section of the config.

        Parameters
        ----------
        section : str
            The section of the config to reference for lookups.
        """
        self.section = section

    def changePath(self, key: str, value: str) -> None:
        """
        Change the path associated with the specified key in the CONFIG[SECTION].

        Parameters
        ----------
        key : str
            The key whose path needs to be changed.
        value : str or Path
            The new path to associate with the key.
        """
        self.config[self.section][key] = str(Path(value).absolute())

    def load(self, ini: Optional[str] = None, section: Optional[str] = None) -> None:
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
        if ini:
            self.ini = Path(ini)

        if section:
            changeSection(section)

        if self.ini.exists():
            self.config.read(self.ini)

            # Retrieve the absolute path
            for key in self.dirs:
                self.changePath(key, self[key])

            Logger.info(f"Loaded ini from: {self.ini}")
        else:
            Logger.info(f"ini does not exist, falling back to defaults: {self.ini}")

    def save(self, ini: Optional[str] = None) -> None:
        """
        Save CONFIG variables to the INI (ini) file.

        Parameters
        ----------
        ini : str or Path, optional
            The path to save the config variables to. If None, the default INI file path is used.
            If provided, sets the global INI for the remainder of the session.
        """
        if ini:
            self.ini = Path(ini)

        self.ini.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.ini, "w") as file:
                self.config.write(file)
        except:
            Logger.exception(f"Failed to dump ini to file: {self.ini}")

    @staticmethod
    def validate(keys: List) -> bool:
        """
        Validates known products. This function is defined by isofit.data.cli.__init__.py
        """
        raise NotImplemented(
            "ISOFIT failed to attach the validation function to this object"
        )
