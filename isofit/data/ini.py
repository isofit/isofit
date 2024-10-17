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
    dirs: List[str] = ["data", "examples", "imagecube", "srtmnet", "sixs", "modtran"]
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

    def changeBase(self, base: str) -> None:
        """
        Changes the base path for each directory.

        Parameters
        ----------
        base : str
            Path to base directory to set
        """
        self.base = Path(base)

        # Re-initialize
        for key in self.dirs:
            self.changePath(key, self.base / key)

    def changeSection(self, section: str) -> None:
        """
        Changes the working section of the config.

        Parameters
        ----------
        section : str
            The section of the config to reference for lookups.
        """
        self.section = section

        if section not in self.config:
            self.config[section] = {}

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
        self.config[self.section][key] = str(Path(value).resolve())

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
            self.changeSection(section)

        if self.ini.exists():
            self.config.read(self.ini)

            # Retrieve the absolute path
            for key in self.dirs:
                self.changePath(key, self[key])

            Logger.info(f"Loaded ini from: {self.ini}")
        else:
            Logger.info(f"ini does not exist, falling back to defaults: {self.ini}")

    def save(self, ini: Optional[str] = None, diff_only: bool = True) -> None:
        """
        Save CONFIG variables to the INI (ini) file.

        Parameters
        ----------
        ini : str or Path, optional
            The path to save the config variables to. If None, the default INI file path is used.
            If provided, sets the global INI for the remainder of the session.
        diff_only : bool, default=True
            Only save if there is a difference between the currently existing ini file and the config in memory.
            If False, will save regardless, possibly overwriting an existing file
        """
        if ini:
            self.ini = Path(ini)

        self.ini.parent.mkdir(parents=True, exist_ok=True)

        save = True
        if diff_only:
            if self.ini.exists():
                save = False

                current = ConfigParser()
                current.read(self.ini)

                if current != self.config:
                    save = True

        if save:
            try:
                with open(self.ini, "w") as file:
                    self.config.write(file)
                print(f"Wrote to file: {self.ini}")
            except:
                Logger.exception(f"Failed to dump ini to file: {self.ini}")

    def path(self, dir: str, path: str) -> Path:
        """
        Retrieves a path under one of the env directories and validates the path exists.

        Parameters
        ----------
        dir : str
            One of the env directories, eg. "data", "examples"
        path : str
            Path to a file under the `dir`

        Returns
        -------
        pathlib.Path
            Validated full path
        """
        self.validate([dir], debug=Logger.debug, error=Logger.error)

        path = (Path(self[dir]) / path).resolve()

        if not path.exists():
            Logger.error(
                f"The following path does not exist, please verify your installation environment: {path}"
            )

        return path

    @staticmethod
    def validate(keys: List) -> bool:
        """
        Validates known products.

        Parameters
        ----------
        keys : list
            List of products to validate
        """
        # Should never be raised as this function is defined and set in isofit.data.cli.__init__
        # If this is hit, there's a critical environment issue
        raise NotImplementedError(
            "ISOFIT failed to attach the validation function to this object"
        )
