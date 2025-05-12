"""
ISOFIT environment module
"""

from __future__ import annotations

import json
import logging
import re
from configparser import ConfigParser
from copy import deepcopy
from pathlib import Path

Logger = logging.getLogger(__file__)


def getWorkingDir(config):
    """
    Attempts to detect if a configuration file sits in an ISOFIT working_directory

    Parameters
    ----------
    config : pathlib.Path
        Path to a config json file

    Returns
    -------
    wd : pathlib.Path | None
        Path to the working directory, if it's detected to be valid
    """
    # [working_directory]/config/config.json
    #              parent/parent/config.json
    wd = config.parent.parent.resolve()

    dirs = ("config", "data", "input", "lut_full", "output")
    for dir in dirs:
        if not (wd / dir).exists():
            return None

    Logger.info(f"Discovered working_directory to be: {wd}")
    return wd


class Ini:
    # Default directories to expect
    _dirs: List[str] = ["data", "examples", "imagecube", "srtmnet", "sixs", "plots"]

    # Additional keys with default values
    _keys: Dict[str, str] = {"srtmnet.file": "", "srtmnet.aux": ""}

    def __init__(self) -> None:
        self.reset()
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

    def __repr__(self) -> str:
        return f"[{self.section}]\n" + "\n".join(
            [f"{key} = {value}" for key, value in self.items()]
        )

    def keys(self) -> Iterable[str]:
        return iter(self.config[self.section])

    def items(self, kind: str = None) -> Iterable[Tuple[str, str]]:
        """
        Passthrough to the items() function on the working section of the config.

        Parameters
        ----------
        kind : "dirs" | "keys" | None
            Returns an iterable for the specific items:
            - "dirs" only keys in Ini._dirs
            - "dirs" only keys in Ini._keys
            - None returns combined both
        """
        items = self.config[self.section].items()
        if kind == "dirs":
            for key, value in items:
                if key in self._dirs:
                    yield key, value
        elif kind == "keys":
            for key, value in items:
                if key in self._keys:
                    yield key, value
        else:
            yield from items

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
        for key in self._dirs:
            self.changePath(key, self.base / key)

    def changeKey(self, key: str, value: str = "") -> None:
        """
        Change the value associated with the specified key in the CONFIG[SECTION].

        Parameters
        ----------
        key : str, dict
            Key to set. Alternatively, can be a dict to iterate over setting multiple
            keys at once.
        value : str, default=""
            The new value to associate with the key.
        """
        if isinstance(key, dict):
            for k, v in key.items():
                self.changeKey(k, v)
            return

        self.config[self.section][key] = str(value)

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
            for key in self._dirs:
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
                Logger.debug(f"Wrote to file: {self.ini}")
            except:
                Logger.exception(f"Failed to dump ini to file: {self.ini}")

    def path(
        self, dir: str, *path: List[str], key: str = None, template: bool = False
    ) -> Path:
        """
        Retrieves a path under one of the env directories and validates the path exists.

        Parameters
        ----------
        dir : str
            One of the env directories, eg. "data", "examples"
        *path : List[str]
            Path to a file under the `dir`
        key : str, default=None
            Optional key value to append to the resolved path. Assumes the path is a
            directory and the key will be a file name
        template : bool, default=False
            Returns the path as a template string. The path will still be validated,
            but the return will be "{env.[dir]}/*path", to be used with Ini.replace

        Returns
        -------
        pathlib.Path
            Validated full path

        Examples
        --------
        >>> from isofit.data import env
        >>> env.load()
        >>> env.path("data")
        ~/.isofit/data
        >>> env.path("examples", "20171108_Pasadena", "configs", "ang20171108t184227_surface.json")
        ~/.isofit/examples/20171108_Pasadena/configs/ang20171108t184227_surface.json
        >>> env.path("srtmnet", key="srtmnet.file")
        ~/.isofit/srtmnet/sRTMnet_v120.h5
        >>> env.path("srtmnet", key="srtmnet.aux")
        ~/.isofit/srtmnet/sRTMnet_v120_aux.npz
        """
        self.validate([dir], debug=Logger.debug, error=Logger.error)

        if template:
            path = Path("{env." + dir + "}", *path)
        else:
            path = Path(self[dir], *path).resolve()

        # Retrieve the value stored for the given key if it's set
        if key and self[key]:
            path /= self[key]

        if not template and not path.exists():
            Logger.error(
                f"The following path does not exist, please verify your installation environment: {path}"
            )

        return path

    def toTemplate(
        self,
        data: str | dict,
        replace="dirs",
        save: bool = True,
        report: bool = True,
        **kwargs,
    ) -> dict:
        """
        Recursively converts string values in a dict to be template values which can be
        converted back using Ini.fromTemplate(). Template values are in the form of
        "{env.[value]}".

        \b
        Parameters
        ----------
        data : str | dict
            The dictionary to walk over and update values. If string, checks if this
            exists as a file and loads that in as the data dict
        replace : "dirs" | "keys" | None, default="dirs"
            Defines what kind of values from the ini to replace in strings:
            - "dirs" only replace directory paths
            - "keys" only replace key strings
            - None replaces both
            Recommended to only use "dirs" to remain consistent. "keys" can have
            unintended consequences and may replace more than it should
        save : bool, default=True
            If the data was a file and this is enabled, saves the converted data dict
            to another file. The new file will simply append ".tmpl" to its name
        report : bool, default=True
            Reports if no value in the input data was changed
        **kwargs : dict
            Additional strings to replace. The values are replaced in a string with the
            key of the kwarg. For example:
            >>> kwargs = {"xyz": "abc"}
            >>> data["some_key"] = "replace abc here"
            will be replaced as:
            >>> data["some_key"] = "replace {xyz} here"
            This is to be used with Ini.fromTemplate to replace values that are not
            found in the ini object

        \b
        Returns
        -------
        data : dict | pathlib.Path
            In-place replaced string values with template values
            If saved as a new file, returns the path instead
        """
        file = None
        if isinstance(data, str):
            if (file := Path(data)).exists():
                with open(data, "rb") as f:
                    data = json.load(f)

                # Attempt to discover the working directory if it's in an ISOFIT output
                if "working_directory" not in kwargs:
                    if wd := getWorkingDir(file):
                        kwargs["working_directory"] = wd
            else:
                raise FileNotFoundError("If `data` is not a dict, it must be a file")

        orig = None
        if report:
            orig = deepcopy(data)

        for key, value in data.items():
            if isinstance(value, dict):
                self.toTemplate(value, report=False, **kwargs)
            elif isinstance(value, str):
                for k, v in self.items(replace):
                    if v in value:
                        Logger.debug(f"{key}: {v} in {value} => env.{k}")
                        value = value.replace(v, "{env." + k + "}")

                for k, v in kwargs.items():
                    if v in value:
                        Logger.debug(f"{key}: {v} in {value} => {k}")
                        value = value.replace(v, "{" + k + "}")

                data[key] = value

        if orig != data:
            if save and file:
                out = file.with_suffix(f"{file.suffix}.tmpl")
                with open(out, "w") as f:
                    f.write(json.dumps(data, indent=4))

                Logger.info(f"Saved converted json to: {out}")
                return out
        elif report:
            Logger.warning(
                "No value in the config was replaced. Are the paths in the ini in the config?"
            )

        return data

    def fromTemplate(
        self, data: str | dict, save: bool = True, prepend: str = None, **kwargs
    ) -> dict:
        """
        Recursively replaces the template values in found in string values with the
        real value from the ini. Template values are in the form of "{env.[value]}".
        This is an in-place operation.

        \b
        Parameters
        ----------
        data : str | dict
            The dictionary to walk over and update values. If string, checks if this
            exists as a file and loads that in as the data dict
        save : bool, default=True
            If the data was a file and this is enabled, saves the converted data dict
            to another file. If the input file ends with ".tmpl" then it will simply be
            cut. If it doesn't or already exists, then the output filename will be the
            input filename prepended with `prepend` value.
        prepend : str, default=None
            Prepend a string to the output filename. If not set and the input filename
            doesn't end with ".tmpl", then this is auto-set to "replaced"
        **kwargs : dict
            Additional strings to replace. The values are replaced in a string with the
            key of the kwarg. For example:
            >>> kwargs = {"xyz": "abc"}
            >>> data["some_key"] = "replace {xyz} here"
            will be replaced as:
            >>> data["some_key"] = "replace abc here"
            This is to be used with Ini.toTemplate to replace values that are not found
            in the ini object

        \b
        Returns
        -------
        data : dict | pathlib.Path
            In-place replaced template values with actual from a loaded ini
            If saved as a new file, returns the path instead
        """
        # On the first call, this may be a file so load it
        file = None
        if isinstance(data, str):
            if (file := Path(data)).exists():
                with open(data, "rb") as f:
                    data = json.load(f)

                # Attempt to discover the working directory if it's in an ISOFIT output
                if "working_directory" not in kwargs:
                    if wd := getWorkingDir(file):
                        kwargs["working_directory"] = wd
            else:
                raise FileNotFoundError("If `data` is not a dict, it must be a file")

        for key, value in data.items():
            if isinstance(value, dict):
                self.fromTemplate(value, **kwargs)
            elif isinstance(value, str):
                # Find all "{env.[value]}"
                for dir in re.findall(r"{env\.(\w+)}", value):
                    # Replace in-place
                    value = value.replace("{env." + dir + "}", self[dir])

                for extra in re.findall(r"{(\w+)}", value):
                    if extra in kwargs:
                        value = value.replace("{" + extra + "}", str(kwargs[extra]))

                data[key] = value

        if save and file:
            if file.suffix == ".tmpl":
                out = file.with_suffix("")
                if out.exists() and not prepend:
                    prepend = "replaced"
            elif not prepend:
                out = file
                prepend = "replaced"

            if prepend:
                out = out.with_name(f"{prepend}.{out.name}")

            with open(out, "w") as f:
                f.write(json.dumps(data, indent=4))

            Logger.info(f"Saved converted json to: {out}")
            return out

        return data

    def reset(self, save: bool = False) -> None:
        """
        Resets the object to the defaults defined by ISOFIT

        Parameters
        ----------
        save : bool, default=False
            Saves the reset to the default ini file: ~/.isofit/isofit.ini
        """
        self.config = ConfigParser()
        self.section = "DEFAULT"

        self.changeBase(Path.home() / ".isofit/")
        self.changeKey(self._keys)

        self.ini = self.base / "isofit.ini"

        if save:
            self.save()

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
