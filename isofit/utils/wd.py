"""
ISOFIT Output Parser
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace

import xarray as xr

from isofit.radiative_transfer import luts


@dataclass(frozen=True)
class FileInfo:
    name: str
    info: Any


class FileFinder:
    """
    Utility class to find files under a directory using various matching strategies
    This must be subclassed and define the following:
        function _load       - The loader for a passed file
        attribute extensions - A list of extensions to retrieve
    """

    cache = None
    patterns = {}
    extensions = []

    def __init__(self, path, cache=True, extensions=[], patterns={}):
        """
        Parameters
        ----------
        path : str
            Path to directory to operate on
        cache : bool, default=True
            Enable caching objects in the .load function
        extensions : list, default=[]
            File extensions to retrieve when searching
        """
        self.path = Path(path)

        if extensions:
            self.extensions = extensions

        if not self.extensions:
            raise AttributeError("One or more extensions must be defined")

        if patterns:
            self.patterns = patterns

        if cache:
            self.cache = {}

        self.log = logging.getLogger(str(self))

    def __repr__(self):
        return f"<{self.__class__.__name__} [{self.path}]>"

    def info(self, file):
        """
        Retrieves known information for a file if its name matches one of the set
        patterns

        Parameters
        ----------
        file : str
            File name to compare against the patterns dict keys

        Returns
        -------
        any
            Returns the value if a regex key in the patterns dict matches the file name
        """
        file = str(file)  # Not pathlib.Path compatible
        for pattern, desc in self.patterns.items():
            if re.match(pattern, file):
                return desc

    @property
    def files(self):
        """
        Passthrough attribute to calling getFlat()
        """
        return self.getFlat()

    def extMatches(self, file):
        """
        Checks if a given file's extension matches in the list of extensions
        Special extension cases include:
            "*" = Match any file
            ""  = Only match files with no extension

        Parameters
        ----------
        file : pathlib.Path
            File path to check

        Returns
        -------
        bool
            True if it matches one of the extensions, False otherwise
        """
        if file.is_dir():
            return False
        if "*" in self.extensions:
            return True
        return file.suffix in self.extensions

    def getTree(self, info=False, *, path=None, tree=None):
        """
        Recursively finds the files under a directory as a dict tree

        Parameters
        ----------
        info : bool, default=False
            Return the found files as objects with their respective info
        path : pathlib.Path, default=None
            Directory to search, defaults to self.path
        tree : dict, default=None
            Tree structure of discovered files

        Returns
        -------
        tree : dict
            Tree structure of discovered files. The keys are the directory names and
            the list values are the found files
        """
        if path is None:
            path = self.path

        if tree is None:
            tree = {"": []}

        for item in path.glob("*"):
            data = item.name
            if info:
                data = FileInfo(item.name, self.info(item.name))

            if item.is_dir():
                self.getTree(info=info, path=item, tree=tree.setdefault(data, {"": []}))
            elif self.extMatches(item):
                tree[""].append(data)

        return tree

    def getFlat(self, path=None):
        """
        Finds all the files under a directory as a flat list with the base path removed

        Parameters
        ----------
        path : pathlib.Path, default=None
            Directory to search, defaults to self.path

        Returns
        -------
        files : list[str]
            List structure of discovered files
        """
        if path is None:
            path = self.path

        files = []
        for file in self.path.rglob("*"):
            if self.extMatches(file):
                name = str(file).replace(f"{path}/", "")
                files.append(name)

        return files

    def ifin(self, name, all=False, exc=[]):
        """
        Simple if name in filename match

        Parameters
        ----------
        name : str
            String to check in the filename
        all : bool, default=False
            Return all files matched instead of the first instance
        exc : str | list[str], default=[]
            A string or list of strings to use to exclude files. If a file contains
            one of the strings in its name, it will not be selected

        Returns
        -------
        str | list | None
            First matched file if all is False, otherwise the full list
        """
        if isinstance(exc, str):
            exc = [exc]

        found = []
        for file in self.getFlat():
            if name in file:
                for string in exc:
                    if string in file:
                        continue
                found.append(file)

        if not all:
            if len(found) > 1:
                self.log.warning(
                    f"{len(found)} files were found containing the provided name {name!r}, try being more specific. Returning just the first instance"
                )

            if found:
                return found[0]
        return found

    def match(self, regex, all=False, exc=[]):
        """
        Find files using a regex match

        Parameters
        ----------
        regex : str
            Regex pattern to match with
        all : bool, default=False
            Return all files matched instead of the first instance
        exc : str | list[str], default=[]
            A string or list of strings to use to exclude files. If a file contains
            one of the strings in its name, it will not be selected

        Returns
        -------
        str | list | None
            First matched file if all is False, otherwise the full list
        """
        if isinstance(exc, str):
            exc = [exc]

        found = []
        for file in self.getFlat():
            try:
                if re.match(regex, file):
                    for string in exc:
                        if string in file:
                            continue
                    found.append(file)
            except Exception as e:
                self.log.exception(f"Is this a valid regex? {regex}")
                raise e

        if not all:
            if len(found) > 1:
                self.log.warning(
                    f"{len(found)} files were found containing the provided regex {regex!r}, try being more specific. Returning just the first instance"
                )

            if found:
                return found[0]
        return found

    def find(self, name, *args, **kwargs):
        """
        Find files using a pre-built regex match. The regex will be in the form of:
            (\S*{part}\S*) for each part in the name split by "/", delineated by "/"

        For example:
            abc/xyz = (\S*abc\S*/\S*xyz\S*)
            part1/part2/part3 = (\S*part1\S*/\S*part2\S*/\S*part3\S*)

        Use .match() for exact control of the regex string

        Parameters
        ----------
        name : str
            Name to parse into a regex string
        *args : list, default=[]
            Additional arguments to pass to match. Refer to that function for
            additional information
        *kwargs : list, default={}
            Additional key-word arguments to pass to match. Refer to that function for
            additional information

        Returns
        -------
        str | list | None
            First matched file if all is False, otherwise the full list
        """
        regex = "/".join([f"\S*{part}\S*" for part in name.split("/")])
        regex = rf"({regex})"

        return self.match(regex, *args, **kwargs)

    def load(self, *, path=None, ifin=None, find=None, match=None):
        """
        Loads a file. One of the key-word arguments must be set. If more than one is
        given, the only first will be used

        Parameters
        ----------
        path : str
            Either the path to an existing file or the name of a file under self.path
        ifin : str
            Use the ifin function to find the file to load
        find : str
            Use the find function to find the file to load
        match : str
            Use the match function to find the file to load

        Returns
        -------
        any
            Returns the subclass's ._load(file)
        """
        args = {path, ifin, find, match} - set([None])
        if not args:
            raise AttributeError("One of the key-word arguments must be set")
        elif len(args) > 1:
            self.log.warning("Only one key-word argument should be set")

        if path:
            file = path
        elif ifin:
            file = self.ifin(ifin)
        elif find:
            file = self.find(find)
        elif match:
            file = self.match(match)

        if file and not Path(file).exists():
            file = self.path / file

        if not file or not (file := Path(file)).exists():
            raise FileNotFoundError(f"Cannot find file to load, attempted: {file}")

        if self.cache is not None:
            if file not in self.cache:
                self.log.debug(f"Loading file: {file}")
                data = self._load(file)
                if data is not None:
                    self.cache[file] = data

            self.log.debug(f"Returning from cache: {file}")
            return self.cache.get(file)

        self.log.debug(f"Returning from load: {file}")
        return self._load(file)

    def _load(self, file):
        raise NotImplementedError("Subclass must define this function")


class Config(FileFinder):
    extensions = [".json"]
    patterns = {
        # Presolve
        r"(.*_h2o.json)": "Presolve configuration produced by apply_oe",
        r"(.*_h2o.json.tmpl)": "Presolve configuration template for developer purposes",
        r"(.*_h2o_tpl.json)": "MODTRAN template configuration for the presolve run",
        # Full
        r"(.*_isofit.json)": "ISOFIT main configuration",
        r"(.*_isofit.json.tmpl)": "ISOFIT main configuration template for developer purposes",
        r"(.*_modtran_tpl.json)": "MODTRAN template configuration for ISOFIT",
    }

    def _load(self, file):
        """
        Loads a JSON file

        Parameters
        ----------
        file : pathlib.Path
            Path to file to load

        Returns
        -------
        dict
            Loaded JSON dict
        """
        with open(file, "rb") as f:
            return json.load(f)


class Data(FileFinder):
    extensions = [".mat", ".txt"]
    patterns = {
        r"(channelized_uncertainty.txt)": None,
        r"(model_discrepancy.mat)": None,
        r"(surface.mat)": None,
        r"(wavelengths.txt)": None,
    }


class LUT(FileFinder):
    extensions = [".nc"]
    patterns = {
        r"(6S.lut.nc)": "LUT produced by the SixS radiative transfer model for sRTMnet",
        r"(lut.nc)": "Look-Up-Table for the radiative transfer model",
        r"(sRTMnet.predicts.nc)": "Output predicts of sRTMnet",
    }

    lut_regex = r"(\w+)-(\d*\.?\d+)_?"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load(self, file):
        return luts.load(file)

    def parseLutFiles(self, ext):
        """
        Parses LUT_*.{ext} file names for the LUT grid

        Parameters
        ----------
        ext : str
            File extension to retrieve:
                sixs    = inp
                modtran = json

        Returns
        -------
        data : dict
            Quantities to a set of their LUT values parsed from the file names
        """
        data = {}
        for file in self.path.glob(f"*.{ext}"):
            matches = re.findall(self.lut_regex, file.stem[4:])  # [4:] skips LUT_
            for name, value in matches:
                quant = data.setdefault(name, set())
                quant.add(float(value))

        return data

    @cached_property
    def sixs(self):
        return self.parseLutFiles("inp")

    @cached_property
    def modtran(self):
        return self.parseLutFiles("json")


class Input(FileFinder):
    extensions = [""]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        files = self.getFlat()

        for file in files:
            if not (self.path / file).with_suffix(".hdr").exists():
                raise FileNotFoundError(f"Missing .hdr file for {file}")

    def _load(self, file):
        """
        Loads an ENVI file

        Parameters
        ----------
        file : pathlib.Path
            Path to file to load

        Returns
        -------
        xr.Dataset | xr.DataArray
            Loaded xarray object from the ENVI. If the Dataset is only one variable,
            returns the DataArray of that variable instead
        """
        if file.suffix:
            file = file.with_suffix("")

        ds = xr.open_dataset(file, engine="rasterio", lock=False)

        if len(ds) == 1:
            return ds[list(ds)[0]]


class Output(FileFinder):
    extensions = [""]
    patterns = {
        # Presolve
        r"(.*_subs_atm)": None,
        r"(.*_subs_h2o)": None,
        r"(.*_subs_rfl)": None,
        r"(.*_subs_state)": None,
        r"(.*_subs_uncert)": None,
        # Full
        r"(.*_atm_interp)": None,
        r"(.*_rfl)": "Reflectance",
        r"(.*_lbl)": None,
        r"(.*_uncert)": None,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        files = self.getFlat()

        for file in files:
            if "subs" in file:
                self.h2o = True
            elif file.endswith("_rfl"):
                self.name = file[:-4]

        self.products = [file.replace(f"{self.name}_", "") for file in files]

        for file in files:
            if not (self.path / file).with_suffix(".hdr").exists():
                raise FileNotFoundError(f"Missing .hdr file for {file}")

    def _load(self, file):
        """
        Loads an ENVI file

        Parameters
        ----------
        file : pathlib.Path
            Path to file to load

        Returns
        -------
        xr.Dataset | xr.DataArray
            Loaded xarray object from the ENVI. If the Dataset is only one variable,
            returns the DataArray of that variable instead
        """
        if file.suffix:
            file = file.with_suffix("")

        ds = xr.open_dataset(file, engine="rasterio", lock=False)

        if len(ds) == 1:
            return ds[list(ds)[0]]

    def rgb(self, r=60, g=40, b=30):
        """
        Returns the RGB data of the RFL product

        Parameters
        ----------
        r : int, default=60
            Red band
        g : int, default=40
            Green band
        b : int, default=30
            Blue band

        Returns
        -------
        xr.DataArray
        """
        data = self.load(path=f"{self.name}_rfl")

        # Retrieve the RGB subset
        rgb = data.sel(band=[r, g, b]).transpose("y", "x", "band")
        rgb /= rgb.max(["x", "y"])  # Brightens image

        # Convert to pixel coords for easier plotting
        rgb["x"] = range(rgb.x.size)
        rgb["y"] = range(rgb.y.size)

        return rgb


class Logs(FileFinder):
    extensions = [".log"]

    file = None

    def __init__(self, *args, **kwargs):
        """
        Auto-loads the first discovered log file
        """
        super().__init__(*args, **kwargs)

        files = self.getFlat()
        if files:
            self.file = files[0]

        # fmt: off
        #                   # Source | Purpose
        self.lines    = []  # build  | The formatted lines (end result of parse->filter->build)
        self.split    = {}  # parse  | Each level containing only the parsed lines of that level
        self.levels   = []  # parse  | Logging levels found in the log
        self.format   = {}  # parse  | Additional formatting options used by build
        self.content  = []  # parse  | Each line parsed into a dict of info
        self.filtered = []  # filter | Lines passing the filter criteria of selected
        self.selected = {}  # parse  | Turn logging levels on/off for the build function
        # fmt: on

    def _load(self, file):
        """
        Loads the lines of a text file

        Parameters
        ----------
        file : str
            File to load

        Returns
        -------
        list[str]
            file.readlines()
        """
        with open(file) as f:
            return f.readlines()

    def load(self, *, path=None, ifin=None, find=None, match=None):
        """
        Loads a file. One of the key-word arguments must be set. If more than one is
        given, the only first will be used

        Parameters
        ----------
        path : str
            Either the path to an existing file or the name of a file under self.path
        ifin : str
            Use the ifin function to find the file to load
        find : str
            Use the find function to find the file to load
        match : str
            Use the match function to find the file to load

        Returns
        -------
        list[str]
            Parsed lines from the log file
        """
        args = {path, ifin, find, match} - set([None])
        if not args:
            path = self.file
        elif len(args) > 1:
            self.log.warning("Only one key-word argument should be set")

        if path:
            file = path
        elif ifin:
            file = self.ifin(ifin)
        elif find:
            file = self.find(find)
        elif match:
            file = self.match(match)

        path = file

        if file and not Path(file).exists():
            file = self.path / file

        if not file or not Path(file).exists():
            raise FileNotFoundError(f"Cannot find file to load, attempted: {file}")

        self.file = path

        return super().load(path=path)

    def parse(self):
        """
        Parses an ISOFIT log file into a dictionary of content that can be used to
        filter and reconstruct lines into different formats

        Returns
        -------
        content : list[dict]
            Parsed content from the log file in the form:
                {
                    "timestamp": str,
                    "level": str,
                    "message": str,
                    "source": {
                        "file": str,
                        "func": str
                    }
                }
        """
        lines = self.load()

        self.content = []
        for line in lines:
            line = line.strip()

            # "[level]:[timestamp] || [source] | [message]"
            if find := re.findall(r"(\w+):(\S+) \|\| (\S+) \| (.*)", line):
                [find] = find
                level = find[0]

                source = find[2].split(":")
                content.append(
                    {
                        "timestamp": find[1],
                        "level": level,
                        "message": find[3],
                        "raw": line,
                        "source": {
                            "file": source[0],
                            "func": source[1],
                        },
                    }
                )
            # "[level]:[timestamp] ||| [message]"
            elif find := re.findall(r"(\w+):(\S+) \|\|\|? (.*)", line):
                [find] = find
                level = find[0]

                self.content.append(
                    {
                        "timestamp": find[1],
                        "level": level,
                        "message": find[2],
                        "raw": line,
                    }
                )
            else:
                self.content[-1]["message"] += f"\n{line}"

        # Split the content dict into a dict of levels for quick reference
        # eg. self.split["INFO"]["message"] to get all the info messages
        self.split = {}
        for line in content:
            for key, value in line.items():
                if key == "level":
                    continue
                level = self.split.setdefault(line["level"], {})
                group = level.setdefault(key, [])
                group.append(value)

        # Extract the levels and sort them per the logging module
        self.levels = sorted(set(split), key=lambda lvl: getattr(logging, lvl))
        self.selected = {lvl: True for lvl in self.levels}
        self.format = {"timestamps": True}

        return self.content

    def extract(self):
        """
        Extracts useful information from the processed logs
        """
        self.stats = []
        stats = SimpleNamespace()

        for i, line in enumerate(self.content):
            message = line["message"]

            if message == "Run ISOFIT initial guess":
                stats.name = "Presolve"

            if message == "Running ISOFIT with full LUT":
                stats.name = "Full Solution"

            if message == "Analytical line inference":
                stats.name = "Analytical Line"

            if find := re.findall(r"Beginning (\d+) inversions", message):
                stats.total = find[0]

            if "inversions complete" in message.lower():
                find = re.findall(r"(\d+\.\d+s?) (\S+)", message.replace(",", ""))

                stats.data = {val: key for key, val in find}

                self.stats.append(stats)

                # Reset the stats object
                stats = SimpleNamespace()

    def filter(self, select=0):
        """
        Filters the content per the `selected` dict

        Parameters
        ----------
        select : str | list[str] | None, default=0
            Toggles selections in the `selected` attribute. Options:
            - "all" = Enable all options
            - None  = Disable all options
            - str   = Enable only this option
            - list  = Enable only these options
            - Anything else, such as the default 0, will do nothing and use the current
              selected dict
        """
        if not self.content:
            self.parse()

        if select == "all":
            for key in self.selected:
                self.selected[key] = True
        elif select is None:
            for key in self.selected:
                self.selected[key] = False
        elif isinstance(select, str):
            for key in self.selected:
                self.selected[key] = False
            if key in self.selected:
                self.selected[key] = True
        elif isinstance(select, list):
            for key in self.selected:
                self.selected[key] = False
            for key in select:
                if key in self.selected:
                    self.selected[key] = True

        self.filtered = []
        for line in self.content:
            if self.selected[line["level"]]:
                self.filtered.append(line)

        return self.filtered

    def toggle(self, key, value=None):
        """
        Sets a key's visibility in either the format dict or the selected dict

        Parameters
        ----------
        key : str
            Key of interest
        value : bool, default=None
            Value to set for the key
        """
        if key in self.format:
            data = self.format
        elif key in self.selected:
            data = self.selected
        else:
            raise AttributeError(
                f"Key not found in either the format dict {list(self.format)} or the level selection dict {list(self.selected)}"
            )

        if value is None:
            value = not data[key]

        data[key] = value

    def build(self):
        """
        Builds the filtered contents dict into a list of tuples to be used for writing.
        Timestamps can be disabled by one of:

            self.format["timestamps"] = False
            self.toggle("timestamps", False)

        Returns
        -------
        lines : list[tuple[str, str, str]]
            Returns a list of 3-pair tuples of strings in the form:
                (timestamp, padded level, log message)
            Timestamp will be an empty string if it is not enabled
            The level is right-padded with whitespace to the length of the longest log
            level (eg. "warning", "debug  ")
            This will also be saved in self.lines
        """
        # Always re-filter
        self.filter()

        padding = len(max(self.levels)) + 1

        lines = []
        for c in self.filtered:
            level = c["level"].ljust(padding)

            ts = ""
            if self.format["timestamps"]:
                ts = c["timestamp"] + " "

            lines.append([ts, level, c["message"]])

        self.lines = lines

        return self.lines


class Unknown(FileFinder):
    extensions = ["*"]
    patterns = {r"(.*)": "Directory unknown, unable to determine this file"}

    def _load(self, *args, **kwargs):
        """
        Files under this class are ignored
        """
        self.log.error(
            "Unable to load file as the parent directory was unable to be parsed"
        )


class IsofitWD(FileFinder):
    extensions = ["*"]
    patterns = {
        r"(config)": "ISOFIT configuration files",
        r"(data)": "Additional data files generated by ISOFIT",
        r"(input)": "Data files inputted to the ISOFIT system",
        r"(lut)": "Look-Up-Table outputs",
        r"(lut_full)": "Look-Up-Table outputs",
        r"(lut_h2o)": "Look-Up-Table outputs for a presolve run",
        r"(output)": "ISOFIT outputs such as reflectance",
    }
    classes = {
        "config": Config,
        "data": Data,
        "input": Input,
        "lut": LUT,
        "output": Output,
    }

    def __init__(self, *args, unknown=True, **kwargs):
        """
        Parameters
        ----------
        unknown : bool, default=True
            If a sub directory type cannot be determined, use the Unknown class to
            retrieve the files. This may help ensure all sub directories behaviour is
            consistent, whereas when this is False, files in unknown subdirs will be
            treated as living on the root
        """
        # This class should not be saving to cache because
        # it defers loading to child classes which have their own cache
        kwargs["cache"] = False

        super().__init__(*args, **kwargs)

        self.logs = Logs(self.path)

        self.dirs = {}

        unkn = []
        dirs = set([self.subpath(file, parent=True)[0] for file in self.files]) - set(
            ["."]
        )
        for subdir in dirs:
            for name, cls in self.classes.items():
                if name in subdir:
                    self.log.debug(f"Initializing {subdir} with class {cls.__name__}")
                    self.dirs[subdir] = cls(self.path / subdir)
                    break
            else:
                unkn.append(subdir)

        if unknown:
            for subdir in unkn:
                self.log.debug(f"Initializing {subdir} with class Unknown")
                self.dirs[subdir] = Unknown(self.path / subdir)

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __getitem__(self, key):
        if key in self.dirs:
            return self.dirs[key]

    def _load(self, file):
        parent, subpath = self.subpath(file, parent=True)

        if parent in self.dirs:
            return self.dirs[parent].load(path=subpath)
        else:
            print(f"Files on the root path are not supported at this time: {subpath}")

    def reset(self, *args, **kwargs):
        """
        Re-initializes the object

        Parameters
        ----------
        *args : list
            Arguments to pass directly to __init__
        **kwargs : dict
            Key-word arguments to pass directly to __init__

        Returns
        -------
        self : IsofitWD
            Re-initialized IsofitWD object
        """
        self.__init__(*args, **kwargs)
        return self

    def subpath(self, path, parent=False):
        """
        Converts an absolute path to a relative path under self.path

        Parameters
        ----------
        path : str
            Either absolute or relative path, will assert it exists under self.path
        parent : bool, default=False
            Split the top-level parent from the subpath

        Returns
        -------
        pathlib.Path | (str, pathlib.Path)
            Relative path
            If parent is enabled, returns the top-level parent separated from the path
        """
        path = Path(path)

        if path.is_absolute():
            path = path.relative_to(self.path)

        if not (file := self.path / path).exists():
            raise FileNotFoundError(file)

        if parent:
            parent = "."
            if len(path.parents) >= 2:
                # -1 = "./" == self.path
                # -2 = "./dir/"
                parent = path.parents[-2].name

            return parent, path.relative_to(parent)
        return path

    def info(self, file):
        """
        Overrides the inherited info function to pass the file to the correct child
        object's info function

        Parameters
        ----------
        file : str
            File name to compare against the patterns dict keys

        Returns
        -------
        any
            Returns the value if a regex key in the patterns dict matches the file name
        """
        parent, subpath = self.subpath(file, parent=True)

        if parent in self.dirs:
            return self.dirs[parent].info(subpath.name)
        return super().info(file)

    def getTree(self, info=False, **kwargs):
        """
        Recursively finds the files under a directory as a dict tree

        Overrides the inherited getTree function to call the getTree of every object
        in self.dirs and merge the returns together. This lets each child handle
        building its own tree

        Parameters
        ----------
        info : bool, default=False
            Return the found files as objects with their respective info
        path : pathlib.Path, default=None
            Directory to search, defaults to self.path
        tree : dict, default=None
            Tree structure of discovered files

        Returns
        -------
        tree : dict
            Tree structure of discovered files. The keys are the directory names and
            the list values are the found files
        """
        tree = {"": []}
        for name, obj in self.dirs.items():
            if info:
                name = FileInfo(name, self.info(name))
            tree[name] = obj.getTree(info, **kwargs)

        for path in self.path.glob("*"):
            if (name := path.name) not in self.dirs:
                if info:
                    name = FileInfo(name, self.info(name))
                tree[""].append(name)

        return tree
