"""
ISOFIT Output Parser
"""

import json
import logging
import re
from functools import cached_property
from pathlib import Path

import xarray as xr

from isofit.radiative_transfer import luts


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

    def getTree(self, *, path=None, tree=None):
        """
        Recursively finds the files under a directory as a dict tree

        Parameters
        ----------
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
            if item.is_dir():
                self.getTree(path=item, tree=tree.setdefault(item.name, {"": []}))
            elif self.extMatches(item):
                tree[""].append(item.name)

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

    def ifin(self, name, all=False):
        """
        Simple if name in filename match

        Parameters
        ----------
        name : str
            String to check in the filename
        all : bool, default=False
            Return all files matched instead of the first instance

        Returns
        -------
        str | list | None
            First matched file if all is False, otherwise the full list
        """
        found = []
        for file in self.getFlat():
            if name in file:
                found.append(file)

        if not all:
            if len(found) > 1:
                self.log.warning(
                    f"{len(found)} files were found containing the provided name {name!r}, try being more specific. Returning just the first instance"
                )

            if found:
                return found[0]
        return found

    def match(self, regex, all=False):
        """
        Find files using a regex match

        Parameters
        ----------
        regex : str
            Regex pattern to match with
        all : bool, default=False
            Return all files matched instead of the first instance

        Returns
        -------
        str | list | None
            First matched file if all is False, otherwise the full list
        """
        found = []
        for file in self.getFlat():
            try:
                if re.match(regex, file):
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

    def find(self, name, all=False):
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
        all : bool, default=False
            Return all files matched instead of the first instance

        Returns
        -------
        str | list | None
            First matched file if all is False, otherwise the full list
        """
        regex = "/".join([f"\S*{part}\S*" for part in name.split("/")])
        regex = rf"({regex})"

        return self.match(regex, all)

    def load(self, *, path=None, ifin=None, find=None, match=None):
        """
        Loads a file

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
        if {path, ifin, find, match} == {
            None,
        }:
            raise AttributeError("At least one of the key-word arguments must be set")

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
        Parses LUT_*.{ext} files

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


class Unknown(FileFinder):
    extensions = ["*"]
    patterns = {r"(.*)": "Directory unknown, unable to determine this file"}

    def _load(self, file):
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
        TODO

        Parameters
        ----------
        unknown : bool, default=True
            If a sub directory type cannot be determined, use the Unknown class to
            retrieve the files. This may help ensure all sub directories behaviour is
            consistent, whereas when this is False, files in unknown subdirs will be
            treated as living on the root
        """
        kwargs["cache"] = False

        super().__init__(*args, **kwargs)

        self.dirs = {}

        unkn = []
        dirs = set([self.subpath(file).parent.name for file in self.files])
        for subdir in dirs:
            for name, cls in self.classes.items():
                if name in subdir:
                    self.log.debug(f"Initializing {subdir} with class {cls.__name__}")
                    self.dirs[subdir] = cls(self.path / subdir)
                    break
            else:
                # Prevent '' (aka root) from being added
                if subdir:
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
