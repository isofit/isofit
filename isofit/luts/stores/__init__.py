"""
Base class for creating ISOFIT LUTs. Must be subclassed to implement the store creation
logic.
"""

from __future__ import annotations

import gc
import importlib
import logging

import numpy as np
from packaging.version import Version

from isofit import __version__

Logger = logging.getLogger(__name__)


def create(path: str, *args, **kwargs):
    """
    Factory function to return the correct Create subclass

    Parameters
    ----------
    path : str
        Store path. Uses the extension to determine the correct subclass
    *args : list
        Arguments to pass to the subclass
    *args : dict
        Key-word arguments to pass to the subclass

    Returns
    -------
    obj
        Create subclass object
    """
    ext = Path(path).suffix

    if ext == ".nc":
        from .netcdf import CreateNetCDF as cls
    elif ext == ".zarr":
        from .zarr import CreateZarr as cls
    else:
        raise AttributeError(
            "The LUT path extension must be one of the supported stores"
        )

    return cls(*args, **kwargs)


class Create:
    def __init__(
        self,
        path: str,
        keys: object,
        wl: np.ndarray,
        grid: dict,
        mode: str = "w",
        attrs: dict = {},
        consts: dict = {},
        onedim: dict = {},
        alldim: dict = {},
        zeros: List[str] = [],
        chunks: List[int] | str = "auto",
        init: bool = True,
        **kwargs,
    ):
        """
        Prepare a LUT path

        Parameters
        ----------
        path : str
            Path for the LUT.
        keys : object
            Default keys to set. This is needed to pre-initilize the LUT store to
            optimize the flushing of data to disk.
        wl : np.ndarray
            The wavelength array.
        grid : dict
            The LUT grid, formatted as {str: Iterable}.
        mode : str, default="w"
            Mode to open with.
        attrs: dict, defaults={}
            Dict of dataset attributes, ie. {"RT_mode": "transm"}
        consts : dict, optional, default={}
            Dictionary of constant values. Appends/replaces current Create.consts list.
        onedim : dict, optional, default={}
            Dictionary of one-dimensional data. Appends/replaces to the current Create.onedim list.
        alldim : dict, optional, default={}
            Dictionary of multi-dimensional data. Appends/replaces to the current Create.alldim list.
        zeros : List[str], optional, default=[]
            List of zero values. Appends to the current Create.zeros list.
        chunks : list[int] | "auto", default="auto"
            Chunking strategy to use on alldim variables. "auto" will chunk along the point dimension
        init : bool, default=True
            Call the initialize function
        *kwargs : dict
            Captures any additional key-word arguments and ignores
        """
        # Track the ISOFIT version that created this LUT
        attrs["ISOFIT version"] = __version__
        attrs["ISOFIT status"] = "<incomplete>"

        self.path = path
        self.keys = keys
        self.wl = wl
        self.grid = {key: np.array(vals) for key, vals in grid.items()}
        self.hold = []

        self.sizes = {"wl": len(self.wl)} | {key: len(val) for key, val in grid.items()}
        self.dims = list(self.sizes)
        self.point_dims = list(grid)

        self.attrs = attrs

        self.consts = {**keys.consts, **consts}
        self.onedim = {**keys.onedim, **onedim}
        self.alldim = {**keys.alldim, **alldim}

        self.chunks = chunks
        if chunks == "auto":
            self.chunks = [len(self.wl)] + [1] * len(grid)

        if init:
            self.initialize()

    def pointIndices(self, point: np.ndarray) -> List[int]:
        """
        Get the indices of the point in the grid.

        Parameters
        ----------
        point : np.ndarray
            The coordinates of the point in the grid.

        Returns
        -------
        List[int]
            Mapped point values to index positions.
        """
        return [
            np.where(self.grid[dim] == val)[0][0] for dim, val in zip(self.grid, point)
        ]

    def queuePoint(self, point: np.ndarray, data: dict) -> None:
        """
        Queues a point and its data to the internal hold list which is used by the
        flush function to write these points to disk.

        Parameters
        ----------
        point : np.ndarray
            The coordinates of the point in the grid.
        data : dict
            Data for this point to write.
        """
        self.hold.append((point, data))

    def flush(self, finalize: bool = False) -> None:
        """
        Flushes the (point, data) pairs held in the hold list to the LUT

        Parameters
        ----------
        finalize : bool, default=False
            Calls the `finalize` function
        """
        # Subclass flush
        unknowns = self._flush()

        self.hold = []
        gc.collect()

        # Reduce the number of warnings produced per flush
        for key in unknowns:
            Logger.warning(
                f"Attempted to assign a key that is not recognized, skipping: {key}"
            )

        if finalize:
            self.finalize()

    def writePoint(self, point: np.ndarray, data: dict) -> None:
        """
        Queues a point and immediately flushes to disk.

        Parameters
        ----------
        point : np.ndarray
            The coordinates of the point in the grid.
        data : dict
            Data for this point to write.
        """
        self.queuePoint(point, data)
        self.flush()

    def finalize(self):
        """
        Finalizes the store by writing any remaining attributes to disk
        """
        self.setAttr("ISOFIT status", "success")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.sizes})"
