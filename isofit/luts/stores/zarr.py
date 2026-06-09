"""
Zarr implementation for LUTs
"""

from __future__ import annotations

import atexit
import logging
import shutil
from collections.abc import Iterable

import numpy as np
import xarray as xr
import zarr

from isofit.core import common
from isofit.luts.stores import Create

Logger = logging.getLogger(__name__)


def cleanup(path):
    """
    Checks the ``ISOFIT status`` attribute on a LUT and removes the file if it is
    incomplete.

    Parameters
    ----------
    path : str
        Path to the store to check
    """
    state = None
    key = "ISOFIT status"
    bad = "<incomplete>"

    if (p := Path(path)).exists():
        store = zarr.open_group(p, mode="r", zarr_format=3)
        state = store.attrs.get(key)

    if state == bad:
        Logger.error(
            f"The LUT status was determined to be incomplete, auto-removing: {p}"
        )
        shutil.rmtree(p)


def shard_to_coord(group, shards, shape):
    """
    Converts a shard index to shard coordinates (the slice of the full store for this
    shard).
    """
    start = np.array(group) * shards
    end = np.minimum(start + shards, shape)
    return tuple(slice(s, e) for s, e in zip(start, end))


def calc_shards(grid, wl, chunk, storage="8gb", min_shards=None, scale=1):
    """
    Determine an optimal Zarr sharding strategy for a LUT and group points by shard.

    This function computes candidate shard shapes that:
    - evenly divide the LUT dimensions
    - do not split chunks
    - approximately match a target storage size per shard

    It then selects the shard shape whose number of chunks per shard is closest
    to the target, and groups LUT points into shard-aligned partitions.

    Parameters
    ----------
    grid : dict[str, Iterable]
        LUT grid definition as {dimension_name: values}. These define the
        non-wavelength dimensions of the LUT.
    wl : iterable
        Wavelength dimension values.
    chunk : iterable of int
        Chunk shape used for Zarr storage. Must align with the full LUT shape
        (wl + grid dimensions).
    storage : str or float, default="8gb"
        Target storage size per shard. Determines how many chunks should be grouped
        into a shard.
    min_shards : int, optional
        Minimum number of shards to produce. Candidate shardings that
        result in fewer shards are discarded.
    scale : int, default=1
        Multiplier applied to the estimated chunk size. Useful when multiple arrays
        are written per chunk, to better approximate the actual memory when in buffered
        mode.

    Returns
    -------
    best : np.array[int]
        Selected shard shape, including the wavelength dimension as the first axis.
    groups : dict[tuple[int, ...], list[tuple]]
        Mapping from shard index (multi-dimensional shard ID) to the list of
        grid points (coordinate tuples) that fall within that shard.
    coords : dict[tuple[int, ...], tuple[slice, ...]]
        Mapping from shard index to the corresponding global array slice
        (including wavelength as the leading dimension). These slices can be
        used directly to write shard-aligned data into a Zarr array.
    """
    dtype = np.dtype("float64").itemsize
    if isinstance(storage, str):
        from isofit.core import units

        storage = units.byte_string_to_float(storage)

    space = (
        np.prod(chunk) * dtype * int(scale)
    )  # Determine how much space each chunk takes
    target = int(storage / space)  # Target number of chunks per shard

    # Not technically the fastest but it's small
    divisors = lambda n: np.unique([n // i for i in range(1, n + 1) if not n % i])

    # Viable shard shapes
    lens = [len(v) for v in grid.values()]
    shape = np.array([len(wl)] + lens)
    candidates = [divisors(s) for s in shape[1:]]

    # Of these, how many files are created by each sharding strategy
    shards = common.combos(candidates)

    # Insert the wavelength dimension as a single shard
    shards = np.column_stack([np.full(shards.shape[0], len(wl)), shards])

    # Ensure shards don't split chunks
    shards = shards[np.sum(shards % chunk, axis=1) == 0]

    # 1 shard == 1 file, only consider shardings that reach the minimum number of files
    if min_shards:
        files = np.prod((shape / shards).astype(int), axis=1)
        shards = shards[files >= min_shards]

    # Chunks per shard
    cpf = np.prod(shards / chunk, axis=1).astype(int)

    # Return the strategy that is closest to the target number of chunks per shard
    idx = np.abs(cpf - target).argmin()
    best = shards[idx]

    # Split the points into shard groups
    points = common.combos(grid.values())
    pidxs = common.combos([np.arange(v) for v in lens])
    indices = pidxs // best[1:]
    groups = {}
    coords = {}
    for i, key in enumerate(indices):
        key = tuple(key)
        groups.setdefault(key, []).append(points[i])
        if key not in coords:
            coord = shard_to_coord(key, best[1:], shape[1:])
            coord = (slice(None),) + coord
            coords[key] = coord

    # Useful information
    Logger.info(f"Number of points: {np.prod(shape):,}")
    Logger.info(f"Target chunks per file: {target} ({target * space / 2**30:.2f}gb)")
    Logger.info(f"Shapes:")
    Logger.info(f"  dimensions: {shape}")
    Logger.info(f"    chunking: {np.array(chunk)}")
    Logger.info(f"Recommended")
    Logger.info(f"    sharding: {best}")
    Logger.info(f"Produces:")
    Logger.info(f"  Chunks per file: {cpf[idx]} ({cpf[idx] * space / 2**30:.2f}gb)")
    Logger.info(f"  Number of files: {np.prod((shape / best).astype(int))}")

    return best, groups, coords


class CreateZarr(Create):
    def __init__(
        self,
        path,
        *args,
        mode="w",
        buffered=False,
        shard_size=None,
        min_shards=1,
        **kwargs,
    ):
        """
        Prepare a Zarr v3 LUT store

        Parameters
        ----------
        path : str
            Path for the LUT.
        mode : str, default="w"
            File mode to open with
        wl : np.ndarray
            The wavelength array.
        grid : dict
            The LUT grid, formatted as {str: Iterable}.
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
        """
        self.store = zarr.storage.LocalStore(path)
        self.z = zarr.open_group(store=self.store, mode=mode, zarr_format=3)

        # TODO: Integrate into config
        self.shards = None
        self.shard_size = shard_size
        self.min_shards = min_shards

        super().__init__(path, *args, mode=mode, **kwargs)

        self.buffer = {}
        if buffered:
            self.reset_buffer(buffered)

        self.data = self.buffer or self.z

        atexit.register(cleanup, self.path)

    def __getitem__(self, key: str) -> Any:
        """
        Retrieves a value from the Zarr store

        Parameters
        ----------
        key : str
            The name of the item to retrieve

        Returns
        -------
        Any
            The value of the item retrieved from the Zarr store
        """
        return self.zarr[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Sets a variable in the Zarr store

        Parameters
        ----------
        key : str
            Key to set
        value : any
            Value to set
        """
        # Automatically insert data into the buffer if it's available instead of writing to disk
        self.data[key][...] = value

    def initialize(self, ret=False) -> None | xr.Dataset:
        """
        Initializes the LUT Zarr by prepopulating it with filler values.

        Parameters
        ----------
        ret : bool, default=False
            If True, returns the dataset instead of saving

        Returns
        -------
        None | xr.Dataset
            The initialized dataset
        """
        if self.shard_size:
            self.shards, self.groups, self.coords = calc_shards(
                self.grid,
                self.wl,
                self.chunks,
                storage=self.shard_size,
                min_shards=self.min_shards,
                scale=len(self.alldim),
            )
            self.setAttr("shards", self.shards.tolist())
            self.setAttr("shard order", list(self.sizes))

        self.z.attrs.update(self.attrs)

        # Coordinates
        array = self.z.create_array(
            name="wl", data=self.wl, fill_value=None, dimension_names=["wl"]
        )

        for key, vals in self.grid.items():
            array = self.z.create_array(
                name=key, data=vals, fill_value=None, dimension_names=[key]
            )

        # Constants
        dims = []
        for key, vals in self.consts.items():
            array = self.z.create_array(
                name=key, data=np.array(vals), fill_value=None, dimension_names=dims
            )

        # One dimensional arrays
        dims = ["wl"]
        shape = (len(self.wl),)
        for key, vals in self.onedim.items():
            if isinstance(vals, Iterable):
                array = self.z.create_array(
                    name=key, data=vals, fill_value=None, dimension_names=dims
                )
            else:
                array = self.z.create_array(
                    name=key,
                    shape=shape,
                    dtype="float64",
                    fill_value=None,
                    dimension_names=dims,
                )

        # Multi dimensional arrays
        dims += list(self.grid)
        shape = list(self.sizes.values())
        chunks = tuple(self.chunks)
        shards = None
        if self.shards is not None:
            shards = tuple(self.shards)

        for key, vals in self.alldim.items():
            if isinstance(vals, Iterable):
                array = self.z.create_array(
                    key=key,
                    data=vals,
                    fill_value=None,
                    chunks=chunks,
                    shards=shards,
                    dimension_names=dims,
                )
            else:
                array = self.z.create_array(
                    name=key,
                    shape=shape,
                    dtype="float64",
                    fill_value=vals,
                    chunks=chunks,
                    shards=shards,
                    dimension_names=dims,
                )

    def _flush(self) -> None:
        """
        Flushes the (point, data) pairs held in the hold list to the LUT Zarr

        Parameters
        ----------
        finalize : bool, default=False
            Calls the `finalize` function
        """
        unknowns = set()
        for point, data in self.hold:
            for key, vals in data.items():
                if key == "wl":
                    continue

                if key in self.consts:
                    self.z[key][...] = vals

                elif key in self.onedim:
                    self.z[key][:] = vals

                elif key in self.alldim:
                    index = self.pointIndices(point)
                    if self.buffer:
                        index = index % self.shards[1:]
                    index = (slice(None),) + tuple(index)
                    self.data[key][index] = vals

                else:
                    unknowns.update([key])

        return unknowns

    def flush_buffer(self, slices):
        """
        Needs to be manually called
        """
        # If the group key is given instead of the slices tuple, retrieve from coords
        if not isinstance(slices[0], slice):
            slices = self.coords[slices]

        for key, vals in self.buffer.items():
            self.z[key][slices] = vals

    def reset_buffer(self, buffered):
        """
        Resets the buffered data to ensure
        """
        if self.buffer:
            for key, vals in self.alldim.items():
                self.buffer[key][:] = vals
        else:
            self.buffer = {
                key: np.full(self.shards, vals, "float64")
                for key, vals in self.alldim.items()
            }

    def queuePoint(self, *args, **kwargs):
        """
        Overrides the inherited queuePoint to enable flushing immediately for this
        subclass only
        """
        super().queuePoint(*args, **kwargs)

        # Don't hold a queue when buffered
        if self.buffer:
            self.flush()

    def getAttr(self, key: str) -> Any:
        """
        Gets an attribute from the Zarr

        Parameters
        ----------
        key : str
            Key to get

        Returns
        -------
        any | None
            Retrieved attribute from Zarr, if it exists
        """
        return self.z.attrs.get(key)

    def setAttr(self, key: str, value: Any) -> None:
        """
        Sets an attribute in the Zarr

        Parameters
        ----------
        key : str
            Key to set
        value : any
            Value to set
        """
        self.attrs[key] = value
        self.z.attrs.update({key: value})

    def finalize(self, *args, **kwargs):
        """
        Finalizes the Zarr store by consolidating the metadata at the end to make
        reading it more efficient via xarray
        """
        super().finalize(*args, **kwargs)

        zarr.consolidate_metadata(self.store, zarr_format=3)
