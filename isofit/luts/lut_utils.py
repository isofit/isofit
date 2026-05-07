import logging

import numpy as np
import xarray as xr

Logger = logging.getLogger(__name__)


def findSlice(dim, val):
    """
    Creates a slice for selecting along a dimension such that a value is encompassed by
    the slice.

    Parameters
    ----------
    dim: array
        Dimension array
    val: float, int
        Value to be encompassed

    Returns
    -------
    slice
        Index slice to encompass the value
    """
    # Increasing is 1, decreasing is -1 for the searchsorted
    orientation = 1
    if dim[0] > dim[-1]:
        orientation = -1

    # Subselect the two points encompassing this interp point
    b = np.searchsorted(dim * orientation, val * orientation)

    # Handle edge cases when val equals first or last lut dim value
    if val <= dim[0]:
        return slice(b, b + 2)

    else:
        return slice(b - 1, b + 1)


def optimizedInterp(ds, strat):
    """
    Optimizes the interpolation step by subselecting along dimensions first then
    interpolating

    Parameters
    ----------
    strat: dict
        Interpolation stategies to perform in the form of {dim: interpolate_values}

    Returns
    -------
    xr.Dataset
        Interpolated dataset
    """
    for key, val in strat.items():
        dim = ds[key]

        if val <= dim[0]:
            Logger.warning(
                f"Scene value for key: {key} of {round(val, 2)} "
                f"is less or equal to minimum LUT value {np.round(dim[0].data, 2)}. "
                "Solutions will use value interpolated to minimum LUT value"
            )

        elif val >= dim[-1]:
            Logger.warning(
                f"Scene value for key: {key} of {round(val, 2)} "
                f"is greater or equal to maximum LUT value {np.round(dim[-1].data, 2)}. "
                "Solutions will use value interpolated to maximum LUT value"
            )

        if isinstance(val, list):
            a = findSlice(dim, val[0])
            b = findSlice(dim, val[-1])

            # Find the correct order
            if a.start < b.start:
                sel = slice(a.start, b.stop)
            else:
                sel = slice(b.start, a.stop)
        else:
            sel = findSlice(dim, val)

        Logger.debug(f"- Subselecting {key}[{sel.start}:{sel.stop}]")
        ds = ds.isel({key: sel})

    Logger.debug("Calling .interp")
    return ds.interp(**strat)


def sel(ds, dim, lt=None, lte=None, gt=None, gte=None, encompass=True):
    """
    Subselects an xarray Dataset object using .sel

    Parameters
    ----------
    ds: xarray.Dataset
        LUT dataset
    dim: str
        Dimension to work on
    lt: float, default=None
        Select along this dim coordinates that are valued less than this
    lte: float, default=None
        Select along this dim coordinates that are valued less than or equal to this
    gt: float, default=None
        Select along this dim coordinates that are valued greater than this
    gte: float, default=None
        Select along this dim coordinates that are valued greater than or equal to this
    encompass: bool, default=True
        Change the values of gte/lte such that these values are encompassed using the
        previous/next valid grid point

    Returns
    -------
    ds: xarray.Dataset
        Subsetted dataset
    """
    assert None in (
        lt,
        lte,
    ), f"Subsetting `lt` and `lte` are mutually exclusive, please only set one for dim {dim}"
    assert None in (
        gt,
        gte,
    ), f"Subsetting `gt` and `gte` are mutually exclusive, please only set one for dim {dim}"

    # Which index in a where to select -- if the dim is in reverse order, this needs to swap
    g, l = -1, 0
    if ds[dim][0] > ds[dim][-1]:
        g, l = 0, -1

    if lt is not None:
        ds = ds.sel({dim: ds[dim] < lt})

    elif lte is not None:
        if encompass:
            where = ds[dim].where(ds[dim] > lte).dropna(dim)
            lte = where[l] if where.size else ds[dim].max()
            Logger.debug(f"Encompass changed lte value to {lte}")

        ds = ds.sel({dim: ds[dim] <= lte})

    if gt is not None:
        ds = ds.sel({dim: gt < ds[dim]})

    elif gte is not None:
        if encompass:
            where = ds[dim].where(ds[dim] < gte).dropna(dim)
            gte = where[g] if where.size else ds[dim].min()
            Logger.debug(f"Encompass changed gte value to {gte}")

        ds = ds.sel({dim: gte <= ds[dim]})

    return ds


def sub(ds: xr.Dataset, dim: str, strat) -> xr.Dataset:
    """
    Subsets a dataset object along a specific dimension in a few supported ways.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to operate on
    dim: str
        Name of the dimension to subset
    strat: float, int, list, dict, str, None
        Strategy to subset the given dimension with

    Returns
    -------
    xr.Dataset
        New subset of the input dataset
    """
    if isinstance(strat, (float, int, list)):
        return ds.sel({dim: strat})

    elif isinstance(strat, str):
        return getattr(ds, strat)(dim)

    elif strat is None:
        return ds  # Take dimension as-is

    elif isinstance(strat, dict):
        if "interp" in strat:
            return ds.interp({dim: strat["interp"]})

        return sel(ds, dim, **strat)

    else:
        Logger.error(f"Unknown subsetting strategy for type: {type(strat)}")
        return ds
