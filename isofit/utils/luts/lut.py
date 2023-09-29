"""
HDF5 file-handling utilities
"""
import logging
import os

import h5py
import numpy as np

Logger = logging.getLogger(__file__)


def writeHDF5(file: str, group: str = "data", **data) -> str:
    """
    Simply writes data from a dictionary into an HDF5 file.

    Parameters
    ----------
    file: str
        Path to write out HDF5 to
    group: str, defaults='data'
        The subgroup of the HDF5 to write to. This allows writing different
        outputs to the same file
    **data: dict
        The data to be saved into the HDF5

    Returns
    -------
    file: str
        Returns the file string path if successful


    TODO: Update to write a chunk of data (1 sim) at a time
    """
    with h5py.File(file, "a") as h5:
        if group in h5:
            Logger.warning(
                f"Group already exists in file and will be overwritten: {group!r}"
            )
            del h5[group]

        group = h5.create_group(group)
        for key, value in data.items():
            group[key] = value

    return file


def readHDF5(file: str, group: str = "data", subset: list = None) -> dict:
    """
    Simply reads data from an HDF5

    Parameters
    ----------
    file: str
        Path to read HDF5 from
    group: str, defaults='data'
        The subgroup of the HDF5 to read from. This allows writing different
        outputs to the same file
    subset: list, defaults=None
        The subset of the HDF5 to retrieve. If None equivalent then defaults to
        all keys available under the group

    Returns
    -------
    data: dict
        [{Key: Value} for each key in subset], all keys available if subset is
        None equivalent
    """
    if not os.path.exists(file):
        raise FileNotFoundError(file)

    with h5py.File(file, "r") as h5:
        # Retrieve the data for this group
        group = h5[group]
        keys = list(group)

        # Use everything if not set
        if not subset:
            subset = keys

        # Only load requested data
        data = {}
        for key in subset:
            value = group[key][:]

            if np.issubdtype(value.dtype, "O"):
                value = value.astype(str)

            data[key] = value

    return data


def verify(file):
    """
    Verifies a LUT NetCDF file is valid as per ISOFIT specifications

    Parameters
    ----------
    file: str
        Path to a luts.nc file

    Returns
    bool
        True is the file is valid, False if not. Logger.error messages will be
        provided
    """
    ...


def runSims(*strings):
    """
    Executes a list of simulation calls in parallel.

    Parameters
    ----------
    *strings
        List of sim strings to execute

    Returns
    -------
    list
        A list of the returns from the simulations

    Notes
    -----
    Phil to implement this function further
    """

    def process(command):
        """ """
        subprocess.run(command, shell=True, check=True)

    func = ray.put(process)
    jobs = [func.remote(cmd) for cmd in strings]

    return ray.get(jobs)
