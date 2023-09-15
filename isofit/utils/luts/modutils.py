"""
General utilities for MODTRAN files. Much of this code is inspired by Nimrod
Carmon's research.
"""
import os
import re

import numpy as np
import pandas as pd

from isofit import ray
from isofit.utils.luts.channel_read import load_chn_single


@ray.remote
def load_chn(file: str, multipart: bool = False) -> (dict, dict):
    """
    Loads a single channel file

    Parameters
    ----------
    file: str
        A MODTRAN file path
    multipart: bool, default=False
        Enables splitting transmittance

    Returns
    -------
    tuple
        (Points: dict, Channel Data: dict)
    """
    return parsePoint(file), load_chn_single(file, multipart)


def load_chns(files: list, multipart: bool = False) -> list:
    """
    Distributed loading channel files

    Parameters
    ----------
    files: list
        List of MODTRAN file paths
    multipart: bool, default=False
        Enables splitting transmittance

    Returns
    -------
    list
        [(Points, Channel Data) for each file]
    """
    jobs = [load_chn.remote(file, multipart) for file in files]
    return ray.get(jobs)


def prepareData(moddata: list) -> dict:
    """
    Prepares MODTRAN data from load_chns for writeHDF5

    Parameters
    ----------
    moddata: list
        Output from load_chns, form: [(points, data) for each channel]

    Returns
    -------
    dict
        Items to be saved into an HDF5 as-is
    """
    # Some variables only use info from one chn, makes code cleaner
    points, channel = moddata[0]

    # Sample space
    sampNames = list(points.keys())
    sampSpace = np.array([[*pnts.values()] for (pnts, data) in moddata])

    # Extract before creating the modData
    wls = np.round(channel["wls"])
    sols = np.squeeze(channel["sols"])

    # deleting makes stacking cleaner
    for (pnts, data) in moddata:
        del data["wls"]
        del data["sols"]

    # MODTRAN output
    modNames = list(channel.keys())
    modData = np.array([[*data.values()] for (pnts, data) in moddata])

    # Data to be passed to writeHDF5 via **
    return dict(
        Sols=sols,
        Wavelengths=wls,
        Dimensions=sampNames,
        SampleSpace=sampSpace,
        Products=modNames,
        Output=modData,
    )


def parsePoint(file: str) -> dict:
    """
    Parses the point values from a MODTRAN filename

    Parameters
    ----------
    file: str
        A MODTRAN file path

    Returns
    -------
    pnts: dict
        {Name: Value} for each point parsed from the filename
    """
    pnts = {}
    grid = os.path.splitext(os.path.basename(file))[0]

    pattern = r"([^_]\w+)-(\d+\.?\d*)+"
    matches = re.findall(pattern, grid)
    for key, value in matches:
        pnts[key] = float(value)

    return pnts


def parseLine(line: str) -> list:
    """
    Parses a single line of a MODTRAN channel file into a list of token values

    Parameters
    ----------
    line: str
        Singular data line of a MODTRAN .chn file

    Returns
    -------
    list
        List of floats parsed from the line
    """
    # Fixes issues in large datasets where irrelevant columns touch which breaks the parsing
    line = line[:17] + " " + line[18:]

    return [float(match) for match in re.findall(r"(\d\S*)", line)]


def process(tokens: list, coszen: float) -> dict:
    """
    Processes tokens returned by parseLine()

    Parameters
    ----------
    tokens: list
        List of floats returned by parseLine()
    coszen: float
        cos(zenith(filename))

    Returns
    -------
    dict
        Dictionary of calculated values using the tokens list
    """
    # Process the tokens
    irr = tokens[18] * 1e6 * np.pi / tokens[8] / coszen  # uW/nm/sr/cm2
    # fmt: off
    return {
        'solar_irr'          : irr,       # Solar irradiance
        'wl'                 : tokens[0], # Wavelength
        'rhoatm'             : tokens[4] * 1e6 * np.pi / (irr * coszen), # uW/nm/sr/cm2
        'width'              : tokens[8],
        'thermal_upwelling'  : (tokens[11] + tokens[12]) / tokens[8] * 1e6, # uW/nm/sr/cm2
        'thermal_downwelling': tokens[16] * 1e6 / tokens[8],
        'path_rdn'           : tokens[14] * 1e6 + tokens[15] * 1e6, # The sum of the (1) single scattering and (2) multiple scattering
        'grnd_rflt'          : tokens[16] * 1e6,        # ground reflected radiance (direct+diffuse+multiple scattering)
        'drct_rflt'          : tokens[17] * 1e6,        # same as 16 but only on the sun->surface->sensor path (only direct)
        'transm'             : tokens[21] + tokens[22], # Total (direct+diffuse) transmittance
        'sphalb'             : tokens[23], #
        'transup'            : tokens[24], #
    }
    # fmt: on


def calcMultipart(
    p0: pd.DataFrame,
    p1: pd.DataFrame,
    p2: pd.DataFrame,
    coszen: float,
    rfl1: float = 0.1,
    rfl2: float = 0.5,
) -> pd.DataFrame:
    """
    Calculates split transmittance values from a multipart file

    Parameters
    ----------
    p0: pandas.DataFrame
        DataFrame of part 0 of the channel file
    p1: pandas.DataFrame
        DataFrame of part 1 of the channel file
    p2: pandas.DataFrame
        DataFrame of part 2 of the channel file
    coszen: float
        cos(zenith(filename))
    rfl1: float, defaults=0.1
        Reflectance scaler 1
    rfl2: float, defaults=0.5
        Reflectance scaler 2

    Returns
    -------
    df: pandas.DataFrame
        DataFrame of relevant information
    """
    # Extract relevant columns
    widths = p0.width.values
    t_up_dirs = p0.transup.values
    toa_irad = p0.solar_irr.values * coszen / np.pi

    # Calculate some fluxes
    directRflt1 = p1.drct_rflt.values
    groundRflt1 = p1.grnd_rflt.values

    directFlux1 = directRflt1 * np.pi / rfl1 / t_up_dirs
    globalFlux1 = groundRflt1 * np.pi / rfl1 / t_up_dirs

    diffuseFlux = globalFlux1 - directFlux1

    globalFlux2 = p2.grnd_rflt.values * np.pi / rfl2 / t_up_dirs

    # Path radiances
    rdn1 = p1.path_rdn.values
    rdn2 = p2.path_rdn.values

    # Path Radiance No Surface
    val1 = rfl1 * globalFlux1  # TODO: Needs a better name
    val2 = rfl2 * globalFlux2  # TODO: Needs a better name
    prns = ((val2 * rdn1) - (val1 * rdn2)) / (val2 - val1)

    # Diffuse upwelling transmittance
    t_up_difs = np.pi * (rdn1 - prns) / (rfl1 * globalFlux1)

    # Spherical Albedo
    sphalbs = (globalFlux1 - globalFlux2) / (val1 - val2)
    dFluxRN = directFlux1 / coszen  # Direct Flux Radiance

    globalFluxNS = globalFlux1 * (1 - rfl1 * sphalbs)  # Global Flux No Surface
    diffusFluxNS = globalFluxNS - dFluxRN * coszen  # Diffused Flux No Surface

    t_down_dirs = dFluxRN * coszen / widths / np.pi / toa_irad
    t_down_difs = diffusFluxNS / widths / np.pi / toa_irad

    transms = (t_down_dirs + t_down_difs) * (t_up_dirs + t_up_difs)

    # Insert the calculated information and return only the first frame
    data = {
        "t_up_dirs": t_up_dirs,
        "t_up_difs": t_up_difs,
        "t_down_dirs": t_down_dirs,
        "t_down_difs": t_down_difs,
        "transm": transms,
        "sphalb": sphalbs,
    }
    df = p0[
        [
            "wl",
            "solar_irr",
            "rhoatm",
            "transm",
            "sphalb",
            "thermal_upwelling",
            "thermal_downwelling",
        ]
    ].copy()
    for key, value in data.items():
        df[key] = value

    return df


def parseChannel(file: str, header: int = 5) -> pd.DataFrame:
    """
    Parses a MODTRAN channel file and extracts relevant data

    Parameters
    ----------
    file: str
        Path to a .chn file
    header: int, defaults=5
        Number of lines to skip for the header

    Returns
    -------
    df: pandas.DataFrame
        DataFrame of relevant information

    Notes
    -----
    file = '20171108_Pasadena/lut_multi/AOT550-0.1000_H2OSTR-1.5000.chn'

    %timeit load_chn_single(file, multipart=True) # Nimrod's
    10.5 ms ± 119 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    %timeit parseChannel(file)
    12.8 ms ± 62 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    with open(file, "r") as f:
        lines = f.readlines()
    data = [lines[header:]]

    # Checks if this is a multipart file, separate if so
    n = int(len(lines) / 3)
    if lines[1] == lines[n + 1]:
        # fmt: off
        data = [
            lines[   :n  ][header:],
            lines[  n:n*2][header:],
            lines[n*2:   ][header:]
        ]
        # fmt: on

    points = parsePoint(file)

    dfs = []
    for part, lines in enumerate(data):
        # Takes the first value in the filename for coszen
        coszen = np.cos(np.deg2rad(points[list(points)[0]]))
        parsed = [process(parseLine(line), coszen) for line in lines]

        # Convert into a DataFrame for easier handling
        df = pd.DataFrame(parsed)
        df["Multipart"] = part
        dfs.append(df)

    if len(dfs) > 1:
        df = calcMultipart(*dfs, coszen)
    else:
        (df,) = dfs

    return points, df


def parseChannelXarray(*args, **kwargs):
    """
    Converts outputs of parseChannel to an Xarray Dataset object
    """
    ps, df = parseChannel(*args, **kwargs)
    ds = df.to_xarray()
    ds = ds.set_index(index="wl")
    ds = ds.rename(index="wl")

    ds.attrs = ps

    return ds


ds = parseChannelXarray(
    "/Users/jamesmo/projects/isofit/examples/20171108_Pasadena/lut_multi/AOT550-0.1000_H2OSTR-1.5000.chn"
)
ds
