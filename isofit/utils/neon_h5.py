"""
Convert NEON AOP HDF5 radiance files to ISOFIT-ready ENVI format.

NEON stores at-sensor radiance in a split-integer encoding to save disk space:

    radiance = RadianceIntegerPart + RadianceDecimalPart / Scale_Factor

This module reconstructs float32 radiance and writes three ENVI binary files
that ISOFIT's apply_oe expects:

- ``.rad``  — at-sensor radiance, Band-Interleaved-by-Line (BIL)
- ``.loc``  — per-pixel location (easting, northing, elevation), BIL
- ``.obs``  — per-pixel observation geometry (angles, path length, time), BIP

Call graph
----------
::

    convert_neon_h5                         # main entry point
    ├── convert_rad                         # reconstruct & write radiance
    │       ├── _get_scale                  # find Scale_Factor in H5 attrs
    │       │       └── _to_float           # safe attr → float cast
    │       ├── _to_float                   # safe attr → float cast
    │       ├── _decode                     # bytes/numpy scalar → str
    │       └── _write_envi_header          # write .hdr text file
    ├── convert_loc                         # write location cube
    │       └── _write_envi_header
    └── convert_obs                         # write geometry cube
            ├── _to_decimal_hours           # normalize time band
            └── _write_envi_header
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np

# Band name constants exposed so callers can reference them without magic strings
LOC_BAND_NAMES = ["easting_utm", "northing_utm", "elevation_m"]

OBS_BAND_NAMES = [
    "path_length",
    "sensor_azimuth",
    "sensor_zenith",
    "solar_azimuth",
    "solar_zenith",
    "toa_azimuth",
    "toa_zenith",
    "view_azimuth",
    "cos_incidence",
    "time_decimal_hours",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


# --------------------------------
# Called by: convert_rad
# Calls:     _decode (recursive)
# --------------------------------
def _decode(x) -> str:
    """Return a plain string from an H5 scalar that may be bytes or a numpy void."""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if hasattr(x, "shape") and x.shape == () and x.dtype.kind in ("S", "O"):
        return _decode(x[()])
    return str(x)


# --------------------------------
# Called by: convert_rad, _get_scale
# Calls:     (none)
# --------------------------------
def _to_float(x, default=None) -> float | None:
    """Safely cast an H5 attribute value to float, returning *default* on failure."""
    try:
        if x is None:
            return default
        if np.isscalar(x):
            return float(x)
        if getattr(x, "shape", ()) == ():
            return float(x[()])
        if isinstance(x, (bytes, str)):
            return float(x)
    except Exception:
        pass
    return default


# --------------------------------
# Called by: convert_rad
# Calls:     _to_float
# --------------------------------
def _get_scale(dec_ds, group) -> Union[float, np.ndarray]:
    """
    Return the radiance scale factor from H5 attributes.

    Searches both the decimal-part dataset and the parent group because the
    attribute name is inconsistent across NEON file versions.  Falls back to
    1.0 (no scaling) if nothing is found.

    Parameters
    ----------
    dec_ds :
        HDF5 dataset for ``RadianceDecimalPart``.
    group :
        HDF5 group containing the radiance datasets.

    Returns
    -------
    float or np.ndarray
        Scalar scale factor, or a 1-D per-band array.
    """
    for obj in (dec_ds, group):
        if obj is None:
            continue
        for key in ("Scale_Factor", "Scale", "scale_factor"):
            if key not in getattr(obj, "attrs", {}):
                continue
            val = obj.attrs[key]
            if np.isscalar(val) or getattr(val, "shape", ()) == ():
                s = _to_float(val)
                if s:
                    return s
            else:
                arr = np.asarray(val)
                if arr.ndim == 1:
                    return arr.astype(np.float32)
    return 1.0


# --------------------------------
# Called by: convert_obs
# Calls:     (none)
# --------------------------------
def _to_decimal_hours(arr: np.ndarray) -> np.ndarray:
    """
    Normalise a time array to decimal hours.

    NEON OBS_Data band 10 (time) may arrive as decimal hours, seconds since
    midnight, or HHMMSS numeric format depending on the file version.

    Parameters
    ----------
    arr : np.ndarray
        Raw time values from OBS_Data.

    Returns
    -------
    np.ndarray
        Time in decimal hours, float32.
    """
    x = arr.astype(np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return arr.astype(np.float32)
    m = np.nanmax(x[finite])
    if m <= 24.5:
        return x.astype(np.float32)  # already decimal hours
    if m < 86_400:
        return (x / 3600.0).astype(np.float32)  # seconds → hours
    # HHMMSS numeric → hours
    h = np.floor(x / 10_000.0)
    mm = np.floor((x - h * 10_000.0) / 100.0)
    ss = x - h * 10_000.0 - mm * 100.0
    return (h + mm / 60.0 + ss / 3600.0).astype(np.float32)


# --------------------------------
# Called by: convert_rad, convert_loc, convert_obs
# Calls:     (none)
# --------------------------------
def _write_envi_header(
    path: str,
    *,
    samples: int,
    lines: int,
    bands: int,
    interleave: str,
    no_data: float = -9999.0,
    wavelengths: np.ndarray | None = None,
    fwhm: np.ndarray | None = None,
    map_info: str | None = None,
    coord_sys: str | None = None,
    band_names: list[str] | None = None,
    description: str = "NEON HDF5 → ENVI",
) -> None:
    """
    Write a minimal ENVI ``.hdr`` file.

    Parameters
    ----------
    path : str
        Output path for the header (typically ``<data_file>.hdr``).
    samples, lines, bands : int
        Spatial and spectral dimensions.
    interleave : str
        ``"bil"`` or ``"bip"``.
    no_data : float
        Fill value written to the ``data ignore value`` field.
    wavelengths : np.ndarray, optional
        Centre wavelengths in nm (written for ``.rad`` files).
    fwhm : np.ndarray, optional
        Full-width half-maximum values in nm.
    map_info : str, optional
        ENVI map info string.
    coord_sys : str, optional
        Coordinate system string.
    band_names : list[str], optional
        Per-band labels.
    description : str
        Short description written into the header.
    """
    rows = [
        "ENVI",
        f"description = {{{description}}}",
        f"samples = {samples}",
        f"lines   = {lines}",
        f"bands   = {bands}",
        "header offset = 0",
        "file type = ENVI Standard",
        "data type = 4",  # float32
        "byte order = 0",  # little-endian
        f"interleave = {interleave.lower()}",
        f"data ignore value = {no_data}",
    ]
    if wavelengths is not None:
        rows.append("wavelength = {" + ", ".join(f"{w:.6f}" for w in wavelengths) + "}")
    if fwhm is not None:
        rows.append("fwhm = {" + ", ".join(f"{v:.6f}" for v in fwhm) + "}")
    if map_info is not None:
        rows.append(f"map info = {{{map_info}}}")
    if coord_sys is not None:
        rows.append(f"coordinate system string = {{{coord_sys}}}")
    if band_names is not None:
        rows.append("band names = {" + ", ".join(band_names) + "}")

    Path(path).write_text("\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# --------------------------------
# Called by: convert_neon_h5
# Calls:     _get_scale, _to_float, _decode, _write_envi_header
# --------------------------------
def convert_rad(
    h5_path: str, out_basename: str, chunk_rows: int = 64
) -> tuple[str, str]:
    """
    Convert a NEON HDF5 radiance file to an ENVI BIL ``.rad`` file.

    NEON stores radiance as two integer datasets to reduce file size.  The
    physical radiance (W m⁻² sr⁻¹ nm⁻¹) is reconstructed as::

        R = RadianceIntegerPart + RadianceDecimalPart / Scale_Factor

    The output is written as float32, Band-Interleaved-by-Line (BIL), which
    is the format expected by ISOFIT's ``apply_oe``.

    Parameters
    ----------
    h5_path : str
        Path to the NEON L1 radiance HDF5 file.
    out_basename : str
        Output path without extension.  A ``.rad`` data file and a ``.hdr``
        header are written next to each other.
    chunk_rows : int, default 64
        Number of image rows processed at once.  Reduce if memory is limited.

    Returns
    -------
    rad_path : str
    hdr_path : str
    """
    rad_path = out_basename + ".rad"
    hdr_path = out_basename + ".rad.hdr"

    with h5py.File(h5_path, "r") as f:
        site = next(iter(f.keys()))
        rad_grp = f[site]["Radiance"]

        ds_int = rad_grp["RadianceIntegerPart"]  # (lines, samples, bands)
        ds_dec = rad_grp["RadianceDecimalPart"]
        n_lines, n_samples, n_bands = ds_int.shape

        scale = _get_scale(ds_dec, rad_grp)
        if not (
            np.isscalar(scale)
            or (isinstance(scale, np.ndarray) and scale.shape == (n_bands,))
        ):
            raise RuntimeError(
                f"Unexpected scale shape {getattr(scale, 'shape', type(scale))}; "
                "expected scalar or 1-D per-band array."
            )

        int_nd = _to_float(ds_int.attrs.get("Data_Ignore_Value"))
        dec_nd = _to_float(ds_dec.attrs.get("Data_Ignore_Value"))

        meta = rad_grp["Metadata"]
        wavelengths = meta["Spectral_Data"]["Wavelength"][:]
        fwhm = meta["Spectral_Data"]["FWHM"][:]
        map_info = _decode(meta["Coordinate_System"]["Map_Info"][()])
        try:
            coord_sys = _decode(
                meta["Coordinate_System"]["Coordinate_System_String"][()]
            )
        except Exception:
            coord_sys = None

        with open(rad_path, "wb") as out:
            row = 0
            while row < n_lines:
                r1 = min(row + chunk_rows, n_lines)
                I = ds_int[row:r1].astype(np.float32)
                D = ds_dec[row:r1].astype(np.float32)

                nodata_mask = np.zeros(I.shape, dtype=bool)
                if int_nd is not None:
                    nodata_mask |= I == int_nd
                if dec_nd is not None:
                    nodata_mask |= D == dec_nd

                if np.isscalar(scale):
                    R = I + D / float(scale)
                else:
                    R = I + D / scale.reshape(1, 1, -1)

                R = R.astype(np.float32)
                R[nodata_mask] = -9999.0

                # BIL: each row on disk is (bands, samples) — transpose (samples, bands)
                for i in range(R.shape[0]):
                    out.write(R[i].T.tobytes(order="C"))
                row = r1

    _write_envi_header(
        hdr_path,
        samples=n_samples,
        lines=n_lines,
        bands=n_bands,
        interleave="bil",
        wavelengths=wavelengths,
        fwhm=fwhm,
        map_info=map_info,
        coord_sys=coord_sys,
        description="NEON HDF5 → ENVI radiance",
    )
    return rad_path, hdr_path


# --------------------------------
# Called by: convert_neon_h5
# Calls:     _write_envi_header
# --------------------------------
def convert_loc(h5_path: str, out_basename: str) -> tuple[str, str]:
    """
    Export NEON IGM_Data (easting, northing, elevation) to an ENVI BIL ``.loc`` file.

    Parameters
    ----------
    h5_path : str
        Path to the NEON L1 radiance HDF5 file.
    out_basename : str
        Output path without extension.

    Returns
    -------
    loc_path : str
    hdr_path : str
    """
    with h5py.File(h5_path, "r") as f:
        site = next(iter(f.keys()))
        loc = f[site]["Radiance"]["Metadata"]["Ancillary_Rasters"]["IGM_Data"][:]

    loc = loc.astype(np.float32)
    loc[loc == -9999] = -9999.0
    n_lines, n_samples, n_bands = loc.shape

    loc_path = out_basename + ".loc"
    with open(loc_path, "wb") as out:
        for i in range(n_lines):
            out.write(loc[i].T.tobytes(order="C"))  # BIL: (bands, samples) per row

    hdr_path = loc_path + ".hdr"
    _write_envi_header(
        hdr_path,
        samples=n_samples,
        lines=n_lines,
        bands=n_bands,
        interleave="bil",
        band_names=LOC_BAND_NAMES,
        description="NEON HDF5 → ENVI location (IGM)",
    )
    return loc_path, hdr_path


# --------------------------------
# Called by: convert_neon_h5
# Calls:     _to_decimal_hours, _write_envi_header
# --------------------------------
def convert_obs(h5_path: str, out_basename: str) -> tuple[str, str]:
    """
    Export NEON OBS_Data (observation geometry) to an ENVI BIP ``.obs`` file.

    The ten output bands are:

    1. path_length — slant range in meters
    2. sensor_azimuth — degrees clockwise from north
    3. sensor_zenith — degrees from nadir
    4. solar_azimuth — degrees clockwise from north
    5. solar_zenith — degrees from zenith
    6. toa_azimuth
    7. toa_zenith
    8. view_azimuth
    9. cos_incidence — cosine of solar incidence angle on the surface
    10. time_decimal_hours — UTC acquisition time in decimal hours

    ISOFIT expects observation geometry in BIP (Band-Interleaved-by-Pixel)
    format, which is why this file uses a different interleave from ``.rad``
    and ``.loc``.

    Parameters
    ----------
    h5_path : str
        Path to the NEON L1 radiance HDF5 file.
    out_basename : str
        Output path without extension.

    Returns
    -------
    obs_path : str
    hdr_path : str
    """
    with h5py.File(h5_path, "r") as f:
        site = next(iter(f.keys()))
        obs = f[site]["Radiance"]["Metadata"]["Ancillary_Rasters"]["OBS_Data"][
            :, :, :10
        ]

    obs = obs.astype(np.float32)
    obs[:, :, 9] = _to_decimal_hours(obs[:, :, 9])
    obs[obs == -9999] = -9999.0
    n_lines, n_samples, n_bands = obs.shape

    obs_path = out_basename + ".obs"
    with open(obs_path, "wb") as out:
        for i in range(n_lines):
            # BIP: each row is (samples, bands) in C order — no transpose needed
            out.write(obs[i].tobytes(order="C"))

    hdr_path = obs_path + ".hdr"
    _write_envi_header(
        hdr_path,
        samples=n_samples,
        lines=n_lines,
        bands=n_bands,
        interleave="bip",
        band_names=OBS_BAND_NAMES,
        description="NEON HDF5 → ENVI observation geometry",
    )
    return obs_path, hdr_path


# --------------------------------
# Called by: (user / external callers)
# Calls:     convert_rad, convert_loc, convert_obs
# --------------------------------
def convert_neon_h5(
    h5_path: str, out_root: str, *, subdir: bool = True
) -> dict[str, str]:
    """
    Convert a NEON HDF5 file to the full ISOFIT input triplet (.rad, .loc, .obs).

    Parameters
    ----------
    h5_path : str
        Path to the NEON L1 radiance HDF5 file.
    out_root : str
        Root output directory.  With ``subdir=True`` (default), files are
        placed in ``<out_root>/<stem>/``; with ``subdir=False`` directly in
        ``<out_root>/``.
    subdir : bool, default True
        Whether to create a per-file subdirectory named after the H5 stem.

    Returns
    -------
    dict[str, str]
        Mapping of ``{"rad", "rad_hdr", "loc", "loc_hdr", "obs", "obs_hdr"}``
        to the written file paths.

    Examples
    --------
    >>> paths = convert_neon_h5("flight.h5", "./envi_output")
    >>> paths["rad"]
    './envi_output/flight/flight.rad'
    """
    h5_path = Path(h5_path)
    out_dir = Path(out_root) / h5_path.stem if subdir else Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = str(out_dir / h5_path.stem)

    rad, rad_hdr = convert_rad(str(h5_path), base)
    loc, loc_hdr = convert_loc(str(h5_path), base)
    obs, obs_hdr = convert_obs(str(h5_path), base)

    return {
        "rad": rad,
        "rad_hdr": rad_hdr,
        "loc": loc,
        "loc_hdr": loc_hdr,
        "obs": obs,
        "obs_hdr": obs_hdr,
    }
