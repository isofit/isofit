"""
Writes the ISOFIT config for the main inversion.

The config is built by :mod:`isofit.retrieval.utils.solve` from the retrieval
config, with the LUT grid and atmosphere statevector derived here from the
scene statistics and the presolve.
"""

import json
import logging
from typing import Optional

import numpy as np
from spectral.io import envi

from isofit.atmosphere.atmosphere import (
    aeronet_model,
    modtran_aot_lowerbound_polynomials,
)
from isofit.core.common import envi_header
from isofit.luts.reader import inspect_lut_dimensions
from isofit.retrieval.stages.scene import Scene
from isofit.retrieval.utils import solve, utils

Logger = logging.getLogger(__name__)

Emulators = ("srtmnet", "sixs", "kernelflows")
"""Engines that grid aerosol as AOT550 rather than MODTRAN's AERFRAC_2"""


@utils.log_elapse
def configure(config) -> None:
    """
    Write the ISOFIT config for the main inversion: the LUT grids, aerosol
    climatology, and statevector derived from the scene and presolve.
    """
    scene = Scene.load(config)
    atm = config.forward_model.atmosphere

    names = list(atm.statevector_names)
    lut_grid = {**scene.grids, "H2OSTR": h2o_grid(config, scene)}

    if climatology := match_climatology(config, scene):
        aerosol = climatology["aerosol_lut_grid"]
        priors = climatology["aerosol_state_vector"]
    else:
        aerosol, priors = aerosol_grid(config, scene)

    # The configured aerosol elements are replaced by those actually gridded
    names = [name for name in names if not name.startswith(("AOT", "AERFRAC"))]
    names += list(aerosol)
    lut_grid.update(aerosol)

    if "CO2" in names:
        lut_grid["CO2"] = solve.grid(*config.lut.co2.range, config.lut.co2)

    lut_grid = {key: val for key, val in lut_grid.items() if val and len(val) > 1}

    if atm.engine.name == "kernelflows":
        from isofit.atmosphere.engines.kernel_flows import bounds_check

        bounds_check(lut_grid, atm.engine.emulator_file, modify=True)

    lut_subset = {key: None for key in lut_grid}
    if config.lut.prebuilt:
        lut_grid, lut_subset = reconcile(config, lut_grid, scene)

    statevector = solve.statevector(
        names=names,
        lut_grid=lut_grid,
        configured=atm.statevector.model_dump(mode="json"),
        priors=priors,
    )

    data = solve.base(config, config.configure.layers)
    data["forward_model"]["atmosphere"].update(
        lut_grid=lut_grid,
        lut_subset=lut_subset,
        statevector=statevector,
        statevector_names=list(statevector),
    )

    if climatology:
        data["forward_model"]["atmosphere"]["engine"]["aerosol_model_file"] = (
            climatology["aerosol_mdl_path"]
        )

    solve.write(data, config.configure.file)


def h2o_grid(config, scene: Scene) -> Optional[list[float]]:
    """
    The H2O LUT grid.

    With the presolve, the range spans the 2nd to 98th percentiles of the
    presolved water vapor plus a margin of half that span, at least
    ``lut.h2o.spacing``. Otherwise it is ``lut.h2o.range``, or 0.2 to the
    scene's maximum water vapor.
    """
    dim = config.lut.h2o
    lo, hi = dim.range or (0.2, scene.h2o_max)

    if config.presolve.enabled:
        file = config.presolve.state
        ds = envi.open(envi_header(str(file)), str(file))
        band = ds.metadata["band names"].index("H2OSTR")

        h2o = ds.read_band(band).ravel()
        h2o = h2o[h2o > lo]
        p02, p50, p98 = np.percentile(h2o, [2, 50, 98])

        margin = max((p98 - p02) / 2, dim.spacing)
        lo, hi = max(lo, p02 - margin), min(scene.h2o_max, max(lo, p98 + margin))

        # Ensure the range is still wide enough to be gridded
        if hi - lo < dim.spacing_min:
            lo = max(lo, p50 - dim.spacing / 2)
            hi = min(scene.h2o_max, max(lo, p50 + dim.spacing / 2))

        Logger.info(f"Presolve H2O range: [{lo:.4f}, {hi:.4f}]")

    grid = solve.grid(lo, hi, dim, config.lut.no_min_spacing)
    if grid is None:
        raise ValueError(
            f"H2O range [{lo}, {hi}] is too small to grid at lut.h2o.spacing_min "
            f"{dim.spacing_min}, but H2O must be a LUT dimension"
        )

    return grid


def aerosol_grid(config, scene: Scene) -> tuple[dict, dict]:
    """
    The aerosol LUT grid and its statevector priors.

    Emulators grid AOT550, MODTRAN grids AERFRAC_2. The range is
    ``lut.aerosol.range`` or the minimum AOT of ``scene.atmosphere_type`` to 1,
    and the prior mean and initial value sit 10% into the range, or follow the
    AERONET climatology when ``"AOT"`` is in
    ``implementation.pixel_heuristic_priors``.

    Returns
    -------
    tuple[dict, dict]
        The LUT grid and statevector priors, keyed by the aerosol LUT name.
        Empty when the aerosol is not gridded.
    """
    dim = config.lut.aerosol
    floor = modtran_aot_lowerbound_polynomials()[config.scene.atmosphere_type](0)
    lo, hi = dim.range or (floor, 1)
    lo = max(lo, floor)

    grid = solve.grid(lo, hi, dim, config.lut.no_min_spacing)
    if grid is None:
        return {}, {}

    if "AOT" in config.implementation.pixel_heuristic_priors:
        init, sigma = aeronet_model(scene.elevation_km, aot_min=lo, aot_max=hi)
        prior = {"init": init, "prior_mean": init, "prior_sigma": sigma}
    else:
        init = lo + (hi - lo) / 10
        prior = {"init": init, "prior_mean": init}

    name = (
        "AOT550"
        if config.forward_model.atmosphere.engine.name in Emulators
        else "AERFRAC_2"
    )

    return {name: grid}, {name: prior}


def match_climatology(config, scene: Scene) -> Optional[dict]:
    """
    Find the first case of ``configure.aerosol_climatology_file`` matching the
    scene's location and acquisition time.

    Each case holds ``criteria`` of ``latitude``, ``longitude``, ``month``, and
    ``year`` intervals, and the ``aerosol_state_vector``, ``aerosol_lut_grid``,
    and ``aerosol_mdl_path`` to use when they all match.
    """
    file = config.configure.aerosol_climatology_file
    if file is None:
        return None

    values = {
        "latitude": scene.latitude,
        "longitude": scene.longitude,
        "month": scene.time.month,
        "year": scene.time.year,
    }

    with open(file) as f:
        cases = json.load(f)["cases"]

    for case in cases:
        if all(lo <= values[key] <= hi for key, (lo, hi) in case["criteria"].items()):
            Logger.info(f"Matched aerosol climatology case: {case['criteria']}")
            return case

    Logger.info("No aerosol climatology case matched the scene")


def reconcile(config, lut_grid: dict, scene: Scene) -> tuple[dict, dict]:
    """
    Reconcile the LUT grid with the dimensions of a prebuilt LUT.

    - A dimension in both is subset to the LUT points encompassing the scene
      grid, clipped to the LUT bounds
    - A dimension only in the LUT is interpolated to the scene mean, or within
      its range for the aerosol and dimensions without a scene mean
    - A dimension only in the scene grid is an error, as the LUT cannot vary it

    Returns
    -------
    tuple[dict, dict]
        The LUT grid and ``lut_subset``.

    Raises
    ------
    ValueError
        If a dimension of the scene grid is not in the prebuilt LUT.
    """
    file = config.forward_model.atmosphere.lut_path
    dims = {
        key: val for key, val in inspect_lut_dimensions(file).items() if key != "wl"
    }

    if missing := set(lut_grid) - set(dims):
        raise ValueError(
            f"LUT dimensions {sorted(missing)} are not in the prebuilt LUT {file}, "
            f"which has: {sorted(dims)}"
        )

    means = {
        "observer_zenith": scene.observer_zenith,
        "solar_zenith": scene.solar_zenith,
        "relative_azimuth": scene.relative_azimuth,
        "surface_elevation_km": scene.elevation_km,
    }

    grid, subset = {}, {}
    for name, points in dims.items():
        lo, hi = float(points.min()), float(points.max())

        if name in lut_grid:
            want = lut_grid[name]
            if want[0] < lo or want[-1] > hi:
                Logger.warning(
                    f"{name} range [{want[0]:.4f}, {want[-1]:.4f}] is clipped to the "
                    f"prebuilt LUT range [{lo:.4f}, {hi:.4f}]"
                )

            below = points[points <= max(want[0], lo)]
            above = points[points >= min(want[-1], hi)]
            keep = points[(points >= below.max()) & (points <= above.min())]

            grid[name] = keep.tolist()
            subset[name] = {
                "gte": grid[name][0],
                "lte": grid[name][-1],
                "encompass": True,
            }
        else:
            if name in means:
                value = means[name]
            elif name.startswith(("AOT", "AERFRAC")):
                value = lo + (hi - lo) / 10
            else:
                value = (lo + hi) / 2

            Logger.info(
                f"{name} is only in the prebuilt LUT, interpolating to {value:.4f}"
            )
            subset[name] = {"interp": float(np.clip(value, lo, hi))}

    return grid, subset
