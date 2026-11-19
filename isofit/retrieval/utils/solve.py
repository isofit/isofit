"""
Writes the ISOFIT configs for the presolve and main inversions.

apply_oe's ``build_config`` assembled each config from ~30 arguments through the
``make_*_config`` helpers, restating defaults the base configs already hold.
Here the retrieval config is already a complete, validated ISOFIT config (the
``tc`` base configs plus any sensor mixins and overrides), so a solve config is
built by layering only what differs on top of it:

1. The retrieval config itself, dumped
2. The stage's ``layers``, e.g. ``presolve.layers``. Only each layer's own keys
   are applied, as their ``^^`` parent is the retrieval config's own base, and
   their ``${...}`` references resolve against the retrieval config, so they
   follow its paths
3. The values derived by the stage: the LUT grid and atmosphere statevector,
   the multisurface classes, and the surface and instrument statevectors (see
   :func:`base`)

The result is validated and written by :func:`write`.
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat
from spectral.io import envi

from isofit.configs import Config
from isofit.configs.loader import expand_paths, load_layer, merge_layer, unalias
from isofit.configs.utils import interpolate
from isofit.core.common import envi_header
from isofit.core.instrument import DefaultRCCPrior
from isofit.core.multistate import SurfaceMapping
from isofit.data import env
from isofit.retrieval.utils import utils

Logger = logging.getLogger(__name__)

# Sections only used by the retrieval pipeline, excluded from a written solve config
Pipeline = ("log", "lut", "setup", "preprocess", "scene", "surface", "segment")
Pipeline += ("presolve", "background", "configure", "invert", "interpolate", "cleanup")

# Files read from a stage's products, keyed by the stage that writes them. When
# the stage is disabled its product does not exist, so the file is unset
Products = {
    "input.skyview_factor_file": "preprocess.svf",
    "input.background_reflectance_file": "background",
    "forward_model.surface.surface_class_file": "surface.classes",
    "forward_model.surface.base_surface_class_file": "surface.classes",
}


def layer(config, ref: str) -> dict:
    """
    Load the own keys of a config layer, interpolated against ``config``.

    Parameters
    ----------
    config : Config
        The retrieval config, against which ``${...}`` references resolve.
    ref : str
        A config reference (see :func:`isofit.configs.loader.resolve`), e.g.
        ``"subs"`` or ``"run.yml:subs"``.

    Returns
    -------
    dict
        The layer's own keys.
    """
    own = load_layer(ref, inherit=False)
    unalias(own)

    data = config.model_dump(mode="json")
    data["layer"] = own
    data["env"] = dict(env.items())
    interpolate.interpolate(data)

    return data["layer"]


def base(config, layers: list[str]) -> dict:
    """
    Build a solve config from the retrieval config, without its atmosphere LUT
    grid and statevector.

    Parameters
    ----------
    config : Config
        The retrieval config.
    layers : list[str]
        Config references layered on top, in order.

    Returns
    -------
    dict
        The solve config, to be completed by the caller and passed to
        :func:`write`.
    """
    data = config.model_dump(mode="json")
    for ref in layers:
        merge_layer(data, layer(config, ref))

    for key, product in Products.items():
        if not produced(config, product):
            interpolate.set_dotted(data, key, None)

    fm = data["forward_model"]
    surface(fm["surface"])
    instrument(fm)

    return data


def produced(config, product: str) -> bool:
    """
    Whether a stage product exists in this retrieval: either a stage that is
    enabled, or a file that is set.
    """
    value = interpolate.get_dotted(config.model_dump(mode="json"), product)
    if isinstance(value, dict):
        return value["enabled"]

    return value is not None


def surface(s: dict) -> None:
    """
    Set the multisurface classes and the surface statevector, in place.

    For a multisurface retrieval (a ``base_surface_class_file`` is set), each
    class present in the classification maps to the model built per surface
    category by the surface stage, ``{stem}_{category}{suffix}`` of
    ``surface_file``. The non-reflectance statevector elements (e.g.
    ``SURF_TEMP_K``) are read from the surface model files, with any configured
    element taking precedence.
    """
    configured = {k: v for k, v in s["statevector"].items() if v is not None}

    files = [s["surface_file"]]
    if classes := s["base_surface_class_file"]:
        model = Path(s["surface_file"])
        ds = envi.open(envi_header(classes), classes)
        present = np.unique(ds.open_memmap(interleave="bip"))

        s["Surfaces"] = {}
        for value in present:
            category = SurfaceMapping[int(value)]
            file = model.with_name(f"{model.stem}_{category}{model.suffix}")
            s["Surfaces"][category] = {
                "surface_int": int(value),
                "surface_file": str(file),
                "surface_category": category,
                "terrain_style": s["terrain_style"],
                "max_slope": s["max_slope"],
                "use_background_rfl": s["use_background_rfl"],
            }

        files = [sub["surface_file"] for sub in s["Surfaces"].values()]
        s.update(multi_surface_flag=True, surface_file=None, surface_category=None)

    s["statevector"] = {**surface_statevector(*files), **configured}

    categories = {s["surface_category"], *s.get("Surfaces", {})}
    if "glint_model_surface" in categories and not s["refractive_index_path"]:
        file = env.path("data", "h2o_real_refractive_index.txt")
        s["refractive_index_path"] = str(file)


def surface_statevector(*files) -> dict:
    """
    The non-reflectance statevector elements stored in surface model files, as
    written by :func:`isofit.utils.surface_model`.
    """
    statevector = {}
    for file in files:
        mat = loadmat(file)
        for i, name in enumerate(mat.get("statevec_names", [])):
            statevector[str(name).strip()] = {
                "bounds": [float(b) for b in mat["bounds"][i]],
                "scale": float(mat["scale"][0][i]),
                "init": float(mat["init"][0][i]),
                "prior_mean": float(mat["prior_mean"][0][i]),
                "prior_sigma": float(mat["prior_sigma"][0][i]),
            }

    return statevector


def instrument(fm: dict) -> None:
    """
    Apply the instrument rules of apply_oe, in place.

    - The model discrepancy replaces the uncorrelated radiometric uncertainty,
      so the two are not both applied
    - Every column of the EOF file is a statevector element ``EOF_{n}``, using
      the default RCC prior unless configured
    """
    inst = fm["instrument"]
    inst["statevector"] = {k: v for k, v in inst["statevector"].items() if v}

    if fm["model_discrepancy_file"]:
        inst["unknowns"]["uncorrelated_radiometric_uncertainty"] = 0

    if eof := inst["eof_path"]:
        columns = np.loadtxt(eof, ndmin=2).shape[1]
        for n in range(1, columns + 1):
            inst["statevector"].setdefault(f"EOF_{n}", DefaultRCCPrior._asdict())


# Default ``(prior_sigma, scale)`` of the atmosphere statevector elements, as set
# by apply_oe. ``AERFRAC`` applies to every ``AERFRAC_{n}``
Defaults = {
    "H2OSTR": (100.0, 1),
    "AOT550": (0.1, 1),
    "AERFRAC": (0.1, 1),
    "surface_elevation_km": (1000.0, 100),
    "CO2": (100.0, 10),
}


def statevector(
    names: list[str],
    lut_grid: dict[str, list[float]],
    configured: dict,
    priors: Optional[dict] = None,
) -> dict:
    """
    Build the atmosphere statevector from the LUT grid.

    Each element is bounded by its LUT grid, and initialized with a prior mean
    at the grid midpoint. Its ``scale`` and ``prior_sigma`` come from the
    configured element when set, otherwise from :data:`Defaults`.
    ``priors`` then overrides any of these, e.g. a heuristic aerosol prior.

    Parameters
    ----------
    names : list[str]
        The elements to retrieve, each of which must be gridded.
    lut_grid : dict[str, list[float]]
        The LUT grid.
    configured : dict
        The configured ``forward_model.atmosphere.statevector``.
    priors : dict, optional
        Element values keyed by element name, taking precedence.

    Returns
    -------
    dict
        The statevector.

    Raises
    ------
    ValueError
        If an element is not in the LUT grid.
    """
    priors = priors or {}

    statevector = {}
    for name in names:
        if name not in lut_grid:
            raise ValueError(
                f"{name!r} is in forward_model.atmosphere.statevector_names but "
                f"is not a LUT dimension, its range may be too small to grid. "
                f"LUT dimensions: {list(lut_grid)}"
            )

        lo, hi = lut_grid[name][0], lut_grid[name][-1]
        key = "AERFRAC" if name.startswith("AERFRAC") else name
        sigma, scale = Defaults.get(key, (np.nan, 1))

        given = configured.get(name) or {}
        if math.isfinite(given.get("prior_sigma", math.nan)):
            sigma = given["prior_sigma"]
        if math.isfinite(given.get("scale", math.nan)):
            scale = given["scale"]

        statevector[name] = {
            "bounds": [lo, hi],
            "scale": scale,
            "init": (lo + hi) / 2,
            "prior_mean": (lo + hi) / 2,
            "prior_sigma": sigma,
            **priors.get(name, {}),
        }

    return statevector


def grid(
    lo: float, hi: float, dim, no_min_spacing: bool = False
) -> Optional[list[float]]:
    """
    Grid ``[lo, hi]`` at roughly ``dim.spacing``.

    Parameters
    ----------
    lo, hi : float
        Range to grid.
    dim : LutDimension
        The ``lut.*`` dimension heuristics.
    no_min_spacing : bool, default=False
        Keep at least 2 points even when their spacing is below
        ``dim.spacing_min``.

    Returns
    -------
    list[float] or None
        The grid, or None when the dimension should not be gridded.

    Examples
    --------
    >>> from isofit.configs.sections.pipeline import LutDimension
    >>> dim = LutDimension(spacing=0.25, spacing_min=0.03, range=None)
    >>> grid(0.5, 1.0, dim)
    [0.5, 0.75, 1.0]
    >>> grid(0.5, 0.51, dim) is None
    True
    """
    if dim.spacing == 0:
        return None

    n = int(np.ceil((hi - lo) / dim.spacing)) + 1
    if no_min_spacing and n == 1 and not np.isclose(hi, lo):
        n = 2

    points = np.linspace(lo, hi, n)
    if dim.spacing_min > 0.0001:
        points = points.round(4)

    if len(points) == 1:
        return None

    # Two points too close together are not worth a dimension
    if (
        len(points) == 2
        and points[1] - points[0] < dim.spacing_min
        and not no_min_spacing
    ):
        return None

    return points.tolist()


def write(data: dict, file: Path) -> Config:
    """
    Validate a solve config and write it to ``file``.

    Paths are written absolute, as ISOFIT expands relative paths against the
    config file's directory. The pipeline sections are excluded, as ISOFIT
    ignores them. The directories of every path in the config are created, as
    the layers may write outside those created by the setup stage.

    Returns
    -------
    Config
        The validated solve config.
    """
    solve = Config.model_validate(data)
    solve.get_config_errors()

    out = solve.model_dump(mode="json", exclude=set(Pipeline))
    expand_paths(out, Path.cwd())
    out["output"]["dir"] = str(Path(out["output"]["dir"]).resolve())

    dirs = {path.parent for path in utils.walk_paths(solve)}
    dirs |= {file.parent, solve.output.dir, solve.forward_model.atmosphere.sim_path}
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)

    Logger.info(f"Writing {file}")
    with open(file, "w") as f:
        json.dump(out, f, indent=4)

    return solve
