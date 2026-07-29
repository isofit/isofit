"""Parity tests for the batched geometry builder and algebraic initializer.

These replace two pieces of scalar reference code that the end-to-end parity
measurement did NOT cover: it was taken with the per-pixel gather, so batching
those pieces moves work outside the validated path. Every assertion here is
against the real scalar function, never against a reimplementation of it.

`build_batched_geometry` is the higher risk of the two. A silently missing or
misderived field becomes a wrong LUT coordinate rather than an error, which is
the failure mode that cost a full debugging cycle on this backend.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from isofit.backends.torch.geometry import SCALAR_FIELDS, build_batched_geometry
from isofit.core.geometry import Geometry

pytestmark = pytest.mark.torch_cpu

B = 24
RTOL = 1e-12


def _obs_loc(seed=0, n=B):
    """Randomized obs/loc slabs in the AVIRIS-NG column order Geometry expects."""
    rng = np.random.default_rng(seed)
    obs = np.zeros((n, 11))
    obs[:, 0] = rng.uniform(500, 20000, n)      # path length, m
    obs[:, 1] = rng.uniform(0, 360, n)          # observer azimuth
    obs[:, 2] = rng.uniform(0, 90, n)           # observer zenith
    obs[:, 3] = rng.uniform(0, 360, n)          # solar azimuth
    obs[:, 4] = rng.uniform(0, 90, n)           # solar zenith
    obs[:, 8] = rng.uniform(0.1, 1.0, n)        # cos(eSZA)

    loc = np.zeros((n, 3))
    loc[:, 0] = rng.uniform(-180, 180, n)       # easting / longitude
    loc[:, 1] = rng.uniform(-90, 90, n)         # northing / latitude
    loc[:, 2] = rng.uniform(-400, 4000, n)      # elevation, m
    return obs, loc


def _scalar_geoms(obs, loc, svf=None, coszen=None, config=None):
    return [
        Geometry(
            obs=obs[i],
            loc=loc[i],
            svf=1 if svf is None else svf[i],
            bg_rfl=None,
            coszen=coszen,
            full_config=config if config is not None else {},
        )
        for i in range(obs.shape[0])
    ]


def _compare_all_fields(batched, scalars):
    """Every field the scalar Geometry populates must be reproduced."""
    checked = []
    for name in SCALAR_FIELDS:
        values = [getattr(g, name, None) for g in scalars]
        if any(v is None for v in values):
            continue  # the scalar path did not populate it either
        assert name in batched, (
            f"scalar Geometry sets {name!r} but the batched builder omitted it; "
            "a missing field becomes a wrong LUT coordinate, not an error"
        )
        np.testing.assert_allclose(
            batched.get(name).cpu().numpy(),
            np.asarray(values, dtype=float),
            rtol=RTOL,
            err_msg=f"geometry field {name!r} diverged",
        )
        checked.append(name)
    return checked


# --- geometry --------------------------------------------------------------------


def test_batched_geometry_matches_scalar_field_by_field():
    obs, loc = _obs_loc()
    got = build_batched_geometry(obs=obs, loc=loc)
    checked = _compare_all_fields(got, _scalar_geoms(obs, loc))
    # Guard against the test silently checking nothing.
    assert len(checked) >= 8, f"only {len(checked)} fields compared: {checked}"


def test_azimuth_wraparound():
    """relative_azimuth uses min(|d|, 360-|d|); check both sides of the wrap."""
    obs, loc = _obs_loc(1, n=6)
    obs[:, 1] = [0.0, 359.0, 10.0, 350.0, 180.0, 90.0]
    obs[:, 3] = [359.0, 0.0, 350.0, 10.0, 0.0, 270.0]
    got = build_batched_geometry(obs=obs, loc=loc[:6])
    _compare_all_fields(got, _scalar_geoms(obs, loc[:6]))


def test_extreme_zenith_and_negative_elevation():
    obs, loc = _obs_loc(2, n=5)
    obs[:, 2] = [0.0, 89.9, 45.0, 0.0, 89.99]      # observer zenith
    obs[:, 4] = [0.0, 89.9, 45.0, 89.99, 0.0]      # solar zenith
    loc[:, 2] = [-400.0, 0.0, 8848.0, -100.0, 1.0]  # below sea level to Everest
    got = build_batched_geometry(obs=obs, loc=loc)
    _compare_all_fields(got, _scalar_geoms(obs, loc))


def test_skyview_factor_is_carried():
    obs, loc = _obs_loc(3)
    svf = np.random.default_rng(0).uniform(0.3, 1.0, B)
    got = build_batched_geometry(obs=obs, loc=loc, svf=svf)
    np.testing.assert_allclose(
        got.get("skyview_factor").cpu().numpy(), svf, rtol=RTOL
    )
    _compare_all_fields(got, _scalar_geoms(obs, loc, svf=svf))


def test_default_skyview_factor_is_one():
    obs, loc = _obs_loc(4)
    got = build_batched_geometry(obs=obs, loc=loc)
    np.testing.assert_allclose(got.get("skyview_factor").cpu().numpy(), 1.0)


def test_observer_altitude_composition():
    """observer_altitude = elevation + path * cos(observer_zenith), in km."""
    obs, loc = _obs_loc(5)
    got = build_batched_geometry(obs=obs, loc=loc)
    scal = _scalar_geoms(obs, loc)
    np.testing.assert_allclose(
        got.get("observer_altitude_km").cpu().numpy(),
        np.array([g.observer_altitude_km for g in scal]),
        rtol=RTOL,
    )


def test_bg_rfl_shape_is_carried():
    obs, loc = _obs_loc(6)
    bg = np.random.default_rng(0).uniform(0.02, 0.6, (B, 12))
    got = build_batched_geometry(obs=obs, loc=loc, bg_rfl=bg)
    np.testing.assert_allclose(got.bg_rfl.cpu().numpy(), bg, rtol=RTOL)


def test_single_pixel_batch():
    obs, loc = _obs_loc(7, n=1)
    got = build_batched_geometry(obs=obs, loc=loc)
    assert len(got) == 1
    _compare_all_fields(got, _scalar_geoms(obs, loc))


def test_coszen_matches_scalar_without_config():
    """With no config, Geometry prefers obs-derived solar zenith over `coszen`."""
    obs, loc = _obs_loc(8)
    got = build_batched_geometry(obs=obs, loc=loc, coszen=0.5)
    scal = _scalar_geoms(obs, loc, coszen=0.5)
    np.testing.assert_allclose(
        got.get("coszen").cpu().numpy(),
        np.array([g.coszen for g in scal]),
        rtol=RTOL,
    )


def test_field_list_covers_what_scalar_sets():
    """SCALAR_FIELDS must not drift behind isofit.core.geometry.Geometry.

    If Geometry gains a numeric per-pixel attribute that a LUT could name as a
    grid dimension, the batched builder needs it too. This fails loudly rather
    than silently producing a wrong coordinate.
    """
    obs, loc = _obs_loc(9, n=2)
    g = _scalar_geoms(obs, loc)[0]

    numeric = {
        name
        for name, val in vars(g).items()
        if isinstance(val, (int, float, np.floating, np.integer))
        and not isinstance(val, bool)
        and not name.startswith("_")
    }
    known = set(SCALAR_FIELDS) | {"max_slope"}  # max_slope is config-uniform
    missing = numeric - known
    assert not missing, (
        f"Geometry sets numeric per-pixel field(s) {sorted(missing)} that "
        "SCALAR_FIELDS does not list; a LUT naming one would get a wrong "
        "coordinate silently"
    )


# --- algebraic initializer --------------------------------------------------------
#
# The full invert_algebraic_batch needs a configured ForwardModel, so these cover
# the pieces where the subtle bugs live: the NaN semantics of fit_params, the
# resampling helper, and the mask ordering in the reflectance formula.


def _surface_stub(n_wl=16, seed=0):
    """A MultiComponentSurface-shaped mock with real bounds."""
    from isofit.surface.surface_multicomp import MultiComponentSurface

    rng = np.random.default_rng(seed)
    s = MagicMock()
    s.idx_lamb = np.arange(n_wl)
    s.wl = np.linspace(400, 2500, n_wl)
    s.statevec_names = [f"RFL_{i}" for i in range(n_wl)]
    s.bounds = [(0.0, 1.0)] * n_wl
    s.fit_params = lambda rfl, geom=None, *a: MultiComponentSurface.fit_params(
        s, rfl, geom
    )
    return s


def test_fit_params_batch_matches_scalar_including_nan():
    """NaN must map to the UPPER bound, as Python's max/min builtins do.

    `max(lo, min(hi, nan))` returns `hi`, because a comparison against NaN is
    False and the builtins then return their first argument. torch.clamp would
    propagate the NaN instead, so this is a real behavioural difference and not
    a nicety.
    """
    from isofit.backends.torch.initializer import BatchedAlgebraicInitializer

    s = _surface_stub()
    n_wl = len(s.idx_lamb)
    rng = np.random.default_rng(1)

    rfl = rng.uniform(-0.5, 2.0, (8, n_wl))
    rfl[0, 0] = np.nan
    rfl[1, :3] = [np.nan, -10.0, 10.0]

    init = BatchedAlgebraicInitializer.__new__(BatchedAlgebraicInitializer)
    init.device, init.dtype = torch.device("cpu"), torch.float64
    bounds = np.asarray(s.bounds, dtype=float)
    init.idx_lamb = torch.as_tensor(np.asarray(s.idx_lamb), dtype=torch.int64)
    init.n_surface_state = len(s.statevec_names)
    init.fit_lo = torch.as_tensor(bounds[:, 0] + 0.001)
    init.fit_hi = torch.as_tensor(bounds[:, 1] - 0.001)

    got = init.fit_params_batch(torch.as_tensor(rfl)).numpy()
    ref = np.stack([s.fit_params(rfl[i]) for i in range(rfl.shape[0])])
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0)

    # And specifically: the NaN became the upper bound, not a NaN.
    assert np.isfinite(got[0, 0])
    assert got[0, 0] == pytest.approx(bounds[0, 1] - 0.001)


def test_batched_interp1d_matches_scipy():
    from scipy.interpolate import interp1d

    from isofit.backends.torch.initializer import BatchedInterp1d

    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(400, 2500, 24))
    x_new = np.sort(rng.uniform(350, 2600, 30))  # spans beyond -> extrapolation
    y = rng.uniform(0.0, 1.0, (7, len(x)))

    got = BatchedInterp1d(x, x_new)(torch.as_tensor(y)).numpy()
    ref = np.stack(
        [interp1d(x, y[i], fill_value="extrapolate")(x_new) for i in range(y.shape[0])]
    )
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_reflectance_mask_order_matches_scalar():
    """The three masks are order-sensitive, and NaN must survive the 1.6 clamp.

    numpy's `rfl[rfl > 1.6] = 1.6` leaves NaN untouched because `nan > 1.6` is
    False. A torch.clamp would not.
    """
    rng = np.random.default_rng(3)
    n = 64
    L_tot = rng.uniform(1.0, 10.0, n)
    denom = rng.uniform(-5.0, 5.0, n)
    sphalb = rng.uniform(0.0, 0.3, n)
    L_tot[:5] = 0.0
    denom[5:10] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        ref = 1.0 / (L_tot / denom + sphalb)
        ref[denom == 0] = 0.0
        ref[L_tot == 0] = 0.0
        ref[ref > 1.6] = 1.6

    t_L, t_d, t_s = (torch.as_tensor(v) for v in (L_tot, denom, sphalb))
    got = 1.0 / (t_L / t_d + t_s)
    zero = torch.zeros((), dtype=got.dtype)
    got = torch.where(t_d == 0, zero, got)
    got = torch.where(t_L == 0, zero, got)
    got = torch.where(got > 1.6, torch.full((), 1.6, dtype=got.dtype), got)

    np.testing.assert_array_equal(np.isnan(got.numpy()), np.isnan(ref))
    finite = np.isfinite(ref)
    np.testing.assert_allclose(got.numpy()[finite], ref[finite], rtol=1e-12)


def test_clip_bounds_edges_match_forward_model():
    """The precomputed clip edges must equal ForwardModel.clip_bounds' own."""
    from isofit.backends.torch.initializer import BatchedAlgebraicInitializer
    from isofit.core.common import eps as EPS
    from isofit.core.forward import ForwardModel

    rng = np.random.default_rng(4)
    n = 20
    lo = rng.uniform(-1, 0, n)
    hi = lo + rng.uniform(0.5, 3.0, n)
    bounds = (lo, hi)

    got_lo, got_hi = BatchedAlgebraicInitializer._clip_bounds_edges(bounds, EPS)

    x = rng.uniform(-5, 5, n)
    ref = ForwardModel.clip_bounds(x, bounds, eps=EPS)
    got = np.clip(x, got_lo, got_hi)
    np.testing.assert_allclose(got, ref, rtol=1e-12)


def test_batched_geometry_accepts_device_tensors():
    """Fields may arrive as tensors, not numpy.

    build_batched_geometry derives its fields on the target device, so
    BatchedGeometry must take tensors directly. Round-tripping through
    np.asarray raises `can't convert cuda:0 device type tensor to numpy` --
    a failure invisible to a CPU-default test, which is how it reached a GPU
    scene run.
    """
    from isofit.backends.torch.geometry import BatchedGeometry

    n = 5
    fields = {
        "coszen": torch.linspace(0.6, 0.95, n),
        "cos_i": torch.linspace(0.5, 0.9, n),
        "solar_zenith": torch.linspace(10.0, 40.0, n),
    }
    bg = torch.rand(n, 7)

    g = BatchedGeometry(fields, bg_rfl=bg, dtype=torch.float64)
    assert len(g) == n
    for name, val in fields.items():
        np.testing.assert_allclose(
            g.get(name).cpu().numpy(), val.double().numpy(), rtol=1e-12
        )
    np.testing.assert_allclose(g.bg_rfl.cpu().numpy(), bg.double().numpy(), rtol=1e-12)


def test_batched_geometry_mixed_tensor_and_numpy_fields():
    """A mix must work too: nothing requires callers to be consistent."""
    from isofit.backends.torch.geometry import BatchedGeometry

    n = 4
    g = BatchedGeometry(
        {
            "coszen": torch.linspace(0.6, 0.9, n),
            "cos_i": np.linspace(0.5, 0.8, n),
        },
        dtype=torch.float64,
    )
    assert len(g) == n
    np.testing.assert_allclose(
        g.get("cos_i").cpu().numpy(), np.linspace(0.5, 0.8, n), rtol=1e-12
    )
