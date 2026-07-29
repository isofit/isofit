"""Tests for the batched driver's H2O_ABSCO Kb column.

Two things are being asserted here, and they pull in opposite directions:

* The default path must stay a finite difference, byte-for-byte the same
  computation :meth:`ForwardModel.drdn_datmosphereb` performs. Silently
  upgrading it would change every retrieval ISOFIT produces.
* The opt-in analytic path must actually be *right*, and right in the same
  units. It is validated against a high-accuracy CENTRAL difference of the
  radiance model, taken with the same multiplicative perturbation convention
  the scalar code uses. The shipped forward difference is deliberately expected
  to miss that reference by O(eps) -- that gap is the error being removed.

The forward model is assembled from mocks in the style of ``test_torch_forward``
so no downloaded LUT is required.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from isofit.backends.torch.driver import AnalyticalBatchSolver
from isofit.backends.torch.forward import (
    TorchRadiance,
    calc_rdn_bgrfl_heterogeneous,
    calc_rdn_bgrfl_homogeneous,
    terrain_rereflection_heterogeneous,
    terrain_rereflection_homogeneous,
)
from isofit.backends.torch.geometry import BatchedGeometry
from isofit.backends.torch.lut import BatchedLUT

pytestmark = pytest.mark.torch_cpu

N_WL = 8
B = 6

COUPLING_TERMS = [
    "transm_down_dir_transm_up_dir",
    "transm_down_dif_transm_up_dir",
    "transm_down_dir_transm_up_dif",
    "transm_down_dif_transm_up_dif",
]
LUT_KEYS = [
    "rhoatm",
    "sphalb",
    "transm_down_dir",
    "transm_down_dif",
    "transm_up_dir",
    "transm_up_dif",
] + COUPLING_TERMS

#: LUT axes: a geometry axis, then the two retrieved state elements.
LUT_NAMES = ["solar_zenith", "H2OSTR", "AOT550"]


def _build(multipart=True, use_background_rfl=True, rt_mode="transm", seed=0):
    """A solver wired to a synthetic LUT, plus a batch of inputs to drive it."""
    from isofit.backends.torch.atmosphere import TorchAtmosphere

    rng = np.random.default_rng(seed)

    grids = [
        np.linspace(10.0, 50.0, 5),
        np.linspace(0.5, 3.0, 6),
        np.linspace(0.0, 0.6, 4),
    ]
    shape = tuple(len(g) for g in grids)
    lut = BatchedLUT(
        grids, {k: rng.uniform(0.05, 0.35, (*shape, N_WL)) for k in LUT_KEYS}
    )

    atm = TorchAtmosphere.__new__(TorchAtmosphere)
    atm.device = torch.device("cpu")
    atm.dtype = torch.float64
    atm.lut = lut
    atm.rt_mode = rt_mode
    atm.multipart_transmittance = multipart
    atm.coupling_terms = COUPLING_TERMS
    atm.esd_correction = 1.03
    atm.solar_irr = torch.as_tensor(rng.uniform(0.5, 1.5, N_WL))
    atm.n_wl = N_WL
    atm.lut_names = LUT_NAMES
    atm.n_dims = len(LUT_NAMES)
    atm.idx_x_RT = [1, 2]
    atm.idx_geom = {0: "solar_zenith"}
    atm.convert_observer_zenith = None

    rad = TorchRadiance.__new__(TorchRadiance)
    rad.device = torch.device("cpu")
    rad.dtype = torch.float64
    rad.atmosphere = atm
    rad.multipart_transmittance = multipart
    rad.use_background_rfl = use_background_rfl
    if use_background_rfl:
        rad.terrain_rereflection = terrain_rereflection_heterogeneous
        rad.calc_rdn_bgrfl = calc_rdn_bgrfl_heterogeneous
    else:
        rad.terrain_rereflection = terrain_rereflection_homogeneous
        rad.calc_rdn_bgrfl = calc_rdn_bgrfl_homogeneous

    fm = MagicMock()
    fm.atmosphere.statevec_names = ["H2OSTR", "AOT550"]

    solver = AnalyticalBatchSolver.__new__(AnalyticalBatchSolver)
    solver.device = torch.device("cpu")
    solver.dtype = torch.float64
    solver.radiance = rad
    solver.fm = fm
    solver.analytic_derivatives = False

    geom = BatchedGeometry(
        {
            "solar_zenith": rng.uniform(15.0, 45.0, B),
            "coszen": rng.uniform(0.6, 0.95, B),
            "cos_i": rng.uniform(0.6, 0.95, B),
            "skyview_factor": rng.uniform(0.8, 1.0, B),
        }
    )

    inputs = dict(
        # Kept well inside the H2OSTR axis so the derivative is not clamped.
        x_atm=torch.as_tensor(
            np.stack([rng.uniform(0.9, 2.6, B), rng.uniform(0.05, 0.5, B)], axis=1)
        ),
        geom=geom,
        rho=torch.as_tensor(rng.uniform(0.05, 0.4, (B, N_WL))),
        rho_dif_dif=torch.as_tensor(rng.uniform(0.05, 0.4, (B, N_WL))),
    )
    inputs["Ls"] = torch.zeros_like(inputs["rho"])
    return solver, inputs


def _rdn(solver, inputs, x_atm=None):
    """Modeled radiance, optionally at a perturbed atmospheric state.

    The four reflectance arguments must match the driver's own ``calc_rdn``
    calls, which in turn match ``ForwardModel.Seps`` (forward.py:659-669):
    both downward-diffuse terms are the background reflectance. If this drifts
    from the driver, the finite-difference reference below stops describing the
    quantity the analytic column computes. See test_torch_bgrfl_reflectance.py.
    """
    rad = solver.radiance
    x_atm = inputs["x_atm"] if x_atm is None else x_atm
    (
        r,
        L_tot,
        L_dir_dir,
        L_dif_dir,
        L_dir_dif,
        L_dif_dif,
    ) = rad.calc_atmosphere_quantities(
        x_atm, inputs["geom"], rho_dif_dif=inputs["rho_dif_dif"]
    )
    rho = inputs["rho"]
    return rad.calc_rdn(
        rho, rho, inputs["rho_dif_dif"], inputs["rho_dif_dif"], inputs["Ls"],
        L_tot, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif, r, inputs["geom"],
    )


def _central_reference(solver, inputs, h=1e-6):
    """``x_H2O * d rdn / d x_H2O`` by central difference.

    Uses the *multiplicative* perturbation ``x * (1 +/- h)``, matching
    ``drdn_datmosphereb``'s convention, so the result is directly comparable to
    both Kb columns rather than to ``d rdn / d x`` alone.
    """

    def bumped(step):
        x = inputs["x_atm"].clone()
        x[:, 0] = x[:, 0] * (1.0 + step)
        return _rdn(solver, inputs, x_atm=x)

    return (bumped(h) - bumped(-h)) / (2 * h)


def _columns(solver, inputs):
    rdn = _rdn(solver, inputs)
    fd = solver._h2o_absco_column_fd(
        inputs["x_atm"], inputs["geom"], inputs["rho"],
        inputs["rho_dif_dif"], inputs["Ls"], rdn,
    )
    analytic = solver._h2o_absco_column_analytic(
        inputs["x_atm"], inputs["geom"], inputs["rho"],
        inputs["rho_dif_dif"], inputs["Ls"],
    )
    return fd, analytic


MODES = [
    dict(multipart=True, use_background_rfl=True, rt_mode="transm"),
    dict(multipart=True, use_background_rfl=False, rt_mode="transm"),
    dict(multipart=True, use_background_rfl=True, rt_mode="rdn"),
    dict(multipart=False, use_background_rfl=False, rt_mode="transm"),
    dict(multipart=False, use_background_rfl=False, rt_mode="rdn"),
]
MODE_IDS = [
    "6comp-hetero-transm",
    "6comp-homog-transm",
    "6comp-hetero-rdn",
    "1comp-transm",
    "1comp-rdn",
]


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_analytic_column_matches_central_difference(mode):
    """The analytic chain rule must reproduce the true derivative.

    The tolerance is set by the central difference's own cancellation floor
    (~1e-10 relative at h=1e-6 in fp64), not by the analytic column.
    """
    solver, inputs = _build(**mode)
    _, analytic = _columns(solver, inputs)
    reference = _central_reference(solver, inputs)

    np.testing.assert_allclose(
        analytic.numpy(), reference.numpy(), rtol=1e-7, atol=1e-9
    )


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_analytic_column_beats_the_finite_difference(mode):
    """The analytic column is closer to the truth than the shipped FD.

    If this ever fails, the analytic path is not buying the accuracy it costs.
    """
    solver, inputs = _build(**mode)
    fd, analytic = _columns(solver, inputs)
    reference = _central_reference(solver, inputs)

    fd_err = float((fd - reference).abs().max())
    analytic_err = float((analytic - reference).abs().max())

    assert fd_err > 0, "the forward difference should carry truncation error"
    assert analytic_err < fd_err / 100


@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
def test_analytic_and_finite_difference_agree_to_truncation_order(mode):
    """The two columns describe the same quantity, in the same units.

    A scaling mistake -- forgetting that ``drdn_datmosphereb`` perturbs
    multiplicatively and so returns ``x * d rdn / dx`` -- would show up here as
    a factor-of-x discrepancy rather than an O(eps) one.
    """
    solver, inputs = _build(**mode)
    fd, analytic = _columns(solver, inputs)

    scale = float(analytic.abs().max())
    assert scale > 0
    assert float((analytic - fd).abs().max()) < 1e-4 * scale


def test_finite_difference_is_the_default():
    """Default numerics must not change: the default column IS the FD column."""
    solver, inputs = _build()
    assert solver.analytic_derivatives is False

    rdn = _rdn(solver, inputs)
    default = solver._h2o_absco_column(
        inputs["x_atm"], inputs["geom"], inputs["rho"],
        inputs["rho_dif_dif"], inputs["Ls"], rdn,
    )
    fd, analytic = _columns(solver, inputs)

    np.testing.assert_array_equal(default.numpy(), fd.numpy())
    assert not np.array_equal(default.numpy(), analytic.numpy())


def test_flag_switches_to_the_analytic_column():
    solver, inputs = _build()
    solver.analytic_derivatives = True

    rdn = _rdn(solver, inputs)
    selected = solver._h2o_absco_column(
        inputs["x_atm"], inputs["geom"], inputs["rho"],
        inputs["rho_dif_dif"], inputs["Ls"], rdn,
    )
    _, analytic = _columns(solver, inputs)

    np.testing.assert_array_equal(selected.numpy(), analytic.numpy())


def test_analytic_column_is_zero_when_h2o_is_off_the_lut_grid():
    """Clamped water vapour means a locally constant interpolant.

    The finite difference returns exactly zero there too -- both perturbed and
    unperturbed states clamp to the same LUT corner -- so this is parity, not a
    divergence.
    """
    solver, inputs = _build()
    inputs["x_atm"][:, 0] = 99.0  # far above the H2OSTR axis

    fd, analytic = _columns(solver, inputs)

    assert torch.count_nonzero(analytic) == 0
    assert torch.count_nonzero(fd) == 0
