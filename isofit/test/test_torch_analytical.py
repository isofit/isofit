"""Parity test for the batched analytical (conditional MAP) retrieval.

This is the test that matters most for the analytical-line GPU path: it drives
the real :func:`isofit.inversion.inverse_simple.invert_analytical` one pixel at a
time and requires the batched implementation to agree.

The scalar function reaches its answer through raw LAPACK/BLAS calls whose
triangle conventions change the arithmetic (see
``isofit.backends.torch.linalg.whiten_innovation``), so agreement here is
evidence the batched path reproduces those conventions and not merely the
textbook formula.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from isofit.backends.torch.analytical import invert_analytical_batch
from isofit.backends.torch.geometry import BatchedGeometry
from isofit.backends.torch.surface import TorchMultiComponentSurface
from isofit.inversion.inverse_simple import invert_analytical
from isofit.surface.surface_multicomp import MultiComponentSurface

pytestmark = pytest.mark.torch_cpu

RTOL = 1e-7
ATOL = 1e-9

N_WL = 16
N_REF = 10
N_COMP = 3
N_ATM = 2
B = 10


def _build(seed=0, n_win=12):
    """Construct a matched scalar/batched surface plus synthetic inputs."""
    rng = np.random.default_rng(seed)

    n_state = N_WL + N_ATM
    idx_lamb = np.arange(N_WL)
    idx_ref = np.arange(N_REF)
    idx_surface = np.arange(N_WL)
    winidx = np.sort(rng.choice(N_WL, n_win, replace=False))

    wl = np.linspace(400, 2500, N_WL)
    component_means = rng.uniform(0.05, 0.6, (N_COMP, N_WL))
    mus = np.stack([m[idx_ref] / np.linalg.norm(m[idx_ref]) for m in component_means])

    covs, inv, inv_sqrt = [], [], []
    for _ in range(N_COMP):
        A = rng.normal(size=(N_WL, N_WL))
        C = A @ A.T + N_WL * np.eye(N_WL)
        covs.append(C)
        Ci = np.linalg.inv(C)
        inv.append(Ci)
        w, V = np.linalg.eigh(Ci)
        inv_sqrt.append(V @ np.diag(np.sqrt(w)) @ V.T)

    s = MagicMock()
    s.n_wl = N_WL
    s.n_comp = N_COMP
    s.n_state = N_WL
    s.wl = wl
    s.idx_ref = idx_ref
    s.idx_lamb = idx_lamb
    s.component_means = component_means
    s.component_covs = np.stack(covs)
    s.Sa_inv_normalized = np.stack(inv)
    s.Sa_inv_sqrt_normalized = np.stack(inv_sqrt)
    s.mus = mus
    s.normalize = "Euclidean"
    s.selection_metric = "Euclidean"
    s.select_on_init = False
    s.statevec_names = [f"RFL_{i}" for i in range(N_WL)]
    s.analytical_iv_idx = np.arange(N_WL)
    s.norm = lambda r: np.linalg.norm(r)
    s.calc_lamb = lambda x, geom=None: MultiComponentSurface.calc_lamb(s, x, geom)
    s.euclidean_distance = MultiComponentSurface.euclidean_distance
    s.component = lambda x, geom=None: MultiComponentSurface.component(s, x, geom)
    s.xa = lambda x, geom=None: MultiComponentSurface.xa(s, x, geom)
    s.Sa = lambda x, geom=None: MultiComponentSurface.Sa(s, x, geom)

    ts = TorchMultiComponentSurface(s)

    # Synthetic per-pixel inputs
    theta = rng.uniform(1.0, 8.0, (B, N_WL))
    x0 = np.concatenate(
        [rng.uniform(0.05, 0.6, (B, N_WL)), rng.uniform(0.1, 2.0, (B, N_ATM))], axis=1
    )
    meas = rng.uniform(1.0, 12.0, (B, N_WL))
    L_atm = rng.uniform(0.1, 1.0, (B, N_WL))
    L_bg = rng.uniform(0.0, 0.3, (B, N_WL))
    eof = np.zeros((B, N_WL))

    Seps = []
    for i in range(B):
        A = rng.normal(size=(n_win, n_win))
        Seps.append(A @ A.T + n_win * np.eye(n_win))
    Seps = np.stack(Seps)

    return dict(
        s=s, ts=ts, winidx=winidx, theta=theta, x0=x0, meas=meas,
        L_atm=L_atm, L_bg=L_bg, eof=eof, Seps=Seps,
        idx_surface=idx_surface, n_state=n_state,
    )


def _scalar_reference(d, num_iter, outside_ret_const):
    """Run invert_analytical once per pixel with matching inputs."""
    s, winidx = d["s"], d["winidx"]
    n_state = d["n_state"]

    trajectories, uncertainties = [], []
    for i in range(B):
        fm = MagicMock()
        fm.surface = s
        fm.idx_surface = d["idx_surface"]
        fm.idx_surf_nonrfl = np.array([], dtype=int)
        fm.surface.analytical_iv_idx = np.arange(N_WL)

        fm.unpack = lambda x: (x[:N_WL].copy(), x[N_WL:], np.array([]))
        fm.calc_rfl = MagicMock(return_value=(np.zeros(N_WL), np.zeros(N_WL)))
        fm.upsample = lambda wl, q: q
        fm.calc_atmosphere_quantities = MagicMock(
            return_value=(
                {"sphalb": np.zeros(N_WL)},
                np.zeros(N_WL), np.zeros(N_WL), np.zeros(N_WL),
                np.zeros(N_WL), np.zeros(N_WL),
            )
        )
        fm.atmosphere.get_L_atm = MagicMock(return_value=d["L_atm"][i])
        fm.calc_rdn_bgrfl = MagicMock(return_value=d["L_bg"][i])
        fm.eof_offset = MagicMock(return_value=d["eof"][i])
        fm.surface.analytical_model = MagicMock(
            return_value=d["theta"][i][:, None] * np.eye(N_WL)
        )
        fm.Seps = MagicMock(
            return_value=_embed(d["Seps"][i], winidx, N_WL)
        )
        # Mirror ForwardModel.Sa, not MultiComponentSurface.Sa: the forward
        # model reconciles the surface's un-normalized covariance against its
        # normalized inverse by dividing through by scale_surface**2
        # (isofit/core/forward.py:259-270). invert_analytical calls fm.Sa, so
        # omitting that step would compare against a prior ISOFIT never uses.
        def _fm_Sa(x, geom, _s=s):
            Sa_surface, Sa_inv_norm, Sa_inv_sqrt_norm = MultiComponentSurface.Sa(
                _s, x[:N_WL], geom
            )
            scale_sq = np.mean(np.diag(Sa_surface))
            return (
                Sa_surface,
                Sa_inv_norm / scale_sq,
                Sa_inv_sqrt_norm / np.sqrt(scale_sq),
            )

        fm.Sa = _fm_Sa
        fm.xa = lambda x, geom: np.concatenate(
            [MultiComponentSurface.xa(s, x[:N_WL], geom), np.zeros(N_ATM)]
        )

        geom = MagicMock(spec=["bg_rfl"])
        geom.bg_rfl = None

        traj, unc = invert_analytical(
            fm,
            winidx,
            d["meas"][i],
            geom,
            d["x0"][i].copy(),
            d["x0"][i].copy(),
            num_iter=num_iter,
            outside_ret_const=outside_ret_const,
        )
        trajectories.append(traj)
        uncertainties.append(unc)

    return np.stack(trajectories), np.stack(uncertainties)


def _embed(sub, winidx, n):
    """Place a windowed covariance back into a full (n, n) matrix."""
    full = np.eye(n)
    full[np.ix_(winidx, winidx)] = sub
    return full


def _batched(d, num_iter, outside_ret_const, strict_parity=True):
    winidx_t = torch.as_tensor(d["winidx"], dtype=torch.int64)
    outside = np.setdiff1d(np.arange(N_WL), d["winidx"])

    return invert_analytical_batch(
        d["ts"],
        winidx_t,
        torch.as_tensor(d["meas"]),
        torch.as_tensor(d["x0"]),
        torch.as_tensor(d["theta"]),
        torch.as_tensor(d["Seps"]),
        torch.as_tensor(d["L_atm"]),
        torch.as_tensor(d["L_bg"]),
        torch.as_tensor(d["eof"]),
        geom=BatchedGeometry({"coszen": np.full(B, 0.8)}),
        idx_surface=torch.as_tensor(d["idx_surface"], dtype=torch.int64),
        outside_ret_windows=torch.as_tensor(outside, dtype=torch.int64),
        num_iter=num_iter,
        outside_ret_const=outside_ret_const,
        strict_parity=strict_parity,
    )


# --- parity ----------------------------------------------------------------------


@pytest.mark.parametrize("num_iter", [1, 3])
def test_matches_scalar_invert_analytical(num_iter):
    """The whole batched update must reproduce the scalar LAPACK chain."""
    d = _build()
    ref_traj, ref_unc = _scalar_reference(d, num_iter, outside_ret_const=-0.01)
    got_traj, got_unc = _batched(d, num_iter, outside_ret_const=-0.01)

    np.testing.assert_allclose(
        got_traj.numpy(), ref_traj, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(got_unc.numpy(), ref_unc, rtol=RTOL, atol=ATOL)


def test_outside_windows_filled_with_prior_when_const_is_none():
    d = _build(seed=1)
    ref_traj, _ = _scalar_reference(d, 1, outside_ret_const=None)
    got_traj, _ = _batched(d, 1, outside_ret_const=None)
    np.testing.assert_allclose(got_traj.numpy(), ref_traj, rtol=RTOL, atol=ATOL)


def test_retrieved_reflectance_only_inside_windows_is_solved():
    """Channels outside the retrieval windows take the fill value, not a solve."""
    d = _build(seed=2)
    got_traj, _ = _batched(d, 1, outside_ret_const=-0.01)
    outside = np.setdiff1d(np.arange(N_WL), d["winidx"])
    np.testing.assert_allclose(
        got_traj.numpy()[:, -1, outside], -0.01, rtol=0, atol=0
    )


def test_uncertainty_is_positive_and_finite():
    d = _build(seed=3)
    _, unc = _batched(d, 1, outside_ret_const=-0.01)
    assert torch.all(torch.isfinite(unc))
    assert torch.all(unc > 0)


def test_trajectory_records_each_iteration():
    d = _build(seed=4)
    traj, _ = _batched(d, 3, outside_ret_const=-0.01)
    assert traj.shape == (B, 4, d["n_state"])
    # the first row is the initial state, untouched
    np.testing.assert_allclose(traj.numpy()[:, 0, :], d["x0"], rtol=0, atol=0)


def test_atmosphere_block_is_left_untouched():
    """Only the surface block is solved; the fixed atmosphere must pass through."""
    d = _build(seed=5)
    traj, _ = _batched(d, 1, outside_ret_const=-0.01)
    np.testing.assert_allclose(
        traj.numpy()[:, -1, N_WL:], d["x0"][:, N_WL:], rtol=0, atol=0
    )


# --- the dsymv parity switch -------------------------------------------------------


def test_strict_parity_differs_from_full_matrix_solution():
    """The two whitening modes must produce different retrievals.

    If they agreed, the strict-parity flag would be meaningless and a future
    change could silently alter results relative to ISOFIT's CPU path.
    """
    d = _build(seed=6)
    strict, _ = _batched(d, 1, -0.01, strict_parity=True)
    full, _ = _batched(d, 1, -0.01, strict_parity=False)
    assert not np.allclose(strict.numpy(), full.numpy())


def test_strict_parity_is_the_default_and_matches_cpu():
    """Default construction must track ISOFIT, not the corrected form."""
    d = _build(seed=7)
    ref_traj, _ = _scalar_reference(d, 1, outside_ret_const=-0.01)

    default, _ = _batched(d, 1, -0.01)
    np.testing.assert_allclose(default.numpy(), ref_traj, rtol=RTOL, atol=ATOL)

    corrected, _ = _batched(d, 1, -0.01, strict_parity=False)
    assert not np.allclose(corrected.numpy(), ref_traj, rtol=RTOL, atol=ATOL)


# --- guardrails --------------------------------------------------------------------


def test_extra_surface_states_without_columns_are_rejected():
    """Extra non-reflectance state needs its dense L columns supplied.

    Refusing is still the right behavior when a caller declares extra surface
    state but does not hand over the columns that describe it -- solving
    without them would silently treat those states as unobserved.
    """
    d = _build(seed=8)
    d["ts"].n_state = N_WL + 2  # pretend there are extra non-rfl states
    with pytest.raises(NotImplementedError, match="non-reflectance element"):
        _batched(d, 1, -0.01)
