"""Parity for glint surfaces: the batched solve vs. the real ``invert_analytical``.

``GlintModelSurface`` appends two state elements, ``SKY_GLINT`` and
``SUN_GLINT``, so the surface linearization ``H`` becomes the multicomponent
diagonal plus exactly two dense columns
(``isofit/surface/surface_glint_model.py:392-401``). ``L = H[winidx][:, iv_idx]``
is then a row-selected diagonal bordered by two dense columns, which the
diagonal fast path cannot express.

Two things make this easy to get subtly wrong, and both are pinned below.

**The triangle.** The scalar computes ``P_tilde = ((L.T @ P) @ L).T`` where ``P``
came from ``dpotri(C, 1)`` and therefore holds only its LOWER triangle. For a
purely diagonal ``L`` the full symmetric inverse gives a numerically identical
answer -- which is why the shipped reflectance-only path uses it and matches the
CPU to 5.96e-08 on a real scene. That coincidence does not survive the dense
columns: using the symmetric inverse there is wrong by ~4e-04 in the retrieved
state, which looks entirely plausible.

**The prior.** ``GlintModelSurface.Sa`` block-diags a constant 2x2 onto the
per-component inverse, and ``ForwardModel.Sa`` divides the whole thing by
``scale_surface**2 = mean(diag(Sa_surface))`` -- a mean over ``n_wl + 2``
entries, including the two glint variances. Averaging over ``n_wl`` alone
mis-weights the entire prior and fails silently.

The reference here is the real ``invert_analytical``, driven one pixel at a
time, so agreement is evidence about the LAPACK chain and not about a textbook
formula.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from scipy.linalg import block_diag

from isofit.backends.torch.analytical import invert_analytical_batch
from isofit.backends.torch.geometry import BatchedGeometry
from isofit.backends.torch.surface import TorchMultiComponentSurface
from isofit.core.common import svd_inv_sqrt
from isofit.inversion.inverse_simple import invert_analytical
from isofit.surface.surface_multicomp import MultiComponentSurface

pytestmark = pytest.mark.torch_cpu

RTOL, ATOL = 1e-7, 1e-9

N_WL, N_REF, N_COMP, N_ATM, N_GLINT = 16, 10, 3, 2, 2
N_STATE_SURF = N_WL + N_GLINT
B = 8

SKY_SIGMA, SUN_SIGMA = 0.04, 0.09
SKY_MEAN, SUN_MEAN = 0.02, 0.05


def _build(seed=0, n_win=12):
    """A glint-shaped surface plus synthetic per-pixel inputs."""
    rng = np.random.default_rng(seed)

    idx_ref = np.arange(N_REF)
    winidx = np.sort(rng.choice(N_WL, n_win, replace=False))
    component_means = rng.uniform(0.05, 0.6, (N_COMP, N_WL))
    mus = np.stack([m[idx_ref] / np.linalg.norm(m[idx_ref]) for m in component_means])

    covs, inv, inv_sqrt = [], [], []
    for _ in range(N_COMP):
        # A reasonably TIGHT prior. With the large covariances used in the
        # reflectance-only fixture the prior barely constrains anything, and
        # with the extra glint columns free the iteration walks away -- the
        # scalar reference alone reaches reflectances of ~100 by the third
        # pass. Real component covariances are informative; mimic that.
        A = rng.normal(size=(N_WL, N_WL)) * 0.05
        C = A @ A.T + 0.02 * np.eye(N_WL)
        covs.append(C)
        Ci = np.linalg.inv(C)
        inv.append(Ci)
        w, V = np.linalg.eigh(Ci)
        inv_sqrt.append(V @ np.diag(np.sqrt(w)) @ V.T)

    # The glint prior, built exactly as GlintModelSurface.__init__ does
    # (surface_glint_model.py:102-105).
    cov_glint = np.array([[SKY_SIGMA, 0.0], [0.0, SUN_SIGMA]])
    Sa_inv_glint, Sa_inv_sqrt_glint = svd_inv_sqrt(cov_glint)

    s = MagicMock()
    s.n_wl = N_WL
    s.n_comp = N_COMP
    s.n_state = N_STATE_SURF
    s.wl = np.linspace(400, 2500, N_WL)
    s.idx_ref = idx_ref
    s.idx_lamb = np.arange(N_WL)
    s.component_means = component_means
    s.component_covs = np.stack(covs)
    s.Sa_inv_normalized = np.stack(inv)
    s.Sa_inv_sqrt_normalized = np.stack(inv_sqrt)
    s.mus = mus
    s.normalize = "Euclidean"
    s.selection_metric = "Euclidean"
    s.select_on_init = False
    s.statevec_names = [f"RFL_{i}" for i in range(N_WL)] + ["SKY_GLINT", "SUN_GLINT"]
    s.analytical_iv_idx = np.arange(N_STATE_SURF)
    s.norm = lambda r: np.linalg.norm(r)
    s.calc_lamb = lambda x, geom=None: MultiComponentSurface.calc_lamb(s, x, geom)
    s.euclidean_distance = MultiComponentSurface.euclidean_distance
    s.component = lambda x, geom=None: MultiComponentSurface.component(s, x, geom)

    # Glint prior attributes the torch surface reads.
    s.Sa_inv_glint = Sa_inv_glint
    s.sky_glint_sigma, s.sun_glint_sigma = SKY_SIGMA, SUN_SIGMA
    s.sky_glint_mean, s.sun_glint_mean = SKY_MEAN, SUN_MEAN

    ts = TorchMultiComponentSurface(s)

    theta = rng.uniform(1.0, 8.0, (B, N_WL))
    # The two dense H columns: ep = L_dif_dir * rho_ls, gam = L_dir_dir * rho_ls.
    extra = rng.uniform(0.05, 0.9, (B, N_WL, N_GLINT))
    x0 = np.concatenate(
        [
            rng.uniform(0.05, 0.6, (B, N_WL)),
            rng.uniform(0.01, 0.2, (B, N_GLINT)),
            rng.uniform(0.1, 2.0, (B, N_ATM)),
        ],
        axis=1,
    )
    L_atm = rng.uniform(0.1, 1.0, (B, N_WL))
    L_bg = rng.uniform(0.0, 0.3, (B, N_WL))
    eof = np.zeros((B, N_WL))

    # Synthesize the measurement THROUGH the linearization rather than drawing
    # it independently. With an unrelated meas the system is ill-posed: the
    # scalar reference itself runs away to reflectances of 6e5 by the second
    # iteration, so a parity test on it would only be comparing two divergences.
    # Building meas = L_atm + L_bg + eof + H @ x_true keeps both paths in a
    # physical regime and makes multi-iteration agreement meaningful.
    x_true = np.concatenate(
        [rng.uniform(0.05, 0.6, (B, N_WL)), rng.uniform(0.01, 0.2, (B, N_GLINT))],
        axis=1,
    )
    meas = L_atm + L_bg + eof
    for i in range(B):
        H_i = np.zeros((N_WL, N_STATE_SURF))
        H_i[np.arange(N_WL), np.arange(N_WL)] = theta[i]
        H_i[:, N_WL:] = extra[i]
        meas[i] += H_i @ x_true[i]

    Seps = np.stack(
        [
            (lambda A: A @ A.T + n_win * np.eye(n_win))(rng.normal(size=(n_win, n_win)))
            for _ in range(B)
        ]
    )

    return dict(
        s=s, ts=ts, winidx=winidx, theta=theta, extra=extra, x0=x0, meas=meas,
        L_atm=L_atm, L_bg=L_bg, eof=eof, Seps=Seps,
        idx_surface=np.arange(N_STATE_SURF),
        n_state=N_STATE_SURF + N_ATM,
        Sa_inv_glint=Sa_inv_glint,
    )


def _embed(sub, winidx, n):
    full = np.eye(n)
    full[np.ix_(winidx, winidx)] = sub
    return full


def _scalar_reference(d, num_iter, outside_ret_const):
    """The real ``invert_analytical``, one pixel at a time, glint-shaped."""
    s, winidx = d["s"], d["winidx"]
    trajectories, uncertainties = [], []

    for i in range(B):
        # H is the diagonal plus the two dense columns, exactly the shape
        # GlintModelSurface.analytical_model returns.
        H = np.zeros((N_WL, N_STATE_SURF))
        H[np.arange(N_WL), np.arange(N_WL)] = d["theta"][i]
        H[:, N_WL:] = d["extra"][i]

        fm = MagicMock()
        fm.surface = s
        fm.idx_surface = d["idx_surface"]
        fm.idx_surf_nonrfl = np.arange(N_WL, N_STATE_SURF)
        fm.unpack = lambda x: (x[:N_STATE_SURF].copy(), x[N_STATE_SURF:], np.array([]))
        fm.calc_rfl = MagicMock(return_value=(np.zeros(N_WL), np.zeros(N_WL)))
        fm.upsample = lambda wl, q: q
        fm.calc_atmosphere_quantities = MagicMock(
            return_value=(
                {"sphalb": np.zeros(N_WL)},
                *(np.zeros(N_WL) for _ in range(5)),
            )
        )
        fm.atmosphere.get_L_atm = MagicMock(return_value=d["L_atm"][i])
        fm.calc_rdn_bgrfl = MagicMock(return_value=d["L_bg"][i])
        fm.eof_offset = MagicMock(return_value=d["eof"][i])
        fm.surface.analytical_model = MagicMock(return_value=H)
        fm.Seps = MagicMock(return_value=_embed(d["Seps"][i], winidx, N_WL))

        def _fm_Sa(x, geom, _s=s):
            # GlintModelSurface.Sa (surface_glint_model.py:142-159) followed by
            # ForwardModel's scale_surface**2 division (forward.py:259-270).
            #
            # MultiComponentSurface.Sa ALREADY embeds the covariance to the full
            # statevec width when statevec_names is longer than idx_lamb
            # (surface_multicomp.py:212-221), so Sa_unnormalized arrives
            # (n_wl+2, n_wl+2) with zeros in the glint corner. Embedding it a
            # second time here would average scale_surface**2 over four extra
            # zeros and mis-weight the whole prior.
            Sa_unnorm, Sa_inv_norm, _ = MultiComponentSurface.Sa(
                _s, x[:N_STATE_SURF], geom
            )
            assert Sa_unnorm.shape == (N_STATE_SURF, N_STATE_SURF), Sa_unnorm.shape
            Sa_unnorm = Sa_unnorm.copy()
            Sa_unnorm[N_WL, N_WL] = SKY_SIGMA
            Sa_unnorm[N_WL + 1, N_WL + 1] = SUN_SIGMA
            Sa_inv_norm = block_diag(Sa_inv_norm, d["Sa_inv_glint"])
            scale_sq = np.mean(np.diag(Sa_unnorm))
            return (Sa_unnorm, Sa_inv_norm / scale_sq, None)

        fm.Sa = _fm_Sa

        def _xa(x, geom, _s=s):
            mu = np.zeros(N_STATE_SURF)
            mu[:N_WL] = MultiComponentSurface.xa(_s, x[:N_STATE_SURF], geom)[:N_WL]
            mu[N_WL], mu[N_WL + 1] = SKY_MEAN, SUN_MEAN
            return np.concatenate([mu, np.zeros(N_ATM)])

        fm.xa = _xa

        geom = MagicMock(spec=["bg_rfl"])
        geom.bg_rfl = None

        traj, unc = invert_analytical(
            fm, winidx, d["meas"][i], geom,
            d["x0"][i].copy(), d["x0"][i].copy(),
            num_iter=num_iter, outside_ret_const=outside_ret_const,
        )
        trajectories.append(traj)
        uncertainties.append(unc)

    return np.stack(trajectories), np.stack(uncertainties)


def _batched(d, num_iter, outside_ret_const, extra_columns=True):
    outside = np.setdiff1d(np.arange(N_WL), d["winidx"])
    return invert_analytical_batch(
        d["ts"],
        torch.as_tensor(d["winidx"], dtype=torch.int64),
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
        extra_columns=torch.as_tensor(d["extra"]) if extra_columns else None,
    )


@pytest.mark.parametrize("num_iter", [1, 3])
def test_glint_state_matches_invert_analytical(num_iter):
    """Full surface state, reflectance and both glint terms."""
    d = _build(seed=1)
    ref_traj, _ = _scalar_reference(d, num_iter, -0.01)
    traj, _ = _batched(d, num_iter, -0.01)

    got = traj.numpy()
    assert got.shape == ref_traj.shape
    np.testing.assert_allclose(got, ref_traj, rtol=RTOL, atol=ATOL)


def test_the_glint_elements_themselves_are_solved():
    """Guard against passing by leaving the glint entries at their input value."""
    d = _build(seed=2)
    ref_traj, _ = _scalar_reference(d, 1, -0.01)
    traj, _ = _batched(d, 1, -0.01)

    x0_glint = d["x0"][:, N_WL:N_STATE_SURF]
    ref_glint = ref_traj[:, -1, N_WL:N_STATE_SURF]
    got_glint = traj.numpy()[:, -1, N_WL:N_STATE_SURF]

    assert not np.allclose(ref_glint, x0_glint), (
        "the reference did not move the glint terms; the fixture is degenerate"
    )
    np.testing.assert_allclose(got_glint, ref_glint, rtol=RTOL, atol=ATOL)


def test_uncertainty_matches():
    d = _build(seed=3)
    _, ref_unc = _scalar_reference(d, 1, -0.01)
    _, unc = _batched(d, 1, -0.01)
    np.testing.assert_allclose(unc.numpy(), ref_unc, rtol=1e-6, atol=1e-9)


def test_symmetric_inverse_for_the_dense_columns_would_diverge():
    """Pin the triangle choice: the symmetric inverse is not equivalent here.

    If this ever stops detecting a difference, the two forms have become
    interchangeable and the comment in ``analytical.py`` about ``tril`` is
    obsolete -- but until then, that comment is load-bearing.
    """
    from isofit.backends.torch.linalg import chol_inv_full

    d = _build(seed=4)
    Seps = torch.as_tensor(d["Seps"])
    P_sym = chol_inv_full(Seps)
    P_low = torch.tril(P_sym)
    winidx = torch.as_tensor(d["winidx"], dtype=torch.int64)
    G = torch.as_tensor(d["extra"]).index_select(1, winidx)

    sym = (G.transpose(-1, -2) @ P_sym @ G).transpose(-1, -2)
    low = (G.transpose(-1, -2) @ P_low @ G).transpose(-1, -2)

    assert not torch.allclose(sym, low), (
        "symmetric and lower-triangle forms agree; the triangle distinction "
        "this implementation depends on has disappeared"
    )


def test_prior_scale_includes_the_glint_variances():
    """scale_surface**2 averages over n_wl + n_glint, not n_wl."""
    d = _build(seed=5)
    ts = d["ts"]
    x_surface = torch.as_tensor(d["x0"][:, :N_STATE_SURF])
    ci = ts.component(x_surface, BatchedGeometry({"coszen": np.full(B, 0.8)}))

    got = ts.prior_scale(x_surface, ci).numpy()

    lamb_ref = x_surface[:, ts.idx_ref].numpy()
    norm_sq = np.linalg.norm(lamb_ref, axis=1) ** 2
    mean_diag = np.stack(
        [np.diag(d["s"].component_covs[c]).mean() for c in ci.numpy()]
    )
    want = (mean_diag * norm_sq * N_WL + SKY_SIGMA + SUN_SIGMA) / N_STATE_SURF

    np.testing.assert_allclose(got, want, rtol=1e-12)
    assert not np.allclose(got, mean_diag * norm_sq), (
        "prior_scale ignored the glint variances"
    )
