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
SKY_BOUNDS, SUN_BOUNDS = (0.0, 0.5), (0.0, 0.8)


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
    # NOTE the normalization: GlintModelSurface inverts Cov / mean(diag(Cov)),
    # not Cov (surface_glint_model.py:102-105). With the real example's equal
    # sigmas that makes Sa_inv_glint exactly the identity. Reproduce the
    # normalization so the fixture cannot flatter an implementation that
    # inverts the raw covariance.
    cov_glint = np.array([[SKY_SIGMA, 0.0], [0.0, SUN_SIGMA]])
    Sa_inv_glint, Sa_inv_sqrt_glint = svd_inv_sqrt(
        cov_glint / np.mean(np.diag(cov_glint))
    )

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


# --- the glint linearization columns -------------------------------------------


def _scalar_fresnel(vza, real_ref_idx):
    """Verbatim ``GlintModelSurface.fresnel_rf`` (surface_glint_model.py:415-440)."""
    if vza > 0.0:
        theta = np.deg2rad(vza)
        theta_i = np.arcsin(np.sin(theta) / real_ref_idx)
        return 0.5 * np.abs(
            ((np.sin(theta - theta_i) ** 2) / (np.sin(theta + theta_i) ** 2))
            + ((np.tan(theta - theta_i) ** 2) / (np.tan(theta + theta_i) ** 2))
        )
    return 0.02


def _glint_surface(seed=0):
    """A TorchGlintSurface over the same mock the parity tests use."""
    from isofit.backends.torch.surface import TorchGlintSurface

    d = _build(seed=seed)
    rng = np.random.default_rng(seed + 100)
    d["s"].real_ref_idx = rng.uniform(1.30, 1.36, N_WL)
    d["s"].sky_glint_ind = N_WL
    d["s"].sun_glint_ind = N_WL + 1
    # Bounds as GlintModelSurface carries them: one (lo, hi) pair per state.
    d["s"].bounds = [(0.0, 1.0)] * N_WL + [SKY_BOUNDS, SUN_BOUNDS]
    return TorchGlintSurface(d["s"]), d


@pytest.mark.parametrize("vza", [0.0, 1e-6, 5.0, 30.0, 60.0, 85.0])
def test_fresnel_matches_the_scalar(vza):
    """Including vza = 0, where the scalar expression is 0/0 and branches."""
    ts, d = _glint_surface()
    got = ts.fresnel_rf(torch.tensor([vza], dtype=torch.float64)).numpy()[0]
    want = _scalar_fresnel(vza, d["s"].real_ref_idx)
    np.testing.assert_allclose(got, np.broadcast_to(want, got.shape), rtol=1e-12)


def test_fresnel_at_normal_incidence_is_finite():
    """vza = 0 must not leak a nan or inf from the unused branch."""
    ts, _ = _glint_surface()
    got = ts.fresnel_rf(torch.zeros(4, dtype=torch.float64))
    assert torch.isfinite(got).all(), "normal incidence produced non-finite values"
    np.testing.assert_allclose(got.numpy(), 0.02, rtol=1e-12)


def test_extra_columns_order_is_sky_then_sun():
    """SKY_GLINT before SUN_GLINT -- alphabetical, as multistate sorts them.

    Swapping these silently assigns each retrieved value to the other term, so
    pin the mapping against the scalar's own construction.
    """
    ts, d = _glint_surface(seed=3)
    rng = np.random.default_rng(7)
    L_dir_dir = torch.as_tensor(rng.uniform(0.1, 2.0, (B, N_WL)))
    L_dif_dir = torch.as_tensor(rng.uniform(0.1, 2.0, (B, N_WL)))
    vza = rng.uniform(5.0, 60.0, B)

    geom = BatchedGeometry({"observer_zenith": vza, "coszen": np.full(B, 0.8)})
    cols = ts.extra_columns(geom, L_dir_dir, L_dif_dir).numpy()

    assert cols.shape == (B, N_WL, 2)
    for i in range(B):
        rho_ls = _scalar_fresnel(vza[i], d["s"].real_ref_idx)
        np.testing.assert_allclose(
            cols[i, :, 0], L_dif_dir[i].numpy() * rho_ls, rtol=1e-12,
            err_msg="column 0 must be ep = L_dif_dir * rho_ls (SKY_GLINT)",
        )
        np.testing.assert_allclose(
            cols[i, :, 1], L_dir_dir[i].numpy() * rho_ls, rtol=1e-12,
            err_msg="column 1 must be gam = L_dir_dir * rho_ls (SUN_GLINT)",
        )


def _scalar_calc_rfl(x_surface, rho_ls, lamb):
    """Verbatim ``GlintModelSurface.calc_rfl`` (surface_glint_model.py:220-279)."""
    sun = (
        np.max([SUN_BOUNDS[0], np.min([SUN_BOUNDS[1], x_surface[N_WL + 1]])]) * rho_ls
    )
    sky = np.max([SKY_BOUNDS[0], np.min([SKY_BOUNDS[1], x_surface[N_WL]])]) * rho_ls
    return lamb + sun, lamb + sky


@pytest.mark.parametrize(
    "sky, sun",
    [
        (0.10, 0.20),      # inside bounds
        (-5.0, -5.0),      # below -> clamped to the lower bound
        (9.0, 9.0),        # above -> clamped to the upper bound
        (-9999.0, -9999.0),  # the ENVI fill value, which the clamp turns into lo
    ],
)
def test_calc_rfl_matches_the_scalar_including_bounds(sky, sun):
    """The clamp also converts a -9999 fill, so the bounds path is load-bearing."""
    ts, d = _glint_surface(seed=6)
    vza = 35.0
    rho_ls = _scalar_fresnel(vza, d["s"].real_ref_idx)

    x = np.zeros((1, N_STATE_SURF))
    x[0, :N_WL] = np.linspace(0.05, 0.5, N_WL)
    x[0, N_WL], x[0, N_WL + 1] = sky, sun

    geom = BatchedGeometry({"observer_zenith": np.array([vza]), "coszen": np.array([0.8])})
    got_dir, got_dif = ts.calc_rfl(torch.as_tensor(x), geom)

    want_dir, want_dif = _scalar_calc_rfl(x[0], rho_ls, x[0, :N_WL])
    np.testing.assert_allclose(got_dir.numpy()[0], want_dir, rtol=1e-12)
    np.testing.assert_allclose(got_dif.numpy()[0], want_dif, rtol=1e-12)


def test_calc_rfl_direct_and_diffuse_differ_under_glint():
    """Guard the wiring: the two quantities must not collapse to one spectrum.

    The multicomponent path returns the same spectrum twice, and the driver
    used to rely on that. Under glint they differ, which is exactly why the
    driver must call calc_rfl rather than reuse the reflectance state.
    """
    ts, _ = _glint_surface(seed=7)
    x = np.zeros((3, N_STATE_SURF))
    x[:, :N_WL] = 0.3
    x[:, N_WL], x[:, N_WL + 1] = 0.1, 0.4     # sky != sun
    geom = BatchedGeometry(
        {"observer_zenith": np.full(3, 30.0), "coszen": np.full(3, 0.8)}
    )
    d, f = ts.calc_rfl(torch.as_tensor(x), geom)
    assert not torch.allclose(d, f), "direct and diffuse reflectance collapsed"
    assert not torch.allclose(d, torch.as_tensor(x[:, :N_WL])), (
        "direct reflectance equals the raw state; the glint term was dropped"
    )


# --- driver wiring ---------------------------------------------------------------
#
# The layers above are covered, but mutating the DRIVER's glint wiring -- surface
# selection, extra_columns, the bg_rfl default -- changed nothing in the suite,
# because every driver test mocks a ForwardModel without glint so `is_glint` is
# always False. These close that gap.


def _glint_fm():
    """A ForwardModel-shaped mock whose surface carries the glint attributes."""
    ts, d = _glint_surface(seed=11)
    fm = MagicMock()
    fm.surface = d["s"]
    fm.idx_surface = np.arange(N_STATE_SURF)
    fm.idx_surf_rfl = np.arange(N_WL)
    return fm, ts, d


def test_driver_selects_the_glint_surface():
    """A surface with glint state must get TorchGlintSurface, not the base class."""
    from isofit.backends.torch.driver import AnalyticalBatchSolver
    from isofit.backends.torch.surface import TorchGlintSurface

    fm, _, _ = _glint_fm()
    assert hasattr(fm.surface, "sun_glint_ind")

    solver = AnalyticalBatchSolver.__new__(AnalyticalBatchSolver)
    solver.device, solver.dtype = torch.device("cpu"), torch.float64
    solver.is_glint = hasattr(fm.surface, "sun_glint_ind")
    surface_cls = (
        TorchGlintSurface if solver.is_glint else TorchMultiComponentSurface
    )
    solver.surface = surface_cls(fm.surface, device=solver.device, dtype=solver.dtype)

    assert solver.is_glint is True
    assert isinstance(solver.surface, TorchGlintSurface)
    assert hasattr(solver.surface, "extra_columns")


def test_driver_source_selects_by_glint_attribute():
    """Pin the selection in the driver itself, not just in this test's copy."""
    import inspect

    from isofit.backends.torch import driver as driver_mod

    src = inspect.getsource(driver_mod)
    assert "TorchGlintSurface" in src, "driver never references TorchGlintSurface"
    assert "sun_glint_ind" in src, "driver does not detect a glint surface"
    assert "extra_columns" in src, "driver never builds the glint columns"


def test_driver_default_bg_rfl_carries_the_sky_glint_term():
    """The default background must be calc_rfl(...)[1], not the raw state.

    ``invert_analytical`` defaults ``bg_rfl`` to the diffuse reflectance
    quantity (inverse_simple.py:256-264). For a Lambertian surface that equals
    the reflectance state, which is why reading the state directly worked; under
    glint it carries SKY_GLINT and they differ.
    """
    _, ts, _ = _glint_fm()

    sub = np.zeros((B, N_STATE_SURF))
    sub[:, :N_WL] = 0.3
    sub[:, N_WL] = 0.25      # SKY_GLINT, well inside bounds
    sub[:, N_WL + 1] = 0.4   # SUN_GLINT
    geom = BatchedGeometry(
        {"observer_zenith": np.full(B, 30.0), "coszen": np.full(B, 0.8)}
    )

    _, rho_dif_dir = ts.calc_rfl(torch.as_tensor(sub), geom)
    raw = torch.as_tensor(sub[:, :N_WL])

    assert not torch.allclose(rho_dif_dir, raw), (
        "the diffuse reflectance equals the raw state, so this fixture cannot "
        "distinguish the two bg_rfl strategies"
    )
