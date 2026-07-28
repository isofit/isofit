"""Parity tests for the batched multicomponent surface prior.

Compared against :class:`isofit.surface.surface_multicomp.MultiComponentSurface`
methods invoked unbound against a mock, so the selection and prior arithmetic is
exercised without the ``.mat`` surface model files a real instance would load.

The component index is a *discrete* choice, so these tests assert exact index
agreement rather than a numerical tolerance: picking a neighbouring component
swaps in an entirely different prior mean and covariance.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from isofit.backends.torch.geometry import BatchedGeometry
from isofit.backends.torch.surface import TorchMultiComponentSurface
from isofit.surface.surface_multicomp import MultiComponentSurface

pytestmark = pytest.mark.torch_cpu

RTOL = 1e-11
N_WL = 20
N_REF = 12
N_COMP = 5
B = 24


def _surface(normalize="Euclidean", metric="Euclidean", select_on_init=False, seed=0):
    """A MultiComponentSurface-shaped mock plus its batched counterpart."""
    rng = np.random.default_rng(seed)

    idx_ref = np.arange(N_REF)
    idx_lamb = np.arange(N_WL)
    wl = np.linspace(400, 2500, N_WL)

    component_means = rng.uniform(0.05, 0.6, (N_COMP, N_WL))
    mus = np.stack(
        [m[idx_ref] / np.linalg.norm(m[idx_ref]) for m in component_means]
    )
    covs = []
    inv = []
    inv_sqrt = []
    for i in range(N_COMP):
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
    s.normalize = normalize
    s.selection_metric = metric
    s.select_on_init = select_on_init
    s.statevec_names = [f"RFL_{i}" for i in range(N_WL)]

    if normalize == "Euclidean":
        s.norm = lambda r: np.linalg.norm(r)
    elif normalize == "RMS":
        s.norm = lambda r: np.sqrt(np.mean(np.power(r, 2)))
    else:
        s.norm = lambda r: 1.0

    # Bind the real collaborator methods. MultiComponentSurface.xa and
    # .component call back into calc_lamb, the distance metrics and component
    # selection; left as MagicMocks those return mocks and the "reference"
    # silently degrades to zeros instead of failing.
    s.calc_lamb = lambda x_surface, geom=None: MultiComponentSurface.calc_lamb(
        s, x_surface, geom
    )
    s.euclidean_distance = MultiComponentSurface.euclidean_distance
    s.spectral_angle_distance = MultiComponentSurface.spectral_angle_distance
    s.spectral_gradient_angle = (
        lambda lamb_ref, mus: MultiComponentSurface.spectral_gradient_angle(
            s, lamb_ref, mus
        )
    )
    s.component = lambda x_surface, geom=None: MultiComponentSurface.component(
        s, x_surface, geom
    )

    return s, TorchMultiComponentSurface(s)


def _states(seed=1):
    return np.random.default_rng(seed).uniform(0.02, 0.7, (B, N_WL))


# --- normalization ----------------------------------------------------------------


@pytest.mark.parametrize("normalize", ["Euclidean", "RMS", "None"])
def test_norm_matches_scalar(normalize):
    s, ts = _surface(normalize=normalize)
    x = _states()
    lamb_ref = x[:, s.idx_ref]

    got = ts.norm(torch.as_tensor(lamb_ref)).numpy()
    ref = np.array([s.norm(lamb_ref[i]) for i in range(B)])
    np.testing.assert_allclose(got, ref, rtol=RTOL)


# --- component selection ----------------------------------------------------------


@pytest.mark.parametrize("metric", ["Euclidean", "SpecAngle", "SGA"])
def test_component_selection_matches_scalar(metric):
    """The chosen index must match exactly; a near-miss picks a different prior."""
    s, ts = _surface(metric=metric)
    x = _states(2)

    got = ts.component(torch.as_tensor(x)).numpy()
    ref = np.array(
        [
            MultiComponentSurface.component(s, x[i], MagicMock(spec=[]))
            for i in range(B)
        ]
    )
    np.testing.assert_array_equal(got, ref)


def test_single_component_surface_returns_zero():
    s, ts = _surface()
    ts.n_comp = 1
    got = ts.component(torch.as_tensor(_states(3)))
    assert torch.all(got == 0)


def test_frozen_component_is_respected():
    """A cached selection must win over recomputation (surf_cmp_init)."""
    s, ts = _surface()
    x = _states(4)
    geom = BatchedGeometry({"coszen": np.full(B, 0.8)})
    frozen = torch.arange(B, dtype=torch.int64) % N_COMP
    geom.surf_cmp_init = frozen

    got = ts.component(torch.as_tensor(x), geom)
    torch.testing.assert_close(got, frozen)


def test_select_on_init_uses_initial_state_and_caches():
    """With select_on_init, selection follows x_surf_init and is then frozen."""
    s, ts = _surface(select_on_init=True)
    x_now = _states(5)
    x_init = _states(6)

    geom = BatchedGeometry({"coszen": np.full(B, 0.8)})
    geom.x_surf_init = torch.as_tensor(x_init)

    got = ts.component(torch.as_tensor(x_now), geom)

    # matches selection from the *initial* state, not the current one
    expected = ts.component(torch.as_tensor(x_init))
    torch.testing.assert_close(got, expected)

    # and the result was cached onto the geometry
    assert geom.surf_cmp_init is not None
    torch.testing.assert_close(geom.surf_cmp_init, expected)


def test_selection_is_stable_under_uniform_scaling():
    """Selection uses normalized reflectance, so brightness must not change it."""
    s, ts = _surface()
    x = _states(7)
    a = ts.component(torch.as_tensor(x))
    b = ts.component(torch.as_tensor(x * 3.0))
    torch.testing.assert_close(a, b)


def test_unknown_metric_rejected():
    s, ts = _surface()
    ts.selection_metric = "Manhattan"
    with pytest.raises(ValueError, match="not valid"):
        ts.component(torch.as_tensor(_states(8)))


# --- prior mean -------------------------------------------------------------------


def test_xa_matches_scalar():
    s, ts = _surface()
    x = _states(9)

    got = ts.xa(torch.as_tensor(x)).numpy()
    ref = np.stack(
        [MultiComponentSurface.xa(s, x[i], MagicMock(spec=[])) for i in range(B)]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


def test_xa_scales_with_brightness():
    """The prior mean is un-normalized by the pixel's own norm."""
    s, ts = _surface()
    x = _states(10)
    base = ts.xa(torch.as_tensor(x)).numpy()
    scaled = ts.xa(torch.as_tensor(x * 2.0)).numpy()
    np.testing.assert_allclose(scaled, base * 2.0, rtol=1e-10)


# --- prior precision --------------------------------------------------------------


def test_add_Sa_inv_gathers_the_right_component():
    s, ts = _surface()
    ci = torch.arange(B, dtype=torch.int64) % N_COMP
    target = torch.zeros((B, N_WL, N_WL), dtype=torch.float64)

    ts.add_Sa_inv(target, ci)

    for i in range(B):
        np.testing.assert_allclose(
            target[i].numpy(), s.Sa_inv_normalized[int(ci[i])], rtol=RTOL
        )


def test_add_Sa_inv_applies_scale():
    s, ts = _surface()
    ci = torch.zeros(B, dtype=torch.int64)
    scale = torch.linspace(1.0, 4.0, B, dtype=torch.float64)
    target = torch.zeros((B, N_WL, N_WL), dtype=torch.float64)

    ts.add_Sa_inv(target, ci, scale=scale)

    for i in range(B):
        np.testing.assert_allclose(
            target[i].numpy(),
            s.Sa_inv_normalized[0] / float(scale[i]),
            rtol=RTOL,
        )


def test_add_Sa_inv_accumulates_into_existing_matrix():
    """It must add, not overwrite: the caller has already built the data term."""
    s, ts = _surface()
    ci = torch.zeros(B, dtype=torch.int64)
    existing = torch.ones((B, N_WL, N_WL), dtype=torch.float64)
    out = ts.add_Sa_inv(existing.clone(), ci)
    np.testing.assert_allclose(
        out[0].numpy(), 1.0 + s.Sa_inv_normalized[0], rtol=RTOL
    )


def test_prior_scale_matches_forward_model_scale_surface():
    """prior_scale must equal ForwardModel.Sa's scale_surface**2.

    That is ``mean(diag(Sa_surface))`` where ``Sa_surface`` is the surface's
    *un-normalized* covariance ``Cov * norm(lamb_ref)**2``
    (isofit/core/forward.py:259-270). An earlier version of this used only the
    norm**2 factor, which silently mis-weighted the prior by the component's
    mean variance; the analytical-line parity test is what caught it.
    """
    s, ts = _surface()
    x = _states(11)
    ci = ts.component(torch.as_tensor(x))
    got = ts.prior_scale(torch.as_tensor(x), ci).numpy()

    ref = []
    for i in range(B):
        Sa_surface, _, _ = MultiComponentSurface.Sa(s, x[i], MagicMock(spec=[]))
        ref.append(np.mean(np.diag(Sa_surface)))
    np.testing.assert_allclose(got, np.array(ref), rtol=RTOL)


def test_prior_scale_includes_component_variance():
    """Guard the specific regression: norm**2 alone is not enough."""
    s, ts = _surface()
    x = _states(12)
    ci = ts.component(torch.as_tensor(x))
    got = ts.prior_scale(torch.as_tensor(x), ci).numpy()
    norm_only = np.array([s.norm(x[i][s.idx_ref]) ** 2 for i in range(B)])
    assert not np.allclose(got, norm_only), (
        "prior_scale must carry the component's mean variance, not just norm**2"
    )


# --- linearization ----------------------------------------------------------------


def test_evaluate_theta_homogeneous_matches_scalar():
    s, ts = _surface()
    rng = np.random.default_rng(12)
    L_tot = rng.uniform(1.0, 10.0, (B, N_WL))
    s_alb = rng.uniform(0.0, 0.3, (B, N_WL))
    bg = rng.uniform(0.05, 0.5, (B, N_WL))

    geom = BatchedGeometry({"coszen": np.full(B, 0.8)}, bg_rfl=bg)
    got = ts.evaluate_theta(
        torch.as_tensor(s_alb), geom, torch.as_tensor(L_tot)
    ).numpy()

    ref = np.stack(
        [
            MultiComponentSurface.evaluate_theta_homogeneous_bgrfl(
                s,
                s_alb[i],
                _geom_with_bg(bg[i]),
                L_tot[i],
                None,
                None,
                None,
                None,
            )
            for i in range(B)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


def _geom_with_bg(bg):
    g = MagicMock()
    g.bg_rfl = bg
    return g


def test_evaluate_theta_heterogeneous_matches_scalar():
    s, ts = _surface()
    rng = np.random.default_rng(13)
    L_dir_dir = rng.uniform(0.5, 5.0, (B, N_WL))
    L_dif_dir = rng.uniform(0.5, 5.0, (B, N_WL))

    geom = BatchedGeometry({"coszen": np.full(B, 0.8)})
    got = ts.evaluate_theta(
        None,
        geom,
        None,
        L_dir_dir=torch.as_tensor(L_dir_dir),
        L_dif_dir=torch.as_tensor(L_dif_dir),
        heterogeneous=True,
    ).numpy()

    ref = np.stack(
        [
            MultiComponentSurface.evaluate_theta_heterogeneous_bgrfl(
                s, None, MagicMock(), None, L_dir_dir[i], None, L_dif_dir[i], None
            )
            for i in range(B)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


def test_evaluate_theta_without_background_is_L_tot():
    """No background reflectance: theta collapses to L_tot."""
    s, ts = _surface()
    L_tot = np.random.default_rng(14).uniform(1.0, 10.0, (B, N_WL))
    geom = BatchedGeometry({"coszen": np.full(B, 0.8)})
    got = ts.evaluate_theta(None, geom, torch.as_tensor(L_tot)).numpy()
    np.testing.assert_allclose(got, L_tot, rtol=RTOL)
