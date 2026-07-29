"""Glint parity against a real ``GlintModelSurface``, not a mock.

Everything in ``test_torch_glint.py`` drives a ``MagicMock`` surface. That
covers the numerics well, but it left the driver's glint path unexecuted:
mutating the surface selection or the ``extra_columns`` call changed nothing in
the whole suite, because every other driver test mocks a ``ForwardModel``
without glint so ``is_glint`` was always False.

This closes that gap using ``examples/20231110_Prism_Multisurface``, the only
glint fixture in the tree. It is a genuine multistate config -- surface class 1
is ``glint_model_surface``, class 0 is ``multicomponent_surface`` -- built from
real PRISM radiance, and it exercises the full chain: a real ``ForwardModel``,
the real ``GlintModelSurface`` with its own priors and bounds, real geometry
with per-pixel view angles, ``AnalyticalBatchSolver``, and the real
``invert_analytical`` as the reference.

Marked ``examples`` because it needs downloaded data, matching how the rest of
the example-driven tests are gated. It does not run in CI.

Setup, if the fixture is missing::

    $ isofit download examples
    $ isofit download data
    # then build configs from templates and run surface_model()

The scene is two pixels, so this says nothing about performance.

WHAT THIS TEST CANNOT SEE. ``Seps`` is strictly diagonal in this configuration
-- ``H2O_ABSCO`` has magnitude 0.0 and no model discrepancy is set, so
``Sy + Kb Sb Kb^T`` has no off-diagonal structure (verified: max off-diagonal
exactly 0.0). A diagonal ``Seps`` has a diagonal inverse, so ``tril(P) == P``
and the triangle distinction the bordered blocks depend on is *unobservable*
here: swapping ``torch.tril(P_sym)`` for ``P_sym`` leaves this test's output
bit-identical. That distinction is covered by the synthetic fixture in
``test_torch_glint.py``, which uses a dense ``Seps``. The two files are
complementary rather than redundant -- this one exercises the real integration,
that one exercises numerics this scene cannot reach.
"""

import os
import warnings
from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isofit.data import env  # noqa: E402

pytestmark = pytest.mark.examples

EXAMPLE = "20231110_Prism_Multisurface"
FLIGHT = "prm20231110t071521"
CONFIG = f"configs/{FLIGHT}_multi_surface_isofit.json"

# fp64 through two Cholesky stages plus a bordered solve, and nothing else.
# These were two to five orders looser until the batched fresnel_rf was made to
# reproduce the scalar's float32 deg2rad/sin chain: that single precision
# mismatch, not the bordered blocks or the prior, dominated the entire glint
# parity budget. Measured after the fix: reflectance 2.0e-13, SKY_GLINT 8.4e-13,
# SUN_GLINT 1.3e-12 against reference values reaching |35|. Kept tight so a
# regression there cannot hide again.
RFL_ATOL = 1e-11
GLINT_RTOL = 1e-10


def _example_dir():
    try:
        path = env.path("examples", EXAMPLE)
    except Exception as exc:  # pragma: no cover - depends on local install
        pytest.skip(f"example path unavailable: {exc}")
    if not os.path.isdir(path):
        pytest.skip(f"{EXAMPLE} not downloaded; run `isofit download examples`")
    if not os.path.isfile(os.path.join(path, CONFIG)):
        pytest.skip(f"{EXAMPLE} not built; {CONFIG} is missing")
    return path


@pytest.fixture(scope="module")
def glint_setup():
    """A real glint ForwardModel, its solver, and the scene's pixels."""
    from spectral.io import envi

    from isofit.configs import configs
    from isofit.core.common import envi_header
    from isofit.core.common import eps as EPS
    from isofit.core.forward import ForwardModel
    from isofit.core.geometry import Geometry
    from isofit.core.multistate import update_config_for_surface
    from isofit.inversion.inverse_simple import invert_algebraic

    root = _example_dir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = configs.create_new_config(os.path.join(root, CONFIG))
        cfg.forward_model.instrument.integrations = 1
        surfaces = list(cfg.forward_model.surface.Surfaces)
        assert "glint_model_surface" in surfaces, surfaces
        sub = update_config_for_surface(deepcopy(cfg), "glint_model_surface")
        fm = ForwardModel(sub)

    surface = fm.surface
    assert hasattr(surface, "sun_glint_ind"), "expected a glint surface"
    assert surface.n_state == surface.n_wl + 2, (
        f"expected two extra glint states, got {surface.n_state - surface.n_wl}"
    )

    wl = np.asarray(surface.wl)
    winidx = np.concatenate(
        [
            np.where((wl > lo) & (wl < hi))[0]
            for lo, hi in sub.implementation.inversion.windows
        ]
    )

    remote = os.path.join(root, "remote")
    opened = {
        k: envi.open(
            envi_header(os.path.join(remote, f"{FLIGHT}_{k}_two_px"))
        ).open_memmap(interleave="bip")
        for k in ("rdn", "loc", "obs")
    }

    meas, geoms, x_atm, sub_state, x0 = [], [], [], [], []
    for r in range(opened["rdn"].shape[0]):
        for c in range(opened["rdn"].shape[1]):
            m = np.array(opened["rdn"][r, c, :], dtype=float)
            geom = Geometry(
                obs=opened["obs"][r, c, :],
                loc=opened["loc"][r, c, :],
                svf=1,
                bg_rfl=None,
                coszen=fm.atmosphere.coszen,
                full_config=sub,
            )
            atm = fm.init[fm.idx_atmosphere].copy()
            xs, _, xi = fm.unpack(fm.init.copy())
            rfl_est, _ = invert_algebraic(fm, xs, atm, xi, m, geom)
            rfl_est = fm.surface.fit_params(rfl_est, geom)
            start = fm.clip_bounds(
                np.concatenate([rfl_est, atm, xi]), fm.bounds, eps=EPS
            )
            meas.append(m)
            geoms.append(geom)
            x_atm.append(atm)
            sub_state.append(fm.init.copy())
            x0.append(start)

    from isofit.backends.torch.driver import AnalyticalBatchSolver

    solver = AnalyticalBatchSolver(
        fm, winidx, device="cpu", dtype=torch.float64, num_iter=1
    )
    return dict(
        fm=fm, solver=solver, winidx=winidx, geoms=geoms,
        meas=np.stack(meas), x_atm=np.stack(x_atm),
        sub=np.stack(sub_state), x0=np.stack(x0),
    )


def _scalar(setup, num_iter=1):
    import copy

    from isofit.inversion.inverse_simple import invert_analytical

    fm, winidx = setup["fm"], setup["winidx"]
    out = []
    for i in range(len(setup["geoms"])):
        # invert_analytical mutates geom.bg_rfl, so each run needs its own copy.
        geom = copy.deepcopy(setup["geoms"][i])
        traj, _ = invert_analytical(
            fm, winidx, setup["meas"][i], geom,
            setup["x0"][i].copy(), setup["sub"][i].copy(), num_iter=num_iter,
        )
        out.append(traj[-1])
    return np.stack(out)


def _batched(setup):
    state, _ = setup["solver"].solve(
        torch.as_tensor(setup["meas"]),
        setup["geoms"],
        torch.as_tensor(setup["x_atm"]),
        torch.as_tensor(setup["sub"]),
        torch.as_tensor(setup["x0"]),
    )
    return np.asarray(state)


def test_driver_builds_the_glint_surface(glint_setup):
    """The real ForwardModel must route to TorchGlintSurface."""
    from isofit.backends.torch.surface import TorchGlintSurface

    solver = glint_setup["solver"]
    assert solver.is_glint is True
    assert isinstance(solver.surface, TorchGlintSurface)
    assert solver.surface.n_extra == 2
    assert solver.surface.Sa_inv_extra is not None


def test_reflectance_matches_invert_analytical(glint_setup):
    ref = _scalar(glint_setup)
    got = _batched(glint_setup)
    n_wl = glint_setup["fm"].surface.n_wl

    d = np.abs(got[:, :n_wl] - ref[:, :n_wl])
    assert d.max() < RFL_ATOL, f"reflectance max|d| = {d.max():.3e}"


def test_glint_states_match_invert_analytical(glint_setup):
    """The two glint terms specifically, which the mock suite cannot reach."""
    ref = _scalar(glint_setup)
    got = _batched(glint_setup)
    surface = glint_setup["fm"].surface
    n_wl = surface.n_wl

    for name, idx in (("SKY_GLINT", surface.sky_glint_ind),
                      ("SUN_GLINT", surface.sun_glint_ind)):
        a, b = got[:, idx], ref[:, idx]
        scale = np.maximum(np.abs(b), 1.0)
        rel = np.abs(a - b) / scale
        assert rel.max() < GLINT_RTOL, (
            f"{name}: max relative difference {rel.max():.3e}; "
            f"batched {a}, scalar {b}"
        )
        assert idx >= n_wl, f"{name} index {idx} is inside the reflectance block"


def test_the_glint_states_are_actually_solved(glint_setup):
    """Guard against agreement achieved by both paths leaving the input alone."""
    ref = _scalar(glint_setup)
    surface = glint_setup["fm"].surface
    start = glint_setup["x0"][:, [surface.sky_glint_ind, surface.sun_glint_ind]]
    end = ref[:, [surface.sky_glint_ind, surface.sun_glint_ind]]
    assert not np.allclose(start, end), (
        "the reference left the glint terms at their initial value, so this "
        "fixture cannot distinguish a working solve from a no-op"
    )


def test_batched_glint_initializer_matches_the_scalar(glint_setup):
    """``fit_params_glint_batch`` vs the real ``GlintModelSurface.fit_params``.

    This is what makes the batched gather usable for glint, and therefore what
    recovers the ~3x the per-pixel gather costs. The scalar picks its bands with
    ``argmin(|target - wl|)`` and clamps with Python builtins, so an
    off-by-one band or a ``torch.clamp`` would diverge here.
    """
    import copy

    from isofit.backends.torch.geometry import BatchedGeometry
    from isofit.backends.torch.initializer import BatchedAlgebraicInitializer

    fm = glint_setup["fm"]
    solver = glint_setup["solver"]

    init = BatchedAlgebraicInitializer(
        fm, solver.radiance, device="cpu", dtype=torch.float64
    )
    assert init.glint is not None, "the initializer did not detect a glint surface"

    rng = np.random.default_rng(0)
    rfl = rng.uniform(0.02, 0.5, (len(glint_setup["geoms"]), fm.surface.n_wl))

    geom_batch = BatchedGeometry.from_geometries(
        glint_setup["geoms"], device="cpu", dtype=torch.float64
    )
    got = init.fit_params_glint_batch(torch.as_tensor(rfl), geom_batch).numpy()

    want = np.stack(
        [
            fm.surface.fit_params(rfl[i].copy(), copy.deepcopy(glint_setup["geoms"][i]))
            for i in range(len(glint_setup["geoms"]))
        ]
    )

    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    surface = fm.surface
    assert not np.allclose(
        want[:, surface.sun_glint_ind], want[0, surface.sun_glint_ind]
    ) or len(want) == 1, "SUN_GLINT is identical across pixels; fixture is degenerate"


def test_glint_gather_is_enabled(glint_setup):
    """A glint surface must now take the batched gather, not fall back."""
    from isofit.backends.torch.initializer import surface_is_batchable

    ok, reason = surface_is_batchable(glint_setup["fm"].surface)
    assert ok, f"glint surface still refused by the batched initializer: {reason}"


@pytest.mark.parametrize("num_iter", [1, 2, 3])
def test_parity_holds_across_map_iterations(glint_setup, num_iter):
    """``Seps`` must be rebuilt from the current state each MAP iteration.

    ``invert_analytical`` evaluates ``fm.Seps(x, meas, geom)`` *inside* its loop
    (inverse_simple.py:313), and Seps depends on the surface state through the
    radiometric ``Kb`` block, ``diagflat(rdn(x))`` (instrument.py:354-360). The
    driver originally built it once from x0, so the two paths solved different
    problems from the second iteration onward -- measured at 1.97e-01 absolute
    reflectance here, against a parity budget of 1e-11.

    Nothing in the suite covered this: every multi-iteration test drove the
    kernel directly with a fixed ``Seps`` tensor handed to both sides, and the
    only real-ForwardModel test ran at num_iter=1. It is also not glint-specific
    -- the reflectance-only path was off by 4.84e-02 at num_iter=2.
    """
    from isofit.backends.torch.driver import AnalyticalBatchSolver

    setup = dict(glint_setup)
    setup["solver"] = AnalyticalBatchSolver(
        setup["fm"], setup["winidx"], device="cpu", dtype=torch.float64,
        num_iter=num_iter,
    )

    ref = _scalar(setup, num_iter=num_iter)
    got = _batched(setup)
    n_state = setup["fm"].surface.n_state

    d = np.abs(got[:, :n_state] - ref[:, :n_state]).max()
    assert d < 1e-9, (
        f"num_iter={num_iter}: max|d| over the surface state = {d:.3e}; "
        "Seps is probably frozen at x0 again"
    )
