"""The four reflectance arguments to ``calc_rdn`` must match the scalar path.

``ForwardModel.calc_rdn`` takes four distinct reflectance quantities. The scalar
``Seps`` builds them like this (``isofit/core/forward.py:655-669``)::

    rho_dir_dir_hi = upsample(rho_dir_dir)
    rho_dif_dir_hi = upsample(rho_dif_dir)
    rho_dir_dif_hi = upsample(geom.bg_rfl) if isinstance(geom.bg_rfl, ndarray) else rho_dir_dir_hi
    rho_dif_dif_hi = upsample(geom.bg_rfl) if isinstance(geom.bg_rfl, ndarray) else rho_dif_dir_hi

and ``invert_analytical`` reaches it having *already* assigned
``geom.bg_rfl = rho_dif_dif`` unconditionally
(``isofit/inversion/inverse_simple.py:259-264``). So by the time
``fm.Seps`` runs at ``inverse_simple.py:313`` the ``isinstance`` test is always
true, and **both** downward-diffuse terms are the background reflectance:

    rho_dir_dif == rho_dif_dif == geom.bg_rfl

The batched driver passed ``(rho, rho, rho, rho_dif_dif)`` instead, using the
surface reflectance for ``rho_dir_dif``. That is invisible without a background
reflectance -- ``rho_dif_dif`` falls back to ``rho_dif_dir``, which equals
``rho`` -- so the 100,000-pixel parity run (no ``--use_background_rfl``) agreed
to 5.96e-08 with the bug present. It only diverges on scenes that supply a
background, which is exactly the adjacency-correction case the term exists for.

These tests pin the argument mapping directly rather than going through a full
``ForwardModel``, because the failure is in *which* tensor is passed, not in the
arithmetic that consumes it.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.torch_cpu

# calc_rdn's reflectance parameters, in positional order.
RHO_ARGS = ("rho_dir_dir", "rho_dif_dir", "rho_dir_dif", "rho_dif_dif")


def _scalar_rho_arguments(rho_dir_dir, rho_dif_dir, bg_rfl):
    """What ``ForwardModel.Seps`` passes, replaying forward.py:655-669.

    ``bg_rfl`` is what ``geom.bg_rfl`` holds when ``Seps`` is called. Per
    inverse_simple.py:264 that is never ``None`` on the analytical-line path.
    """
    is_array = isinstance(bg_rfl, np.ndarray)
    return {
        "rho_dir_dir": rho_dir_dir,
        "rho_dif_dir": rho_dif_dir,
        "rho_dir_dif": bg_rfl if is_array else rho_dir_dir,
        "rho_dif_dif": bg_rfl if is_array else rho_dif_dir,
    }


class _SpyRadiance:
    """Records the reflectance arguments of the last ``calc_rdn`` call."""

    def __init__(self):
        self.seen = None

    def calc_rdn(self, rho_dir_dir, rho_dif_dir, rho_dir_dif, rho_dif_dif, *rest):
        self.seen = {
            "rho_dir_dir": rho_dir_dir,
            "rho_dif_dir": rho_dif_dir,
            "rho_dir_dif": rho_dir_dif,
            "rho_dif_dif": rho_dif_dif,
        }
        return torch.zeros_like(rho_dir_dir)


def _driver_rho_arguments(rho, rho_dif_dif, rho_dif_dir=None):
    """What the driver passes, replaying its ``calc_rdn`` call verbatim.

    Kept as a literal transcription of the call in
    ``isofit/backends/torch/driver.py`` so that this test tracks the source: if
    the driver's argument order changes, update this to match and the
    assertions below will judge the new order.
    """
    import inspect
    import re

    from isofit.backends.torch import driver as driver_mod

    source = inspect.getsource(driver_mod)
    calls = re.findall(r"\.calc_rdn\(\s*\n\s*(.+?),\s*Ls,", source)
    assert calls, "could not locate the driver's calc_rdn calls"

    rho_dif_dir = rho if rho_dif_dir is None else rho_dif_dir
    spy = _SpyRadiance()
    results = []
    for call in calls:
        names = [a.strip() for a in call.split(",")]
        assert len(names) == 4, f"expected 4 reflectance arguments, got {names}"
        # Local names the driver uses for the two calc_rfl quantities. rho_dd /
        # rho_fd are the per-iteration rebuild's names for the same pair.
        env = {
            "rho": rho,
            "rho_dir_dir": rho,
            "rho_dd": rho,
            "rho_dif_dir": rho_dif_dir,
            "rho_fd": rho_dif_dir,
            "rho_dif_dif": rho_dif_dif,
        }
        spy.calc_rdn(*[env[n] for n in names], None)
        results.append(dict(spy.seen))
    return results


def test_all_calc_rdn_call_sites_agree():
    """The driver must not use different reflectance arguments in different places."""
    rho = torch.tensor([0.10, 0.11, 0.12], dtype=torch.float64)
    bg = torch.tensor([0.50, 0.51, 0.52], dtype=torch.float64)
    calls = _driver_rho_arguments(rho, bg)
    assert len(calls) >= 3, f"expected at least 3 calc_rdn call sites, found {len(calls)}"
    first = calls[0]
    for i, other in enumerate(calls[1:], start=1):
        for arg in RHO_ARGS:
            assert torch.equal(first[arg], other[arg]), (
                f"calc_rdn call site {i} passes a different {arg} than call site 0; "
                "the modeled radiance for Kb must be built identically everywhere"
            )


def test_downward_diffuse_terms_use_the_background_reflectance():
    """With a background present, both *_dif terms must be it, per forward.py:659-669."""
    rho_np = np.array([0.10, 0.11, 0.12])
    bg_np = np.array([0.50, 0.51, 0.52])
    rho = torch.as_tensor(rho_np)
    bg = torch.as_tensor(bg_np)

    expected = _scalar_rho_arguments(rho_np, rho_np, bg_np)
    got = _driver_rho_arguments(rho, bg)[0]

    for arg in RHO_ARGS:
        assert np.allclose(got[arg].numpy(), expected[arg]), (
            f"{arg}: driver passes {got[arg].numpy()}, scalar Seps uses "
            f"{expected[arg]} (forward.py:659-669 with geom.bg_rfl set by "
            "inverse_simple.py:264)"
        )


def test_without_a_background_the_mapping_is_unchanged():
    """No background: rho_dif_dif falls back to rho, so all four are rho.

    This is the configuration the 100,000-pixel parity run used, and it must
    keep agreeing -- the fix must not perturb the case that was already correct.
    """
    rho_np = np.array([0.10, 0.11, 0.12])
    rho = torch.as_tensor(rho_np)

    # bg_rfl absent -> inverse_simple.py:262 sets rho_dif_dif = rho_dif_dir,
    # then line 264 assigns it to geom.bg_rfl, so Seps sees that array.
    expected = _scalar_rho_arguments(rho_np, rho_np, rho_np)
    got = _driver_rho_arguments(rho, rho)[0]

    for arg in RHO_ARGS:
        assert np.allclose(got[arg].numpy(), expected[arg]), f"{arg} changed"
