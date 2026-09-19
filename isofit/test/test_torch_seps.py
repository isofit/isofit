"""Parity tests for the batched observation-error covariance.

Compared against :meth:`isofit.core.forward.ForwardModel.Seps` invoked unbound
against a mock, so the ``Sy + Kb·Sb·Kbᵀ + Γ`` assembly is exercised directly.

The structural claim under test is that the radiometric ``Kb`` block, being
``diagflat(meas)``, contributes only a diagonal -- and therefore that a config
with no dense ``Kb`` columns and no model discrepancy has a strictly diagonal
``Seps``. That is what determines whether the upstream ``dsymv`` issue can
affect a given run at all.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from isofit.backends.torch.seps import seps_batch
from isofit.core.forward import ForwardModel

pytestmark = pytest.mark.torch_cpu

RTOL = 1e-11
N_CHAN = 24
B = 8


def _inputs(seed=0, n_dense=0, with_gamma=False):
    rng = np.random.default_rng(seed)
    meas = rng.uniform(1.0, 10.0, (B, N_CHAN))
    sy_diag = rng.uniform(1e-4, 1e-2, (B, N_CHAN))
    sb_rad = rng.uniform(1e-5, 1e-3, N_CHAN)

    dense = rng.normal(size=(B, N_CHAN, n_dense)) if n_dense else None
    sb_dense = rng.uniform(1e-5, 1e-3, n_dense) if n_dense else None

    gamma = None
    if with_gamma:
        A = rng.normal(size=(N_CHAN, N_CHAN))
        gamma = A @ A.T * 1e-4

    return meas, sy_diag, sb_rad, dense, sb_dense, gamma


def _scalar_seps(meas_i, sy_i, sb_rad, dense_i, sb_dense, gamma):
    """Assemble Seps the way ForwardModel.Seps does, for one pixel."""
    n_dense = 0 if dense_i is None else dense_i.shape[1]

    Kb = np.hstack(
        [np.diagflat(meas_i)] + ([dense_i] if n_dense else [])
    )
    sb = np.concatenate([sb_rad] + ([sb_dense] if n_dense else []))
    Sb = np.diagflat(sb)
    Sy = np.diagflat(sy_i)

    fm = MagicMock()
    fm.model_discrepancy = gamma
    fm.Sb = MagicMock(return_value=Sb)
    fm.Kb = MagicMock(return_value=Kb)
    fm.instrument.Sy = MagicMock(return_value=Sy)

    return ForwardModel.Seps(fm, x=None, meas=meas_i, geom=None)


def _run(meas, sy_diag, sb_rad, dense, sb_dense, gamma, winidx=None):
    return seps_batch(
        torch.as_tensor(sy_diag),
        torch.as_tensor(meas),
        torch.as_tensor(sb_rad),
        dense_columns=None if dense is None else torch.as_tensor(dense),
        sb_dense=None if sb_dense is None else torch.as_tensor(sb_dense),
        model_discrepancy=None if gamma is None else torch.as_tensor(gamma),
        winidx=None if winidx is None else torch.as_tensor(winidx, dtype=torch.int64),
    )


# --- parity ----------------------------------------------------------------------


@pytest.mark.parametrize("n_dense", [0, 1, 3])
@pytest.mark.parametrize("with_gamma", [False, True])
def test_matches_forward_model_seps(n_dense, with_gamma):
    meas, sy, sb_rad, dense, sb_dense, gamma = _inputs(
        seed=n_dense + int(with_gamma) * 10, n_dense=n_dense, with_gamma=with_gamma
    )

    got = _run(meas, sy, sb_rad, dense, sb_dense, gamma).numpy()
    ref = np.stack(
        [
            _scalar_seps(
                meas[i],
                sy[i],
                sb_rad,
                None if dense is None else dense[i],
                sb_dense,
                gamma,
            )
            for i in range(B)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


def test_windowed_matches_windowed_scalar():
    """Windowing must be applied consistently with the scalar slicing."""
    meas, sy, sb_rad, dense, sb_dense, gamma = _inputs(seed=5, n_dense=2, with_gamma=True)
    winidx = np.sort(np.random.default_rng(0).choice(N_CHAN, 15, replace=False))

    got = _run(meas, sy, sb_rad, dense, sb_dense, gamma, winidx=winidx).numpy()
    ref = np.stack(
        [
            _scalar_seps(meas[i], sy[i], sb_rad, dense[i], sb_dense, gamma)[
                np.ix_(winidx, winidx)
            ]
            for i in range(B)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


# --- structure -------------------------------------------------------------------


def test_seps_is_diagonal_without_dense_columns_or_gamma():
    """The premise behind the upstream dsymv impact analysis.

    With only diagonal Sy and the radiometric Kb block, Seps has no
    off-diagonal content -- so discarding off-diagonals changes nothing.
    """
    meas, sy, sb_rad, _, _, _ = _inputs(seed=7)
    got = _run(meas, sy, sb_rad, None, None, None).numpy()

    for i in range(B):
        off = got[i] - np.diag(np.diag(got[i]))
        assert np.allclose(off, 0.0), "expected a strictly diagonal Seps"


def test_dense_column_introduces_off_diagonal_content():
    """A single dense Kb column is enough to make the off-diagonals matter."""
    meas, sy, sb_rad, dense, sb_dense, _ = _inputs(seed=8, n_dense=1)
    got = _run(meas, sy, sb_rad, dense, sb_dense, None).numpy()

    off = got[0] - np.diag(np.diag(got[0]))
    assert np.abs(off).max() > 0, "a dense Kb column must produce off-diagonals"


def test_result_is_symmetric():
    meas, sy, sb_rad, dense, sb_dense, gamma = _inputs(seed=9, n_dense=2, with_gamma=True)
    got = _run(meas, sy, sb_rad, dense, sb_dense, gamma).numpy()
    np.testing.assert_allclose(got, got.transpose(0, 2, 1), rtol=1e-12)


def test_result_is_positive_definite():
    """Seps is inverted downstream; a non-PD result would fail Cholesky."""
    meas, sy, sb_rad, dense, sb_dense, gamma = _inputs(
        seed=10, n_dense=2, with_gamma=True
    )
    got = _run(meas, sy, sb_rad, dense, sb_dense, gamma)
    torch.linalg.cholesky(got)  # raises if not PD


def test_pushbroom_full_noise_matrix_is_added():
    """The pushbroom model supplies a shared full covariance rather than a diagonal."""
    meas, _, sb_rad, _, _, _ = _inputs(seed=11)
    rng = np.random.default_rng(12)
    A = rng.normal(size=(N_CHAN, N_CHAN))
    sy_full = A @ A.T + N_CHAN * np.eye(N_CHAN)

    got = seps_batch(
        None,
        torch.as_tensor(meas),
        torch.as_tensor(sb_rad),
        sy_full=torch.as_tensor(sy_full),
    ).numpy()

    for i in range(B):
        expected = sy_full + np.diagflat(meas[i] ** 2 * sb_rad)
        np.testing.assert_allclose(got[i], expected, rtol=RTOL)


def test_dense_columns_without_sb_raises():
    meas, sy, sb_rad, dense, _, _ = _inputs(seed=13, n_dense=2)
    with pytest.raises(ValueError, match="sb_dense is required"):
        seps_batch(
            torch.as_tensor(sy),
            torch.as_tensor(meas),
            torch.as_tensor(sb_rad),
            dense_columns=torch.as_tensor(dense),
        )
