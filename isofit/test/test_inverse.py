from unittest.mock import MagicMock

import numpy as np

from isofit.inversion.inverse import Inversion, error_code


def test_error_code():
    assert error_code == -1


# Helpers


def _mock_inv(nstate, inds_free, inds_fixed=None, x_fixed=None):
    inv = MagicMock()
    inv.fm.nstate = nstate
    inv.inds_free = np.asarray(inds_free, dtype=int)
    inv.inds_fixed = np.asarray(inds_fixed if inds_fixed is not None else [], dtype=int)
    inv.x_fixed = x_fixed
    return inv


# full_statevector


def test_full_statevector_all_free():
    """When all indices are free and x_fixed is None, output equals x_free."""
    inv = _mock_inv(nstate=4, inds_free=[0, 1, 2, 3])
    x_free = np.array([1.0, 2.0, 3.0, 4.0])
    result = Inversion.full_statevector(inv, x_free)
    assert np.array_equal(result, x_free)


def test_full_statevector_with_fixed():
    """Fixed indices are filled from x_fixed; free indices come from x_free."""
    inv = _mock_inv(
        nstate=5,
        inds_free=[0, 2, 4],
        inds_fixed=[1, 3],
        x_fixed=np.array([10.0, 20.0]),
    )
    x_free = np.array([1.0, 3.0, 5.0])
    result = Inversion.full_statevector(inv, x_free)
    assert np.array_equal(result, [1.0, 10.0, 3.0, 20.0, 5.0])


def test_full_statevector_unspecified_indices_are_zero():
    """Indices not in inds_free or inds_fixed default to zero."""
    inv = _mock_inv(nstate=4, inds_free=[0, 3])
    result = Inversion.full_statevector(inv, np.array([7.0, 8.0]))
    assert result[0] == 7.0 and result[3] == 8.0
    assert result[1] == 0.0 and result[2] == 0.0


# loss_function


def _mock_inv_for_loss(winidx, x_free, xa_free, Sa_inv_sqrt, est_meas):
    """Build a minimal mock Inversion for loss_function calls."""
    inv = MagicMock()
    inv.winidx = np.asarray(winidx, dtype=int)
    inv.full_statevector = MagicMock(return_value=x_free)
    inv.fm.calc_meas = MagicMock(return_value=est_meas)
    n_free = len(x_free)
    inv.calc_conditional_prior = MagicMock(
        return_value=(xa_free, np.eye(n_free), Sa_inv_sqrt @ Sa_inv_sqrt, Sa_inv_sqrt)
    )
    return inv


def test_loss_function_perfect_fit_returns_zero():
    """When est_meas == meas and x_free == xa_free, total residual is zero."""
    winidx = [2, 4, 6, 8]
    n_win = len(winidx)
    n_free = 4
    meas = np.ones(10)
    x_free = np.full(n_free, 0.5)

    inv = _mock_inv_for_loss(
        winidx, x_free, xa_free=x_free.copy(), Sa_inv_sqrt=np.eye(n_free), est_meas=meas
    )

    residual, x_out = Inversion.loss_function(
        inv, x_free, MagicMock(), np.eye(n_win), meas
    )

    assert np.allclose(residual, 0.0)
    assert np.array_equal(x_out, x_free)


def test_loss_function_meas_residual_nonzero_prior_zero():
    """Measurement mismatch produces non-zero meas residual; satisfied prior stays zero."""
    winidx = [2, 4, 6, 8]
    n_win = len(winidx)
    n_free = 4
    meas = np.ones(10)
    est_meas = meas.copy()
    est_meas[winidx] += 2.0  # force measurement discrepancy
    x_free = np.full(n_free, 0.5)

    inv = _mock_inv_for_loss(
        winidx,
        x_free,
        xa_free=x_free.copy(),
        Sa_inv_sqrt=np.eye(n_free),
        est_meas=est_meas,
    )

    residual, _ = Inversion.loss_function(inv, x_free, MagicMock(), np.eye(n_win), meas)

    assert np.all(np.abs(residual[:n_win]) > 0)  # measurement residual non-zero
    assert np.allclose(residual[n_win:], 0.0)  # prior residual zero


def test_loss_function_prior_residual_nonzero_meas_zero():
    """Departure from prior produces non-zero prior residual; satisfied measurement stays zero."""
    winidx = [1, 3, 5, 7]
    n_win = len(winidx)
    n_free = 4
    meas = np.ones(10)
    x_free = np.full(n_free, 0.5)
    xa_free = np.zeros(n_free)  # prior mean differs from x_free

    inv = _mock_inv_for_loss(
        winidx, x_free, xa_free=xa_free, Sa_inv_sqrt=np.eye(n_free), est_meas=meas
    )

    residual, _ = Inversion.loss_function(inv, x_free, MagicMock(), np.eye(n_win), meas)

    assert np.allclose(residual[:n_win], 0.0)  # measurement residual zero
    assert np.all(np.abs(residual[n_win:]) > 0)  # prior residual non-zero


# jacobian


def test_jacobian_shape():
    """Jacobian must have shape (n_win + n_free, n_free)."""
    n_state = 6
    n_free = 5
    winidx = [0, 2, 4]
    n_win = len(winidx)

    inv = MagicMock()
    inv.winidx = np.asarray(winidx, dtype=int)
    inv.inds_free = np.arange(n_free, dtype=int)
    inv.full_statevector = MagicMock(return_value=np.zeros(n_free))

    # K has shape (n_state, n_state); window rows then free cols are selected
    rng = np.random.default_rng(42)
    inv.fm.K = MagicMock(return_value=rng.normal(size=(n_state, n_state)))

    inv.calc_conditional_prior = MagicMock(
        return_value=(np.zeros(n_free), np.eye(n_free), np.eye(n_free), np.eye(n_free))
    )

    result = Inversion.jacobian(inv, np.zeros(n_free), MagicMock(), np.eye(n_win))

    assert result.shape == (n_win + n_free, n_free)


def test_jacobian_meas_rows_scaled_by_Seps():
    """Measurement rows of the jacobian are Seps_inv_sqrt @ K_win[:,free]."""
    n_state = 4
    n_free = 4
    winidx = [0, 1]
    n_win = len(winidx)

    K = np.arange(1.0, n_state * n_state + 1).reshape(n_state, n_state)
    Seps_inv_sqrt = 3.0 * np.eye(n_win)

    inv = MagicMock()
    inv.winidx = np.asarray(winidx, dtype=int)
    inv.inds_free = np.arange(n_free, dtype=int)
    inv.full_statevector = MagicMock(return_value=np.zeros(n_free))
    inv.fm.K = MagicMock(return_value=K)
    inv.calc_conditional_prior = MagicMock(
        return_value=(np.zeros(n_free), np.eye(n_free), np.eye(n_free), np.eye(n_free))
    )

    result = Inversion.jacobian(inv, np.zeros(n_free), MagicMock(), Seps_inv_sqrt)

    expected_meas_rows = Seps_inv_sqrt @ K[np.ix_(winidx, list(range(n_free)))]
    assert np.allclose(result[:n_win], expected_meas_rows)
