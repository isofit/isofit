from unittest.mock import MagicMock

import numpy as np
import pytest

from isofit.core.instrument import Instrument, wl_tol


def test_wl_tol():
    assert wl_tol == 0.01


# Helpers


def _mock_inst(n_chan=5, n_state=0, statevec_names=None):
    """Minimal Instrument mock with attributes set directly."""
    inst = MagicMock(spec=Instrument)
    inst.n_chan = n_chan
    inst.n_state = n_state
    inst.statevec_names = statevec_names or []
    inst.wl_init = np.linspace(400, 2500, n_chan)
    inst.fwhm_init = np.full(n_chan, 10.0)
    inst.init = np.zeros(n_state)
    inst.eof = None
    inst.eof_idx = []
    inst.unknowns = None
    return inst


# xa / Sa


def test_xa_returns_copy_of_init():
    inst = _mock_inst(n_state=3)
    inst.init = np.array([1.0, 2.0, 3.0])
    result = Instrument.xa(inst)
    assert np.array_equal(result, inst.init)
    # Must be a copy, not the same object
    result[0] = 99.0
    assert inst.init[0] == 1.0


def test_Sa_returns_cached_matrix():
    inst = _mock_inst(n_state=2)
    inst.Sa_cached = np.eye(2) * 4.0
    result = Instrument.Sa(inst)
    assert np.array_equal(result, inst.Sa_cached)


def test_Sa_zero_state_returns_empty():
    inst = _mock_inst(n_state=0)
    result = Instrument.Sa(inst)
    assert result.shape == (0, 0)


# Sy — SNR noise model


def test_Sy_snr_diagonal():
    """Sy for SNR model is diagonal with (meas/snr)^2 on the diagonal."""
    n = 6
    inst = _mock_inst(n_chan=n)
    inst.model_type = "SNR"
    inst.snr = 100.0
    inst.dn_uncertainty_embedding = None
    inst.integrations = 1

    meas = np.linspace(1.0, 10.0, n)
    Sy = Instrument.Sy(inst, meas, MagicMock())

    expected_diag = np.power(meas / inst.snr, 2)
    assert np.allclose(np.diag(Sy), expected_diag)
    assert Sy.shape == (n, n)


def test_Sy_snr_clamps_small_noise():
    """Very small meas values are clamped so noise stays positive."""
    n = 4
    inst = _mock_inst(n_chan=n)
    inst.model_type = "SNR"
    inst.snr = 1e6  # forces tiny nedl
    inst.dn_uncertainty_embedding = None
    inst.integrations = 1

    meas = np.full(n, 1e-10)
    Sy = Instrument.Sy(inst, meas, MagicMock())

    assert np.all(np.diag(Sy) > 0)


# sample — fast path


def test_sample_fastpath_same_wavelengths():
    """When calibration is fixed and wavelengths match within wl_tol, rdn_hi is returned as-is."""
    n = 8
    wl = np.linspace(400, 2500, n)
    inst = _mock_inst(n_chan=n)
    inst.calibration_fixed = True
    rdn_hi = np.random.default_rng(0).uniform(0, 1, n)

    result = Instrument.sample(inst, np.array([]), wl, rdn_hi)
    assert np.array_equal(result, rdn_hi)


def test_sample_fastpath_skipped_when_lengths_differ():
    """When wl_hi has a different length, the fast path is bypassed and calibration() is called."""
    n = 8
    inst = _mock_inst(n_chan=n)
    inst.calibration_fixed = True
    # wl_hi has length n+1 → different from wl_init (length n), triggers resampling
    wl_hi = np.linspace(400, 2500, n + 1)
    rdn_hi = np.ones(n + 1)

    # calibration() must be called; set it to return wl_init and fwhm_init
    inst.calibration = MagicMock(return_value=(inst.wl_init, inst.fwhm_init))

    Instrument.sample(inst, np.array([]), wl_hi, rdn_hi)
    inst.calibration.assert_called_once()


# calibration — WL_SHIFT and GROW_FWHM


def test_calibration_no_statevec_returns_init():
    """With no state vector elements, calibration returns wl_init and fwhm_init unchanged."""
    inst = _mock_inst(n_chan=6)
    wl, fwhm = Instrument.calibration(inst, np.array([]))
    assert np.allclose(wl, inst.wl_init)
    assert np.allclose(fwhm, inst.fwhm_init)


def test_calibration_wl_shift():
    """WL_SHIFT element translates all wavelengths by the given offset."""
    n = 6
    inst = _mock_inst(n_chan=n, n_state=1, statevec_names=["WL_SHIFT"])
    shift = 5.0
    x = np.array([shift])

    wl, fwhm = Instrument.calibration(inst, x)
    assert np.allclose(wl, inst.wl_init + shift)
    assert np.allclose(fwhm, inst.fwhm_init)


def test_calibration_grow_fwhm():
    """GROW_FWHM element broadens all channels by the given delta."""
    n = 6
    inst = _mock_inst(n_chan=n, n_state=1, statevec_names=["GROW_FWHM"])
    delta = 2.5
    x = np.array([delta])

    wl, fwhm = Instrument.calibration(inst, x)
    assert np.allclose(fwhm, inst.fwhm_init + delta)
    assert np.allclose(wl, inst.wl_init)


# eof_offset


def test_eof_offset_no_eof_returns_zeros():
    inst = _mock_inst(n_chan=5)
    result = Instrument.eof_offset(inst, np.array([]))
    assert np.allclose(result, 0.0)
    assert result.shape == (5,)


def test_eof_offset_single_eof_column():
    """Offset equals eof_column * x_instrument[eof_idx]."""
    n = 4
    inst = _mock_inst(n_chan=n, n_state=2)
    inst.eof = np.ones((n, 2))  # both columns are all-ones
    inst.eof_idx = [1]  # retrieve second column
    x_instrument = np.array([0.0, 3.0])

    result = Instrument.eof_offset(inst, x_instrument)
    assert np.allclose(result, 3.0)


# DN_additive_uncertainty (static method)


def test_dn_additive_uncertainty_inflation():
    """Doubling inflation doubles the output."""
    meas = np.array([100.0, 200.0, 300.0])
    rcc = np.ones_like(meas)  # DN = meas
    interp = lambda dn: np.full_like(dn, 1.1)  # noise ratio always 1.1

    u1 = Instrument.DN_additive_uncertainty(meas, rcc, interp, inflation=1.0)
    u2 = Instrument.DN_additive_uncertainty(meas, rcc, interp, inflation=2.0)
    assert np.allclose(u2, 2 * u1)


def test_dn_additive_uncertainty_zero_for_unity_ratio():
    """When noise ratio is exactly 1.0, uncertainty is zero."""
    meas = np.array([50.0, 75.0])
    rcc = np.ones_like(meas)
    interp = lambda dn: np.ones_like(dn)  # ratio == 1 → noise-free

    result = Instrument.DN_additive_uncertainty(meas, rcc, interp, inflation=1.0)
    assert np.allclose(result, 0.0)


# summarize


def test_summarize_empty_statevector():
    inst = _mock_inst()
    assert Instrument.summarize(inst, np.array([]), MagicMock()) == ""


def test_summarize_nonempty_statevector():
    inst = _mock_inst()
    result = Instrument.summarize(inst, np.array([1.5, 2.0]), MagicMock())
    assert result.startswith("Instrument:")
    assert "1.500" in result and "2.000" in result
