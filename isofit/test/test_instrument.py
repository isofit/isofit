from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.io import savemat
from scipy.signal import convolve

from isofit.configs.sections.instrument_config import InstrumentConfig
from isofit.core.common import spectral_response_function
from isofit.core.instrument import Instrument, NoiseModel, wl_tol


def test_wl_tol():
    assert wl_tol == 0.01


# Helpers


def _mock_inst(n_chan=5, n_state=0, **kwargs):
    """Minimal Instrument mock with attributes set directly."""
    inst = MagicMock(spec=Instrument)
    inst.n_chan = n_chan
    inst.n_state = n_state
    inst.statevec_names = []
    inst.wl_init = np.linspace(400, 2500, n_chan)
    inst.fwhm_init = np.full(n_chan, 10.0)
    inst.init = np.zeros(n_state)
    inst.state_idx = {}
    inst.config_state_names = []
    inst.unknowns = None

    for key, value in kwargs.items():
        setattr(inst, key, value)

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
    Sy = NoiseModel.Sy_SNR(inst, meas, MagicMock())

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
    Sy = NoiseModel.Sy_SNR(inst, meas, MagicMock())

    assert np.all(np.diag(Sy) > 0)


# sample — fast path


def test_sample_fastpath_same_wavelengths():
    """When calibration is fixed and wavelengths match within wl_tol, rdn_hi is returned as-is."""
    n = 8
    wl = np.linspace(400, 2500, n)
    inst = _mock_inst(n_chan=n)
    inst.wavelengths_fixed = True
    rdn_hi = np.random.default_rng(0).uniform(0, 1, n)

    Instrument.calibration(inst, np.array([]))

    result = Instrument.sample(inst, np.array([]), wl, rdn_hi)
    assert np.array_equal(result, rdn_hi)


def test_sample_fastpath_skipped_when_lengths_differ():
    """When wl_hi has a different length, the fast path is bypassed and calibration() is called."""
    n = 8
    inst = _mock_inst(n_chan=n)
    inst.wavelengths_fixed = True
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
    inst = _mock_inst(
        n_chan=n,
        n_state=1,
        statevec_names=["WL_SHIFT"],
        config_state_names=["WL_SHIFT"],
        state_idx={"WL_SHIFT": [0]},
        calibration_fixed=False,
    )
    shift = 5.0
    x = np.array([shift])

    wl, fwhm = Instrument.calibration(inst, x)
    assert np.allclose(wl, inst.wl_init + shift)
    assert np.allclose(fwhm, inst.fwhm_init)


def test_calibration_grow_fwhm():
    """GROW_FWHM element broadens all channels by the given delta."""
    n = 6
    inst = _mock_inst(
        n_chan=n,
        n_state=1,
        statevec_names=["GROW_FWHM"],
        config_state_names=["GROW_FWHM"],
        state_idx={"GROW_FWHM": [0]},
        calibration_fixed=False,
    )
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
    inst.state_idx["EOF"] = [1]  # retrieve second column
    x_instrument = np.array([0.0, 3.0])

    result = Instrument.eof_offset(inst, x_instrument)
    assert np.allclose(result, 3.0)


# rcc_factor / nonlinearity


def _nl_inst(n_chan=5, fit_rcc=True, fit_nl=True):
    """Bare Instrument (no __init__) with optional PER_WL_RCC and NL_ALPHA/NL_BETA."""
    inst = Instrument.__new__(Instrument)
    inst.n_chan = n_chan
    inst.wl_init = np.linspace(400, 2500, n_chan)
    inst.wavelengths_fixed = True
    inst.eof = np.array([])
    inst.dn_mapping = np.linspace(50.0, 150.0, n_chan)
    inst.fit_rcc = fit_rcc
    inst.fit_nl = fit_nl
    inst.state_idx = {}
    n_state = 0
    if fit_rcc:
        inst.state_idx["PER_WL_RCC"] = list(range(n_chan))
        n_state = n_chan
    if fit_nl:
        inst.state_idx["NL_ALPHA"] = [n_state]
        inst.state_idx["NL_BETA"] = [n_state + 1]
        n_state += 2
    inst.n_state = n_state

    rng = np.random.default_rng(0)
    rcc = rng.uniform(0.9, 1.1, n_chan) if fit_rcc else []
    nl = [0.05, 200.0] if fit_nl else []
    x = np.concatenate([rcc, nl])
    rdn = rng.uniform(0.5, 3.0, n_chan)
    return inst, x, rdn


def _modeled_meas(inst, x, rdn):
    meas = inst.sample(x, inst.wl_init, rdn)
    return meas * inst.rcc_factor(x, meas) + inst.eof_offset(x)


def test_rcc_factor_is_unity_without_rcc_or_nl():
    inst, x, rdn = _nl_inst(fit_rcc=False, fit_nl=False)
    assert np.array_equal(inst.rcc_factor(x, rdn), np.ones(inst.n_chan))
    assert np.array_equal(inst.rcc_chain_factor(x, rdn), np.ones(inst.n_chan))


def test_rcc_factor_nonlinearity():
    inst, x, rdn = _nl_inst()
    rcc, alpha, beta = x[: inst.n_chan], x[-2], x[-1]
    expected = rcc * (1.0 + alpha * np.exp(-rdn * inst.dn_mapping / beta))
    assert np.allclose(inst.rcc_factor(x, rdn), expected)


@pytest.mark.parametrize("fit_rcc", [True, False])
def test_rcc_chain_factor_matches_finite_difference(fit_rcc):
    inst, x, rdn = _nl_inst(fit_rcc=fit_rcc)
    h = 1e-6
    fd = (_modeled_meas(inst, x, rdn + h) - _modeled_meas(inst, x, rdn - h)) / (2 * h)
    assert np.allclose(inst.rcc_chain_factor(x, rdn), fd, rtol=1e-6)


@pytest.mark.parametrize("fit_rcc", [True, False])
def test_dmeas_dinstrument_matches_finite_difference(fit_rcc):
    inst, x, rdn = _nl_inst(fit_rcc=fit_rcc)
    K = inst.dmeas_dinstrument(x, inst.wl_init, rdn)

    fd = np.zeros_like(K)
    for i in range(len(x)):
        h = 1e-6 * max(1.0, abs(x[i]))
        dx = np.zeros_like(x)
        dx[i] = h
        fd[:, i] = (
            _modeled_meas(inst, x + dx, rdn) - _modeled_meas(inst, x - dx, rdn)
        ) / (2 * h)

    assert np.allclose(K, fd, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("fit_rcc", [True, False])
def test_dmeas_dinstrumentb_propagates_rcc_and_nl(fit_rcc):
    """Relative calibration columns scale the full modeled signal; spectral unknowns
    perturb the sampled radiance, which is then propagated through rcc and nl."""
    inst, x, rdn = _nl_inst(fit_rcc=fit_rcc)
    n = inst.n_chan
    inst.unknowns = MagicMock(
        wavelength_calibration_uncertainty=0.1, stray_srf_uncertainty=0.1
    )
    inst.bvec = [f"Cal_Relative_{i}" for i in range(n)]
    inst.bvec += ["Cal_Spectral", "Cal_Stray_SRF"]
    inst.cal_spectral_idx, inst.cal_stray_idx = n, n + 1

    Kb = inst.dmeas_dinstrumentb(x, inst.wl_init, rdn)
    assert Kb.shape == (n, n + 2)
    assert np.allclose(Kb[:, :n], np.diag(_modeled_meas(inst, x, rdn)))

    h = 1e-6
    ssrf = spectral_response_function(np.arange(-10, 11), 0, 4)
    perturbations = {
        inst.cal_spectral_idx: np.hstack((np.diff(rdn), [0])),
        inst.cal_stray_idx: convolve(rdn, ssrf, mode="same") - rdn,
    }
    for col, d in perturbations.items():
        fd = (
            _modeled_meas(inst, x, rdn + h * d) - _modeled_meas(inst, x, rdn - h * d)
        ) / (2 * h)
        assert np.allclose(Kb[:, col], fd, rtol=1e-6, atol=1e-12)


# __init__ with nonlinearity and RCC prior file


def _real_inst(tmp_path, statevector, n_chan=4, rcc_prior=True):
    wl = np.linspace(400, 2500, n_chan)
    wl_file = tmp_path / "wl.txt"
    np.savetxt(wl_file, np.column_stack([wl, np.full(n_chan, 10.0)]))
    instrument = {
        "wavelength_file": str(wl_file),
        "SNR": 300.0,
        "integrations": 1,
        "statevector": statevector,
    }
    if rcc_prior:
        prior_file = tmp_path / "rcc_prior.mat"
        savemat(
            prior_file,
            {
                "cov": np.eye(n_chan) * 0.01,
                "mean": np.full(n_chan, 1.0),
                "bounds": np.array([[0.5, 2.0]] * n_chan),
            },
        )
        instrument["rcc_prior_file"] = str(prior_file)

    full_config = MagicMock()
    full_config.forward_model.instrument = InstrumentConfig(instrument)
    return Instrument(full_config)


_RCC_SV = {
    "bounds": [0.5, 2.0],
    "init": 1.0,
    "prior_mean": 0.0,
    "prior_sigma": 0.0,
    "scale": 1.0,
}
_ALPHA_SV = {
    "bounds": [-1.0, 1.0],
    "init": 0.0,
    "prior_mean": 0.0,
    "prior_sigma": 0.2,
    "scale": 1.0,
}
_BETA_SV = {
    "bounds": [10.0, 20000.0],
    "init": 1000.0,
    "prior_mean": 1000.0,
    "prior_sigma": 1000.0,
    "scale": 1000.0,
}


def test_rcc_prior_file_only_overrides_rcc_elements(tmp_path):
    """NL_* sort ahead of PER_WL_RCC, so the prior file must be applied by index."""
    inst = _real_inst(
        tmp_path,
        {"PER_WL_RCC": _RCC_SV, "NL_ALPHA": _ALPHA_SV, "NL_BETA": _BETA_SV},
    )
    assert inst.fit_rcc and inst.fit_nl

    rcc_idx = inst.state_idx["PER_WL_RCC"]
    assert np.allclose(inst.prior_mean[rcc_idx], 1.0)
    assert np.allclose([inst.bounds[i] for i in rcc_idx], [0.5, 2.0])

    for name, sv in [("NL_ALPHA", _ALPHA_SV), ("NL_BETA", _BETA_SV)]:
        i = inst.state_idx[name][0]
        assert np.allclose(inst.bounds[i], sv["bounds"])
        assert inst.init[i] == sv["init"]
        assert inst.prior_mean[i] == sv["prior_mean"]
        assert inst.scale[i] == sv["scale"]
        assert np.isclose(inst.Sa()[i, i], sv["prior_sigma"] ** 2)


def test_nl_alpha_without_beta_raises(tmp_path):
    with pytest.raises(ValueError, match="NL_ALPHA and NL_BETA"):
        _real_inst(tmp_path, {"NL_ALPHA": _ALPHA_SV}, rcc_prior=False)


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
