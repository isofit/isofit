"""Parity tests for the batched instrument noise model.

Compared against :meth:`isofit.core.instrument.Instrument.Sy` invoked unbound
against a mock, so the noise arithmetic is exercised without a configured
instrument (and without the wavelength/response files that would require).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from isofit.backends.torch.instrument import TorchInstrument
from isofit.core.instrument import Instrument

pytestmark = pytest.mark.torch_cpu

RTOL = 1e-12
N_CHAN = 32
B = 12


def _meas(seed=0, lo=0.1, hi=10.0):
    return np.random.default_rng(seed).uniform(lo, hi, (B, N_CHAN))


def _scalar(model_type, **attrs):
    inst = MagicMock()
    inst.model_type = model_type
    inst.dn_uncertainty_embedding = None
    for k, v in attrs.items():
        setattr(inst, k, v)
    return inst


def _batched(model_type, **attrs):
    inst = _scalar(model_type, **attrs)
    inst.n_chan = N_CHAN
    return TorchInstrument(inst)


# --- SNR ------------------------------------------------------------------------


def test_snr_matches_scalar():
    snr = np.random.default_rng(1).uniform(50, 500, N_CHAN)
    meas = _meas()

    got = _batched("SNR", snr=snr, integrations=1).Sy_diagonal(
        torch.as_tensor(meas)
    ).numpy()

    ref = np.stack(
        [
            np.diag(Instrument.Sy(_scalar("SNR", snr=snr), meas[i], geom=None))
            for i in range(B)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


def test_snr_clamps_nonpositive_noise():
    """Zero radiance would give zero noise and divide-by-zero downstream."""
    snr = np.full(N_CHAN, 100.0)
    meas = np.zeros((B, N_CHAN))

    got = _batched("SNR", snr=snr, integrations=1).Sy_diagonal(
        torch.as_tensor(meas)
    ).numpy()

    ref = np.stack(
        [
            np.diag(Instrument.Sy(_scalar("SNR", snr=snr), meas[i].copy(), geom=None))
            for i in range(B)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)
    assert np.all(got > 0)


# --- parametric ------------------------------------------------------------------


@pytest.mark.parametrize("integrations", [1, 40])
def test_parametric_matches_scalar(integrations):
    rng = np.random.default_rng(2)
    noise = np.stack(
        [
            rng.uniform(0.001, 0.01, N_CHAN),
            rng.uniform(0.1, 1.0, N_CHAN),
            rng.uniform(0.0, 0.05, N_CHAN),
        ],
        axis=1,
    )
    meas = _meas(3)

    got = _batched("parametric", noise=noise, integrations=integrations).Sy_diagonal(
        torch.as_tensor(meas)
    ).numpy()

    ref = np.stack(
        [
            np.diag(
                Instrument.Sy(
                    _scalar("parametric", noise=noise, integrations=integrations),
                    meas[i],
                    geom=None,
                )
            )
            for i in range(B)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


def test_parametric_handles_nonpositive_noise_plus_meas():
    """Negative radiance can drive noise+meas <= 0; both paths clamp to 1e-5."""
    rng = np.random.default_rng(4)
    noise = np.stack(
        [
            rng.uniform(0.001, 0.01, N_CHAN),
            np.full(N_CHAN, 0.5),
            rng.uniform(0.0, 0.05, N_CHAN),
        ],
        axis=1,
    )
    meas = np.full((B, N_CHAN), -2.0)

    got = _batched("parametric", noise=noise, integrations=1).Sy_diagonal(
        torch.as_tensor(meas)
    ).numpy()

    ref = np.stack(
        [
            np.diag(
                Instrument.Sy(
                    _scalar("parametric", noise=noise, integrations=1),
                    meas[i].copy(),
                    geom=None,
                )
            )
            for i in range(B)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


def test_parametric_superpixel_integrations_reduce_noise():
    """Superpixel means average ~segsize spectra, so noise falls as 1/sqrt(n)."""
    noise = np.stack(
        [np.full(N_CHAN, 0.01), np.full(N_CHAN, 0.5), np.zeros(N_CHAN)], axis=1
    )
    meas = _meas(5)
    one = _batched("parametric", noise=noise, integrations=1).Sy_diagonal(
        torch.as_tensor(meas)
    )
    forty = _batched("parametric", noise=noise, integrations=40).Sy_diagonal(
        torch.as_tensor(meas)
    )
    np.testing.assert_allclose((one / forty).numpy(), 40.0, rtol=1e-10)


# --- NEDT -------------------------------------------------------------------------


def test_nedt_matches_scalar():
    nesr = np.random.default_rng(6).uniform(0.01, 0.5, N_CHAN)
    meas = _meas(7)

    got = _batched("NEDT", noise_NESR=nesr, integrations=1).Sy_diagonal(
        torch.as_tensor(meas)
    ).numpy()

    ref = np.stack(
        [
            np.diag(
                Instrument.Sy(_scalar("NEDT", noise_NESR=nesr), meas[i], geom=None)
            )
            for i in range(B)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


# --- pushbroom --------------------------------------------------------------------


def test_pushbroom_is_full_matrix_shared_across_batch():
    rng = np.random.default_rng(8)
    A = rng.normal(size=(5, N_CHAN, N_CHAN))
    covs = A @ A.transpose(0, 2, 1)
    meas = _meas(9)

    ti = _batched("pushbroom", covs=covs, integrations=4)
    assert not ti.sy_is_diagonal

    ref = Instrument.Sy(
        _scalar("pushbroom", covs=covs, integrations=4), meas[0], geom=None
    )
    np.testing.assert_allclose(ti.Sy_shared.numpy(), ref, rtol=RTOL)

    got = ti.Sy(torch.as_tensor(meas)).numpy()
    assert got.shape == (B, N_CHAN, N_CHAN)
    for i in range(B):
        np.testing.assert_allclose(got[i], ref, rtol=RTOL)


def test_pushbroom_diagonal_accessor_raises():
    """Asking for a diagonal from a correlated model should fail loudly."""
    rng = np.random.default_rng(10)
    A = rng.normal(size=(3, N_CHAN, N_CHAN))
    ti = _batched("pushbroom", covs=A @ A.transpose(0, 2, 1), integrations=1)
    with pytest.raises(ValueError, match="not diagonal"):
        ti.Sy_diagonal(torch.as_tensor(_meas(11)))


# --- dense form and validation ----------------------------------------------------


def test_dense_Sy_matches_scalar_for_diagonal_models():
    snr = np.random.default_rng(12).uniform(50, 500, N_CHAN)
    meas = _meas(13)

    got = _batched("SNR", snr=snr, integrations=1).Sy(torch.as_tensor(meas)).numpy()
    ref = np.stack(
        [
            Instrument.Sy(_scalar("SNR", snr=snr), meas[i], geom=None)
            for i in range(B)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL)


def test_unknown_noise_model_rejected():
    with pytest.raises(ValueError, match="Unsupported instrument noise model"):
        _batched("magic", integrations=1)
