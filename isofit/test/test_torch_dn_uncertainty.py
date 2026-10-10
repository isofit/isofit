"""The DN-linearity uncertainty must match the scalar path, or be refused.

``Instrument`` can fold an imperfect-linearity-correction term into either
``Sy`` or ``Sb``, selected by the ``embedding`` field of the ``dn_uncertainty``
.mat file (``isofit/core/instrument.py:88-127``). Both branches were wrong in
the batched backend, in opposite directions:

* ``Sy``: the scalar adds ``DN_additive_uncertainty`` **unsquared** to the
  variance diagonal (``instrument.py:305-320``). ``TorchInstrument.Sy_diagonal``
  squared it. Squaring is arguably the dimensionally sensible thing -- the term
  is in radiance units and is being added to a variance -- but this backend's
  contract is parity with the CPU path as it behaves today, the same contract
  that makes ``whiten_innovation(strict_parity=True)`` the default. Diverging
  "helpfully" is what the contract exists to prevent, so the dimensional
  question belongs upstream, not in a silent behavior change here.

* ``Sb``: the scalar adds ``DN_additive_uncertainty ** 2`` to the radiometric
  block (``instrument.py:245-255``). ``sb_diagonal`` dropped the term entirely
  while advertising in its own docstring that ``meas`` was "only used when the
  DN-linearity uncertainty is embedded in ``Sb``" -- and never reading ``meas``.
  Silently dropping a term of ``Sb`` understates observation error, which is the
  failure mode this PR refuses everywhere else, so it now raises.

Neither branch is reachable without an explicit ``dn_uncertainty_file``, so no
default ``apply_oe`` run is affected.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.torch_cpu

N_CHAN = 6


def _scalar_dn_additive(meas, rcc, interp, inflation):
    """Verbatim ``Instrument.DN_additive_uncertainty`` (instrument.py:479-483)."""
    dn_est = np.maximum(meas / rcc, 0)
    noise_est = interp(dn_est)
    return np.abs(meas * (noise_est - 1) * inflation)


class _Interp:
    """Stand-in for the scipy response-curve interpolator."""

    def __call__(self, dn):
        return 1.0 + 0.05 * np.sqrt(np.asarray(dn))


def _instrument(embedding):
    """A minimal Instrument-like object carrying the DN uncertainty fields."""
    from unittest.mock import MagicMock

    inst = MagicMock()
    inst.n_chan = N_CHAN
    inst.model_type = "SNR"
    inst.snr = np.full(N_CHAN, 300.0)
    inst.integrations = 1
    inst.bval = np.zeros(N_CHAN)
    inst.dn_uncertainty_embedding = embedding
    inst.dn_uncertainty_rcc = 2.0
    inst.dn_uncertainty_interp = _Interp()
    inst.dn_uncertainty_inflation = 1.5
    inst.unknowns = None
    return inst


def _torch_instrument(embedding):
    from isofit.backends.torch.instrument import TorchInstrument

    ti = TorchInstrument.__new__(TorchInstrument)
    inst = _instrument(embedding)
    ti.instrument = inst
    ti.device = torch.device("cpu")
    ti.dtype = torch.float64
    ti.model_type = "SNR"
    ti.snr = torch.as_tensor(inst.snr)
    ti.integrations = 1
    ti.sy_is_diagonal = True
    ti.dn_uncertainty_embedding = embedding
    return ti


MEAS = np.linspace(2.0, 9.0, N_CHAN)[None, :]


def test_dn_additive_matches_the_scalar():
    """The term itself must agree before the Sy/Sb question is meaningful."""
    ti = _torch_instrument("Sy")
    got = ti.dn_additive_uncertainty(torch.as_tensor(MEAS)).numpy()
    want = _scalar_dn_additive(MEAS, 2.0, _Interp(), 1.5)
    assert np.allclose(got, want, rtol=1e-12), "dn_additive_uncertainty diverges"


def test_sy_adds_the_term_unsquared_like_the_scalar():
    """Parity: instrument.py:305-320 adds it unsquared, so this must too."""
    ti = _torch_instrument("Sy")
    meas_t = torch.as_tensor(MEAS)

    got = ti.Sy_diagonal(meas_t).numpy()

    nedl = np.maximum(MEAS / 300.0, np.sqrt(1e-7))
    want = nedl**2 + _scalar_dn_additive(MEAS, 2.0, _Interp(), 1.5)

    assert np.allclose(got, want, rtol=1e-12), (
        "Sy_diagonal does not match the scalar; if it squares the DN term it "
        "will read high by orders of magnitude wherever the term is small"
    )


def test_sy_without_the_embedding_is_untouched():
    """The common case must not move."""
    ti = _torch_instrument(None)
    got = ti.Sy_diagonal(torch.as_tensor(MEAS)).numpy()
    nedl = np.maximum(MEAS / 300.0, np.sqrt(1e-7))
    assert np.allclose(got, nedl**2, rtol=1e-12)


def test_sb_embedding_is_refused_rather_than_dropped():
    """Dropping a term of Sb understates observation error; it must raise."""
    from isofit.backends.torch.seps import sb_diagonal

    inst = _instrument("Sb")
    with pytest.raises(NotImplementedError, match="Sb"):
        sb_diagonal(inst, torch.as_tensor(MEAS), dtype=torch.float64,
                    device=torch.device("cpu"))


def test_sy_embedding_does_not_block_sb_assembly():
    """Only the Sb embedding is refused; the Sy one flows through sb_diagonal."""
    from isofit.backends.torch.seps import sb_diagonal

    inst = _instrument("Sy")
    out = sb_diagonal(inst, torch.as_tensor(MEAS), dtype=torch.float64,
                      device=torch.device("cpu"))
    assert out.shape == (N_CHAN,)
