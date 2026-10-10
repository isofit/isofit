#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
#
"""Batched instrument noise model.

Batched counterpart of :meth:`isofit.core.instrument.Instrument.Sy`. Three of
the four noise models (SNR, parametric, NEDT) produce a *diagonal* covariance,
so this module returns the diagonal as a ``(B, n_chan)`` tensor and lets callers
add it to a covariance without ever materializing ``B`` dense diagonal matrices.
Only the pushbroom model needs a full matrix, and that one is shared across the
batch.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

Logger = logging.getLogger(__name__)

# Matches the noise floor in Instrument.Sy for the SNR model.
MINIMUM_NOISE = np.sqrt(1e-7)


class TorchInstrument:
    """Batched instrument noise for a fixed instrument model.

    Args:
        instrument: A built :class:`isofit.core.instrument.Instrument`.
        device: Torch device.
        dtype: Torch dtype.

    Attributes:
        sy_is_diagonal: True when :meth:`Sy_diagonal` is meaningful. False only
            for the pushbroom model, whose covariance is a full matrix shared by
            every pixel (see :attr:`Sy_shared`).
    """

    def __init__(self, instrument, device=None, dtype=torch.float64):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.instrument = instrument

        self.model_type = instrument.model_type
        self.n_chan = int(instrument.n_chan)
        self.integrations = float(instrument.integrations)
        self.dn_uncertainty_embedding = getattr(
            instrument, "dn_uncertainty_embedding", None
        )

        self.sy_is_diagonal = self.model_type != "pushbroom"
        self.Sy_shared = None

        if self.model_type == "SNR":
            self.snr = self._t(instrument.snr)
        elif self.model_type == "parametric":
            # Columns are the three coefficients of the parametric noise curve.
            self.noise = self._t(instrument.noise)
        elif self.model_type == "NEDT":
            self.noise_NESR = self._t(instrument.noise_NESR)
        elif self.model_type == "pushbroom":
            # Batch-shared constant: mean over cross-track positions.
            C = np.squeeze(np.asarray(instrument.covs).mean(axis=0))
            self.Sy_shared = self._t(C / np.sqrt(self.integrations))
        else:
            raise ValueError(f"Unsupported instrument noise model: {self.model_type!r}")

    def _t(self, array):
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)

    def Sy_diagonal(self, meas: torch.Tensor) -> torch.Tensor:
        """Diagonal of the measurement noise covariance.

        Args:
            meas: ``(B, n_chan)`` measured radiance.

        Returns:
            ``(B, n_chan)`` variances, i.e. ``diag(Sy)`` per pixel.

        Raises:
            ValueError: The configured model has a full (non-diagonal)
                covariance; use :attr:`Sy_shared` instead.
        """
        if not self.sy_is_diagonal:
            raise ValueError(
                f"The {self.model_type!r} noise model is not diagonal; "
                "use Sy_shared for the batch-shared full covariance"
            )

        if self.model_type == "SNR":
            nedl = (1.0 / self.snr).unsqueeze(0) * meas
            # Clamp non-positive noise to avoid dividing by zero downstream.
            nedl = torch.clamp(nedl, min=MINIMUM_NOISE)

        elif self.model_type == "parametric":
            noise_plus_meas = self.noise[:, 1].unsqueeze(0) + meas
            noise_plus_meas = torch.where(
                noise_plus_meas <= 0,
                torch.full_like(noise_plus_meas, 1e-5),
                noise_plus_meas,
            )
            nedl = torch.abs(
                self.noise[:, 0].unsqueeze(0) * torch.sqrt(noise_plus_meas)
                + self.noise[:, 2].unsqueeze(0)
            )
            nedl = nedl / np.sqrt(self.integrations)

        else:  # NEDT
            nedl = self.noise_NESR.unsqueeze(0).expand(meas.shape[0], -1)

        diag = nedl**2

        if self.dn_uncertainty_embedding:
            # UNSQUARED, matching Instrument.Sy (instrument.py:305-320), which
            # adds this straight onto the variance diagonal. The term is in
            # radiance units, so squaring is arguably what was meant -- but the
            # contract here is parity with the CPU path as it behaves today, the
            # same contract that makes whiten_innovation(strict_parity=True) the
            # default. Raise it upstream rather than diverging quietly.
            diag = diag + self.dn_additive_uncertainty(meas)

        return diag

    def dn_additive_uncertainty(self, meas: torch.Tensor) -> torch.Tensor:
        """Uncertainty from imperfect linearity correction.

        Mirrors :meth:`Instrument.DN_additive_uncertainty`. The response curve
        is a scipy interpolator, so this necessarily round-trips through numpy;
        it only runs when the option is enabled.
        """
        inst = self.instrument
        dn_est = np.maximum(meas.detach().cpu().numpy() / inst.dn_uncertainty_rcc, 0)
        noise_est = inst.dn_uncertainty_interp(dn_est)
        out = np.abs(
            meas.detach().cpu().numpy()
            * (noise_est - 1)
            * inst.dn_uncertainty_inflation
        )
        return torch.as_tensor(out, dtype=self.dtype, device=self.device)

    def Sy(self, meas: torch.Tensor) -> torch.Tensor:
        """Full measurement noise covariance, ``(B, n_chan, n_chan)``.

        Provided for parity testing and for callers that genuinely need the
        dense form. Prefer :meth:`Sy_diagonal` in batched code: at 425 channels
        and a batch of 512 the dense form is ~740 MiB of mostly zeros.
        """
        if self.sy_is_diagonal:
            return torch.diag_embed(self.Sy_diagonal(meas))

        B = meas.shape[0]
        Sy = self.Sy_shared.unsqueeze(0).expand(B, -1, -1).clone()
        if self.dn_uncertainty_embedding:
            extra = self.dn_additive_uncertainty(meas) ** 2
            Sy = Sy + torch.diag_embed(extra)
        return Sy
