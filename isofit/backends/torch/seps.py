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
"""Batched observation-error covariance.

``Seps = Sy + Kb·Sb·Kbᵀ + Γ`` (:meth:`isofit.core.forward.ForwardModel.Seps`)
combines instrument noise, uncertainty from unretrieved unknowns, and an
optional model-discrepancy term.

Building it densely per pixel would be wasteful, because most of it is not
dense. ``Kb``'s radiometric block is ``diagflat(meas)``
(``isofit/core/instrument.py:355-359``), so its contribution to ``Kb·Sb·Kbᵀ`` is
the diagonal ``meas² · sb_rad``. Only a handful of columns are genuinely dense --
the H2O_ABSCO derivative and, when configured, the spectral-calibration and
stray-light terms -- and each contributes a rank-1 update. ``Γ`` is shared by the
whole batch.

So the assembly is: one diagonal, a few rank-1 outer products, and one shared
matrix.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

Logger = logging.getLogger(__name__)


def seps_batch(
    sy_diag: torch.Tensor,
    rdn_modeled: torch.Tensor,
    sb_radiometric: torch.Tensor,
    dense_columns: torch.Tensor = None,
    sb_dense: torch.Tensor = None,
    model_discrepancy: torch.Tensor = None,
    winidx: torch.Tensor = None,
    sy_full: torch.Tensor = None,
) -> torch.Tensor:
    """Assemble the observation-error covariance for a batch of pixels.

    Args:
        sy_diag: ``(B, n_chan)`` diagonal of the instrument noise covariance.
            Ignored when ``sy_full`` is given.
        rdn_modeled: ``(B, n_chan)`` MODELED radiance at the current state.
            This forms the radiometric ``Kb`` block. Note ``Instrument
            .dmeas_dinstrumentb`` names its local ``meas`` but assigns it
            ``self.sample(x_instrument, wl_hi, rdn_hi)`` -- the modeled
            radiance, not the measurement (isofit/core/instrument.py:353).
        sb_radiometric: ``(n_chan,)`` radiometric entries of ``diag(Sb)``.
        dense_columns: Optional ``(B, n_chan, n_dense)`` dense ``Kb`` columns
            (H2O_ABSCO, spectral calibration, stray light).
        sb_dense: Optional ``(n_dense,)`` matching ``Sb`` entries.
        model_discrepancy: Optional ``(n_chan, n_chan)`` shared ``Γ``.
        winidx: Optional ``(nw,)`` retrieval-window channels to restrict to.
            Applying this here avoids ever forming the full ``(B, n_chan,
            n_chan)`` matrix.
        sy_full: Optional ``(n_chan, n_chan)`` shared full noise covariance,
            for the pushbroom model.

    Returns:
        ``(B, nw, nw)`` when ``winidx`` is given, else ``(B, n_chan, n_chan)``.
    """
    B = rdn_modeled.shape[0]
    device, dtype = rdn_modeled.device, rdn_modeled.dtype

    if winidx is not None:
        rdn_w = rdn_modeled.index_select(1, winidx)
        sb_rad_w = sb_radiometric.index_select(0, winidx)
        n = winidx.numel()
    else:
        rdn_w = rdn_modeled
        sb_rad_w = sb_radiometric
        n = rdn_modeled.shape[1]

    # Diagonal part: instrument noise plus the radiometric Kb block.
    diag = rdn_w**2 * sb_rad_w.unsqueeze(0)
    if sy_full is None:
        sy_w = (
            sy_diag.index_select(1, winidx) if winidx is not None else sy_diag
        )
        diag = diag + sy_w

    out = torch.diag_embed(diag)

    if sy_full is not None:
        block = (
            sy_full.index_select(0, winidx).index_select(1, winidx)
            if winidx is not None
            else sy_full
        )
        out = out + block.unsqueeze(0)

    # Dense Kb columns: one rank-1 update each.
    if dense_columns is not None and dense_columns.shape[-1]:
        cols = (
            dense_columns.index_select(1, winidx)
            if winidx is not None
            else dense_columns
        )
        if sb_dense is None:
            raise ValueError("sb_dense is required when dense_columns is given")
        scaled = cols * sb_dense.view(1, 1, -1)
        out = out + scaled @ cols.transpose(-1, -2)

    if model_discrepancy is not None:
        gamma = (
            model_discrepancy.index_select(0, winidx).index_select(1, winidx)
            if winidx is not None
            else model_discrepancy
        )
        out = out + gamma.unsqueeze(0)

    return out


def sb_diagonal(instrument, meas: torch.Tensor, dtype=None, device=None):
    """Diagonal of ``Sb`` for a batch (:meth:`Instrument.Sb`).

    Args:
        instrument: A built :class:`isofit.core.instrument.Instrument`.
        meas: ``(B, n_chan)`` measured radiance. Only used when the DN-linearity
            uncertainty is embedded in ``Sb``.

    Returns:
        ``(n_bvec,)`` when the result is shared across the batch, or
        ``(B, n_bvec)`` when it depends on the measurement.

    Notes:
        The radiometric entries accumulate in quadrature and are floored before
        the square root, exactly as the scalar implementation does
        (``isofit/core/instrument.py:221-263``); the floor exists to keep a
        zero-uncertainty channel from producing a singular covariance.
    """
    from isofit.core.common import eps

    dtype = dtype or torch.float64
    device = device or meas.device

    bval = np.array(instrument.bval, dtype=float).copy()
    n_chan = int(instrument.n_chan)

    unknowns = getattr(instrument, "unknowns", None)
    if unknowns:
        f = getattr(unknowns, "channelized_radiometric_uncertainty_file", None)
        if f is not None:
            u = np.loadtxt(f, comments="#")
            if len(u.shape) > 0 and u.ndim > 1 and u.shape[1] > 1:
                u = u[:, 1]
            bval[:n_chan] = bval[:n_chan] + np.power(u, 2)

        u = getattr(unknowns, "uncorrelated_radiometric_uncertainty", None)
        if u:
            bval[:n_chan] = bval[:n_chan] + np.power(np.ones(n_chan) * u, 2)

    bval[:n_chan] = np.maximum(bval[:n_chan], np.ones(n_chan) * eps)
    bval[:n_chan] = np.sqrt(bval[:n_chan])

    return torch.as_tensor(bval**2, dtype=dtype, device=device)
