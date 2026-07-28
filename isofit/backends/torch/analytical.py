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
"""Batched analytical (conditional MAP) surface retrieval.

Batched counterpart of :func:`isofit.inversion.inverse_simple.invert_analytical`
-- the "inner loop" of Susiluoto et al. (2025) that ``analytical_line`` runs once
per pixel over a whole image. The atmosphere is fixed per pixel, so the update is
a closed-form MAP solve rather than an iterative optimization, and every pixel
has identical matrix shapes. That makes it the most naturally batched stage in
ISOFIT.

Structure exploited here
------------------------
For a multicomponent surface the linearization ``H`` is diagonal
(``surface_multicomp.py:362-384``), so ``L = H[winidx][:, iv_idx]`` has a single
nonzero per row. Forming it densely would cost ``(B, nw, ns)`` -- roughly 500 MiB
at ``B=512`` for EMIT-sized data -- and the subsequent GEMMs would be almost
entirely multiplications by zero. Instead the two places ``L`` appears are
computed directly:

* ``Lᵀ · Seps⁻¹ · L``  -> an outer-product scaling of ``Seps⁻¹``, scattered to
  the retrieval-window positions.
* ``Lᵀ · z``           -> an elementwise scale, scattered the same way.

Both are ``O(B·nw²)`` elementwise work with no dense ``(nw, ns)`` intermediate.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from isofit.backends.torch.linalg import chol_inv_full, upper_read_sym, whiten_innovation

Logger = logging.getLogger(__name__)


def invert_analytical_batch(
    surface,
    winidx: torch.Tensor,
    meas: torch.Tensor,
    x0: torch.Tensor,
    theta: torch.Tensor,
    Seps: torch.Tensor,
    L_atm: torch.Tensor,
    L_bg: torch.Tensor,
    eof_offset: torch.Tensor,
    geom=None,
    idx_surface: torch.Tensor = None,
    outside_ret_windows: torch.Tensor = None,
    num_iter: int = 1,
    diag_uncert: bool = True,
    outside_ret_const: float = -0.01,
    strict_parity: bool = True,
):
    """Conditional MAP surface retrieval for a batch of pixels.

    Args:
        surface: :class:`TorchMultiComponentSurface` supplying the prior.
        winidx: ``(nw,)`` retrieval-window channel indices.
        meas: ``(B, n_meas)`` measured radiance.
        x0: ``(B, n_state)`` initial full state vector.
        theta: ``(B, n_wl)`` surface linearization coefficients, i.e. the
            diagonal of ``H``.
        Seps: ``(B, nw, nw)`` measurement covariance, already windowed.
        L_atm: ``(B, n_meas)`` atmospheric path radiance.
        L_bg: ``(B, n_meas)`` background radiance contribution.
        eof_offset: ``(B, n_meas)`` instrument EOF offset.
        geom: :class:`BatchedGeometry`, used for frozen component selection.
        idx_surface: ``(n_surface,)`` indices of the surface block within the
            state vector. Defaults to the leading ``n_surface`` entries.
        outside_ret_windows: Surface indices outside the retrieval windows,
            which are filled with ``outside_ret_const`` rather than solved.
        num_iter: MAP iterations (default 1, as in ``analytical_line``).
        diag_uncert: Return the diagonal posterior standard deviation.
        outside_ret_const: Fill value outside the retrieval windows. ``None``
            uses the prior mean instead.
        strict_parity: Reproduce the CPU path's diagonal-only whitening of the
            innovation. See :func:`isofit.backends.torch.linalg.whiten_innovation`.

    Returns:
        ``(trajectory, uncertainty)`` where trajectory is
        ``(B, num_iter + 1, n_state)`` and uncertainty is ``(B, n_state)``.

    Raises:
        NotImplementedError: The surface carries non-reflectance state elements
            (e.g. the glint model). Those add dense columns to ``L`` that the
            diagonal fast path does not cover.
    """
    B, n_state = x0.shape
    device, dtype = x0.device, x0.dtype

    n_surface = surface.n_state
    if idx_surface is None:
        idx_surface = torch.arange(n_surface, dtype=torch.int64, device=device)

    iv_idx = torch.arange(n_surface, dtype=torch.int64, device=device)
    if n_surface != surface.n_wl:
        raise NotImplementedError(
            "invert_analytical_batch currently supports reflectance-only surface "
            f"states (got n_state={n_surface} vs n_wl={surface.n_wl}). Surfaces "
            "with extra non-reflectance elements, such as the glint model, add "
            "dense columns to L that the diagonal fast path does not handle."
        )

    x = x0.clone()
    trajectory = torch.empty((B, num_iter + 1, n_state), dtype=dtype, device=device)
    trajectory[:, 0, :] = x

    theta_w = theta.index_select(1, winidx)  # (B, nw)

    # Innovation: measured minus every modeled contribution that does not depend
    # on the surface state being solved for.
    y = (
        meas.index_select(1, winidx)
        - L_atm.index_select(1, winidx)
        - eof_offset.index_select(1, winidx)
        - L_bg.index_select(1, winidx)
    )

    C_rcond = None
    for _ in range(num_iter):
        x_surface = x.index_select(1, idx_surface)

        ci = surface.component(x_surface, geom)
        xa_surface = surface.xa(x_surface, geom, ci=ci)
        scale = surface.prior_scale(x_surface, ci)

        # Seps⁻¹, full and symmetric.
        P_sym = chol_inv_full(Seps)

        # Data term. Lᵀ·Seps⁻¹·L for diagonal L is an outer-product scaling of
        # Seps⁻¹ restricted to the retrieval windows.
        blk = (theta_w.unsqueeze(-1) * theta_w.unsqueeze(-2)) * P_sym  # (B, nw, nw)

        A = torch.zeros((B, n_surface, n_surface), dtype=dtype, device=device)
        rows = winidx.view(-1, 1).expand(-1, winidx.numel())
        cols = winidx.view(1, -1).expand(winidx.numel(), -1)
        A[:, rows, cols] = blk

        # Prior precision, gathered per component and un-normalized.
        surface.add_Sa_inv(A, ci, scale=scale)

        # dpotrf reads only the upper triangle; model that explicitly so any
        # asymmetry introduced above is resolved the same way LAPACK would.
        A = upper_read_sym(A)
        C_rcond = chol_inv_full(A)

        # Right-hand side: whitened innovation scattered back to channel
        # positions, plus the prior term.
        z = whiten_innovation(P_sym, y, strict_parity=strict_parity)
        rhs = torch.zeros((B, n_surface), dtype=dtype, device=device)
        rhs.index_add_(1, winidx, theta_w * z)

        prprod = torch.zeros((B, n_surface), dtype=dtype, device=device)
        prprod = _prior_product(surface, ci, xa_surface, scale, prprod)
        rhs = rhs + prprod

        xk = (C_rcond @ rhs.unsqueeze(-1)).squeeze(-1)

        x_surface = x_surface.clone()
        x_surface[:, iv_idx] = xk
        if outside_ret_windows is not None and outside_ret_windows.numel():
            if outside_ret_const is None:
                x_surface[:, outside_ret_windows] = xa_surface[:, outside_ret_windows]
            else:
                x_surface[:, outside_ret_windows] = outside_ret_const

        x = x.clone()
        x[:, idx_surface] = x_surface
        trajectory[:, _ + 1, :] = x

    unc = torch.ones((B, n_state), dtype=dtype, device=device)
    if diag_uncert and C_rcond is not None:
        unc[:, iv_idx] = torch.sqrt(
            torch.diagonal(C_rcond, dim1=-2, dim2=-1).clamp(min=0)
        )
        return trajectory, unc
    return trajectory, C_rcond


def _prior_product(surface, ci, xa_surface, scale, out):
    """``Sa_inv @ xa`` per pixel, looping over components.

    Same rationale as :meth:`TorchMultiComponentSurface.add_Sa_inv`: gathering a
    per-pixel ``(B, n, n)`` prior would dominate memory, while there are only a
    handful of distinct components.
    """
    n = out.shape[1]
    for c in range(surface.n_comp):
        mask = ci == c
        if not bool(mask.any()):
            continue
        block = surface.Sa_inv_normalized[c, :n, :n]
        vals = xa_surface[mask] @ block.T
        out[mask] = vals / scale[mask].unsqueeze(1)
    return out
