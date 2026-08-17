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
"""Batched linear algebra for the torch backend.

Two things live here that are easy to get subtly wrong:

1. **Eigenvalue stabilization.** ``isofit.core.common.svd_inv_sqrt`` does not
   invert covariances naively -- it eigendecomposes and, when the result has
   negative or NaN eigenvalues, retries with a progressively larger diagonal
   offset. Measurement covariances near the edges of retrieval windows are
   close to singular often enough that a plain ``torch.linalg.inv`` diverges.
   :func:`svd_inv_sqrt_batch` reproduces that ladder per pixel.

2. **LAPACK triangle conventions.** ``invert_analytical`` calls ``dpotrf`` /
   ``dpotri`` / ``dsymv`` directly, and those routines read and write only one
   triangle. The resulting arithmetic is *not* the textbook formula, so
   reproducing it requires modelling which triangle each step actually touched.
   See :func:`upper_read_sym` and :func:`whiten_innovation`.
"""

from __future__ import annotations

import logging

import torch

Logger = logging.getLogger(__name__)

#: Diagonal offsets tried, in order, when an eigendecomposition comes back with
#: negative or NaN eigenvalues (isofit/core/common.py:457).
INV_EPS_LADDER = (1e-6, 1e-5, 1e-4)


#: Maximum bytes of matrix data passed to a single batched hipBLAS call.
#: ROCm's ``hipblasXtrsmBatched`` fails with ``HIPBLAS_STATUS_ALLOC_FAILED`` once
#: ``batch * n * n * itemsize`` exceeds roughly 24 MB. This is NOT a VRAM
#: shortage -- it reproduces with 31 GiB free on a 32 GiB card. Measured on
#: ROCm 7.2.4 / gfx1201 (Radeon AI PRO R9700), n=425, float64:
#:     batch 16 -> 22.0 MB  OK
#:     batch 20 -> 27.6 MB  HIPBLAS_STATUS_ALLOC_FAILED
#: 16 MiB leaves margin below the observed cliff. CUDA is unaffected, but the
#: chunking is harmless there: it is the same op over disjoint slices.
_BATCHED_TRSM_MAX_BYTES = 16 * 1024**2


def _chunked_cholesky_inverse(L: torch.Tensor) -> torch.Tensor:
    """``torch.cholesky_inverse`` split so each hipBLAS call stays under the limit.

    Numerically identical to the unchunked call -- ``cholesky_inverse`` is
    independent per batch element, so slicing changes nothing but the call size.
    """
    batch = L.shape[0]
    per_matrix = L.shape[-1] * L.shape[-2] * L.element_size()
    chunk = max(1, _BATCHED_TRSM_MAX_BYTES // max(per_matrix, 1))
    if chunk >= batch:
        return torch.cholesky_inverse(L)
    return torch.cat(
        [torch.cholesky_inverse(L[i : i + chunk]) for i in range(0, batch, chunk)]
    )


def chol_inv_full(S: torch.Tensor) -> torch.Tensor:
    """Invert a batch of symmetric positive-definite matrices.

    Args:
        S: ``(B, n, n)`` SPD matrices.

    Returns:
        ``(B, n, n)`` full symmetric inverses.

    Notes:
        The CPU path reaches the same inverse via ``dpotrf``/``dpotri``, which
        leave the opposite triangle untouched (zero). This returns the *full*
        symmetric inverse; callers that need to mimic a one-triangle read must
        do so explicitly.
    """
    L, info = torch.linalg.cholesky_ex(S)
    failed = info != 0
    if bool(failed.any()):
        # torch.linalg.cholesky raises for the WHOLE batch when any element is
        # not positive definite, which would discard thousands of good pixels
        # for one bad one. The scalar path cannot fail this way: it discards
        # LAPACK's info (``dpotrf(P_rcond)[0]`` at inverse_simple.py:333) and
        # continues with whatever is in the factor, so a non-PD pixel there
        # yields garbage rather than an exception.
        #
        # Reproducing uninitialized LAPACK output is neither possible nor
        # desirable. Isolate the failures instead: the surviving pixels keep
        # their real answer, and the caller can mark the rest.
        n_failed = int(failed.sum())
        Logger.warning(
            f"Cholesky failed for {n_failed} of {S.shape[0]} pixels "
            "(matrix not positive definite). Their output is not meaningful; "
            "the scalar backend silently returns garbage for the same pixels."
        )
        # Substitute the identity so the batched factorization completes; the
        # affected pixels are reported through `chol_inv_full.last_failed`.
        eye = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device)
        L = torch.where(failed.view(-1, 1, 1), eye, L)
    chol_inv_full.last_failed = failed
    return _chunked_cholesky_inverse(L)


#: Per-pixel mask from the most recent :func:`chol_inv_full` call. Set on every
#: call so a caller can attribute failures without changing the return type.
chol_inv_full.last_failed = None


def upper_read_sym(X: torch.Tensor) -> torch.Tensor:
    """Symmetrize a batch of matrices from their upper triangles.

    ``dpotrf`` with its default ``lower=0`` reads only the upper triangle of its
    argument and implicitly treats the matrix as symmetric. When the matrix
    handed to it is *not* symmetric -- which happens in ``invert_analytical``
    because ``P_tilde`` is built from a one-triangle inverse -- the effective
    matrix being factorized is the one reconstructed here, not the input.

    Args:
        X: ``(B, n, n)`` matrices.

    Returns:
        ``(B, n, n)`` symmetric matrices sharing ``X``'s upper triangle.

    Notes:
        The reflected triangle is accumulated in place. The naive
        ``upper + triu(X, 1).T`` holds three ``(B, n, n)`` buffers alongside
        ``X`` at once, and this is called on the largest matrix in the solve --
        one extra ``n_surface**2`` buffer per pixel is ~1.4 MiB at
        ``n_surface=425``, a quarter of the whole per-pixel budget. The
        additions performed are identical, so the result is bit-for-bit the
        same.
    """
    upper = torch.triu(X)
    reflected = torch.triu(X, diagonal=1).transpose(-1, -2)
    return upper.add_(reflected)


def whiten_innovation(
    Seps_inv: torch.Tensor, y: torch.Tensor, strict_parity: bool = True
) -> torch.Tensor:
    """Apply the inverse measurement covariance to the innovation.

    Args:
        Seps_inv: ``(B, nw, nw)`` full symmetric inverse covariance.
        y: ``(B, nw)`` innovation (measured minus modeled radiance).
        strict_parity: When True (default), reproduce the CPU path exactly --
            which uses only the *diagonal* of ``Seps_inv``. When False, apply
            the full matrix.

    Returns:
        ``(B, nw)`` whitened innovation.

    Notes:
        The CPU code computes this as ``dsymv(1, P, y)`` where ``P`` came from
        ``dpotri(C, 1)`` (isofit/inversion/inverse_simple.py:336-341). ``dpotri``
        with ``lower=1`` writes the inverse into the lower triangle only, while
        ``dsymv`` defaults to ``lower=0`` and therefore reads the upper triangle
        -- which is all zeros apart from the diagonal. The product is thus
        ``diag(P) * y``, discarding every off-diagonal term of the measurement
        error covariance.

        This is reproduced by default so the batched backend matches current
        ISOFIT results. It is reported upstream as a suspected bug; if it is
        confirmed and fixed there, ``strict_parity`` should default to False.
    """
    if strict_parity:
        return torch.diagonal(Seps_inv, dim1=-2, dim2=-1) * y
    return torch.einsum("bij,bj->bi", Seps_inv, y)


def svd_inv_sqrt_batch(C: torch.Tensor, max_tries: int = None):
    """Batched stabilized inverse and inverse square root.

    Batched counterpart of :func:`isofit.core.common.svd_inv_sqrt`.

    Args:
        C: ``(B, n, n)`` symmetric matrices.
        max_tries: Unused; present for signature symmetry with the ladder length.

    Returns:
        ``(Cinv, Cinv_sqrt)``, each ``(B, n, n)``.

    Raises:
        ValueError: A matrix still had negative or NaN eigenvalues after every
            offset in :data:`INV_EPS_LADDER`.

    Notes:
        The retry ladder is applied only to the pixels that need it, rather than
        re-decomposing the whole batch: in a typical scene a small minority of
        pixels are ill-conditioned, and redoing thousands of good ones would
        dominate the cost.

        The hashtable cache used by the scalar implementation is intentionally
        dropped. It pays off when the same covariance recurs across calls, which
        does not happen once pixels are batched.
    """
    if C.ndim != 3 or C.shape[-1] != C.shape[-2]:
        raise ValueError(f"expected a batch of square matrices, got {tuple(C.shape)}")

    B, n = C.shape[0], C.shape[-1]
    eye = torch.eye(n, dtype=C.dtype, device=C.device)

    D, P = _safe_eigh(C)
    bad = _needs_retry(D)

    for inv_eps in INV_EPS_LADDER:
        if not bool(bad.any()):
            break
        idx = torch.nonzero(bad, as_tuple=True)[0]
        Dr, Pr = _safe_eigh(C.index_select(0, idx) + eye * inv_eps)
        still_bad = _needs_retry(Dr)

        D = D.index_copy(0, idx, Dr)
        P = P.index_copy(0, idx, Pr)
        bad = bad.clone()
        bad[idx] = still_bad

    if bool(bad.any()):
        n_bad = int(bad.sum())
        raise ValueError(
            f"Matrix inversion contains negative values for {n_bad} of {B} "
            f"matrices, even after adding {INV_EPS_LADDER[-1]} to the diagonal."
        )

    # L = P diag(1/sqrt(D)); Cinv = L Lᵀ; Cinv_sqrt = L Pᵀ (common.py:473-476)
    L = P * (1.0 / torch.sqrt(D)).unsqueeze(-2)
    Cinv = L @ L.transpose(-1, -2)
    Cinv_sqrt = L @ P.transpose(-1, -2)
    return Cinv, Cinv_sqrt


def _safe_eigh(C: torch.Tensor):
    """Eigendecomposition that degrades to NaN instead of raising.

    A batched ``eigh`` fails for the whole batch if any single matrix fails, so
    a hard failure is converted into NaN eigenvalues that the retry ladder can
    then handle per pixel.
    """
    try:
        D, P = torch.linalg.eigh(C)
    except Exception as exc:  # cuSOLVER raises for the entire batch
        Logger.debug(f"batched eigh failed ({exc}); falling back per matrix")
        Ds, Ps = [], []
        for i in range(C.shape[0]):
            try:
                d, p = torch.linalg.eigh(C[i])
            except Exception:
                d = torch.full(
                    (C.shape[-1],), float("nan"), dtype=C.dtype, device=C.device
                )
                p = torch.eye(C.shape[-1], dtype=C.dtype, device=C.device)
            Ds.append(d)
            Ps.append(p)
        D, P = torch.stack(Ds), torch.stack(Ps)
    return D, P


def _needs_retry(D: torch.Tensor) -> torch.Tensor:
    """True for each matrix whose eigenvalues are negative or NaN."""
    return ((D < 0) | torch.isnan(D)).any(dim=-1)
