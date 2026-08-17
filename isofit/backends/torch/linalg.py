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


# ---------------------------------------------------------------------------
# ROCm workaround: batched Cholesky inverse via rocSOLVER potri
#
# torch.cholesky_inverse dispatches to rocblas_batched_dtrsm, which fails with
# HIPBLAS_STATUS_ALLOC_FAILED once batch*n*n*itemsize exceeds ~44 MB on gfx1201
# (ROCm 7.2.4). At n=425 that caps the batch at 16 (float64) or 32 (float32) --
# regardless of free VRAM (reproduced with 19.8 GiB free). No environment knob
# (ROCBLAS_DEVICE_MEMORY_SIZE, ROCBLAS_STREAM_ORDER_ALLOC,
# TORCH_BLAS_PREFER_HIPBLASLT), linalg backend, or reformulation
# (cholesky_solve, linalg.solve, solve_triangular) avoids it.
#
# rocSOLVER's potri_strided_batched does not use that path and scales cleanly to
# batch 2048+. It is also the routine the scalar backend already relies on
# (LAPACK dpotri), so this is the semantically matching primitive, not a hack.
#
# Measured, n=425, float64: 28.8 ms/matrix chunked -> 0.32 ms/matrix (90x).
# ---------------------------------------------------------------------------
_ROCSOLVER = {"tried": False, "lib": None, "handles": {}}


def _rocsolver():
    """Lazily load rocSOLVER and create a rocBLAS handle. Returns None if unusable."""
    if _ROCSOLVER["tried"]:
        return _ROCSOLVER["lib"]
    _ROCSOLVER["tried"] = True
    if getattr(torch.version, "hip", None) is None:
        return None  # CUDA build: torch's own path is fine
    try:
        import ctypes

        lib = ctypes.CDLL("librocsolver.so")
        rb = ctypes.CDLL("librocblas.so")
        rb.rocblas_create_handle.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        # A SEPARATE handle per precision. Sharing one handle across dpotri and
        # spotri corrupts rocSOLVER state: a float32 call issued after a float64
        # call on the same handle silently returns NaN (reproduced on ROCm 7.2.4
        # / gfx1201 -- fp32 alone is fine, fp32-after-fp64 is not).
        handles = {}
        for key in (torch.float32, torch.float64):
            h = ctypes.c_void_p()
            if rb.rocblas_create_handle(ctypes.byref(h)) != 0:
                return None
            handles[key] = h
        for name in ("rocsolver_dpotri_strided_batched", "rocsolver_spotri_strided_batched"):
            fn = getattr(lib, name)
            fn.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_int,
            ]
            fn.restype = ctypes.c_int
        _ROCSOLVER["lib"] = lib
        _ROCSOLVER["handles"] = handles
        Logger.debug("rocSOLVER potri available for batched Cholesky inverse")
        return lib
    except Exception as e:  # pragma: no cover - depends on local ROCm install
        Logger.debug(f"rocSOLVER unavailable ({e}); using chunked torch path")
        return None


_ROCBLAS_FILL_LOWER = 122


def _rocsolver_cholesky_inverse(L: torch.Tensor):
    """Full symmetric inverse from a lower Cholesky factor, via rocSOLVER potri.

    Returns None if rocSOLVER cannot service this call, so the caller can fall back.
    """
    lib = _rocsolver()
    if lib is None or L.ndim != 3 or not L.is_cuda:
        return None
    if L.dtype == torch.float64:
        fn = lib.rocsolver_dpotri_strided_batched
    elif L.dtype == torch.float32:
        fn = lib.rocsolver_spotri_strided_batched
    else:
        return None

    import ctypes

    b, n, _ = L.shape
    if b == 0:
        return L.clone()
    # rocSOLVER is column-major. A row-major L^T is exactly L in column-major,
    # so transpose+contiguous hands it a lower-triangular factor as it expects.
    #
    # CAREFUL: torch.linalg.cholesky_ex returns L in COLUMN-MAJOR layout, which
    # makes L.transpose(-1, -2) already contiguous -- so .contiguous() is a no-op
    # that returns the SAME storage. potri is in-place, so without the explicit
    # clone below it silently corrupts the caller's L. That produced a correct
    # result here and garbage (1e9 relative error) in any later consumer of L.
    work = L.transpose(-1, -2).contiguous()
    if work.data_ptr() == L.data_ptr():
        work = work.clone()
    info = torch.zeros(b, dtype=torch.int32, device=L.device)
    status = fn(
        _ROCSOLVER["handles"][L.dtype], _ROCBLAS_FILL_LOWER, n,
        ctypes.c_void_p(work.data_ptr()), n, n * n,
        ctypes.c_void_p(info.data_ptr()), b,
    )
    if status != 0:
        Logger.debug(f"rocSOLVER potri returned status {status}; falling back")
        return None
    if bool((info != 0).any()):
        Logger.debug("rocSOLVER potri reported singular factors; falling back")
        return None
    # potri wrote the inverse into the same triangle; mirror it to full symmetric.
    out = work.transpose(-1, -2)
    result = torch.tril(out) + torch.tril(out, -1).transpose(-1, -2)
    # Guard: rocSOLVER has been observed returning status 0 with NaN output under
    # handle-state corruption. Never hand a silently-wrong inverse to a retrieval.
    if not bool(torch.isfinite(result).all()):
        Logger.warning("rocSOLVER potri returned non-finite values; falling back")
        return None
    return result

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
        # Stabilize with the same diagonal-offset ladder ISOFIT uses elsewhere
        # for exactly this situation (`isofit.core.common.svd_inv_sqrt` retries
        # eigh on ``C + I*eps`` for eps in 1e-6, 1e-5, 1e-4), rather than
        # substituting a matrix unrelated to the input.
        #
        # Why not substitute the identity: Seps^-1 is on the order of
        # 1/noise-variance, i.e. large. Identity makes it ~1, so every
        # downstream term weighted by Seps^-1 is wrong by ten orders of
        # magnitude. Measured on a marginally non-PD covariance (LAPACK
        # info=58): scalar gives mean |Seps^-1| = 2.36e+09, identity gives
        # 1.67e-02 - a factor of 7e10. On a 100,000-pixel AVIRIS-NG scene that
        # produced 91 pixels with reflectances up to 3.0e+05 where the scalar
        # backend stayed inside [0, 1].
        #
        # Why not keep the raw partial factor either: its undefined region
        # drives the downstream posterior solve to +/-inf. Measured on the same
        # scene: 191 bad pixels and a non-finite cube. Worse than the identity.
        #
        # The ladder is the only one of the three that yields a genuinely valid
        # inverse. Pixels that remain non-PD after the largest offset keep the
        # identity fallback so the batch still completes, and every failure is
        # reported through `chol_inv_full.last_failed`.
        n_failed = int(failed.sum())
        eye = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device)
        still = failed.clone()
        for inv_eps in (1e-6, 1e-5, 1e-4):
            if not bool(still.any()):
                break
            idx = still.nonzero(as_tuple=True)[0]
            L_try, info_try = torch.linalg.cholesky_ex(S[idx] + eye * inv_eps)
            ok = info_try == 0
            if bool(ok.any()):
                good_idx = idx[ok]
                L = L.index_copy(0, good_idx, L_try[ok])
                still[good_idx] = False
        n_rescued = n_failed - int(still.sum())
        Logger.warning(
            f"Cholesky failed for {n_failed} of {S.shape[0]} pixels "
            f"(matrix not positive definite); {n_rescued} recovered by the "
            f"diagonal-offset ladder, {int(still.sum())} unrecoverable. "
            "See chol_inv_full.last_failed to identify them."
        )
        if bool(still.any()):
            L = torch.where(still.view(-1, 1, 1), eye, L)
    chol_inv_full.last_failed = failed
    out = _rocsolver_cholesky_inverse(L)
    if out is not None:
        return out
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
