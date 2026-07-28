"""Parity tests for the batched linear algebra.

Two things are pinned down here:

* :func:`svd_inv_sqrt_batch` against ``isofit.core.common.svd_inv_sqrt``,
  including matrices constructed to land on each rung of the stabilization
  ladder and one that should fall off the end of it.
* The LAPACK triangle conventions in ``invert_analytical``, against the literal
  ``dpotrf``/``dpotri``/``dsymv`` call sequence.
"""

import numpy as np
import pytest
import torch
from scipy.linalg.blas import dsymv
from scipy.linalg.lapack import dpotrf, dpotri

from isofit.backends.torch.linalg import (
    INV_EPS_LADDER,
    chol_inv_full,
    svd_inv_sqrt_batch,
    upper_read_sym,
    whiten_innovation,
)
from isofit.core.common import svd_inv_sqrt

pytestmark = pytest.mark.torch_cpu

RTOL = 1e-8
N = 24
B = 8


def _spd(n=N, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return (A @ A.T + n * np.eye(n)) * scale


def _spd_batch(b=B, n=N, seed=0):
    return np.stack([_spd(n, seed + i) for i in range(b)])


# --- chol_inv_full ---------------------------------------------------------------


def test_chol_inv_full_inverts():
    S = _spd_batch()
    inv = chol_inv_full(torch.as_tensor(S)).numpy()
    for i in range(B):
        np.testing.assert_allclose(inv[i] @ S[i], np.eye(N), atol=1e-9)


def test_chol_inv_full_is_symmetric():
    inv = chol_inv_full(torch.as_tensor(_spd_batch())).numpy()
    np.testing.assert_allclose(inv, inv.transpose(0, 2, 1), rtol=1e-12)


# --- svd_inv_sqrt ----------------------------------------------------------------


def test_svd_inv_sqrt_matches_scalar():
    S = _spd_batch()
    Cinv, Cinv_sqrt = svd_inv_sqrt_batch(torch.as_tensor(S))

    for i in range(B):
        ref_inv, ref_sqrt = svd_inv_sqrt(S[i])
        np.testing.assert_allclose(Cinv[i].numpy(), ref_inv, rtol=RTOL, atol=1e-12)
        # eigenvectors are sign/order ambiguous, so compare the products the
        # callers actually use rather than the factor itself
        np.testing.assert_allclose(
            Cinv_sqrt[i].numpy() @ Cinv_sqrt[i].numpy().T,
            ref_sqrt @ ref_sqrt.T,
            rtol=RTOL,
            atol=1e-12,
        )


def test_inv_sqrt_squares_to_inverse():
    """The defining property: Cinv_sqrt @ Cinv_sqrt == Cinv."""
    S = _spd_batch()
    Cinv, Cinv_sqrt = svd_inv_sqrt_batch(torch.as_tensor(S))
    np.testing.assert_allclose(
        (Cinv_sqrt @ Cinv_sqrt).numpy(), Cinv.numpy(), rtol=1e-8, atol=1e-12
    )


def test_negative_eigenvalue_matrix_is_stabilized():
    """A matrix with a small negative eigenvalue must climb the eps ladder."""
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(N, N)))
    eigs = np.linspace(1.0, 5.0, N)
    eigs[0] = -1e-9  # just negative: fixed by the first rung
    S = Q @ np.diag(eigs) @ Q.T
    S = (S + S.T) / 2

    batch = np.stack([S] + [_spd(N, 10 + i) for i in range(3)])
    Cinv, _ = svd_inv_sqrt_batch(torch.as_tensor(batch))

    ref_inv, _ = svd_inv_sqrt(S)
    np.testing.assert_allclose(Cinv[0].numpy(), ref_inv, rtol=1e-6, atol=1e-8)


def test_only_bad_matrices_are_retried():
    """Well-conditioned members of the batch are untouched by the ladder."""
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.normal(size=(N, N)))
    eigs = np.linspace(1.0, 5.0, N)
    eigs[0] = -1e-9
    bad = Q @ np.diag(eigs) @ Q.T
    bad = (bad + bad.T) / 2

    good = _spd(N, 21)
    batch = np.stack([good, bad, good])

    Cinv, _ = svd_inv_sqrt_batch(torch.as_tensor(batch))
    ref_good, _ = svd_inv_sqrt(good)

    # The good matrices must match the unstabilized reference exactly; if the
    # ladder had been applied to the whole batch they would be offset.
    np.testing.assert_allclose(Cinv[0].numpy(), ref_good, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Cinv[2].numpy(), ref_good, rtol=1e-10, atol=1e-12)


def test_hopeless_matrix_raises():
    """Large negative eigenvalues survive every rung and must raise."""
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.normal(size=(N, N)))
    eigs = np.linspace(1.0, 5.0, N)
    eigs[0] = -10.0  # far beyond the largest offset
    S = Q @ np.diag(eigs) @ Q.T
    S = (S + S.T) / 2

    with pytest.raises(ValueError, match="negative values"):
        svd_inv_sqrt_batch(torch.as_tensor(np.stack([S, _spd(N, 6)])))


def test_error_names_the_final_offset():
    """The message should say how far the ladder got, as the scalar one does."""
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.normal(size=(N, N)))
    eigs = np.linspace(1.0, 5.0, N)
    eigs[0] = -10.0
    S = Q @ np.diag(eigs) @ Q.T
    S = (S + S.T) / 2
    with pytest.raises(ValueError, match=str(INV_EPS_LADDER[-1])):
        svd_inv_sqrt_batch(torch.as_tensor(S[None]))


def test_rejects_non_square_input():
    with pytest.raises(ValueError, match="batch of square matrices"):
        svd_inv_sqrt_batch(torch.zeros((4, 3, 5), dtype=torch.float64))


# --- LAPACK triangle conventions -------------------------------------------------


def test_whiten_innovation_reproduces_dsymv_quirk():
    """The CPU data term uses only diag(Seps_inv); parity mode must match it.

    ``dpotri(C, 1)`` fills the lower triangle only, and ``dsymv`` defaults to
    reading the upper triangle, so the product collapses to the diagonal. See
    isofit/inversion/inverse_simple.py:336-341.
    """
    rng = np.random.default_rng(8)
    for i in range(4):
        S = _spd(N, 30 + i)
        y = rng.normal(size=N)

        C = dpotrf(S, 1)[0]
        P = dpotri(C, 1)[0]
        ref = dsymv(1, P, y)

        Seps_inv = chol_inv_full(torch.as_tensor(S[None]))
        got = whiten_innovation(Seps_inv, torch.as_tensor(y[None]))[0].numpy()
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_whiten_innovation_strict_parity_is_not_the_full_product():
    """Guard against 'fixing' the quirk by accident: the two must differ."""
    S = _spd(N, 40)
    y = np.random.default_rng(9).normal(size=N)
    Seps_inv = chol_inv_full(torch.as_tensor(S[None]))
    yt = torch.as_tensor(y[None])

    strict = whiten_innovation(Seps_inv, yt, strict_parity=True)[0].numpy()
    full = whiten_innovation(Seps_inv, yt, strict_parity=False)[0].numpy()

    assert not np.allclose(strict, full), "strict parity should discard off-diagonals"
    np.testing.assert_allclose(full, np.linalg.inv(S) @ y, rtol=1e-8, atol=1e-10)


def test_upper_read_sym_matches_what_dpotrf_factorizes():
    """dpotrf reads only the upper triangle of a non-symmetric input."""
    rng = np.random.default_rng(10)
    A = rng.normal(size=(N, N))
    M = A @ A.T + N * np.eye(N)
    # Perturb the lower triangle only: dpotrf(upper) must be blind to it.
    M_perturbed = M.copy()
    M_perturbed[np.tril_indices(N, -1)] += 5.0

    ref = dpotrf(M_perturbed)[0]
    same = dpotrf(M)[0]
    np.testing.assert_allclose(ref, same, rtol=1e-12)

    sym = upper_read_sym(torch.as_tensor(M_perturbed[None]))[0].numpy()
    np.testing.assert_allclose(sym, np.triu(M) + np.triu(M, 1).T, rtol=1e-12)
    np.testing.assert_allclose(sym, sym.T, rtol=1e-12)


def test_full_analytical_chain_matches_lapack_sequence():
    """End-to-end: the whole dpotrf/dpotri/dsymv chain, batched.

    Mirrors inverse_simple.py:327-341 for a multicomponent-style diagonal H,
    which is the shape that actually occurs in analytical_line.
    """
    rng = np.random.default_rng(11)
    nw, ns = 10, 14
    winidx = np.array([0, 1, 3, 4, 6, 7, 8, 10, 11, 13])
    theta = rng.uniform(0.5, 2.0, ns)

    L = np.zeros((nw, ns))
    for i, w in enumerate(winidx):
        L[i, w] = theta[w]

    Seps = _spd(nw, 50)
    Sa_inv = _spd(ns, 51)
    y = rng.normal(size=nw)
    prprod = rng.normal(size=ns)

    # --- scalar reference, literal LAPACK calls
    C = dpotrf(Seps, 1)[0]
    P = dpotri(C, 1)[0]
    P_tilde = ((L.T @ P) @ L).T
    P_rcond = Sa_inv + P_tilde
    C_rcond = dpotri(dpotrf(P_rcond)[0])[0]
    ref = dsymv(1, C_rcond, (L.T @ dsymv(1, P, y) + prprod))

    # --- batched
    Seps_t = torch.as_tensor(Seps[None])
    P_sym = chol_inv_full(Seps_t)
    z = whiten_innovation(P_sym, torch.as_tensor(y[None]))
    L_t = torch.as_tensor(L[None])

    M = (L_t.transpose(-1, -2) @ torch.tril(P_sym)) @ L_t
    A = upper_read_sym(torch.as_tensor(Sa_inv[None]) + M.transpose(-1, -2))
    C_rcond_t = chol_inv_full(A)

    rhs = (L_t.transpose(-1, -2) @ z.unsqueeze(-1)).squeeze(-1) + torch.as_tensor(
        prprod[None]
    )
    got = (C_rcond_t @ rhs.unsqueeze(-1)).squeeze(-1)[0].numpy()

    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10)
