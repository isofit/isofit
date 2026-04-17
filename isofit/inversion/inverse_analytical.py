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
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
from __future__ import annotations

from typing import OrderedDict

import numpy as np
from scipy.linalg.blas import dsymv
from scipy.linalg.lapack import dpotrf, dpotri

from isofit.core.common import eps


def map_solve(L, P, prprod, Sa_inv, y, iv_idx):
    P_tilde = ((L.T @ P) @ L).T
    P_rcond = Sa_inv[iv_idx, :][:, iv_idx] + P_tilde

    LI_rcond = dpotrf(P_rcond)[0]
    C_rcond = dpotri(LI_rcond)[0]

    xk = dsymv(
        1,
        C_rcond,
        (L.T @ dsymv(1, P, y) + prprod[iv_idx]),
    )

    return xk, C_rcond


def map_solve_constrained(
    L: np.ndarray,
    iv_idx: np.ndarray,
    Sa_inv: np.ndarray,
    P: np.ndarray,
    y: np.ndarray,
    prprod: np.ndarray,
    bounds: np.ndarray,
    n_max_fixed: int,
    chunk_n: int = -1,
):
    """More complex solver that iteratively enforces bounds on AOE solutions.
    It does this by "freezing" the index of the n-"greatest" bounds violation.
    It uses a for loop to deal with the covariance constraint between statevector elements.
    """
    n = len(iv_idx)
    fixed_set = np.zeros((n, 2), dtype=bool)
    free_mask = np.min(~fixed_set, axis=1)
    y_constrained = y.copy()
    Py = dsymv(1, P, y_constrained)
    P_tilde_0 = (L.T @ P) @ L
    for _ in range(n_max_fixed + 1):
        iv_free = iv_idx[free_mask]
        L_free = L[:, free_mask]
        free_local = np.where(free_mask)[0]

        # Don't need to recompute every iter. Can just slice.
        P_tilde = P_tilde_0[np.ix_(free_local, free_local)]
        P_rcond = Sa_inv[np.ix_(iv_free, iv_free)] + P_tilde
        LI = dpotrf(P_rcond)[0]
        C_free = dpotri(LI)[0]

        xk_free = dsymv(
            1,
            C_free,
            (L_free.T @ Py + prprod[iv_free]),
        )

        # Reconstruct full solution
        xk = np.zeros(n)
        xk[free_mask] = xk_free
        xk[~free_mask] = bounds[fixed_set]

        # Bounds check
        violations = np.zeros((n, 2), dtype=bool)
        violations[xk < bounds[:, 0], 0] = True
        violations[xk > bounds[:, 1], 1] = True

        # Found solution
        if not np.any(violations):
            break

        # Replace the n-worst offenders
        violation_magnitudes = violations * np.concatenate(
            [
                ((bounds[:, 0] - xk) / (bounds[:, 0] + eps) ** 2)[..., None],
                ((xk - bounds[:, 1]) / (bounds[:, 1] + eps) ** 2)[..., None],
            ],
            axis=1,
        )

        n_violations = np.sum(violation_magnitudes > 0)
        if chunk_n == -1 or chunk_n >= n_violations:
            worst = np.where(violation_magnitudes > 0)
        else:
            worst = np.unravel_index(
                np.argpartition(violation_magnitudes.ravel(), -chunk_n)[-chunk_n:],
                violations.shape,
            )
        fixed_set[worst[0], worst[1]] = True
        free_mask = ~np.any(fixed_set, axis=1)

        # Subtract active (fixed) variable contributions from y
        delta = L[:, worst[0]] @ bounds[worst[0], worst[1]]
        y_constrained -= delta
        Py = dsymv(1, P, y_constrained)

    C_rcond = np.zeros((n, n))
    C_rcond[np.ix_(free_mask, free_mask)] = C_free
    # Set the posterior uncertainty to prior variance in the fixed cases
    fixed_mask = ~free_mask
    C_rcond[fixed_mask, fixed_mask] = (
        1.0 / Sa_inv[iv_idx[fixed_mask], iv_idx[fixed_mask]]
    )

    return xk, C_rcond, fixed_mask


def invert_analytical(
    fm: ForwardModel,
    winidx: np.array,
    meas: np.array,
    geom: Geometry,
    x0: np.array,
    sub_state,
    num_iter: int = 1,
    hash_table: OrderedDict = None,
    hash_size: int = None,
    diag_uncert: bool = True,
    outside_ret_const: float = -0.01,
    fill_value: float = -9999.0,
    enforce_bounds: bool = False,
):
    """Perform an analytical estimate of the conditional MAP estimate for
    a fixed atmosphere.  Based on the "Inner loop" from Susiluoto et al. (2025).
    doi: https://doi.org/10.3390/rs17223719

    Args:
        fm: isofit forward model
        winidx: indices of surface components of state vector (to be solved)
        meas: a one-D numpy vector of radiance in uW/nm/sr/cm2
        geom: geometry object corresponding to given measurement
        x0: the initialization state including surface from the superpixel
            and the atm from the smoothed atmosphere.
        num_iter: number of interations to run through
        hash_table: a hash table to use locally
        hash_size: max size of given hash table
        diag_uncert: flag indicating whether to diagonalize the uncertainty

    Returns:
        x: MAP estimate of the mean
        S: diagonal conditional posterior covariance estimate
    """
    from scipy.linalg.blas import dsymv
    from scipy.linalg.lapack import dpotrf, dpotri

    x = x0.copy()
    x_surface, x_RT, x_instrument = fm.unpack(x)

    # Get all the RT quantities
    (r, L_tot, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif) = fm.RT.calc_RT_quantities(
        x_RT, geom
    )

    # Path radiance and spherical albedo
    L_atm = fm.RT.get_L_atm(x_RT, geom)
    s = r["sphalb"]

    # Get all the surface quantities for the super pixel
    sub_surface, sub_RT, sub_instrument = fm.unpack(sub_state)

    # Surface reflectance at the wl resolution of fm.RT
    rho_dir_dir, rho_dif_dir = fm.calc_rfl(sub_surface, geom)
    rho_dir_dir = fm.upsample(fm.surface.wl, rho_dir_dir)
    rho_dif_dir = fm.upsample(fm.surface.wl, rho_dif_dir)

    # Background conditions equal to the superpixel reflectance or geom background
    rho_dir_dif = fm.upsample(
        fm.surface.wl,
        (geom.bg_rfl if isinstance(geom.bg_rfl, np.ndarray) else rho_dir_dir),
    )
    rho_dif_dif = fm.upsample(
        fm.surface.wl,
        (geom.bg_rfl if isinstance(geom.bg_rfl, np.ndarray) else rho_dif_dir),
    )

    bg = s * rho_dif_dif
    L_bg, eq_11 = fm.RT.get_L_bg(
        rho_dir_dir,
        rho_dif_dir,
        rho_dir_dif,
        rho_dif_dif,
        L_dir_dif,
        L_dif_dif,
        L_tot,
        s,
        geom,
    )

    # Get superpixel EOF shift if used
    eof_offset = fm.eof_offset(sub_surface, sub_RT, sub_instrument)

    # Set inversion target
    y = meas[winidx] - L_atm[winidx] - eof_offset[winidx] - L_bg[winidx]

    # Get the inversion indices; Include glint indices if applicable
    full_idx = np.concatenate((winidx, fm.idx_surf_nonrfl), axis=0)
    outside_ret_windows = np.ones(len(fm.idx_surface), dtype=bool)
    outside_ret_windows[full_idx] = False
    outside_ret_windows = np.where(outside_ret_windows)[0]

    # TODO Support flexible AOE and smoothing indices
    iv_idx = fm.surface.analytical_iv_idx

    # The H matrix does not change as a function of x-vector
    H = fm.surface.analytical_model(
        bg,
        L_tot=L_tot,
        geom=geom,
        L_dir_dir=L_dir_dir,
        L_dir_dif=L_dir_dif,
        L_dif_dir=L_dif_dir,
        L_dif_dif=L_dif_dif,
    )
    # Sample just the wavelengths and states of interest
    L = H[winidx, :][:, iv_idx]

    # Use cached scaling factor from inital normalized inverse (outside of loop).
    Sa, Sa_inv, Sa_inv_sqrt = fm.Sa(x, geom)
    Sa_inv = Sa_inv[fm.idx_surface, :][:, fm.idx_surface]
    Sa_inv_sqrt = Sa_inv_sqrt[fm.idx_surface, :][:, fm.idx_surface]

    trajectory = np.zeros((num_iter + 1, len(x)))
    trajectory[0, :] = x

    # Bounded
    bounds = np.array(fm.surface.bounds)

    for n in range(num_iter):
        # Measurement uncertainty
        Seps = fm.Seps(x, meas, geom)[winidx, :][:, winidx]

        # Prior mean
        xa_full = fm.xa(x, geom)
        xa_surface = xa_full[fm.idx_surface]

        # Save the product of the prior covariance and mean
        prprod = Sa_inv @ xa_surface

        x_surface, x_RT, x_instrument = fm.unpack(x)

        C = dpotrf(Seps, 1)[0]
        P = dpotri(C, 1)[0]

        if enforce_bounds:
            xk, C_rcond, fixed_set = map_solve_constrained(
                L, iv_idx, Sa_inv, P, y, prprod, bounds, len(iv_idx) - 1, 2
            )
        else:
            xk, C_rcond = map_solve(L, P, prprod, Sa_inv, y, iv_idx)

        # Save trajectory step:
        x_surface[iv_idx] = xk
        if outside_ret_const is None:
            x_surface[outside_ret_windows] = xa_surface[outside_ret_windows]
        else:
            x_surface[outside_ret_windows] = outside_ret_const

        x[fm.idx_surface] = x_surface
        trajectory[n + 1, :] = x

    if diag_uncert:
        if len(C_rcond):
            full_unc = np.ones(len(x))
            full_unc[iv_idx] = np.sqrt(np.diag(C_rcond))
        else:
            full_unc = np.ones(len(x))
            full_unc[iv_idx] = [-9999 for i in x[iv_idx]]

        return trajectory, full_unc
    else:
        return trajectory, C_rcond
