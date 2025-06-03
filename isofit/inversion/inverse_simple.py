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

import os
from typing import OrderedDict

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, minimize
from scipy.optimize import minimize_scalar as min1d

from isofit.core.common import (
    emissive_radiance,
    eps,
    get_refractive_index,
    svd_inv_sqrt,
)
from isofit.data import env


def heuristic_atmosphere(
    RT: RadiativeTransfer,
    instrument: Instrument,
    x_RT: np.array,
    x_instrument: np.array,
    meas: np.array,
    geom: Geometry,
):
    """From a given radiance, estimate atmospheric state with band ratios.
    Used to initialize gradient descent inversions.

    Args:
        RT: radiative transfer model to use
        instrument: instrument for noise characterization
        x_RT: radiative transfer portion of the state vector
        x_instrument: instrument portion of the state vector
        meas: a one-D numpy vector of radiance in uW/nm/sr/cm2
        geom: geometry object corresponding to given measurement

    Returns:
        x_new: updated estimate of x_RT
    """

    # Identify the latest instrument wavelength calibration (possibly
    # state-dependent) and identify channel numbers for the band ratio.
    wl, fwhm = instrument.calibration(x_instrument)
    b865 = np.argmin(abs(wl - 865))
    b945 = np.argmin(abs(wl - 945))
    b1040 = np.argmin(abs(wl - 1040))
    if not (any(RT.wl > 850) and any(RT.wl < 1050)):
        return x_RT
    x_new = x_RT.copy()

    # Figure out which RT object we are using
    # TODO: this is currently very specific to vswir-tir 2-mode, eventually generalize
    my_RT = None
    for rte in RT.rt_engines:
        if rte.treat_as_emissive is False:
            my_RT = rte
            break
    if not my_RT:
        raise ValueError("No suitable RT object for initialization")

    # Band ratio retrieval of H2O.  Depending on the radiative transfer
    # model we are using, this state parameter could go by several names.
    for h2oname in ["H2OSTR", "h2o"]:
        if h2oname not in RT.statevec_names:
            continue

        # ignore unused names
        if h2oname not in my_RT.lut_names:
            continue

        # find the index in the lookup table associated with water vapor
        ind_sv = RT.statevec_names.index(h2oname)
        h2os, ratios = [], []

        # We iterate through every possible grid point in the lookup table,
        # calculating the band ratio that we would see if this were the
        # atmospheric H2O content.  It assumes that defaults for all other
        # atmospheric parameters (such as aerosol, if it is there).
        for h2o in my_RT.lut_grid[h2oname]:
            # Get Atmospheric terms at high spectral resolution
            x_RT_2 = x_RT.copy()
            x_RT_2[ind_sv] = h2o
            rhi = RT.get_shared_rtm_quantities(x_RT_2, geom)
            rhoatm = instrument.sample(x_instrument, RT.wl, rhi["rhoatm"])
            if (
                not isinstance(rhi["transm_up_dir"], np.ndarray)
                or len(rhi["transm_up_dir"]) == 1
            ):
                # 1c case
                transm = instrument.sample(x_instrument, RT.wl, rhi["transm_down_dif"])
            else:
                # 4c case
                transm = instrument.sample(
                    x_instrument,
                    RT.wl,
                    (
                        (rhi["transm_down_dir"] + rhi["transm_down_dif"])
                        * (rhi["transm_up_dir"] + rhi["transm_up_dif"])
                    ),
                )
            sphalb = instrument.sample(x_instrument, RT.wl, rhi["sphalb"])
            solar_irr = instrument.sample(x_instrument, RT.wl, RT.solar_irr)

            # Assume no surface emission.  "Correct" the at-sensor radiance
            # using this presumed amount of water vapor, and measure the
            # resulting residual (as measured from linear interpolation across
            # the absorption feature)
            coszen, cos_i = geom.check_coszen_and_cos_i(RT.coszen)
            if my_RT.rt_mode == "rdn":
                rho = meas
            else:
                rho = RT.rdn_to_rho(meas, coszen, solar_irr)

            r = 1.0 / (transm / (rho - rhoatm) + sphalb)
            ratios.append((r[b945] * 2.0) / (r[b1040] + r[b865]))
            h2os.append(h2o)

        # Finally, interpolate to determine the actual water vapor level that
        # would optimize the continuum-relative correction
        p = interp1d(h2os, ratios)
        bounds = (h2os[0] + 0.001, h2os[-1] - 0.001)
        best = min1d(lambda h: abs(1 - p(h)), bounds=bounds, method="bounded")
        x_new[ind_sv] = best.x
    return x_new


def invert_algebraic(
    surface: Surface,
    RT: RadiativeTransfer,
    instrument: Instrument,
    x_surface: np.array,
    x_RT: np.array,
    x_instrument: np.array,
    meas: np.array,
    geom: Geometry,
):
    """Inverts radiance algebraically using Lambertian assumptions to get a
    reflectance.

    Args:
        surface: surface model
        RT: radiative transfer model to use
        instrument: instrument model
        x_surface: surface portion of the state vector
        x_RT: radiative transfer portion of the state vector
        x_instrument: instrument portion of the state vector
        meas: a one-D numpy vector of radiance in uW/nm/sr/cm2
        geom: geometry object corresponding to given measurement

    Return:
        rfl_est: estimate of the surface reflectance based on the given surface model and specified atmospheric state
        Ls: estimate of the emitted surface leaving radiance
        coeffs: atmospheric parameters for the forward model
    """

    # Get atmospheric optical parameters (possibly at high
    # spectral resolution) and resample them if needed.
    rhi = RT.get_shared_rtm_quantities(x_RT, geom)
    wl, fwhm = instrument.calibration(x_instrument)
    rhoatm = instrument.sample(x_instrument, RT.wl, rhi["rhoatm"])
    transm = instrument.sample(
        x_instrument, RT.wl, rhi["transm_down_dir"] + rhi["transm_down_dif"]
    )  # REVIEW: Changed from transm
    solar_irr = instrument.sample(x_instrument, RT.wl, RT.solar_irr)
    sphalb = instrument.sample(x_instrument, RT.wl, rhi["sphalb"])
    transup = instrument.sample(
        x_instrument, RT.wl, rhi["transm_up_dir"]
    )  # REVIEW: Changed from transup

    # Figure out which RT object we are using
    # TODO: this is currently very specific to vswir-tir 2-mode, eventually generalize
    my_RT = None
    for rte in RT.rt_engines:
        if rte.treat_as_emissive is False:
            my_RT = rte
            break
    if not my_RT:
        raise ValueError("No suitable RT object for initialization")

    # Prevent NaNs
    transm[transm == 0] = 1e-5

    # Calculate the initial emission and subtract from the measurement.
    # Surface and measured wavelengths may differ.
    Ls = surface.calc_Ls(x_surface, geom)
    Ls_meas = interp1d(surface.wl, Ls, fill_value="extrapolate")(wl)
    rdn_solrfl = meas - (transup * Ls_meas)

    # Now solve for the reflectance at measured wavelengths,
    # and back-translate to surface wavelengths
    coszen, cos_i = geom.check_coszen_and_cos_i(RT.coszen)
    if my_RT.rt_mode == "rdn":
        rho = rdn_solrfl
    else:
        rho = RT.rdn_to_rho(rdn_solrfl, coszen, solar_irr)

    rfl = 1.0 / (transm / (rho - rhoatm) + sphalb)
    rfl[rfl > 1.0] = 1.0
    rfl_est = interp1d(wl, rfl, fill_value="extrapolate")(surface.wl)

    # Some downstream code will benefit from our precalculated
    # atmospheric optical parameters
    coeffs = rhoatm, sphalb, transm, solar_irr, coszen, transup
    return rfl_est, Ls, coeffs


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
):
    """Perform an analytical estimate of the conditional MAP estimate for
    a fixed atmosphere.  Based on the "Inner loop" from Susiluoto, 2022.

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

    EXIT_CODE = 0

    # Note, this will fail if x_instrument is populated
    if len(fm.idx_instrument) > 0:
        raise AttributeError(
            "Invert analytical not currently set to "
            "handle instrument state variable indexing"
        )

    x = x0.copy()
    x_surface, x_RT, x_instrument = fm.unpack(x)

    # Get all the RT quantities
    (r, L_tot, L_down_dir, L_down_dif, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif) = (
        fm.RT.calc_RT_quantities(x_RT, geom)
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

    # Background conditions equal to the superpixel reflectance
    bg = s * rho_dif_dir

    # Special case: 1-component model
    if type(L_tot) != np.ndarray or len(L_tot) == 1:
        L_tot = L_down_dir + L_down_dif

    # Get the inversion indices; Include glint indices if applicable
    full_idx = np.concatenate((winidx, fm.idx_surf_nonrfl), axis=0)
    outside_ret_windows = np.ones(len(fm.idx_surface), dtype=bool)
    outside_ret_windows[full_idx] = False
    outside_ret_windows = np.where(outside_ret_windows)[0]
    iv_idx = fm.surface.analytical_iv_idx

    # The H matrix does not change as a function of x-vector
    H = fm.surface.analytical_model(
        bg,
        L_down_dir,
        L_down_dif,
        L_tot,
        geom,
        L_dir_dir=L_dir_dir,
        L_dir_dif=L_dir_dif,
        L_dif_dir=L_dif_dir,
        L_dif_dif=L_dif_dif,
    )
    # Sample just the wavelengths and states of interest
    L = H[winidx, :][:, iv_idx]

    trajectory = np.zeros((num_iter + 1, len(x)))
    trajectory[0, :] = x
    for n in range(num_iter):
        # Measurement uncertainty
        Seps = fm.Seps(x, meas, geom)[winidx, :][:, winidx]

        # Prior covariance
        try:
            Sa = fm.Sa(x, geom)
            Sa_surface = Sa[fm.idx_surface, :][:, fm.idx_surface]
            Sa_inv = svd_inv_sqrt(Sa_surface, hash_table, hash_size)[0]

        except (np.linalg.LinAlgError, ValueError) as e:
            C_rcond = []
            trajectory[n + 1, :] = [fill_value] * len(x)
            if isinstance(e, np.linalg.LinAlgError):
                EXIT_CODE = -15
                continue
            elif isinstance(e, ValueError):
                EXIT_CODE = -11
                continue

        # Prior mean
        xa_full = fm.xa(x, geom)
        xa_surface = xa_full[fm.idx_surface]

        # Save the product of the prior covariance and mean
        prprod = Sa_inv @ xa_surface

        x_surface, x_RT, x_instrument = fm.unpack(x)

        C = dpotrf(Seps, 1)[0]
        P = dpotri(C, 1)[0]

        P_tilde = ((L.T @ P) @ L).T
        P_rcond = Sa_inv[iv_idx, :][:, iv_idx] + P_tilde

        LI_rcond = dpotrf(P_rcond)[0]
        C_rcond = dpotri(LI_rcond)[0]
        xk = dsymv(
            1,
            C_rcond,
            (L.T @ dsymv(1, P, meas[winidx] - L_atm[winidx]) + prprod[iv_idx]),
        )

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

        return trajectory, full_unc, EXIT_CODE
    else:
        return trajectory, C_rcond, EXIT_CODE


def invert_simple(forward: ForwardModel, meas: np.array, geom: Geometry):
    """Find an initial guess at the state vector. This currently uses
    traditional (non-iterative, heuristic) atmospheric correction.

    Args:
        forward: isofit forward model
        meas: a one-D numpy vector of radiance in uW/nm/sr/cm2
        geom: geometry object corresponding to given measurement

    Returns:
        x: estimate of the full statevector based on initial conditions, geometry, and a heuristic guess
    """

    surface = forward.surface
    RT = forward.RT
    instrument = forward.instrument

    vswir_present = False
    if any(forward.surface.wl < 2600):
        vswir_present = True

    tir_present = False
    if any(forward.surface.wl > 2600):
        tir_present = True

    # First step is to get the atmosphere. We start from the initial state
    # and estimate atmospheric terms using traditional heuristics.
    x = forward.init.copy()
    x_surface, x_RT, x_instrument = forward.unpack(x)

    if vswir_present:
        x[forward.idx_RT] = heuristic_atmosphere(
            RT, instrument, x_RT, x_instrument, meas, geom
        )

    # Now, with atmosphere fixed, we can invert the radiance algebraically
    # via Lambertian approximations to get reflectance
    x_surface, x_RT, x_instrument = forward.unpack(x)
    rfl_est, Ls_est, coeffs = invert_algebraic(
        surface, RT, instrument, x_surface, x_RT, x_instrument, meas, geom
    )

    # Condition thermal part on the VSWIR portion. Only works for
    # Multicomponent surfaces. Finds the cluster nearest the VSWIR heuristic
    # inversion and uses it for the TIR suface initialization.
    if tir_present:
        tir_idx = np.where(forward.surface.wl > 3000)[0]

        if vswir_present:
            x_surface_temp = x_surface.copy()
            x_surface_temp[: len(rfl_est)] = rfl_est
            mu = forward.surface.xa(x_surface_temp, geom)
            rfl_est[tir_idx] = mu[tir_idx]
        else:
            rfl_est = 0.03 * np.ones(len(forward.surface.wl))

    # Now we have an estimated reflectance. Fit the surface parameters.
    x_surface[forward.idx_surface] = forward.surface.fit_params(rfl_est, geom)

    # Find temperature of emissive surfaces
    if tir_present:
        # Estimate the total radiance at sensor, leaving out surface emission
        # Radiate transfer calculations could take place at high spectral resolution
        # so we upsample the surface reflectance
        rfl_hi = forward.upsample(forward.surface.wl, rfl_est)
        rhoatm, sphalb, transm, solar_irr, coszen, transup = coeffs

        L_atm = RT.get_L_atm(x_RT, geom)
        L_down_transmitted, _, _ = RT.get_L_down_transmitted(x_RT, geom)
        L_total_without_surface_emission = L_atm + L_down_transmitted * rfl_hi / (
            1.0 - sphalb * rfl_hi
        )

        # These tend to have high transmission factors; the emissivity of most
        # materials is nearly 1 for these bands, so they are good for
        # initializing the surface temperature.
        clearest_wavelengths = [10125.0, 10390.00, 10690.00]

        # This is fragile if other instruments have different wavelength
        # spacing or range
        clearest_indices = [
            np.argmin(np.absolute(RT.wl - w)) for w in clearest_wavelengths
        ]

        # Error function for nonlinear temperature fit
        def err(z):
            T = z
            emissivity = forward.surface.emissivity_for_surface_T_init
            Ls_est, d = emissive_radiance(
                emissivity, T, forward.surface.wl[clearest_indices]
            )
            resid = (
                transup[clearest_indices] * Ls_est
                + L_total_without_surface_emission[clearest_indices]
                - meas[clearest_indices]
            )
            return sum(resid**2)

        # Fit temperature, set bounds,  and set the initial values
        idx_T = forward.surface.surf_temp_ind
        Tinit = np.array([forward.surface.init[idx_T]])
        Tbest = minimize(err, Tinit).x
        T = max(
            forward.surface.bounds[idx_T][0] + eps,
            min(Tbest, forward.surface.bounds[idx_T][1] - eps),
        )
        x_surface[idx_T] = Tbest
        forward.surface.init[idx_T] = T

    # Update the full state vector
    x[forward.idx_surface] = x_surface

    # If available, get initial guess of surface elevation from location file.
    if geom.surface_elevation_km and "surface_elevation_km" in RT.statevec_names:
        ind_sv = forward.idx_RT[RT.statevec_names.index("surface_elevation_km")]
        if geom.surface_elevation_km < 0.0:
            x[ind_sv] = 0.0
        else:
            x[ind_sv] = geom.surface_elevation_km

    # We record these initial values in the geometry object - the only
    # "stateful" part of the retrieval
    geom.x_surf_init = x[forward.idx_surface]
    geom.x_RT_init = x[forward.idx_RT]

    return x


def invert_liquid_water(
    rfl_meas: np.array,
    wl: np.array,
    l_shoulder: float = 850,
    r_shoulder: float = 1100,
    lw_init: tuple = (0.02, 0.3, 0.0002),
    lw_bounds: tuple = ([0, 0.5], [0, 1.0], [-0.0004, 0.0004]),
    ewt_detection_limit: float = 0.5,
    return_abs_co: bool = False,
):
    """Given a reflectance estimate, fit a state vector including liquid water path length
    based on a simple Beer-Lambert surface model.

    Args:
        rfl_meas:            surface reflectance spectrum
        wl:                  instrument wavelengths, must be same size as rfl_meas
        l_shoulder:          wavelength of left absorption feature shoulder
        r_shoulder:          wavelength of right absorption feature shoulder
        lw_init:             initial guess for liquid water path length, intercept, and slope
        lw_bounds:           lower and upper bounds for liquid water path length, intercept, and slope
        ewt_detection_limit: upper detection limit for ewt
        return_abs_co:       if True, returns absorption coefficients of liquid water

    Returns:
        solution: estimated liquid water path length, intercept, and slope based on a given surface reflectance
    """

    # params needed for liquid water fitting
    lw_feature_left = np.argmin(abs(l_shoulder - wl))
    lw_feature_right = np.argmin(abs(r_shoulder - wl))
    wl_sel = wl[lw_feature_left : lw_feature_right + 1]

    # adjust upper detection limit for ewt if specified
    if ewt_detection_limit != 0.5:
        lw_bounds[0][1] = ewt_detection_limit

    # load imaginary part of liquid water refractive index and calculate wavelength dependent absorption coefficient
    path_k = env.path("data", "iop", "k_liquid_water_ice.csv")

    k_wi = pd.read_csv(path_k)
    wl_water, k_water = get_refractive_index(
        k_wi=k_wi, a=0, b=982, col_wvl="wvl_6", col_k="T = 20°C"
    )
    kw = np.interp(x=wl_sel, xp=wl_water, fp=k_water)
    abs_co_w = 4 * np.pi * kw / wl_sel

    rfl_meas_sel = rfl_meas[lw_feature_left : lw_feature_right + 1]

    x_opt = least_squares(
        fun=beer_lambert_model,
        x0=lw_init,
        jac="2-point",
        method="trf",
        bounds=(
            np.array([lw_bounds[ii][0] for ii in range(3)]),
            np.array([lw_bounds[ii][1] for ii in range(3)]),
        ),
        max_nfev=15,
        args=(rfl_meas_sel, wl_sel, abs_co_w),
    )

    solution = x_opt.x

    if return_abs_co:
        return solution, abs_co_w
    else:
        return solution


def beer_lambert_model(x, y, wl, alpha_lw):
    """Function, which computes the vector of residuals between measured and modeled surface reflectance optimizing
    for path length of surface liquid water based on the Beer-Lambert attenuation law.

    Args:
        x:        state vector (liquid water path length, intercept, slope)
        y:        measurement (surface reflectance spectrum)
        wl:       instrument wavelengths
        alpha_lw: wavelength dependent absorption coefficients of liquid water

    Returns:
        resid: residual between modeled and measured surface reflectance
    """

    attenuation = np.exp(-x[0] * 1e7 * alpha_lw)
    rho = (x[1] + x[2] * wl) * attenuation
    resid = rho - y

    return resid
