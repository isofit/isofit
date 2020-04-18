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
#

import sys
import scipy as s
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar as min1d
from scipy.optimize import minimize

from .common import conditional_gaussian, emissive_radiance, eps


def heuristic_atmosphere(RT, instrument, x_RT, x_instrument,  meas, geom):
    """From a given radiance, estimate atmospheric state with band ratios.
    Used to initialize gradient descent inversions."""

    # Identify the latest instrument wavelength calibration (possibly
    # state-dependent) and identify channel numbers for the band ratio.
    wl, fwhm = instrument.calibration(x_instrument)
    b865 = s.argmin(abs(wl-865))
    b945 = s.argmin(abs(wl-945))
    b1040 = s.argmin(abs(wl-1040))
    if not (any(RT.wl > 850) and any(RT.wl < 1050)):
        return x_RT
    x_new = x_RT.copy()

    # Figure out which RT object we are using
    my_RT = None
    for rt_name in ['modtran_vswir', 'sixs_vswir', 'libradtran_vswir']:
        if rt_name in RT.RTs:
            my_RT = RT.RTs[rt_name]
            continue
    if not my_RT:
        raise ValueError('No suiutable RT object for initialization')

    # Band ratio retrieval of H2O.  Depending on the radiative transfer
    # model we are using, this state parameter could go by several names.
    for h2oname in ['H2OSTR', 'h2o']:

        if h2oname not in RT.statevec:
            continue

        # ignore unused names
        if h2oname not in my_RT.lut_names:
            continue

        # find the index in the lookup table associated with water vapor
        ind_lut = my_RT.lut_names.index(h2oname)
        ind_sv = RT.statevec.index(h2oname)
        h2os, ratios = [], []

        # We iterate through every possible grid point in the lookup table,
        # calculating the band ratio that we would see if this were the
        # atmospheric H2O content.  It assumes that defaults for all other
        # atmospheric parameters (such as aerosol, if it is there).
        for h2o in my_RT.lut_grids[ind_lut]:

            # Get Atmospheric terms at high spectral resolution
            x_RT_2 = x_RT.copy()
            x_RT_2[ind_sv] = h2o
            rhi = RT.get(x_RT_2, geom)
            rhoatm = instrument.sample(x_instrument, RT.wl, rhi['rhoatm'])
            transm = instrument.sample(x_instrument, RT.wl, rhi['transm'])
            sphalb = instrument.sample(x_instrument, RT.wl, rhi['sphalb'])
            solar_irr = instrument.sample(x_instrument, RT.wl, RT.solar_irr)

            # Assume no surface emission.  "Correct" the at-sensor radiance
            # using this presumed amount of water vapor, and measure the
            # resulting residual (as measured from linear interpolation across
            # the absorption feature)
            rho = meas * s.pi / (solar_irr * RT.coszen)
            r = 1.0 / (transm / (rho - rhoatm) + sphalb)
            ratios.append((r[b945]*2.0)/(r[b1040]+r[b865]))
            h2os.append(h2o)

        # Finally, interpolate to determine the actual water vapor level that
        # would optimize the continuum-relative correction
        p = interp1d(h2os, ratios)
        bounds = (h2os[0]+0.001, h2os[-1]-0.001)
        best = min1d(lambda h: abs(1-p(h)), bounds=bounds, method='bounded')
        x_new[ind_sv] = best.x
    return x_new


def invert_algebraic(surface, RT, instrument, x_surface, x_RT, x_instrument,
                     meas, geom):
    """Inverts radiance algebraically using Lambertian assumptions to get a 
        reflectance."""

    # Get atmospheric optical parameters (possibly at high
    # spectral resolution) and resample them if needed.
    rhi = RT.get(x_RT, geom)
    wl, fwhm = instrument.calibration(x_instrument)
    rhoatm = instrument.sample(x_instrument, RT.wl, rhi['rhoatm'])
    transm = instrument.sample(x_instrument, RT.wl, rhi['transm'])
    solar_irr = instrument.sample(x_instrument, RT.wl, RT.solar_irr)
    sphalb = instrument.sample(x_instrument, RT.wl, rhi['sphalb'])
    transup = instrument.sample(x_instrument, RT.wl, rhi['transup'])
    coszen = RT.coszen

    # Calculate the initial emission and subtract from the measurement.
    # Surface and measured wavelengths may differ.
    Ls = surface.calc_Ls(x_surface, geom)
    Ls_meas = interp1d(surface.wl, Ls, fill_value='extrapolate')(wl)
    rdn_solrfl = meas - (transup * Ls_meas)

    # Now solve for the reflectance at measured wavelengths,
    # and back-translate to surface wavelengths
    rho = rdn_solrfl * s.pi / (solar_irr * coszen)
    rfl = 1.0 / (transm / (rho - rhoatm) + sphalb)
    rfl_est = interp1d(wl, rfl, fill_value='extrapolate')(surface.wl)

    # Some downstream code will benefit from our precalculated
    # atmospheric optical parameters
    coeffs = rhoatm, sphalb, transm, solar_irr, coszen, transup
    return rfl_est, Ls, coeffs


def invert_simple(forward, meas, geom):
    """Find an initial guess at the state vector. This currently uses
    traditional (non-iterative, heuristic) atmospheric correction."""

    surface = forward.surface
    RT = forward.RT
    instrument = forward.instrument

    # First step is to get the atmosphere. We start from the initial state
    # and estimate atmospheric terms using traditional heuristics.
    x = forward.init.copy()
    x_surface, x_RT, x_instrument = forward.unpack(x)
    x[forward.idx_RT] = heuristic_atmosphere(RT, instrument,
                                             x_RT, x_instrument,  meas, geom)

    # Now, with atmosphere fixed, we can invert the radiance algebraically
    # via Lambertian approximations to get reflectance
    x_surface, x_RT, x_instrument = forward.unpack(x)
    rfl_est, Ls_est, coeffs = invert_algebraic(surface, RT,
                                               instrument, x_surface, x_RT,
                                               x_instrument, meas, geom)

    # Condition thermal part on the VSWIR portion. Only works for
    # Multicomponent surfaces. Finds the cluster nearest the VSWIR heuristic
    # inversion and uses it for the TIR suface initialization.
    if any(forward.surface.wl > 3000):
        rfl_idx = s.array([i for i, v in enumerate(forward.surface.statevec)
                           if 'RFL' in v])
        tir_idx = s.where(forward.surface.wl > 3000)[0]
        vswir_idx = s.where(forward.surface.wl < 3000)[0]
        vswir_idx = s.array([i for i in vswir_idx if i in
                             forward.surface.idx_ref])
        x_surface_temp = x_surface.copy()
        x_surface_temp[:len(rfl_est)] = rfl_est
        mu = forward.surface.xa(x_surface_temp, geom)
        C = forward.surface.Sa(x_surface_temp, geom)
        rfl_est[tir_idx] = mu[tir_idx]

    # Now we have an estimated reflectance. Fit the surface parameters.
    x_surface[forward.idx_surface] = forward.surface.fit_params(rfl_est, geom)

    # Find temperature of emissive surfaces
    if forward.surface.emissive:

        # Estimate the total radiance at sensor, leaving out surface emission
        rhoatm, sphalb, transm, solar_irr, coszen, transup = coeffs
        L_atm = RT.get_L_atm(x_RT, geom)
        L_down_transmitted = RT.get_L_down_transmitted(x_RT, geom)
        L_total_without_surface_emission = \
            L_atm + L_down_transmitted * rfl_est / (1. - sphalb * rfl_est)

        # These tend to have high transmission factors; the emissivity of most
        # materials is nearly 1 for these bands, so they are good for
        # initializing the surface temperature.
        clearest_wavelengths = [10125., 10390.00, 10690.00]

        # This is fragile if other instruments have different wavelength
        # spacing or range
        clearest_indices = [s.argmin(s.absolute(RT.wl - w))
                            for w in clearest_wavelengths]

        # Error function for nonlinear temperature fit
        def err(z):
            T = z
            emissivity = forward.surface.emissivity_for_surface_T_init
            Ls_est, d = emissive_radiance(emissivity, T,
                                          forward.surface.wl[clearest_indices])
            resid = transup[clearest_indices] * Ls_est + \
                L_total_without_surface_emission[clearest_indices] - \
                meas[clearest_indices]
            return sum(resid**2)

        # Fit temperature, set bounds,  and set the initial values
        idx_T = forward.surface.surf_temp_ind
        Tinit = s.array([forward.surface.init[idx_T]])
        Tbest = minimize(err, Tinit).x
        T = max(forward.surface.bounds[idx_T][0]+eps,
                min(Tbest, forward.surface.bounds[idx_T][1]-eps))
        x_surface[idx_T] = Tbest
        forward.surface.init[idx_T] = T

    # Update the full state vector
    x[forward.idx_surface] = x_surface

    # We record these initial values in the geometry object - the only
    # "stateful" part of the retrieval
    geom.x_surf_init = x[forward.idx_surface]
    geom.x_RT_init = x[forward.idx_RT]

    return x
