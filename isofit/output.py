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

import scipy as s
import pylab as plt
from common import load_spectrum, resample_spectrum
from scipy.linalg import inv, norm, sqrtm, det
from scipy.io import savemat
from inverse_simple import invert_simple, invert_algebraic


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


class Output:

    def __init__(self, config, forward, inverse):
        """Initialization specifies retrieval subwindows for calculating
        measurement cost distributions"""

        self.iv = inverse
        self.fm = forward

        self.output = Bunch(config['output'])
        for field in ['data_dump_file', 'algebraic_inverse_file',
                      'estimated_reflectance_file', 'modeled_radiance_file',
                      'posterior_errors_file', 'estimated_state_file', 'path_radiance_file',
                      'atmospheric_coefficients_file', 'plot_directory_file',
                      'radiometry_correction_file', 'measured_radiance_file']:
            if not hasattr(self.output, field):
                setattr(self.output, field, None)

        self.ref_wl, self.ref_rfl = None, None

        if 'input' in config:
            self.inputs = Bunch(config['input'])
            for field in ['reference_reflectance_file']:
                if not hasattr(self.inputs, field):
                    setattr(self.inputs, field, None)

            if self.inputs.reference_reflectance_file is not None:
                self.ref_rfl, self.ref_wl = \
                    load_spectrum(self.inputs.reference_reflectance_file)

    def write_spectrum(self, x, lamb_est, rdn_est, path_est, meas, rdn_sim,
                       geom):
        """Write the output products for a single spectrum.  This code 
        should probably be consolidated."""

        basic_products = {"estimated_reflectance_file": lamb_est,
                          "modeled_radiance_file": rdn_est,
                          "simulated_radiance_file": rdn_sim,
                          "path_radiance_file": path_est}

        for field, prod in basic_products.items():
            if hasattr(self.output, field) and \
               getattr(self.output, field) is not None:
                with open(getattr(self.output, field), 'w') as fout:
                    wl, fwhm = self.fm.calibration(x)
                    for w, v in zip(wl, prod):
                        fout.write('%7.5f %7.5f\n' % (w, v))

        if hasattr(self.output, "estimated_state_file") and \
           getattr(self.output, "estimated_state_file") is not None:
            with open(getattr(self.output, "estimated_state_file"), 'w') as fout:
                for v in x:
                    fout.write('%7.5f\n' % v)

        if self.output.radiometry_correction_file is not None:
            if self.ref_wl is not None and self.ref_rfl is not None:
                resamp = resample_spectrum(self.ref_rfl, self.ref_wl, 
                        self.fm.surface.wl, self.fm.surface.fwhm, fill=True)
                rdn_predict = self.fm.calc_meas(x, geom, rfl=resamp)
                factors = rdn_predict / _meas
                wl, fwhm = self.fm.calibration(x)
                with open(self.output.radiometry_correction_file, 'w') as fout:
                    for w, v in zip(wl, factors):
                        fout.write('%7.5f %7.5f\n' % (w, v))
            else:
                raise ValueError(
                    'Need reference spectrum for radiometry correction')

        if self.output.posterior_errors_file is not None:
            S_hat, K, G = self.iv.calc_posterior(x, geom, meas)
            package = {'K': K, 'S_hat': S_hat, 'G': G}
            savemat(self.output.posterior_errors_file, package)

        if self.output.atmospheric_coefficients_file:
            rfl_alg_opt, Ls, coeffs = invert_algebraic(x, meas, geom)
            rhoatm, sphalb, transm, solar, coszen = coeffs
            package = {'rhoatm': rhoatm, 'transm': transm, 'sphalb': sphalb,
                       'solarirr': solar, 'coszen': coszen}
            savemat(self.output.atmospheric_coefficients_file, package)

        if self.output.data_dump_file is not None:
            self.data_dump(x, meas, geom, self.output.data_dump_file)

        if self.output.algebraic_inverse_file is not None:
            rfl_alg_init = invert_simple(self.fm, meas, geom)
            x_surface, x_RT, x_instrument = self.fm.unpack(x)
            rfl_alg_opt, Ls, coeffs = invert_algebraic(self.fm.surface, 
                self.fm.RT, self.fm.instrument, x_surface, x_RT, 
                x_instrument, meas, geom)
            with open(self.output.algebraic_inverse_file, 'w') as fout:
                for w, v, u in zip(self.fm.surface.wl, rfl_alg_init, rfl_alg_opt):
                    fout.write('%7.5f %7.5f %7.5f\n' % (w, v, u))

    def data_dump(self, x, meas, geom, fname):
        """Dump diagnostic data to a file."""

        Seps_inv, Seps_inv_sqrt = self.iv.calc_Seps(x, meas, geom)
        rdn_est = self.fm.calc_rdn(x, geom)
        rdn_est_window = rdn_est[self.iv.winidx]
        meas_window = meas[self.iv.winidx]
        meas_resid = (rdn_est_window-meas_window).dot(Seps_inv_sqrt)
        xa, Sa, Sa_inv, Sa_inv_sqrt = self.iv.calc_prior(x, geom)
        prior_resid = (x - xa).dot(Sa_inv_sqrt)
        x_surface, x_RT, x_instrument = self.fm.unpack(x)
        xopt, Ls, coeffs  = invert_algebraic(self.fm.surface, self.fm.RT, 
                    self.fm.instrument, x_surface, x_RT, x_instrument, meas, geom)
        rhoatm, sphalb, transm, solar_irr, coszen = coeffs
        Ls = self.fm.surface.calc_Ls(x[self.fm.idx_surface], geom)

        # jacobian of cost
        Kb = self.fm.Kb(x, geom)
        K = self.fm.K(x, geom)
        xinit = invert_simple(self.fm, meas, geom)
        Sy = self.fm.instrument.Sy(meas, geom)
        S_hat, K, G = self.iv.calc_posterior(x, geom, meas)
        lamb_est = self.fm.calc_lamb(x, geom)
        cost_jac_prior = s.diagflat(x - xa).dot(Sa_inv_sqrt)
        cost_jac_meas = Seps_inv_sqrt.dot(K[self.iv.winidx, :])
        meas_Cov = self.fm.Seps(x, meas, geom)
        wl, fwhm = self.fm.calibration(x)
        mdict = {'K': K, 'G': G, 'S_hat': S_hat, 'prior_mu': xa, 'Ls': Ls,
                 'prior_Cov': Sa, 'meas': meas, 'rdn_est': rdn_est,
                 'x': x, 'meas_Cov': meas_Cov, 'wl': wl,
                 'lamb_est': lamb_est, 'cost_jac_prior': cost_jac_prior,
                 'Kb': Kb, 'cost_jac_meas': cost_jac_meas, 
                 'winidx': self.iv.winidx,
                 'meas_resid': meas_resid, 'prior_resid': prior_resid,
                 'noise_Cov': Sy, 'xinit': xinit, 'rhoatm': rhoatm,
                 'sphalb': sphalb, 'transm': transm, 'solar_irr': solar_irr,
                 'coszen': coszen}
        savemat(fname, mdict)

    def plot_spectrum(self, x, meas, geom, fname=None):

        if fname is None and hasattr(self.output, 'plot_directory') and\
                self.output.plot_directory is not None:
            fname = self.output.plot_directory+'/frame_%i.png' % self.iv.counts
        else:
            return

        plt.cla()
        wl, fwhm = self.fm.calibration(x)
        xmin, xmax = min(wl), max(wl)
        fig = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplot(1, 2, 1)
        rdn_est = self.fm.calc_rdn(x, geom)
        for lo, hi in self.iv.windows:
            idx = s.where(s.logical_and(wl>lo, wl<hi))[0]
            p1 = plt.plot(wl[idx], meas[idx], color=[0.7, 0.2, 0.2], 
                    linewidth=2)
            plt.hold(True)
            p2 = plt.plot(wl, rdn_est, color='k', linewidth=2)
        plt.title("Radiance")
        ymax = max(meas)*1.25
        plt.text(500, ymax*0.92, "Measured", color=[0.7, 0.2, 0.2])
        plt.text(500, ymax*0.86, "Model", color='k')
        plt.ylabel("$\mu$W nm$^{-1}$ sr$^{-1}$ cm$^{-2}$")
        plt.xlabel("Wavelength (nm)")
        plt.ylim([-0.001, ymax])
        plt.xlim([xmin, xmax])

        plt.subplot(1, 2, 2)
        lamb_est = self.fm.calc_lamb(x, geom)
        ymax = min(max(lamb_est)*1.25, 0.10)
        for lo, hi in self.iv.windows:
            if self.ref_wl is not None and self.ref_rfl is not None:
                # red line
                idx = s.where(s.logical_and(
                    self.ref_wl > lo, self.ref_wl < hi))[0]
                p1 = plt.plot(self.ref_wl[idx], self.ref_rfl[idx],
                              color=[0.7, 0.2, 0.2], linewidth=2)
                ymax = max(max(self.ref_rfl[idx]*1.2), ymax)
                plt.hold(True)
            # black line
            idx = s.where(s.logical_and(wl > lo, wl < hi))[0]
            p2 = plt.plot(wl[idx], lamb_est[idx], 'k', linewidth=2)
            ymax = max(max(lamb_est[idx]*1.2), ymax)
            # green and blue lines - surface components
            if hasattr(self.fm.surface, 'components'):
                idx = s.where(s.logical_and(self.fm.surface.wl > lo, 
                        self.fm.surface.wl < hi))[0]
                p3 = plt.plot(self.fm.surface.wl[idx], self.fm.xa(x, geom)[idx],
                              'b', linewidth=2)
                for j in range(len(self.fm.surface.components)):
                    z = self.fm.surface.norm(
                        lamb_est[self.fm.surface.idx_ref])
                    mu = self.fm.surface.components[j][0] * z
                    plt.plot(self.fm.surface.wl[idx], mu[idx], 'g:', linewidth=1)
        plt.ylim([-0.0010, ymax])
        plt.xlim([xmin, xmax])
        plt.title("Reflectance")
        plt.xlabel("Wavelength (nm)")
        if self.ref_rfl is not None:
            plt.text(500, ymax*0.92, "In situ reference",
                     color=[0.7, 0.2, 0.2])
            plt.text(500, ymax*0.86, "Remote estimate", color='k')
            plt.text(500, ymax*0.80, "Prior mean state ", color='b')
            plt.text(500, ymax*0.74, "Surface components ", color='g')

        plt.savefig(fname)
        plt.close()

