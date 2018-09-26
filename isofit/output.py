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
from common import spectrumLoad, spectrumResample
from scipy.linalg import inv, norm, sqrtm, det
from scipy.io import savemat


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


class Output:

    def __init__(self, config, inverse):
        """Initialization specifies retrieval subwindows for calculating
        measurement cost distributions"""

        self.iv = inverse
        self.wl = self.iv.fm.instrument.wl
        self.fwhm = self.iv.fm.instrument.fwhm
        self.windows = inverse.windows
        self.winidx = inverse.winidx

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
                    spectrumLoad(self.inputs.reference_reflectance_file)

    def write_spectrum(self, x, lrfl_est, rdn_est, path_est, rdn_meas, rdn_sim,
                       geom):
        """Write the output products for a single spectrum.  This code 
        should probably be consolidated."""

        basic_products = {"estimated_reflectance_file": lrfl_est,
                          "modeled_radiance_file": rdn_est,
                          "simulated_radiance_file": rdn_sim,
                          "path_radiance_file": path_est}

        for field, prod in basic_products.items():
            if hasattr(self.output, field) and \
               getattr(self.output, field) is not None:
                with open(getattr(self.output, field), 'w') as fout:
                    for w, v in zip(self.iv.fm.instrument.wl, prod):
                        fout.write('%7.5f %7.5f\n' % (w, v))

        if hasattr(self.output, "estimated_state_file") and \
           getattr(self.output, "estimated_state_file") is not None:
            with open(getattr(self.output, "estimated_state_file"), 'w') as fout:
                for v in x:
                    fout.write('%7.5f\n' % v)

        if self.output.radiometry_correction_file is not None:
            if self.ref_wl is not None and self.ref_rfl is not None:
                resamp = spectrumResample(self.ref_rfl, self.ref_wl, self.wl,
                                          self.fwhm, fill=True)
                rdn_reference = self.iv.fm.calc_rdn(x, geom, rfl=resamp)
                factors = rdn_reference / rdn_meas
                with open(self.output.radiometry_correction_file, 'w') as fout:
                    for w, v in zip(self.iv.fm.instrument.wl, factors):
                        fout.write('%7.5f %7.5f\n' % (w, v))
            else:
                raise ValueError(
                    'Need reference spectrum for radiometry correction')

        if self.output.posterior_errors_file is not None:
            S_hat, K, G = self.iv.calc_posterior(x, geom, rdn_meas)
            package = {'K': K, 'S_hat': S_hat, 'G': G}
            savemat(self.output.posterior_errors_file, package)

        if self.output.atmospheric_coefficients_file:
            rfl_alg_init, rfl_alg_opt, coeffs = \
                self.iv.invert_algebraic(rdn_meas, x, geom)
            rhoatm, sphalb, transm, solar, coszen = coeffs
            package = {'rhoatm': rhoatm, 'transm': transm, 'sphalb': sphalb,
                       'solarirr': solar, 'coszen': coszen}
            savemat(self.output.atmospheric_coefficients_file, package)

        if self.output.data_dump_file is not None:
            self.data_dump(x, rdn_meas, geom, self.output.data_dump_file)

        if self.output.algebraic_inverse_file is not None:
            rfl_alg_init, rfl_alg_opt, coeffs = \
                self.iv.invert_algebraic(rdn_meas, x, geom)
            with open(self.output.algebraic_inverse_file, 'w') as fout:
                for w, v, u in zip(self.iv.fm.instrument.wl, rfl_alg_init, rfl_alg_opt):
                    fout.write('%7.5f %7.5f %7.5f\n' % (w, v, u))

    def data_dump(self, x, rdn_meas, geom, fname):
        """Dump diagnostic data to a file."""

        Seps_inv, Seps_inv_sqrt = self.iv.calc_Seps(rdn_meas, geom)
        rdn_est = self.iv.fm.calc_rdn(x, geom)
        rdn_est_window = rdn_est[self.winidx]
        meas_window = rdn_meas[self.winidx]
        meas_resid = (rdn_est_window-meas_window).dot(Seps_inv_sqrt)
        xa, Sa, Sa_inv, Sa_inv_sqrt = self.iv.calc_prior(x, geom)
        prior_resid = (x - xa).dot(Sa_inv_sqrt)
        xopt, coeffs = self.iv.fm.invert_algebraic(x, rdn_meas, geom)
        rhoatm, sphalb, transm, solar_irr, coszen = coeffs
        Ls = self.iv.fm.surface.calc_Ls(x[self.iv.fm.surface_inds], geom)

        # jacobian of cost
        Kb = self.iv.fm.Kb(rdn_meas, geom)
        K = self.iv.fm.K(x, geom)
        xinit = self.iv.fm.init(rdn_meas, geom)
        Sy = self.iv.fm.instrument.Sy(rdn_meas, geom)
        S_hat, K, G = self.iv.calc_posterior(x, geom, rdn_meas)
        lrfl_est = self.iv.fm.calc_lrfl(x, geom)
        cost_jac_prior = s.diagflat(x - xa).dot(Sa_inv_sqrt)
        cost_jac_meas = Seps_inv_sqrt.dot(K[self.winidx, :])
        meas_Cov = self.iv.fm.Seps(rdn_meas, geom)
        mdict = {'K': K, 'G': G, 'S_hat': S_hat, 'prior_mu': xa, 'Ls': Ls,
                 'prior_Cov': Sa, 'rdn_meas': rdn_meas, 'rdn_est': rdn_est,
                 'x': x, 'meas_Cov': meas_Cov, 'wl': self.iv.fm.wl,
                 'lrfl_est': lrfl_est, 'cost_jac_prior': cost_jac_prior,
                 'Kb': Kb, 'cost_jac_meas': cost_jac_meas, 'winidx': self.winidx,
                 'meas_resid': meas_resid, 'prior_resid': prior_resid,
                 'noise_Cov': Sy, 'xinit': xinit, 'rhoatm': rhoatm,
                 'sphalb': sphalb, 'transm': transm, 'solar_irr': solar_irr,
                 'coszen': coszen}
        savemat(fname, mdict)

    def plot_spectrum(self, x, rdn_meas, geom, fname=None):

        if fname is None and hasattr(self.output, 'plot_directory') and\
                self.output.plot_directory is not None:
            fname = self.output.plot_directory+'/frame_%i.png' % self.iv.counts
        else:
            return

        plt.cla()
        xmin, xmax = min(self.wl), max(self.wl)
        fig = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplot(1, 2, 1)
        rdn_est = self.iv.fm.calc_rdn(x, geom)
        for lo, hi in self.windows:
            idx = s.where(s.logical_and(self.wl > lo, self.wl < hi))[0]
            p1 = plt.plot(self.iv.fm.wl[idx], rdn_meas[idx],
                          color=[0.7, 0.2, 0.2], linewidth=2)
            plt.hold(True)
            p2 = plt.plot(self.iv.fm.wl, rdn_est, color='k', linewidth=2)
        plt.title("Radiance")
        ymax = max(rdn_meas)*1.25
        plt.text(500, ymax*0.92, "Measured", color=[0.7, 0.2, 0.2])
        plt.text(500, ymax*0.86, "Model", color='k')
        plt.ylabel("$\mu$W nm$^{-1}$ sr$^{-1}$ cm$^{-2}$")
        plt.xlabel("Wavelength (nm)")
        plt.ylim([-0.001, ymax])
        plt.xlim([xmin, xmax])

        plt.subplot(1, 2, 2)
        lrfl_est = self.iv.fm.calc_lrfl(x, geom)
        ymax = min(max(lrfl_est)*1.25, 0.7)
        for lo, hi in self.windows:
            if self.ref_wl is not None and self.ref_rfl is not None:
                # red line
                idx = s.where(s.logical_and(
                    self.ref_wl > lo, self.ref_wl < hi))[0]
                p1 = plt.plot(self.ref_wl[idx], self.ref_rfl[idx],
                              color=[0.7, 0.2, 0.2], linewidth=2)
                ymax = max(max(self.ref_rfl[idx]*1.2), ymax)
                plt.hold(True)
            # black line
            idx = s.where(s.logical_and(self.wl > lo, self.wl < hi))[0]
            p2 = plt.plot(self.iv.fm.wl[idx], lrfl_est[idx],
                          'k', linewidth=2)
            ymax = max(max(lrfl_est[idx]*1.2), ymax)
            # green and blue lines - surface components
            if hasattr(self.iv.fm.surface, 'components'):
                p3 = plt.plot(self.iv.fm.wl[idx], self.iv.fm.xa(x, geom)[idx],
                              'b', linewidth=2)
                for j in range(len(self.iv.fm.surface.components)):
                    z = self.iv.fm.surface.norm(
                        lrfl_est[self.iv.fm.surface.refidx])
                    mu = self.iv.fm.surface.components[j][0] * z
                    plt.plot(self.iv.fm.wl[idx], mu[idx], 'g:', linewidth=1)
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


if __name__ == '__main__':
    main()
