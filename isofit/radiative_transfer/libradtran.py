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

import os
import logging
import numpy as np
from scipy.interpolate import interp1d

from ..core.common import resample_spectrum, VectorInterpolator
from .look_up_tables import TabularRT, FileExistsError

from isofit.configs.sections.radiative_transfer_config import RadiativeTransferEngineConfig
from isofit.configs import Config


### Variables ###

eps = 1e-5  # used for finite difference derivative calculations

### Classes ###


class LibRadTranRT(TabularRT):
    """A model of photon transport including the atmosphere."""

    def __init__(self, engine_config: RadiativeTransferEngineConfig, full_config: Config):

        self.angular_lut_keys_degrees = ['OBSZEN', 'TRUEAZ', 'viewzen', 'viewaz',
                                         'solzen', 'solaz']
        self.angular_lut_keys_radians = []

        super().__init__(engine_config, full_config)
        self.treat_as_emissive = False
        self.libradtran_dir = self.find_basedir(engine_config)
        self.libradtran_template_file = engine_config.template_file
        self.libradtran_environment = engine_config.environment

        self.lut_quantities = ['rhoatm', 'transm', 'sphalb', 'transup']

        # Build the lookup table
        self.build_lut()

    def find_basedir(self, config: RadiativeTransferEngineConfig):
        """Seek out a libradtran base directory."""

        if config.engine_base_dir is not None:
            return config.engine_base_dir

        try:
            return os.getenv('LIBRADTRAN_DIR')
        except KeyError:
            pass
        return None

    def rebuild_cmd(self, point, fn):
        """."""

        # start with defaults
        vals = {'atmosphere': 'midlatitude_summer'}
        for n, v in zip(self.lut_names, point):
            vals[n] = v

        # Translate a couple of special cases
        if 'AOT550' in self.lut_names:
            vals['aerosol_visibility'] = self.ext550_to_vis(vals['AOT550'])
        if 'H2OSTR' in self.lut_names:
            vals['h2o_mm'] = vals['H2OSTR']*10.0

        with open(self.libradtran_template_file, 'r') as fin:
            template = fin.read()
            dict0, dict025, dict05 = [dict(vals).copy() for q in (1, 2, 3)]
            dict0['albedo'] = '0.0'
            dict025['albedo'] = '0.25'
            dict05['albedo'] = '0.5'
            libradtran_config_str0 = template.format(**dict0)
            libradtran_config_str025 = template.format(**dict025)
            libradtran_config_str05 = template.format(**dict05)

        # Check rebuild conditions: LUT is missing or from a different config
        infilename0 = 'LUT_'+fn+'_alb0.inp'
        infilename05 = 'LUT_'+fn+'_alb05.inp'
        infilename025 = 'LUT_'+fn+'_alb025.inp'
        infilepath0 = os.path.join(self.lut_dir, infilename0)
        infilepath05 = os.path.join(self.lut_dir, infilename05)
        infilepath025 = os.path.join(self.lut_dir, infilename025)

        outfilename0 = 'LUT_'+fn+'_alb0.out'
        outfilename05 = 'LUT_'+fn+'_alb05.out'
        outfilename025 = 'LUT_'+fn+'_alb025.out'
        outfilenamezen = 'LUT_'+fn+'.zen'
        outfilepath0 = os.path.join(self.lut_dir, outfilename0)
        outfilepath05 = os.path.join(self.lut_dir, outfilename05)
        outfilepath025 = os.path.join(self.lut_dir, outfilename025)
        outfilepathzen = os.path.join(self.lut_dir, outfilenamezen)

        scriptfilename = 'LUT_'+fn+'.sh'
        scriptfilepath = os.path.join(self.lut_dir, scriptfilename)

        # Are all files present?
        rebuild = False
        for path in [infilepath0, infilepath05, infilepath025,
                     outfilepath0, outfilepath05, outfilepath025,
                     outfilepathzen, scriptfilepath]:
            if not os.path.exists(path):
                rebuild = True

        # Has configuration changed?
        if not rebuild:
            current0 = open(infilepath0, 'r').read()
            current05 = open(infilepath05, 'r').read()
            current025 = open(infilepath025, 'r').read()
            rebuild = (rebuild or (libradtran_config_str0 != current0))
            rebuild = (rebuild or (libradtran_config_str025 != current025))
            rebuild = (rebuild or (libradtran_config_str05 != current05))

        if not rebuild:
            raise FileExistsError('Files exist')

        if self.libradtran_dir is None:
            logging.error('Specify a LibRadTran installation')
            raise KeyError('Specify a LibRadTran installation')

        # write config files
        with open(infilepath0, 'w') as f:
            f.write(libradtran_config_str0)
        with open(infilepath025, 'w') as f:
            f.write(libradtran_config_str025)
        with open(infilepath05, 'w') as f:
            f.write(libradtran_config_str05)

        # Find the location and time for solar zenith caching
        with open(infilepath0, 'r') as fin:
            lat, lon, yr, mon, day, hour, mn = \
                None, None, None, None, None, None, None
            for line in fin.readlines():
                # Skip comment lines
                if line.strip().startswith("#"):
                    continue
                if 'latitude N' in line:
                    lat = float(line.split()[-1])
                elif 'latitude S' in line:
                    lat = -float(line.split()[-1])
                elif 'longitude W' in line:
                    lon = float(line.split()[-1])
                elif 'longitude E' in line:
                    lon = -float(line.split()[-1])
                elif 'time' in line:
                    yr, mon, day, hour, mn, sec = [
                        float(q) for q in line.split()[1:]]

        # Write runscript file
        with open(scriptfilepath, 'w') as f:
            f.write('#!/usr/bin/bash\n')
            f.write('%s\n' % self.libradtran_environment)
            f.write('(cd %s/bin && ./uvspec < %s > %s)\n' %
                    (self.libradtran_dir, infilepath0, outfilepath0))
            f.write('(cd %s/bin && ./uvspec < %s > %s)\n' %
                    (self.libradtran_dir, infilepath05, outfilepath05))
            f.write('(cd %s/bin && ./uvspec < %s > %s)\n' %
                    (self.libradtran_dir, infilepath025, outfilepath025))
            f.write('(cd %s/bin && ./zenith %s -a %s -o %s -y %s %s %s %s %s > %s)\n' %
                    (self.libradtran_dir,
                     '-s 0 -q', lat, lon, yr, day, mon, hour, mn,
                     outfilepathzen))

        return 'bash '+scriptfilepath

    def load_rt(self, fn):
        """Load the results of a LibRadTran run."""

        _, rdn0,   _ = np.loadtxt(self.lut_dir+'/LUT_'+fn+'_alb0.out').T
        _, rdn025, _ = np.loadtxt(self.lut_dir+'/LUT_'+fn+'_alb025.out').T
        wl, rdn05,  irr = np.loadtxt(self.lut_dir+'/LUT_'+fn+'_alb05.out').T

        # Replace a few zeros in the irradiance spectrum via interpolation
        good = irr > 1e-15
        bad = np.logical_not(good)
        irr[bad] = interp1d(wl[good], irr[good])(wl[bad])

        # Translate to Top of Atmosphere (TOA) reflectance
        rhoatm = rdn0 / 10.0 / irr * np.pi  # Translate to uW nm-1 cm-2 sr-1
        rho025 = rdn025 / 10.0 / irr * np.pi
        rho05 = rdn05 / 10.0 / irr * np.pi

        # Resample TOA reflectances to simulate the instrument observation
        rhoatm = resample_spectrum(rhoatm, wl, self.wl, self.fwhm)
        rho025 = resample_spectrum(rho025, wl, self.wl, self.fwhm)
        rho05 = resample_spectrum(rho05,  wl, self.wl, self.fwhm)
        irr = resample_spectrum(irr,    wl, self.wl, self.fwhm)

        # Calculate some atmospheric optical constants NOTE: This calc is not
        # numerically stable for cases where rho025 and rho05 are the same.
        # Anecdotally, in all of these cases, they are also the same as rhoatm,
        # so the equation reduces to 0 / 0. Therefore, we assume that spherical
        # albedo here is zero. Any other non-finite results are (currently)
        # unexpected, so we convert them to errors.
        bad = np.logical_and(rho025 == rhoatm, rho05 == rhoatm)
        sphalb = 2.8*(2.0*rho025-rhoatm-rho05)/(rho025-rho05)
        if np.sum(bad) > 0:
            logging.debug('Setting sphalb = 0 where rho025 == rho05 == rhoatm.')
            sphalb[bad] = 0

        if not np.all(np.isfinite(sphalb)):
            raise AttributeError('Non-finite values in spherical albedo calculation')

        transm = (rho05-rhoatm)*(2.0-sphalb)

        # For now, don't estimate this term!!
        # TODO: Have LibRadTran calculate it directly
        transup = np.zeros(self.wl.shape)

        # Get solar zenith, translate to irradiance at zenith = 0
        # HACK: If a file called `prescribed_geom` exists in the LUT directory,
        # use that instead of the LibRadtran calculated zenith angle. This is
        # not the most elegant or efficient solution, but it seems to work.
        zenfile = os.path.join(self.lut_dir, "prescribed_geom")
        if not os.path.exists(zenfile):
            zenfile = os.path.join(self.lut_dir, 'LUT_'+fn+'.zen')
        with open(zenfile, 'r') as fin:
            output = fin.read().split()
            solzen, solaz = [float(q) for q in output[1:]]

        self.coszen = np.cos(solzen/360.0*2.0*np.pi)
        irr = irr / self.coszen
        self.solar_irr = irr.copy()

        results = {"wl": self.wl, 'solzen': solzen, 'irr': irr,
                   "solzen": solzen, "rhoatm": rhoatm, "transm": transm,
                   "sphalb": sphalb, "transup": transup}
        return results

    def ext550_to_vis(self, ext550):
        return np.log(50.0) / (ext550 + 0.01159)

    def build_lut(self, rebuild=False):

        TabularRT.build_lut(self, rebuild)

        librt_outputs = []
        for point, fn in zip(self.points, self.files):
            librt_outputs.append(self.load_rt(fn))

        self.cache = {}
        dims_aug = self.lut_dims + [self.n_chan]
        for key in self.lut_quantities:
            temp = np.zeros(dims_aug, dtype=float)
            for librt_output, point in zip(librt_outputs, self.points):
                ind = [np.where(g == p)[0] for g, p in
                       zip(self.lut_grids, point)]
                ind = tuple(ind)
                temp[ind] = librt_output[key]

            self.luts[key] = VectorInterpolator(self.lut_grids, temp,
                                                self.lut_interp_types)

    def _lookup_lut(self, point):
        ret = {}
        for key, lut in self.luts.items():
            ret[key] = np.array(lut(point)).ravel()
        return ret

    def get(self, x_RT, geom):
        point = np.zeros((self.n_point,))
        for point_ind, name in enumerate(self.lut_grid_config):
            if name in self.statevector_names:
                ix = self.statevector_names.index(name)
                point[point_ind] = x_RT[ix]
            elif name == "OBSZEN":
                point[point_ind] = geom.OBSZEN
            elif name == "GNDALT":
                point[point_ind] = geom.GNDALT
            elif name == "viewzen":
                point[point_ind] = geom.observer_zenith
            elif name == "viewaz":
                point[point_ind] = geom.observer_azimuth
            elif name == "solaz":
                point[point_ind] = geom.solar_azimuth
            elif name == "solzen":
                point[point_ind] = geom.solar_zenith
            elif name == "TRUEAZ":
                point[point_ind] = geom.TRUEAZ
            elif name == 'phi':
                point[point_ind] = geom.phi
            elif name == 'umu':
                point[point_ind] = geom.umu
            else:
                # If a variable is defined in the lookup table but not
                # specified elsewhere, we will default to the minimum
                point[point_ind] = min(self.lut_grid_config[name])
        return self._lookup_lut(point)

    def get_L_atm(self, x_RT, geom):
        r = self.get(x_RT, geom)
        rho = r['rhoatm']
        rdn = rho / np.pi*(self.solar_irr * self.coszen)
        return rdn

    def get_L_down_transmitted(self, x_RT, geom):
        r = self.get(x_RT, geom)
        rdn = (self.solar_irr * self.coszen) / np.pi * r['transm']
        return rdn
