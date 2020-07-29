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

from sys import platform
import os
import logging
import re
import json
from copy import deepcopy
import numpy as np
import scipy.stats
import scipy.interpolate

from ..radiative_transfer.look_up_tables import TabularRT, FileExistsError
from ..core.common import json_load_ascii, recursive_replace

from ..core.common import VectorInterpolator
from isofit.configs.sections.radiative_transfer_config import RadiativeTransferEngineConfig
from isofit.configs import Config
from isofit.core.geometry import Geometry
import subprocess


### Variables ###

eps = 1e-5  # used for finite difference derivative calculations

### Classes ###


class ModtranRT(TabularRT):
    """A model of photon transport including the atmosphere."""

    def __init__(self, engine_config: RadiativeTransferEngineConfig, full_config: Config):
        """."""

        # Specify which of the potential MODTRAN LUT parameters are angular, which will be handled differently
        self.angular_lut_keys_degrees = ['OBSZEN', 'TRUEAZ', 'viewzen', 'viewaz', 'solzen', 'solaz']
        self.angular_lut_keys_radians = []

        super().__init__(engine_config, full_config)

        # Flag to determine if MODTRAN should operate with reflectivity = 1
        # (enabling thermal_upwelling and thermal_downwelling to be determined - see comments below)
        self.treat_as_emissive = False
        if self.wl[0] > 2500:
            self.treat_as_emissive = True

        self.modtran_dir = self.find_basedir(engine_config)
        flt_name = 'wavelengths_{}_{}_{}.flt'.format(
            engine_config.engine_name, self.wl[0], self.wl[-1])
        self.filtpath = os.path.join(self.lut_dir, flt_name)
        self.template = deepcopy(json_load_ascii(engine_config.template_file)['MODTRAN'])

        # Insert aerosol templates, if specified
        if engine_config.aerosol_model_file is not None:
            self.template[0]['MODTRANINPUT']['AEROSOLS'] = \
                deepcopy(json_load_ascii(engine_config.aerosol_template_file))

        # Insert aerosol data, if specified
        if engine_config.aerosol_model_file is not None:
            aer_data = np.loadtxt(engine_config.aerosol_model_file)
            self.aer_wl = aer_data[:, 0]
            aer_data = np.transpose(aer_data[:, 1:])
            self.naer = int(len(aer_data)/3)
            aer_absc, aer_extc, aer_asym = [], [], []
            for i in range(self.naer):
                aer_extc.append(aer_data[i*3])
                aer_absc.append(aer_data[i*3+1])
                aer_asym.append(aer_data[i*3+2])
            self.aer_absc = np.array(aer_absc)
            self.aer_extc = np.array(aer_extc)
            self.aer_asym = np.array(aer_asym)

        self.modtran_lut_names = ['rhoatm', 'transm', 'sphalb', 'transup']
        if self.treat_as_emissive:
            self.modtran_lut_names = ['thermal_upwelling',
                                      'thermal_downwelling'] + self.modtran_lut_names

        self.last_point_looked_up = np.zeros(self.n_point)
        self.last_point_lookup_values = np.zeros(self.n_point)

        # Build the lookup table
        self.build_lut()

    def find_basedir(self, config):
        """Seek out a modtran base directory."""

        if config.engine_base_dir is not None:
            return config.engine_base_dir

        try:
            return os.getenv('MODTRAN_DIR')
        except KeyError:
            logging.error('I could not find the MODTRAN base directory')
            raise KeyError('I could not find the MODTRAN base directory')

    def load_tp6(self, infile):
        """Load a '.tp6' file. This contains the solar geometry. We 
           Return cosine of mean solar zenith."""

        with open(infile, 'r') as f:
            ts, te = -1, -1  # start and end indices
            lines = []
            while len(lines) == 0 or len(lines[-1]) > 0:
                try:
                    lines.append(f.readline())
                except UnicodeDecodeError:
                    pass

            for i, line in enumerate(lines):
                if "SINGLE SCATTER SOLAR" in line:
                    ts = i+5
                if ts >= 0 and len(line) < 5:
                    te = i
                    break
            if ts < 0:
                logging.error('%s is missing solar geometry' % infile)
                raise ValueError('%s is missing solar geometry' % infile)
        szen = np.array([float(lines[i].split()[3])
                         for i in range(ts, te)]).mean()
        return szen

    def load_chn(self, infile, coszen):
        """Load a '.chn' output file and parse critical coefficient vectors. 

           These are:
             * wl      - wavelength vector
             * sol_irr - solar irradiance
             * sphalb  - spherical sky albedo at surface
             * transm  - diffuse and direct irradiance along the
                          sun-ground-sensor path
             * transup - transmission along the ground-sensor path only

           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            Be careful with these! They are to be used only by the
            modtran_tir functions because MODTRAN must be run with a
            reflectivity of 1 for them to be used in the RTM defined
            in radiative_transfer.py.

            * thermal_upwelling - atmospheric path radiance
            * thermal_downwelling - sky-integrated thermal path radiance
                reflected off the ground and back into the sensor.

           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

           We parse them one wavelength at a time."""

        with open(infile) as f:
            sols, transms, sphalbs, wls, rhoatms, transups = \
                [], [], [], [], [], []
            thermal_upwellings, thermal_downwellings = [], []
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i < 5:
                    continue
                toks = line.strip().split(' ')
                toks = re.findall(r"[\S]+", line.strip())
                wl, wid = float(toks[0]), float(toks[8])  # nm
                solar_irr = float(toks[18]) * 1e6 * \
                    np.pi / wid / coszen  # uW/nm/sr/cm2
                rdnatm = float(toks[4]) * 1e6  # uW/nm/sr/cm2
                rhoatm = rdnatm * np.pi / (solar_irr * coszen)
                sphalb = float(toks[23])
                transm = float(toks[22]) + float(toks[21])
                transup = float(toks[24])

                # Be careful with these! See note in function comments above
                thermal_emission = float(toks[11])
                thermal_scatter = float(toks[12])
                thermal_upwelling = (thermal_emission + thermal_scatter) / \
                    wid * 1e6  # uW/nm/sr/cm2

                # Be careful with these! See note in function comments above
                # grnd_rflt already includes ground-to-sensor transmission
                grnd_rflt = float(toks[16])
                thermal_downwelling = grnd_rflt / wid * 1e6  # uW/nm/sr/cm2

                sols.append(solar_irr)
                transms.append(transm)
                sphalbs.append(sphalb)
                rhoatms.append(rhoatm)
                transups.append(transup)

                thermal_upwellings.append(thermal_upwelling)
                thermal_downwellings.append(thermal_downwelling)

                wls.append(wl)
        params = [np.array(i) for i in [wls, sols, rhoatms, transms, sphalbs,
                                        transups, thermal_upwellings, thermal_downwellings]]
        return tuple(params)

    def ext550_to_vis(self, ext550):
        """."""

        return np.log(50.0) / (ext550 + 0.01159)

    def modtran_driver(self, overrides):
        """Write a MODTRAN 6.0 input file."""

        param = deepcopy(self.template)

        if hasattr(self, 'aer_absc'):
            fracs = np.zeros((self.naer))

        # Perform overrides
        for key, val in overrides.items():
            recursive_replace(param, key, val)

            if key.startswith('AER'):
                i = int(key.split('_')[-1])
                fracs[i] = val

            elif key == 'EXT550' or key == 'AOT550' or key == 'AOD550':
                # MODTRAN 6.0 convention treats negative visibility as AOT550
                recursive_replace(param, 'VIS', -val)

            elif key == 'FILTNM':
                param[0]['MODTRANINPUT']['SPECTRAL']['FILTNM'] = val

            elif key in ['ITYPE', 'H1ALT', 'IDAY', 'IPARM', 'PARM1',
                         'PARM2', 'GMTIME', 'TRUEAZ', 'OBSZEN']:
                param[0]['MODTRANINPUT']['GEOMETRY'][key] = val

        # For custom aerosols, specify final extinction and absorption
        # MODTRAN 6.0 convention treats negative visibility as AOT550
        if hasattr(self, 'aer_absc'):
            total_aot = fracs.sum()
            recursive_replace(param, 'VIS', -total_aot)
            total_extc = self.aer_extc.T.dot(fracs)
            total_absc = self.aer_absc.T.dot(fracs)
            norm_fracs = fracs/(fracs.sum())
            total_asym = self.aer_asym.T.dot(norm_fracs)

            # Normalize to 550 nm
            total_extc550 = scipy.interpolate.interp1d(self.aer_wl, total_extc)(0.55)
            lvl0 = param[0]['MODTRANINPUT']['AEROSOLS']['IREGSPC'][0]
            lvl0['NARSPC'] = len(self.aer_wl)
            lvl0['VARSPC'] = [float(v) for v in self.aer_wl]
            lvl0['ASYM'] = [float(v) for v in total_asym]
            lvl0['EXTC'] = [float(v) / total_extc550 for v in total_extc]
            lvl0['ABSC'] = [float(v) / total_extc550 for v in total_absc]

        return json.dumps({"MODTRAN": param}), param

    def build_lut(self, rebuild=False):
        """Each LUT is associated with a source directory.

        We build a lookup table by: 
              (1) defining the LUT dimensions, state vector names, and the grid 
                  of values; 
              (2) running modtran if needed, with each MODTRAN run defining a 
                  different point in the LUT; and 
              (3) loading the LUTs, one per key atmospheric coefficient vector,
                  into memory as VectorInterpolator objects.
        """

        # Regenerate MODTRAN input wavelength file
        if not os.path.exists(self.filtpath):
            self.wl2flt(self.wl, self.fwhm, self.filtpath)

        # Check that the H2OSTR value, if present, is not too high.
        # MODTRAN caps the value at 5x profile specified value or 100% RH, as
        # defined in PDF-page 52 of the MODTRAN user guide.
        if 'H2OSTR' in self.lut_names:
            if 'H2OOPT' in self.template[0]['MODTRANINPUT']['ATMOSPHERE'].keys() and self.template[0]['MODTRANINPUT']['ATMOSPHERE']['H2OOPT'] == '+':
                logging.info('H2OOPT found in MODTRAN template - ignoring H2O upper bound')
            else:
                # Only do this check if we don't have a LUT provided:
                need_to_rebuild = np.any([not self.required_results_exist(x) for x in self.get_lut_filenames()])
                if need_to_rebuild:

                    # Define a realistic point, based on lut grid
                    point = np.array([x[-1] for x in self.lut_grids])

                    # Set the H2OSTR value as arbitrarily high - 50 g/cm2 in this case
                    point[self.lut_names.index('H2OSTR')] = 50

                    filebase = os.path.join(os.path.dirname(self.files[-1]), 'H2O_bound_test')
                    cmd = self.rebuild_cmd(point, filebase)

                    # Run MODTRAN for up to 10 seconds - this should be plenty of time
                    if os.path.isdir(self.lut_dir) is False:
                        os.mkdir(self.lut_dir)
                    try:
                        subprocess.call(cmd, shell=True, timeout=10,
                                        cwd=self.lut_dir)
                    except:
                        pass


                    max_water = None
                    with open(os.path.join(self.lut_dir,filebase + '.tp6'), errors='ignore') as tp6file:
                        for count, line in enumerate(tp6file):
                            if 'The water column is being set to the maximum' in line:
                                max_water = line.split(',')[1].strip()
                                max_water = float(max_water.split(' ')[0])
                                break

                    if max_water is None:
                        logging.error('Could not find MODTRAN H2O upper bound in file {}'.format(filebase + '.tp6'))
                        raise KeyError('Could not find MODTRAN H2O upper bound')

                    if np.max(self.lut_grids[self.lut_names.index('H2OSTR')]) > max_water:
                        logging.error('MODTRAN max H2OSTR with current profile is {}, while H2O lut_grid is {}.  Either adjust MODTRAN profile or lut_grid.  To over-ride MODTRANs maximum allowable value, set H2OOPT to "+"'.format(max_water, self.lut_grids[self.lut_names.index('H2OSTR')]))
                        raise KeyError('MODTRAN H2O lut grid is invalid - see logs for details.')


        TabularRT.build_lut(self, rebuild)

        mod_outputs = []
        for point, fn in zip(self.points, self.files):
            mod_outputs.append(self.load_rt(fn))

        self.wl = mod_outputs[0]['wl']
        self.solar_irr = mod_outputs[0]['sol']
        self.coszen = np.cos(mod_outputs[0]['solzen'] * np.pi / 180.0)

        dims_aug = self.lut_dims + [self.n_chan]
        for key in self.modtran_lut_names:
            temp = np.zeros(dims_aug, dtype=float)
            for mod_output, point in zip(mod_outputs, self.points):
                ind = [np.where(g == p)[0] for g, p in
                       zip(self.lut_grids, point)]
                ind = tuple(ind)
                temp[ind] = mod_output[key]

            self.luts[key] = VectorInterpolator(self.lut_grids, temp,
                                                self.lut_interp_types)


    def rebuild_cmd(self, point, fn):
        """."""

        if not fn:
            logging.error("Function is not defined.")
            raise SystemExit("Function is not defined.")

        vals = dict([(n, v) for n, v in zip(self.lut_names, point)])
        vals['DISALB'] = True
        vals['NAME'] = fn
        vals['FILTNM'] = os.path.normpath(self.filtpath)
        modtran_config_str, modtran_config = self.modtran_driver(dict(vals))

        # Check rebuild conditions: LUT is missing or from a different config
        infilename = 'LUT_'+fn+'.json'
        infilepath = os.path.join(self.lut_dir, 'LUT_'+fn+'.json')

        if not self.required_results_exist(fn):
            rebuild = True
        else:
            # We compare the two configuration files, ignoring names and
            # wavelength paths which tend to be non-portable
            with open(infilepath, 'r') as fin:
                current_config = json.load(fin)['MODTRAN']
                current_config[0]['MODTRANINPUT']['NAME'] = ''
                modtran_config[0]['MODTRANINPUT']['NAME'] = ''
                current_config[0]['MODTRANINPUT']['SPECTRAL']['FILTNM'] = ''
                modtran_config[0]['MODTRANINPUT']['SPECTRAL']['FILTNM'] = ''
                current_str = json.dumps(current_config)
                modtran_str = json.dumps(modtran_config)
                rebuild = (modtran_str.strip() != current_str.strip())

        if not rebuild:
            raise FileExistsError('File exists')

        # write_config_file
        with open(infilepath, 'w') as f:
            f.write(modtran_config_str)

        # Specify location of the proper MODTRAN 6.0 binary for this OS
        xdir = {
            'linux': 'linux',
            'darwin': 'macos',
            'windows': 'windows'
        }

        # If self.modtran_dir is not defined, raise an exception
        # This occurs e.g., when MODTRAN is not installed
        if not self.modtran_dir:
            logging.error("MODTRAN directory not defined in config file.")
            raise SystemExit("MODTRAN directory not defined in config file.")

        # Generate the CLI path
        cmd = os.path.join(self.modtran_dir, 'bin', xdir[platform], 'mod6c_cons ' + infilename)
        return cmd

    def required_results_exist(self, fn):
        infilename = os.path.join(self.lut_dir, 'LUT_'+fn+'.json')
        outchnname = os.path.join(self.lut_dir, fn+'.chn')
        outtp6name = os.path.join(self.lut_dir, fn+'.tp6')

        if os.path.isfile(infilename) and os.path.isfile(outchnname) and os.path.isfile(outtp6name):
            return True
        else:
            return False

    def load_rt(self, fn):
        """."""

        tp6file = os.path.join(self.lut_dir, fn+'.tp6')
        solzen = self.load_tp6(tp6file)
        coszen = np.cos(solzen * np.pi / 180.0)

        chnfile = os.path.join(self.lut_dir, fn+'.chn')
        params = self.load_chn(chnfile, coszen)

        # Be careful with the two thermal values! They can only be used in
        # the modtran_tir functions as they require the modtran reflectivity
        # be set to 1 in order to use them in the RTM in radiative_transfer.py.
        # Don't add these to the VSWIR functions!
        names = ['wl', 'sol', 'rhoatm', 'transm', 'sphalb', 'transup']

        # Don't include the thermal terms in VSWIR runs to avoid incorrect usage
        if self.treat_as_emissive:
            names = names + ['thermal_upwelling', 'thermal_downwelling']

        results_dict = {name: param for name, param in zip(names, params)}
        results_dict['solzen'] = solzen
        results_dict['coszen'] = coszen
        return results_dict

    def _lookup_lut(self, point):
        if np.all(np.equal(point, self.last_point_looked_up)):
            return self.last_point_lookup_values
        else:
            ret = {}
            for key, lut in self.luts.items():
                ret[key] = np.array(lut(point)).ravel()

            self.last_point_looked_up = point
            self.last_point_lookup_values = ret
            return ret

    def get(self, x_RT: np.array, geom: Geometry) -> dict:
        """ Get interpolated MODTRAN results at a particular location

        Args:
            x_RT: radiative-transfer portion of the statevector
            geom: local geometry conditions for lookup

        Returns:
            interpolated modtran result

        """
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

    def get_L_atm(self, x_RT: np.array, geom: Geometry) -> np.array:
        """ Get the interpolated MODTRAN modeled atmospheric reflectance (aka path radiance).

        Args:
            x_RT: radiative-transfer portion of the statevector
            geom: local geometry conditions for lookup

        Returns:
            the interpolated MODTRAN modeled atmospheric reflectance

        """
        if self.treat_as_emissive:
            return self._get_L_atm_tir(x_RT, geom)
        else:
            return self._get_L_atm_vswir(x_RT, geom)

    def _get_L_atm_vswir(self, x_RT: np.array, geom: Geometry) -> np.array:
        r = self.get(x_RT, geom)
        rho = r['rhoatm']
        rdn = rho/np.pi*(self.solar_irr*self.coszen)
        return rdn

    def _get_L_atm_tir(self, x_RT: np.array, geom: Geometry) -> np.array:
        r = self.get(x_RT, geom)
        return r['thermal_upwelling']

    def get_L_down_transmitted(self, x_RT: np.array, geom: Geometry) -> np.array:
        """ Get the interpolated MODTRAN downward atmospheric transmittance.

        Args:
            x_RT: radiative-transfer portion of the statevector
            geom: local geometry conditions for lookup

        Returns:
            The interpolated MODTRAN downward atmospheric transmittance
        """

        if self.treat_as_emissive:
            return self._get_L_down_transmitted_tir(x_RT, geom)
        else:
            return self._get_L_down_transmitted_vswir(x_RT, geom)

    def _get_L_down_transmitted_vswir(self, x_RT, geom):
        r = self.get(x_RT, geom)
        rdn = (self.solar_irr*self.coszen) / np.pi * r['transm']
        return rdn

    def _get_L_down_transmitted_tir(self, x_RT, geom):
        """thermal_downwelling already includes the transmission factor. Also
        assume there is no multiple scattering for TIR.
        """
        r = self.get(x_RT, geom)
        return r['thermal_downwelling']

    def wl2flt(self, wavelengths: np.array, fwhms: np.array, outfile: str) -> None:
        """Helper function to generate Gaussian distributions around the
        center wavelengths.

        Args:
            wavelengths: wavelength centers
            fwhms: full width at half max
            outfile: file to write to

        """

        sigmas = fwhms/2.355
        span = 2.0 * (wavelengths[1]-wavelengths[0])  # nm
        steps = 101

        with open(outfile, 'w') as fout:

            fout.write('Nanometer data for sensor\n')
            for wl, fwhm, sigma in zip(wavelengths, fwhms, sigmas):

                ws = wl + np.linspace(-span, span, steps)
                vs = scipy.stats.norm.pdf(ws, wl, sigma)
                vs = vs/vs[int(steps/2)]
                wns = 10000.0/(ws/1000.0)

                fout.write('CENTER:  %6.2f NM   FWHM:  %4.2f NM\n' %
                           (wl, fwhm))

                for w, v, wn in zip(ws, vs, wns):
                    fout.write(' %9.4f %9.7f %9.2f\n' % (w, v, wn))
