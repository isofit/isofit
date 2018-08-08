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
import json
import os
import re
import scipy as s
from common import json_load_ascii, combos, VectorInterpolator
from common import recursive_replace
from copy import deepcopy
from scipy.stats import norm as normal
from scipy.interpolate import interp1d
from rt_lut import TabularRT, FileExistsError

eps = 1e-5  # used for finite difference derivative calculations


class ModtranRT(TabularRT):
    """A model of photon transport including the atmosphere."""

    def __init__(self, config, instrument):

        TabularRT.__init__(self, config, instrument)

        self.modtran_dir = self.find_basedir(config)
        self.modtran_template_file = config['modtran_template_file']
        self.template = deepcopy(json_load_ascii(
            self.modtran_template_file)['MODTRAN'])
        self.filtpath = os.path.join(self.lut_dir, 'wavelengths.flt')

        if 'aerosol_model_file' in config:
            self.aerosol_model_file = config['aerosol_model_file']
            self.aerosol_template = config['aerosol_template_file']
            self.build_aerosol_model()

        # Build the lookup table
        self.build_lut(instrument)

    def find_basedir(self, config):
        '''Seek out a modtran base directory'''

        try:
            return config['modtran_directory']
        except KeyError:
            pass  # fall back to environment variable
        try:
            return os.getenv('MODTRAN_DIR')
        except KeyError:
            raise KeyError('I could not find the MODTRAN base directory')

    def load_tp6(self, infile):
        '''Load a .tp6 file.  This contains the solar geometry.  We 
           Return cosine of mean solar zenith'''

        with open(infile, 'r') as f:
            ts, te = -1, -1  # start and end indices
            lines = []
            while len(lines) == 0 or len(lines[-1]) > 0:
                try:
                    lines.append(f.readline())
                except UnicodeDecodeError:
                    pass
            #lines = f.readlines()
            for i, line in enumerate(lines):
                if "SINGLE SCATTER SOLAR" in line:
                    ts = i+5
                if ts >= 0 and len(line) < 5:
                    te = i
                    break
            if ts < 0:
                raise ValueError('Could not find solar geometry in .tp6 file')
        szen = s.array([float(lines[i].split()[3])
                        for i in range(ts, te)]).mean()
        return szen

    def load_chn(self, infile, coszen):
        """Load a .chn output file and parse critical coefficient vectors.  
           These are:
             wl      - wavelength vector
             sol_irr - solar irradiance
             sphalb  - spherical sky albedo at surface
             transm  - diffuse and direct irradiance along the 
                          sun-ground-sensor path
             transup - transmission along the ground-sensor path only 
           We parse them one wavelength at a time."""

        with open(infile) as f:
            sols, transms, sphalbs, wls, rhoatms, transups = [], [], [], [], [], []
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i < 5:
                    continue
                toks = line.strip().split(' ')
                toks = re.findall(r"[\S]+", line.strip())
                wl, wid = float(toks[0]), float(toks[8])  # nm
                solar_irr = float(toks[18]) * 1e6 * \
                    s.pi / wid / coszen  # uW/nm/sr/cm2
                rdnatm = float(toks[4]) * 1e6  # uW/nm/sr/cm2
                rhoatm = rdnatm * s.pi / (solar_irr * coszen)
                sphalb = float(toks[23])
                transm = float(toks[22]) + float(toks[21])
                transup = float(toks[24])
                sols.append(solar_irr)
                transms.append(transm)
                sphalbs.append(sphalb)
                rhoatms.append(rhoatm)
                transups.append(rhoatm)
                wls.append(wl)
        params = [s.array(i) for i in
                  [wls, sols, rhoatms, transms, sphalbs, transups]]
        return tuple(params)

    def ext550_to_vis(self, ext550):
        return s.log(50.0) / (ext550 + 0.01159)

    def modtran_driver(self, overrides):
        """Write a MODTRAN 6.0 input file"""

        param = deepcopy(self.template)

        # Insert basic aerosol template, if needed
        for aer_key in ['VIS', 'AERTYPE', 'AOT550', 'AOD550', 'EXT550']:
            if aer_key in overrides.keys():
                aerosol_template = deepcopy(
                    json_load_ascii(self.aerosol_template))
                param[0]['MODTRANINPUT']['AEROSOLS'] = aerosol_template

        # Other overrides
        for key, val in overrides.items():
            recursive_replace(param, key, val)

            if key == 'AERTYPE':

                custom_aerosol = self.get_aerosol(val)
                wl, absc, extc, asym = [list(q) for q in custom_aerosol]
                try:
                    del param[0]['MODTRANINPUT']['AEROSOLS']['IHAZE']
                except KeyError:
                    pass
                nwl = len(wl)
                param[0]['MODTRANINPUT']['AEROSOLS']['IREGSPC'][0]['NARSPC'] = nwl
                param[0]['MODTRANINPUT']['AEROSOLS']['IREGSPC'][0]['VARSPC'] = wl
                param[0]['MODTRANINPUT']['AEROSOLS']['IREGSPC'][0]['EXTC'] = extc
                param[0]['MODTRANINPUT']['AEROSOLS']['IREGSPC'][0]['ABSC'] = absc
                param[0]['MODTRANINPUT']['AEROSOLS']['IREGSPC'][0]['ASYM'] = asym

            elif key == 'EXT550' or key == 'AOT550' or key == 'AOD550':
                # MODTRAN 6.0 convention treats negative visibility as AOT550
                recursive_replace(param, 'VIS', -val)

            elif key == 'FILTNM':
                param[0]['MODTRANINPUT']['SPECTRAL']['FILTNM'] = val

            elif key in ['ITYPE', 'H1ALT', 'IDAY', 'IPARM', 'PARM1',
                         'PARM2', 'GMTIME', 'TRUEAZ', 'OBSZEN']:
                param[0]['MODTRANINPUT']['GEOMETRY'][key] = val

        return json.dumps({"MODTRAN": param})

    def build_aerosol_model(self):
        aer_data = s.loadtxt(self.aerosol_model_file)
        self.aer_wl = aer_data[:, 0]
        aer_data = aer_data[:, 1:].T
        self.naer = int(len(aer_data)/2)
        self.aer_grid = s.linspace(0, 1, self.naer)
        self.aer_asym = s.ones(len(self.aer_wl)) * 0.65  # heuristic
        aer_absc, aer_extc = [], []
        for i in range(self.naer):
            aer_extc.append(aer_data[i*2])
            aer_ssa = aer_data[i*2+1]
            aer_absc.append(aer_extc[-1] * (1.0 - aer_ssa))
        self.aer_absc = s.array(aer_absc)
        self.aer_extc = s.array(aer_extc)
        self.aer_absc_interp, self.aer_extc_interp = [], []
        for i in range(len(self.aer_wl)):
            self.aer_absc_interp.append(
                interp1d(self.aer_grid, self.aer_absc[:, i]))
            self.aer_extc_interp.append(
                interp1d(self.aer_grid, self.aer_extc[:, i]))

    def get_aerosol(self, val):
        """ Interpolation in lookup table """
        extc = s.array([p(val) for p in self.aer_extc_interp])
        absc = s.array([p(val) for p in self.aer_absc_interp])
        return deepcopy(self.aer_wl), absc, extc, self.aer_asym

    def build_lut(self, instrument, rebuild=False):
        """ Each LUT is associated with a source directory.  We build a 
            lookup table by: 
              (1) defining the LUT dimensions, state vector names, and the grid 
                  of values; 
              (2) running modtran if needed, with each MODTRAN run defining a 
                  different point in the LUT; and 
              (3) loading the LUTs, one per key atmospheric coefficient vector,
                  into memory as VectorInterpolator objects."""

        # Regenerate MODTRAN input wavelength file
        if not os.path.exists(self.filtpath):
            self.wl2flt(instrument.wl, instrument.fwhm, self.filtpath)

        TabularRT.build_lut(self, instrument, rebuild)

    def rebuild_cmd(self, point, fn):

        vals = dict([(n, v) for n, v in zip(self.lut_names, point)])
        vals['DISALB'] = True
        vals['NAME'] = fn
        vals['FILTNM'] = os.path.normpath(self.filtpath)
        modtran_config_str = self.modtran_driver(dict(vals))

        # Check rebuild conditions: LUT is missing or from a different config
        infilename = 'LUT_'+fn+'.json'
        infilepath = os.path.join(self.lut_dir, infilename)
        outchnname = fn+'.chn'
        outchnpath = os.path.join(self.lut_dir, outchnname)
        if not os.path.exists(infilepath) or\
           not os.path.exists(outchnpath):
            rebuild = True
        else:
            with open(infilepath, 'r') as f:
                current = f.read()
                rebuild = (modtran_config_str.strip() != current.strip())

        if not rebuild:
            raise FileExistsError('File exists')

        # write_config_file
        with open(infilepath, 'w') as f:
            f.write(modtran_config_str)

        # Specify location of the proper MODTRAN 6.0 binary for this OS
        xdir = {'linux': 'linux', 'darwin': 'macos', 'windows': 'windows'}
        cmd = self.modtran_dir+'/bin/'+xdir[platform]+'/mod6c_cons '+infilename
        return cmd

    def load_rt(self, point, fn):
        tp6file = self.lut_dir+'/'+fn+'.tp6'
        solzen = self.load_tp6(tp6file)
        coszen = s.cos(solzen * s.pi / 180.0)
        chnfile = self.lut_dir+'/'+fn+'.chn'
        wl, sol, rhoatm, transm, sphalb, transup = self.load_chn(
            chnfile, coszen)
        return wl, sol, solzen, rhoatm, transm, sphalb, transup

    def wl2flt(self, wls, fwhms, outfile):
        """ helper function to generate Gaussian distributions around the center 
            wavelengths """
        I = None
        sigmas = fwhms/2.355
        span = 2.0 * (wls[1]-wls[0])  # nm
        steps = 101

        with open(outfile, 'w') as fout:

            fout.write('Nanometer data for sensor\n')
            for wl, fwhm, sigma in zip(wls, fwhms, sigmas):

                ws = wl + s.linspace(-span, span, steps)
                vs = normal.pdf(ws, wl, sigma)
                vs = vs/vs[int(steps/2)]
                wns = 10000.0/(ws/1000.0)

                fout.write('CENTER:  %6.2f NM   FWHM:  %4.2f NM\n' %
                           (wl, fwhm))

                for w, v, wn in zip(ws, vs, wns):
                    fout.write(' %9.4f %9.7f %9.2f\n' % (w, v, wn))
