#
#  Copyright 2019 California Institute of Technology
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
# Author: Philip G Brodrick, philip.brodrick@jpl.nasa.gov
#

import os
import logging
import datetime
from sys import platform
import numpy as np
from copy import deepcopy
import yaml

from isofit.core.common import resample_spectrum, load_wavelen, VectorInterpolator
from .look_up_tables import TabularRT, FileExistsError
from isofit.core.geometry import Geometry
from isofit.configs import Config
from isofit.configs.sections.radiative_transfer_config import RadiativeTransferEngineConfig
from isofit.core.sunposition import sunpos
from isofit.radiative_transfer.six_s import SixSRT

from tensorflow import keras
from sklearn.preprocessing import StandardScaler

class SimulatedModtranRT(TabularRT):
    """A hybrid simulator and emulator of MODTRAN-like results

    Args:
        engine_config: the configuration for this particular engine
        full_config (Config): the global configuration
    """   

    def __init__(self, engine_config: RadiativeTransferEngineConfig, full_config: Config):

        # Specify which of the potential MODTRAN LUT parameters are angular, which will be handled differently
        self.angular_lut_keys_degrees = ['OBSZEN', 'TRUEAZ', 'viewzen', 'viewaz', 'solzen', 'solaz']
        self.angular_lut_keys_radians = []

        super().__init__(engine_config, full_config)

        #self.lut_quantities = ['rhoatm', 'transm', 'sphalb', 'transup']
        self.lut_quantities = ['transm', 'rhoatm', 'sphalb']
        self.treat_as_emissive = False

        # Load in the emulator
        emulator = keras.models.load_model(engine_config.emulator_file)
        emulator_aux = np.load(engine_config.emulator_aux_file)

        #if len(self.wl) != len(emulator_aux['wavelengths']) or np.any(self.wl != emulator_aux['wavelengths']):
        #    raise AttributeError('Emulator wavelengths do not match simulator wavelengths')

        if len(self.lut_quantities) != len(emulator_aux['rt_quantities']) or \
            np.any(np.array(self.lut_quantities) != np.array(emulator_aux['rt_quantities'])):
            raise AttributeError('lut_quantities: {} do not match emulator_aux rt_quantities: {}'.format(self.lut_quantities, emulator_aux['rt_quantities']))
        
        # Build a new config for sixs simulation runs using existing config
        sixs_config: RadiativeTransferEngineConfig = deepcopy(engine_config)
        sixs_config.aerosol_model_file = None
        sixs_config.aerosol_template_file = None
        sixs_config.emulator_file = None

        # Populate the sixs-values from the modtran template file
        with open(sixs_config.template_file, 'r') as infile:
            modtran_input = yaml.safe_load(infile)['MODTRAN'][0]['MODTRANINPUT']

        # Do a quickk conversion to put things in solar azimuth/zenith terms for 6s
        dt = datetime.datetime(2000, 1, 1) + datetime.timedelta(days=modtran_input['GEOMETRY']['IDAY'] - 1)  + \
             datetime.timedelta(hours = modtran_input['GEOMETRY']['GMTIME'])

        solar_azimuth, solar_zenith, ra, dec, h = sunpos(dt, modtran_input['GEOMETRY']['PARM1'],-modtran_input['GEOMETRY']['PARM2'],modtran_input['SURFACE']['GNDALT'] * 1000.0)  

        sixs_config.day = dt.day
        sixs_config.month = dt.month
        sixs_config.elev = modtran_input['SURFACE']['GNDALT']
        sixs_config.alt = modtran_input['GEOMETRY']['H1ALT']
        sixs_config.solzen = solar_zenith
        sixs_config.solaz = solar_azimuth
        sixs_config.viewzen = 180 - modtran_input['GEOMETRY']['OBSZEN']
        sixs_config.viewaz = modtran_input['GEOMETRY']['TRUEAZ']
        sixs_config.wlinf = 0.35
        sixs_config.wlsup = 2.5
        #sixs_config.earth_sun_distance_file = None
        #sixs_config.irradiance_file = None

        # Build the simulator
        sixs_rte = SixSRT(sixs_config, full_config, build_lut_only = False)
        self.solar_irr = sixs_rte.solar_irr
        self.esd = sixs_rte.esd
        self.coszen = sixs_rte.coszen
        #self.solar_irr = emulator_aux['solar_irr']

        emulator_irr = emulator_aux['solar_irr']
        irr_factor_ref = sixs_rte.esd[200, 1]
        irr_factor_current = sixs_rte.esd[sixs_rte.day_of_year-1, 1]

        # First, convert our irr to date 0, then back to current date
        self.solar_irr = emulator_irr  * irr_factor_ref**2 / irr_factor_current**2

        simulator_wavelengths = emulator_aux['simulator_wavelengths']
        emulator_wavelengths = emulator_aux['emulator_wavelengths']
        n_simulator_chan = len(simulator_wavelengths)
        n_emulator_chan = len(emulator_wavelengths)
        
        # Couple emulator-simulator
        emulator_inputs = np.zeros((sixs_rte.points.shape[0],n_simulator_chan*len(emulator_aux['rt_quantities'])), dtype=float)


        logging.info('loading 6s results for emulator')
        for ind, (point, fn) in enumerate(zip(self.points, sixs_rte.files)):
            simulator_output = sixs_rte.load_rt(fn)
            for keyind, key in enumerate(emulator_aux['rt_quantities']):
                emulator_inputs[ind,keyind*n_simulator_chan:(keyind+1)*n_simulator_chan] = simulator_output[key]
        
        logging.debug('loading SimulatedModtran feature scaler')
        feature_scaler = StandardScaler()
        feature_scaler.mean_ = emulator_aux['feature_scaler_mean']
        feature_scaler.var_ = emulator_aux['feature_scaler_var']
        feature_scaler.scale_ = emulator_aux['feature_scaler_scale']

        logging.debug('loading SimulatedModtran response scaler')
        response_scaler = StandardScaler()
        response_scaler.mean_ = emulator_aux['response_scaler_mean']
        response_scaler.var_ = emulator_aux['response_scaler_var']
        response_scaler.scale_ = emulator_aux['response_scaler_scale']

        logging.debug('Emulating')
        emulator_outputs = emulator.predict(emulator_inputs)

        emulator_outputs = emulator.predict(feature_scaler.transform(emulator_inputs))
        emulator_outputs = response_scaler.inverse_transform(emulator_outputs)

        emulator_outputs = resample_spectrum(emulator_outputs, emulator_wavelengths, self.wl, self.fwhm)

        dims_aug = self.lut_dims + [self.n_chan]
        for key_ind, key in enumerate(self.lut_quantities):
            interpolator_inputs = np.zeros(dims_aug, dtype=float)
            for point_ind, point in enumerate(self.points):
                ind = [np.where(g == p)[0] for g, p in
                       zip(self.lut_grids, point)]
                ind = tuple(ind)
                interpolator_inputs[ind] = emulator_outputs[point_ind:point_ind+1, key_ind*self.n_chan: (key_ind+1)*self.n_chan]

            if key == 'transup':
                interpolator_inputs[...] = 0

            self.luts[key] = VectorInterpolator(self.lut_grids, interpolator_inputs,
                                                self.lut_interp_types)
        self.luts['transup'] = VectorInterpolator(self.lut_grids, interpolator_inputs*0, self.lut_interp_types )


        
    def recursive_dict_search(self, indict, key):
        for k, v  in indict.items():
            if k == key:
                return v
            elif isinstance(v, dict):
                found = self.recursive_dict_search(v, key) 
                if found is not None: 
                    return found 


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
                point[point_ind] = geom.ground_elevation_km
            elif name == "H1ALT":
                point[point_ind] = geom.observer_altitude_km
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
