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

import datetime
import logging
import os
import pickle
from collections import OrderedDict
from copy import deepcopy
from sys import platform

import numpy as np
import yaml
from scipy import interpolate
from tensorflow import keras

from isofit import ray
from isofit.configs import Config
from isofit.configs.sections.radiative_transfer_config import (
    RadiativeTransferEngineConfig,
)
from isofit.core.common import VectorInterpolator, load_wavelen, resample_spectrum
from isofit.core.sunposition import sunpos
from isofit.radiative_transfer.six_s import SixSRT

from .look_up_tables import TabularRT


@ray.remote
def resample_single(ind, ind_emulator_output, emulator_wavelengths, wavelengths, fwhm):
    return ind, resample_spectrum(
        ind_emulator_output, emulator_wavelengths, wavelengths, fwhm
    )


class SimulatedModtranRT(TabularRT):
    """A hybrid surrogate-model and emulator of MODTRAN-like results.  A description of the model can be found in:

     P.G. Brodrick, D.R. Thompson, J.E. Fahlen, M.L. Eastwood, C.M. Sarture, S.R. Lundeen, W. Olson-Duvall,
     N. Carmon, and R.O. Green. Generalized radiative transfer emulation for imaging spectroscopy reflectance
     retrievals. Remote Sensing of Environment, 261:112476, 2021.doi: 10.1016/j.rse.2021.112476.

    Args:
        engine_config: the configuration for this particular engine
        full_config (Config): the global configuration
    """

    def __init__(
        self, engine_config: RadiativeTransferEngineConfig, full_config: Config
    ):
        # Specify which of the potential MODTRAN LUT parameters are angular, which will be handled differently
        self.angular_lut_keys_degrees = [
            "OBSZEN",
            "TRUEAZ",
            "viewzen",
            "viewaz",
            "solzen",
            "solaz",
        ]
        self.angular_lut_keys_radians = []

        super().__init__(engine_config, full_config)

        self.lut_quantities = ["rhoatm", "transm", "sphalb", "transup"]
        self.treat_as_emissive = False

        # Load in the emulator aux - hold off on emulator till last
        # second, as the model is large, and we don't want to load in
        # parallel if possible
        emulator_aux = np.load(engine_config.emulator_aux_file)

        simulator_wavelengths = np.arange(350, 2500 + 2.5, 2.5)
        emulator_wavelengths = emulator_aux["emulator_wavelengths"]
        n_simulator_chan = len(simulator_wavelengths)
        n_emulator_chan = len(emulator_wavelengths)

        for lq in self.lut_quantities:
            if lq not in emulator_aux["rt_quantities"].tolist() and lq != "transup":
                raise AttributeError(
                    "lut_quantities: {} do not match emulator_aux rt_quantities: {}".format(
                        self.lut_quantities, emulator_aux["rt_quantities"]
                    )
                )

        interpolator_disk_paths = [
            engine_config.interpolator_base_path + "_" + rtq + ".pkl"
            for rtq in self.lut_quantities
        ]

        # Build a new config for sixs simulation runs using existing config
        sixs_config: RadiativeTransferEngineConfig = deepcopy(engine_config)
        sixs_config.aerosol_model_file = None
        sixs_config.aerosol_template_file = None
        sixs_config.emulator_file = None

        # Populate the sixs-values from the modtran template file
        with open(sixs_config.template_file, "r") as infile:
            modtran_input = yaml.safe_load(infile)["MODTRAN"][0]["MODTRANINPUT"]

        # Do a quickk conversion to put things in solar azimuth/zenith terms for 6s
        dt = (
            datetime.datetime(2000, 1, 1)
            + datetime.timedelta(days=modtran_input["GEOMETRY"]["IDAY"] - 1)
            + datetime.timedelta(hours=modtran_input["GEOMETRY"]["GMTIME"])
        )

        solar_azimuth, solar_zenith, ra, dec, h = sunpos(
            dt,
            modtran_input["GEOMETRY"]["PARM1"],
            -modtran_input["GEOMETRY"]["PARM2"],
            modtran_input["SURFACE"]["GNDALT"] * 1000.0,
        )

        sixs_config.day = dt.day
        sixs_config.month = dt.month
        sixs_config.elev = modtran_input["SURFACE"]["GNDALT"]
        sixs_config.alt = modtran_input["GEOMETRY"]["H1ALT"]
        sixs_config.solzen = solar_zenith
        sixs_config.solaz = solar_azimuth
        sixs_config.viewzen = 180 - modtran_input["GEOMETRY"]["OBSZEN"]
        sixs_config.viewaz = modtran_input["GEOMETRY"]["TRUEAZ"]
        sixs_config.wlinf = 0.35
        sixs_config.wlsup = 2.5

        # Build the simulator
        logging.debug("Create RTE simulator")
        sixs_rte = SixSRT(
            sixs_config,
            full_config,
            build_lut_only=False,
            wavelength_override=simulator_wavelengths,
            fwhm_override=np.ones(n_simulator_chan) * 2.0,
            modtran_emulation=True,
        )
        logging.debug("Initialize basic values")
        self.solar_irr = sixs_rte.solar_irr
        self.esd = sixs_rte.esd
        self.coszen = sixs_rte.coszen

        emulator_irr = emulator_aux["solar_irr"]
        irr_factor_ref = sixs_rte.esd[200, 1]
        irr_factor_current = sixs_rte.esd[sixs_rte.day_of_year - 1, 1]

        # First, convert our irr to date 0, then back to current date
        self.solar_irr = resample_spectrum(
            emulator_irr * irr_factor_ref**2 / irr_factor_current**2,
            emulator_wavelengths,
            self.wl,
            self.fwhm,
        )

        # First, check if we've already got the vector interpolators built on disk:
        prebuilt = np.all([os.path.isfile(x) for x in interpolator_disk_paths])
        if not prebuilt or self.overwrite_interpolator:
            # Load the emulator
            logging.debug("Load emulator")
            emulator = keras.models.load_model(engine_config.emulator_file)

            # Couple emulator-simulator
            emulator_inputs = np.zeros(
                (
                    sixs_rte.points.shape[0],
                    n_simulator_chan * len(emulator_aux["rt_quantities"]),
                ),
                dtype=float,
            )

            logging.info("loading 6s results for emulator")
            for ind, (point, fn) in enumerate(zip(self.points, sixs_rte.files)):
                simulator_output = sixs_rte.load_rt(fn, resample=False)
                for keyind, key in enumerate(emulator_aux["rt_quantities"]):
                    emulator_inputs[
                        ind, keyind * n_simulator_chan : (keyind + 1) * n_simulator_chan
                    ] = simulator_output[key]
            emulator_inputs[np.isnan(emulator_inputs)] = 0
            emulator_inputs_match_output = np.zeros(
                (
                    emulator_inputs.shape[0],
                    n_emulator_chan * len(emulator_aux["rt_quantities"]),
                )
            )
            logging.debug("Interpolate 6s results")
            for key_ind, key in enumerate(emulator_aux["rt_quantities"]):
                band_range_o = np.arange(
                    n_emulator_chan * key_ind, n_emulator_chan * (key_ind + 1)
                )
                band_range_i = np.arange(
                    n_simulator_chan * key_ind, n_simulator_chan * (key_ind + 1)
                )

                finterp = interpolate.interp1d(
                    simulator_wavelengths, emulator_inputs[:, band_range_i]
                )
                emulator_inputs_match_output[:, band_range_o] = finterp(
                    emulator_wavelengths
                )

            if "response_scaler" in emulator_aux.keys():
                response_scaler = emulator_aux["response_scaler"]
            else:
                response_scaler = 100.0

            logging.debug("Emulating")
            emulator_outputs = emulator.predict(emulator_inputs) / response_scaler
            emulator_outputs = emulator_outputs + emulator_inputs_match_output

            inputs = {}
            dims = self.lut_dims + [self.n_chan]
            for ki, key in enumerate(emulator_aux["rt_quantities"]):
                # Transup is always appended after
                if key == "transup":
                    continue

                data = np.zeros(dims, dtype=float)

                jobs = []
                for pi, point in enumerate(self.points):
                    ind = tuple(
                        [np.where(g == p)[0] for g, p in zip(self.lut_grids, point)]
                    )
                    leo = emulator_outputs[
                        pi, ki * n_emulator_chan : (ki + 1) * n_emulator_chan
                    ]
                    jobs.append(
                        resample_single.remote(
                            ind, leo, emulator_wavelengths, self.wl, self.fwhm
                        )
                        # resample_single(ind, leo, emulator_wavelengths, self.wl, self.fwhm)
                    )

                results = ray.get(jobs)
                # results = jobs
                for ind, res in results:
                    data[ind] = res

                del jobs, results

                inputs[key] = data

            inputs["transup"] = np.zeros(dims, dtype=float)

            self.luts = {
                key: VectorInterpolator(
                    self.lut_grids, data, self.lut_interp_types, self.interpolator_style
                )
                for key, data in inputs.items()
            }

            for i, key in enumerate(self.lut_quantities):
                with open(interpolator_disk_paths[i], "wb") as file:
                    pickle.dump(self.luts[key], file, protocol=4)
        else:
            logging.info("Prebuilt LUT interpolators found, loading from disk")

            self.luts = {}
            for i, key in enumerate(self.lut_quantities):
                with open(interpolator_disk_paths[i], "rb") as file:
                    self.luts[key] = pickle.load(file)

    def recursive_dict_search(self, indict, key):
        for k, v in indict.items():
            if k == key:
                return v
            elif isinstance(v, dict):
                found = self.recursive_dict_search(v, key)
                if found is not None:
                    return found

    def _lookup_lut(self, point):
        """
        Cache assumes Python >= 3.7 for deterministic dicts
        """
        # type(point) == numpy.ndarray which are unhashable, cast to str for caching
        key = str(point)
        if key in self.cache:
            return self.cache[key]
        else:
            ret = {key: lut(point) for key, lut in self.luts.items()}

            # If the cache is at its limit, delete the first key (FIFO)
            if self.cache_size > 0:
                if len(self.cache) == self.cache_size:
                    del self.cache[next(iter(self.cache))]

                self.cache[key] = ret

        return ret

    def get(self, x_RT, geom):
        """ """
        point = np.zeros((self.n_point,))
        for point_ind, name in enumerate(self.lut_grid_config):
            if name in self.statevector_names:
                ix = self.statevector_names.index(name)
                point[point_ind] = x_RT[ix]
            elif name == "OBSZEN":
                point[point_ind] = geom.OBSZEN
            elif name == "GNDALT":
                point[point_ind] = geom.surface_elevation_km
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
            elif name == "phi":
                point[point_ind] = geom.phi
            elif name == "umu":
                point[point_ind] = geom.umu
            else:
                # If a variable is defined in the lookup table but not
                # specified elsewhere, we will default to the minimum
                point[point_ind] = min(self.lut_grid_config[name])

        return self._lookup_lut(point)

    def get_L_atm(self, x_RT, geom):
        r = self.get(x_RT, geom)
        rho = r["rhoatm"]
        rdn = rho / np.pi * (self.solar_irr * self.coszen)
        return rdn

    def get_L_down_transmitted(self, x_RT, geom):
        r = self.get(x_RT, geom)
        rdn = (self.solar_irr * self.coszen) / np.pi * r["transm"]
        return rdn
