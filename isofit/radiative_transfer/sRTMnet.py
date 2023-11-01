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
from copy import deepcopy

import numpy as np
import yaml
from scipy.interpolate import interp1d
from tensorflow import keras

from isofit.configs.sections.radiative_transfer_config import (
    RadiativeTransferEngineConfig,
)
from isofit.core.common import resample_spectrum
from isofit.core.sunposition import sunpos
from isofit.luts import netcdf as luts
from isofit.radiative_transfer.radiative_transfer_engine import RadiativeTransferEngine
from isofit.radiative_transfer.six_s import SixSRT

Logger = logging.getLogger(__file__)


class SimulatedModtranRTv2(RadiativeTransferEngine):
    """
    A hybrid surrogate-model and emulator of MODTRAN-like results.  A description of
    the model can be found in:

        P.G. Brodrick, D.R. Thompson, J.E. Fahlen, M.L. Eastwood, C.M. Sarture, S.R. Lundeen, W. Olson-Duvall,
        N. Carmon, and R.O. Green. Generalized radiative transfer emulation for imaging spectroscopy reflectance
        retrievals. Remote Sensing of Environment, 261:112476, 2021.doi: 10.1016/j.rse.2021.112476.
    """

    lut_quantities = {
        "rhoatm",
        "transm_down_dif",  # REVIEW: Formerly transm
        "sphalb",
        "transm_up_dir",  # REVIEW: Formerly transup
    }

    def __init__(
        self,
        engine_config: RadiativeTransferEngineConfig,
        **kwargs,
    ):
        self.predict_path = os.path.join(
            self.engine_config.sim_path, "sRTMnet.predicts.nc"
        )

        super().__init__(engine_config, **kwargs)

    @staticmethod
    def build_sixs_config(engine_config):
        """
        Builds a configuration object for a 6S simulation
        """
        if not os.path.exists(config.template_file):
            raise FileNotFoundError(
                f"MODTRAN template file does not exist: {config.template_file}"
            )

        # First create a copy of the starting config
        config = deepcopy(engine_config)

        # Populate the 6S parameter values from a modtran template file
        with open(config.template_file, "r") as file:
            data = yaml.safe_load(file)["MODTRAN"][0]["MODTRANINPUT"]

        # Do a quickk conversion to put things in solar azimuth/zenith terms for 6s
        dt = (
            datetime.datetime(2000, 1, 1)
            + datetime.timedelta(days=data["GEOMETRY"]["IDAY"] - 1)
            + datetime.timedelta(hours=data["GEOMETRY"]["GMTIME"])
        )

        solar_azimuth, solar_zenith, ra, dec, h = sunpos(
            dt,
            data["GEOMETRY"]["PARM1"],
            -data["GEOMETRY"]["PARM2"],
            data["SURFACE"]["GNDALT"] * 1000.0,
        )

        # Tweak parameter values for sRTMnet
        config.aerosol_model_file = None
        config.aerosol_template_file = None
        config.emulator_file = None
        config.day = dt.day
        config.month = dt.month
        config.elev = data["SURFACE"]["GNDALT"]
        config.alt = data["GEOMETRY"]["H1ALT"]
        config.solzen = solar_zenith
        config.solaz = solar_azimuth
        config.viewzen = 180 - data["GEOMETRY"]["OBSZEN"]
        config.viewaz = data["GEOMETRY"]["TRUEAZ"]
        config.wlinf = 0.35
        config.wlsup = 2.5

        # Save 6S to a different lut file
        config.lut_path += "6S"

        return config

    def preSim(self):
        """
        sRTMnet leverages 6S to simulate results which is best done before sRTMnet begins
        simulations itself
        """
        Logger.info("Creating a simulator configuration")
        # Create a copy of the engine_config and populate it with 6S parameters
        config = build_sixs_config(self.engine_config)

        # Emulator Aux
        aux = np.load(config.emulator_aux_file)

        # Verify expected keys exist
        # missing = self.lut_quantities - set(aux["rt_quantities"].tolist())
        # if missing:
        #     raise AttributeError(
        #         f"Emulator Aux rt_quantities does not contain the following required keys: {missing}"
        #     )

        # Emulator keys (sRTMnet)
        self.emu_wl = aux["emulator_wavelengths"]

        # Simulation wavelengths overrides, REVIEW: hardcoded?
        self.sim_wl = np.arange(350, 2500 + 2.5, 2.5)
        self.sim_fwhm = np.full(self.sim_wl.size, 2.0)

        # Build the 6S simulations
        Logger.info("Building simulator and executing (6S)")
        sim = SixSRT(
            config,
            wl=sim_wl,
            fwhm=sim_fwhm,
            lut_path=config.lut_path,
            lut_grid=lut_grid,
            wavelength_file=self.wavelength_file,
            modtran_emulation=True,
            build_interpolators=False,
        )

        self.esd = sim.esd

        ## Prepare the sim results for the emulator
        # Create the resampled input data for the emulator
        resample = sim.lut[["point"]].copy()
        resample["wl"] = emu_wl

        # FIXME: Temporarily add transm until sRTMnet's aux file is updated
        sim.lut["transm"] = sim.lut["transm_down_dir"]

        # Interpolate the sim results from its wavelengths to the emulator wavelengths
        Logger.info("Interpolating simulator quantities to emulator size")
        for key in aux["rt_quantities"]:
            interpolate = interp1d(sim.wl, sim[key])
            resample[key] = (("point", "wl"), interpolate(emu_wl))

        # Stack the quantities together along a new dimension named `quantity`
        resample = resample.to_array("quantity").stack(stack=["wl", "quantity"])

        ## Reduce from 3D to 2D by stacking along the wavelength dim for each quantity
        # Convert to DataArray to stack the variables along a new `quantity` dimension
        data = sim.lut[aux["rt_quantities"]]
        data = data.to_array("quantity").stack(stack=["wl", "quantity"])

        scaler = aux.get("response_scaler", 100.0)

        # Now predict, scale, and add the interpolations
        Logger.info("Loading and predicting with emulator")
        emulator = keras.models.load_model(self.engine_config.emulator_file)
        predicts = emulator.predict(data) / scaler
        predicts += resample

        # Unstack back to a dataset and save
        predicts = predicts.unstack("stack").to_dataset("quantity")

        Logger.info(f"Saving intermediary prediction results to: {self.predict_path}")
        luts.saveDataset(self.predict_path, predicts)

        # Convert our irradiance to date 0 then back to current date
        sol_irr = aux["solar_irr"]
        irr_ref = sim.esd[200, 1]  # Irradiance factor
        irr_cur = sim.esd[config.day - 1, 1]  # Factor for current date
        sol_irr = sol_irr * irr_ref**2 / irr_cur**2

        # Insert these into the LUT file
        return {
            "coszen": sim["coszen"],
            "solar_irr": resample_spectrum(solar_irr, self.emu_wl, sim.wl, sim.fwhm),
        }

    def makeSim(self, point):
        """
        sRTMnet does not implement a makeSim because it leverages 6S as its simulator
        As such, preSim() to create 6S, readSim() to process the 6S results
        """
        pass

    def readSim(self, point):
        """
        Resamples the predicts produced by preSim to be saved in self.lut_path
        """
        data = luts.load(self.predict_path).sel(point=point).load()
        return {
            key: resample_spectrum(values.data, self.emu_wl, self.sim_wl, self.sim_fwhm)
            for key, values in data.items()
        }

    def get_L_atm(self, x_RT, geom):
        r = self.get(x_RT, geom)
        rho = r["rhoatm"]
        rdn = rho / np.pi * (self.solar_irr * self.coszen)
        return rdn

    def get_L_down_transmitted(self, x_RT, geom):
        r = self.get(x_RT, geom)
        rdn = (self.solar_irr * self.coszen) / np.pi * r["transm"]
        return rdn


def recursive_dict_search(indict, key):
    for k, v in indict.items():
        if k == key:
            return v
        elif isinstance(v, dict):
            found = self.recursive_dict_search(v, key)
            if found is not None:
                return found
