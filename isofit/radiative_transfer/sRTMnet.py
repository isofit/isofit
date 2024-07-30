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
from pathlib import Path

import dask.array as da
import h5py
import numpy as np
import yaml
from scipy.interpolate import interp1d

from isofit.configs.sections.radiative_transfer_config import (
    RadiativeTransferEngineConfig,
)
from isofit.core.common import resample_spectrum
from isofit.core.sunposition import sunpos
from isofit.radiative_transfer import luts
from isofit.radiative_transfer.radiative_transfer_engine import RadiativeTransferEngine
from isofit.radiative_transfer.six_s import SixSRT

Logger = logging.getLogger(__file__)


class tfLikeModel:
    def __init__(self, input_file):
        self.weights = []
        self.biases = []
        self.input_file = input_file
        self.model = h5py.File(input_file, "r")

        weights = []
        biases = []
        for _n, n in enumerate(self.model["model_weights"].keys()):
            if "dense" in n:
                weights.append(np.array(self.model["model_weights"][n][n]["kernel:0"]))
                biases.append(np.array(self.model["model_weights"][n][n]["bias:0"]))

        self.weights = weights
        self.biases = biases
        self.input_file = input_file

    def leaky_re_lu(self, x, alpha=0.4):
        return np.maximum(alpha * x, x)

    def predict(self, x):
        xi = x.copy()
        for i, (M, b) in enumerate(zip(self.weights, self.biases)):
            yi = np.dot(xi, M) + b
            # apply leaky_relu unless we're at the output layer
            if i < len(self.weights) - 1:
                xi = self.leaky_re_lu(yi)
        return yi


class SimulatedModtranRT(RadiativeTransferEngine):
    """
    A hybrid surrogate-model and emulator of MODTRAN-like results.  A description of
    the model can be found in:

        P.G. Brodrick, D.R. Thompson, J.E. Fahlen, M.L. Eastwood, C.M. Sarture, S.R. Lundeen, W. Olson-Duvall,
        N. Carmon, and R.O. Green. Generalized radiative transfer emulation for imaging spectroscopy reflectance
        retrievals. Remote Sensing of Environment, 261:112476, 2021.doi: 10.1016/j.rse.2021.112476.
    """

    lut_quantities = {
        "rhoatm",
        "sphalb",
        "transm_down_difs",
        "transm_down_dif",  # NOTE: Formerly transm
        "transm_up_difs",
        "transm_up_dir",  # NOTE: Formerly transup
    }

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

        # TODO: Disable when sRTMnet_v120_aux is updated
        aux_rt_quantities = np.where(
            aux["rt_quantities"] == "transm", "transm_down_dif", aux["rt_quantities"]
        )

        # TODO: Re-enable when sRTMnet_v120_aux is updated
        # Verify expected keys exist
        # missing = self.lut_quantities - set(aux["rt_quantities"].tolist())
        # if missing:
        #     raise AttributeError(
        #         f"Emulator Aux rt_quantities does not contain the following required keys: {missing}"
        #     )

        # Emulator keys (sRTMnet)
        self.emu_wl = aux["emulator_wavelengths"]

        # Simulation wavelengths overrides, always fixed size
        self.sim_wl = np.arange(350, 2500 + 2.5, 2.5)
        self.sim_fwhm = np.full(self.sim_wl.size, 2.0)

        # Build the 6S simulations
        Logger.info("Building simulator and executing (6S)")
        sim = SixSRT(
            config,
            wl=self.sim_wl,
            fwhm=self.sim_fwhm,
            lut_path=config.lut_path,
            lut_grid=self.lut_grid,
            modtran_emulation=True,
            build_interpolators=False,
        )
        # Extract useful information from the sim
        self.esd = sim.esd
        self.sim_lut_path = config.lut_path

        ## Prepare the sim results for the emulator
        # In some atmospheres the values get down to basically 0, which 6S can’t quite handle and will resolve to NaN instead of 0
        # Safe to replace here
        if sim.lut[aux_rt_quantities].isnull().any():
            Logger.debug("Simulator detected to have NaNs, replacing with 0s")
            sim.lut = sim.lut.fillna(0)

        # Interpolate the sim results from its wavelengths to the emulator wavelengths
        Logger.info("Interpolating simulator quantities to emulator size")
        sixs = sim.lut[aux_rt_quantities]
        resample = sixs.interp({"wl": aux["emulator_wavelengths"]})

        # Stack the quantities together along a new dimension named `quantity`
        resample = resample.to_array("quantity").stack(stack=["quantity", "wl"])

        ## Reduce from 3D to 2D by stacking along the wavelength dim for each quantity
        # Convert to DataArray to stack the variables along a new `quantity` dimension
        data = sixs.to_array("quantity").stack(stack=["quantity", "wl"])

        scaler = aux.get("response_scaler", 100.0)

        # Now predict, scale, and add the interpolations
        Logger.info("Loading and predicting with emulator")
        emulator = tfLikeModel(self.engine_config.emulator_file)
        predicts = da.from_array(emulator.predict(data))
        predicts /= scaler
        predicts += resample

        # Unstack back to a dataset and save
        predicts = predicts.unstack("stack").to_dataset("quantity")

        self.predict_path = os.path.join(
            self.engine_config.sim_path, "sRTMnet.predicts.nc"
        )
        Logger.info(f"Saving intermediary prediction results to: {self.predict_path}")
        luts.saveDataset(self.predict_path, predicts)

        # Convert our irradiance to date 0 then back to current date
        sol_irr = aux["solar_irr"]
        irr_ref = sim.esd[200, 1]  # Irradiance factor
        irr_cur = sim.esd[sim.day_of_year - 1, 1]  # Factor for current date
        sol_irr = sol_irr * irr_ref**2 / irr_cur**2

        # Insert these into the LUT file
        return {
            "coszen": sim["coszen"],
            "solzen": sim["solzen"],
            "solar_irr": resample_spectrum(sol_irr, self.emu_wl, self.wl, self.fwhm),
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
        # REVIEW: Likely should chunk along the point dim to improve this
        data = luts.load(self.predict_path).sel(point=tuple(point)).load()
        return {
            key: resample_spectrum(values.data, self.emu_wl, self.wl, self.fwhm)
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


def build_sixs_config(engine_config):
    """
    Builds a configuration object for a 6S simulation using a MODTRAN template
    """
    if not os.path.exists(engine_config.template_file):
        raise FileNotFoundError(
            f"MODTRAN template file does not exist: {engine_config.template_file}"
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

    relative_azimuth = data["GEOMETRY"]["PARM1"]
    observer_azimuth = data["GEOMETRY"]["TRUEAZ"]
    # RT simulations commonly only depend on the relative azimuth,
    # so we don't care if we do view azimuth + or - relative azimuth.
    # In addition, sRTMnet was only trained on relative azimuth = 0°,
    # so providing different values here would have no implications.
    solar_azimuth = np.minimum(
        observer_azimuth + relative_azimuth, observer_azimuth - relative_azimuth
    )
    solar_zenith = data["GEOMETRY"]["PARM2"]

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
    # the MODTRAN config provides the view zenith in MODTRAN convention,
    # so substract from 180 here as 6s follows the ANG OBS file convention
    config.viewzen = 180 - data["GEOMETRY"]["OBSZEN"]
    config.viewaz = observer_azimuth
    config.wlinf = 0.35
    config.wlsup = 2.5

    # Save 6S to a different lut file, prepend 6S to the sRTMnet lut_path
    # REVIEW: Should this write to sim_path instead? I think so
    path = Path(config.lut_path)
    config.lut_path = path.parent / f"6S.{path.name}"

    return config


def recursive_dict_search(indict, key):
    for k, v in indict.items():
        if k == key:
            return v
        elif isinstance(v, dict):
            found = self.recursive_dict_search(v, key)
            if found is not None:
                return found
