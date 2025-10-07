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
from __future__ import annotations

import datetime
import logging
import os
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.interpolate import interp1d

from isofit import ray
from isofit.core import units
from isofit.core.common import calculate_resample_matrix, resample_spectrum
from isofit.radiative_transfer import luts
from isofit.radiative_transfer.engines import SixSRT
from isofit.radiative_transfer.radiative_transfer_engine import RadiativeTransferEngine

Logger = logging.getLogger(__file__)


class tfLikeModel:
    def __init__(self, input_file, weights=None, biases=None):
        if input_file is None and weights is not None and biases is not None:
            # If we have weights and biases provided directly
            self.weights = weights
            self.biases = biases
            self.input_file = None

        elif input_file is not None:
            self.input_file = input_file
            model = h5py.File(input_file, "r")

            weights = []
            biases = []
            for n in model["model_weights"].keys():
                if "dense" in n:
                    if "kernel:0" in model["model_weights"][n][n]:
                        weights.append(
                            np.array(model["model_weights"][n][n]["kernel:0"])
                        )
                        biases.append(np.array(model["model_weights"][n][n]["bias:0"]))
                    else:
                        weights.append(np.array(model["model_weights"][n][n]["kernel"]))
                        biases.append(np.array(model["model_weights"][n][n]["bias"]))

            self.weights = weights
            self.biases = biases
            self.input_file = input_file
        else:
            raise ValueError(
                "You must provide either an input_file or both weights and biases."
            )

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
        "transm_down_dir",
        "transm_down_dif",  # NOTE: Formerly transm
        "transm_up_dif",
        "transm_up_dir",  # NOTE: Formerly transup
    }
    _disable_makeSim = True

    def preSim(self):
        """
        sRTMnet leverages 6S to simulate results which is best done before sRTMnet begins
        simulations itself
        """

        Logger.info("Creating a simulator configuration")
        # Create a copy of the engine_config and populate it with 6S parameters
        config = build_sixs_config(self.engine_config)

        # Emulator Aux
        aux = np.load(config.emulator_aux_file, allow_pickle=True)

        # TODO: Disable when sRTMnet_v120_aux is updated
        aux_rt_quantities = np.where(
            aux["rt_quantities"] == "transm", "transm_down_dif", aux["rt_quantities"]
        )

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

        if self.engine_config.rte_configure_and_exit:
            return

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

        self.predict_path = os.path.join(
            self.engine_config.sim_path, "sRTMnet.predicts.nc"
        )
        if os.path.exists(self.predict_path):
            Logger.info(f"Loading sRTMnet predicts from: {self.predict_path}")
            predicts = luts.load(self.predict_path, mode="r")
            if self.engine_config.emulator_file.endswith(".h5"):
                self.component_mode = "1c"
            elif self.engine_config.emulator_file.endswith(".npz"):
                self.component_mode = "6c"
        else:
            Logger.info("Loading and predicting with emulator")
            if self.engine_config.emulator_file.endswith(".h5"):
                Logger.debug("Detected hdf5 (3c) emulator file format")
                self.component_mode = "1c"

                # Stack the quantities together along a new dimension named `quantity`
                resample = resample.to_array("quantity").stack(stack=["quantity", "wl"])

                ## Reduce from 3D to 2D by stacking along the wavelength dim for each quantity
                # Convert to DataArray to stack the variables along a new `quantity` dimension
                data = sixs.to_array("quantity").stack(stack=["quantity", "wl"])

                scaler = aux.get("response_scaler", 100.0)
                response_offset = aux.get("response_offset", 0.0)

                # Now predict, scale, and add the interpolations
                emulator = tfLikeModel(self.engine_config.emulator_file)
                predicts = emulator.predict(data)
                predicts /= scaler
                predicts += response_offset
                predicts += resample

                # Unstack back to a dataset and save
                predicts = predicts.unstack("stack").to_dataset("quantity")

            elif self.engine_config.emulator_file.endswith(".npz"):
                Logger.debug("Detected npz (6c) emulator file format")
                self.component_mode = "6c"

                # This is an array of feature points tacked onto the interpolated 6s values
                feature_point_names = aux["feature_point_names"].tolist()
                if len(feature_point_names) > 0:

                    # Populate the 6S parameter values from a modtran template file
                    with open(self.engine_config.template_file, "r") as file:
                        data = yaml.safe_load(file)["MODTRAN"][0]["MODTRANINPUT"]

                    add_vector = np.zeros(
                        (self.points.shape[0], len(feature_point_names))
                    )
                    for _fpn, fpn in enumerate(feature_point_names):
                        if fpn in self.lut_names:
                            add_vector[:, feature_point_names.index(fpn)] = self.points[
                                :, self.lut_names.index(fpn)
                            ]
                        elif fpn == "H2OSTR":
                            add_vector[:, _fpn] = 2.5
                            Logger.warning(f"Using default const H2OSTR of 2.5 g/cm2.")
                        elif fpn == "AERFRAC_2" or fpn == "AOT550":
                            add_vector[:, _fpn] = 0.06
                            Logger.warning(f"Using default const AOD of 0.06.")
                        elif fpn == "observer_altitude_km":
                            add_vector[:, _fpn] = data["GEOMETRY"]["H1ALT"]
                        elif fpn == "surface_elevation_km":
                            add_vector[:, _fpn] = data["SURFACE"]["GNDALT"]
                        else:
                            raise ValueError(f"Feature point {fpn} not found in points")

                predicts = resample.copy(deep=True)
                for key in aux_rt_quantities:
                    Logger.debug(f"Loading emulator {key}")
                    emulator = tfLikeModel(
                        None, weights=aux[f"weights_{key}"], biases=aux[f"biases_{key}"]
                    )
                    Logger.debug(f"Emulating {key}")
                    if len(feature_point_names) > 0:
                        lp = emulator.predict(np.hstack((sixs[key].values, add_vector)))
                    else:
                        lp = emulator.predict(sixs[key].values)
                    Logger.debug(f"Cleanup {key}")
                    lp /= aux["response_scaler"].item()[key]
                    lp += aux["response_offset"].item()[key]

                    # filter out negative values for numerical stability before integrating
                    ltz = resample[key].values + lp < 0
                    lp[ltz] = -1 * resample[key].values[ltz]

                    predicts[key] = resample[key] + lp

            # TODO: Is saving the predicts necessary anymore now that makeSim isn't used? (was at one point)
            # Logger.info(
            #     f"Saving intermediary prediction results to: {self.predict_path}"
            # )
            # luts.saveDataset(self.predict_path, predicts)

        # Convert our irradiance to date 0 then back to current date
        # sc - If statement to make sure tsis solar model is used if supplied
        if os.path.basename(config.irradiance_file) == "tsis_f0_0p1.txt":
            # Load coarser TSIS model to match emulator expectations
            _, sol_irr = np.loadtxt(
                os.path.split(config.irradiance_file)[0] + "/tsis_f0_0p5.txt"
            ).T
            sol_irr = sol_irr / 10  # Convert to uW cm-2 sr-1 nm-1
        else:
            sol_irr = aux["solar_irr"]  # Otherwise, use sRTMnet f0
        irr_ref = sim.esd[200, 1]  # Irradiance factor
        irr_cur = sim.esd[sim.day_of_year - 1, 1]  # Factor for current date
        sol_irr = sol_irr * irr_ref**2 / irr_cur**2

        self.emulator_sol_irr = sol_irr
        self.emulator_coszen = sim["coszen"]
        self.emulator_H = calculate_resample_matrix(self.emu_wl, self.wl, self.fwhm)

        Logger.debug("Adjusting predictions")
        self.adjust(predicts)

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
        return {}

    def postSim(self):
        pass

    def adjust(self, predicts):
        """
        Post-simulation adjustments for sRTMnet.
        """
        convert = {"dir-dir", "dir-dif", "dif-dir", "dif-dif", "rhoatm"}

        # Do conversions upfront
        for key, values in predicts.items():
            if key in convert and self.component_mode == "6c":
                predicts[key] = units.transm_to_rdn(
                    values.data, self.emulator_coszen, self.emulator_sol_irr
                )

        # Place everything into shared memory
        args = [ray.put(obj) for obj in (predicts, self.emu_wl, self.wl, self.fwhm)]
        kwargs = {"H": ray.put(self.emulator_H)}

        # Create a wrapper function to return the key of the data being resampled
        wrapper = lambda key, data, *args, **kwargs: (
            key,
            resample_spectrum(data[key].data, *args, **kwargs),
        )
        resample = ray.remote(wrapper)

        # Create and launch jobs
        Logger.debug("Executing resamples in parallel")
        jobs = [
            resample.remote(key, *args, **kwargs)
            for key, data in predicts.items()
            if len(data.shape) > 1
        ]
        outdict = dict(ray.get(jobs))
        Logger.debug("Resampling finished")

        Logger.debug("Setting up lut cache")
        for _point, point in enumerate(predicts["point"].values):
            self.lut.queuePoint(
                np.array(point),
                {key: outdict[key][_point, :] for key in outdict.keys()},
            )

        Logger.debug("Flushing lut to file")
        self.lut.flush()

        # This is crude - we should revise the LUT naming and store L_* to make this
        # more explicit
        if "dir-dir" in outdict:
            self.rt_mode = "rdn"
            self.lut.setAttr("RT_mode", "rdn")
        Logger.debug("sRTMnet finished")


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
