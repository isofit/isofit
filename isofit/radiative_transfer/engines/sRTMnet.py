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
import time
from copy import deepcopy
from pathlib import Path

import dask.array as da
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
    def __init__(self, input_file=None, key=None, layer_read=True):
        if input_file is not None and key is None:
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
            if layer_read:
                self.input_file = input_file
                self.key = key
                with h5py.File(input_file, "r") as model:
                    self.layers = len(model[f"weights_{key}"])

            else:
                model = h5py.File(input_file, "r")
                model[f"weights_{key}"].keys()
                with h5py.File(input_file, "r") as model:
                    self.layers = len(model[f"weights_{key}"])
                    self.weights = [
                        model[f"weights_{key}"][layer][:]
                        for layer in model[f"weights_{key}"].keys()
                    ]
                    self.biases = [
                        model[f"biases_{key}"][layer][:]
                        for layer in model[f"biases_{key}"].keys()
                    ]

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

    def load_arrays(self, i):
        weights = h5py.File(self.input_file, "r")[f"weights_{self.key}"]
        biases = h5py.File(self.input_file, "r")[f"biases_{self.key}"]

        layer = weights[f"layer_{i}"]
        offset = layer.id.get_offset()
        weight = np.memmap(
            self.input_file,
            layer.dtype,
            "r",
            offset=offset,
            shape=layer.shape,
        )

        layer = biases[f"layer_{i}"]
        offset = layer.id.get_offset()
        bias = np.memmap(
            self.input_file,
            layer.dtype,
            "r",
            offset=offset,
            shape=layer.shape,
        )

        return weight, bias


@ray.remote(num_cpus=1)
def ray_predict(model, x, layer_read=True):
    xi = x.copy()
    for i in range(model.layers):
        if layer_read:
            M, b = model.load_arrays(i)
        else:
            M, b = model.weights[i], model.biases[i]

        yi = np.dot(xi, M) + b

        # apply leaky_relu unless we're at the output layer
        if i < model.layers - 1:
            xi = model.leaky_re_lu(yi)
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
    aux_quantities = {
        "lut_names": str,
        "feature_point_names": str,
        "rt_quantities": str,
        "solar_irr": np.float64,
        "emulator_wavelengths": np.float64,
        "simulator_wavelengths": np.float64,
        "response_scaler": dict,
        "response_offset": dict,
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

        # Track the sRTMnet file used in the LUT attributes
        self.lut.setAttr("sRTMnet", str(config.emulator_file))

        # Get the component mode up front
        if self.engine_config.emulator_file.endswith(".h5"):
            self.component_mode = "3c"

        elif self.engine_config.emulator_file.endswith(".6c"):
            self.component_mode = "6c"

        else:
            raise ValueError(
                f"Invalid extension for emulator aux file. Use .npz or .6c"
            )

        # Pack the emulator Aux the same regardless of input file type.
        # Enforce types
        if self.component_mode == "3c":
            aux = dict(np.load(config.emulator_aux_file, allow_pickle=True))
            aux_dict = {}
            for key, value in self.aux_quantities.items():
                if len(aux.get(key, [])):
                    aux_dict[key] = aux.get(key)

            aux = aux_dict

        else:
            aux = {}
            with h5py.File(config.emulator_file, "r") as model:
                for key, value in self.aux_quantities.items():
                    if value == dict:
                        aux[key] = {
                            model_: model[key][model_][:].astype(np.float64)
                            for model_ in model[key].keys()
                        }
                    else:
                        aux[key] = model[key][:].astype(value)

        # TODO: Disable when sRTMnet_v120_aux is updated
        aux_rt_quantities = np.where(
            aux["rt_quantities"] == "transm", "transm_down_dif", aux["rt_quantities"]
        ).astype(str)

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
            self.component_mode = predicts.attrs.get("component_mode", "3c")

        else:
            Logger.info("Loading and predicting with emulator")
            if self.component_mode == "3c":
                Logger.debug("Detected hdf5 (3c) emulator file format")

                # Stack the quantities together along a new dimension
                # named `quantity`
                resample = resample.to_array("quantity").stack(stack=["quantity", "wl"])

                ## Reduce from 3D to 2D by stacking along the wavelength
                # dim for each quantity. Convert to DataArray to stack
                # the variables along a new `quantity` dimension
                data = sixs.to_array("quantity").stack(stack=["quantity", "wl"])

                scaler = aux.get("response_scaler", 100.0)
                response_offset = aux.get("response_offset", 0.0)

                # Now predict, scale, and add the interpolations
                emulator = tfLikeModel(self.engine_config.emulator_file)
                predicts = da.from_array(emulator.predict(data))
                predicts /= scaler
                predicts += response_offset
                predicts += resample

                # Unstack back to a dataset and save
                predicts = predicts.unstack("stack").to_dataset("quantity")
                predicts.attrs["component_mode"] = "3c"

            else:
                Logger.debug("Detected 6c emulator file format")

                # This is an array of feature points tacked onto the interpolated 6s values
                feature_point_names = aux["feature_point_names"].astype(str).tolist()
                if len(feature_point_names) > 0 and feature_point_names[0] != "None":

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

                total_start_time = time.time()
                for key in aux_rt_quantities:
                    key_start_time = time.time()
                    Logger.debug(f"Loading emulator {key}")

                    emulator = tfLikeModel(
                        input_file=self.engine_config.emulator_file,
                        key=key,
                        layer_read=self.engine_config.parallel_layer_read,
                    )

                    Logger.info(f"Emulating {key}")
                    if (
                        len(feature_point_names) > 0
                        and feature_point_names[0] != "None"
                    ):
                        data = np.hstack((sixs[key].values, add_vector))
                    else:
                        data = sixs[key].values

                    # run predictions
                    n_chunks = self.engine_config.predict_parallel_chunks
                    data_chunks = np.array_split(data, n_chunks, axis=0)

                    model_ref = ray.put(emulator)
                    result_refs = [
                        ray_predict.remote(
                            model_ref, x, self.engine_config.parallel_layer_read
                        )
                        for x in data_chunks
                    ]

                    lp = np.concatenate(ray.get(result_refs), axis=0)
                    Logger.debug(f"Cleanup {key}")
                    lp /= aux["response_scaler"][key]
                    lp += aux["response_offset"][key]

                    ltz = resample[key].values + lp < 0
                    lp[ltz] = -1 * resample[key].values[ltz]

                    predicts[key] = resample[key] + lp

                    elapsed_time = time.time() - key_start_time
                    Logger.debug(f"Predict time ({key}): {elapsed_time} seconds")
                    del result_refs, model_ref, emulator

                predicts.attrs["component_mode"] = "6c"

                elapsed_time = time.time() - total_start_time
                Logger.info(f"Total prediction: {elapsed_time} seconds")

            Logger.info(
                f"Saving intermediary prediction results to: {self.predict_path}"
            )
            luts.saveDataset(self.predict_path, predicts)

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
        """
        Post-simulation adjustments for sRTMnet.
        """
        # Update engine to run in RDN mode
        data = luts.load(self.predict_path, mode="r")
        outdict = {}
        Logger.debug("Resampling components")
        for key, values in data.items():
            Logger.debug(f"Resampling {key}")
            if (
                key in ["dir-dir", "dir-dif", "dif-dir", "dif-dif", "rhoatm"]
                and self.component_mode == "6c"
            ):
                fullspec_val = units.transm_to_rdn(
                    data[key].data, self.emulator_coszen, self.emulator_sol_irr
                )
            else:
                fullspec_val = data[key].data

            # Only resample and store valid keys
            if len(data[key].data.shape) > 0:
                outdict[key] = resample_spectrum(
                    fullspec_val, self.emu_wl, self.wl, self.fwhm, H=self.emulator_H
                )

        Logger.debug("Setting up lut cache")
        for _point, point in enumerate(data["point"].values):
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
        Logger.debug("Complete")


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
