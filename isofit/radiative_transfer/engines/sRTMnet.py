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
import torch
import yaml
from scipy.interpolate import interp1d

from isofit.core import units
from isofit.core.common import calculate_resample_matrix, resample_spectrum
from isofit.radiative_transfer import luts
from isofit.radiative_transfer.engines import SixSRT
from isofit.radiative_transfer.radiative_transfer_engine import RadiativeTransferEngine

Logger = logging.getLogger(__file__)


class SRTMnetModel3c(torch.nn.Module):
    def __init__(self, input_file: str, n_cores: int = 1):
        """Initializes the SRTMnet model by loading weights and biases from an HDF5 file.
        This new version uses torch, but shoudl give the same results as the
        sRTMnet (tensorflow) and previous isofit (numpy) implementations.

        Args:
            input_file (str, optional): Path to the file containing model weights.
            n_cores (int, optional): Number of CPU cores to use if using cpu.
        """
        super().__init__()
        weights_list = []
        biases_list = []

        with h5py.File(input_file, "r") as model:
            for n in model["model_weights"].keys():
                if "dense" in n:
                    group = model["model_weights"][n][n]
                    if "kernel:0" in group:
                        w = group["kernel:0"][:]
                        b = group["bias:0"][:]
                    else:
                        w = group["kernel"][:]
                        b = group["bias"][:]
                    weights_list.append(torch.tensor(w, dtype=torch.float32))
                    biases_list.append(torch.tensor(b, dtype=torch.float32))

        # Register as parameters or buffers
        # Since we are not training, buffers might be arguably better, but Parameter is standard for weights.
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(w) for w in weights_list]
        )
        self.biases = torch.nn.ParameterList(
            [torch.nn.Parameter(b) for b in biases_list]
        )

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")

        self.to(self.device)
        self.eval()

        if self.device.type == "cpu":
            # This seems to work well despite global OMP / MKL options, though some systmes may need
            # those to be set differently
            torch.set_num_threads(n_cores)

    def forward(self, x):
        """Forward model structure
        Args:
            x (torch.Tensor): batched input data

        Returns:
            torch.Tensor: batched emulated data
        """
        # x: (batch, input_dim)
        # weights: (input_dim, output_dim)
        # (batch, input_dim) @ (input_dim, output_dim) -> (batch, output_dim)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.matmul(x, W) + b
            if i < len(self.weights) - 1:
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.4)
        return x

    @torch.inference_mode()
    def predict(self, 
                surrogate_data, 
                surrogate_data_emulator_wl, 
                batch_size=4096,
                response_scaler=None,
                response_offset=None,
                resample_dict=None,
                ):
        """predict model output

        Args:
            x: input data as numpy or dask array
            batch_size (int, optional): Size of batch to process. Defaults to 4096.

        Returns:
            np.array: emulated output
        """
        # Handle numpy input
        x_np = surrogate_data
        if isinstance(surrogate_data, da.Array):
            x_np = surrogate_data.compute()

        # Convert to tensor
        x_tensor = torch.as_tensor(x_np, dtype=torch.float32)

        results = []
        n = x_tensor.shape[0]

        for i in range(0, n, batch_size):
            batch = x_tensor[i : i + batch_size].to(self.device)
            out = self(batch)
            results.append(out.cpu().numpy())

        return np.concatenate(results, axis=0)


class SRTMnetModel6c(torch.nn.Module):
    def __init__(self, input_file: str, key: str = None, n_cores: int = 1):
        """Initializes the SRTMnet model by loading weights and biases from an HDF5 file.
        This new version uses torch, but shoudl give the same results as the
        sRTMnet (tensorflow) and previous isofit (numpy) implementations.

        Args:
            input_file (str, optional): Path to the file containing model weights.
            key (str, optional): Key to select specific weights in the file (6c format). If '3c', read all (sRTMnet 3c format).
            n_cores (int, optional): Number of CPU cores to use if using cpu.
        """
        super().__init__()

        # some keys, we'll legit go through uniquely.  Some, we must go through in pairs
        self.weights = torch.nn.ModuleDict()
        self.biases = torch.nn.ModuleDict()
        if key == "dir-dir":
            self.component_keys = ["transm_down_dir", "transm_up_dir"]
            self.product_name = key
        elif key == "dir-dif":
            self.component_keys = ["transm_down_dir", "transm_up_dif"]
            self.product_name = key
        elif key == "dif-dir":
            self.component_keys = ["transm_down_dif", "transm_up_dir"]
            self.product_name = key
        elif key == "dif-dif":
            self.component_keys = ["transm_down_dif", "transm_up_dif"]
            self.product_name = key
        elif key == '3c':
            self.component_keys = ['transm_down_dif', 'rhoatm', 'sphalb']
        else:
            self.component_keys = [key]

        if key != "3c": # 6c model
            with h5py.File(input_file, "r") as model:
                for ckey in self.component_keys:
                    w_list, b_list = [], []
                    w_group = model[f"weights_{ckey}"]
                    b_group = model[f"biases_{ckey}"]

                    for layer in w_group.keys():
                        w_list.append(
                            torch.nn.Parameter(
                                torch.tensor(w_group[layer][:], dtype=torch.float32)
                            )
                        )
                        b_list.append(
                            torch.nn.Parameter(
                                torch.tensor(b_group[layer][:], dtype=torch.float32)
                            )
                        )
                    # Register as parameters - could (perhaps should) use buffers
                    self.weights[ckey] = torch.nn.ParameterList(w_list)
                    self.biases[ckey] = torch.nn.ParameterList(b_list)
        else: # 3c model
            with h5py.File(input_file, "r") as model:
                w_list, b_list = [], []
                for n in model["model_weights"].keys():
                    if "dense" in n:
                        group = model["model_weights"][n][n]
                        if "kernel:0" in group:
                            w = group["kernel:0"][:]
                            b = group["bias:0"][:]
                        else:
                            w = group["kernel"][:]
                            b = group["bias"][:]
                        w_list.append(torch.tensor(w, dtype=torch.float32))
                        b_list.append(torch.tensor(b, dtype=torch.float32))
            self.weights['3c'] = torch.nn.ParameterList(w_list)
            self.biases['3c'] = torch.nn.ParameterList(b_list)

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")

        self.to(self.device)
        self.eval()

        if self.device.type == "cpu":
            # This seems to work well despite global OMP / MKL options, though some systmes may need
            # those to be set differently
            torch.set_num_threads(n_cores)

    def forward(self, x, key):
        """Forward model structure
        Args:
            x (torch.Tensor): batched input data

        Returns:
            torch.Tensor: batched emulated data
        """
        # x: (batch, input_dim)
        # weights: (input_dim, output_dim)
        # (batch, input_dim) @ (input_dim, output_dim) -> (batch, output_dim)
        for i, (W, b) in enumerate(zip(self.weights[key], self.biases[key])):
            x = torch.matmul(x, W) + b
            if i < len(self.weights[key]) - 1:
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.4)
        return x
    
    def batch_resample(self, out, key, resample_dict=None):
        # Resample the direct product, converting to radiance for rhoatm
        if resample_dict is not None:
            if key == "rhoatm":
                out_r = units.transm_to_rdn(
                    out,
                    resample_dict["emulator_coszen"],
                    resample_dict["emulator_sol_irr"],
                )
            else:
                out_r = out.copy()

            out_r = resample_spectrum(
                out_r,
                resample_dict["emu_wl"],
                resample_dict["wl"],
                resample_dict["fwhm"],
                H=resample_dict["emulator_H"],
            )
        return out_r

    @torch.inference_mode()
    def predict(
        self,
        surrogate_data,
        surrogate_data_emulator_wl,
        batch_size=4096,
        response_scaler=None,
        response_offset=None,
        resample_dict=None,
    ):
        """predict model output

        Args:
            x: input data as numpy or dask array
            batch_size (int, optional): Size of batch to process. Defaults to 4096.

        Returns:
            np.array: emulated output
        """
        # Handle numpy input
        x_tensor = [
            _x.compute() if isinstance(_x, da.Array) else _x for _x in surrogate_data
        ]
        x_tensor = [torch.as_tensor(_x, dtype=torch.float32) for _x in x_tensor]
        n = x_tensor[0].shape[0]

        outdict = {}
        for key in self.weights.keys():
            if key != '3c':
                outdict[key] = []
            else:
                outdict['transm_down_dif'] = []
                outdict['rhoatm'] = []
                outdict['sphalb'] = []

        is_paired = len(self.weights.keys()) > 1
        if is_paired:
            outdict[self.product_name] = []

        for i in range(0, n, batch_size):
            product = None
            for _key, key in enumerate(self.weights.keys()):
                batch_slice = slice(i, min(i + batch_size, n))
                batch = x_tensor[_key][batch_slice].to(self.device)

                out = self(batch, key)
                out = out.cpu().numpy()
                if response_scaler is not None:
                    out /= response_scaler[_key]
                if response_offset is not None:
                    out += response_offset[_key]
                out += surrogate_data_emulator_wl[_key][batch_slice]

                # Resample the direct product, converting to radiance for rhoatm
                if key != '3c':
                    outdict[key].append(self.batch_resample(out, key, resample_dict))

                    # For paired terms, convert to radiance and multiply
                    if is_paired:
                        if product is None:
                            product = out
                        else:
                            product *= out
                else:
                    nc = int(out.shape[1] / len(self.component_keys))
                    for _ckey, ckey in enumerate(self.component_keys):
                        outdict[ckey].append(self.batch_resample(out[:,_ckey * nc: (_ckey + 1)*nc], ckey, resample_dict))

            if is_paired: # only happens with 6c
                if resample_dict is not None:
                    product = units.transm_to_rdn(
                        product,
                        resample_dict["emulator_coszen"],
                        resample_dict["emulator_sol_irr"],
                    )
                    product = resample_spectrum(
                        product,
                        resample_dict["emu_wl"],
                        resample_dict["wl"],
                        resample_dict["fwhm"],
                        H=resample_dict["emulator_H"],
                    )
                outdict[self.product_name].append(product)

        # Concatenate all outputs from all batches
        for key in outdict.keys():
            outdict[key] = np.concatenate(outdict[key], axis=0)

        return outdict


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

        # Pack into dictionary for passing convenience to torch
        resample_dict = {
            "emu_wl": self.emu_wl,
            "wl": self.wl,
            "fwhm": self.fwhm,
            "emulator_H": self.emulator_H,
            "emulator_coszen": self.emulator_coszen,
            "emulator_sol_irr": self.emulator_sol_irr,
        }

        batch_size = 4096
        # Optional...perhaps we should just be explicit in the config about the batch sizing
        # if self.engine_config.predict_parallel_chunks > 0:
        #    batch_size = int(np.ceil(data.shape[0] / self.engine_config.predict_parallel_chunks))


        import multiprocessing

        n_cores = multiprocessing.cpu_count()
        Logger.info(f"Loading and predicting with emulator on {n_cores} cores")
        if self.component_mode == "3c":
            Logger.debug("Detected hdf5 (3c) emulator file format")

            # Stack the quantities together along a new dimension
            # named `quantity`
            resample = resample.to_array("quantity").stack(stack=["quantity", "wl"])

            ## Reduce from 3D to 2D by stacking along the wavelength
            # dim for each quantity. Convert to DataArray to stack
            # the variables along a new `quantity` dimension
            data = sixs.to_array("quantity").stack(stack=["quantity", "wl"])
            response_scaler = aux.get("response_scaler", 100.0)
            response_offset = aux.get("response_offset", 0.0)

            emulator = SRTMnetModel6c(
                    input_file=self.engine_config.emulator_file,
                    key='3c',
                    n_cores=n_cores,
                )
            lp = emulator.predict(
                    [data.values],  # surrogate data (6S)
                    [resample.values], #  stacked 3c data interpolated to emulator wl
                    batch_size=batch_size,
                    response_scaler=[response_scaler],
                    response_offset=[response_offset],
                    resample_dict=resample_dict,
                )
            outshape = (len(self.wl),) + tuple(
                len(self.lut_grid[n]) for n in self.lut_grid
            )
            for outkey in lp.keys():
                self.lut[outkey] = lp[outkey].T.reshape(outshape)
            self.lut.flush()



            # Now predict, scale, and add the interpolations
            #emulator = SRTMnetModel3c(self.engine_config.emulator_file, n_cores=n_cores)
            #predicts = da.from_array(emulator.predict([data]))
            #predicts /= scaler
            #predicts += response_offset
            #predicts += resample

            # Unstack back to a dataset and save
            #predicts = predicts.unstack("stack").to_dataset("quantity")
            #predicts.attrs["component_mode"] = "3c"

        else:
            Logger.debug("Detected 6c emulator file format")

            # This is an array of feature points tacked onto the interpolated 6s values
            feature_point_names = aux["feature_point_names"].astype(str).tolist()
            add_vector = None
            if len(feature_point_names) > 0 and feature_point_names[0] != "None":
                # Populate the 6S parameter values from a modtran template file
                with open(self.engine_config.template_file, "r") as file:
                    data = yaml.safe_load(file)["MODTRAN"][0]["MODTRANINPUT"]

                add_vector = np.zeros((self.points.shape[0], len(feature_point_names)))
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

            #predicts = resample.copy(deep=True)

            total_start_time = time.time()

            mapping = {
                "dir-dir": ["transm_down_dir", "transm_up_dir"],
                "dir-dif": ["transm_down_dir", "transm_up_dif"],
                "dif-dir": ["transm_down_dif", "transm_up_dir"],
                "dif-dif": ["transm_down_dif", "transm_up_dif"],
                "rhoatm": ["rhoatm"],
                "sphalb": ["sphalb"],
            }
            # for key in aux_rt_quantities:
            for key in mapping.keys():
                key_start_time = time.time()
                Logger.debug(f"Loading emulator {key}")

                emulator = SRTMnetModel6c(
                    input_file=self.engine_config.emulator_file,
                    key=key,
                    n_cores=n_cores,
                )

                Logger.info(f"Emulating {key}")
                response_scaler = [aux["response_scaler"][x] for x in mapping[key]]
                response_offset = [aux["response_offset"][x] for x in mapping[key]]

                lp = emulator.predict(
                    [sixs[x].values for x in mapping[key]],  # surrogate data (6S)
                    [
                        resample[x].values for x in mapping[key]
                    ],  #  6S data interpolated to emulator wl
                    batch_size=batch_size,
                    response_scaler=response_scaler,
                    response_offset=response_offset,
                    resample_dict=resample_dict,
                )
                Logger.debug(f"Cleanup {key}")
                del emulator

                outshape = (len(self.wl),) + tuple(
                    len(self.lut_grid[n]) for n in self.lut_grid
                )
                for outkey in lp.keys():
                    self.lut[outkey] = lp[outkey].T.reshape(outshape)
                self.lut.flush()

                elapsed_time = time.time() - key_start_time
                Logger.debug(f"Predict time ({key}): {elapsed_time} seconds")

            # predicts.attrs["component_mode"] = "6c"
            elapsed_time = time.time() - total_start_time
            Logger.info(f"Total prediction: {elapsed_time} seconds")

            self.rt_mode = "rdn"
            self.lut.setAttr("RT_mode", "rdn")
            self.lut.flush()

        # Logger.info(
        #    f"Saving intermediary prediction results to: {self.predict_path}"
        # )
        # luts.saveDataset(self.predict_path, predicts)

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
        #if self.engine_config.resample_inline is False:
        #    data = luts.load(self.predict_path, mode="r")
        #    outdict = {}
        #    Logger.debug("Resampling components")
        #    for key, values in data.items():
        #        Logger.debug(f"Resampling {key}")
        #        if (
        #            key in ["dir-dir", "dir-dif", "dif-dir", "dif-dif", "rhoatm"]
        #            and self.component_mode == "6c"
        #        ):
        #            fullspec_val = units.transm_to_rdn(
        #                data[key].data, self.emulator_coszen, self.emulator_sol_irr
        #            )
        #        else:
        #            fullspec_val = data[key].data

        #        # Only resample and store valid keys
        #        if len(data[key].data.shape) > 0:
        #            outdict[key] = resample_spectrum(
        #                fullspec_val, self.emu_wl, self.wl, self.fwhm, H=self.emulator_H
        #            )

        #    Logger.debug("Setting up lut cache")
        #    for _point, point in enumerate(data["point"].values):
        #        self.lut.queuePoint(
        #            np.array(point),
        #            {key: outdict[key][_point, :] for key in outdict.keys()},
        #        )
        #    Logger.debug("Flushing lut to file")
        #    self.lut.flush()

        #    # This is crude - we should revise the LUT naming and store L_* to make this
        #    # more explicit
        #    if "dir-dir" in outdict:
        #        self.rt_mode = "rdn"
        #        self.lut.setAttr("RT_mode", "rdn")
        #    Logger.debug("Complete")


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
