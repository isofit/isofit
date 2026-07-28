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
# Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

import os
from typing import Dict, List, Type

from isofit import __version__
from isofit.configs.base_config import BaseConfigSection
from isofit.configs.sections.inversion_config import InversionConfig

# Imported for the device-string pattern only; isofit.core.backend defers its
# torch import, so this stays cheap and safe to import at config time.
from isofit.core.backend import DEVICE_RE


class ImplementationConfig(BaseConfigSection):
    def __init__(self, sub_configdic: dict = None):
        """
        Input file(s) configuration.
        """
        super().__init__()

        self._mode_type = str
        self.mode = "inversion"
        """
        str: Defines the operating mode for isofit. Current options are: inversion, inversion_mcmc,
        and 'simulation'.
        """

        self._inversion_type = InversionConfig
        self.inversion: InversionConfig = None
        """InversionConfig: optional config for running in inversion mode."""

        self._n_cores_type = int
        self.n_cores = None
        """int: number of cores to use."""

        self._task_inflation_factor_type = int
        self.task_inflation_factor = 10
        """int: Submit task_inflation_factor*n_cores number of tasks."""

        self._ip_head_type = str
        self.ip_head = None
        """str: Ray - parameter.  IP-head (for multi-node runs)."""

        self._redis_password_type = str
        self.redis_password = None
        """str: Ray - parameter.  Redis-password (for multi-node runs)."""

        self._ray_include_dashboard_type = bool
        self.ray_include_dashboard = False
        """str: Ray - parameter.  Boolean to include dashboard."""

        self._ray_temp_dir_type = str
        self.ray_temp_dir = "/tmp/ray"
        """str: Overrides the standard ray temporary directory.  Useful for multiuser systems."""

        self._ray_ignore_reinit_error_type = bool
        self.ray_ignore_reinit_error = True
        """bool: Boolean to tell ray to ignore re-initilaization.  Can be convenient for multiple Isofit instances."""

        self._io_buffer_size_type = int
        self.io_buffer_size = 100
        """bool: Integer indicating how large (how many spectra) of chunks to read/process/write.  A
        buffer size of 1 means pixels are processed independently.  Large buffers can help prevent IO choke points,
        especially if the """

        self._max_hash_table_size_type = int
        self.max_hash_table_size = 50
        """int: The maximum size of inversion hash tables.  Can provide speedups with redundant surfaces, but comes
        with increased memory costs.
        """

        self._per_pixel_heuristic_prior_type = bool
        self.per_pixel_heuristic_prior = False
        """bool: define prior mean setting scheme.
        True -> use per-pixel heuristic
        False -> Use image-wide "unverisal" value
        """

        self._debug_mode_type = bool
        self.debug_mode = False
        """bool: A flag to run the code in debug mode, which circumvents ray.
        """

        self._backend_type = str
        self.backend = "numpy"
        """str: Numerical backend used for retrievals. Options are:
        'numpy' (default) -> the standard per-pixel scipy/numpy code path,
        'torch' -> an opt-in batched backend that inverts many pixels at once,
        optionally on a GPU. The default is unchanged from historical ISOFIT
        behavior; the torch backend is selected only when explicitly requested.
        """

        self._torch_device_type = str
        self.torch_device = "auto"
        """str: Device for the torch backend. Options are 'auto', 'cpu', 'mps',
        'cuda', or 'cuda:N'. 'auto' prefers cuda, then mps, then cpu. An
        explicitly requested device that is unavailable raises an error rather
        than silently falling back to cpu. Ignored when backend is 'numpy'.
        """

        self._torch_batch_size_type = str
        self.torch_batch_size = "auto"
        """str: Number of spectra inverted per batched torch call, as an integer
        string or 'auto'. 'auto' sizes the batch from available device memory.
        Larger batches improve GPU utilization at the cost of memory. Ignored
        when backend is 'numpy'.
        """

        self._torch_dtype_type = str
        self.torch_dtype = "auto"
        """str: Floating point precision for the torch backend. Options are
        'auto', 'float32', or 'float64'. 'auto' selects float64 on cuda/cpu and
        float32 on mps (which does not implement float64). float64 is required
        for production fidelity; float32 is a development/performance mode.
        Ignored when backend is 'numpy'.
        """

        self._torch_num_gpu_workers_type = int
        self.torch_num_gpu_workers = None
        """int: Number of GPU worker actors for the torch backend. Defaults to
        the number of visible CUDA devices (one actor per GPU). Note that
        n_cores does not control GPU worker count.
        """

        self._isofit_version_type = str
        self.isofit_version = __version__
        """str: ISOFIT version used."""

        self.set_config_options(sub_configdic)

    def _check_config_validity(self) -> List[str]:
        errors = list()
        warnings = list()

        valid_implementation_modes = ["inversion", "mcmc_inversion", "simulation"]
        if self.mode not in valid_implementation_modes:
            errors.append(
                "Invalid implementation mode: {}.  Valid options are: {}".format(
                    self.mode, valid_implementation_modes
                )
            )

        if self.mode != "simulation" and self.inversion is None:
            errors.append(
                "If running outside of simulation mode, Inversion must be defined"
            )
        elif self.mode == "simulation" and self.inversion is None:
            # TODO: fix this
            errors.append(
                "Even in simulation mode, and inversion config must be defined, though"
                "it may be blank."
            )

        if int(self.ip_head is not None) + int(self.redis_password is not None) == 1:
            errors.append(
                "If either ip_head or redis_password are specified, both must be"
                " specified"
            )

        errors_b, warnings_b = self._check_backend_validity()
        errors += errors_b
        warnings += warnings_b

        return errors, warnings

    def _check_backend_validity(self) -> (List[str], List[str]):
        """Validate the numerical backend options.

        Device availability is deliberately NOT checked here: config objects are
        constructed inside Ray workers, so probing hardware at validation time
        would be both wrong and expensive. Availability is enforced at runtime by
        isofit.core.backend.resolve_device.
        """
        errors = list()
        warnings = list()

        valid_backends = ["numpy", "torch"]
        if self.backend not in valid_backends:
            errors.append(
                f"Invalid backend: {self.backend}.  Valid options are:"
                f" {valid_backends}"
            )

        if not DEVICE_RE.match(str(self.torch_device)):
            errors.append(
                f"Invalid torch_device: {self.torch_device}.  Valid options are:"
                " 'auto', 'cpu', 'mps', 'cuda', or 'cuda:N'"
            )

        valid_dtypes = ["auto", "float32", "float64"]
        if self.torch_dtype not in valid_dtypes:
            errors.append(
                f"Invalid torch_dtype: {self.torch_dtype}.  Valid options are:"
                f" {valid_dtypes}"
            )

        # MPS has no float64 implementation at all, so this combination can never
        # be satisfied - catch it at config time rather than deep in a worker.
        if self.torch_device == "mps" and self.torch_dtype == "float64":
            errors.append(
                "torch_dtype 'float64' is not supported on torch_device 'mps'."
                "  Use 'float32' or 'auto', or run on cuda/cpu for float64"
            )

        if str(self.torch_batch_size) != "auto":
            try:
                if int(self.torch_batch_size) <= 0:
                    errors.append(
                        f"torch_batch_size must be positive or 'auto', got:"
                        f" {self.torch_batch_size}"
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"Invalid torch_batch_size: {self.torch_batch_size}.  Must be"
                    " a positive integer or 'auto'"
                )

        if self.torch_num_gpu_workers is not None and self.torch_num_gpu_workers <= 0:
            errors.append(
                "torch_num_gpu_workers must be positive, got:"
                f" {self.torch_num_gpu_workers}"
            )

        if self.backend == "numpy":
            set_torch_opts = [
                name
                for name, default in (
                    ("torch_device", "auto"),
                    ("torch_batch_size", "auto"),
                    ("torch_dtype", "auto"),
                    ("torch_num_gpu_workers", None),
                )
                if getattr(self, name) != default
            ]
            if set_torch_opts:
                warnings.append(
                    f"The following options are ignored because backend is"
                    f" 'numpy': {set_torch_opts}.  Set backend to 'torch' to use"
                    " them"
                )

        if self.backend == "torch" and self.n_cores is not None:
            warnings.append(
                "n_cores does not control the number of GPU workers for the torch"
                " backend; use torch_num_gpu_workers.  n_cores still governs CPU"
                " stages such as segmentation and atmospheric interpolation"
            )

        return errors, warnings
