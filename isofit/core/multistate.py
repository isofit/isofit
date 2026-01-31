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
# Author: Evan Greenberg, evan.greenberg@jpl.nasa.gov
#
from __future__ import annotations

import logging
import time

import numpy as np
from scipy.interpolate import interp1d
from spectral.io import envi

from isofit.configs.sections.statevector_config import StateVectorElementConfig
from isofit.core.common import envi_header
from isofit.core.instrument import Instrument
from isofit.surface import Surface


class SurfaceMapping:
    surface_classes = {
        0: "multicomponent_surface",
        1: "glint_model_surface",
        2: "surface_lut",
        3: "surface_thermal",
    }

    def __class_getitem__(self, val):
        error_message = (
            "Classification int does not match any supported "
            "surface model categories. "
            "Check classification file values or supported "
            "category mappings."
        )
        if isinstance(val, int):
            surface_class = self.surface_classes.get(val)
            if not surface_class:
                raise ValueError(error_message)
            return surface_class

        elif isinstance(val, str):
            match_i = -1
            for i, value in self.surface_classes.items():
                if value == val:
                    match_i = i
            if match_i == -1:
                raise ValueError(error_message)
            return match_i

        else:
            raise ValueError("Invalid input data type")


def construct_full_state(full_config):
    """
    Get the full statevec. I don't love how this is done.
    Should be set up so that it can be pulled right out of the config.
    This is not currently possible. Have to pull rfl states from
    initialized surface class. Looks at all the model-states present in
    the config and collapses them into a single image-universal statevector.
    Returns both the names and indexes of the image-wide statevector.
    Args:
        full_config: full Isofit config
    Returns:
        full_statevec: list of the combined rfl, surf_non_rfl, RT and instrument state names
        full_idx_surface: np.array of the combined rfl, surf_non_rfl state indexes
        full_idx_surf_rfl: np.array of the combined rfl state indexes
        full_idx_surf_nonrfl: np.array of the combined surf_non_rfl state indexes
        full_idx_RT: np.array of the combined RT state indexes
        full_idx_instrument: np.array of the combined instrument state indexes
    """
    rfl_states = []
    nonrfl_states = []

    # Not sure if there are ever instrument statevec elements?
    instrument = Instrument(full_config)
    instrument_states = instrument.statevec_names

    # Pull the rt names from the config. Seems to be most commonly present.
    rt_config = full_config.forward_model.radiative_transfer

    rt_states = vars(rt_config.radiative_transfer_engines[0])["statevector_names"]
    if not rt_states:
        rt_states = sorted(rt_config.radiative_transfer_engines[0].lut_names.keys())

    # Check for config type
    if full_config.forward_model.surface.multi_surface_flag:
        # Iterate through the different surfaces to find overlapping state names
        for surface_class_str in full_config.forward_model.surface.Surfaces.keys():
            full_config = update_config_for_surface(full_config, surface_class_str)
            surface = Surface(full_config)
            rfl_states += surface.statevec_names[: len(surface.idx_lamb)]
            nonrfl_states += surface.statevec_names[len(surface.idx_lamb) :]

    else:
        surface = Surface(full_config)
        rfl_states += surface.statevec_names[: len(surface.idx_lamb)]
        nonrfl_states += surface.statevec_names[len(surface.idx_lamb) :]

    # Find unique state elements and collapse - ALPHABETICAL
    rfl_states = sorted(list(set(rfl_states)))
    nonrfl_states = sorted(list(set(nonrfl_states)))

    # Rejoin in the same order as the original statevector object
    full_statevec = rfl_states + nonrfl_states + rt_states + instrument_states

    # Set up full idx arrays
    full_idx_surface = np.arange(0, len(rfl_states) + len(nonrfl_states))

    start = 0
    full_idx_surf_rfl = np.arange(start, len(rfl_states))

    start += len(rfl_states)
    full_idx_surf_nonrfl = np.arange(start, start + len(nonrfl_states))

    start += len(nonrfl_states)
    full_idx_rt = np.arange(start, start + len(rt_states))

    start += len(rt_states)
    full_idx_instrument = np.arange(start, start + len(instrument_states))

    return (
        full_statevec,
        full_idx_surface,
        full_idx_surf_rfl,
        full_idx_surf_nonrfl,
        full_idx_rt,
        full_idx_instrument,
    )


def index_spectra_by_surface(config, index_pairs, force_full_res=False):
    """
    Indexes an image by a provided surface class file.
    Could extend it to be indexed by an atomspheric classification
    file as well if you want to vary both surface and atmospheric
    state.
    Args:
        surface_config: (Config object) The surface component of the
                        main config.
        index_pairs:
    Returns:
        class_groups: (dict) where keys are the pixel classification (name)
                      and values are tuples of rows and columns for each
                      group.
    """
    surface_config = config.forward_model.surface

    """Check if the class files exist. Defaults to run all pixels.
    This accomodates the test cases where we test the multi-surface,
    but don't use a classification file."""
    if not surface_config.surface_class_file:
        return {"uniform_surface": index_pairs}

    if force_full_res:
        class_file = surface_config.base_surface_class_file
    else:
        class_file = surface_config.surface_class_file

    classes = np.squeeze(
        envi.open(envi_header(class_file)).open_memmap(interleave="bip"), axis=-1
    ).astype(int)

    class_groups = {}
    for surface_name, surface_dict in surface_config.Surfaces.items():
        surface_index_pairs = np.argwhere(classes == SurfaceMapping[surface_name])
        class_groups[surface_name] = surface_index_pairs

    del classes

    return class_groups


def index_spectra_by_surface_view(config, index_pairs, force_full_res=False):
    """
    Indexes an image by a provided surface class file.
    Could extend it to be indexed by an atomspheric classification
    file as well if you want to vary both surface and atmospheric
    state.
    Args:
        surface_config: (Config object) The surface component of the
                        main config.
    Returns:
        class_groups: (dict) where keys are the pixel classification (name)
                      and values are tuples of rows and columns for each
                      group.
    """

    start_time = time.time()
    surface_config = config.forward_model.surface

    """Check if the class files exist. Defaults to run all pixels.
    This accomodates the test cases where we test the multi-surface,
    but don't use a classification file."""
    if not surface_config.surface_class_file:
        return {"uniform_surface": index_pairs}

    if force_full_res:
        class_file = surface_config.base_surface_class_file
    else:
        class_file = surface_config.surface_class_file

    classes = np.squeeze(
        envi.open(envi_header(class_file)).open_memmap(interleave="bip"), axis=-1
    )

    class_groups = {}
    for c, surface_sub_config in surface_config.Surfaces.items():
        surface_pixel_list = np.ascontiguousarray(
            np.argwhere(classes == surface_sub_config["surface_int"]).astype(int)
        )

        if not len(surface_pixel_list):
            continue

        # The strategy here is to produce a view where the columns are read together.
        # Both the index_pairs and surface_pixel_list have to be contiguous arrays.
        ncols = index_pairs.shape[1]
        dtype = {
            "names": ["{}".format(i) for i in range(ncols)],
            "formats": ncols * [index_pairs.dtype],
        }

        surface_index_pairs = np.intersect1d(
            index_pairs.view(dtype), surface_pixel_list.view(dtype)
        )
        surface_index_pairs = np.reshape(
            surface_index_pairs.view(index_pairs.dtype), (len(surface_index_pairs), 2)
        )

        class_groups[c] = surface_index_pairs

    del classes

    return class_groups


def update_config_for_surface(config, surface_class_str, clouds=True):
    """
    This is the primary mechanism by which isofit changes its configuration
    across surface classifications. It will leverage the Surfaces dict,
    and then update the primary config key to reflect that surface.
    Args:
        config: (Config object) Full isofit config object.
        surface_class_str: (str) string that corresponds to a surface class.
    Returns:
        config: (Config object) Update full isofit config object
    """
    if not config.forward_model.surface.surface_class_file:
        return config

    isurface = config.forward_model.surface.Surfaces.get(surface_class_str)

    if not isurface:
        raise KeyError("Multi-surface flag used, but no valid multi-surface config")

    surface_category = isurface.get("surface_category")
    surface_file = isurface.get("surface_file")
    glint_model = isurface.get("glint_model")

    config.forward_model.surface.surface_category = surface_category
    config.forward_model.surface.surface_file = surface_file
    config.forward_model.surface.glint_model = glint_model

    # Experimental: added statevector elements
    for key, value in isurface.get("rt_statevector_elements", {}).items():
        # Add the statevector params
        config.forward_model.radiative_transfer.statevector.surface_elevation_km = (
            StateVectorElementConfig(value)
        )

        # Add the statevector names
        config.forward_model.radiative_transfer.radiative_transfer_engines[
            0
        ].statevector_names.append(key)

    return config


def match_statevector(
    state_data: np.array, full_statevec: list, fm_statevec: list, null_value=-9999.0
):
    """
    A multi-class surface requires some merging across statevectors
    of different length. This function maps the fm-specific state
    to the io-state that captures all state elements present in the
    image. The full_state will record a Non
    Args:
        state_data: (n,) numpy array with the fm-specific state vector
        full_statevec: [m] list of state-names of the image-universal combined statevector
        fm_statevec: [n] list of state-names of the fm-specific state vector
        null_value: (optional) value to fill in the statevector elements that aren't present at a pixel
    returns:
        full_state: (np.array) Populated full state with null_values in missing elements
    """
    full_state = np.zeros((len(full_statevec))) + null_value
    idx = []
    for fm_name in fm_statevec:
        for i, full_name in enumerate(full_statevec):
            if fm_name == full_name:
                idx.append(i)
    full_state[idx] = state_data

    return full_state
