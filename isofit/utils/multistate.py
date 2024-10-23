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
from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d
from spectral.io import envi

from isofit.configs.sections.statevector_config import StateVectorElementConfig
from isofit.core.common import envi_header
from isofit.core.instrument import Instrument
from isofit.surface import Surface


def construct_full_state(full_config):
    """
    Get the full statevec. I don't like how this is done.
    Should be set up so that it can be pulled right out of the config

    Looks at all the model-states present in the config and collapses
    them into a single image-universal statevector. Returns both
    the names and indexes of the image-wide statevector.

    Returns:
        self.full_statevec: [m] list of the combined
                       rfl, surf_non_rfl, RT and instrument state names
        self.full_idx_surface: [n] np.array of the combined
                       rfl, surf_non_rfl state indexes
        self.full_idx_surf_rfl: [n] np.array of the combined
                       rfl state indexes
        self.full_idx_surf_nonrfl: [n] np.array of the combined
                       surf_non_rfl state indexes
        self.full_idx_RT: [n] np.array of the combined
                       RT state indexes
        self.full_idx_instrument: [n] np.array of the combined
                       instrument state indexes
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

    return full_statevec, full_idx_surface, full_idx_surf_rfl, full_idx_rt


def index_spectra_by_surface(config, index_pairs, sub=True):
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

    surface_config = config.forward_model.surface

    """Check if the class files exist. Defaults to run all pixels.
    This accomodates the test cases where we test the multi-surface,
    but don't use a classification file."""
    if (
        not surface_config.sub_surface_class_file
        and not surface_config.surface_class_file
    ):
        return {"uniform_surface": index_pairs}

    if vars(surface_config).get("sub_surface_class_file") and sub:
        class_file = surface_config.sub_surface_class_file
    else:
        class_file = surface_config.surface_class_file

    classes = np.squeeze(
        envi.open(envi_header(class_file)).open_memmap(interleave="bip"), axis=-1
    )

    class_groups = {}
    for c, surface_sub_config in surface_config.Surfaces.items():
        surface_pixel_list = np.argwhere(
            classes == surface_sub_config["surface_int"]
        ).astype(int)

        if not len(surface_pixel_list):
            continue

        # Find intersection between index_pairs and pixel_list
        in_surface_index = (index_pairs[:, None] == surface_pixel_list).all(-1).any(1)
        surface_index_pairs = index_pairs[in_surface_index, ...]

        class_groups[c] = surface_index_pairs

    del classes

    return class_groups


def index_spectra_by_surface_and_sub(config, lbl_file):
    """
    Indexes an image by surface class file and lbl_file.
    This is needed for the analytical line where each pixel needs to
    inherit the surface classification of the slic pixel that it belongs
    to. This function looks at the slic pixel indexing and then creates
    a list of all full img pixels found within each slic pixel for each
    surface class.

    Args:
        config: (Config object) Full isofit config object.
              file to use.
        lbl_file: Path to the label file produced by the slic algorithm.

    Returns:
        pixel_index: (list) List of row-col pairs in each class.
                     index of list matches the class key. Empty list
                     returned if there is no multistate.
    """
    # Get all index pairs in image
    lbl = envi.open(envi_header(lbl_file)).open_memmap(interleave="bip")
    lbl_shape = (range(lbl.shape[0]), range(lbl.shape[1]))
    index_pairs = np.vstack([x.flatten(order="f") for x in np.meshgrid(*lbl_shape)]).T

    sub_pixel_index = index_spectra_by_surface(config, index_pairs)

    pixel_index = {}
    class_groups = {}
    for surface_class_str, class_subs in sub_pixel_index.items():
        if not len(class_subs):
            continue

        class_pixel_index = []
        for i in class_subs:
            class_pixel_index += np.argwhere(lbl == i).tolist()

        class_groups[surface_class_str] = class_pixel_index

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
    isurface = config.forward_model.surface.Surfaces.get(surface_class_str)

    if not isurface:
        raise KeyError("Multi-surface flag used, but no multi-surface config")

    surface_category = isurface.get("surface_category")
    surface_file = isurface.get("surface_file")
    glint_model = isurface.get("glint_model")

    if (not surface_category) or (not surface_file):
        raise KeyError("Failed to parse multi-surface config")

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
