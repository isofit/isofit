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
import logging

import numpy as np
from scipy.interpolate import interp1d
from spectral.io import envi

from isofit.configs import Config
from isofit.core.common import envi_header, load_spectrum, load_wavelen
from isofit.core.forward import ForwardModel
from isofit.core.instrument import Instrument
from isofit.radiative_transfer.radiative_transfer import RadiativeTransfer
from isofit.surface.surfaces import Surfaces


def match_class(class_groups, row, col):
    """
    Pass this function the row column pair and it will return the
    key from class_groups for which that row-col belongs.

    Args:

        class_groups: (dict) Keys are the pixel groups. Values are tuples
                      of rows and column that belong in the respective groups.
        row: (int) row of queried pixel
        col: (int) col of queried pixel
    """
    # If there is no class index, return base
    if not len(class_groups):
        return "0"

    # else match
    matches = np.zeros((len(class_groups))).astype(int)
    for i, group in enumerate(class_groups):
        if [row, col, 0] in group:
            matches[i] = 1
        else:
            matches[i] = 0

    if len(matches[np.where(matches)]) < 1:
        logging.exception(
            "Pixel did not match any class. \
                         Something is wrong"
        )
        raise ValueError

    elif len(matches[np.where(matches)]) > 1:
        logging.exception(
            "Pixel matches too many classes. \
                         Something is wrong"
        )
        raise ValueError

    return str(np.argwhere(matches)[0][0])


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

    """
    This method of retrieving the rt states is giving me issues between
    legacy configs and current configs. What is the most stable place
    to pull the statevector name list?.
    statevector_names not always present.
    is lut_names always present? Is always a dict?

    most stable is to iterate across statevector config, 
    but I have to match out the _type -> bad
    """
    rt_states = sorted(rt_config.radiative_transfer_engines[0].lut_names.keys())

    # Without changing where the nonrfl surface elements are defined
    surface_config = full_config.forward_model.surface
    params = surface_config.surface_params

    # Iterate through the different surfaces to find overlapping state names
    for i, surface_config in full_config.forward_model.surface.Surfaces.items():
        surface = Surfaces[surface_config["surface_category"]](surface_config, params)
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


def index_image_by_class(surface_config, subs=True):
    """
    Indexes an image by a provided surface class file.
    Could extend it to be indexed by an atomspheric classification
    file as well if you want to vary both surface and atmospheric
    state.

    Args:
        surface_config: (Config object) The surface component of the
                        main config.
        subs: (optional) (bool) that tells function which classification
              file to use.

    Returns:
        class_groups: (dict) where keys are the pixel classification (index)
                      and values are tuples of rows and columns for each
                      group.
    """

    if vars(surface_config).get("sub_surface_class_file") and subs:
        class_file = surface_config.sub_surface_class_file
    else:
        class_file = surface_config.surface_class_file

    classes = envi.open(envi_header(class_file)).open_memmap(interleave="bip")

    class_groups = []
    for c in surface_config.Surfaces.keys():
        pixel_list = np.argwhere(classes == int(c)).astype(int).tolist()
        class_groups.append(pixel_list)

    del classes

    return class_groups


def index_image_by_class_and_sub(config, lbl_file):
    """
    Indexes an image by surface class file and lbl_file.
    This is needed for the analytical line where each pixel needs to
    inherit the surface classification of the slic pixel that is belongs
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
    ds = envi.open(envi_header(lbl_file))
    im = ds.load()
    if config.forward_model.surface.multi_surface_flag:
        sub_pixel_index = index_image_by_class(config.forward_model.surface)
        pixel_index = []
        for class_subs in sub_pixel_index:
            if not len(class_subs):
                pixel_index.append([])
                continue

            class_pixel_index = []
            for i in class_subs:
                class_pixel_index += np.argwhere(im == i).tolist()

            pixel_index.append(class_pixel_index)
    else:
        pixel_index = []

    return pixel_index
