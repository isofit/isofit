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
from isofit.core.instrument import Instrument
from isofit.core.state import StateVector
from isofit.radiative_transfer.radiative_transfer import RadiativeTransfer
from isofit.surface.surfaces import Surfaces

from ..core.common import envi_header, load_spectrum, load_wavelen


def index_image_by_class(surface_config, subs=True):
    if subs:
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


def match_class(class_groups, row, col):
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

    return str(matches[np.where(matches)][0])


def construct_full_state(full_config):
    """
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
    RT_states = []
    instrument_states = []

    instrument = Instrument(full_config)
    RT = RadiativeTransfer(full_config)

    surface_config = full_config.forward_model.surface
    params = surface_config.surface_params

    # Iterate through the different states to find overlapping state names
    for i, surface in full_config.forward_model.surface.Surfaces.items():
        surface = Surfaces[surface["surface_category"]](surface, params)
        state = StateVector(instrument, RT, surface)

        rfl_states += [state.statevec[i] for i in state.idx_surf_rfl]
        nonrfl_states += [state.statevec[i] for i in state.idx_surf_nonrfl]
        RT_states += [state.statevec[i] for i in state.idx_RT]
        instrument_states += [state.statevec[i] for i in state.idx_instrument]

    # Find unique state elements and collapse - ALPHABETICAL
    rfl_states = sorted(list(set(rfl_states)))
    nonrfl_states = sorted(list(set(nonrfl_states)))
    RT_states = sorted(list(set(RT_states)))
    instrument_states = sorted(list(set(instrument_states)))

    # Rejoin in the same order as the original statevector object
    full_statevec = rfl_states + nonrfl_states + RT_states + instrument_states

    # Set up full idx arrays
    full_idx_surface = np.arange(0, len(rfl_states) + len(nonrfl_states))

    start = 0
    full_idx_surf_rfl = np.arange(start, len(rfl_states))

    start += len(rfl_states)
    full_idx_surf_nonrfl = np.arange(start, start + len(nonrfl_states))

    start += len(nonrfl_states)
    full_idx_RT = np.arange(start, start + len(RT_states))

    start += len(RT_states)
    full_idx_instrument = np.arange(start, start + len(instrument_states))

    return (
        full_statevec,
        full_idx_surface,
        full_idx_surf_rfl,
        full_idx_surf_nonrfl,
        full_idx_RT,
        full_idx_instrument,
    )
