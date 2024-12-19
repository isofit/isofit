#! /usr/bin/env python3
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
# Author: Evan Greenberg, evan.greenberg@jpl.nasa.gov
import click
import numpy as np
from spectral.io import envi

from isofit.core.common import envi_header


def reconstruct_subs(input_subs_path, output_path, lbl_working_path):
    """Helper function to take the flat array that the superpixel
    algorithms work with and turn them into images at the full resolution
    of the input/output file. They will have the full array-resolution,
    but appear as coarser pixel-resolution images.\n

    args:

        input_subs_path:    Input subs file path.

        output_path:        Output reconstructed file path.

        lbl_working_path:   File path to label file for reconstruction.

    returns:
        None
    """
    # Load the input data
    subs_input = envi.open(envi_header(input_subs_path))
    subs_input_ar = np.squeeze(subs_input.open_memmap(interleave="bip"))

    lbl = envi.open(envi_header(lbl_working_path))
    lbl_ar = np.squeeze(lbl.open_memmap(interleave="bip"))

    # Make the reconstructed file
    sub_full = np.zeros((lbl_ar.shape[0], lbl_ar.shape[1], subs_input_ar.shape[1]))
    labels = [0] + list(np.unique(lbl_ar))
    for label, lbl_val in zip(labels, subs_input_ar):
        row, col = np.where(lbl_ar == label)
        sub_full[row, col, :] = lbl_val

    # Construct header and init file
    output_metadata = subs_input.metadata
    output_metadata["samples"] = sub_full.shape[1]
    output_metadata["lines"] = sub_full.shape[0]
    output_metadata["interleave"] = "bil"
    out = envi.create_image(
        envi_header(output_path),
        ext="",
        metadata=output_metadata,
        force=True,
    )
    del out

    # Write file
    output = envi.open(envi_header(output_path)).open_memmap(
        interleave="source", writable=True
    )
    output[...] = np.swapaxes(sub_full, 1, 2)
    del output


# Input arguments
@click.command(
    name="reconstruct_subs", help=reconstruct_subs.__doc__, no_args_is_help=True
)
@click.argument("input_subs_path")
@click.argument("output_path")
@click.argument("lbl_working_path")
def cli(**kwargs):

    reconstruct_subs(**kwargs)
    click.echo("Done")
