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
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
import os

import click
import numpy as np
from spectral.io import envi

from isofit.core.common import envi_header

# Test args
root = "/Users/bgreenbe/Projects/MultiSurface/Notebooks/Glint/ang20230723t170821_sample/output"
inp = os.path.join(root, "ang20230723t170821_subs_rfl")
lbl_file = os.path.join(root, "ang20230723t170821_lbl")

output_root = "/Users/bgreenbe/Projects/MultiSurface/Debug/codes/reconstruction"
outp = os.path.join(output_root, "ang20230723t170821_recon_subs_rfl")


def reconstruct_subs(inp, outp, lbl_file):
    """Helper function to take the flat array that the superpixel
    algorithms work with and turn them into images at the full resolution
    of the input/output file. They will have the full array-resolution,
    but appear as coarser pixel-resolution images.

    args:
        inp: input file path
        outp: output file path
        lbl_file: file path to the label file to guide the reconstruction
    returns:
        None
    """
    # Load the input data
    subs_input = envi.open(envi_header(inp))
    subs_input_ar = np.squeeze(subs_input.open_memmap(interleave="bip"))

    lbl = envi.open(envi_header(lbl_file))
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
        envi_header(outp),
        ext="",
        metadata=output_metadata,
        force=True,
    )
    del out

    # Write file
    output = envi.open(envi_header(outp)).open_memmap(
        interleave="source", writable=True
    )
    output[...] = np.swapaxes(sub_full, 1, 2)
    del output


# Input arguments
@click.command(name="reconstruct_subs")
@click.argument("inp")
@click.argument("outp")
@click.argument("lbl_file")
def cli_reconstruct_subs(**kwargs):
    """Reconstruct a subs file to full resolution"""

    reconstruct_subs(**kwargs)
    click.echo("Done")
