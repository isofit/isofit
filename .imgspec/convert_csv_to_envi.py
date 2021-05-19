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
# Author: Winston Olson-Duvall, winston.olson-duvall@jpl.nasa.gov
#

import sys

import pandas as pd
import spectral.io.envi as envi


def main():
    # Read in CSV file and convert to BIP ENVI file

    csv_path = sys.argv[1]
    output_path = csv_path.replace(".csv", "")
    output_hdr_path = output_path + ".hdr"

    spectra_df = pd.read_csv(csv_path, dtype="float32")
    spectra_df = spectra_df.fillna(-9999)
    lines, bands = spectra_df.shape
    wavelengths = spectra_df.keys().values
    hdr = {
        "lines": str(lines),
        "samples": "1",
        "bands": str(bands),
        "header offset": "0",
        "file type": "ENVI Standard",
        "data type": "4",
        "interleave": "bip",
        "byte order": "0",
        "data ignore value": "-9999",
        "wavelength": wavelengths
    }
    out_file = envi.create_image(output_hdr_path, hdr, ext='', force=True)
    output_mm = out_file.open_memmap(interleave='source', writable=True)

    # Iterate through dataframe and write to output memmap
    for index, row in spectra_df.iterrows():
        output_mm[index, 0, :] = row.values
    # Write to disk
    del output_mm


if __name__ == "__main__":
    main()