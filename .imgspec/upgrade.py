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

import argparse
import numpy as np
import os
import scipy as s
import spectral.io.envi as envi


# Return the header associated with an image file
def find_header(imgfile):
    if os.path.exists(imgfile + '.hdr'):
        return imgfile + '.hdr'
    ind = imgfile.rfind('.raw')
    if ind >= 0:
        return imgfile[0:ind] + '.hdr'
    ind = imgfile.rfind('.img')
    if ind >= 0:
        return imgfile[0:ind] + '.hdr'
    raise IOError('No header found for file {0}'.format(imgfile));


# parse the command line (perform the correction on all command line arguments)
def main():
    parser = argparse.ArgumentParser(description="Upgrade AVIRIS-C radiances")
    parser.add_argument('infile', type=str, metavar='INPUT')
    parser.add_argument('outfile', type=str, metavar='OUTPUT')
    parser.add_argument('--scaling', '-s', action='store')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    hdrfile = find_header(args.infile)
    hdr = envi.read_envi_header(hdrfile)
    hdr['data type'] = '4'
    hdr['byte order'] = '0'
    if hdr['interleave'] != 'bip':
        raise ValueError('I expected BIP interleave')
    hdr['interleave'] = 'bil'
    hdr['data ignore value'] = '-9999'
    envi.write_envi_header(args.outfile + '.hdr', hdr)
    lines = int(hdr['lines'])
    samples = int(hdr['samples'])
    bands = int(hdr['bands'])
    frame = samples * bands
    if args.verbose:
        print('Lines: %i  Samples: %i  Bands: %i\n' % (lines, samples, bands))

    if args.scaling is None:
        scaling = np.ones(bands, dtype=s.float32)
    else:
        scaling = s.loadtxt(args.scaling)

    prefix = os.path.split(args.infile)[-1][:3]
    if prefix in ['f95', 'f96', 'f97', 'f98', 'f99', 'f00',
                  'f01', 'f02', 'f03', 'f04', 'f05']:
        gains = s.r_[50.0 * np.ones(160), 100.0 * np.ones(64)]
    elif prefix in ['f06', 'f07', 'f08', 'f09', 'f10', 'f11',
                    'f12', 'f13', 'f14', 'f15', 'f16', 'f17',
                    'f18', 'f19', 'f20', 'f21']:
        gains = s.r_[300.0 * np.ones(110), 600.0 * np.ones(50), 1200.0 * np.ones(64)]
    else:
        raise ValueError('Unrecognized year prefix "%s"' % prefix)

    with open(args.infile, 'rb') as fin:
        with open(args.outfile, 'wb') as fout:
            for line in range(lines):
                X = np.fromfile(fin, dtype=s.int16, count=frame)
                X.byteswap(True)
                X = X.flatten()
                bad = X < -49
                X = np.array(X, dtype=s.float32)
                X = np.array(X.reshape((samples, bands)))
                X = scaling * X / gains
                X = X.flatten()
                X[bad] = -9999.0
                X = X.reshape((samples, bands))
                X = X.T.flatten()  # convert from BIP to BIL
                X = np.array(X, dtype=s.float32)
                X.tofile(fout)


if __name__ == "__main__":
    main()
