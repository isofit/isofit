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

import os
import sys
import argparse
import scipy as s
from numpy.random import multivariate_normal
from os.path import abspath, split

sys.path.insert(0, '../isofit/')
from geometry import Geometry
from instrument import Instrument
from common import find_header, expand_path, json_load_ascii
from common import expand_all_paths, json_load_ascii, spectrumLoad


hdr_template = '''ENVI
samples = {samples}
lines   = {lines}
bands   = 1
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bsq
byte order = 0
'''


def percentile(X, p):
    S = sorted(X)
    return S[int(s.floor(len(S)*(p/100.0)))]


# Return the header associated with an image file
def find_header(imgfile):
    if os.path.exists(imgfile+'.hdr'):
        return imgfile+'.hdr'
    ind = imgfile.rfind('.raw')
    if ind >= 0:
        return imgfile[0:ind]+'.hdr'
    ind = imgfile.rfind('.img')
    if ind >= 0:
        return imgfile[0:ind]+'.hdr'
    raise IOError('No header found for file {0}'.format(imgfile))


# parse the command line (perform the correction on all command line arguments)
def main():

    desc = "Add noise to a radiance spectrum or image"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('config', type=str, metavar='INPUT')
    args = parser.parse_args(sys.argv[1:])
    config = json_load_ascii(args.config, shell_replace=True)
    configdir, configfile = split(abspath(args.config))

    infile = expand_path(configdir, config['input_radiance_file'])
    outfile = expand_path(configdir, config['output_radiance_file'])
    instrument = Instrument(config['instrument_model'])
    geom = Geometry()

    if infile.endswith('txt'):

        rdn, wl = spectrumLoad(infile)
        Sy = instrument.Sy(rdn, geom)
        rdn_noise = rdn + multivariate_normal(zeros(rdn.shape), Sy)

        with open(outfile, 'w') as fout:
            for w, r in zip(wl, rdn_noise):
                fout.write('%8.5f %8.5f' % (w, r))
    else:

        raise ValueError('image cubes not yet implemented')


if __name__ == "__main__":
    main()
