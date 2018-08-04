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

import pep8
import sys
from os.path import expandvars, split, abspath, join
from glob import glob

testdir, fname = split(abspath(__file__))
config_file = testdir+'/data/pep8_config.txt'
excludes = ['sunposition.py']

def test_pep8_conformance():
     """Test that we conform to PEP8."""
     directories = [testdir + '/../isofit/', testdir + '/../utils/']
     files = []
     for directory in directories:
        for fi in glob(directory+'*py'):
            if not any([e in fi for e in excludes]):
                files.append(fi)

     pep8style = pep8.StyleGuide(config_file = config_file, quiet = False)
     result = pep8style.check_files(files)
     if result.total_errors != 0:
        print('Found PEP8 conformance error.')
        print('Please fix your style with autopep8.')
     assert result.total_errors == 0


