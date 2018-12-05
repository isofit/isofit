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

from io import open
from setuptools import setup, find_packages

with open('README.rst', 'r') as f:
    long_description = f.read()
     
lic = 'Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)'

setup(name='isofit',
    version='0.5.2',
    url='http://github.com/davidraythompson/isofit/',
    license=lic,
    author='David R. Thompson, Winston Olson-Duvall, and Team',
    author_email='david.r.thompson@jpl.nasa.gov',
    description='Imaging Spectrometer Optimal FITting',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    packages=find_packages(),
    install_requires=['scipy>=1.1.0',
                      'numba>=0.38.0',
                      'matplotlib>=2.2.2',
                      'scikit-learn>=0.19.1',
                      'spectral>=0.19',
                      'pytest>=3.5.1',
                      'pep8>=1.7.1',
                      'xxhash>=1.2.0'],
    python_requires='>=3',
    platforms='any',
    classifiers=['Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent'])
