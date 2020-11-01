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
from isofit import __version__

with open('README.rst', 'r') as f:
    LONG_DESCRIPTION = f.read()

LICENSE = "Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)"

setup(name='isofit',
      version=__version__,
      url='http://github.com/isofit/isofit/',
      license=LICENSE,
      author='David R. Thompson, Winston Olson-Duvall, Philip G. Brodrick, and Team',
      author_email='david.r.thompson@jpl.nasa.gov',
      description='Imaging Spectrometer Optimal FITting',
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/x-rst',
      packages=find_packages(),
      include_package_data=True,
      scripts=['bin/isofit',
               'bin/sunposition'],
      install_requires=['numpy>=1.11,<1.19.0',
                        'scipy>=1.3.0',
                        'matplotlib>=2.2.2',
                        'scikit-learn>=0.19.1',
                        'scikit-image>=0.17.0',
                        'spectral>=0.19',
                        'pytest>=3.5.1',
                        'pep8>=1.7.1',
                        'xxhash>=1.2.0',
                        'pyyaml>=5.3.1',
                        'ray==0.8.5',
                        'pandas>=0.24',
                        'tensorflow>=2.0.1'],
      python_requires='>=3',
      platforms='any',
      classifiers=['Programming Language :: Python :: 3',
                   'License :: OSI Approved :: Apache Software License',
                   'Operating System :: OS Independent'])
