#! /usr/bin/env python
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
#         Philip G Brodrick, philip.brodrick@jpl.nasa.gov
#
__version__ = "3.4.3"


import logging

from threadpoolctl import threadpool_info

from isofit.debug import ray

Logger = logging.getLogger("isofit")


def checkNumThreads():
    """
    Checks the num_threads setting in the environment and raises a strong warning if it
    is not set to 1 .
    """
    threads = "unknown"
    error = False
    if info := threadpool_info():
        threads = info[0]["num_threads"]
        if threads > 1:
            error = "greater than"
    else:
        error = "not set to"

    if error:
        Logger.warning(
            f"""
******************************************************************************************
! Number of threads is {error} 1 (currently: {threads}), this may greatly impact performance
! Please set this the environment variables 'MKL_NUM_THREADS' and 'OMP_NUM_THREADS' to '1'
******************************************************************************************\
"""
        )
