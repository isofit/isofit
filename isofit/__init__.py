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
### Variables ###
import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)

import logging
import sys
from pathlib import Path

from threadpoolctl import threadpool_info

from isofit.debug import ray

Logger = logging.getLogger("isofit")


def checkNumThreads():
    """
    Checks the num_threads setting in the environment and raises a strong warning if it
    is not set to 1 .
    """
    error = False
    if info := threadpool_info():
        if info[0]["num_threads"] > 1:
            error = "greater than"
    else:
        error = "not set to"

    if error:
        Logger.warning(
            f"""
******************************************************************************************
! Number of threads is {error} 1, this may greatly impact performance
! Please set this the environment variables 'MKL_NUM_THREADS' and 'OMP_NUM_THREADS' to '1'
******************************************************************************************\
"""
        )


def setupLogging(level="INFO", path=None, reset=False):
    """
    Initializes the ISOFIT logger

    Parameters
    ----------
    TODO
    """
    formats = {
        "DEBUG": logging.Formatter(
            fmt="{asctime} | {levelname:7} | {filename}:{funcName}:{lineno} | {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
        ),
        "INFO": logging.Formatter(
            fmt="{asctime} | {levelname:7} | {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
        ),
    }

    terminal = logging.StreamHandler(sys.stdout)
    terminal.setLevel(level)
    terminal.setFormatter(formats.get(level, formats["INFO"]))

    handlers = [terminal]

    if path:
        path = Path(path)
        mode = "a"
        if path.exists() and reset:
            mode = "w"

        # Single-file logging is always debug
        if path.suffix:
            fh = logging.FileHandler(path, mode=mode)
            fh.setLevel("DEBUG")
            fh.setFormatter(formats["DEBUG"])
            handlers.append(fh)

        # Multi-file logging provides both info and debug
        else:
            path.mkdir(exist_ok=True, parents=True)
            for lvl in ("INFO", "DEBUG"):
                fh = logging.FileHandler(path / f"{lvl.lower()}.log", mode=mode)
                fh.setLevel(lvl)
                fh.setFormatter(formats[lvl])
                handlers.append(fh)

    logging.basicConfig(
        level="DEBUG",
        handlers=handlers,
    )
