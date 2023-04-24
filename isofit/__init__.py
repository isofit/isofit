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

warnings_enabled = False

import logging
import os

Logger = logging.getLogger("isofit")

if os.environ.get("ISOFIT_DEBUG"):
    Logger.info("Using ISOFIT internal ray")
    from .wrappers import ray
else:
    import ray

import click


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("-v", "--version", help="Print the current version", is_flag=True)
@click.option("-p", "--path", help="Print the installation path", is_flag=True)
def cli(ctx, version, path):
    """\
    This houses the subcommands of ISOFIT
    """
    if ctx.invoked_subcommand is None:
        if version:
            click.echo(__version__)

        if path:
            click.echo(__path__[0])


# Import all of the files that define a _cli command to register them
import isofit.core.isofit
import isofit.utils.add_HRRR_profiles_to_modtran_config
import isofit.utils.analytical_line
import isofit.utils.apply_oe
import isofit.utils.multisurface_oe
import isofit.utils.solar_position
