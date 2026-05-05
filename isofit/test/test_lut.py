#!/usr/bin/env python3
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
# Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
#          Niklas Bohn, urs.n.bohn@jpl.nasa.gov
#          James Montgomery, j.montgomery@jpl.nasa.gov
#

import pytest

from isofit.configs import configs
from isofit.atmosphere.engines import ModtranRT


@pytest.mark.xfail
def test_combined(monkeypatch):
    """Test class reuse."""

    monkeypatch.chdir("examples/20171108_Pasadena/")

    print("Loading config file from the Pasadena example.")
    config_file = "configs/ang20171108t184227_beckmanlawn-multimodtran-topoflux.json"
    full_config = configs.create_new_config(config_file)

    print("Initialize radiative transfer engine without prebuilt LUT file.")
    engine = ModtranRT(full_config=full_config)

    print("Run radiative transfer simulations.")
    engine.runSimulations()
    del engine
