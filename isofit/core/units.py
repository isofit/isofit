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
# Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
#          Niklas Bohn, urs.n.bohn@jpl.nasa.gov
#          Evan Greenberg

import numpy as np


def E_to_L(E, coszen):
    """Convert irradiance to radiance.

    Args:
        E:         input irradiance vector
        coszen:    cosine of solar zenith angle

    Returns:
        Data vector converted to radiance
    """
    L = E * coszen / np.pi
    return L

def L_to_E(L, coszen):
    """Convert radiance to irradiance.

    Args:
        L:         input radiance vector
        coszen:    cosine of solar zenith angle

    Returns:
        Data vector converted to irradiance
    """
    E = L * np.pi / coszen
    return E

def transm_to_rdn(transm, coszen, solar_irr):
    """Function to convert a unitless atmospheric vector to radiance units.

    Args:
        transm:    input data vector in unitless atmospheric units
        coszen:    cosine of solar zenith angle
        solar_irr: solar irradiance vector

    Returns:
        Data vector converted to radiance
    """
    rdn = transm * E_to_L(solar_irr, coszen)
    return rdn


def rdn_to_transm(rdn, coszen, solar_irr):
    """Function to convert a radiance vector to transmittance.

    Args:
        rdn:       input data vector in radiance
        coszen:    cosine of solar zenith angle
        solar_irr: solar irradiance vector

    Returns:
        Data vector converted to unitless atmospheric units
    """
    transm = rdn / E_to_L(solar_irr, coszen)
    return transm

