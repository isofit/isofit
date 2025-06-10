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
#          Evan Greenberg, evan.greenberg@jpl.nasa.gov

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


def m_wavenumber_to_nm(wavenumber):
    """
    Function to convert wavenumber to nm.

    Args:
        wavenumber:     value(s) in wavenumber in 1/m

    Returns:
        value(s) in nanometers
    """
    nm = 1e9 / (wavenumber)
    return nm


def cm_wavenumber_to_nm(wavenumber):
    """
    Function to convert wavenumber to nm.

    Args:
        wavenumber:     value(s) in wavenumber in 1/cm

    Returns:
        value(s) in nanometers
    """
    nm = 1e7 / (wavenumber)
    return nm


def micron_to_nm(micron):
    """
    Function to convert microns to nanometers

    Args:
        micron: value(s) in microns

    Returns:
        value(s) in nanometers
    """
    nm = micron * 1000
    return nm


def nm_to_micron(nm):
    """
    Function to convert nanometers to microns

    Args:
        nm:     value(s) in nanometers

    Returns:
        value(s) in microns
    """
    micron = nm / 1000
    return micron


def vis_to_aod(vis):
    """
    Converts VIS modtran parameter to the AOD550 .
    This formula comes from page 50 of the MODTRAN 6 user manual,
    which relates ViS to the extinction coefficient at 550 nm.
    The constant, 0.01159 is the Rayleigh scattering coefficient at
    550 nm in 1/km.

    Args:
        vis:    visibility in km

    Returns:
        Data vector converted to exctinction at 550 nm
    """
    aod = (np.log(50) / vis) - 0.01159
    return aod


def aod_to_vis(aod):
    """
    Converts AOD550 to VIS modtran parameter.
    Formula comes from page 50 of the MODTRAN 6 users manual,
    which relates ViS to the extinction coefficient at 550 nm.
    The constant, 0.01159 is the Rayleigh scattering coefficient at
    550 nm in 1/km.

    Args:
        aod:    extinction at 550 nm

    Returns:
        Data vector converted to visibility in km
    """
    vis = np.log(50) / (aod + 0.01159)
    return vis


def rfl_to_rrs(rfl):
    """
    Converts unitless reflectance to remote sensing reflectance

    Args:
        rfl:    unitless reflectance

    Returns:
        Data vector of remote sensing reflectance in units 1/sr
    """
    rrs = rfl / np.pi
    return rrs


def rrs_to_rfl(rrs):
    """
    Converts remote sensing reflectance to unitless reflectance

    Args:
        rrs:   remote sensing reflectance in 1/sr

    Returns:
        Data vector of unitless reflectance
    """
    rfl = rrs * np.pi
    return rfl


def Wm2_to_uWcm2(wm):
    """
    Converts value of units Watts / square meter to units
    of micro-watts / square centimeter

    Args:
        wm:     value(s) in units of W/m2

    Returns:
        Value(s) in units of uW/cm2
    """
    uwcm = wm * 1e2
    return uwcm


def uWcm2_to_Wm2(uwcm):
    """
    Converts value of micro-watts / square centimeter to units Watts / square meter

    Args:
        uwcm:     value(s) in units of uW/cm2

    Returns:
        Value(s) in units of W/m2
    """
    wm = uwcm / 1e2
    return wm


def m_to_km(m):
    """
    Converts value of units meters to units kilometer

    Args:
        m:     value(s) in units of meters

    Returns:
        Value(s) in units of kilometers
    """
    km = m / 1000
    return km


def km_to_m(km):
    """
    Converts value of units kilometers to unit meters

    Args:
        km:     value(s) in units of kilometers

    Returns:
        Value(s) in units of meters
    """
    m = km * 1000
    return m


def m_to_ft(m):
    """
    Converts value of units meters to units feet

    Args:
        m:     value(s) in units of meters

    Returns:
        Value(s) in units of feet
    """
    ft = m * 3.280839895
    return ft


def ft_to_m(ft):
    """
    Converts value of units ft to units m

    Args:
        ft:     value(s) in units of feet

    Returns:
        Value(s) in units of meters
    """
    m = ft / 3.280839895
    return m
