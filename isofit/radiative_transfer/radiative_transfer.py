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
#          Jay E. Fahlen, jay.e.fahlen@jpl.nasa.gov
#

import numpy as np

from isofit.configs import Config
from isofit.configs.sections.radiative_transfer_config import (
    RadiativeTransferEngineConfig,
)
from isofit.core.geometry import Geometry

from ..core.common import eps
from ..radiative_transfer.libradtran import LibRadTranRT
from ..radiative_transfer.modtran import ModtranRT
from ..radiative_transfer.six_s import SixSRT
from ..radiative_transfer.sRTMnet import SimulatedModtranRT


class RadiativeTransfer:
    """This class controls the radiative transfer component of the forward
    model. An ordered dictionary is maintained of individual RTMs (MODTRAN,
    for example). We loop over the dictionary concatenating the radiation
    and derivatives from each RTM and interval to form the complete result.

    In general, some of the state vector components will be shared between
    RTMs and bands. For example, H20STR is shared between both VISNIR and
    TIR. This class maintains the master list of statevectors.
    """

    def __init__(self, full_config: Config):
        # Maintain order when looping for indexing convenience
        config = full_config.forward_model.radiative_transfer

        self.statevec_names = config.statevector.get_element_names()
        self.lut_grid = config.lut_grid

        # TODO: rework this so that we instead initialize an interpolator, that calls
        # RTEs as necessary based on LUT grid or other parameters..which may happen higher up
        self.rt_engines = []
        for idx in range(len(config.radiative_transfer_engines)):
            rte_config: RadiativeTransferEngineConfig = (
                config.radiative_transfer_engines[idx]
            )

            if rte_config.engine_name == "modtran":
                rte = ModtranRT(rte_config, full_config)
            elif rte_config.engine_name == "libradtran":
                rte = LibRadTranRT(rte_config, full_config)
            elif rte_config.engine_name == "6s":
                rte = SixSRT(rte_config, full_config)
            elif rte_config.engine_name == "sRTMnet":
                rte = SimulatedModtranRT(rte_config, full_config)
            else:
                # Should never get here, checked in config
                raise AttributeError(
                    "Invalid radiative transfer engine name: {}".format(
                        rte_config.engine_name
                    )
                )

            self.rt_engines.append(rte)

        # The rest of the code relies on sorted order of the individual RT engines which cannot
        # be guaranteed by the dict JSON or YAML input
        self.rt_engines.sort(key=lambda x: x.wl[0])

        # Retrieved variables.  We establish scaling, bounds, and
        # initial guesses for each state vector element.  The state
        # vector elements are all free parameters in the RT lookup table,
        # and they all have associated dimensions in the LUT grid.
        self.bounds, self.scale, self.init = [], [], []
        self.prior_mean, self.prior_sigma = [], []

        for sv, sv_name in zip(*config.statevector.get_elements()):
            self.bounds.append(sv.bounds)
            self.scale.append(sv.scale)
            self.init.append(sv.init)
            self.prior_sigma.append(sv.prior_sigma)
            self.prior_mean.append(sv.prior_mean)

        self.bounds = np.array(self.bounds)
        self.scale = np.array(self.scale)
        self.init = np.array(self.init)
        self.prior_mean = np.array(self.prior_mean)
        self.prior_sigma = np.array(self.prior_sigma)

        self.wl = np.concatenate([RT.wl for RT in self.rt_engines])

        self.bvec = config.unknowns.get_element_names()
        self.bval = np.array([x for x in config.unknowns.get_elements()[0]])

        self.solar_irr = np.concatenate([RT.solar_irr for RT in self.rt_engines])
        # These should all be the same so just grab one
        self.coszen = [RT.coszen for RT in self.rt_engines][0]

        self.topography_model = config.topography_model

    def xa(self):
        """Pull the priors from each of the individual RTs."""
        return self.prior_mean

    def Sa(self):
        """Pull the priors from each of the individual RTs."""
        return np.diagflat(np.power(np.array(self.prior_sigma), 2))

    def get_shared_rtm_quantities(self, x_RT, geom):
        """Return only the set of RTM quantities (transup, sphalb, etc.) that are contained
        in all RT engines.
        """

        ret = []
        for RT in self.rt_engines:
            ret.append(RT.get(x_RT, geom))

        return self.pack_arrays(ret)

    def calc_rdn(self, x_RT, rfl, Ls, geom):
        r = self.get_shared_rtm_quantities(x_RT, geom)
        L_atm = self.get_L_atm(x_RT, geom)
        L_up = Ls * r["transup"]

        if geom.bg_rfl is not None:
            # adjacency effects are counted
            I = (self.solar_irr * self.coszen) / np.pi
            bg = geom.bg_rfl
            t_down = r["t_down_dif"] + r["t_down_dir"]

            ret = (
                L_atm
                + I / (1.0 - r["sphalb"] * bg) * bg * t_down * r["t_up_dif"]
                + I / (1.0 - r["sphalb"] * bg) * rfl * t_down * r["t_up_dir"]
                + L_up
            )

        elif self.topography_model:
            I = (self.solar_irr) / np.pi
            t_dir_down = r["t_down_dir"]
            t_dif_down = r["t_down_dif"]
            if geom.cos_i is None:
                cos_i = self.coszen
            else:
                cos_i = geom.cos_i
            t_total_up = r["t_up_dif"] + r["t_up_dir"]
            s_alb = r["sphalb"]
            # topographic flux (topoflux) effect corrected
            ret = (
                L_atm
                + (
                    I * cos_i / (1.0 - s_alb * rfl) * t_dir_down
                    + I * self.coszen / (1.0 - s_alb * rfl) * t_dif_down
                )
                * rfl
                * t_total_up
            )

        else:
            L_down_transmitted = self.get_L_down_transmitted(x_RT, geom)

            ret = L_atm + L_down_transmitted * rfl / (1.0 - r["sphalb"] * rfl) + L_up

        return ret

    def get_L_atm(self, x_RT: np.array, geom: Geometry) -> np.array:
        """Get the interpolated modeled atmospheric reflectance (aka path radiance).

        Args:
            x_RT: radiative-transfer portion of the statevector
            geom: local geometry conditions for lookup

        Returns:
            the interpolated modeled atmospheric reflectance
        """
        L_atms = []
        for RT in self.rt_engines:
            if RT.treat_as_emissive:
                r = RT.get(x_RT, geom)
                rdn = r["thermal_upwelling"]
                L_atms.append(rdn)
            else:
                r = RT.get(x_RT, geom)
                rho = r["rhoatm"]
                rdn = rho / np.pi * (self.solar_irr * self.coszen)
                L_atms.append(rdn)
        return np.hstack(L_atms)

    def get_L_down_transmitted(self, x_RT: np.array, geom: Geometry) -> np.array:
        """Get the interpolated total downward atmospheric transmittance.
        Thermal_downwelling already includes the transmission factor. Also
        assume there is no multiple scattering for TIR.

        Args:
            x_RT: radiative-transfer portion of the statevector
            geom: local geometry conditions for lookup

        Returns:
            The interpolated totel downward atmospheric transmittance
        """
        L_downs = []
        for RT in self.rt_engines:
            if RT.treat_as_emissive:
                r = RT.get(x_RT, geom)
                rdn = r["thermal_downwelling"]
                L_downs.append(rdn)
            else:
                r = RT.get(x_RT, geom)
                rdn = (self.solar_irr * self.coszen) / np.pi * r["t_down_dif"]
                L_downs.append(rdn)
        return np.hstack(L_downs)

    def drdn_dRT(self, x_RT, rfl, drfl_dsurface, Ls, dLs_dsurface, geom):
        # first the rdn at the current state vector
        rdn = self.calc_rdn(x_RT, rfl, Ls, geom)

        # perturb each element of the RT state vector (finite difference)
        K_RT = []
        x_RTs_perturb = x_RT + np.eye(len(x_RT)) * eps
        for x_RT_perturb in list(x_RTs_perturb):
            rdne = self.calc_rdn(x_RT_perturb, rfl, Ls, geom)
            K_RT.append((rdne - rdn) / eps)
        K_RT = np.array(K_RT).T

        # Get K_surface
        r = self.get_shared_rtm_quantities(x_RT, geom)

        if geom.bg_rfl is not None:
            # adjacency effects are counted
            I = (self.solar_irr * self.coszen) / np.pi
            bg = geom.bg_rfl
            t_down = r["t_down_dif"] + r["t_down_dir"]
            drdn_drfl = I / (1.0 - r["sphalb"] * bg) * t_down * r["t_up_dir"]

        elif self.topography_model:
            # jac w.r.t. topoflux correct radiance
            I = self.solar_irr / np.pi
            t_dir_down = r["t_down_dir"]
            t_dif_down = r["t_down_dif"]
            if geom.cos_i is None:
                cos_i = self.coszen
            else:
                cos_i = geom.cos_i
            t_total_up = r["t_up_dif"] + r["t_up_dir"]
            s_alb = r["sphalb"]

            a = t_total_up * (I * cos_i * t_dir_down + I * self.coszen * t_dif_down)
            drdn_drfl = a / (1 - s_alb * rfl) ** 2
        else:
            L_down_transmitted = self.get_L_down_transmitted(x_RT, geom)

            # The reflected downwelling light is:
            # L_down_transmitted * rfl / (1.0 - r['sphalb'] * rfl), or
            # L_down_transmitted * rho_scaled_for_multiscattering
            # This term is the derivative of rho_scaled_for_multiscattering
            drho_scaled_for_multiscattering_drfl = 1.0 / (1 - r["sphalb"] * rfl) ** 2

            drdn_drfl = L_down_transmitted * drho_scaled_for_multiscattering_drfl

        drdn_dLs = r["transup"]
        K_surface = (
            drdn_drfl[:, np.newaxis] * drfl_dsurface
            + drdn_dLs[:, np.newaxis] * dLs_dsurface
        )

        return K_RT, K_surface

    def drdn_dRTb(self, x_RT, rfl, Ls, geom):
        if len(self.bvec) == 0:
            Kb_RT = np.zeros((0, len(self.wl.shape)))

        else:
            # first the radiance at the current state vector
            r = self.get_shared_rtm_quantities(x_RT, geom)
            rdn = self.calc_rdn(x_RT, rfl, Ls, geom)

            # unknown parameters modeled as random variables per
            # Rodgers et al (2000) K_b matrix.  We calculate these derivatives
            # by finite differences
            Kb_RT = []
            perturb = 1.0 + eps
            for unknown in self.bvec:
                if unknown == "H2O_ABSCO" and "H2OSTR" in self.statevec_names:
                    i = self.statevec_names.index("H2OSTR")
                    x_RT_perturb = x_RT.copy()
                    x_RT_perturb[i] = x_RT[i] * perturb
                    rdne = self.calc_rdn(x_RT_perturb, rfl, Ls, geom)
                    Kb_RT.append((rdne - rdn) / eps)

        Kb_RT = np.array(Kb_RT).T
        return Kb_RT

    def summarize(self, x_RT, geom):
        ret = []
        for RT in self.rt_engines:
            ret.append(RT.summarize(x_RT, geom))
        ret = "\n".join(ret)
        return ret

    def pack_arrays(self, rtm_quantities_from_RT_engines):
        """Take the list of dict outputs from each RT engine and
        stack their internal arrays in the same order. Keep only
        those quantities that are common to all RT engines.
        """

        # Get the intersection of the sets of keys from each of the rtm_quantities_from_RT_engines
        shared_rtm_keys = set(rtm_quantities_from_RT_engines[0].keys())
        if len(rtm_quantities_from_RT_engines) > 1:
            for rtm_quantities_from_one_RT_engine in rtm_quantities_from_RT_engines[1:]:
                shared_rtm_keys.intersection_update(
                    rtm_quantities_from_one_RT_engine.keys()
                )

        # Concatenate the different band ranges
        rtm_quantities_concatenated_over_RT_bands = {}
        for key in shared_rtm_keys:
            temp = [x[key] for x in rtm_quantities_from_RT_engines]
            rtm_quantities_concatenated_over_RT_bands[key] = np.hstack(temp)

        return rtm_quantities_concatenated_over_RT_bands


def ext550_to_vis(ext550):
    return np.log(50.0) / (ext550 + 0.01159)


def three_albedo_method(
    transups,
    drct_rflts_1,
    grnd_rflts_1,
    grnd_rflts_2,
    lp_1,
    lp_2,
    sols,
    coszen,
    test_rfls,
    widths,
):
    """This implementation follows Guanter et al. (2009) (DOI:10.1080/01431160802438555),
    with modifications by Nimrod Carmon. It is called the "2-albedo" method, referring to running
    MODTRAN with 2 different surface albedos. The 3-albedo method is similar to this one with
    the single difference where the "path_radiance_no_surface" variable is taken from a
    zero-surface-reflectance MODTRAN run instead of being calculated from 2 MODTRAN outputs.
    There are a few argument as to why this approach is beneficial:
    (1) for each grid point on the lookup table you sample MODTRAN 2 or 3 times, i.e., you get
    2 or 3 "data points" for the atmospheric parameter of interest. This in theory allows us
    to use a lower band model resolution for the MODTRAN run, which is much faster, while keeping
    high accuracy. (2) we use the decoupled transmittance products to expand
    the forward model and account for more physics, currently topography and glint."""

    t_up_dirs = np.array(transups)
    direct_ground_reflected_1 = np.array(drct_rflts_1)
    total_ground_reflected_1 = np.array(grnd_rflts_1)
    total_ground_reflected_2 = np.array(grnd_rflts_2)
    path_radiance_1 = np.array(lp_1)
    path_radiance_2 = np.array(lp_2)
    TOA_Irad = np.array(sols) * coszen / np.pi
    rfl_1 = test_rfls[1]
    rfl_2 = test_rfls[2]
    mus = coszen

    direct_flux_1 = direct_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs
    global_flux_1 = total_ground_reflected_1 * np.pi / rfl_1 / t_up_dirs

    global_flux_2 = total_ground_reflected_2 * np.pi / rfl_2 / t_up_dirs

    path_radiance_no_surface = (
        rfl_2 * path_radiance_1 * global_flux_2
        - rfl_1 * path_radiance_2 * global_flux_1
    ) / (rfl_2 * global_flux_2 - rfl_1 * global_flux_1)

    # Diffuse upwelling transmittance
    t_up_difs = (
        np.pi * (path_radiance_1 - path_radiance_no_surface) / (rfl_1 * global_flux_1)
    )

    # Spherical Albedo
    sphalbs = (global_flux_1 - global_flux_2) / (
        rfl_1 * global_flux_1 - rfl_2 * global_flux_2
    )
    direct_flux_radiance = direct_flux_1 / mus

    global_flux_no_surface = global_flux_1 * (1.0 - rfl_1 * sphalbs)
    diffuse_flux_no_surface = global_flux_no_surface - direct_flux_radiance * coszen

    t_down_dirs = (direct_flux_radiance * coszen / widths / np.pi) / TOA_Irad
    t_down_difs = (diffuse_flux_no_surface / widths / np.pi) / TOA_Irad

    # total transmittance
    transms = (t_down_dirs + t_down_difs) * (t_up_dirs + t_up_difs)

    return transms, t_down_dirs, t_down_difs, t_up_dirs, t_up_difs, sphalbs
