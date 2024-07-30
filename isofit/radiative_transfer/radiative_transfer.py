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
from types import SimpleNamespace

import numpy as np

from isofit.configs import Config
from isofit.core.geometry import Geometry

from ..core.common import eps
from ..radiative_transfer.kernel_flows import KernelFlowsRT
from ..radiative_transfer.modtran import ModtranRT
from ..radiative_transfer.six_s import SixSRT
from ..radiative_transfer.sRTMnet import SimulatedModtranRT

# Match config string options to modules
RTE = {
    "modtran": ModtranRT,
    "6s": SixSRT,
    "sRTMnet": SimulatedModtranRT,
    "KernelFlowsGP": KernelFlowsRT,
}


def confPriority(key, configs):
    """
    Selects a key from a config if the value for that key is not None
    Prioritizes returning the first value found in the configs list

    TODO: ISOFIT configs are annoying and will create keys to NoneTypes
    Should use mlky to handle key discovery at runtime instead of like this
    """
    value = None
    for config in configs:
        if hasattr(config, key):
            value = getattr(config, key)
            if value is not None:
                break
    return value


class RadiativeTransfer:
    """This class controls the radiative transfer component of the forward
    model. An ordered dictionary is maintained of individual RTMs (MODTRAN,
    for example). We loop over the dictionary concatenating the radiation
    and derivatives from each RTM and interval to form the complete result.

    In general, some of the state vector components will be shared between
    RTMs and bands. For example, H20STR is shared between both VISNIR and
    TIR. This class maintains the master list of statevectors.
    """

    # Keys to retrieve from 3 sections to use the preferred
    # Prioritizes retrieving from radiative_transfer_engines first, then instrument, then radiative_transfer
    _keys = [
        "interpolator_style",
        "overwrite_interpolator",
        "lut_grid",
        "lut_path",
        "wavelength_file",
    ]

    def __init__(self, full_config: Config):
        config = full_config.forward_model.radiative_transfer
        confIT = full_config.forward_model.instrument

        self.lut_grid = config.lut_grid
        self.statevec_names = config.statevector.get_element_names()

        self.rt_engines = []
        for idx in range(len(config.radiative_transfer_engines)):
            confRT = config.radiative_transfer_engines[idx]

            # Generate the params for this RTE
            params = {
                key: confPriority(key, [confRT, confIT, config]) for key in self._keys
            }
            params["engine_config"] = confRT

            # Select the right RTE and initialize it
            rte = RTE[confRT.engine_name](**params)
            self.rt_engines.append(rte)

        # If any engine is true, self is true
        self.topography_model = any([rte.topography_model for rte in self.rt_engines])
        self.glint_model = any([rte.glint_model for rte in self.rt_engines])

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

        # TODO: Is code for this missing? We have if statements that rely on this
        self.glint_model = False

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

    @property
    def coszen(self):
        """
        Backwards compatibility until Geometry takes over this param
        Return some child RTE coszen
        """
        for child in self.rt_engines:
            if "coszen" in child.lut:
                return child.lut.coszen.data

    def calc_rdn(self, x_RT, x_surface, rfl, Ls, geom):
        r = self.get_shared_rtm_quantities(x_RT, geom)
        L_atm = self.get_L_atm(x_RT, geom)
        L_up = Ls * (r["transm_up_dir"] + r["transm_up_dif"])

        if geom.bg_rfl is not None:
            # adjacency effects are counted
            I = (self.solar_irr * self.coszen) / np.pi
            bg = geom.bg_rfl
            t_down = r["transm_down_dif"] + r["transm_down_dir"]

            ret = (
                L_atm
                + I / (1.0 - r["sphalb"] * bg) * bg * t_down * r["transm_up_dif"]
                + I / (1.0 - r["sphalb"] * bg) * rfl * t_down * r["transm_up_dir"]
                + L_up
            )

        elif self.topography_model:
            I = self.solar_irr / np.pi
            t_dir_down = r["transm_down_dir"]
            t_dif_down = r["transm_down_dif"]
            if geom.cos_i is None:
                cos_i = self.coszen
            else:
                cos_i = geom.cos_i
            t_total_up = r["transm_up_dif"] + r["transm_up_dir"]
            t_total_down = t_dir_down + t_dif_down
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

        elif self.glint_model:
            L_down_transmitted = self.get_L_down_transmitted(x_RT, geom)

            E_dd = (self.solar_irr * self.coszen) / np.pi * r["t_down_dir"]
            E_ds = (self.solar_irr * self.coszen) / np.pi * r["t_down_dif"]
            E_d = E_dd + E_ds
            L_sky = x_surface[-2] * E_dd + x_surface[-1] * E_ds

            rho_ls = 0.02  # fresnel reflectance factor (approx. 0.02 for nadir view)
            glint = rho_ls * (L_sky / E_d)

            ret = (
                L_atm
                + L_down_transmitted
                * (rfl + glint)
                / (1.0 - r["sphalb"] * (rfl + glint))
                + L_up
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
            interpolated modeled atmospheric reflectance
        """
        L_atms = []
        for RT in self.rt_engines:
            if RT.treat_as_emissive:
                r = RT.get(x_RT, geom)
                rdn = r["thermal_upwelling"]
                L_atms.append(rdn)
            else:
                r = RT.get(x_RT, geom)
                rdn = r["rhoatm"]
                if RT.rt_mode == "transm":
                    rdn = (self.solar_irr * self.coszen) / np.pi * rdn
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
            interpolated total downward atmospheric transmittance
        """
        L_downs = []
        for RT in self.rt_engines:
            if RT.treat_as_emissive:
                r = RT.get(x_RT, geom)
                rdn = r["thermal_downwelling"]
                L_downs.append(rdn)
            else:
                r = RT.get(x_RT, geom)
                rdn = r["transm_down_dir"] + r["transm_down_dif"]
                if RT.rt_mode == "transm":
                    rdn = (self.solar_irr * self.coszen) / np.pi * rdn
                L_downs.append(rdn)
        return np.hstack(L_downs)

    def drdn_dRT(
        self, x_RT, x_surface, rfl, drfl_dsurface, Ls, dLs_dsurface, geom: Geometry
    ):
        # first the rdn at the current state vector
        rdn = self.calc_rdn(x_RT, x_surface, rfl, Ls, geom)

        # perturb each element of the RT state vector (finite difference)
        K_RT = []
        x_RTs_perturb = x_RT + np.eye(len(x_RT)) * eps
        for x_RT_perturb in list(x_RTs_perturb):
            rdne = self.calc_rdn(x_RT_perturb, x_surface, rfl, Ls, geom)
            K_RT.append((rdne - rdn) / eps)
        K_RT = np.array(K_RT).T

        # Get K_surface
        r = self.get_shared_rtm_quantities(x_RT, geom)

        if geom.bg_rfl is not None:
            # adjacency effects are counted
            I = (self.solar_irr * self.coszen) / np.pi
            bg = geom.bg_rfl
            t_down = r["transm_down_dif"] + r["transm_down_dir"]
            drdn_drfl = I / (1.0 - r["sphalb"] * bg) * t_down * r["transm_up_dir"]

        elif self.topography_model:
            # jac w.r.t. topoflux correct radiance
            I = self.solar_irr / np.pi
            t_dir_down = r["transm_down_dir"]
            t_dif_down = r["transm_down_dif"]
            if geom.cos_i is None:
                cos_i = self.coszen
            else:
                cos_i = geom.cos_i
            t_total_up = r["transm_up_dif"] + r["transm_up_dir"]
            t_total_down = t_dir_down + t_dif_down
            s_alb = r["sphalb"]

            a = t_total_up * (I * cos_i * t_dir_down + I * self.coszen * t_dif_down)
            drdn_drfl = a / (1 - s_alb * rfl) ** 2

        elif self.glint_model:
            L_down_transmitted = self.get_L_down_transmitted(x_RT, geom)

            E_dd = (self.solar_irr * self.coszen) / np.pi * r["t_down_dir"]
            E_ds = (self.solar_irr * self.coszen) / np.pi * r["t_down_dif"]
            E_d = E_dd + E_ds
            L_sky = x_surface[-2] * E_dd + x_surface[-1] * E_ds

            rho_ls = 0.02  # fresnel reflectance factor (approx. 0.02 for nadir view)
            glint = rho_ls * (L_sky / E_d)

            drho_scaled_for_multiscattering_drfl = (
                1.0 / (1 - r["sphalb"] * (rfl + glint)) ** 2
            )
            drdn_drfl = L_down_transmitted * drho_scaled_for_multiscattering_drfl

        else:
            L_down_transmitted = self.get_L_down_transmitted(x_RT, geom)

            # The reflected downwelling light is:
            # L_down_transmitted * rfl / (1.0 - r['sphalb'] * rfl), or
            # L_down_transmitted * rho_scaled_for_multiscattering
            # This term is the derivative of rho_scaled_for_multiscattering
            drho_scaled_for_multiscattering_drfl = 1.0 / (1 - r["sphalb"] * rfl) ** 2

            drdn_drfl = L_down_transmitted * drho_scaled_for_multiscattering_drfl

        drdn_dLs = r["transm_up_dir"] + r["transm_up_dif"]
        K_surface = (
            drdn_drfl[:, np.newaxis] * drfl_dsurface
            + drdn_dLs[:, np.newaxis] * dLs_dsurface
        )

        return K_RT, K_surface

    def drdn_dRTb(self, x_RT, x_surface, rfl, Ls, geom):
        if len(self.bvec) == 0:
            Kb_RT = np.zeros((0, len(self.wl.shape)))
        # currently, the K_b matrix only covers forward model derivatives due to H2O_ABSCO unknowns,
        # so that subsequent errors might occur when water vapor is not part of the state
        # vector (which is very unlikely though). the following statement captures this case,
        # but might need to be modified as soon as we add more unknowns
        # ToDo: might require modification in case more unknowns are added
        elif len(self.bvec) > 0 and "H2OSTR" not in self.statevec_names:
            Kb_RT = np.zeros((1, len(self.wl)))
        else:
            # first the radiance at the current state vector
            rdn = self.calc_rdn(x_RT, x_surface, rfl, Ls, geom)

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
                    rdne = self.calc_rdn(x_RT_perturb, x_surface, rfl, Ls, geom)
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
    """VIS is defined as a function of the surface aerosol extinction coefficient
    at 550 nm in km-1, EXT550, by the formula VIS[km] = ln(50) / (EXT550 + 0.01159),
    where 0.01159 is the surface Rayleigh scattering coefficient at 550 nm in km-1
    (see MODTRAN6 manual, p. 50).
    """
    return np.log(50.0) / (ext550 + 0.01159)
