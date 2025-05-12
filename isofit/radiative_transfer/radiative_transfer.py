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
from __future__ import annotations

import logging

import numpy as np

from isofit.core.common import eps
from isofit.radiative_transfer.engines import Engines

Logger = logging.getLogger(__file__)


RTE = ["modtran", "sRTMnet", "KernelFlowsGP"]


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

            if confRT.engine_name not in Engines:
                raise AttributeError(
                    f"Invalid radiative transfer engine choice. Got: {confRT.engine_name}; Must be one of: {RTE}"
                )

            # Generate the params for this RTE
            params = {
                key: confPriority(key, [confRT, confIT, config]) for key in self._keys
            }
            params["engine_config"] = confRT

            # Select the right RTE and initialize it
            rte = Engines[confRT.engine_name](**params)
            self.rt_engines.append(rte)

            # Make sure the length of the config statevectores match the engine's assumed statevectors
            if (expected := len(config.statevector.get_element_names())) != (
                got := len(rte.indices.x_RT)
            ):
                error = f"Mismatch between the number of elements for the config statevector and LUT.indices.x_RT: {expected=}, {got=}"
                Logger.error(error)
                raise AttributeError(error)

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

    def calc_rdn(
        self,
        x_RT,
        rho_dir_dir,
        rho_dif_dir,
        Ls,
        L_tot,
        L_dir_dir,
        L_dif_dir,
        L_dir_dif,
        L_dif_dif,
        r,
        geom,
    ):
        """
        Physics-based forward model to calculate at-sensor radiance.
        Includes topography, background reflectance, and glint.
        """
        # Adjacency effects
        # ToDo: we need to think about if we want to obtain the background reflectance from the Geometry object
        #  or from the surface model, i.e., the same way as we do with the target pixel reflectance

        rho_dir_dif = (
            geom.bg_rfl if isinstance(geom.bg_rfl, np.ndarray) else rho_dir_dir
        )
        rho_dif_dif = (
            geom.bg_rfl if isinstance(geom.bg_rfl, np.ndarray) else rho_dif_dir
        )

        # Atmospheric path radiance
        L_atm = self.get_L_atm(x_RT, geom)

        # Atmospheric spherical albedo
        s_alb = r["sphalb"]
        atm_surface_scattering = s_alb * rho_dif_dif

        # Special case: 1-component model
        if not isinstance(L_dir_dir, np.ndarray) or len(L_dir_dir) == 1:
            # we assume rho_dir_dir = rho_dif_dir = rho_dir_dif = rho_dif_dif
            rho_dif_dif = rho_dir_dir
            # eliminate spherical albedo and one reflectance term from numerator if using 1-component model
            atm_surface_scattering = 1

        # Thermal transmittance
        L_up = Ls * (r["transm_up_dir"] + r["transm_up_dif"])

        # Our radiance model follows the physics as presented in Guanter (2006), Vermote et al. (1997), and
        # Tanre et al. (1983). This particular formulation facilitates the consideration of topographic effects,
        # glint, or BRDF modeling in general. The contribution of the target to the signal at the top of the atmosphere
        # is decomposed as the sum of four terms:

        # 1. photons directly transmitted from the sun to the target and directly reflected back to the sensor
        #    rho_dir_dir => directional-directional surface reflectance of the target
        # 2. photons scattered by the atmosphere then reflected by the target and directly transmitted to the sensor
        #    rho_dif_dir => surface diffuse-directional reflectance
        # 3. photons directly transmitted to the target but scattered by the atmosphere on their way to the sensor
        #    rho_dir_dif => surface directional-diffuse reflectance
        # 4. photons having at least two interactions with the atmosphere and one with the target
        #    rho_dif_dif => surface diffuse-diffuse reflectance

        # These terms are also called coupling terms, as they are responsible for the coupling between atmospheric
        # radiative transfer and the surface reflectance properties.

        # The coupling terms are multiplied by four different combinations of direct and diffuse radiance terms:
        # 1. L_dir_dir => downward direct * upward direct
        # 2. L_dif_dir => downward diffuse * upward direct
        # 3. L_dir_dif => downward direct * upward diffuse
        # 4. L_dif_dif => downward diffuse * upward diffuse

        # When separated radiance terms and/or a BRDF model of the surface are not available,
        # the Lambertian assumption is made for the target reflectance:
        # rho_dir_dir = rho_dif_dir = rho_dir_dif = rho_dif_dif
        # In this case, our radiance model reduces to:
        # L_atm + (L_tot * rho_dir_dir) / (1 - S * rho_dir_dir) + L_up,
        # with L_tot being the total radiance (downward * upward, direct + diffuse).

        # TOA radiance model
        ret = (
            L_atm
            + L_dir_dir * rho_dir_dir
            + L_dif_dir * rho_dif_dir
            + L_dir_dif * rho_dir_dif
            + L_dif_dif * rho_dif_dif
            + (L_tot * atm_surface_scattering * rho_dif_dif) / (1 - s_alb * rho_dif_dif)
            + L_up
        )

        return ret

    def rdn_to_rho(self, rdn, coszen, solar_irr=None):
        """Function to convert a radiance vector to transmittance.

        Args:
            rdn:       input data vector in radiance
            coszen:    cosine of solar zenith angle
            solar_irr: solar irradiance vector (optional)

        Returns:
            Data vector converted to transmittance
        """
        if solar_irr is None:
            solar_irr = self.solar_irr
        return rdn * np.pi / (solar_irr * coszen)

    def rho_to_rdn(self, rho, coszen, solar_irr=None):
        """Function to convert a transmittance vector to radiance.

        Args:
            rho:       input data vector in transmittance
            coszen:    cosine of solar zenith angle
            solar_irr: solar irradiance vector (optional)

        Returns:
            Data vector converted to radiance
        """
        if solar_irr is None:
            solar_irr = self.solar_irr
        return (solar_irr * coszen) / np.pi * rho

    def get_L_atm(self, x_RT: np.array, geom: Geometry) -> np.array:
        """Get the interpolated modeled atmospheric path radiance.

        Args:
            x_RT: radiative-transfer portion of the statevector
            geom: local geometry conditions for lookup

        Returns:
            interpolated modeled atmospheric path radiance
        """
        L_atms = []

        coszen, cos_i = geom.check_coszen_and_cos_i(self.coszen)

        for RT in self.rt_engines:
            if RT.treat_as_emissive:
                r = RT.get(x_RT, geom)
                rdn = r["thermal_upwelling"]
                L_atms.append(rdn)
            else:
                r = RT.get(x_RT, geom)
                if RT.rt_mode == "rdn":
                    L_atm = r["rhoatm"]
                else:
                    rho_atm = r["rhoatm"]
                    L_atm = self.rho_to_rdn(rho_atm, coszen)
                L_atms.append(L_atm)
        return np.hstack(L_atms)

    def get_L_down_transmitted(self, x_RT: np.array, geom: Geometry) -> np.array:
        """Get the interpolated direct and diffuse downward radiance on the sun-to-surface path.
        Thermal_downwelling already includes the transmission factor.
        Also assume there is no multiple scattering for TIR.

        Args:
            x_RT: radiative-transfer portion of the statevector
            geom: local geometry conditions for lookup

        Returns:
            interpolated total, direct, and diffuse downward atmospheric radiance
        """
        L_downs = []
        L_downs_dir = []
        L_downs_dif = []

        coszen, cos_i = geom.check_coszen_and_cos_i(self.coszen)

        for RT in self.rt_engines:
            if RT.treat_as_emissive:
                r = RT.get(x_RT, geom)
                rdn = r["thermal_downwelling"]
                L_downs.append(rdn)
            else:
                r = RT.get(x_RT, geom)
                if RT.rt_mode == "rdn":
                    L_down_dir = r["transm_down_dir"]
                    L_down_dif = r["transm_down_dif"]
                else:
                    # Transform downward transmittance to radiance
                    L_down_dir = self.rho_to_rdn(r["transm_down_dir"], coszen)
                    L_down_dif = self.rho_to_rdn(r["transm_down_dif"], coszen)

                L_down = L_down_dir + L_down_dif

                L_downs.append(L_down)
                L_downs_dir.append(L_down_dir)
                L_downs_dif.append(L_down_dif)

        return np.hstack(L_downs), np.hstack(L_downs_dir), np.hstack(L_downs_dif)

    def get_L_coupled(self, r: dict, geom: Geometry):
        """Get the interpolated radiance terms on the sun-to-surface-to-sensor path.
        These follow the physics as presented in Guanter (2006), Vermote et al. (1997), and Tanre et al. (1983).

        Args:
            r:      interpolated radiative transfer quantities from the LUT
            coszen: top-of-atmosphere solar zenith angle
            cos_i:  local solar zenith angle at the surface

        Returns:
            interpolated radiances along all optical paths:
            L_dir_dir => downward direct * upward direct
            L_dif_dir => downward diffuse * upward direct
            L_dir_dif => downward direct * upward diffuse
            L_dif_dif => downward diffuse * upward diffuse
        """
        # Check coszen against cos_i
        coszen, cos_i = geom.check_coszen_and_cos_i(self.coszen)

        # radiances along all optical paths
        L_coupled = []

        if any(
            [
                not isinstance(r[key], np.ndarray) or len(r[key]) == 1
                for key in self.rt_engines[0].coupling_terms
            ]
        ):
            # In case of the 1-component model, we cannot populate the coupling terms
            L_coupled = [
                0,
                0,
                0,
                0,
            ]
        else:
            for key in self.rt_engines[0].coupling_terms:
                L_coupled.append(
                    self.solar_irr * coszen / np.pi * r[key]
                    if self.rt_engines[0].rt_mode == "transm"
                    else r[key]
                )

        # assigning coupled terms, unscaling and rescaling downward direct radiance by local solar zenith angle
        L_dir_dir = L_coupled[0] / coszen * cos_i
        L_dif_dir = L_coupled[1]
        L_dir_dif = L_coupled[2] / coszen * cos_i
        L_dif_dif = L_coupled[3]

        return L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif

    def calc_RT_quantities(self, x_RT: np.ndarray, geom: Geometry):
        """Retrieves the RT quantities including the LUT sample (r),
        and the radiances (L). This function handles the hand-off between
        the 1c and 4c model.

        In the 1c case, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif = 0,
        and L_tot, L_down_dir, and L_down_dif are populated within the
        if statement.

        In the 4c case, we always use returns from get_L_coupled
        """
        # Propogate LUT
        r = self.get_shared_rtm_quantities(x_RT, geom)

        # Default: get directional radiances
        L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif = self.get_L_coupled(r, geom)
        L_tot = L_dir_dir + L_dif_dir + L_dir_dif + L_dif_dif

        # Handle case for 1c vs 4c model
        if not isinstance(L_tot, np.ndarray) or len(L_tot) == 1:
            # 1c model w/in if clause
            L_tot, L_down_dir, L_down_dif = self.get_L_down_transmitted(x_RT, geom)
        else:
            # 4c model w/in else clause
            L_down_dir = L_dir_dir + L_dif_dir
            L_down_dif = L_dif_dir + L_dif_dir

        return (
            r,
            L_tot,
            L_down_dir,
            L_down_dif,
            L_dir_dir,
            L_dif_dir,
            L_dir_dif,
            L_dif_dif,
        )

    def drdn_dRT(self, x_RT, geom, rho_dir_dir, rho_dif_dir, Ls, rdn):
        """Derivative of estimated radiance w.r.t. RT statevector elements.
        We use a numerical approach to approximate dRT with a constant surface
        reflectance. This is a reasonable approx. for the multicomponent surface.

        When using the glint model however, this does not take into account
        the dependence of the surface reflectance on the atmosphere.
        """
        # perturb each element of the RT state vector (finite difference)
        K_RT = []
        x_RTs_perturb = x_RT + np.eye(len(x_RT)) * eps
        for x_RT_perturb in list(x_RTs_perturb):
            (
                r,
                L_tot,
                L_down_dir,
                L_down_dif,
                L_dir_dir,
                L_dif_dir,
                L_dir_dif,
                L_dif_dif,
            ) = self.calc_RT_quantities(x_RT_perturb, geom)

            # Surface state is held constant?
            rdne = self.calc_rdn(
                x_RT_perturb,
                rho_dir_dir,
                rho_dif_dir,
                Ls,
                L_tot,
                L_dir_dir,
                L_dif_dir,
                L_dir_dif,
                L_dif_dif,
                r,
                geom,
            )
            K_RT.append((rdne - rdn) / eps)

        K_RT = np.array(K_RT).T

        return K_RT

    def drdn_dRTb(self, x_RT, geom, rho_dir_dir, rho_dif_dir, Ls, rdn):
        """Derivative of estimated rdn w.r.t. H2O_ABSCO

        Currently, the K_b matrix only covers forward model derivatives
        due to H2O_ABSCO unknowns, so that subsequent errors might occur
        when water vapor is not part of the statevector
        (which is very unlikely though).
        """
        if len(self.bvec) == 0:
            Kb_RT = np.zeros((0, len(self.wl.shape)))

        # ToDo: might require modification in case more unknowns are added
        # The following statement captures the case that H2O is not part
        # of the statevector.
        # but might need to be modified as soon as we add more unknowns
        elif len(self.bvec) > 0 and "H2OSTR" not in self.statevec_names:
            Kb_RT = np.zeros((1, len(self.wl)))
        else:
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
                    (
                        r,
                        L_tot,
                        L_down_dir,
                        L_down_dif,
                        L_dir_dir,
                        L_dif_dir,
                        L_dir_dif,
                        L_dif_dif,
                    ) = self.calc_RT_quantities(x_RT_perturb, geom)

                    rdne = self.calc_rdn(
                        x_RT_perturb,
                        rho_dir_dir,
                        rho_dif_dir,
                        Ls,
                        L_tot,
                        L_dir_dir,
                        L_dif_dir,
                        L_dir_dif,
                        L_dif_dif,
                        r,
                        geom,
                    )
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
