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
# Author: Philip G Brodrick, philip.brodrick@jpl.nasa.gov
#
from __future__ import annotations

import logging
from os.path import abspath, exists, join

import numpy as np


try:
    from juliacall import Main as jl

    jl.seval("using vSmartMOM")
    jl.seval("using vSmartMOM.CoreRT")
    jl.seval("using vSmartMOM.InelasticScattering")

except ImportError:
    raise ImportError(
        "juliacall is required for vSmartMOM usage in ISOFIT. "
        "You can install via: pip install isofit[juliacall]"
    )


from isofit.atmosphere.atmosphere import BaseAtmosphere
from isofit.core.common import (
    calculate_resample_matrix,
    json_load_ascii,
    load_esd,
    resample_spectrum,
)
from isofit.core import units
from isofit.data import env
from isofit.luts.writer import Writer

Logger = logging.getLogger(__name__)


class VSmartMomRT(BaseAtmosphere, Writer):

    albedos = [0.0, 0.2]
    wl_start = 350.0
    wl_end = 2510.0

    # TODO
    # - Figure out a way to suppress @progressbar and other logs/prints vSmartMOM passes through...

    # - tau dimension patch, see method below (`check_tau_dim()`). Do we still even need it in this python implementation?

    # - implement multiband model (similarly to TP7 PR)

    # - enable option not to write to disk and just pass arrays from makeSim to readSim

    # - double check RAA based on Phils PR-1070

    # -  Bring in H2O molar mass, air mass etc, into units

    # - UserWarning: torch was imported before juliacall. This may cause a segfault.
    #     To avoid this, import juliacall before importing torch.
    #     For updates, see https://github.com/pytorch/pytorch/issues/78829.

    # - Check UV and Conda install to check changes needed

    # - Fully hookup through apply_oe and template construction

    # - Don't forget default wl is at 10 cm-1 for testing but should not keep this for the PR

    def __init__(self, full_config, **kwargs):
        self.config = full_config.forward_model.atmosphere
        self.sim_path = self.config.sim_path

        super().__init__(full_config, **kwargs)

    def preSim(self):

        # define the output wl grid [nm], and stash matrix H
        # The wn grid is also stored to self and used for vsmartmom sims.
        wn1 = units.nm_to_wavenumber(self.wl_start)
        wn2 = units.nm_to_wavenumber(self.wl_end)
        start_wn = min(wn1, wn2)
        end_wn = max(wn1, wn2)
        self.wn_grid = np.arange(start_wn, end_wn + 1e-8, self.config.engine.wl_cm_1)
        self.vsmartmom_wl = np.sort(units.wavenumber_to_nm(self.wn_grid))
        self.matrix_H = calculate_resample_matrix(self.vsmartmom_wl, self.wl, self.fwhm)

        # load static information from MODTRAN template
        self.template = json_load_ascii(self.config.engine.template_file)["MODTRAN"]
        self.modtran_geom = self.template[0]["MODTRANINPUT"]["GEOMETRY"]
        self.modtran_surf = self.template[0]["MODTRANINPUT"]["SURFACE"]
        self.modtran_atmos = self.template[0]["MODTRANINPUT"]["ATMOSPHERE"]

        self.atm_type = "atm_midlat_summer"
        if self.modtran_atmos["M1"] is not None and isinstance(
            self.modtran_atmos["M1"], str
        ):
            self.atm_type = self.modtran_atmos["M1"].lower()

        self.day_of_year = self.modtran_geom["IDAY"]

        # Set up the irradiance spectrum
        self.path_solar = env.path("data", "oci", "tsis_f0_0p1.txt")
        solar_data = np.loadtxt(self.path_solar)
        wl_solar = solar_data[:, 0].T
        solar_irr = solar_data[:, 1].T

        if np.nanmax(solar_irr) > 300.0:
            solar_irr = solar_irr / 10.0
        elif 3.0 < np.nanmax(solar_irr) < 30.0:
            solar_irr = solar_irr * 10.0
        elif 0.3 < np.nanmax(solar_irr) < 3.0:
            solar_irr = solar_irr * 100.0
        elif 30.0 <= np.nanmax(solar_irr) <= 300.0:
            pass
        else:
            raise ValueError("Verify units of solar irradiance are in uW/nm/cm2.")

        # apply ESD correction
        self.esd = load_esd()
        irr_ref = self.esd[200, 1]
        irr_cur = self.esd[self.day_of_year - 1, 1]
        solar_irr = solar_irr * irr_ref**2 / irr_cur**2

        self.solar_irr_sensor = resample_spectrum(
            solar_irr, wl_solar, self.wl, self.fwhm
        )

        return {
            "coszen": np.cos(np.radians(self.modtran_geom["PARM2"])),
            "solzen": self.modtran_geom["PARM2"],
            "solar_irr": self.solar_irr_sensor,
        }

    def makeSim(self, point: np.array):
        """
        Perform vSmartMOM simulations

        Parameters
        ----------
        point: np.array
            Point to process
        """
        name = self.point_to_filename(point)
        vals = self.extract_point_values(point)

        # Only execute when the output file is missing
        out_path = abspath(join(self.sim_path, f"{name}_vsmartmom.csv"))
        if exists(out_path):
            Logger.warning(f"vSmartMOM sim files already exist for point {point}")
            return

        if not self.config.configure_and_exit:
            self.run_vsmartmom(vals, out_path)

    def readSim(self, point):
        name = self.point_to_filename(point)
        out_path = abspath(join(self.sim_path, f"{name}_vsmartmom.csv"))

        # Load output data from vSmartMOM, with each sim as a csv
        data = np.loadtxt(out_path, delimiter=",", skiprows=1)
        # wl = data[:, 0]
        rhoatm = data[:, 1]
        sphalb = data[:, 2]
        transm_down_dir = data[:, 3]
        transm_down_dif = data[:, 4]
        transm_up_dir = data[:, 5]
        transm_up_dif = data[:, 6]

        # get cos_sza
        vals = self.extract_point_values(point)
        cos_sza = np.cos(np.radians(vals["sza"]))

        results = {
            "rhoatm": rhoatm,
            "sphalb": sphalb,
            "transm_down_dif": transm_down_dif,
            "transm_down_dir": transm_down_dir,
            "transm_up_dif": transm_up_dif,
            "transm_up_dir": transm_up_dir,
            "dir-dir": transm_down_dir * transm_up_dir,
            "dif-dir": transm_down_dif * transm_up_dir,
            "dir-dif": transm_down_dir * transm_up_dif,
            "dif-dif": transm_down_dif * transm_up_dif,
        }

        # resample all quantities to sensor wavelengths using H matrix
        results = {
            key: resample_spectrum(
                data, self.vsmartmom_wl, self.wl, self.fwhm, H=self.matrix_H
            )
            for key, data in results.items()
        }

        # set these keys to be appropriate for rdn mode
        for key in ["rhoatm", "dir-dir", "dif-dir", "dir-dif", "dif-dif"]:
            results[key] = units.transm_to_rdn(
                results[key], self.solar_irr_sensor, cos_sza
            )

        return results

    def postSim(self):
        self.rt_mode = "rdn"
        self.lut.setAttr("RT_mode", "rdn")

    def extract_point_values(self, point):
        """
        In vSmartMOM they have the following convention for vaz:

        vaz = phi_view - phi_sun ; 0 deg: Sun behind satellite; 180 deg: opposite.
        """
        translation = {
            "surface_elevation_km": "GNDALT",
            "observer_altitude_km": "H1ALT",
            "observer_azimuth": "TRUEAZ",
            "observer_zenith": "OBSZEN",
        }

        vals = {translation.get(n, n): v for n, v in zip(self.lut_names, point)}

        sza_deg = self.modtran_geom["PARM2"]
        rel_az = self.modtran_geom["PARM1"]
        vaa_deg = vals.get("TRUEAZ", self.modtran_geom["TRUEAZ"])
        saa_deg = (vaa_deg - rel_az) % 360

        vza = vals.get("OBSZEN", self.modtran_geom["OBSZEN"])
        vza_deg = 180.0 - vza if vza > 90.0 else vza
        vza_deg = max(0.0, vza_deg)

        elev = abs(max(vals.get("GNDALT", self.modtran_surf["GNDALT"]), 0.0))
        vals["observer_elev_km"] = vals.get("H1ALT", self.modtran_geom["H1ALT"])

        h2o = vals.get("H2OSTR", self.modtran_atmos["H2OSTR"])
        aot = vals.get("AOT550", vals.get("AERFRAC_2", vals.get("AERFRAC_1", 0.10)))
        co2 = vals.get("CO2", self.modtran_atmos["CO2MX"])

        if "solar_zenith" in vals:
            sza_deg = vals["solar_zenith"]

        # TODO Double check this based on Phils RAA PR
        if "relative_azimuth" in vals:
            saa_deg = (vaa_deg - vals["relative_azimuth"]) % 360

        return {
            "sza": float(sza_deg),
            "vza": float(vza_deg),
            "vaz": float(vaa_deg - saa_deg),
            "elev": float(elev),
            "h2o": float(h2o),
            "aot": float(aot),
            "co2": float(co2),
            "surface_elev_km": float(elev),
            "observer_elev_km": float(vals["observer_elev_km"]),
        }

    def run_vsmartmom(self, vals, out_path):

        # Constants
        GRAVITY = 9.80665
        O2_TOTAL = 167820
        O2_VMR = 0.2095
        TOA_KM = 98.0
        P_SEALEVEL = 1013.25
        H2O_MOLAR_MASS = 18.015
        AIR_MOLAR_MASS = 28.97

        h2o_vmr = vals["h2o"] / (
            ((P_SEALEVEL * 10.0) / GRAVITY) * (H2O_MOLAR_MASS / AIR_MOLAR_MASS)
        )
        co2_vmr = (vals["co2"] / O2_TOTAL) * O2_VMR

        atm_type = self.atm_type

        # We initialize based on default yaml (based on Version 2.2.0)
        # https://github.com/RemoteSensingTools/vSmartMOM.jl/blob/release/v2.2-consolidation/src/CoreRT/DefaultParameters.yaml
        params = jl.vSmartMOM.default_parameters()
        params.nstreams = self.config.engine.nstreams_mom
        params.architecture = jl.seval(self.config.engine.architecture)
        params.float_type = jl.seval(self.config.engine.float_type)

        # Assumes O2 scaling factor is constant
        # Table is approx. based on MODTRAN profiles
        # http://modtran.spectral.com/modtran_faq
        jl.ATM_VMRS = {
            "atm_tropical": {
                "O2": 0.2095,
                "CH4": 1.649e-6,
                "O3": 3.453e-7,
                "N2O": 3.069e-7,
                "CO": 1.092e-7,
                "NO2": 2.625e-10,
                "SO2": 1.344e-10,
            },
            "atm_midlat_summer": {
                "O2": 0.2095,
                "CH4": 1.583e-6,
                "O3": 4.141e-7,
                "N2O": 2.964e-7,
                "CO": 1.095e-7,
                "NO2": 2.724e-10,
                "SO2": 1.353e-10,
            },
            "atm_midlat_winter": {
                "O2": 0.2095,
                "CH4": 1.594e-6,
                "O3": 4.691e-7,
                "N2O": 2.992e-7,
                "CO": 1.121e-7,
                "NO2": 2.470e-10,
                "SO2": 1.400e-10,
            },
            "atm_subarc_summer": {
                "O2": 0.2095,
                "CH4": 1.576e-6,
                "O3": 4.329e-7,
                "N2O": 2.751e-7,
                "CO": 1.105e-7,
                "NO2": 2.704e-10,
                "SO2": 1.372e-10,
            },
            "atm_subarc_winter": {
                "O2": 0.2095,
                "CH4": 1.592e-6,
                "O3": 4.702e-7,
                "N2O": 3.004e-7,
                "CO": 1.139e-7,
                "NO2": 2.338e-10,
                "SO2": 1.428e-10,
            },
            "atm_us_standard_1976": {
                "O2": 0.2095,
                "CH4": 1.652e-6,
                "O3": 4.298e-7,
                "N2O": 3.077e-7,
                "CO": 1.110e-7,
                "NO2": 2.553e-10,
                "SO2": 1.375e-10,
            },
        }

        # Truncate pressure, surface, and q if above sea level
        if vals["surface_elev_km"] > 0.0:
            p_surf = P_SEALEVEL * np.exp(-vals["surface_elev_km"] / 8.0)

            p_array = np.array(params.p)
            T_array = np.array(params.T)

            # find where p_surf is in and set the new surface
            idx = np.searchsorted(p_array, p_surf, side="left")
            idx = max(1, min(idx, len(p_array)))
            params.p = np.concatenate((p_array[:idx], [p_surf]))

            # get midpoint temperature for new surface
            orig_centers = 0.5 * (p_array[:-1] + p_array[1:])
            new_centers = 0.5 * (params.p[:-1] + params.p[1:])
            params.T = np.interp(new_centers, orig_centers, T_array)

        # ISOFIT defintion - to ensure h2o absorption is driven by total water column vapor input, q is set to zero.
        # NOTE This is a vSmartMOM hack (v2.2.0) which may be version dependent.
        # Alternatively, we could change the definition of H2O to be altitude dependent, but this would be a larger discussion.
        params.q = jl.zeros(jl.Float32, len(params.p))

        # Only update observer altitude if we are below TOA
        if vals["observer_elev_km"] < TOA_KM:
            params.obs_alt = [vals["observer_elev_km"]]

        # The vza and vaz have to be wrapped by a list -- even if it's a scalar.
        params.sza = vals["sza"]
        params.vza = [vals["vza"]]
        params.vaz = [vals["vaz"]]

        # Scattering portion of params
        params.scattering_params.λ_ref = self.config.engine.aot_ref
        params.scattering_params.rt_aerosols[0].τ_ref = vals["aot"]
        setattr(
            params.scattering_params.rt_aerosols[0].aerosol,
            "nᵣ",
            self.config.engine.aerosol_nr,
        )
        setattr(
            params.scattering_params.rt_aerosols[0].aerosol,
            "nᵢ",
            self.config.engine.aerosol_ni,
        )
        params.scattering_params.rt_aerosols[0].aerosol.size_distribution = (
            jl.vSmartMOM.Scattering.LogNormal(
                self.config.engine.aerosol_mu, self.config.engine.aerosol_sigma
            )
        )

        # # Absorption poriton of params
        # ~/data/HITRAN_LUTs/{O2,CO2,H2O,CH4,O3,N2O,CO,NO2,SO2}.jld2` (9 species)
        params.absorption_params.fixed_molecules = [
            ["O2", "CO2", "H2O", "CH4", "O3", "N2O", "CO", "NO2", "SO2"]
        ]
        params.absorption_params.vmr["H2O"] = h2o_vmr
        params.absorption_params.vmr["CO2"] = co2_vmr

        atm_dict = jl.ATM_VMRS[atm_type]
        for mol, vmr_val in atm_dict.items():
            params.absorption_params.vmr[mol] = vmr_val

        # Set same wavelegth grid that was set in preSim
        params.spec_bands = [self.wn_grid]

        FT = params.float_type

        # NOTE This currently assumes noRS. See note from vSmartMOM docs below:
        # "The bulk of retrievals (NIR shortwave) use `noRS` — the inelastic correction is small there.
        # UV/Vis retrievals (Ring effect, O2A-band line shapes, ozone DOAS) need RRS."
        RS = jl.vSmartMOM.InelasticScattering.noRS[FT]()

        # TODO - Figure out why this needed for some of the taus
        # There is likely a better fix...
        def check_tau_dim(tau):
            if tau.ndim == 3:
                return np.sum(tau, axis=(0, 2))
            elif tau.ndim == 2:
                return np.sum(tau, axis=1)
            return tau

        R_results = []
        hem_dw_results = []
        tau_results = []

        for alb in self.albedos:
            params.brdf = jl.seval(
                "(alb, FT) -> [vSmartMOM.LambertianSurfaceScalar{FT}(alb)]"
            )(alb, FT)
            model = jl.vSmartMOM.CoreRT.model_from_parameters(params)

            # rt_run returns 7-tuple (R_SFI, T_SFI, ieR_SFI, ieT_SFI, hdr, bhr_uw, bhr_dw)
            R_SFI_tot, _, _, _, _, _, hem_T_tot = jl.vSmartMOM.CoreRT.rt_run(RS, model, 1)

            R_tot_I = np.array(R_SFI_tot)[0, 0, :]

            tau_total = (
                check_tau_dim(np.array(model.τ_abs[0]))
                + check_tau_dim(np.array(model.τ_rayl[0]))
                + check_tau_dim(np.array(model.τ_aer[0]))
            )

            R_results.append(R_tot_I)
            hem_dw_results.append(np.array(hem_T_tot).flatten())
            tau_results.append(tau_total)

        # math for MODTRAN equivalent field via Suniti's code in vSmartMOM test codes
        L0 = R_results[0]
        L2 = R_results[1]
        hT0 = hem_dw_results[0]
        hT2 = hem_dw_results[1]
        tau_tot = tau_results[0]

        mu_s = np.cos(np.radians(vals["sza"]))
        mu_v = np.cos(np.radians(vals["vza"]))

        # path reflectance for zero albedo
        rhoatm = np.pi * L0 / mu_s

        # spherical albedo
        sphalb = np.where(
            hT2 > np.finfo(float).eps, (1.0 - hT0 / hT2) / self.albedos[1], 0.0
        )
        sphalb = np.clip(sphalb, 0.0, 1.0)

        # split transmittances
        transm_down_dir = np.exp(-tau_tot / mu_s)
        T_down_tot = hT0 / mu_s
        transm_down_dif = np.maximum(T_down_tot - transm_down_dir, 0.0)

        transm_up_dir = np.exp(-tau_tot / mu_v)
        alpha_rho = np.pi * (L2 - L0) / mu_s
        T_up_tot = np.where(
            hT0 > np.finfo(float).eps,
            alpha_rho
            * (1.0 - self.albedos[1] * sphalb)
            / (self.albedos[1] * T_down_tot),
            transm_up_dir,
        )
        transm_up_dif = np.maximum(T_up_tot - transm_up_dir, 0.0)

        transm_down_dif = np.maximum(transm_down_dif, 0)
        transm_up_dif = np.maximum(transm_up_dif, 0)
        sphalb = np.maximum(sphalb, 0)

        wl = units.wavenumber_to_nm(np.array(params.spec_bands[0]))

        # Flip arrays - to increase in wl from VIS to SWIR
        wl = wl[::-1]
        rhoatm = rhoatm[::-1]
        sphalb = sphalb[::-1]
        transm_down_dir = transm_down_dir[::-1]
        transm_down_dif = transm_down_dif[::-1]
        transm_up_dir = transm_up_dir[::-1]
        transm_up_dif = transm_up_dif[::-1]

        data = np.column_stack(
            [
                wl,
                rhoatm,
                sphalb,
                transm_down_dir,
                transm_down_dif,
                transm_up_dir,
                transm_up_dif,
            ]
        )
        header = "wl,rhoatm,sphalb,transm_down_dir,transm_down_dif,transm_up_dir,transm_up_dif"
        np.savetxt(out_path, data, delimiter=",", header=header, comments="")

        return
