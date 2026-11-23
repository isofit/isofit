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
    # wrapper for rt_run() call that suppresses the progressbars/logs derived from julia process
    jl.seval(
        """
        using Logging
        if isdefined(Main, :ProgressMeter) || Base.find_package("ProgressMeter") !== nothing
            @eval begin
                import ProgressMeter
                ProgressMeter.updateProgress!(p, args...; kwargs...) = nothing
                ProgressMeter.printover(io, args...; kwargs...) = nothing
            end
        end
        function _run_vsmartmom_quiet(f)
            old_logger = global_logger(NullLogger())
            try
                redirect_stdout(devnull) do
                    redirect_stderr(devnull) do
                        with_logger(NullLogger()) do
                            return f()
                        end
                    end
                end
            finally
                global_logger(old_logger)
            end
        end
        """
    )
    _jl_run_vsmartmom_quiet = jl.seval("_run_vsmartmom_quiet")
    _jl_deepcopy = jl.seval("deepcopy")

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
_sim_cache: dict[str, np.ndarray] = {}


class VSmartMomRT(BaseAtmosphere, Writer):

    albedos = [0.0, 0.2]

    # TODO

    # - think about how we want to interface with scattering parameters (e.g., angstrom exponent)

    # - Chappuis still pending issue

    # - double check RAA based on Phils PR-1070

    # - Check UV and Conda install to check changes needed

    # - Bring in atm profile into isofit-data

    def __init__(self, full_config, **kwargs):
        self.config = full_config.forward_model.atmosphere
        self.sim_path = self.config.sim_path

        self.o2_total = 167820
        self.o2_vmr = 0.2095  # treated as constant vertically
        self.so2_vmr = (
            1.353e-10  # SO2 effect above 350 nm ~ zero, value represents midlat summer
        )
        self.toa_km = 100.0

        super().__init__(full_config, **kwargs)

    def preSim(self):

        # define the output wl grid [nm], and stash matrix H
        # The wn (wavenumber) grid is also stored to self and used for vsmartmom sims.
        # Loop through each, sorted by resolution
        wn, intervals = [], []
        for wl1, wl2, step_wn in sorted(
            self.config.engine.wl_resolutions, key=lambda b: b[2]
        ):
            wn1 = units.nm_to_wavenumber(wl1)
            wn2 = units.nm_to_wavenumber(wl2)
            start_wn = min(wn1, wn2)
            end_wn = max(wn1, wn2)
            g = np.arange(start_wn, end_wn + 1e-8, step_wn)

            # guard against empty range and keep low wl (end_wn)
            if len(g) == 0 or abs(g[-1] - end_wn) > 1e-6:
                g = np.append(g, end_wn)

            # drop points that are within finer resolution intervals
            to_drop = np.zeros(len(g), dtype=bool)
            for lo, hi in intervals:
                to_drop |= (g >= lo - 1e-6) & (g <= hi + 1e-6)
            g = g[~to_drop]

            wn.append(g)
            intervals.append((start_wn, end_wn))

        # join and deduplicate points
        wn = np.sort(np.concatenate(wn))
        self.wn_grid = wn[np.concatenate(([True], np.diff(wn) > 1e-6))]

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

        # Read the atm profiles data from isofit-data
        # Korkin, S., Sayer, A. M., Ibrahim, A., & Lyapustin, A. (2025).
        # A practical guide to coding line-by-line trace gas absorption in Earth's atmosphere.
        # Journal of Quantitative Spectroscopy and Radiative Transfer, 337, 109345.
        atm_profile_path = env.path("data", "atm_1996_profiles.csv")
        atm_profile_data = np.genfromtxt(
            atm_profile_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
        )

        atm_idx = atm_profile_data["atm_type"].astype(str) == self.atm_type

        fields = [
            "z_km",
            "p_mbar",
            "t_kelvin",
            "q_kgkg",
            "o3_vmr",
            "n2o_vmr",
            "co_vmr",
            "ch4_vmr",
            "no2_vmr",
        ]

        self.atm_profile = {
            k: np.asarray(atm_profile_data[k][atm_idx], dtype=float) for k in fields
        }

        self.p_def = self.atm_profile["p_mbar"]
        self.T_def = self.atm_profile["t_kelvin"]
        self.q_def = self.atm_profile["q_kgkg"]
        self.log_p_def = np.log(self.p_def)

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

        # initialize vsmartmom static params based on default yaml (based on Version 2.2.0)
        # https://github.com/RemoteSensingTools/vSmartMOM.jl/blob/release/v2.2-consolidation/src/CoreRT/DefaultParameters.yaml
        self.params = jl.vSmartMOM.default_parameters()
        self.params.nstreams = self.config.engine.nstreams_mom
        self.params.l_trunc = 2 * self.config.engine.nstreams_mom - 1
        self.params.stream_l_cap = 2 * self.config.engine.nstreams_mom - 1
        self.params.architecture = jl.seval(self.config.engine.architecture)
        self.params.float_type = jl.seval(self.config.engine.float_type)
        self.params.scattering_params.λ_ref = self.config.engine.aot_ref
        self.params.spec_bands = [self.wn_grid]
        setattr(
            self.params.scattering_params.rt_aerosols[0].aerosol,
            "nᵣ",
            self.config.engine.aerosol_nr,
        )
        setattr(
            self.params.scattering_params.rt_aerosols[0].aerosol,
            "nᵢ",
            self.config.engine.aerosol_ni,
        )
        self.params.scattering_params.rt_aerosols[0].aerosol.size_distribution = (
            jl.vSmartMOM.Scattering.LogNormal(
                self.config.engine.aerosol_mu, self.config.engine.aerosol_sigma
            )
        )
        self.params.absorption_params.fixed_molecules = [
            ["O2", "CO2", "CH4", "O3", "N2O", "CO", "NO2", "SO2"]
        ]

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

        if not self.config.engine.write_sim:
            if not self.config.configure_and_exit:
                _sim_cache[name] = self.run_vsmartmom(vals, out_path=None)
                return

        if not self.config.configure_and_exit:
            self.run_vsmartmom(vals, out_path)

    def readSim(self, point):
        name = self.point_to_filename(point)
        out_path = abspath(join(self.sim_path, f"{name}_vsmartmom.csv"))

        if not self.config.engine.write_sim:
            data = _sim_cache[name]
            _sim_cache.pop(name)
        else:
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

        params = _jl_deepcopy(self.params)

        p_surf = units.standard_atmosphere()
        if vals["surface_elev_km"] > 0.0:
            p_surf = units.standard_atmosphere() * np.exp(
                -vals["surface_elev_km"] / 8.0
            )

        co2_vmr = (vals["co2"] / self.o2_total) * self.o2_vmr

        # Truncate pressure grid if above sea level
        if vals["surface_elev_km"] > 0.0:
            idx = np.searchsorted(self.p_def, p_surf, side="left")
            idx = max(1, min(idx, len(self.p_def) - 1))
            params.p = np.concatenate((self.p_def[:idx], [p_surf]))
        else:
            params.p = self.p_def

        p_final = np.array(params.p)
        p_centers = 0.5 * (p_final[:-1] + p_final[1:])

        log_p_mid = np.log(np.clip(p_centers, self.p_def.min(), self.p_def.max()))
        params.T = np.interp(log_p_mid, self.log_p_def, self.T_def)
        q_array = np.interp(log_p_mid, self.log_p_def, self.q_def)
        z_centers = np.interp(log_p_mid, self.log_p_def, self.atm_profile["z_km"])

        gas_profiles = {
            "O3": np.interp(log_p_mid, self.log_p_def, self.atm_profile["o3_vmr"]),
            "N2O": np.interp(log_p_mid, self.log_p_def, self.atm_profile["n2o_vmr"]),
            "CO": np.interp(log_p_mid, self.log_p_def, self.atm_profile["co_vmr"]),
            "CH4": np.interp(log_p_mid, self.log_p_def, self.atm_profile["ch4_vmr"]),
            "NO2": np.interp(log_p_mid, self.log_p_def, self.atm_profile["no2_vmr"]),
        }

        above_toa = z_centers > self.toa_km
        if np.any(above_toa):
            for k in gas_profiles:
                gas_profiles[k] = np.where(above_toa, 0.0, gas_profiles[k])

        # Absorption portion of params
        # H2O is set via q and not absorption_params.vmr
        dp_gcm2 = units.hPa_to_gcm2(np.abs(np.diff(p_final)))
        w_base_gcm2 = np.sum(q_array * dp_gcm2)
        params.q = q_array * ((vals["h2o"] / w_base_gcm2) if w_base_gcm2 > 0.0 else 0.0)

        params.absorption_params.vmr["CO2"] = co2_vmr
        params.absorption_params.vmr["O2"] = self.o2_vmr
        params.absorption_params.vmr["SO2"] = self.so2_vmr

        for mol, vmr_profile in gas_profiles.items():
            params.absorption_params.vmr[mol] = vmr_profile

        # Only update observer altitude if we are below TOA
        if vals["observer_elev_km"] < self.toa_km:
            params.obs_alt = [vals["observer_elev_km"]]

        # The vza and vaz have to be wrapped by a list -- even if it's a scalar.
        params.sza = vals["sza"]
        params.vza = [vals["vza"]]
        params.vaz = [vals["vaz"]]

        # Set aerosol optical thickness
        params.scattering_params.rt_aerosols[0].τ_ref = vals["aot"]

        # This currently assumes no raman scattering. See note from vSmartMOM docs below:
        # "The bulk of retrievals (NIR shortwave) use `noRS` — the inelastic correction is small there.
        # UV/Vis retrievals (Ring effect, O2A-band line shapes, ozone DOAS) need RRS."
        FT = params.float_type
        RS = jl.vSmartMOM.InelasticScattering.noRS[FT]()

        def sum_tau(tau):
            if tau.ndim == 3:
                return np.sum(tau, axis=(0, 2))
            elif tau.ndim == 2:
                return np.sum(tau, axis=1)
            return tau

        R_results = []
        hem_dw_results = []
        tau_results = []

        def loop_albedos():
            for alb in self.albedos:
                params.brdf = jl.seval(
                    "(alb, FT) -> [vSmartMOM.LambertianSurfaceScalar{FT}(alb)]"
                )(alb, FT)
                model = jl.vSmartMOM.CoreRT.model_from_parameters(params)

                # rt_run returns 7-tuple (R_SFI, T_SFI, ieR_SFI, ieT_SFI, hdr, bhr_uw, bhr_dw)
                R_SFI_tot, _, _, _, _, _, hem_T_tot = jl.vSmartMOM.CoreRT.rt_run(
                    RS, model, 1
                )

                R_tot_I = np.array(R_SFI_tot)[0, 0, :]

                # total column optical depth composed of gaseous absorption, rayleigh, and aerosol
                tau_total = (
                    sum_tau(np.array(model.τ_abs[0]))
                    + sum_tau(np.array(model.τ_rayl[0]))
                    + sum_tau(np.array(model.τ_aer[0]))
                )

                R_results.append(R_tot_I)
                hem_dw_results.append(np.array(hem_T_tot).flatten())
                tau_results.append(tau_total)

        _jl_run_vsmartmom_quiet(loop_albedos)

        # math for MODTRAN equivalent field via Suniti's code in vSmartMOM test codes
        L0 = R_results[0]
        L2 = R_results[1]
        hT0 = hem_dw_results[0]
        hT2 = hem_dw_results[1]
        tau_tot = tau_results[0]

        mu_s = np.cos(np.radians(vals["sza"]))
        mu_v = np.cos(np.radians(vals["vza"]))

        # ignore divide by zero warnings
        with np.errstate(divide="ignore", invalid="ignore"):

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

        header = "wl,rhoatm,sphalb,transm_down_dir,transm_down_dif,transm_up_dir,transm_up_dif"
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
        if out_path is None:
            return data
        else:
            np.savetxt(out_path, data, delimiter=",", header=header, comments="")

        return
