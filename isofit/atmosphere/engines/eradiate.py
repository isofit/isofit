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
import os

import numpy as np
import xarray as xr

# Eradiate is an optional dependency. Raise rather than print the recommendation
# so isofit.atmosphere.engines can still defer it until the engine is requested
try:
    import eradiate
    import joseki
    from eradiate.attrs import AUTO
    from eradiate.config import ProgressLevel, settings
    from eradiate.experiments import AtmosphereExperiment
    from eradiate.radprops import ParticleProperties, absdb_factory
    from eradiate.rng import SeedState
    from eradiate.scenes.atmosphere import (
        HeterogeneousAtmosphere,
        MolecularAtmosphere,
        ParticleLayer,
    )
    from eradiate.scenes.bsdfs import LambertianBSDF
    from eradiate.scenes.illumination import DirectionalIllumination
    from eradiate.scenes.integrators import PiecewiseVolPathIntegrator
    from eradiate.scenes.measure import MultiDistantMeasure
    from eradiate.scenes.surface import BasicSurface
    from eradiate.spectral import BandSRF, CKDSpectralIndex, SpectralIndex
    from eradiate.units import to_quantity
    from eradiate.units import unit_registry as ureg
except ImportError as e:
    from isofit.data.cli.eradiate import print_install

    raise ImportError(
        "Eradiate is not installed, it is an optional dependency of ISOFIT\n"
        'To install it, run: pip install "isofit[eradiate]"\n'
        "Or if using uv: uv sync --extra eradiate"
    ) from e

from isofit.atmosphere.atmosphere import BaseAtmosphere
from isofit.core import units
from isofit.core.common import json_load_ascii
from isofit.data.cli.eradiate import set_data_path
from isofit.luts.writer import Writer

Logger = logging.getLogger(__name__)


class EradiateRT(BaseAtmosphere, Writer):
    # ...
    # Eradiate's Monte Carlo integrator reports total TOA radiance per
    # run, whereas MODTRAN splits path radiance from ground-
    # reflected radiance in its channel output. To reuse `two_albedo_method`
    # we reconstruct the same inputs by:
    #   - computing direct transmittances analytically (Beer-Lambert,
    #     from the atmosphere's own extinction profile), alongside an analytical
    #     correlation scalar (C_ckd) to handle correlated-k gas absorption.
    #   - adding a second, BOA, nadir-viewing sensor to every run (rendered
    #     alongside the main TOA-view sensor in the same pass) to directly
    #     measure the Lambertian surface's leaving radiance, from which the
    #     ground-reflected and path-radiance components follow algebraically.
    # ...

    # Surface albedos used for the two-albedo decomposition, matching
    # modtran.py convention
    albedos = [0.0, 0.1, 0.5]

    # Reference wavelength for aerosol optical thickness (nm)
    aot_w_ref = 550.0

    # Fallbacks for the background gas and aerosol load, used only when
    # neither a LUT dimension nor the MODTRAN template supplies them (see
    # resolve_point). AFGL 1986's own values are badly dated -- it carries
    # CO2 at 330 ppm and CH4 at 1.7 ppm at the surface. CO2 is uniform, CH4 a
    # surface value the profile's shape is scaled to.
    default_co2_ppm = 420.0
    default_ch4_ppm = 1.93
    default_aot550 = 0.1

    # Height of the BOA "surface-leaving radiance" sensor above ground (m)
    boa_sensor_height = 1.0

    # Per-channel Gaussian SRF discretization: srf_n_inner points span +/-
    # srf_sigma_inner standard deviations (resolving the response's shape),
    # bracketed by an explicit zero at +/- srf_sigma_outer sigma (BandSRF
    # requires the response to reach zero at its support boundary).
    # We need to rethink this for the 0.1 nm mono solution - this would
    # ridiculously oversample things...just what I used for the broader
    # band tests.
    srf_n_inner = 9
    srf_sigma_inner = 2.5
    srf_sigma_outer = 3.0

    # when true, clip terms between 0 and 1
    clip_01 = True

    def __init__(self, full_config, wl=[], fwhm=[], **kwargs):
        """Inits the engine and loads the base atmosphere/aerosol datasets.

        Args:
            full_config: Full ISOFIT config.
            wl: Wavelength override (nm).
            fwhm: FWHM override (nm).
            **kwargs: Forwarded to BaseAtmosphere.__init__.
        """
        self.config = full_config.forward_model.atmosphere

        # Eradiate data always come from the ini eradiate path
        set_data_path()
        self.set_progress(self.config.engine.eradiate_progress)

        self.eradiate_mode = self.infer_spectral_mode(
            self.config.engine.eradiate_absorption_database
        )

        self.engine_base_dir = getattr(self.config.engine, "base_dir", None)
        self.sim_path = self.config.sim_path

        eradiate.set_mode(self.eradiate_mode)

        # Parsed once here rather than per-point in readSim, which runs in
        # ray workers
        self._template_defaults = self.template_defaults(
            self.config.engine.template_file
        )

        # Loaded once; per-point gas/aerosol scaling starts from these and
        # never mutates them in place.
        self._base_thermoprops = self.default_thermoprops()
        self._base_particle_properties = self.load_particle_properties(
            self.config.engine.eradiate_particle_properties
        )

        # Terms readSim computes and SRF-averages. Set before super().__init__,
        # which builds the LUT and so already calls readSim
        self.output_terms = list(self.lut_keys.alldim) + ["solar_irr"]

        super().__init__(full_config, wl=wl, fwhm=fwhm, **kwargs)

    def preSim(self):
        """Records engine configuration on the LUT for provenance."""
        self.lut.setAttr("eradiate_mode", self.eradiate_mode)
        self.lut.setAttr(
            "eradiate_absorption_database",
            self.config.engine.eradiate_absorption_database,
        )
        self.lut.setAttr(
            "eradiate_particle_properties",
            self.config.engine.eradiate_particle_properties,
        )

        # Not required, but choosing to carry in radiance units
        self.lut.setAttr("RT_mode", "rdn")

    def makeSim(self, point: np.array, template_only: bool = False):
        """No-op: eradiate needs no on-disk input deck, everything happens in
        readSim.

        Args:
            point: Unused.
            template_only: Unused.
        """

    def readSim(self, point: np.array):
        """Runs the three albedo simulations for a LUT point and reduces
        them to standard terms via two_albedo_method, once per channel's
        sub-wavelength grid, then SRF-averages the resulting terms.
        This only gives 6c results...obviously it could be rebuilt to
        do 3c, but I opted to move everything in the exclusive 6c direction.

        Args:
            point: LUT point to process.

        Returns:
            dict: LUT variable names to per-wavelength values.
        """
        eradiate.set_mode(self.eradiate_mode)  # per-process; cheap & idempotent
        set_data_path()  # per-process too; ray workers re-import eradiate
        self.set_progress(self.config.engine.eradiate_progress)

        vals = self.resolve_point(point)

        wl = np.asarray(self.wl, dtype=float)  # nm
        fwhm = np.asarray(self.fwhm, dtype=float)  # nm
        coszen = np.cos(np.deg2rad(vals["solzen"]))
        cosvza = np.cos(np.deg2rad(vals["viewzen"]))

        atmosphere = self.build_atmosphere(vals)

        exp, measures_toa, measures_boa = self.build_experiment(
            atmosphere, vals, wl, fwhm
        )
        exp.init()

        # Common random numbers: re-seeding to the same, point-derived value
        # before every albedo re-run means the three Monte Carlo estimates
        # share the same underlying photon-path realizations.
        seed = tuple(int(round(p * 1e6)) for p in point)

        results = {}
        for albedo in self.albedos:
            self.set_reflectance(exp, albedo)
            results[albedo] = eradiate.run(
                exp,
                spp=self.config.engine.eradiate_spp,
                seed_state=SeedState(seed=seed),
            )

        rfl_1, rfl_2 = self.albedos[1], self.albedos[2]

        data = {key: np.empty_like(wl, dtype=float) for key in self.output_terms}
        for i in range(len(wl)):
            data_i = self.decompose_channel(
                atmosphere,
                exp,
                results,
                measures_toa[i],
                measures_boa[i],
                wl[i],
                fwhm[i],
                coszen,
                cosvza,
                rfl_1,
                rfl_2,
            )
            for key in self.output_terms:
                data[key][i] = data_i[key]

        data["wl"] = wl
        data["solzen"] = vals["solzen"]
        data["coszen"] = coszen
        return data

    def decompose_channel(
        self,
        atmosphere,
        exp,
        results: dict,
        measure_toa,
        measure_boa,
        center: float,
        fwhm: float,
        coszen: float,
        cosvza: float,
        rfl_1: float,
        rfl_2: float,
    ) -> dict:
        def read(results_dict, albedo, measure):
            radiance = results_dict[albedo][measure.id]["radiance"].squeeze()
            order = np.argsort(radiance.coords["w"].values)
            w = radiance.coords["w"].values[order]
            values = (
                radiance.values[order]
                * ureg(radiance.attrs.get("units", "W/m^2/sr/nm"))
            ).m_as("uW/cm^2/sr/nm")
            return w, values

        w_sub = None
        toa_sub, boa_sub = {}, {}
        for albedo in self.albedos:
            w_sub, toa_sub[albedo] = read(results, albedo, measure_toa)
            _, boa_sub[albedo] = read(results, albedo, measure_boa)

        t_down_sub = np.empty_like(w_sub)
        t_up_sub = np.empty_like(w_sub)
        c_ckd_sub = np.empty_like(w_sub)
        for j, w in enumerate(w_sub):
            t_down_sub[j], t_up_sub[j], c_ckd_sub[j] = self.wavelength_transmittances(
                atmosphere, exp, w, coszen, cosvza
            )

        irr_sub = exp.illumination.irradiance.eval_mono(w=w_sub * ureg.nm).m_as(
            "uW/cm^2/nm"
        )

        case_0 = {
            "width": np.ones_like(w_sub),
            "transm_up_dir": t_up_sub,
            "solar_irr": irr_sub,
            "wl": w_sub,
            "rhoatm": toa_sub[0.0],
            "thermal_upwelling": np.zeros_like(w_sub),
            "thermal_downwelling": np.zeros_like(w_sub),
        }

        def subcase(rfl):
            # 100% exact analytical direct ground-reflected radiance (zero MC noise)
            drct_rflt = (
                rfl * (irr_sub * coszen / np.pi) * t_down_sub * t_up_sub * c_ckd_sub
            )

            # Surface-leaving radiance scaled by direct up transmittance and analytical CKD correlation
            grnd_rflt = boa_sub[rfl] * t_up_sub * c_ckd_sub

            return {
                "grnd_rflt": grnd_rflt,
                "drct_rflt": drct_rflt,
                "path_rdn": toa_sub[rfl] - grnd_rflt,
            }

        sub_data = self.two_albedo_method(
            case_0, subcase(rfl_1), subcase(rfl_2), coszen, rfl_1, rfl_2
        )

        # Prevent exploding values
        valid_signal = (t_down_sub * t_up_sub) > 1e-8
        for key in ["transm_up_dif", "transm_down_dif", "sphalb"]:
            sub_data[key] = np.where(valid_signal, sub_data[key], 0.0)

        # Optional 0-1 clip.
        if self.clip_01:
            for key in [
                "sphalb",
                "transm_up_dir",
                "transm_up_dif",
                "transm_down_dir",
                "transm_down_dif",
            ]:
                if key in sub_data:
                    sub_data[key] = np.clip(sub_data[key], 0.0, 1.0)

        # Pair the transmittances here rather than leaving it to couple() at
        # load time: pairing before the SRF average keeps the product
        # consistent with the sub-wavelength transmittances rather than the
        # channel means
        l_solar = units.E_to_L(irr_sub, coszen)
        sub_data = self.couple(sub_data)
        for term in self.coupling_terms:
            sub_data[term] = sub_data[term] * l_solar

        sigma = fwhm / 2.3548200450309493
        response = np.exp(-0.5 * ((w_sub - center) / sigma) ** 2)
        return {
            key: self.srf_weighted_average(w_sub, response, sub_data[key])
            for key in self.output_terms
        }

    def resolve_point(self, point: np.array) -> dict:
        """Merges fixed engine-config geometry with this LUT point's
        values, mapping ISOFIT's geometry-key conventions onto eradiate's.

        Values resolve in order of decreasing specificity:
            1) Lut point
            2) engine config
            3) MODTRAN template
            4) class defaults

        Args:
            point: LUT point to process.

        Returns:
            dict: Geometry and state-vector values keyed by name.
        """

        def setting(key, default=None):
            val = getattr(self.config.engine, key, None)
            if val is None:
                val = self._template_defaults.get(key, default)
            return val

        vals = {
            "solzen": setting("solzen"),
            "solaz": setting("solaz"),
            "viewzen": setting("viewzen"),
            "viewaz": setting("viewaz"),
            "elev": abs(max(setting("elev", 0.0) or 0.0, 0.0)),
            # None means "satellite/TOA sensor" (infinite distance); only a
            # value strictly below the atmosphere top switches to a finite,
            # airborne ray_offset (see build_experiment).
            "alt": setting("alt"),
            # Gases and aerosol the state vector doesn't always retrieve, so
            # they need a sensible default value
            "CO2": self._template_defaults.get("CO2", self.default_co2_ppm),
            "CH4": self._template_defaults.get("CH4", self.default_ch4_ppm),
            "AOT550": self._template_defaults.get("AOT550", self.default_aot550),
        }

        # A LUT dimension of the same name always takes precedence
        for key, val in zip(self.lut_names, point):
            vals[key] = val

        if "surface_elevation_km" in vals:
            vals["elev"] = abs(max(vals["surface_elevation_km"], 0.0))
        if "observer_altitude_km" in vals:
            vals["alt"] = vals["observer_altitude_km"]
        if "observer_azimuth" in vals:
            vals["viewaz"] = vals["observer_azimuth"]
        if "observer_zenith" in vals:
            vals["viewzen"] = vals["observer_zenith"]
        if "solar_zenith" in vals:
            vals["solzen"] = vals["solar_zenith"]
        if "solar_azimuth" in vals:
            vals["solaz"] = vals["solar_azimuth"]
        if "relative_azimuth" in vals:
            vals["solaz"] = vals["viewaz"] + vals["relative_azimuth"]

        return vals

    def build_atmosphere(self, vals: dict):
        """Builds a HeterogeneousAtmosphere for a LUT point's state vector.

        Args:
            vals: Geometry and state-vector values keyed by name.

        Returns:
            HeterogeneousAtmosphere: Molecular + aerosol atmosphere.
        """
        thermoprops = self._base_thermoprops
        if "H2OSTR" in vals:
            thermoprops = self.scale_h2o_column(thermoprops, vals["H2OSTR"])
        if "CO2" in vals:
            thermoprops = self.set_uniform_mixing_ratio(
                thermoprops, "x_CO2", vals["CO2"]
            )
        if "CH4" in vals:
            thermoprops = self.scale_surface_mixing_ratio(
                thermoprops, "x_CH4", vals["CH4"]
            )
        if "O3" in vals:
            thermoprops = self.set_uniform_mixing_ratio(thermoprops, "x_O3", vals["O3"])

        molecular = MolecularAtmosphere(
            thermoprops=thermoprops,
            absorption_data=self.config.engine.eradiate_absorption_database,
        )

        particle_properties = self._base_particle_properties
        if "AERANGSTROM" in vals:
            particle_properties = self.angstrom_particle_properties(
                particle_properties,
                vals["AERANGSTROM"],
                self.aot_w_ref * ureg.nm,
            )

        aerosol = ParticleLayer(
            bottom=0.0 * ureg.km,
            top=2.0 * ureg.km,
            tau_ref=max(vals.get("AOT550", 0.0), 1e-6),
            w_ref=self.aot_w_ref * ureg.nm,
            particle_properties=particle_properties,
        )

        return HeterogeneousAtmosphere(
            molecular_atmosphere=molecular, particle_layers=[aerosol]
        )

    def build_experiment(
        self,
        atmosphere,
        vals: dict,
        wl: np.array,
        fwhm: np.array,
    ):
        """Builds an AtmosphereExperiment with one TOA (or mid-column) and
        one BOA nadir sensor per wavelength channel, each pair sharing a
        Gaussian BandSRF built from that channel's own FWHM.

        Args:
            atmosphere: Atmosphere to place in the experiment.
            vals: Geometry and state-vector values keyed by name.
            wl: Channel center wavelengths (nm).
            fwhm: Channel FWHM (nm), same length as wl.

        Returns:
            tuple: (AtmosphereExperiment, TOA measures, BOA measures).
        """
        illumination = DirectionalIllumination(
            zenith=vals["solzen"] * ureg.deg,
            azimuth=vals["solaz"] * ureg.deg,
        )

        # Sensor altitude: a finite ray_offset for airborne cases, otherwise
        # an infinite-distance (satellite/TOA) sensor. `alt` is None (the
        # config default) or >= the profile top for satellite/TOA sensors.
        toa_km = float(atmosphere.geometry.zgrid.levels.max().m_as(ureg.km))
        alt_km, ground_km = vals["alt"], vals["elev"]
        if alt_km is not None and alt_km < toa_km:
            cosvza = max(np.cos(np.deg2rad(vals["viewzen"])), 1e-6)
            ray_offset_toa = max(alt_km - ground_km, 0.0) / cosvza * ureg.km
        else:
            ray_offset_toa = None

        measures_toa = []
        measures_boa = []
        for i, (center, fw) in enumerate(zip(wl, fwhm)):
            w_sub, response = self.srf_grid(center, fw)
            srf = BandSRF(wavelengths=w_sub * ureg.nm, values=response)
            measures_toa.append(
                MultiDistantMeasure.from_angles(
                    angles=[[vals["viewzen"], vals["viewaz"]]],
                    ray_offset=ray_offset_toa,
                    srf=srf,
                    id=f"toa_{i}",
                )
            )
            measures_boa.append(
                MultiDistantMeasure.from_angles(
                    angles=[[0.0, 0.0]],
                    ray_offset=self.boa_sensor_height * ureg.m,
                    srf=srf,
                    id=f"boa_{i}",
                )
            )

        surface = BasicSurface(bsdf=LambertianBSDF(reflectance=0.0))

        exp = AtmosphereExperiment(
            geometry="plane_parallel",
            atmosphere=atmosphere,
            surface=surface,
            illumination=illumination,
            measures=measures_toa + measures_boa,
        )
        return exp, measures_toa, measures_boa

    @staticmethod
    def template_defaults(template_file: str) -> dict:
        """Pulls the scene's geometry, gas and aerosol defaults out of the
        MODTRAN template apply_oe writes, mapping MODTRAN's parameter
        names onto the engine's own. Follows the same mapping
        template_construction uses to recover scene means for a prebuilt LUT.

        Args:
            template_file: Path to the MODTRAN template, or None.

        Returns:
            dict: Engine-keyed defaults found in the template. Keys absent
            from the template are left out entirely so resolve_point can fall
            back to the class defaults.
        """
        if not template_file or not os.path.exists(template_file):
            return {}

        template = json_load_ascii(template_file)
        if "cases" in template:
            template = template["cases"][0]
        template = template["MODTRAN"][0]["MODTRANINPUT"]

        geometry = template.get("GEOMETRY", {})
        atmosphere = template.get("ATMOSPHERE", {})

        defaults = {
            "solzen": geometry.get("PARM2"),
            "viewaz": geometry.get("TRUEAZ"),
            "elev": template.get("SURFACE", {}).get("GNDALT"),
            "alt": geometry.get("H1ALT"),
            "CO2": atmosphere.get("CO2MX"),
            "CH4": atmosphere.get("CH4MX"),
        }

        # MODTRAN reports the observer zenith in its own convention
        obszen = geometry.get("OBSZEN")
        if obszen is not None:
            defaults["viewzen"] = 180.0 - obszen

        # IPARM 12 carries the solar azimuth relative to the view azimuth,
        # the same sum resolve_point applies to a relative_azimuth dimension
        trueaz, parm1 = geometry.get("TRUEAZ"), geometry.get("PARM1")
        if trueaz is not None and parm1 is not None:
            defaults["solaz"] = trueaz + parm1

        # MODTRAN carries the aerosol load as a visibility, not an AOT
        vis = template.get("AEROSOLS", {}).get("VIS")
        if vis is not None:
            defaults["AOT550"] = units.vis_to_aod(vis)

        return {key: val for key, val in defaults.items() if val is not None}

    @staticmethod
    def infer_spectral_mode(database: str) -> str:
        """Infers eradiate's mono/ckd mode from the absorption database name
        (gecko/komodo are mono-only, monotropa/mycena/panellus/tuber are
        CKD-only), so mode and database can't disagree.

        Args:
            database: Absorption database name.

        Returns:
            str: "mono" or "ckd".
        """
        is_ckd = "CKD" in type(absdb_factory.create(database)).__name__
        return "ckd" if is_ckd else "mono"

    @staticmethod
    def set_progress(show: bool):
        """Turns eradiate's progress bars on or off. Eradiate reads this
        setting on every run, so it can be changed after import, but it must
        be set in each process (eg. ray workers).

        Args:
            show: Show the spectral loop and kernel progress bars.
        """
        level = ProgressLevel.KERNEL if show else ProgressLevel.NONE
        settings.set("PROGRESS", level)

    @classmethod
    def default_thermoprops(cls):
        """Loads the default AFGL US-standard thermophysical profile
        (0-120 km), matching MolecularAtmosphere's own default, then brings
        the background gases up to present-day concentrations. AFGL 1986 is
        a 1986 profile: it carries CO2 at 330 ppm and CH4 at 1.7 ppm, which
        is sadly dated. CO2 is well mixed so it is set uniformly,
        whereas CH4 is only well mixed through the troposphere and so scales
        the profile instead.

        Returns:
            xr.Dataset: Thermophysical profile.
        """
        thermoprops = joseki.make(
            identifier="afgl_1986-us_standard",
            z=np.linspace(0.0, 120.0, 121) * ureg.km,
        )
        thermoprops = cls.set_uniform_mixing_ratio(
            thermoprops, "x_CO2", cls.default_co2_ppm
        )
        return cls.scale_surface_mixing_ratio(thermoprops, "x_CH4", cls.default_ch4_ppm)

    @staticmethod
    def load_particle_properties(dataset_id: str):
        """Loads a named (or path-specified) aerosol particle-properties
        dataset.

        Args:
            dataset_id: Dataset name or path.

        Returns:
            ParticleProperties: Loaded aerosol optical properties.
        """
        return ParticleProperties.convert(dataset_id)

    @staticmethod
    def trapz(y: np.array, x: np.array) -> float:
        """Trapezoidal integral of y(x). A local replacement for the removed
        np.trapz (np.trapezoid, its numpy>=2.0 replacement, isn't available
        on all supported numpy versions either, so this avoids both names).

        Args:
            y: Values to integrate.
            x: Sample points, same length as y.

        Returns:
            float: Trapezoidal integral.
        """
        dx = np.diff(x)
        return float(np.sum(0.5 * (y[:-1] + y[1:]) * dx))

    @staticmethod
    def scale_h2o_column(thermoprops, h2ostr_g_cm2: float):
        """Scales the profile's water vapor mixing ratio so its column
        matches H2OSTR (g/cm2), preserving the default profile's vertical
        shape.

        Args:
            thermoprops: Thermophysical profile to scale.
            h2ostr_g_cm2: Target water vapor column (g/cm^2).

        Returns:
            xr.Dataset: Thermophysical profile with scaled x_H2O.
        """
        n = to_quantity(thermoprops.n).m_as("1/cm^3")  # number density
        x_h2o = thermoprops["x_H2O"].values
        z_cm = to_quantity(thermoprops.z).m_as("cm")

        # Mass column (g/cm^2) of water vapor implied by the current profile.
        mass_density = x_h2o * n * units.h2o_molar_mass() / units.avagadro()
        current_column = EradiateRT.trapz(mass_density, z_cm)

        if current_column <= 0:
            Logger.warning("Default H2O column is non-positive, cannot scale H2OSTR")
            return thermoprops

        scale = h2ostr_g_cm2 / current_column
        scaled = thermoprops.copy(deep=True)
        scaled["x_H2O"] = thermoprops["x_H2O"] * scale
        return scaled

    @staticmethod
    def set_uniform_mixing_ratio(thermoprops, variable: str, target_ppm: float):
        """Overwrites a well-mixed gas's mixing ratio profile with a
        uniform value, e.g. setting a target CO2/CH4 concentration in ppm.

        Args:
            thermoprops: Thermophysical profile to modify.
            variable: Mixing-ratio variable name, e.g. "x_CO2".
            target_ppm: Target uniform mixing ratio (ppm).

        Returns:
            xr.Dataset: Thermophysical profile with the uniform ratio.
        """
        scaled = thermoprops.copy(deep=True)
        scaled[variable] = thermoprops[variable] * 0.0 + (target_ppm * 1e-6)
        return scaled

    @staticmethod
    def scale_surface_mixing_ratio(thermoprops, variable: str, target_ppm: float):
        """Scales a mixing ratio profile so its surface value matches
        target_ppm, preserving the default profile's vertical shape.

        Suits gases that are well mixed only through the troposphere, where
        set_uniform_mixing_ratio would be wrong (e.g. methane).

        Args:
            thermoprops: Thermophysical profile to scale.
            variable: Mixing-ratio variable name, e.g. "x_CH4".
            target_ppm: Target surface mixing ratio (ppm).

        Returns:
            xr.Dataset: Thermophysical profile with the rescaled gas.
        """
        surface = float(thermoprops[variable].isel(z=0))

        if surface <= 0:
            Logger.warning(
                f"Default {variable} surface value is non-positive, cannot scale"
            )
            return thermoprops

        scaled = thermoprops.copy(deep=True)
        scaled[variable] = thermoprops[variable] * (target_ppm * 1e-6 / surface)
        return scaled

    @staticmethod
    def angstrom_particle_properties(base, angstrom: float, w_ref):
        """Returns a ParticleProperties with the base dataset's single-
        scattering albedo/phase function kept as-is, but its extinction
        replaced by an Angstrom power-law spectral shape. ParticleLayer
        normalizes extinction by its value at w_ref (see
        ParticleLayer.eval_sigma_t), so only the relative shape across
        wavelength matters here.

        Args:
            base: Base aerosol particle-properties dataset.
            angstrom: Angstrom exponent.
            w_ref: Reference wavelength (quantity).

        Returns:
            ParticleProperties: Copy of base with reshaped extinction.
        """
        w = to_quantity(base.data["w"])
        ext_shape = (w / w_ref).m_as("dimensionless") ** (-float(angstrom))

        data = base.data.copy(deep=True)
        data["ext"] = xr.DataArray(
            ext_shape, dims=data["ext"].dims, attrs=data["ext"].attrs
        )
        return ParticleProperties(data)

    @classmethod
    def srf_grid(cls, center: float, fwhm: float):
        """Samples a channel's Gaussian spectral response function:
        srf_n_inner points spanning +/- srf_sigma_inner standard deviations,
        bracketed by an explicit zero at +/- srf_sigma_outer sigma (BandSRF
        requires the response to reach zero at its support boundary).

        Args:
            center: Channel center wavelength (nm).
            fwhm: Channel FWHM (nm).

        Returns:
            tuple: (sub-channel wavelengths nm ascending, Gaussian response
            at each wavelength, zero at the two endpoints).
        """
        sigma = fwhm / 2.3548200450309493  # FWHM -> Gaussian standard deviation
        w_inner = np.linspace(
            center - cls.srf_sigma_inner * sigma,
            center + cls.srf_sigma_inner * sigma,
            cls.srf_n_inner,
        )
        w = np.concatenate(
            [
                [center - cls.srf_sigma_outer * sigma],
                w_inner,
                [center + cls.srf_sigma_outer * sigma],
            ]
        )
        response = np.exp(-0.5 * ((w - center) / sigma) ** 2)
        response[0] = response[-1] = 0.0
        return w, response

    @classmethod
    def srf_weighted_average(
        cls, w: np.array, response: np.array, values: np.array
    ) -> float:
        """Weighted average of ``values`` (sampled at ``w``) using
        ``response`` as the weight, integrated by the trapezoidal rule --
        matching how eradiate's own BandSRF pipeline step combines
        sub-wavelength samples for a Monte Carlo measure, so the analytic
        terms stay consistent with the MC ones.

        Args:
            w: Sub-channel wavelengths (nm).
            response: SRF weight at each wavelength in w.
            values: Values to average, same length as w.

        Returns:
            float: SRF-weighted average of values.
        """
        return cls.trapz(values * response, w) / cls.trapz(response, w)

    @staticmethod
    def wavelength_transmittances(
        atmosphere, exp, w: float, coszen: float, cosvza: float
    ):
        """Direct downward/upward transmittance (Beer-Lambert) at a single
        wavelength, along with an analytical CKD correlation correction scalar.

        In Correlated K-Distribution (CKD) mode, a spectral bin is represented
        by an ensemble of quadrature points (g-points). Because the downward
        and upward direct paths through the atmosphere share the exact same gas
        absorption realization at a given g-point, they are strongly positively
        correlated. Consequently, the product of their averages underestimates
        the true average of their product:

            <T_down> * <T_up>  <  <T_down * T_up>

        Multiplying the band-averaged transmittances directly leads to severe
        underestimation of the ground-reflected radiance inside deep absorption
        bands, causing inverted artifacts, particularly in the diffuse transmittance
        retrievals.

        To fix this, we compute an exact analytical correlation scalar, C_ckd,
        directly from the layer optical depths:

            C_ckd = <T_down * T_up> / (<T_down> * <T_up>)

        This scalar is applied to the BOA radiance product in `decompose_channel`.
        In monochromatic mode (line-by-line),
        there are no g-points to correlate, and C_ckd=1. This is probably the
        'right' answer for fine-spectral resolution runs, but the C_ckd solution
        allows fore reasonable testing with the band models.

        Args:
            atmosphere: Pre-init Atmosphere, as returned by build_atmosphere.
                Must NOT be exp.atmosphere (see init mutation warnings).
            exp: Initialized AtmosphereExperiment (only its CKD quadrature
                config is used here, which isn't affected by init).
            w: Wavelength (nm).
            coszen: Cosine of the solar zenith angle.
            cosvza: Cosine of the view zenith angle.

        Returns:
            tuple: (transm_down_dir, transm_up_dir, c_ckd) at w.
        """
        zgrid = atmosphere.geometry.zgrid
        layer_height = zgrid.layer_height.m_as("km")

        def tau(si) -> float:
            sigma_t = atmosphere.eval_sigma_t(si, zgrid).reshape(zgrid.layers.shape)
            return float(np.sum(sigma_t.m_as("1/km") * layer_height))

        mu0 = max(coszen, 1e-6)
        muv = max(cosvza, 1e-6)

        if eradiate.mode().is_ckd:
            quad = exp.ckd_quad_config.get_quad(wcenter=w * ureg.nm)
            nodes = quad.eval_nodes([0, 1])
            weights = quad.weights  # native [-1, 1] convention, sums to 2

            tau_g = np.array(
                [tau(CKDSpectralIndex(w=w * ureg.nm, g=float(g))) for g in nodes]
            )

            t_down_g = np.exp(-tau_g / mu0)
            t_up_g = np.exp(-tau_g / muv)

            t_down = 0.5 * np.dot(weights, t_down_g)
            t_up = 0.5 * np.dot(weights, t_up_g)

            # Exact correlated product across g-points
            t_both_correlated = 0.5 * np.dot(weights, t_down_g * t_up_g)
            uncorrelated = t_down * t_up
            c_ckd = t_both_correlated / uncorrelated if uncorrelated > 1e-12 else 1.0
        else:
            si = SpectralIndex.new(w=w * ureg.nm)
            tau_dir = tau(si)
            t_down = np.exp(-tau_dir / mu0)
            t_up = np.exp(-tau_dir / muv)
            c_ckd = 1.0

        return t_down, t_up, c_ckd

    @staticmethod
    def set_reflectance(exp, albedo: float):
        """Mutates the surface's Lambertian reflectance in place so
        repeated eradiate.run() calls reuse the already-initialized scene
        instead of rebuilding it from scratch.

        Args:
            exp: Initialized AtmosphereExperiment.
            albedo: Lambertian reflectance to set.
        """
        exp.surface.bsdf.reflectance.value = albedo * ureg.dimensionless
