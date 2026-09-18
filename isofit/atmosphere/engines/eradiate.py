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

import eradiate
import joseki
import numpy as np
import xarray as xr
from eradiate.attrs import AUTO
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

from isofit.atmosphere.atmosphere import BaseAtmosphere
from isofit.core import units
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

    def __init__(self, full_config, wl=[], fwhm=[], **kwargs):
        """Inits the engine and loads the base atmosphere/aerosol datasets.

        Args:
            full_config: Full ISOFIT config.
            wl: Wavelength override (nm).
            fwhm: FWHM override (nm).
            **kwargs: Forwarded to BaseAtmosphere.__init__.
        """
        self.config = full_config.forward_model.atmosphere

        self.eradiate_absorption_database = (
            self.config.eradiate_absorption_database or "komodo"
        )
        self.eradiate_mode = self.infer_spectral_mode(self.eradiate_absorption_database)
        self.eradiate_particle_properties = (
            self.config.eradiate_particle_properties or "govaerts_2021-continental"
        )
        self.eradiate_spp = self.config.eradiate_spp or 1000

        self.engine_base_dir = self.config.engine_base_dir
        self.sim_path = self.config.sim_path

        eradiate.set_mode(self.eradiate_mode)

        # Loaded once; per-point gas/aerosol scaling starts from these and
        # never mutates them in place.
        self._base_thermoprops = self.default_thermoprops()
        self._base_particle_properties = self.load_particle_properties(
            self.eradiate_particle_properties
        )

        super().__init__(full_config, wl=wl, fwhm=fwhm, **kwargs)

    def preSim(self):
        """Records engine configuration on the LUT for provenance."""
        self.lut.setAttr("eradiate_mode", self.eradiate_mode)
        self.lut.setAttr(
            "eradiate_absorption_database", self.eradiate_absorption_database
        )
        self.lut.setAttr(
            "eradiate_particle_properties", self.eradiate_particle_properties
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

    # Output terms two_albedo_method computes or passes through per
    # sub-wavelength, SRF-averaged in readSim to get each channel's final
    # LUT value.
    output_terms = (
        "rhoatm",
        "sphalb",
        "transm_down_dir",
        "transm_down_dif",
        "transm_up_dir",
        "transm_up_dif",
        "solar_irr",
        "thermal_upwelling",
        "thermal_downwelling",
    )

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
                exp, spp=self.eradiate_spp, seed_state=SeedState(seed=seed)
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

        sigma = fwhm / 2.3548200450309493
        response = np.exp(-0.5 * ((w_sub - center) / sigma) ** 2)
        return {
            key: self.srf_weighted_average(w_sub, response, sub_data[key])
            for key in self.output_terms
        }

    def resolve_point(self, point: np.array) -> dict:
        """Merges fixed engine-config geometry with this LUT point's
        values, mapping ISOFIT's geometry-key conventions onto eradiate's.

        Args:
            point: LUT point to process.

        Returns:
            dict: Geometry and state-vector values keyed by name.
        """
        vals = {
            "solzen": self.config.solzen,
            "solaz": self.config.solaz,
            "viewzen": self.config.viewzen,
            "viewaz": self.config.viewaz,
            "elev": abs(max(self.config.elev or 0.0, 0.0)),
            # None means "satellite/TOA sensor" (infinite distance); only a
            # value strictly below the atmosphere top switches to a finite,
            # airborne ray_offset (see build_experiment).
            "alt": self.config.alt,
        }

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
            thermoprops = self.set_uniform_mixing_ratio(
                thermoprops, "x_CH4", vals["CH4"]
            )

        molecular = MolecularAtmosphere(
            thermoprops=thermoprops,
            absorption_data=self.eradiate_absorption_database,
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
    def default_thermoprops():
        """Loads the default AFGL US-standard thermophysical profile
        (0-120 km), matching MolecularAtmosphere's own default.

        Returns:
            xr.Dataset: Thermophysical profile.
        """
        return joseki.make(
            identifier="afgl_1986-us_standard",
            z=np.linspace(0.0, 120.0, 121) * ureg.km,
        )

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
