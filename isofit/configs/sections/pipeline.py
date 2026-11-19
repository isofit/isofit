"""
Retrieval pipeline configuration.

This module defines the Pydantic models for the sections only used by
``isofit retrieval``: logging, the LUT grid heuristics, and one section per
pipeline stage (see :mod:`isofit.retrieval.pipeline`). Every stage section
carries an ``enabled`` flag, which both gates the stage in
``isofit retrieval run`` and tells the downstream stages whether its products
are part of the retrieval (e.g. ``presolve.enabled`` narrows the H2O LUT in
``configure``).
"""

import logging
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    FilePath,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from isofit.configs.base import SectionModel


class LogConfig(SectionModel):
    file: Optional[Path] = Field(description="")

    level: Optional[str] = Field(description="")

    append: Optional[bool] = Field(description="")

    config: Optional[Path] = Field(description="")

    resources: Union[bool, Path, None] = Field(
        description=(
            "Track system resources to a JSON lines file. Either a path to the "
            "file or True to write it adjacent to ``file`` as ``.resources.jsonl``"
        )
    )

    @field_validator("level")
    @classmethod
    def check_level(cls, value):
        value = value.upper()
        levels = logging.getLevelNamesMapping()

        if value not in levels:
            raise ValueError(f"Must be one of {levels}")

        return value

    @model_validator(mode="after")
    def resolve_resources(self) -> "LogConfig":
        """Resolve ``resources`` to a path, deriving it from ``file`` when True."""
        if self.resources is True:
            if self.file is None:
                raise ValueError("log.resources=True requires log.file to be set")
            self.resources = self.file.with_suffix(".resources.jsonl")
        elif self.resources is False:
            self.resources = None

        return self


class Stage(SectionModel):
    """
    Base for the retrieval pipeline stage sections.
    """

    enabled: bool = Field(
        description=(
            "Include this stage in the retrieval: it is executed by "
            "``isofit retrieval run``, and its products are used by the "
            "downstream stages"
        )
    )


class LutDimension(BaseModel):
    """
    Grid heuristic for one LUT dimension.

    A grid spans the scene's range of the dimension (or :attr:`range`, when
    set) at roughly :attr:`spacing`. The dimension is dropped from the LUT, and
    interpolated to the scene mean instead, when the grid would collapse to one
    point or its points would be closer than :attr:`spacing_min`.
    """

    spacing: NonNegativeFloat = Field(
        description="Target spacing between grid points; 0 disables the grid"
    )

    spacing_min: NonNegativeFloat = Field(
        description="Minimum spacing for a grid to be used"
    )

    range: Optional[tuple[float, float]] = Field(
        description="Fixed [min, max] of the grid; derived from the scene when unset"
    )


class LutConfig(Stage):
    """
    Configuration for the retrieval ``lut`` stage, and the LUT grid heuristics
    used by the ``scene``, ``presolve``, and ``configure`` stages.

    The grid heuristics replace apply_oe's ``LUTConfig`` and its
    ``--lut_config_file``.
    """

    section = "lut"

    exit: bool = Field(
        description=(
            "Stop ``isofit retrieval run`` once the LUT is built, before the "
            "inversion. Resume with ``lut.exit=False``, or ``lut.enabled=False``"
        )
    )

    prebuilt: bool = Field(
        description=(
            "``forward_model.atmosphere.lut_path`` is a prebuilt LUT to subset "
            "to the scene rather than a LUT to build (apply_oe: --prebuilt_lut)"
        )
    )

    no_min_spacing: bool = Field(
        description=(
            "Keep a dimension with at least 2 points even when its spacing is "
            "below ``spacing_min`` (apply_oe: --no_min_lut_spacing)"
        )
    )

    elevation: LutDimension = Field(description="Surface elevation, km")
    observer_zenith: LutDimension = Field(description="To-sensor zenith, degrees")
    solar_zenith: LutDimension = Field(description="To-sun zenith, degrees")
    relative_azimuth: LutDimension = Field(description="Relative azimuth, degrees")

    h2o: LutDimension = Field(
        description=(
            "Water vapor, g/cm2. An unset range spans 0.2 to the maximum water "
            "of ``scene.atmosphere_type`` at the scene elevation"
        )
    )

    aerosol: LutDimension = Field(
        description=(
            "Aerosol optical thickness. Gridded as AOT550 for emulators and "
            "AERFRAC_2 for MODTRAN. An unset range spans the minimum AOT of "
            "``scene.atmosphere_type`` to 1"
        )
    )

    co2: LutDimension = Field(
        description="CO2, ppm. Only used when CO2 is in the atmosphere statevector"
    )


class SetupStage(Stage):
    """
    Configuration for the retrieval ``setup`` stage.

    The radiance, obs, and loc the later stages read are :attr:`rdn`,
    :attr:`obs`, and :attr:`loc`. Each defaults to its ``input`` file, and when
    set to another path the input is copied there (see the ``copy_inputs``
    mixin of the ``tc`` base config).

    The ``*_file`` fields are the source files that setup stages into
    ``${output.dir}``. Each is copied to the path set by its matching
    ``forward_model`` field, which is what the written ISOFIT configs reference:

    =============================  ===================================================================
    Source                         Destination
    =============================  ===================================================================
    channelized_uncertainty_file   forward_model.instrument.unknowns.channelized_radiometric_uncertainty_file
    model_discrepancy_file         forward_model.model_discrepancy_file
    eof_file                       forward_model.instrument.eof_path
    wavelength_file                forward_model.instrument.wavelength_file
    =============================  ===================================================================

    Examples
    --------
    >>> from isofit.configs.sections.pipeline import SetupStage
    >>> SetupStage(base="pipeline").rdn is None
    True
    """

    section = "setup"

    rdn: Optional[Path] = Field(description="Working radiance, staged from the input")
    obs: Optional[Path] = Field(description="Working obs, staged from the input")
    loc: Optional[Path] = Field(description="Working loc, staged from the input")

    wavelength_file: Optional[FilePath] = Field(
        description=(
            "Wavelength file (channel, wavelength, FWHM columns) to use instead of "
            "the radiance header. Must have one row per radiance channel"
        )
    )

    channelized_uncertainty_file: Optional[FilePath] = Field(
        description="Per-channel radiometric uncertainty file to stage"
    )

    model_discrepancy_file: Optional[FilePath] = Field(
        description="Model discrepancy covariance (.mat) file to stage"
    )

    eof_file: Optional[FilePath] = Field(
        description="Empirical Orthogonal Function (EOF) file to stage"
    )


class PreprocessStage(Stage):
    """
    Configuration for the retrieval ``preprocess`` stage.
    """

    section = "preprocess"

    interpolate_radiance: bool = Field(
        description=(
            "Spectrally interpolate across partially no-data or NaN channels "
            "of each radiance spectrum, from ``setup.rdn`` to :attr:`rdn` "
            "(apply_oe: --interpolate_bad_rdn)"
        )
    )

    rdn: Optional[Path] = Field(
        description=(
            "Radiance read by the later stages: the interpolated radiance, "
            "otherwise ``setup.rdn``"
        )
    )

    skyview: Optional[Literal["slope", "horizon"]] = Field(
        description=(
            "Compute the sky view factor to :attr:`svf`, either approximated from "
            "the obs slope as cos^2(slope/2) or from the loc elevation horizons "
            "(apply_oe: --skyview_factor)"
        )
    )

    svf: Optional[Path] = Field(
        description=(
            "Sky view factor read by the later stages: the computed factor, "
            "otherwise ``input.skyview_factor_file``. None for no sky view factor"
        )
    )


class SceneStage(Stage):
    """
    Configuration for the retrieval ``scene`` stage.
    """

    section = "scene"

    atmosphere_type: Literal[
        "ATM_MIDLAT_SUMMER",
        "ATM_MIDLAT_WINTER",
        "ATM_SUBARC_SUMMER",
        "ATM_SUBARC_WINTER",
        "ATM_TROPICAL",
        "ATM_US_STANDARD_1976",
    ] = Field(
        description=(
            "Atmospheric profile for the MODTRAN template, which also bounds "
            "the H2O and aerosol LUT ranges"
        )
    )

    trim_lines: NonNegativeInt = Field(
        description=(
            "Lines at the start and end of the obs/loc to exclude from the "
            "scene statistics"
        )
    )

    max_flight_duration: PositiveFloat = Field(
        description="Maximum acquisition length, hours, to detect a UTC day rollover"
    )

    file: Optional[Path] = Field(
        description="Scene statistics (JSON) read by the later stages"
    )


class SurfaceStage(Stage):
    """
    Configuration for the retrieval ``surface`` stage.

    Not to be confused with ``forward_model.surface``, which is the surface
    model the written ISOFIT configs use: this stage builds or checks the
    model at :attr:`file` and stages it to ``forward_model.surface.surface_file``.
    """

    section = "surface"

    file: Optional[FilePath] = Field(
        description=(
            "Surface model (.mat) or surface model config (.json) to build one "
            "from (apply_oe: --surface_path)"
        )
    )

    classify: bool = Field(
        description=(
            "Classify the scene into the surface categories of a .json surface "
            "config, writing :attr:`classes` (apply_oe: --classify_multisurface)"
        )
    )

    classes: Optional[Path] = Field(
        description=(
            "Surface classification (ENVI) for a multisurface retrieval, either "
            "given or written by :attr:`classify`. None for a single surface "
            "(apply_oe: --surface_class_file)"
        )
    )


class SegmentStage(Stage):
    """
    Configuration for the retrieval ``segment`` stage.
    """

    section = "segment"

    size: PositiveInt = Field(
        description="Target superpixel size, pixels (apply_oe: --segmentation_size)"
    )

    npca: PositiveInt = Field(
        description="Principal components used for the SLIC segmentation"
    )

    labels: Optional[Path] = Field(description="Superpixel labels of every pixel")

    # Per-superpixel means of the working files, see stages.segment.Sources
    rdn: Optional[Path] = Field(description="Superpixel radiance")
    obs: Optional[Path] = Field(description="Superpixel obs")
    loc: Optional[Path] = Field(description="Superpixel loc")
    svf: Optional[Path] = Field(
        description="Superpixel sky view factor, when ``preprocess.svf`` is set"
    )
    surface_class: Optional[Path] = Field(
        description="Superpixel surface classes, when ``surface.classes`` is set"
    )


class PresolveStage(Stage):
    """
    Configuration for the retrieval ``presolve`` stage.
    """

    section = "presolve"

    grid_points: int = Field(
        ge=2, description="Number of points in the presolve 1-D H2O LUT"
    )

    layers: list[str] = Field(
        description=(
            "Config references (e.g. ``tc.yml`` run sections) layered onto this "
            "config to write the presolve ISOFIT config. Only each layer's own "
            "keys are applied, see :mod:`isofit.retrieval.utils.solve`"
        )
    )

    file: Optional[Path] = Field(description="Presolve ISOFIT config, written")
    template: Optional[Path] = Field(
        description="Presolve MODTRAN template, written by the scene stage"
    )
    state: Optional[Path] = Field(
        description="Presolve state, read by the configure and background stages"
    )


class BackgroundStage(Stage):
    """
    Configuration for the retrieval ``background`` stage.
    """

    section = "background"

    rfl: Optional[Path] = Field(description="Background reflectance of every pixel")
    subs: Optional[Path] = Field(description="Superpixel background reflectance")


class ConfigureStage(Stage):
    """
    Configuration for the retrieval ``configure`` stage.
    """

    section = "configure"

    aerosol_climatology_file: Optional[FilePath] = Field(
        description=(
            "Aerosol climatology cases (JSON) matched against the scene location "
            "and time (apply_oe: --aerosol_climatology_path)"
        )
    )

    layers: list[str] = Field(
        description=(
            "Config references (e.g. ``tc.yml`` run sections) layered onto this "
            "config to write the main ISOFIT config. Only each layer's own "
            "keys are applied, see :mod:`isofit.retrieval.utils.solve`"
        )
    )

    file: Optional[Path] = Field(
        description="Main ISOFIT config, read by the lut, invert, and interpolate stages"
    )


class InvertStage(Stage):
    """
    Configuration for the retrieval ``invert`` stage.
    """

    section = "invert"


class InterpolateStage(Stage):
    """
    Configuration for the retrieval ``interpolate`` stage.

    Extends the superpixel products of the main inversion to every pixel,
    writing the ``output`` products of this config.
    """

    section = "interpolate"

    method: Literal["empirical", "analytical"] = Field(
        description=(
            "Empirical line or analytical line "
            "(apply_oe: --empirical_line / --analytical_line)"
        )
    )

    num_neighbors: list[PositiveInt] = Field(
        description=(
            "Neighbors used for interpolation, derived from ``segment.size`` "
            "when empty. Multiple values are only valid for the analytical line"
        )
    )

    atm_sigma: list[float] = Field(
        description=(
            "Smoothing sigma per atmospheric parameter, or one broadcast to all. "
            "Analytical line only"
        )
    )


class CleanupStage(Stage):
    """
    Configuration for the retrieval ``cleanup`` stage.
    """

    section = "cleanup"

    remove: list[str] = Field(description="Glob patterns of the files to remove")
