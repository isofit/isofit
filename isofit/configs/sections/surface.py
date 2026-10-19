"""
Surface configuration for ISOFIT.

This module defines surface model configuration including surface type,
priors, multi-surface mode, terrain handling, and surface state vector
parameters for temperature and glint effects.
"""

from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FilePath, NonNegativeInt
from scipy.io import loadmat

from isofit.configs.sections.statevector import (
    StateVectorConfig,
    StateVectorElementConfig,
)

SurfaceCategory = Literal[
    "surface",
    "multicomponent_surface",
    "glint_model_surface",
    "thermal_surface",
    "lut_surface",
]


def statevector_mismatches(
    file: Path, configured: Dict[str, StateVectorElementConfig]
) -> list[str]:
    """
    Compare configured statevector elements with those stored in a surface file.

    Parameters
    ----------
    file : Path
        Surface model file (.mat format). If it doesn't exist yet (e.g. it is
        built later by ``isofit surface_model``), nothing is compared.
    configured : dict[str, StateVectorElementConfig]
        Configured statevector elements by name. Elements stored in the file
        but not configured here are skipped, as are fields missing from the
        file. Values unset (NaN) in both are considered equal.

    Returns
    -------
    list[str]
        One ``"NAME: field (file)"`` entry per mismatching value.
    """
    if not file.exists():
        return []

    mismatch = []
    model_dict = loadmat(file)
    for i, name in enumerate(model_dict.get("statevec_names", [])):
        name = name.strip()
        if name not in configured:
            continue
        for key in StateVectorElementConfig.model_fields:
            if key not in model_dict:
                continue
            values = model_dict[key]
            if key == "bounds":
                stored = np.reshape(values, (-1, 2))[i]
            else:
                stored = np.ravel(values)[i]
            value = getattr(configured[name], key)
            if not np.allclose(value, stored, equal_nan=True):
                mismatch.append(f"{name}: {key} ({file})")

    return mismatch


class SurfaceStateVectorConfig(StateVectorConfig):
    """
    Surface state vector configuration.

    Defines surface parameters that can be optimized during inversion,
    including surface temperature for thermal retrievals and glint parameters
    for water surfaces.

    Attributes
    ----------
    SURF_TEMP_K : StateVectorElementConfig, optional
        Surface temperature in Kelvin. Used for thermal emission modeling
        in thermal infrared (TIR) retrievals.
    SKY_GLINT : StateVectorElementConfig, optional
        Sky glint parameter for modeling specular reflection of sky radiance
        from water surfaces.
    SUN_GLINT : StateVectorElementConfig, optional
        Sun glint parameter for modeling specular reflection of direct solar
        radiance from water surfaces.

    Examples
    --------
    >>> from isofit.configs.sections.surface import SurfaceStateVectorConfig
    >>> from isofit.configs.sections.statevector import StateVectorElementConfig
    >>> ssv = SurfaceStateVectorConfig()
    >>> ssv.SURF_TEMP_K = StateVectorElementConfig(
    ...     bounds=[250.0, 350.0],
    ...     scale=50.0,
    ...     init=300.0
    ... )

    Notes
    -----
    Glint parameters are primarily used for aquatic remote sensing applications
    where specular reflection from the water surface is significant.
    """

    SURF_TEMP_K: Optional[StateVectorElementConfig] = Field(
        description="Surface temperature in Kelvin"
    )
    SKY_GLINT: Optional[StateVectorElementConfig] = Field(
        description="Sky glint parameter"
    )
    SUN_GLINT: Optional[StateVectorElementConfig] = Field(
        description="Sun glint parameter"
    )


class SubSurfaceConfig(BaseModel):
    """
    Configuration for one surface class in multi-surface mode.

    Each entry of :attr:`SurfaceConfig.Surfaces` maps a classification value
    to the surface model used for those pixels. When a class is processed,
    :meth:`SurfaceConfig.update_from_subconfig` copies these settings onto the
    active surface config.

    Attributes
    ----------
    surface_category : {"surface", "multicomponent_surface", "glint_model_surface", "thermal_surface", "lut_surface"}
        Type of surface model for this class.
    surface_file : Path
        Path to the surface model file (.mat format) for this class. Need not
        exist at load time, as it may be built later (e.g. by
        ``isofit surface_model``).
    surface_int : int
        Value of this class in the surface classification file.
    statevector : dict[str, StateVectorElementConfig]
        Non-reflectance statevector elements expected in ``surface_file``.
        Only used by :meth:`SurfaceConfig.get_config_issues`; runs use
        :attr:`SurfaceConfig.statevector`.
    rt_statevector_elements : dict[str, StateVectorElementConfig]
        Experimental: statevector elements added to the atmosphere for this
        class.

    Notes
    -----
    Any other key (e.g. ``terrain_style``, ``max_slope``) is kept as-is and
    overwrites the matching :class:`SurfaceConfig` field for this class.
    """

    model_config = ConfigDict(extra="allow")

    surface_category: SurfaceCategory = Field(
        description="Type of surface model for this class"
    )
    surface_file: Path = Field(description="Path to surface model file (.mat format)")
    surface_int: NonNegativeInt = Field(
        description="Value of this class in the surface classification file"
    )
    statevector: Dict[str, StateVectorElementConfig] = Field(
        default_factory=dict,
        description="Non-reflectance statevector elements expected in surface_file",
    )
    rt_statevector_elements: Dict[str, StateVectorElementConfig] = Field(
        default_factory=dict,
        description="Experimental: atmosphere statevector elements for this class",
    )

    def statevector_mismatches(
        self, base: Dict[str, StateVectorElementConfig]
    ) -> list[str]:
        """
        Compare the statevector configured for this class with its surface file.

        Parameters
        ----------
        base : dict[str, StateVectorElementConfig]
            Statevector elements configured for the parent surface. This
            class's own ``statevector`` overrides them by name.

        Returns
        -------
        list[str]
            Mismatches found, see :func:`statevector_mismatches`.
        """
        return statevector_mismatches(self.surface_file, {**base, **self.statevector})


class SurfaceConfig(BaseModel):
    """
    Surface configuration.

    Configures surface reflectance model including surface type, priors,
    multi-surface mode for classification-based processing, terrain handling,
    and surface state vector parameters.

    Attributes
    ----------
    multi_surface_flag : bool
        Enable multi-surface mode where different surface models are used
        based on classification map. Default is False.
    use_background_rfl : bool
        Use background reflectance for spatial inference. Default is False.
    surface_file : FilePath, optional
        Path to surface model file in .mat format containing reflectance
        priors, basis vectors, and covariance information.
    surface_category : {"surface", "multicomponent_surface", "glint_model_surface", "thermal_surface", "lut_surface"}, optional
        Type of surface model to use. Determines which surface class is
        instantiated.
    surface_class_file : FilePath, optional
        Path to surface classification file (ENVI format) for multi-surface mode.
    base_surface_class_file : FilePath, optional
        Path to base surface classification file for hierarchical classification.
    Surfaces : dict[str, SubSurfaceConfig]
        Multi-surface configuration dictionary mapping surface class names to
        their configurations. Each entry must have 'surface_category',
        'surface_file', and 'surface_int' (integer mapping).
    wavelength_file : FilePath, optional
        Path to wavelength file for surface model spectral sampling.
    select_on_init : bool
        Force use of initialization state and never change surface model
        selection. Default is True.
    selection_metric : {"Euclidean", "SGA", "SpecAngle"}
        Metric for selecting the surface prior component: Euclidean distance,
        "SGA" (Spectral Gradient Angle), or "SpecAngle" (spectral angle).
        Default is "Euclidean".
    statevector : SurfaceStateVectorConfig
        Surface state vector configuration for temperature and glint parameters.
    emissivity_for_surface_T_init : float
        Initial emissivity value for surface temperature estimation. Value
        of 0.98 recommended by Glynn Hulley for most surfaces. Default is 0.98.
    terrain_style : {"flat", "dem", "solved"}
        Style of terrain to use in forward model. "flat" assumes level surface,
        "dem" uses digital elevation model, "solved" optimizes terrain parameters.
        Default is "flat".
    max_slope : float
        Maximum slope value in degrees for LUT calculations. Only relevant
        for terrain_style="dem". Default is 90.0.
    refractive_index_path : str
        Path to refractive index data for water surface modeling. Default is "".

    Examples
    --------
    >>> from isofit.configs.sections.surface import SurfaceConfig
    >>> surf = SurfaceConfig(
    ...     surface_file="surface_priors.mat",
    ...     surface_category="multicomponent_surface",
    ...     terrain_style="flat"
    ... )

    Multi-surface configuration example:

    >>> multi_surf = SurfaceConfig(
    ...     multi_surface_flag=True,
    ...     surface_class_file="classification.img",
    ...     Surfaces={
    ...         "vegetation": {
    ...             "surface_category": "multicomponent_surface",
    ...             "surface_file": "veg_prior.mat",
    ...             "surface_int": 1
    ...         },
    ...         "soil": {
    ...             "surface_category": "multicomponent_surface",
    ...             "surface_file": "soil_prior.mat",
    ...             "surface_int": 2
    ...         }
    ...     }
    ... )

    Notes
    -----
    The surface model is critical for accurate atmospheric correction. Poor
    surface priors or incorrect surface type selection can degrade retrieval
    quality.

    Multi-surface mode allows using different priors for different surface
    types (vegetation, soil, water, etc.) based on a classification map,
    improving retrieval accuracy in heterogeneous scenes.

    See Also
    --------
    SurfaceStateVectorConfig : Surface parameters for optimization
    SubSurfaceConfig : Per-class configuration in multi-surface mode
    """

    multi_surface_flag: bool = Field(description="Enable multi-surface mode")

    use_background_rfl: bool = Field(description="Use background reflectance")

    surface_file: Optional[FilePath] = Field(
        description="Path to surface model file (.mat format)"
    )

    surface_category: Optional[SurfaceCategory] = Field(
        description="Type of surface model to use"
    )

    surface_class_file: Optional[FilePath] = Field(
        description="Path to surface classification file"
    )

    base_surface_class_file: Optional[FilePath] = Field(
        description="Path to base surface classification file"
    )

    Surfaces: Dict[str, SubSurfaceConfig] = Field(
        description="Multi-surface configuration dictionary"
    )

    wavelength_file: Optional[FilePath] = Field(description="Path to wavelength file")

    select_on_init: bool = Field(
        description="Force use of initialization state and never change"
    )

    selection_metric: Literal["Euclidean", "SGA", "SpecAngle"] = Field(
        description="Metric for surface selection"
    )

    statevector: SurfaceStateVectorConfig = Field(
        description="Surface state vector configuration",
    )

    emissivity_for_surface_T_init: float = Field(
        description="Initial emissivity value for surface temperature (recommended by Glynn Hulley)",
    )

    terrain_style: Literal["flat", "dem", "solved"] = Field(
        description="Style of terrain to use in the forward model"
    )

    max_slope: float = Field(
        description="Max slope value for LUT calculations (only relevant for 'dem' terrain_style)",
    )

    refractive_index_path: str = Field(description="Path to refractive index data")

    def update_from_subconfig(self, subconfig: SubSurfaceConfig):
        """
        Overwrite fields from a per-class multi-surface sub-config.

        In multi-surface mode each surface class carries its own settings; this
        copies them onto the active ``SurfaceConfig``. Structural fields that
        describe the multi-surface machinery itself (or the state vector) are
        never overwritten, and unknown keys are ignored.

        Parameters
        ----------
        subconfig : SubSurfaceConfig
            Settings for the selected surface class.
        """
        keys_to_ignore = {
            "multi_surface_flag",
            "Surfaces",
            "surface_class_file",
            "base_surface_class_file",
            "statevector",
        }

        # Iterating a model yields its declared fields followed by its extras
        for key, value in subconfig:
            if key in keys_to_ignore:
                continue
            if key in type(self).model_fields:
                setattr(self, key, value)

    def get_config_issues(self) -> tuple[list[str], list[str]]:
        """
        Check the config for completeness and against the surface model files.

        These checks run from :meth:`isofit.configs.Config.get_config_errors`
        rather than at validation time: base configs (e.g. ``default.yml``)
        intentionally leave the surface file unset, and the surface model files
        may not have been built yet when the config is loaded.

        Returns
        -------
        tuple[list[str], list[str]]
            Errors and warnings found.

        Notes
        -----
        Checks performed:
        - Outside multi-surface mode, surface_file and surface_category must be set
        - Statevector elements (e.g. SURF_TEMP_K) configured for each surface
          file should match those stored in it; a mismatch is a warning, as the
          configured values are used. The base statevector applies to every
          file, and a sub-surface's own statevector overrides it for that
          sub-surface's file. Files that don't exist yet are skipped
        """
        errors, warnings = [], []

        if not self.Surfaces:
            if self.surface_file is None:
                errors.append(
                    "surface_file not specified. All current supported surface models require surface_file to be provided"
                )
            if self.surface_category is None:
                errors.append("surface->surface_category or Surfaces must be specified")

        elements, names = self.statevector.get_elements()
        statevector = dict(zip(names, elements))

        mismatch = []
        if self.surface_file is not None:
            mismatch += statevector_mismatches(self.surface_file, statevector)
        for sub_surface in self.Surfaces.values():
            mismatch += sub_surface.statevector_mismatches(statevector)

        if mismatch:
            warnings.append(
                "Configured non-reflectance surface statevector elements do not "
                "match .mat file.\nRun will use configured values in the config."
                "\nMismatching values:\n" + "\n".join(mismatch)
            )

        return errors, warnings
