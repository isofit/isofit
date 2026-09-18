"""
Surface configuration for ISOFIT.

This module defines surface model configuration including surface type,
priors, multi-surface mode, terrain handling, and surface state vector
parameters for temperature and glint effects.
"""

from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from isofit.configs.sections.statevector import StateVector, StateVectorElement
from isofit.configs.utils.validators import PathExists


class SurfaceStateVector(StateVector):
    """
    Surface state vector configuration.

    Defines surface parameters that can be optimized during inversion,
    including surface temperature for thermal retrievals and glint parameters
    for water surfaces.

    Attributes
    ----------
    SURF_TEMP_K : StateVectorElement, optional
        Surface temperature in Kelvin. Used for thermal emission modeling
        in thermal infrared (TIR) retrievals.
    SKY_GLINT : StateVectorElement, optional
        Sky glint parameter for modeling specular reflection of sky radiance
        from water surfaces.
    SUN_GLINT : StateVectorElement, optional
        Sun glint parameter for modeling specular reflection of direct solar
        radiance from water surfaces.

    Examples
    --------
    >>> from isofit.configs.sections.surface import SurfaceStateVector
    >>> from isofit.configs.sections.statevector import StateVectorElement
    >>> ssv = SurfaceStateVector()
    >>> ssv.SURF_TEMP_K = StateVectorElement(
    ...     bounds=[250.0, 350.0],
    ...     scale=50.0,
    ...     init=300.0
    ... )

    Notes
    -----
    Glint parameters are primarily used for aquatic remote sensing applications
    where specular reflection from the water surface is significant.
    """

    SURF_TEMP_K: Optional[StateVectorElement] = Field(
        default=None, description="Surface temperature in Kelvin"
    )
    SKY_GLINT: Optional[StateVectorElement] = Field(
        default=None, description="Sky glint parameter"
    )
    SUN_GLINT: Optional[StateVectorElement] = Field(
        default=None, description="Sun glint parameter"
    )


class Surface(BaseModel):
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
    surface_file : PathExists, optional
        Path to surface model file in .mat format containing reflectance
        priors, basis vectors, and covariance information.
    surface_category : {"surface", "multicomponent_surface", "glint_model_surface", "thermal_surface", "lut_surface"}, optional
        Type of surface model to use. Determines which surface class is
        instantiated.
    surface_class_file : PathExists, optional
        Path to surface classification file (ENVI format) for multi-surface mode.
    base_surface_class_file : PathExists, optional
        Path to base surface classification file for hierarchical classification.
    Surfaces : dict[str, dict]
        Multi-surface configuration dictionary mapping surface class names to
        their configurations. Each entry must have 'surface_category',
        'surface_file', and 'surface_int' (integer mapping).
    wavelength_file : PathExists, optional
        Path to wavelength file for surface model spectral sampling.
    select_on_init : bool
        Force use of initialization state and never change surface model
        selection. Default is True.
    selection_metric : {"Euclidean", "SGA", "NormSGA"}
        Metric for surface model selection in multi-surface mode. "SGA"
        (Spectral Gradient Angle) is recommended. Default is "SGA".
    statevector : SurfaceStateVector
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
    >>> from isofit.configs.sections.surface import Surface
    >>> surf = Surface(
    ...     surface_file="surface_priors.mat",
    ...     surface_category="multicomponent_surface",
    ...     terrain_style="flat"
    ... )

    Multi-surface configuration example:

    >>> multi_surf = Surface(
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
    SurfaceStateVector : Surface parameters for optimization
    """

    multi_surface_flag: bool = Field(
        default=False, description="Enable multi-surface mode"
    )

    use_background_rfl: bool = Field(
        default=False, description="Use background reflectance"
    )

    surface_file: Optional[PathExists] = Field(
        default=None, description="Path to surface model file (.mat format)"
    )

    surface_category: Optional[
        Literal[
            "surface",
            "multicomponent_surface",
            "glint_model_surface",
            "thermal_surface",
            "lut_surface",
        ]
    ] = Field(default=None, description="Type of surface model to use")

    surface_class_file: Optional[PathExists] = Field(
        default=None, description="Path to surface classification file"
    )

    base_surface_class_file: Optional[PathExists] = Field(
        default=None, description="Path to base surface classification file"
    )

    Surfaces: Dict[str, Dict] = Field(
        default_factory=dict, description="Multi-surface configuration dictionary"
    )

    wavelength_file: Optional[PathExists] = Field(
        default=None, description="Path to wavelength file"
    )

    select_on_init: bool = Field(
        default=True, description="Force use of initialization state and never change"
    )

    selection_metric: Literal["Euclidean", "SGA", "NormSGA"] = Field(
        default="SGA", description="Metric for surface selection"
    )

    statevector: SurfaceStateVector = Field(
        default_factory=SurfaceStateVector,
        description="Surface state vector configuration",
    )

    emissivity_for_surface_T_init: float = Field(
        default=0.98,
        description="Initial emissivity value for surface temperature (recommended by Glynn Hulley)",
    )

    terrain_style: Literal["flat", "dem", "solved"] = Field(
        default="flat", description="Style of terrain to use in the forward model"
    )

    max_slope: float = Field(
        default=90.0,
        description="Max slope value for LUT calculations (only relevant for 'dem' terrain_style)",
    )

    refractive_index_path: str = Field(
        default="", description="Path to refractive index data"
    )

    @field_validator("Surfaces")
    @classmethod
    def validate_surfaces(cls, v: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Validate multi-surface configuration.

        Parameters
        ----------
        v : dict[str, dict]
            Multi-surface configuration dictionary to validate.

        Returns
        -------
        dict[str, dict]
            The validated configuration.

        Raises
        ------
        ValueError
            If any surface configuration is missing required fields
            (surface_category, surface_file, or valid surface_int).

        Notes
        -----
        Each surface entry must have:
        - surface_category: Type of surface model
        - surface_file: Path to prior/model file
        - surface_int: Integer >= 0 mapping to classification values
        """
        if v:
            for name, config in v.items():
                if not config.get("surface_category"):
                    raise ValueError(f"Surface '{name}' missing surface_category")
                if not config.get("surface_file"):
                    raise ValueError(f"Surface '{name}' missing surface_file")
                if config.get("surface_int", -1) < 0:
                    raise ValueError(
                        f"Surface '{name}' missing valid surface_int mapping"
                    )
        return v
