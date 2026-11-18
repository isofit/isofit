"""
Scene statistics shared by the downstream stages.

apply_oe derived the acquisition time, mean geometry, and LUT grids inline and
passed them between steps in memory. Here the ``scene`` stage computes them once
and saves a :class:`Scene` to ``scene.file``, which the surface, presolve,
background, and configure stages load, so any of those can be restarted on
their own.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

from isofit.retrieval.utils import utils

Logger = logging.getLogger(__name__)


@dataclass
class Scene:
    """
    Acquisition time, mean geometry, and LUT grids of a scene.
    """

    time: datetime
    """Acquisition time at the mean UTC time of the obs"""

    path_km: float
    observer_azimuth: float
    observer_zenith: float
    solar_azimuth: float
    solar_zenith: float
    relative_azimuth: float
    latitude: float
    longitude: float
    elevation_km: float
    altitude_km: float

    h2o_max: float
    """Water vapor upper bound for ``scene.atmosphere_type`` at the lowest
    elevation of the scene, g/cm2"""

    grids: dict[str, Optional[list[float]]] = field(default_factory=dict)
    """LUT grid of each geometry dimension, keyed by its LUT name. None when the
    dimension is not gridded, in which case its mean is used"""

    @property
    def day_of_year(self) -> int:
        return self.time.timetuple().tm_yday

    @property
    def gmtime(self) -> float:
        """Decimal UTC hours"""
        return self.time.hour + self.time.minute / 60

    def save(self, file) -> None:
        data = asdict(self)
        data["time"] = self.time.isoformat()

        with open(file, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, config) -> "Scene":
        """
        Load the scene saved by the scene stage.
        """
        file = config.scene.file
        if not file.exists():
            raise FileNotFoundError(
                f"Scene statistics not found, run the scene stage first: {file}"
            )

        with open(file) as f:
            data = json.load(f)

        data["time"] = datetime.fromisoformat(data["time"])

        return cls(**data)


@utils.log_elapse
def scene(config) -> None:
    """
    Derive the acquisition time, mean geometry, and LUT grids from the obs and
    loc, and write the MODTRAN templates.
    """
    # TODO: Port from apply_oe / template_construction
    # 1. Acquisition date from the fid: move get_sensor_metadata_from_fid's
    #    per-sensor datetime parsing into utils.sensor alongside FID_PARSERS.
    #    The inversion windows it also returned are already in the tc sensor
    #    mixins, so only the datetime is needed
    # 2. get_metadata_from_obs on setup.obs with
    #    scene.trim_lines and scene.max_flight_duration: mean geometry, UTC time
    #    and day rollover, and the observer_zenith/solar_zenith/relative_azimuth
    #    grids from config.lut. Replace LUTConfig.get_grid with a function of a
    #    config.lut dimension
    # 3. get_metadata_from_loc on setup.loc: mean lat/lon,
    #    elevation, and the elevation grid, expanded by 2 km when
    #    surface_elevation_km is in the atmosphere statevector (apply_oe:
    #    --pressure_elevation)
    # 4. Engine adjustments: clip elevations below 0 km and altitude above 99 km
    #    for 6S-based engines (srtmnet, sixs); error on a negative altitude
    # 5. h2o_max from modtran_water_upperbound_polynomials
    # 6. Write the MODTRAN templates (write_modtran_template): the main one to
    #    forward_model.atmosphere.engine.template_file, and, when presolve is
    #    enabled, the AER_NONE one to presolve.template
    # 7. Scene(...).save(scene.file), skipped when it exists
    raise NotImplementedError("scene")
