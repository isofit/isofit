import logging

import numpy as np

from isofit.retrieval.stages.scene import Scene
from isofit.retrieval.utils import solve, utils

Logger = logging.getLogger(__name__)


@utils.log_elapse
def presolve(config) -> None:
    """
    Run a 1-D water vapor retrieval, over the superpixels when segmenting, to
    narrow the H2O range of the main LUT.
    """
    from isofit.core.isofit import Isofit

    pre = config.presolve
    if utils.skip(pre.state):
        return

    scene = Scene.load(config)
    h2o = np.linspace(0.2, scene.h2o_max - 0.01, pre.grid_points)
    h2o = h2o.round(2).tolist()
    Logger.info(f"Presolve H2O grid: {h2o}")

    statevector = solve.statevector(
        names=["H2OSTR"],
        lut_grid={"H2OSTR": h2o},
        configured=config.forward_model.atmosphere.statevector.model_dump(mode="json"),
    )

    data = solve.base(config, pre.layers)
    data["forward_model"]["atmosphere"].update(
        lut_grid={"H2OSTR": h2o},
        lut_subset={"H2OSTR": None},
        statevector=statevector,
        statevector_names=list(statevector),
    )
    solve.write(data, pre.file)

    # The H2O range is derived from this output by the configure stage, so the
    # presolve does not need to be rerun to change how the range is derived
    Logger.info(f"Running the presolve with {pre.file}")
    Isofit(str(pre.file), level=config.log.level, logfile=config.log.file).run()
