import logging

from isofit.retrieval.utils import utils

Logger = logging.getLogger(__name__)


@utils.log_elapse
def lut(config) -> None:
    """
    Build the LUT of the main ISOFIT config written by the configure stage,
    running the radiative transfer simulations of its grid. With ``lut.exit``,
    ``isofit retrieval run`` stops after this stage.
    """
    from isofit import ray
    from isofit.atmosphere import Atmosphere
    from isofit.configs import create_new_config

    file = config.configure.file
    if not file.exists():
        raise FileNotFoundError(f"Run the configure stage first, not found: {file}")

    solve = create_new_config(file)
    path = solve.forward_model.atmosphere.lut_path

    if path.exists():
        Logger.info(f"LUT already exists, skipping: {path}")
        return

    impl = solve.implementation
    ray.init(
        num_cpus=impl.n_cores,
        _temp_dir=impl.ray_temp_dir,
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    # Constructing the atmosphere builds its LUT when one does not exist
    Logger.info(f"Building the LUT: {path}")
    Atmosphere(solve)
