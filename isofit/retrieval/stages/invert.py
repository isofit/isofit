import logging

from isofit.retrieval.utils import utils

Logger = logging.getLogger(__name__)


@utils.log_elapse
def invert(config) -> None:
    """
    Run the main ISOFIT inversion written by the configure stage, over the
    superpixels when interpolating, otherwise over every pixel.
    """
    from isofit.configs import create_new_config
    from isofit.core.isofit import Isofit

    file = config.configure.file
    if not file.exists():
        raise FileNotFoundError(f"Run the configure stage first, not found: {file}")

    solve = create_new_config(file)
    outputs = [file for file in solve.output.get_output_files() if file]
    if outputs and all(utils.skip(output) for output in outputs):
        return

    Logger.info(f"Running ISOFIT with {file}")
    Isofit(str(file), level=config.log.level, logfile=config.log.file).run()
