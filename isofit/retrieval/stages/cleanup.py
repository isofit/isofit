import logging
from glob import glob
from pathlib import Path

from isofit.retrieval.utils import utils

Logger = logging.getLogger(__name__)


@utils.log_elapse
def cleanup(config) -> None:
    """
    Remove the intermediate files matched by ``cleanup.remove``, such as the
    MODTRAN run files and the background reflectance.
    """
    files = sorted(
        {Path(file) for pattern in config.cleanup.remove for file in glob(pattern)}
    )

    Logger.info(f"Removing {len(files)} intermediate files")
    utils.remove(*[file for file in files if file.is_file()])
