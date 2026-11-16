"""
Pipeline stages for ISOFIT retrieval.

Each stage is a function of the run
and communicates with the other stages only through files in the working
directory, so a stage may be run on its own once the stages before it have
completed. Expensive stages skip work whose outputs already exist, which makes
re-running the pipeline a resume.

The stages, in order, and the apply_oe step each replaces:

1. setup - Validate inputs, create the working directories, stage inputs, write
   the wavelength file
2. surface - Build/stage the surface model(s) and classify a multisurface scene
3. preprocess - Compute the sky view factor and interpolate bad radiance
4. segment - SLIC superpixel segmentation and per-superpixel extraction
5. presolve - 1-D water vapor retrieval to narrow the H2O LUT range
6. configure - Write the MODTRAN template and the main ISOFIT config
   (apply_oe's ``config_only`` stops after this stage)
7. background - Background reflectance from the presolve
8. invert - Run ISOFIT over the superpixels or every pixel
9. interpolate - Empirical or analytical line from superpixels to every pixel
10. cleanup - Remove intermediate RTM and background reflectance files
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import ray
from rich.progress import Progress
from spectral.io import envi

from isofit.retrieval import Console, utils

Logger = logging.getLogger(__name__)


@utils.log_elapse
def run(config) -> None:
    """
    Execute every enabled stage of the ISOFIT retrieval pipeline.

    Stages are gated by their ``{stage}.enabled`` config flag and executed in
    the order of :data:`STAGES`.

    Parameters
    ----------
    config
        The fully loaded configuration.
    """
    steps = [name for name in STAGES if getattr(c, name).enabled]
    Logger.info(f"Executing stages: {', '.join(steps)}")

    # tracker = None
    # if c.log.resources:
    #     if c.log.file:
    #         jsonl = Path(c.log.file).with_suffix(".resources.jsonl")
    #         tracker = FileResources(jsonl, reset=True, cores=c.n_cores)
    #         tracker.start()
    #     else:
    #         Logger.error("log.resources requires log.file to be set")

    try:
        with Progress(*Progress.get_default_columns(), console=Console) as progress:
            task = progress.add_task("Executing pipeline", total=len(steps))
            for step in steps:
                Console.rule(style="dim")
                progress.update(task, description=f"Executing: {step}")

                func = STAGES[step]
                utils.log_elapse(func, config)

                progress.advance(task)
    finally:
        ray.shutdown()
        # if tracker:
        #     tracker.stop()


@utils.log_elapse
def setup(config) -> None:
    """
    Validate the inputs, create the working directories, stage the input files,
    and write the wavelength file.
    """
    Logger.info("STAGE: setup")
    Logger.info(f"Config:\n{config.to_yaml()}")
    return

    paths.make_directories()
    paths.stage_files()

    tmpl.write_wavelength_file(paths.wavelength_path, *config.wavelengths)


STAGES = {
    "setup": setup,
    # "surface": surface,
    # "preprocess": preprocess,
    # "segment": segment,
    # "presolve": presolve,
    # "configure": configure,
    # "background": background,
    # "invert": invert,
    # "interpolate": interpolate,
    # "cleanup": cleanup,
}
