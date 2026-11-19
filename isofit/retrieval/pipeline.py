"""
Pipeline stages for ISOFIT retrieval.

Each stage is a function of the config and communicates with the other stages
only through the files set in the config, e.g. ``segment.labels``, so a stage
may be run on its own once the stages before it have completed. The ``tc`` base
config sets every file relative to ``output.dir``, and its mixins choose between
alternatives by repointing them (see ``isofit/configs/templates/bases/tc.yml``
and ``isofit/configs/templates/mixins/``). Expensive stages skip work whose
outputs already exist, which makes re-running the pipeline a resume.

The stages, in order, and the apply_oe step each replaces:

1. setup - Validate inputs, create the working directories, stage the inputs
   and instrument files, write the wavelength file
2. preprocess - Interpolate bad radiance channels, compute the sky view factor
3. scene - Acquisition time, mean geometry, and LUT grids from the obs/loc, and
   the MODTRAN templates (apply_oe's inline metadata derivation)
4. surface - Build/check the surface model(s), classify a multisurface scene
5. segment - SLIC superpixel segmentation and per-superpixel extraction
6. presolve - 1-D water vapor retrieval to narrow the H2O LUT range
7. background - Background reflectance from the presolve
8. configure - Write the main ISOFIT config (apply_oe's ``config_only`` is
   running the stages up to here)
9. lut - Build the main LUT, then stop the pipeline when ``lut.exit`` is set.
   Disable it to build the LUT as part of the inversion instead
10. invert - Run ISOFIT over the superpixels or every pixel
11. interpolate - Empirical or analytical line from superpixels to every pixel
12. cleanup - Remove intermediate files
"""

import logging

from rich.progress import Progress

from isofit import ray
from isofit.debug.resource_tracker import FileResources
from isofit.retrieval import Console
from isofit.retrieval.stages import Stages, check
from isofit.retrieval.utils import utils

Logger = logging.getLogger(__name__)


@utils.log_elapse
def run(config) -> None:
    """
    Execute every enabled stage of the ISOFIT retrieval pipeline.

    Stages are gated by their ``{stage}.enabled`` config flag and executed in
    the order of :data:`~isofit.retrieval.stages.Stages`. With ``lut.exit``,
    the pipeline stops once the LUT is built.

    Parameters
    ----------
    config
        The fully loaded configuration.
    """
    check(config)

    steps = [name for name in Stages if getattr(config, name).enabled]
    if config.lut.exit:
        steps = steps[: steps.index("lut") + 1]
    Logger.info(f"Executing stages: {', '.join(steps)}")

    tracker = None
    if file := config.log.resources:
        # n_cores=None lets Ray use every core
        cores = config.implementation.n_cores or "all"
        tracker = FileResources(file, reset=True, cores=cores)
        tracker.start()

    try:
        with Progress(*Progress.get_default_columns(), console=Console) as progress:
            task = progress.add_task("Executing pipeline", total=len(steps))
            for step in steps:
                Console.rule(style="dim")
                progress.update(task, description=f"Executing: {step}")

                Stages[step](config)

                progress.advance(task)
    finally:
        ray.shutdown()
        if tracker:
            tracker.stop()
