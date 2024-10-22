import logging
import os

Logger = logging.getLogger(__file__)

if os.environ.get("ISOFIT_DEBUG") == "1":
    Logger.info("Using ISOFIT internal ray")
    from isofit.debug import ray_bypass as ray
else:
    import ray
