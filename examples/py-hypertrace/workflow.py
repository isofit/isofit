#!/usr/bin/env python3
#
# Authors: Alexey Shiklomanov

import sys
import json
import itertools
import logging

from hypertrace import do_hypertrace, mkabs

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

clean = False
if len(sys.argv) > 1:
    configfile = sys.argv[1]
    if "--clean" in sys.argv:
        clean = True
else:
    configfile = "./config.json"
configfile = mkabs(configfile)
logger.info("Using config file `%s`", configfile)

with open(configfile) as f:
    config = json.load(f)

wavelength_file = mkabs(config["wavelength_file"])
reflectance_file = mkabs(config["reflectance_file"])
if "libradtran_template_file" in config:
    raise Exception("`libradtran_template_file` is deprecated. Use `rtm_template_file` instead.")
rtm_template_file = mkabs(config["rtm_template_file"])
lutdir = mkabs(config["lutdir"])
outdir = mkabs(config["outdir"])

if clean and outdir.exists():
    import shutil
    shutil.rmtree(outdir)

isofit_config = config["isofit"]
hypertrace_config = config["hypertrace"]

# Make RTM paths absolute
vswir_settings = isofit_config["forward_model"]["radiative_transfer"]["radiative_transfer_engines"]["vswir"]
for key in ["lut_path", "template_file", "engine_base_dir"]:
    if key in vswir_settings:
        vswir_settings[key] = str(mkabs(vswir_settings[key]))

# Create iterable config permutation object
ht_iter = itertools.product(*hypertrace_config.values())
logger.info("Starting Hypertrace workflow.")
for ht in ht_iter:
    argd = dict()
    for key, value in zip(hypertrace_config.keys(), ht):
        argd[key] = value
    logger.info("Running config: %s", argd)
    do_hypertrace(isofit_config, wavelength_file, reflectance_file,
                  rtm_template_file, lutdir, outdir,
                  **argd)
logging.info("Workflow completed successfully.")
