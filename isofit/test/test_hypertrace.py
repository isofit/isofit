"""Test ISOFIT against the Hypertrace workflow"""

import itertools
import json
import os
import sys
from unittest import mock

import pytest

# Mark entire file with the 'hypertrace' marker.
pytestmark = pytest.mark.hypertrace


# The 'hypertrace' module lives within 'examples/py-hypertrace', so need to
# enable files in the current working directory. This may have other side
# effects depending on how/wat the 'isofit' library imports.
_NEWPATH = sys.path.copy()
_NEWPATH.insert(0, ".")


@mock.patch("sys.path", new=_NEWPATH)
def test_hypertrace(monkeypatch):
    monkeypatch.chdir("examples/py-hypertrace")

    # This is a module that lives in 'examples/py-hypertrace'
    from hypertrace import do_hypertrace, mkabs

    configfile = "configs/example-srtmnet-ci.json"

    configfile = mkabs(configfile)
    print(f"Using config file `{configfile}`")

    with open(configfile) as f:
        config = json.load(f)

    wavelength_file = mkabs(config["wavelength_file"])
    reflectance_file = mkabs(config["reflectance_file"])
    if "libradtran_template_file" in config:
        raise Exception(
            "`libradtran_template_file` is deprecated. Use `rtm_template_file` instead."
        )
    rtm_template_file = mkabs(config["rtm_template_file"])
    lutdir = mkabs(config["lutdir"])
    outdir = mkabs(config["outdir"])

    isofit_config = config["isofit"]
    hypertrace_config = config["hypertrace"]

    # Make RTM paths absolute
    vswir_settings = isofit_config["forward_model"]["radiative_transfer"][
        "radiative_transfer_engines"
    ]["vswir"]
    for key in ["lut_path", "template_file", "engine_base_dir"]:
        if key in vswir_settings:
            vswir_settings[key] = str(mkabs(vswir_settings[key]))

    # Create iterable config permutation object
    ht_iter = itertools.product(*hypertrace_config.values())
    print("Starting Hypertrace workflow.")

    # Print iter list to record what iterations will be run
    print(
        "\n".join(
            [
                ",".join(map(str, item))
                for item in list(itertools.product(*hypertrace_config.values()))
            ]
        )
    )
    with open("py-hypertrace_iteration_list.txt", "w") as outfile:
        outfile.write(
            "\n".join(
                [
                    ",".join(map(str, item))
                    for item in list(itertools.product(*hypertrace_config.values()))
                ]
            )
        )

    for ht in ht_iter:
        argd = dict()
        for key, value in zip(hypertrace_config.keys(), ht):
            argd[key] = value
        print("Running config: %s", argd)
        do_hypertrace(
            isofit_config,
            wavelength_file,
            reflectance_file,
            rtm_template_file,
            lutdir,
            outdir,
            **argd,
        )
    print("Workflow completed successfully.")
