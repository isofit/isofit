import json
from pathlib import Path

import pytest

from isofit.configs import create_new_config, legacy, load_config_dict
from isofit.configs.loader import base_config_name, resolve_config_dict
from isofit.data import env

Data = Path(__file__).parent / "data"
Wavelengths = str(Data / "wavelengths.txt")


def legacy_config(engine: dict) -> dict:
    """A minimal config using the legacy forward_model.radiative_transfer layout."""
    return {
        "input": {},
        "output": {"dir": "/tmp/isofit-test-output"},
        "forward_model": {
            "instrument": {"wavelength_file": Wavelengths, "SNR": 500},
            "surface": {"surface_category": "surface"},
            "radiative_transfer": {
                "radiative_transfer_engines": {"vswir": engine},
                "statevector": {
                    "H2OSTR": {
                        "bounds": [0.5, 5.0],
                        "scale": 0.01,
                        "prior_sigma": 100.0,
                        "prior_mean": 1.0,
                        "init": 0.75,
                    }
                },
                "lut_grid": {"H2OSTR": [0.5, 1.0, 2.0]},
                "unknowns": {"H2O_ABSCO": 0.01},
            },
        },
        "implementation": {"inversion": {"windows": [[400.0, 1000.0]]}},
    }


def test_legacy_radiative_transfer():
    """The radiative_transfer section is converted, splitting engine/atmosphere keys."""
    config = legacy.convert(
        legacy_config(
            {
                "engine_name": "6s",
                "engine_base_dir": "/sixs",
                "day": 26,
                "lut_path": "lut.nc",
                "lut_names": {"H2OSTR": None},
                "irradiance_file": "irr.txt",
            }
        )
    )

    forward_model = config["forward_model"]
    assert "radiative_transfer" not in forward_model

    atmosphere = forward_model["atmosphere"]
    assert atmosphere["engine"] == {"name": "sixs", "base_dir": "/sixs", "day": 26}
    assert atmosphere["lut_path"] == "lut.nc"
    assert atmosphere["lut_subset"] == {"H2OSTR": None}
    assert atmosphere["irradiance_file"] == "irr.txt"
    assert list(atmosphere["statevector"]) == ["H2OSTR"]
    assert atmosphere["lut_grid"] == {"H2OSTR": [0.5, 1.0, 2.0]}
    assert atmosphere["unknowns"] == {"H2O_ABSCO": 0.01}


def test_legacy_nested_engine():
    """
    Older apply_oe configs nested every atmosphere key under `engine`, with
    engine_base_dir meaning the 6S path for sRTMnet.
    """
    atmosphere = legacy.convert_atmosphere(
        {
            "engine": {
                "engine_name": "sRTMnet",
                "engine_base_dir": "/sixs",
                "emulator_file": "em.h5",
                "multipart_transmittance": True,
                "lut_path": "lut.zarr",
                "lut_names": {"H2OSTR": None},
                "statevector_names": ["H2OSTR"],
            },
            "lut_grid": {"H2OSTR": [1.0, 2.0]},
        }
    )

    assert atmosphere == {
        "engine": {"name": "srtmnet", "sixs_dir": "/sixs", "emulator_file": "em.h5"},
        "multipart_transmittance": True,
        "lut_path": "lut.zarr",
        "lut_subset": {"H2OSTR": None},
        "statevector_names": ["H2OSTR"],
        "lut_grid": {"H2OSTR": [1.0, 2.0]},
    }


def test_legacy_shared_keys():
    """multipart_transmittance is a field of both the atmosphere and MODTRAN."""
    atmosphere = legacy.convert_atmosphere(
        {"engine": {"name": "modtran"}, "multipart_transmittance": True}
    )
    assert atmosphere["multipart_transmittance"] is True
    assert atmosphere["engine"]["multipart_transmittance"] is True


@pytest.mark.parametrize(
    "name, expected",
    [("6S", "sixs"), ("sRTMnet", "srtmnet"), ("KernelFlowsGP", "kernelflows")],
)
def test_legacy_engine_names(name, expected):
    assert legacy.engine_name(name) == expected


def test_legacy_prefers_current_key():
    """When both a legacy key and its replacement are set, the replacement wins."""
    atmosphere = legacy.convert_atmosphere(
        {"engine": {"name": "sixs", "base_dir": "/new", "engine_base_dir": "/old"}}
    )
    assert atmosphere["engine"] == {"name": "sixs", "base_dir": "/new"}


def test_legacy_current_schema_unchanged():
    """Configs already in the current schema pass through untouched."""
    config = resolve_config_dict(legacy_config({"engine_name": "6s"}))
    assert legacy.convert(config) == config


def test_legacy_does_not_mutate_input():
    data = legacy_config({"engine_name": "6s", "engine_base_dir": "/sixs"})
    snapshot = json.dumps(data, sort_keys=True)
    legacy.convert(data)
    assert json.dumps(data, sort_keys=True) == snapshot


@pytest.mark.parametrize(
    "engine, expected",
    [
        ({"name": "sixs"}, "sixs"),
        ({"name": "6S"}, "sixs"),
        ({"name": "sRTMnet"}, "srtmnet"),
        ({"name": "not-an-engine"}, "default"),
        (None, "default"),
    ],
)
def test_base_config_name(engine, expected):
    config = {"forward_model": {"atmosphere": {"engine": engine}}}
    assert base_config_name(config) == expected


def test_legacy_config_loads():
    """
    A sparse legacy config validates: defaults come from the base config, and the
    legacy radiative_transfer section is converted rather than dropped.
    """
    config = load_config_dict(
        legacy_config({"engine_name": "6s", "engine_base_dir": env.sixs, "day": 26})
    )

    atmosphere = config.forward_model.atmosphere
    assert atmosphere.engine.name == "sixs"
    assert atmosphere.engine.day == 26
    assert str(atmosphere.engine.base_dir) == str(Path(env.sixs))
    assert list(atmosphere.statevector.root) == ["H2OSTR"]
    assert atmosphere.lut_grid == {"H2OSTR": [0.5, 1.0, 2.0]}
    assert atmosphere.unknowns.root == {"H2O_ABSCO": 0.01}

    # Defaults from configs/default.yml
    assert config.implementation.mode == "inversion"
    assert config.implementation.inversion.least_squares_params.method == "trf"
    assert config.forward_model.surface.selection_metric == "SGA"


def test_defaults_interpolate_after_overlay():
    """Base-config references (e.g. ${output.dir}) resolve against user values."""
    config = load_config_dict(legacy_config({"engine_name": "6s"}))
    assert config.forward_model.atmosphere.lut_path == Path(
        "/tmp/isofit-test-output/lut/lut.zarr"
    )


def test_resolve_is_idempotent():
    """Resolving an already-resolved config (e.g. apply_oe output) is a no-op."""
    once = resolve_config_dict(legacy_config({"engine_name": "6s"}))
    assert "radiative_transfer" not in once["forward_model"]
    assert "env" not in once
    assert resolve_config_dict(once) == once


def test_resolve_does_not_mutate_input():
    data = legacy_config({"engine_name": "6s"})
    snapshot = json.dumps(data, sort_keys=True)
    resolve_config_dict(data, base_dir="/tmp")
    assert json.dumps(data, sort_keys=True) == snapshot


def test_create_new_config_relative_paths(tmp_path):
    """Relative paths resolve against the config file's directory."""
    data = legacy_config({"engine_name": "6s"})
    data["forward_model"]["instrument"]["wavelength_file"] = "wl.txt"
    (tmp_path / "wl.txt").write_text(Path(Wavelengths).read_text())

    file = tmp_path / "config.json"
    file.write_text(json.dumps(data))

    config = create_new_config(file)
    assert config.forward_model.instrument.wavelength_file == tmp_path / "wl.txt"


# Example run configs hand-written against the legacy config format, relative to
# the examples directory. These are what `isofit run` loads in test_examples.py
# fmt: off
ExampleConfigs = [
    "20151026_SantaMonica/configs/run/prm20151026t173213_D8W_6s.json",
    "20151026_SantaMonica/configs/run/prm20151026t173213_D8p5W_6s.json",
    "20151026_SantaMonica/configs/run/prm20151026t173213_D9W_6s.json",
    "20151026_SantaMonica/configs/run/prm20151026t173213_D9p5W_6s.json",
    "20171108_Pasadena/configs/libradtran/and20171108t184227_beckmanlawn-libradtran.json",
    "20171108_Pasadena/configs/modtran/ang20171108t173546_darklot.json",
    "20171108_Pasadena/configs/modtran/ang20171108t173546_horse.json",
    "20171108_Pasadena/configs/modtran/ang20171108t184227_astrogreen.json",
    "20171108_Pasadena/configs/modtran/ang20171108t184227_astrored.json",
    "20171108_Pasadena/configs/modtran/ang20171108t184227_beckmanlawn.json",
    "20171108_Pasadena/configs/modtran/ang20171108t184227_beckmanlawn-oversmoothed.json",
    "20171108_Pasadena/configs/modtran/ang20171108t184227_beckmanlawn-undersmoothed.json",
    "20171108_Pasadena/configs/topoflux/ang20171108t184227_beckmanlawn-multimodtran-topoflux.json",
    "20231110_Prism_Multisurface/configs/prm20231110t071521_multi_surface_isofit.json",
]
# fmt: on


@pytest.mark.examples
@pytest.mark.parametrize("path", ExampleConfigs)
def test_example_configs(path, monkeypatch):
    """The example configs validate against the new config models."""
    file = Path(env.examples) / path
    if not file.exists():
        pytest.skip(f"Example not installed: {file}")

    # These examples ship prebuilt LUTs, so MODTRAN need not be installed
    monkeypatch.delitem(env.config[env.section], "modtran", raising=False)

    config = create_new_config(file)
    config.get_config_errors()
