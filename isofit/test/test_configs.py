import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.io import savemat

from isofit.configs import Config, create_new_config, legacy
from isofit.configs.sections.atmosphere import AtmosphereConfig
from isofit.configs.sections.atmosphere.engines.srtmnet import sRTMnetConfig
from isofit.configs.sections.implementation import ImplementationConfig
from isofit.configs.sections.inversion import InversionConfig
from isofit.configs.sections.io import OutputConfig
from isofit.core.multistate import update_config_for_surface
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


def validate(data: dict) -> Config:
    """Upgrade a (possibly legacy) config dict and validate it."""
    return Config.model_validate(legacy.convert(data))


def resolve(data: dict) -> dict:
    """Validate a config dict and dump the complete result, as apply_oe writes it."""
    return validate(data).model_dump(mode="json")


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
    config = resolve(legacy_config({"engine_name": "6s"}))
    assert legacy.convert(config) == config


def test_legacy_does_not_mutate_input():
    data = legacy_config({"engine_name": "6s", "engine_base_dir": "/sixs"})
    snapshot = json.dumps(data, sort_keys=True)
    legacy.convert(data)
    assert json.dumps(data, sort_keys=True) == snapshot


def test_legacy_config_loads():
    """
    A sparse legacy config validates: defaults come from the base config, and the
    legacy radiative_transfer section is converted rather than dropped.
    """
    config = validate(
        legacy_config({"engine_name": "6s", "engine_base_dir": env.sixs, "day": 26})
    )

    atmosphere = config.forward_model.atmosphere
    assert atmosphere.engine.name == "sixs"
    assert atmosphere.engine.day == 26
    assert str(atmosphere.engine.base_dir) == str(Path(env.sixs))
    assert list(atmosphere.statevector.root) == ["H2OSTR"]
    assert atmosphere.lut_grid == {"H2OSTR": [0.5, 1.0, 2.0]}
    assert atmosphere.unknowns.root == {"H2O_ABSCO": 0.01}

    # Defaults from isofit/configs/templates/default.yml
    assert config.implementation.mode == "inversion"
    assert config.implementation.inversion.least_squares_params.method == "trf"
    assert config.forward_model.surface.selection_metric == "Euclidean"
    assert config.forward_model.atmosphere.interpolator_style == "mlg_numba"


@pytest.mark.parametrize(
    "lut_grid, match",
    [
        ({"H2OSTR": [1.0]}, "less than the required 2"),
        ({"H2OSTR": [1.0, 1.0, 2.0]}, "duplicate values"),
    ],
)
def test_invalid_lut_grid(lut_grid, match):
    data = legacy_config({"engine_name": "6s"})
    data["forward_model"]["radiative_transfer"]["lut_grid"] = lut_grid
    with pytest.raises(ValidationError, match=match):
        validate(data)


def test_invalid_interpolator_style():
    data = legacy_config({"engine_name": "6s", "interpolator_style": "nds-3"})
    with pytest.raises(ValidationError, match="interpolator_style"):
        validate(data)


def test_windows_required_outside_simulation():
    data = legacy_config({"engine_name": "6s"})
    data["implementation"]["inversion"]["windows"] = []
    with pytest.raises(ValidationError, match="inversion.windows"):
        validate(data)


def test_surface_file_required():
    """Without Surfaces, surface_file must be set (checked by get_config_errors)."""
    config = validate(legacy_config({"engine_name": "6s"}))
    errors, _ = config.forward_model.surface.get_config_issues()
    assert any("surface_file not specified" in e for e in errors)
    with pytest.raises(AttributeError):
        config.get_config_errors()


def test_surface_statevector_mismatch_warns(tmp_path):
    """Configured statevector values differing from the .mat file warn."""
    mat = tmp_path / "surface.mat"
    savemat(
        mat,
        {
            "statevec_names": ["SURF_TEMP_K"],
            "bounds": [[250.0, 400.0]],
            "scale": [100.0],
            "prior_mean": [300.0],
            "prior_sigma": [1.0],
            "init": [300.0],
        },
    )

    data = legacy_config({"engine_name": "6s"})
    data["forward_model"]["surface"] = {
        "surface_category": "thermal_surface",
        "surface_file": str(mat),
        "statevector": {
            "SURF_TEMP_K": {
                "bounds": [250.0, 400.0],
                "scale": 100.0,
                "prior_mean": 310.0,
                "prior_sigma": 1.0,
                "init": 300.0,
            }
        },
    }
    config = validate(data)

    errors, warnings = config.forward_model.surface.get_config_issues()
    assert errors == []
    assert len(warnings) == 1
    assert "SURF_TEMP_K: prior_mean" in warnings[0]


def glint_mat(path, scale):
    """Write a surface .mat file storing SKY_GLINT and SUN_GLINT elements."""
    savemat(
        path,
        {
            "statevec_names": ["SKY_GLINT", "SUN_GLINT"],
            "bounds": [[0.0, 1.0], [0.0, 10.0]],
            "scale": [scale, scale],
            "prior_mean": [0.02, 0.5],
            "prior_sigma": [1.0, 1.0],
            "init": [0.02, 0.5],
        },
    )


Glint = {
    "SKY_GLINT": {"bounds": [0.0, 1.0], "scale": 1.0, "prior_mean": 0.02,
                  "prior_sigma": 1.0, "init": 0.02},
    "SUN_GLINT": {"bounds": [0.0, 10.0], "scale": 1.0, "prior_mean": 0.5,
                  "prior_sigma": 1.0, "init": 0.5},
}  # fmt: skip


def test_multi_surface_statevector_mismatch_warns(tmp_path):
    """Each sub-surface file is checked against its own statevector, and every
    mismatching element is reported."""
    matching, differing = tmp_path / "matching.mat", tmp_path / "differing.mat"
    glint_mat(matching, scale=1.0)
    glint_mat(differing, scale=2.0)

    data = legacy_config({"engine_name": "6s"})
    data["forward_model"]["surface"] = {
        "multi_surface_flag": True,
        "Surfaces": {
            "glint_model_surface": {
                "surface_category": "glint_model_surface",
                "surface_file": str(matching),
                "surface_int": 1,
                "statevector": Glint,
            },
            "other_glint": {
                "surface_category": "glint_model_surface",
                "surface_file": str(differing),
                "surface_int": 2,
                "statevector": Glint,
            },
            "not_built_yet": {
                "surface_category": "multicomponent_surface",
                "surface_file": str(tmp_path / "missing.mat"),
                "surface_int": 0,
            },
        },
    }
    config = validate(data)

    errors, warnings = config.forward_model.surface.get_config_issues()
    assert errors == []
    assert len(warnings) == 1
    assert f"SKY_GLINT: scale ({differing})" in warnings[0]
    assert f"SUN_GLINT: scale ({differing})" in warnings[0]
    assert str(matching) not in warnings[0]


def test_unset_statevector_values_match_nan(tmp_path):
    """Values unset in both the config and the .mat file (NaN) don't warn, and
    fields missing from the .mat file are skipped."""
    mat = tmp_path / "surface.mat"
    savemat(
        mat,
        {
            "statevec_names": ["SURF_TEMP_K"],
            "bounds": [[250.0, 400.0]],
            "scale": [np.nan],
        },
    )

    data = legacy_config({"engine_name": "6s"})
    data["forward_model"]["surface"] = {
        "surface_category": "thermal_surface",
        "surface_file": str(mat),
        "statevector": {"SURF_TEMP_K": {"bounds": [250.0, 400.0]}},
    }
    config = validate(data)

    assert config.forward_model.surface.get_config_issues() == ([], [])


@pytest.mark.parametrize(
    "sub_surface, match",
    [
        ({"surface_file": "a.mat", "surface_int": 0}, "surface_category"),
        ({"surface_category": "surface", "surface_int": 0}, "surface_file"),
        ({"surface_category": "surface", "surface_file": "a.mat"}, "surface_int"),
        (
            {"surface_category": "surface", "surface_file": "a.mat", "surface_int": -1},
            "surface_int",
        ),
        (
            {
                "surface_category": "surface",
                "surface_file": "a.mat",
                "surface_int": 0,
                "statevector": {"SKY_GLINT": {"scale": "big"}},
            },
            "scale",
        ),
    ],
)
def test_sub_surface_validation(sub_surface, match):
    """Malformed Surfaces entries fail at load time."""
    data = legacy_config({"engine_name": "6s"})
    data["forward_model"]["surface"] = {"Surfaces": {"bad": sub_surface}}
    with pytest.raises(ValidationError, match=match):
        validate(data)


def test_update_from_subconfig():
    """Sub-surface settings, including extra keys, overwrite the active surface."""
    data = legacy_config({"engine_name": "6s"})
    data["forward_model"]["surface"] = {
        "Surfaces": {
            "glint_model_surface": {
                "surface_category": "glint_model_surface",
                "surface_file": "glint.mat",
                "surface_int": 1,
                "terrain_style": "dem",
            }
        },
    }
    surface = validate(data).forward_model.surface
    surface.update_from_subconfig(surface.Surfaces["glint_model_surface"])

    assert surface.surface_category == "glint_model_surface"
    assert surface.surface_file == Path("glint.mat")
    assert surface.terrain_style == "dem"
    assert surface.statevector.SKY_GLINT is None


def test_update_config_for_surface_rt_statevector_elements():
    """A sub-surface's rt_statevector_elements are added to the atmosphere
    under their own names, once, even when updated repeatedly."""
    data = legacy_config({"engine_name": "6s"})
    data["forward_model"]["surface"] = {
        "surface_class_file": str(Wavelengths),
        "Surfaces": {
            "glint_model_surface": {
                "surface_category": "glint_model_surface",
                "surface_file": "glint.mat",
                "surface_int": 1,
                "rt_statevector_elements": {
                    "AOT550": {"bounds": [0.001, 1.0], "scale": 1.0},
                    "GNDALT": {"bounds": [0.0, 3.0], "scale": 1.0},
                },
            }
        },
    }
    config = validate(data)
    config.forward_model.atmosphere.statevector_names = ["H2OSTR"]
    for _ in range(2):
        config = update_config_for_surface(config, "glint_model_surface")

    atmosphere = config.forward_model.atmosphere
    assert atmosphere.statevector_names == ["AOT550", "GNDALT", "H2OSTR"]
    assert atmosphere.statevector.root["GNDALT"].bounds == [0.0, 3.0]
    assert "surface_elevation_km" not in atmosphere.statevector.root


def test_defaults_interpolate_after_overlay():
    """Base-config references (e.g. ${output.dir}) resolve against user values."""
    config = validate(legacy_config({"engine_name": "6s"}))
    assert config.forward_model.atmosphere.lut_path == Path(
        "/tmp/isofit-test-output/lut/lut.zarr"
    )


def test_resolve_is_idempotent():
    """Resolving an already-resolved config (e.g. apply_oe output) is a no-op."""
    once = resolve(legacy_config({"engine_name": "6s"}))
    assert "radiative_transfer" not in once["forward_model"]
    assert "env" not in once
    assert resolve(once) == once


def test_resolve_does_not_mutate_input():
    data = legacy.convert(legacy_config({"engine_name": "6s"}))
    data["output"]["estimated_state_file"] = "${output.dir}/state"
    snapshot = json.dumps(data, sort_keys=True)
    Config.model_validate(data)
    assert json.dumps(data, sort_keys=True) == snapshot


VswirWindows = [[380.0, 1325.0], [1435.0, 1770.0], [1965.0, 2500.0]]


def test_section_from_base():
    """A section seeded with base= takes that base config's subsection."""
    impl = ImplementationConfig(base="vswir")
    assert impl.mode == "inversion"
    assert impl.inversion.windows == VswirWindows


def test_section_from_base_overrides():
    """Keyword arguments are deep-merged over the base config's defaults."""
    assert ImplementationConfig(base="vswir", n_cores=4).n_cores == 4

    inversion = InversionConfig(base="vswir", mcmc={"iterations": 5})
    assert inversion.mcmc.iterations == 5
    assert inversion.mcmc.burnin == 200
    assert inversion.windows == VswirWindows

    output = OutputConfig(base="vswir", estimated_state_file=None)
    assert output.estimated_state_file is None


def test_section_from_base_interpolates():
    """References to other sections resolve, and follow the overrides."""
    lut_path = AtmosphereConfig(base="vswir").lut_path
    assert lut_path.parts[-3:] == ("isofit-output", "lut", "lut.zarr")

    output = OutputConfig(base="vswir", dir="/x")
    assert output.estimated_state_file == Path("/x/estimated_state")


def test_section_from_base_nested():
    """Engine models and nested model instances work with base=."""
    assert sRTMnetConfig(base="vswir").name == "srtmnet"

    impl = ImplementationConfig(base="vswir", inversion=InversionConfig(base="thermal"))
    assert impl.inversion.windows == VswirWindows + [[7500.0, 12000.0]]

    config = Config(base="vswir", implementation={"n_cores": 4})
    assert config.implementation.n_cores == 4
    assert config.forward_model.atmosphere.engine.name == "srtmnet"


def test_section_from_unknown_base():
    with pytest.raises(ValidationError, match="Available base configs"):
        ImplementationConfig(base="nope")


def test_instrument_spline_statevector():
    """Dynamically named instrument elements (wavelength_cal splines) are kept."""
    element = {"bounds": [-5, 5], "init": 0, "prior_sigma": 100.0, "scale": 1}
    data = legacy_config({"engine_name": "6s"})
    data["forward_model"]["instrument"]["statevector"] = {
        "GROW_FWHM": element,
        "WLSPL_3": element,
        "FWHMSPL_1": element,
    }

    statevector = validate(data).forward_model.instrument.statevector
    assert statevector.get_element_names() == ["FWHMSPL_1", "GROW_FWHM", "WLSPL_3"]
    assert statevector.WLSPL_3.bounds == [-5.0, 5.0]

    data["forward_model"]["instrument"]["statevector"]["WLSPL_3"] = {"bounds": "x"}
    with pytest.raises(ValidationError, match="WLSPL_3"):
        validate(data)


def test_atmosphere_wavelength_file_survives_resolve():
    """An atmosphere-level edit to a resolved config (as wavelength_cal makes) is kept."""
    once = resolve(legacy_config({"engine_name": "6s"}))
    once["forward_model"]["atmosphere"]["wavelength_file"] = Wavelengths
    config = validate(once)
    assert config.forward_model.atmosphere.wavelength_file == Path(Wavelengths)


@pytest.mark.parametrize(
    "emulator, engine",
    [("emulator.jld2", "kernelflows"), ("emulator.h5", "srtmnet")],
)
def test_make_atmosphere_config_emulators(tmp_path, monkeypatch, emulator, engine):
    """template_construction only sets the fields the selected emulator accepts."""
    from isofit.atmosphere.engines import kernel_flows
    from isofit.configs.sections.atmosphere import AtmosphereConfig
    from isofit.utils.template_construction import make_atmosphere_config

    # The KernelFlows bounds check reads the emulator, which is a stub here
    monkeypatch.setattr(kernel_flows, "bounds_check", lambda *args, **kwargs: None)

    for name in (emulator, "emulator_aux.npz", "template.json"):
        (tmp_path / name).touch()

    atmosphere = make_atmosphere_config(
        lut_directory=str(tmp_path),
        modtran_template_path=str(tmp_path / "template.json"),
        emulator_base=str(tmp_path / emulator),
        sixs_path=str(tmp_path),
        h2o_lut_grid=np.array([0.5, 1.0]),
    )
    validated = AtmosphereConfig.model_validate(atmosphere)
    assert validated.engine.name == engine


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
