import json

import pytest
from click.testing import CliRunner

from isofit.configs.loader import load_base_config
from isofit.configs.utils import interpolate
from isofit.configs.utils.interpolate.cli import cli

# Mirrors the ini `env` section, where literal dotted keys sit next to a plain
# key that is a prefix of them
Env = {"srtmnet": "/p", "srtmnet.file": "f.h5", "path.mingw": "/m"}


def test_get_dotted():
    data = {"a": {"b": {"c": 1}}, "env": Env}
    assert interpolate.get_dotted(data, "a.b.c") == 1
    assert interpolate.get_dotted(data, "a.b") == {"c": 1}
    assert interpolate.get_dotted(data, "env.srtmnet") == "/p"
    assert interpolate.get_dotted(data, "env.srtmnet.file") == "f.h5"
    assert interpolate.get_dotted(data, "env.path.mingw") == "/m"

    for key in ("a.x", "a.b.c.d", "env.srtmnet.aux"):
        with pytest.raises(KeyError):
            interpolate.get_dotted(data, key)


def test_set_dotted():
    data = {"a": {"b": 1}, "n": None, "env": dict(Env)}
    interpolate.set_dotted(data, "a.b", 2)
    interpolate.set_dotted(data, "x.y.z", 3)
    interpolate.set_dotted(data, "n.k", 4)
    interpolate.set_dotted(data, "env.srtmnet.file", "g.h5")
    assert data["a"] == {"b": 2}
    assert data["x"] == {"y": {"z": 3}}
    assert data["n"] == {"k": 4}
    assert data["env"]["srtmnet.file"] == "g.h5"
    assert data["env"]["srtmnet"] == "/p"

    with pytest.raises(TypeError, match="not a mapping"):
        interpolate.set_dotted(data, "a.b.c", 5)


def test_merge():
    src = {"s": {"b": None, "l": [3]}, "t": {"u": 1}}
    dst = interpolate.merge({"s": {"a": 1, "b": 2, "l": [1, 2]}, "t": 0}, src)
    assert dst == {"s": {"a": 1, "b": None, "l": [3]}, "t": {"u": 1}}

    # Values from src are copied, not shared
    dst["s"]["l"].append(4)
    dst["t"]["u"] = 2
    assert src == {"s": {"b": None, "l": [3]}, "t": {"u": 1}}


def test_read(tmp_path):
    (tmp_path / "a.yml").write_text("x: 1\nn: null\n")
    (tmp_path / "b.json").write_text('{"x": 1}')
    (tmp_path / "empty.yml").write_text("")
    (tmp_path / "list.yml").write_text("- 1\n")
    (tmp_path / "c.toml").write_text("")

    assert interpolate.read(tmp_path / "a.yml") == {"x": 1, "n": None}
    assert interpolate.read(tmp_path / "b.json") == {"x": 1}
    assert interpolate.read(tmp_path / "empty.yml") == {}
    with pytest.raises(ValueError, match="mapping"):
        interpolate.read(tmp_path / "list.yml")
    with pytest.raises(ValueError, match="Unsupported"):
        interpolate.read(tmp_path / "c.toml")


def test_patch_precedence_and_cross_file(tmp_path):
    (tmp_path / "bases").mkdir()
    (tmp_path / "bases" / "common.yml").write_text(
        "shared: {x: 0, y: 0, z: 0, nested: {a: 0, b: 0}}\n"
    )
    (tmp_path / "run.yml").write_text(
        "A: {^^: [B, C], x: 3, nested: {a: 3}}\n"
        "B: {^^: D, y: 1}\n"
        "C: {y: 2, z: 2}\n"
        "D: {^^: bases/common.yml:shared, x: 1}\n"
    )
    box = interpolate.load(str(tmp_path / "run.yml"), section="A")
    assert box == {"x": 3, "y": 2, "z": 2, "nested": {"a": 3, "b": 0}}


def test_patch_does_not_alias_sections(tmp_path):
    """Two sections inheriting the same parent get independent copies."""
    (tmp_path / "run.yml").write_text("base: {l: [1], s: {a: 1}}\nA: {^^: base}\n")
    full = interpolate.read(tmp_path / "run.yml")
    box = interpolate.patch(full["A"], full)
    box["l"].append(2)
    box["s"]["a"] = 2
    assert full["base"] == {"l": [1], "s": {"a": 1}}


def test_patch_errors(tmp_path):
    (tmp_path / "run.yml").write_text(
        "A: {^^: A}\nB: {^^: nope}\nC: {^^: missing.yml}\n"
    )
    path = str(tmp_path / "run.yml")
    with pytest.raises(ValueError, match="Circular"):
        interpolate.load(path, section="A")
    with pytest.raises(KeyError, match="unknown section"):
        interpolate.load(path, section="B")
    with pytest.raises(FileNotFoundError):
        interpolate.load(path, section="C")
    with pytest.raises(KeyError, match="available sections: A, B, C"):
        interpolate.load(path, section="typo")


def test_override():
    box = {"a": {"b": 1, "keep": 0}, "n": None}
    interpolate.override(
        box, ["--a.b", "[1, 2]", "--n.k", "x", "--new.flag", "--a.none", "None"]
    )
    assert box == {
        "a": {"b": [1, 2], "keep": 0, "none": None},
        "n": {"k": "x"},
        "new": {"flag": True},
    }

    interpolate.override(box, {"a.b": "500"})
    assert box["a"]["b"] == "500"


def test_interpolate():
    box = {
        "n": 4,
        "dir": "/data",
        "env": dict(Env),
        "s": {
            "abs": "${dir}/x",
            "int": "${n}0",
            "rel": "${.c}",
            "c": 7,
            "env": "${env.srtmnet}/${env.srtmnet.file}",
            "lst": ["${dir}", "${n}", {"c": 1, "rel": "${.c}"}],
        },
        "chain": "${s.abs}/y",
    }
    interpolate.interpolate(box)
    assert box["s"] == {
        "abs": "/data/x",
        "int": 40,
        "rel": 7,
        "c": 7,
        "env": "/p/f.h5",
        "lst": ["/data", 4, {"c": 1, "rel": 1}],
    }
    assert box["chain"] == "/data/x/y"


def test_interpolate_errors():
    with pytest.raises(ValueError, match="Circular"):
        interpolate.interpolate({"a": "${b}", "b": "${a}"})
    with pytest.raises(KeyError, match="not found"):
        interpolate.interpolate({"a": "${nope}"})


def test_load_base_config_returns_dicts():
    """The resolved tree is plain dicts/lists and does not mutate overrides."""
    overrides = {"output.dir": "/x"}
    cfg = load_base_config("vswir", overrides=overrides)
    assert overrides == {"output.dir": "/x"}
    assert cfg["forward_model"]["atmosphere"]["lut_path"] == "/x/lut/lut.zarr"

    def walk(node):
        assert type(node) in (dict, list, str, int, float, bool, type(None))
        for val in node.values() if isinstance(node, dict) else node:
            if isinstance(val, (dict, list)):
                walk(val)

    walk(cfg)


def test_cli_copy(tmp_path):
    runner = CliRunner()
    for ext in ("json", "yml"):
        out = tmp_path / f"vswir.{ext}"
        result = runner.invoke(
            cli, ["copy", "vswir", "-o", str(out), "--output.dir", "/scratch"]
        )
        assert result.exit_code == 0, result.output
        data = interpolate.read(out)
        assert data["output"]["dir"] == "/scratch"
        assert data["forward_model"]["atmosphere"]["lut_path"] == (
            "/scratch/lut/lut.zarr"
        )
