import io
import json

import pytest
from rich.console import Console

from isofit.configs import create_new_config
from isofit.configs.loader import list_bases, list_mixins, load
from isofit.configs.utils import browser, interpolate


def render(*items) -> str:
    console = Console(file=io.StringIO(), width=200, color_system=None)
    for item in items:
        console.print(item)
    return console.file.getvalue()


def test_sections(tmp_path):
    """Sections run to the next key, keeping their leading comments."""
    file = tmp_path / "t.yml"
    file.write_text("# header\n\n# about a\na:\n  x: 1\n\nb:  # inline\n  y: 2\n")
    assert browser.sections(file) == {
        "a": (4, 3, "# about a\na:\n  x: 1"),
        "b": (7, 7, "b:  # inline\n  y: 2"),
    }


def test_entries():
    """Every base and mixin is collected with its source and description."""
    items = browser.entries()
    names = [entry.name for entry in items]
    assert names == [*list_bases(), *list_mixins()]

    emit = browser.find(items, "emit")
    assert emit.kind == "mixin" and emit.group == "sensors"
    assert emit.description.startswith("EMIT instrument noise")
    assert emit.source.startswith(f"# {emit.description}\nemit:")  # Its comment
    assert emit.start == emit.line - 1

    vswir = browser.find(items, "vswir")
    assert vswir.kind == "base" and vswir.inherits == ["default", "srtmnet"]
    assert browser.find(items, "mcmc").inherits == ["vswir"]
    assert browser.find(items, "srtmnet").kind == "mixin"
    assert browser.find(items, str(names.index("vswir") + 1)) is vswir
    assert browser.find(items, "nope") is None


def test_render():
    items = browser.entries()
    out = render(browser.overview(items), browser.details(browser.find(items, "oci")))
    assert "mixins/sensors.yml" in out and "PACE OCI" in out
    assert "parametric_noise_file: ${env.data}/oci/oci_noise.txt" in out


def browse(monkeypatch, *lines) -> str:
    """Run the browser on piped input, a key or answer per line."""
    monkeypatch.setattr("sys.stdin", io.StringIO("".join(f"{l}\n" for l in lines)))
    console = Console(file=io.StringIO(), width=200, color_system=None)
    browser.browse(console)
    return console.file.getvalue()


def test_browse_view(monkeypatch):
    """`v` shows a base or mixin, suggesting close names for typos."""
    out = browse(monkeypatch, "v", "oci", "v", "ocii", "v", "", "q")
    assert "oci_noise.txt" in out
    assert "did you mean" in out
    assert out.count("› view") == 3


def test_browse_buttons(monkeypatch):
    """Composing buttons are hidden and refused until a config is composed."""
    out = browse(monkeypatch, "m", "x", "l", "q")
    assert "No config is being composed" in out
    assert "No button 'x'" in out
    assert out.count("Base configs") == 2

    session = browser.Session(Console(), browser.entries())
    assert (
        "compose" in session.buttons().plain and "mixin" not in session.buttons().plain
    )
    session.base, session.mixins = "vswir", ["emit"]
    assert session.buttons().plain.startswith("Composing vswir + emit\n")
    assert all(label in session.buttons().plain for label in browser.Buttons.values())

    # Ends cleanly on end of input
    assert "› quit" not in browse(monkeypatch, "l")


def test_changes():
    """Changed leaves are listed, and new or replaced mappings as a whole."""
    before = {"a": {"b": 1, "c": 2}, "s": {"EOF_1": None}, "x": [1], "gone": 1}
    after = {
        "a": {"b": 3, "c": 2, "n": {"p": 1}},
        "s": {"EOF_1": {"init": 1}},
        "x": [1],
    }
    assert browser.changes(before, after) == [
        (("a", "b"), 1, 3),
        (("a", "n"), browser.Unset, {"p": 1}),
        (("s", "EOF_1"), None, {"init": 1}),
        (("gone",), 1, browser.Unset),
    ]
    assert browser.changes(after, after) == []


def test_layers():
    """Each layer records only what it changed, in order."""
    stack, data = browser.layers(
        "pipeline", ["empirical_line", "analytical_line"], {"segment.size": 10}
    )
    assert [layer.kind for layer in stack] == ["base", "mixin", "mixin", "override"]
    assert (("interpolate", "method"), "analytical", "empirical") in stack[1].changes
    assert (("interpolate", "method"), "empirical", "analytical") in stack[2].changes
    assert stack[3].changes == [(("segment", "size"), 40, 10)]
    assert data["interpolate"]["method"] == "analytical"


def test_compose():
    out = render(browser.compose("pipeline", ["oci"], section="forward_model"))
    assert "pipeline + oci" in out and "Valid config" in out
    assert "SNR  500 → null" in out
    assert "parametric_noise_file:" in out and "input:" not in out

    out = render(browser.compose("vswir", overrides={"implementation.mode": "bogus"}))
    assert "validation error" in out and "bogus" in out


def test_browse_compose(monkeypatch):
    """Compose a base, layer and remove mixins, return to it, and reset."""
    out = browse(
        monkeypatch,
        "c",
        "vswir",  # Start composing
        "m",
        "emit, tanager",  # Layer two mixins
        "m",
        "emit",  # Already layered
        "u",
        "emit",  # Remove a chosen mixin
        "v",
        "av3",  # View an entry, leaving the composition
        "c",  # Back to the composition
        "u",  # Remove the last mixin
        "u",  # Nothing left to remove
        "r",  # Discard it
        "c",
        "",  # Cancel starting another
        "q",
    )
    shown = [
        line for line in out.splitlines() if line.startswith("╭") and "vswir" in line
    ]
    titles = [line.strip("╭─╮ ") for line in shown]
    assert titles == [
        "vswir",
        "vswir + emit + tanager",
        "vswir + emit + tanager",
        "vswir + tanager",
        "vswir + tanager",
        "vswir",
    ]
    assert "Already layered: emit" in out
    assert "AVIRIS-3 inversion windows" in out
    assert "No mixins or overrides to remove" in out
    assert "Composition cleared" in out
    assert "Composing vswir + emit + tanager" in out


def test_browse_compose_errors(monkeypatch):
    """Unknown or wrong-kind selections leave the composition unchanged."""
    out = browse(monkeypatch, "c", "emit", "c", "vswir", "m", "vswir", "m", "nope", "q")
    assert "Unknown base 'emit'" in out
    assert "Unknown mixin 'vswir'" in out
    assert "Unknown mixin 'nope'" in out
    assert "Composing vswir\n" in out


def test_parse():
    """Overrides typed at the prompt parse like those on the command line."""
    assert browser.parse("output.dir /x --implementation.n_cores=8") == {
        "output.dir": "/x",
        "implementation.n_cores": 8,
    }
    assert browser.parse('a=[1, 2] b {"c": 1} d "e f"') == {
        "a": [1, 2],
        "b": {"c": 1},
        "d": "e f",
    }
    for bad in ("output.dir", 'a "b', "a [1"):
        with pytest.raises(ValueError):
            browser.parse(bad)


def test_save(tmp_path):
    """The composed config is written as YAML or JSON, and loads back."""
    overrides = {"output.dir": "/scratch", "implementation.n_cores": 8}

    file = browser.save(tmp_path / "out" / "run.yml", "vswir", ["emit"], overrides)
    data = interpolate.read(file)
    assert data["output"]["dir"] == "/scratch"
    assert data["output"]["estimated_state_file"] == "/scratch/estimated_state"
    assert data["implementation"]["n_cores"] == 8
    assert "windows: [[" in file.read_text()  # Lists in flow style
    assert create_new_config(file).implementation.n_cores == 8

    # Raw keeps the references, so they still follow the config
    file = browser.save(
        tmp_path / "run.json", "vswir", ["emit"], overrides, interp=False
    )
    data = json.loads(file.read_text())
    assert data["output"]["estimated_state_file"] == "${output.dir}/estimated_state"
    assert load(str(file), interp=True)["output"]["estimated_state_file"] == (
        "/scratch/estimated_state"
    )

    with pytest.raises(ValueError, match="Unsupported file extension"):
        browser.save(tmp_path / "run.txt", "vswir")


def test_browse_override_save(monkeypatch, tmp_path):
    """Overrides are applied and undone in order, and the composition saved."""
    file = tmp_path / "run.yml"
    out = browse(
        monkeypatch,
        "o",
        "output.dir /x",  # Start without a composition
        "c",
        "vswir",
        "o",
        "output.dir /scratch implementation.n_cores=8",
        "m",
        "emit",
        "o",
        "output.dir /scratch2",  # Replaces the earlier value
        "o",
        "nope",  # No value
        "o",
        "output.dir.x 1",  # Can't be applied
        "s",
        f"{file}",
        "s",
        f"{tmp_path / 'raw.json'} raw",
        "s",
        f"{tmp_path / 'run.txt'}",
        "u",
        "",  # The last added: output.dir
        "u",
        "emit",
        "u",  # The only one left: implementation.n_cores
        "s",
        "",  # As last saved: raw.json, raw, without confirmation
        "s",
        f"{file}",
        "y",  # Overwriting an earlier save is confirmed
        "q",
    )
    assert "No config is being composed" in out
    assert "Composing vswir  --output.dir /scratch  --implementation.n_cores 8" in out
    assert (
        "Composing vswir + emit  --implementation.n_cores 8  --output.dir /scratch2"
        in out
    )
    assert "Override 'nope' has no value" in out
    assert "Could not apply the overrides" in out
    assert "Unsupported file extension '.txt'" in out
    assert "Removed --output.dir" in out and "Removed emit" in out
    assert "Removed --implementation.n_cores" in out
    assert out.count("✔ Saved the raw config") == 2
    assert out.count("✔ Saved the interpolated config") == 2

    assert interpolate.read(file) == load("vswir", interp=True)
    raw = json.loads((tmp_path / "raw.json").read_text())
    assert raw == load("vswir")


def test_browse_save_overwrite(monkeypatch, tmp_path):
    """An existing file is only overwritten when confirmed."""
    file = tmp_path / "run.yml"
    file.write_text("keep: true\n")

    out = browse(monkeypatch, "c", "vswir", "s", str(file), "n", "q")
    assert "Not saved" in out and file.read_text() == "keep: true\n"

    out = browse(monkeypatch, "c", "vswir", "s", str(file), "y", "q")
    assert "✔ Saved" in out and interpolate.read(file)["output"]["dir"]


def test_cli_save(tmp_path):
    """`browse BASE -o FILE` saves the composition with its overrides."""
    from click.testing import CliRunner

    from isofit.configs.utils.interpolate.cli import cli

    file = tmp_path / "run.yml"
    args = [
        "browse",
        "vswir",
        "-m",
        "emit",
        "--output.dir",
        "/scratch",
        "-o",
        str(file),
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert interpolate.read(file)["output"]["dir"] == "/scratch"

    result = CliRunner().invoke(cli, ["browse", "vswir", "-o", str(tmp_path / "x.txt")])
    assert result.exit_code == 1


def test_keylines():
    """Each line of the shown config maps to its key, wrapped lines included."""
    assert browser.keylines("a:\n  b: 1\n  c: [1,\n    2]\nd: x\n") == {
        1: ("a",),
        2: ("a", "b"),
        3: ("a", "c"),
        4: ("a", "c"),
        5: ("d",),
    }

    text = browser.dump(load("pipeline", interp=True))
    lines = browser.keylines(text)
    assert set(lines) == set(range(1, len(text.splitlines()) + 1))
    for number, line in enumerate(text.splitlines(), 1):
        assert line.strip().startswith(f"{lines[number][-1]}:") or lines[number] == (
            "cleanup",
            "remove",
        )


def test_literal():
    """Values typed for a line parse as literals, YAML scalars, or text."""
    assert browser.literal("8") == 8
    assert browser.literal("[1, 2]") == [1, 2]
    assert browser.literal("true") is True and browser.literal("null") is None
    assert browser.literal('"/my scratch"') == "/my scratch"
    assert browser.literal("two words") == "two words"


def line(text: str, key: str) -> int:
    """The line of `key` in the shown config `text`."""
    lines = browser.keylines(text)
    return next(n for n, path in lines.items() if ".".join(path) == key)


def test_browse_override_lines(monkeypatch):
    """`o` takes line numbers of the shown composition, then their values."""
    text = browser.dump(load("vswir", interp=True))
    cores = line(text, "implementation.n_cores")
    mode = line(text, "implementation.mode")
    level = line(text, "log.level")
    windows = line(text, "implementation.inversion.windows")
    first = min(mode, cores)

    out = browse(
        monkeypatch,
        "c",
        "vswir",
        "o",
        f"{cores}",
        "8",  # One line
        "o",
        f"{level}, {windows}",
        "DEBUG",
        "[[400, 1300], [1450, 2400]]",  # Several
        "o",
        f"{first}-{first + 1}",
        "",
        "",  # A range, both kept
        "o",
        "99999",  # Not a line
        "o",
        f"{mode}",
        "[1, 2",  # Unclosed
        "u",
        "",
        "q",
    )
    assert "Composing vswir  --implementation.n_cores 8" in out
    assert (
        "--implementation.n_cores 8  --log.level DEBUG  "
        "--implementation.inversion.windows [[400, 1300], [1450, 2400]]"
    ) in out
    assert "implementation.n_cores (now null; new value, blank to keep)" in out
    assert "No overrides set" in out
    assert "Line 99999 is not in the composition" in out
    assert "Unclosed bracket" in out
    assert "Removed --implementation.inversion.windows" in out


def test_session_keys():
    """Line selections resolve to keys, deduplicated, and need a shown config."""
    session = browser.Session(Console(file=io.StringIO()), browser.entries())
    session.base = "vswir"
    session.show()

    text = browser.dump(load("vswir", interp=True))
    remove = line(text, "cleanup.remove")
    assert session.keys("output.dir /x") is None
    assert session.keys(f"{remove}") == ["cleanup.remove"]
    assert session.keys("1 2,3-4") == [
        "input",
        "input.sensor",
        "input.fid",
        "input.measured_radiance_file",
    ]
    with pytest.raises(ValueError, match="not in the composition"):
        session.keys("0")


def test_layers_overrides():
    """Each override is its own layer, after the mixins, in order."""
    stack, _ = browser.layers(
        "vswir", ["oci"], {"output.dir": "/x", "log.level": "INFO"}
    )
    assert [layer.ref for layer in stack] == [
        "vswir",
        "oci",
        "--output.dir",
        "--log.level",
    ]
    assert stack[2].changes == [(("output", "dir"), "./isofit-output/", "/x")]
    assert stack[3].changes == []  # Already INFO


def test_changes_table():
    """The table lists each mixin's and override's changes, and no-op overrides."""
    overrides = {
        "output.dir": "/scratch",
        "forward_model.instrument.SNR": 300,
        "log.level": "INFO",
    }
    out = render(browser.changes_table("vswir", ["oci"], overrides))
    assert "Changes vswir + oci + 3 overrides" in out
    rows = [" ".join(line.strip("│ ").split()) for line in out.splitlines()]
    assert (
        "oci forward_model.atmosphere.irradiance_file null ${env.data}/oci/tsis_f0_0p1.txt"
        in rows
    )
    assert "forward_model.instrument.SNR 500 null" in rows
    assert "override output.dir ./isofit-output/ /scratch" in rows
    assert "override forward_model.instrument.SNR null 300" in rows  # After the mixin
    assert "override log.level no change" in rows

    # Within a section
    out = render(browser.changes_table("vswir", ["oci"], overrides, section="output"))
    assert "output.dir" in out and "SNR" not in out and "log.level" not in out

    assert "No changes on top of vswir" in render(browser.changes_table("vswir"))


def test_browse_diff(monkeypatch):
    """`d` shows the changes table of the active composition."""
    out = browse(
        monkeypatch, "d", "c", "vswir", "d", "m", "oci", "o", "output.dir /x", "d", "q"
    )
    assert "No config is being composed" in out
    assert "No changes on top of vswir" in out
    assert "Changes vswir + oci + 1 override" in out
    assert out.count("› diff") == 3


def test_references():
    """Each reference resolves on its own, noting unset keys and failures."""
    found = {
        key: rest
        for key, *rest in browser.references("pipeline", [], {"output.dir": "/x"})
    }
    written, resolved, unset = found["output.estimated_state_file"]
    assert written == "${output.dir}/output/${input.fid}_state"
    assert resolved == "/x/output/None_state" and unset == ["input.fid"]
    assert found["cleanup.remove[0]"][1] == "/x/lut_*/*r_k"  # List items
    assert found["log.file"][1] == "/x/isofit.log"  # Relative references

    found = browser.references(
        "pipeline", [], {"input.fid": "f", "output.dir": "${nope}"}, section="output"
    )
    assert {key for key, *_ in found} == {
        "output.dir",
        "output.estimated_state_file",
        "output.estimated_reflectance_file",
        "output.posterior_uncertainty_file",
    }
    assert all(isinstance(resolved, KeyError) for _, _, resolved, _ in found)


def test_references_table():
    out = render(
        browser.references_table("pipeline", [], {"output.dir": "/x"}, section="output")
    )
    assert "Interpolation pipeline + 1 override" in out
    assert "/x/output/None_state" in out and "⚠ until input.fid is set" in out
    assert "3 references, 3 pending unset keys" in out

    out = render(
        browser.references_table(
            "vswir", [], {"output.dir": "${nope}"}, section="output"
        )
    )
    assert "✘ Interpolation reference" in out and "4 unresolved" in out

    assert "No ${...} references" in render(
        browser.references_table("default", section="implementation")
    )


def test_browse_interpolate(monkeypatch):
    """`i` previews the references of the composition, optionally of a section."""
    out = browse(
        monkeypatch,
        "i",
        "c",
        "pipeline",
        "o",
        "input.fid emit2022",
        "i",
        "output",
        "i",
        "nope",
        "i",
        "",
        "q",
    )
    assert "No config is being composed" in out
    assert "/output/emit2022_state" in out
    assert "Section 'nope' is not in the config" in out
    assert out.count("Interpolation pipeline + 1 override") == 2


def test_compose_as_written(monkeypatch):
    """The composition shows ${...} references as written, with lines to match."""
    out = render(
        browser.compose("vswir", overrides={"output.dir": "/scratch"}, section="output")
    )
    assert "estimated_state_file: ${output.dir}/estimated_state" in out
    assert "/scratch/estimated_state" not in out

    out = render(
        browser.compose("vswir", overrides={"output.dir": "/scratch"}, interp=True)
    )
    assert "estimated_state_file: /scratch/estimated_state" in out

    # The session's line numbers index the config as shown
    session = browser.Session(
        Console(file=io.StringIO()), browser.entries(), base="vswir"
    )
    session.show()
    shown = session.console.file.getvalue()
    text = browser.dump(load("vswir"))
    number = line(text, "forward_model.atmosphere.lut_path")
    assert f"{number} " in shown and "lut_path: ${output.dir}/lut/lut.zarr" in shown
    assert session.lines[number] == ("forward_model", "atmosphere", "lut_path")


def test_cli_browse_interpolate():
    """`browse BASE -i` shows the config as written plus the preview."""
    from click.testing import CliRunner

    from isofit.configs.utils.interpolate.cli import cli

    result = CliRunner().invoke(cli, ["browse", "vswir", "-s", "output", "-i"])
    assert result.exit_code == 0, result.output
    assert "estimated_state_file: ${output.dir}/estimated_state" in result.output
    assert "Interpolation vswir" in result.output
