"""
Rich browser for the base configs and mixins shipped under
``isofit/configs/templates/``.

:func:`overview` renders every base and mixin as tables grouped by template
file, and :func:`details` renders a single one: its description, where it is
defined, and its YAML exactly as written in the file. :func:`compose` renders
the full config of a base with mixins layered on top, along with what each
mixin changed. :func:`browse` combines them into a prompt loop, where a single
number or name shows that entry and a base followed by mixins composes them:

    $ isofit config browse
    $ isofit config browse emit
    $ isofit config browse pipeline -m emit -m empirical_line
"""

import copy
import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import ValidationError
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from isofit.configs import Config, loader
from isofit.configs.utils import interpolate
from isofit.configs.utils.console import errors_table

# A top-level key of a YAML template, e.g. `vswir:`
TopLevel = re.compile(r"^([A-Za-z_][\w-]*)\s*:")

Styles = {"base": "cyan", "mixin": "green"}
Quit = {"q", "quit", "exit"}
List = {"l", "ls", "list"}

# Separates the names of a composition, e.g. `pipeline + emit, empirical_line`
Separators = re.compile(r"[\s,+]+")

# A key absent from a config, as distinct from one set to `null`
Unset = object()


@dataclass
class Entry:
    """
    A base config or mixin, as defined in its template file.
    """

    name: str
    kind: str
    file: Path
    line: int
    start: int
    source: str
    description: Optional[str] = None
    inherits: list[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        """The template file, relative to the templates directory."""
        return str(self.file.relative_to(loader.ConfigsDir))

    @property
    def group(self) -> str:
        """The template file's name without its extension, e.g. ``sensors``."""
        return self.file.stem


def sections(file: Path) -> dict[str, tuple[int, int, str]]:
    """
    Split a YAML template into the source text of each top-level section.

    A section runs from its key to the next top-level key. Comments directly
    above a key, with no blank line in between, belong to that key's section.

    Parameters
    ----------
    file : pathlib.Path
        The template file.

    Returns
    -------
    dict[str, tuple[int, int, str]]
        Mapping of section name to the 1-indexed line of its key, the 1-indexed
        line its source starts on, and the source.
    """
    lines = file.read_text().splitlines()
    keys = [(i, m[1]) for i, line in enumerate(lines) if (m := TopLevel.match(line))]

    found = {}
    for n, (key, name) in enumerate(keys):
        start = key
        while start > 0 and lines[start - 1].startswith("#"):
            start -= 1

        end = keys[n + 1][0] if n + 1 < len(keys) else len(lines)
        while end > key + 1 and (
            not lines[end - 1].strip() or lines[end - 1].startswith("#")
        ):
            end -= 1

        found[name] = (key + 1, start + 1, "\n".join(lines[start:end]))

    return found


def short(ref: str, file: Path) -> str:
    """
    Shorten a ``^^`` reference to a template section to the section's name,
    e.g. ``../mixins/engines.yml:srtmnet`` to ``srtmnet``.

    References to other files, or to a section whose name is shadowed by
    another template, are left as written.
    """
    name, ext = ref, next(
        (ext for ext in interpolate.Loaders if f"{ext}:" in ref), None
    )
    path = file
    if ext:
        target, name = ref.split(f"{ext}:", 1)
        path = (file.parent / f"{target}{ext}").resolve()

    index = {**loader.list_mixins(), **loader.list_bases()}
    return name if index.get(name) == path else ref


def entries() -> list[Entry]:
    """
    Collect every base config, then every mixin, in template file order.

    Returns
    -------
    list[Entry]
        The bases and mixins.
    """
    items = []
    for kind, index in (("base", loader.list_bases()), ("mixin", loader.list_mixins())):
        for name, file in index.items():
            data = interpolate.read(file)[name] or {}
            if name in (source := sections(file)):
                line, start, text = source[name]
            else:
                # Not a YAML key found by line, e.g. a JSON template
                line, start, text = 1, 1, yaml.safe_dump({name: data}, sort_keys=False)

            inherits = data.get("^^") or []
            if isinstance(inherits, str):
                inherits = [inherits]
            inherits = [short(ref, file) for ref in inherits]

            items.append(
                Entry(
                    name=name,
                    kind=kind,
                    file=file,
                    line=line,
                    start=start,
                    source=text,
                    description=data.get("description"),
                    inherits=inherits,
                )
            )

    return items


def overview(items: list[Entry]) -> Group:
    """
    Render the bases and mixins as numbered tables, grouped by template file.

    Entries are numbered in the order given, so the numbers match those
    accepted by :func:`browse`.

    Parameters
    ----------
    items : list[Entry]
        The entries to list, e.g. from :func:`entries`.

    Returns
    -------
    rich.console.Group
        A panel of base configs followed by a panel of mixins.
    """
    panels = []
    for kind, title in (("base", "Base configs"), ("mixin", "Mixins")):
        style = Styles[kind]
        table = Table(box=box.SIMPLE_HEAD, expand=True, header_style=f"bold {style}")
        table.add_column("#", justify="right", style="dim", no_wrap=True)
        table.add_column("Name", style=f"bold {style}", no_wrap=True)
        if kind == "base":
            table.add_column("Inherits", style="magenta", no_wrap=True)
        table.add_column("Description", ratio=1)

        group = None
        for number, entry in enumerate(items, 1):
            if entry.kind != kind:
                continue

            if entry.file != group:
                if group is not None:
                    table.add_section()
                group = entry.file
                heading = Text(entry.path, style="italic yellow")
                table.add_row("", heading, *[""] * (len(table.columns) - 2))

            row = [str(number), entry.name]
            if kind == "base":
                row.append(", ".join(entry.inherits) or Text("-", style="dim"))
            row.append(entry.description or Text("No description", style="dim italic"))
            table.add_row(*row)

        panels.append(
            Panel(table, title=f"[bold]{title}[/]", border_style=style, box=box.ROUNDED)
        )

    return Group(*panels)


def details(entry: Entry) -> Panel:
    """
    Render a single base or mixin: its description, where it is defined, and
    its YAML as written in the template file.

    Parameters
    ----------
    entry : Entry
        The base or mixin.

    Returns
    -------
    rich.panel.Panel
        The rendered entry.
    """
    style = Styles[entry.kind]

    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold", no_wrap=True)
    meta.add_column()
    meta.add_row(
        "Description",
        entry.description or Text("No description", style="dim italic"),
    )
    if entry.kind == "base":
        meta.add_row(
            "Inherits",
            (
                Text(", ".join(entry.inherits), style="magenta")
                if entry.inherits
                else Text("Nothing", style="dim")
            ),
        )
    else:
        meta.add_row("Group", Text(entry.group, style="yellow"))
    meta.add_row("File", Text(f"{entry.path}:{entry.line}", style="yellow"))

    source = Syntax(
        entry.source,
        "yaml",
        line_numbers=True,
        start_line=entry.start,
        word_wrap=True,
        background_color="default",
    )

    return Panel(
        Group(meta, Rule(style="dim"), source),
        title=f"[bold {style}]{entry.name}[/]",
        subtitle=f"[{style}]{entry.kind}[/]",
        border_style=style,
        box=box.ROUNDED,
    )


def find(items: list[Entry], choice: str) -> Optional[Entry]:
    """
    Select an entry by its number in :func:`overview` or by name.

    A base takes precedence over a mixin of the same name, matching
    :func:`isofit.configs.loader.resolve`.
    """
    if choice.isdigit() and 1 <= int(choice) <= len(items):
        return items[int(choice) - 1]

    return next((entry for entry in items if entry.name == choice), None)


def flatten(data: Any, path: tuple = ()) -> dict[tuple, Any]:
    """
    Flatten nested mappings into ``{path: value}`` leaves, keyed by tuples of
    keys.

    Lists are leaves, as are empty mappings so their key is not lost.
    """
    if not isinstance(data, dict) or not data:
        return {path: data}

    flat = {}
    for key, val in data.items():
        flat.update(flatten(val, (*path, key)))

    return flat


def lookup(data: dict, path: tuple) -> Any:
    """The value at ``path`` in nested mappings, or :data:`Unset`."""
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return Unset
        data = data[key]

    return data


def changes(before: dict, after: dict) -> list[tuple[tuple, Any, Any]]:
    """
    List the leaves that differ between two configs.

    A mapping that is new in ``after``, e.g. a statevector element a mixin adds,
    is reported as one change rather than one per leaf, as is a value replaced
    by a mapping.

    Parameters
    ----------
    before, after : dict
        The configs to compare.

    Returns
    -------
    list[tuple[tuple, Any, Any]]
        The path of keys, the value before (:data:`Unset` if absent), and the
        value after (:data:`Unset` if removed), in the order of ``after``.
    """
    old, new = flatten(before), flatten(after)

    found, added = [], set()
    for path, val in new.items():
        if path in old:
            if old[path] != val:
                found.append((path, old[path], val))
            continue

        # Report the outermost key that was missing from `before`, or was not
        # a mapping there (e.g. `EOF_1: null`), as a whole
        head = next(
            path[:n]
            for n in range(1, len(path) + 1)
            if not isinstance(lookup(before, path[:n]), dict)
        )
        if head not in added:
            added.add(head)
            found.append((head, lookup(before, head), lookup(after, head)))

    for path, val in old.items():
        if path not in new and not any(path[: len(a)] == a for a in added):
            found.append((path, val, Unset))

    return found


@dataclass
class Layer:
    """
    One layer of a composed config: the base, a mixin, or the overrides.
    """

    ref: str
    kind: str
    description: Optional[str]
    changes: list[tuple[tuple, Any, Any]]


def layers(
    base: str, mixins: list[str], overrides: Optional[dict] = None
) -> tuple[list[Layer], dict]:
    """
    Layer mixins and overrides on top of a base, as :func:`loader.load` does,
    recording what each layer changes.

    The values compared are as written in the templates, before ``${...}``
    interpolation.

    Parameters
    ----------
    base : str
        A config reference for the base; see :func:`loader.resolve`.
    mixins : list[str]
        Config references layered on top of ``base``, in order.
    overrides : dict, optional
        Dotted-key overrides applied last.

    Returns
    -------
    tuple[list[Layer], dict]
        The layers, and the composed config before interpolation.
    """
    data = loader.load_layer(base)
    stack = [Layer(base, "base", loader.describe(base), [])]

    for ref in mixins:
        before = copy.deepcopy(data)
        loader.merge_layer(data, loader.load_layer(ref, inherit=False))
        stack.append(Layer(ref, "mixin", loader.describe(ref), changes(before, data)))

    if overrides:
        before = copy.deepcopy(data)
        interpolate.override(data, overrides)
        stack.append(Layer("overrides", "override", None, changes(before, data)))

    return stack, data


class Dumper(yaml.SafeDumper):
    """
    Dumps mappings in block style and lists in flow style, e.g. ``[1, 2]``, as
    the templates are written.
    """


Dumper.add_representer(
    list,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    ),
)


def show(value: Any) -> Text:
    """Render a config value compactly for a table cell."""
    if value is Unset:
        return Text("unset", style="dim italic")
    if value is None:
        return Text("null", style="dim")
    if isinstance(value, str):
        return Text(value, style="yellow" if "${" in value else "")

    return Text(json.dumps(value), style="bright_blue")


def changes_tree(layer: Layer, rows: list[tuple[tuple, Any, Any]], style: str) -> Tree:
    """
    Render a layer's changes as a tree of their keys, with shared parent keys
    merged, e.g.::

        emit
        └── forward_model
            └── instrument
                ├── SNR  500 → null
                └── parametric_noise_file  null → ${env.data}/emit_noise.txt
    """
    tree = Tree(Text(layer.ref, style=f"bold {style}"), guide_style="dim")
    nodes = {(): tree}
    for path, old, new in rows:
        for n in range(1, len(path)):
            if path[:n] not in nodes:
                nodes[path[:n]] = nodes[path[: n - 1]].add(Text(str(path[n - 1])))

        label = Text.assemble(
            (str(path[-1]), "bold"), "  ", show(old), (" → ", "dim"), show(new)
        )
        nodes[path[:-1]].add(label)

    return tree


def compose(
    base: str,
    mixins: Optional[list[str]] = None,
    overrides: Optional[dict] = None,
    section: Optional[str] = None,
    interp: bool = True,
) -> Group:
    """
    Render the full config of a base with mixins layered on top.

    Shows the layer stack with each layer's description, a tree of what each
    mixin (and the overrides) changed, whether the interpolated result
    validates as a :class:`~isofit.configs.Config`, and the composed config
    itself.

    Parameters
    ----------
    base : str
        A config reference for the base; see :func:`loader.resolve`.
    mixins : list[str], optional
        Config references layered on top of ``base``, in order.
    overrides : dict, optional
        Dotted-key overrides applied last.
    section : str, optional
        Dotted path of the subsection to show, e.g. ``"forward_model"``. The
        whole config is still validated.
    interp : bool, default=True
        Show the config with ``${...}`` references resolved. Validation is
        always of the resolved config.

    Returns
    -------
    rich.console.Group
        The rendered composition.

    Raises
    ------
    ValueError
        If ``base`` or a mixin is not a known config reference.
    """
    mixins = list(mixins or [])
    stack, _ = layers(base, mixins, overrides)

    # The stack, lowest priority first
    stacked = Table.grid(padding=(0, 1))
    stacked.add_column(justify="right", style="dim", no_wrap=True)
    stacked.add_column(no_wrap=True)
    stacked.add_column(style="dim")
    stacked.add_column()
    styles = {**Styles, "override": "magenta"}
    for n, layer in enumerate(stack):
        mark = "" if n == 0 else "+"
        count = "" if n == 0 else f"{len(layer.changes)} change(s)"
        name = Text(layer.ref, style=f"bold {styles[layer.kind]}")
        stacked.add_row(mark, name, count, layer.description or "")

    # What each layer changed, as a tree of keys per layer
    prefix = tuple(section.split(".")) if section else ()
    diffs = []
    for layer in stack[1:]:
        rows = [c for c in layer.changes if c[0][: len(prefix)] == prefix]
        if rows:
            diffs.append(changes_tree(layer, rows, styles[layer.kind]))

    # The composed config, validated once interpolated
    raw = loader.load(base, mixin=mixins, overrides=overrides)
    data, error = raw, None
    try:
        resolved = loader.load(base, mixin=mixins, overrides=overrides, interp=True)
    except (KeyError, ValueError) as e:
        error = e
        status = Panel(
            Text(f"Interpolation failed: {e}", style="red"),
            border_style="red",
            box=box.ROUNDED,
        )
    else:
        data = resolved if interp else raw
        try:
            Config.model_validate(resolved)
            status = Text("✔ Valid config", style="bold green")
        except ValidationError as e:
            status = errors_table(e)

    try:
        shown = interpolate.get_dotted(data, section) if section else data
    except (KeyError, IndexError, TypeError):
        shown = None
        status = Group(
            status, Text(f"Section {section!r} is not in the config", style="red")
        )

    yml = yaml.dump(shown, Dumper=Dumper, sort_keys=False, width=100)
    source = Syntax(yml.rstrip(), "yaml", line_numbers=True, background_color="default")

    title = " + ".join([base, *mixins]) + (" + overrides" if overrides else "")
    parts = [
        Panel(stacked, title="[bold]Layers[/]", border_style="blue", box=box.ROUNDED),
    ]
    if diffs:
        parts.append(
            Panel(
                Group(*diffs),
                title="[bold]Changes[/]",
                border_style="blue",
                box=box.ROUNDED,
            )
        )
    parts += [
        status,
        Panel(
            source,
            title=f"[bold]{title}[/]",
            subtitle=section or ("interpolated" if interp and not error else "raw"),
            border_style="cyan",
            box=box.ROUNDED,
        ),
    ]
    return Group(*parts)


def unknown(choice: str, names: list[str], what: str = "base or mixin") -> str:
    """A message for an unknown selection, suggesting close names."""
    msg = f"[red]Unknown {what} [bold]{choice!r}[/][/]"
    if close := difflib.get_close_matches(choice, names, n=3):
        msg += f"[red], did you mean:[/] {', '.join(close)}"

    return msg


def pick(items: list[Entry], choice: str, kind: str) -> str:
    """
    Select a ``kind`` of entry by number or name for :func:`compose`.

    Raises
    ------
    ValueError
        With a message from :func:`unknown` if there is no such entry.
    """
    entry = find(items, choice)
    if entry is None or entry.kind != kind:
        names = [e.name for e in items if e.kind == kind]
        raise ValueError(unknown(choice, names, kind))

    return entry.name


def select(items: list[Entry], choice: str) -> Union[Panel, Group, str]:
    """
    Render a selection from the :func:`browse` prompt.

    A single number or name renders that entry with :func:`details`. A base
    followed by mixins, separated by spaces, commas, or ``+``, renders their
    composition with :func:`compose`, as does a base with a trailing ``+``
    (e.g. ``vswir +``) to show its full config.

    Parameters
    ----------
    items : list[Entry]
        The entries, numbered as in :func:`overview`.
    choice : str
        The selection, e.g. ``"emit"``, ``"17"``, or ``"pipeline + emit"``.

    Returns
    -------
    rich.panel.Panel or rich.console.Group or str
        The rendered entry or composition, or a message if a name is unknown.
    """
    base, *mixins = Separators.split(choice.strip().strip(",")) or [""]
    mixins = [m for m in mixins if m]

    if not mixins and "+" not in choice:
        if entry := find(items, base):
            return details(entry)
        return unknown(base, [entry.name for entry in items])

    try:
        return compose(
            pick(items, base, "base"), [pick(items, m, "mixin") for m in mixins]
        )
    except ValueError as e:
        return str(e)


def browse(console: Optional[Console] = None) -> None:
    """
    Interactively browse the bases and mixins.

    Lists them with :func:`overview`, then prompts for a selection and renders
    it with :func:`select`, until ``q`` or end of input: a number or name shows
    that entry, and a base followed by mixins (e.g. ``pipeline + emit``) shows
    the composed config. ``l`` lists them again.

    Parameters
    ----------
    console : rich.console.Console, optional
        Console to render to and read from. Defaults to a new console.
    """
    console = console or Console()
    items = entries()

    console.print(overview(items))

    prompt = (
        "[bold]Select a base or mixin[/] [dim](number or name, or "
        "[cyan]base + mixin + ...[/] to compose, [cyan]l[/] to list, "
        "[cyan]q[/] to quit)[/]"
    )
    while True:
        try:
            choice = Prompt.ask(prompt, console=console).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not choice:
            continue
        if choice.lower() in Quit:
            break
        if choice.lower() in List:
            console.print(overview(items))
            continue

        console.print(select(items, choice))
