"""
Rich browser for the base configs and mixins shipped under
``isofit/configs/templates/``.

:func:`overview` renders every base and mixin as tables grouped by template
file, and :func:`details` renders a single one: its description, where it is
defined, and its YAML exactly as written in the file. :func:`compose` renders
the full config of a base with mixins and overrides layered on top, along with
what each changed, and :func:`save` writes it to a file. :func:`browse` combines
them into an interactive session driven by single-key buttons (see
:data:`Buttons`), composing a config one mixin or override at a time:

    $ isofit config browse
    $ isofit config browse emit
    $ isofit config browse pipeline -m emit -m empirical_line
    $ isofit config browse vswir -m emit --output.dir /scratch -o run.yml
"""

import copy
import difflib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import ValidationError
from rich import box
from rich.console import Console, Group
from rich.markup import escape
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

# The keys of the interactive browser, and the label of each
Buttons = {
    "c": "compose",
    "d": "diff",
    "i": "interpolate",
    "m": "mixin",
    "o": "override",
    "u": "undo",
    "r": "reset",
    "s": "save",
    "v": "view",
    "l": "list",
    "q": "quit",
}

# Buttons that only apply while a config is being composed
Composing = ("d", "i", "m", "o", "u", "r", "s")

# The formats :func:`save` writes, by file extension
Formats = (".yml", ".yaml", ".json")

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


def overview(items: list[Entry], kinds: tuple[str, ...] = ("base", "mixin")) -> Group:
    """
    Render the bases and mixins as numbered tables, grouped by template file.

    Entries are numbered in the order given, so the numbers match those
    accepted by :func:`find`, whichever ``kinds`` are shown.

    Parameters
    ----------
    items : list[Entry]
        The entries to list, e.g. from :func:`entries`.
    kinds : tuple[str, ...], default=("base", "mixin")
        The kinds of entry to show.

    Returns
    -------
    rich.console.Group
        A panel of base configs and/or a panel of mixins.
    """
    panels = []
    titles = {"base": "Base configs", "mixin": "Mixins"}
    for kind, title in ((kind, titles[kind]) for kind in kinds):
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
        Dotted-key overrides applied last, in order, each as its own layer.

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

    # Each override is its own layer, so its changes are attributed to it
    for key, val in (overrides or {}).items():
        before = copy.deepcopy(data)
        interpolate.override(data, {key: val})
        stack.append(Layer(f"--{key}", "override", None, changes(before, data)))

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


def dump(data: Any) -> str:
    """Dump a config as YAML in the style of the templates (see :class:`Dumper`)."""
    return yaml.dump(data, Dumper=Dumper, sort_keys=False, width=100)


def keylines(text: str, prefix: tuple = ()) -> dict[int, tuple]:
    """
    Map each line of a YAML config, as numbered in :func:`compose`, to the
    path of the key it belongs to.

    A line belongs to the innermost key whose value spans it, so the lines a
    long list wraps onto map to that list's key.

    Parameters
    ----------
    text : str
        The YAML, e.g. from :func:`dump`.
    prefix : tuple, optional
        Path the YAML is nested under, e.g. its section in the full config.

    Returns
    -------
    dict[int, tuple]
        Mapping of 1-indexed line number to the path of its key.

    Examples
    --------
    >>> keylines("a:\\n  b: 1\\n  c: [1,\\n    2]\\n")
    {1: ('a',), 2: ('a', 'b'), 3: ('a', 'c'), 4: ('a', 'c')}
    """
    found = {}

    def walk(node, path):
        if not isinstance(node, yaml.MappingNode):
            return
        for key, value in node.value:
            sub = (*path, key.value)
            end = value.end_mark.line - (value.end_mark.column == 0)
            for line in range(key.start_mark.line, max(end, key.start_mark.line) + 1):
                found[line + 1] = sub
            walk(value, sub)

    walk(yaml.compose(text, Loader=yaml.SafeLoader), prefix)
    return found


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


def changes_table(
    base: str,
    mixins: Optional[list[str]] = None,
    overrides: Optional[dict] = None,
    section: Optional[str] = None,
) -> Union[Panel, Text]:
    """
    Render what each mixin and override changes on top of a base, as a table
    with a row per changed key: the layer, the key, and its value before and
    after. Values are as written in the templates, before ``${...}``
    interpolation.

    Parameters
    ----------
    base : str
        A config reference for the base; see :func:`loader.resolve`.
    mixins : list[str], optional
        Config references layered on top of ``base``, in order.
    overrides : dict, optional
        Dotted-key overrides applied last, each as its own layer.
    section : str, optional
        Dotted path of a section to only show the changes within.

    Returns
    -------
    rich.panel.Panel or rich.text.Text
        The table, or a note if nothing changes.
    """
    stack, _ = layers(base, list(mixins or []), overrides)
    prefix = tuple(section.split(".")) if section else ()
    styles = {**Styles, "override": "magenta"}

    table = Table(box=box.SIMPLE_HEAD, expand=True, header_style="bold")
    table.add_column("Layer", no_wrap=True)
    table.add_column("Key", style="bold", ratio=4, overflow="fold")
    table.add_column("Before", ratio=3, overflow="fold")
    table.add_column("After", ratio=3, overflow="fold")

    for layer in stack[1:]:
        rows = [c for c in layer.changes if c[0][: len(prefix)] == prefix]
        if layer.kind == "override":
            # Named by the key column, which says which override it is
            name = Text("override", style=f"bold {styles['override']}")
            if not rows and (not prefix or layer.ref[2:].startswith(".".join(prefix))):
                # An override that sets a key to the value it already had
                rows = [(tuple(layer.ref[2:].split(".")), Unset, Unset)]
        else:
            name = Text(layer.ref, style=f"bold {styles[layer.kind]}")

        for path, old, new in rows:
            key = ".".join(map(str, path))
            if old is Unset and new is Unset:
                table.add_row(name, key, Text("no change", style="dim italic"), "")
            else:
                table.add_row(name, key, show(old), show(new))
            name = ""

        # Overrides share one section, each mixin has its own
        if rows and layer.kind != "override":
            table.add_section()

    if not table.rows:
        return Text(f"No changes on top of {base}", style="dim")

    title = " + ".join([base, *(mixins or [])])
    count = sum(1 for layer in stack[1:] if layer.kind == "override")
    if count:
        title += f" + {count} override{'s' if count > 1 else ''}"

    return Panel(
        table,
        title=f"[bold]Changes[/] [dim]{escape(title)}[/]",
        subtitle=section,
        border_style="blue",
        box=box.ROUNDED,
    )


def references(
    base: str,
    mixins: Optional[list[str]] = None,
    overrides: Optional[dict] = None,
    section: Optional[str] = None,
) -> list[tuple[str, str, Any]]:
    """
    Resolve each ``${...}`` reference of a composed config on its own.

    Every value holding a reference is interpolated against the full config,
    with the ISOFIT environment exposed under ``env`` as :func:`loader.load`
    does, independently of the others, so one unresolvable reference does not
    hide what the rest become. References to keys that are unset are noted, as
    the value will change once they are set (e.g. ``${input.fid}`` within a
    file name).

    Parameters
    ----------
    base : str
        A config reference for the base; see :func:`loader.resolve`.
    mixins : list[str], optional
        Config references layered on top of ``base``, in order.
    overrides : dict, optional
        Dotted-key overrides applied last.
    section : str, optional
        Dotted path of a section to only resolve the references within.

    Returns
    -------
    list[tuple[str, str, Any, list[str]]]
        The dotted key, the value as written, the value it resolves to (or the
        exception if it does not resolve), and the referenced keys that are
        unset (``null``), in config order.
    """
    from isofit.data import env

    raw = loader.load(base, mixin=list(mixins or []), overrides=overrides)
    prefix = tuple(section.split(".")) if section else ()

    def resolve(item, parent, data):
        try:
            return interpolate.interp(item, parent, data)
        except (KeyError, ValueError) as e:
            return e

    found = []
    for path, value in flatten(raw).items():
        if path[: len(prefix)] != prefix:
            continue

        # Containers are flattened to their leaves, except lists
        values = enumerate(value) if isinstance(value, list) else [(None, value)]
        for index, item in values:
            if not (isinstance(item, str) and interpolate.Interp.search(item)):
                continue

            key = ".".join(map(str, path)) + (f"[{index}]" if index is not None else "")
            data = copy.deepcopy(raw)
            data["env"] = {"modtran": None, **dict(env.items())}
            resolved = resolve(item, lookup(data, path[:-1]), data)

            # The references that resolve to null, so set to "None" in a string
            unset = []
            if not isinstance(resolved, Exception):
                for ref in interpolate.Interp.findall(item):
                    data = copy.deepcopy(raw)
                    data["env"] = {"modtran": None, **dict(env.items())}
                    value = resolve(f"${{{ref}}}", lookup(data, path[:-1]), data)
                    if value is None:
                        unset.append(ref.lstrip("."))

            found.append((key, item, resolved, unset))

    return found


def references_table(
    base: str,
    mixins: Optional[list[str]] = None,
    overrides: Optional[dict] = None,
    section: Optional[str] = None,
) -> Union[Panel, Text]:
    """
    Render a preview of what each ``${...}`` reference of a composed config
    resolves to (see :func:`references`): a row per reference with its key,
    the value as written, and the resolved value, or why it does not resolve.

    Parameters
    ----------
    base : str
        A config reference for the base; see :func:`loader.resolve`.
    mixins : list[str], optional
        Config references layered on top of ``base``, in order.
    overrides : dict, optional
        Dotted-key overrides applied last.
    section : str, optional
        Dotted path of a section to only show the references within.

    Returns
    -------
    rich.panel.Panel or rich.text.Text
        The table, or a note if there are no references.
    """
    # Errors are shown in the table rather than logged
    logging.disable(logging.ERROR)
    try:
        found = references(base, mixins, overrides, section)
    finally:
        logging.disable(logging.NOTSET)

    if not found:
        return Text(f"No ${{...}} references in {base}", style="dim")

    table = Table(box=box.SIMPLE_HEAD, expand=True, header_style="bold")
    table.add_column("Key", style="bold", ratio=3, overflow="fold")
    table.add_column("Written", ratio=3, overflow="fold")
    table.add_column("Resolves to", ratio=4, overflow="fold")

    failed = pending = 0
    for key, written, resolved, unset in found:
        if isinstance(resolved, Exception):
            failed += 1
            reason = str(resolved).strip("'\"")
            table.add_row(key, show(written), Text(f"✘ {reason}", style="red"))
            continue

        value = show(resolved)
        if unset:
            pending += 1
            names = ", ".join(unset)
            value.append(
                f"\n⚠ until {names} {'is' if len(unset) == 1 else 'are'} set",
                style="yellow",
            )
        table.add_row(key, show(written), value)

    count = f"{len(found)} reference{'s' if len(found) != 1 else ''}"
    if pending:
        count += f", [yellow]{pending} pending unset keys[/]"
    if failed:
        count += f", [red]{failed} unresolved[/]"

    title = " + ".join([base, *(mixins or [])])
    if overrides:
        title += f" + {len(overrides)} override{'s' if len(overrides) > 1 else ''}"

    return Panel(
        table,
        title=f"[bold]Interpolation[/] [dim]{escape(title)}[/]",
        subtitle=f"{section + ' · ' if section else ''}{count}",
        border_style="yellow",
        box=box.ROUNDED,
    )


def compose(
    base: str,
    mixins: Optional[list[str]] = None,
    overrides: Optional[dict] = None,
    section: Optional[str] = None,
    interp: bool = False,
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
    interp : bool, default=False
        Show the config with ``${...}`` references resolved, rather than as
        written (see :func:`references_table` to preview what they resolve
        to). Validation is always of the resolved config.

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

    # The overrides as one layer; see `changes_table` for each on its own
    if overrides:
        applied = [layer for layer in stack if layer.kind == "override"]
        stack = [layer for layer in stack if layer.kind != "override"]
        merged = [change for layer in applied for change in layer.changes]
        stack.append(Layer("overrides", "override", None, merged))

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

    yml = dump(shown)
    source = Syntax(yml.rstrip(), "yaml", line_numbers=True, background_color="default")

    title = " + ".join([base, *mixins])
    if overrides:
        title += f" + {len(overrides)} override{'s' if len(overrides) > 1 else ''}"
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
            subtitle=section
            or ("interpolated" if interp and not error else "as written"),
            border_style="cyan",
            box=box.ROUNDED,
        ),
    ]
    return Group(*parts)


def tokens(text: str) -> list[str]:
    """
    Split text on whitespace, keeping quoted strings and bracketed values
    (``[...]``, ``{...}``, ``(...)``) together, e.g. ``a=[1, 2] b "c d"`` to
    ``['a=[1, 2]', 'b', 'c d']``. Quotes around a whole token are removed.

    Raises
    ------
    ValueError
        If a quote or bracket is not closed.
    """
    found, token, quote, depth = [], "", None, 0
    for char in text:
        if quote:
            token += char
            quote = None if char == quote else quote
        elif char in "'\"":
            token, quote = token + char, char
        elif char in "[{(":
            token, depth = token + char, depth + 1
        elif char in "]})":
            token, depth = token + char, depth - 1
        elif char.isspace() and depth == 0:
            if token:
                found.append(token)
            token = ""
        else:
            token += char

    if quote or depth:
        raise ValueError(f"Unclosed {quote or 'bracket'} in {text!r}")
    if token:
        found.append(token)

    # A quoted token is a string, e.g. `"/my scratch"`
    return [
        t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "'\"" else t for t in found
    ]


def parse(text: str) -> dict[str, Any]:
    """
    Parse overrides typed at the :func:`browse` prompt into a ``{dotted.key:
    value}`` dict, as :func:`interpolate.parse_overrides` does for the command
    line.

    Each override is a dotted key and a value, separated by a space or ``=``,
    with an optional leading ``--``. Values are parsed as Python literals where
    they can be, so may be lists or dicts, and quoted values may contain
    spaces, e.g. ``output.dir "/my scratch"``.

    Raises
    ------
    ValueError
        If a key has no value, or a quote or bracket is not closed.

    Examples
    --------
    >>> parse("output.dir /scratch --implementation.n_cores=8")
    {'output.dir': '/scratch', 'implementation.n_cores': 8}
    >>> parse("implementation.inversion.windows=[[400, 1300], [1450, 2400]]")
    {'implementation.inversion.windows': [[400, 1300], [1450, 2400]]}
    """
    split = []
    for token in tokens(text):
        key, eq, val = token.partition("=")
        if eq and key and key[0] not in "[{(":
            split += [key, val]
        else:
            split.append(token)

    keys, values = split[0::2], split[1::2]
    if len(keys) != len(values):
        raise ValueError(f"Override {keys[-1].removeprefix('--')!r} has no value")

    return {key.removeprefix("--"): literal(val) for key, val in zip(keys, values)}


def literal(text: str) -> Any:
    """
    Parse a value typed at the :func:`browse` prompt as on the command line: a
    Python literal where it is one (e.g. ``8``, ``[1, 2]``, ``None``), else the
    text itself, spaces included. A wholly quoted value is a string, e.g.
    ``"/my scratch"``. The YAML ``true``, ``false``, and ``null`` the config is
    shown with are accepted too.

    Raises
    ------
    ValueError
        If a quote or bracket is not closed.

    Examples
    --------
    >>> literal("8"), literal("[1, 2]"), literal('"/my scratch"'), literal("two words")
    (8, [1, 2], '/my scratch', 'two words')
    >>> literal("true"), literal("null")
    (True, None)
    """
    found = tokens(text)
    token = found[0] if len(found) == 1 else text.strip()
    if token.lower() in YamlScalars and token == text.strip():
        return YamlScalars[token.lower()]

    return interpolate.parse_overrides(["--value", token])["value"]


# The YAML scalars :func:`literal` accepts beyond Python literals
YamlScalars = {"true": True, "false": False, "null": None, "~": None}


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


def save(
    file: Union[str, Path],
    base: str,
    mixins: Optional[list[str]] = None,
    overrides: Optional[dict] = None,
    interp: bool = True,
) -> Path:
    """
    Write the config of a base with mixins and overrides layered on top.

    The format follows the file extension: YAML (``.yml`` / ``.yaml``), in the
    style of the templates, or JSON (``.json``).

    Parameters
    ----------
    file : str or pathlib.Path
        The file to write. Missing parent directories are created.
    base : str
        A config reference for the base; see :func:`loader.resolve`.
    mixins : list[str], optional
        Config references layered on top of ``base``, in order.
    overrides : dict, optional
        Dotted-key overrides applied last.
    interp : bool, default=True
        Write the config with ``${...}`` references resolved. Otherwise they
        are kept, so the written config still follows e.g. ``output.dir``.

    Returns
    -------
    pathlib.Path
        The written file.

    Raises
    ------
    ValueError
        If the extension is not one of :data:`Formats`, ``base`` or a mixin is
        not a known config reference, or interpolation fails.
    KeyError
        If a ``${...}`` reference cannot be resolved.

    Examples
    --------
    >>> save("run.yml", "vswir", ["emit"], {"output.dir": "/scratch"})  # doctest: +SKIP
    PosixPath('run.yml')
    """
    file = Path(file).expanduser()
    if file.suffix.lower() not in Formats:
        raise ValueError(
            f"Unsupported file extension {file.suffix!r} for {str(file)!r}, "
            f"expected one of {', '.join(Formats)}"
        )

    data = loader.load(
        base, mixin=list(mixins or []), overrides=overrides, interp=interp
    )
    if file.suffix.lower() == ".json":
        text = json.dumps(data, indent=4, ensure_ascii=False) + "\n"
    else:
        text = dump(data)

    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text)

    return file


def readkey() -> str:
    """
    Read a single key press, without waiting for Enter.

    Falls back to reading a line when stdin is not a terminal (e.g. piped
    input), returning its first character.

    Returns
    -------
    str
        The key, lowercased, or ``""`` for a key with no character (e.g. an
        arrow key).

    Raises
    ------
    EOFError
        On end of input, or Ctrl-D.
    KeyboardInterrupt
        On Ctrl-C.
    """
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line.strip()[:1].lower()

    try:
        import msvcrt
    except ImportError:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            # cbreak keeps Ctrl-C as an interrupt; read a whole escape
            # sequence at once so its remainder isn't taken as the next key
            tty.setcbreak(fd)
            key = os.read(fd, 8).decode(errors="ignore")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    else:
        key = msvcrt.getwch()
        if key == "\x03":
            raise KeyboardInterrupt

    if key in ("\x04", ""):
        raise EOFError

    return key.lower() if key.isprintable() and len(key) == 1 else ""


@dataclass
class Session:
    """
    The state of an interactive :func:`browse`: the config being composed, if
    any, as a base with the mixins layered on it in order and the overrides
    applied last, and the file it was last saved to and whether raw.
    """

    console: Console
    items: list[Entry]
    base: Optional[str] = None
    mixins: list[str] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)
    saved: Optional[Path] = None
    raw: bool = False

    # The mixins and overrides in the order they were added, for undo
    history: list[tuple[str, str]] = field(default_factory=list)

    # The key of each line of the composition as last shown, for selecting
    # overrides by line number
    lines: dict[int, tuple] = field(default_factory=dict)

    @property
    def composing(self) -> bool:
        """Whether a config is being composed."""
        return self.base is not None

    def ask(self, prompt: str) -> str:
        """Prompt for a line of input, ``""`` if left blank."""
        answer = Prompt.ask(
            prompt, default="", show_default=False, console=self.console
        )
        return answer.strip()

    def buttons(self) -> Text:
        """Render the active composition, if any, and the available buttons."""
        text = Text()
        if self.composing:
            text.append("Composing ", style="dim")
            text.append(self.base, style=f"bold {Styles['base']}")
            for mixin in self.mixins:
                text.append(" + ", style="dim")
                text.append(mixin, style=f"bold {Styles['mixin']}")
            for key, val in self.overrides.items():
                text.append(f"  --{key} ", style="magenta")
                text.append(show(val))
            text.append("\n")

        for key, label in Buttons.items():
            if key in Composing and not self.composing:
                continue
            text.append(f" {key} ", style="bold reverse")
            text.append(f" {label}  ")

        return text

    def show(self) -> None:
        """Print the active composition, as written, and index its lines by key."""
        self.console.print(compose(self.base, self.mixins, overrides=self.overrides))
        data = loader.load(self.base, mixin=self.mixins, overrides=self.overrides)
        self.lines = keylines(dump(data))

    def keys(self, choice: str) -> Optional[list[str]]:
        """
        Resolve line numbers of the composition as last shown, e.g. ``12``,
        ``12 15``, or ``12-15``, to the dotted keys on those lines, deduplicated
        in order. ``None`` if ``choice`` is not line numbers.

        Raises
        ------
        ValueError
            If a line is not in the composition.
        """
        if not re.fullmatch(r"\d+(-\d+)?([\s,]+\d+(-\d+)?)*", choice):
            return None

        numbers = []
        for part in Separators.split(choice.replace(",", " ")):
            first, _, last = part.partition("-")
            numbers += range(int(first), int(last or first) + 1)

        if missing := [n for n in numbers if n not in self.lines]:
            raise ValueError(
                f"Line {missing[0]} is not in the composition "
                f"(lines 1-{max(self.lines, default=0)})"
            )

        paths = dict.fromkeys(self.lines[n] for n in numbers)
        return [".".join(map(str, path)) for path in paths]

    def values(self, keys: list[str]) -> Optional[dict[str, Any]]:
        """
        Prompt for a new value for each of ``keys``, showing its current one,
        parsed as by :func:`parse`. Blank keeps a key as it is; ``None`` if
        every key was kept.

        Raises
        ------
        ValueError
            If a value's quote or bracket is not closed.
        """
        current = loader.load(self.base, mixin=self.mixins, overrides=self.overrides)
        overrides = {}
        for key in keys:
            value = interpolate.get_dotted(current, key)
            now = "a mapping" if isinstance(value, dict) else escape(show(value).plain)
            text = self.ask(
                f"[bold magenta]{escape(key)}[/] [dim](now {now}; new value, blank to keep)[/]"
            )
            if text:
                overrides[key] = literal(text)

        return overrides or None

    def diff(self) -> None:
        """Show the table of what each mixin and override changes."""
        self.console.print(changes_table(self.base, self.mixins, self.overrides))

    def interpolate(self) -> None:
        """
        Preview what each ``${...}`` reference of the composition resolves to,
        optionally within a section.
        """
        section = (
            self.ask(
                "[bold yellow]Section[/] [dim](e.g. [yellow]forward_model.atmosphere[/], "
                "blank for the whole config)[/]"
            )
            or None
        )
        if section:
            raw = loader.load(self.base, mixin=self.mixins, overrides=self.overrides)
            if lookup(raw, tuple(section.split("."))) is Unset:
                return self.console.print(
                    f"[red]Section {section!r} is not in the config[/]"
                )

        self.console.print(
            references_table(self.base, self.mixins, self.overrides, section=section)
        )

    def compose(self) -> None:
        """
        Show the active composition, or start one by prompting for its base.
        """
        if not self.composing:
            self.console.print(overview(self.items, kinds=("base",)))
            choice = self.ask(
                "[bold cyan]Base[/] [dim](number or name, blank to cancel)[/]"
            )
            if not choice:
                return
            try:
                self.base = pick(self.items, choice, "base")
            except ValueError as e:
                return self.console.print(str(e))

        self.show()

    def mixin(self) -> None:
        """Prompt for mixins to layer on top of the composition, in order."""
        self.console.print(overview(self.items, kinds=("mixin",)))
        choice = self.ask(
            "[bold green]Mixins[/] [dim](numbers or names in order, blank to cancel)[/]"
        )
        if not choice:
            return

        try:
            mixins = [
                pick(self.items, ref, "mixin")
                for ref in Separators.split(choice)
                if ref
            ]
        except ValueError as e:
            return self.console.print(str(e))

        if layered := [m for m in mixins if m in self.mixins]:
            names = ", ".join(layered)
            self.console.print(f"[yellow]Already layered:[/] {names}")
        for mixin in dict.fromkeys(mixins):
            if mixin not in self.mixins:
                self.mixins.append(mixin)
                self.history.append(("mixin", mixin))
        self.show()

    def override(self) -> None:
        """
        Prompt for overrides applied on top of the mixins, either:

        - Line numbers of the composition as last shown, e.g. ``12``,
          ``12 15``, or ``12-15``, then a new value for the key on each line
        - Dotted-key overrides, as on the command line: ``output.dir
          /scratch``, ``--implementation.n_cores 8``, or ``key=value``, several
          separated by spaces

        Values are parsed as Python literals where they can be, e.g. ``8`` or
        ``[1, 2]``.
        """
        choice = self.ask(
            "[bold magenta]Overrides[/] [dim](line numbers, e.g. [magenta]12[/] or "
            "[magenta]12-15[/], or [magenta]output.dir /scratch[/]; blank to cancel)[/]"
        )
        if not choice:
            return

        try:
            if (keys := self.keys(choice)) is not None:
                overrides = self.values(keys)
                if overrides is None:
                    return self.console.print("No overrides set")
            else:
                overrides = parse(choice)
        except ValueError as e:
            return self.console.print(f"[red]{e}[/]")

        # Check that the overrides apply before keeping them
        try:
            loader.load(
                self.base, mixin=self.mixins, overrides={**self.overrides, **overrides}
            )
        except (KeyError, ValueError, TypeError) as e:
            return self.console.print(f"[red]Could not apply the overrides: {e}[/]")

        for key, val in overrides.items():
            self.history = [h for h in self.history if h != ("override", key)]
            self.history.append(("override", key))
            self.overrides.pop(key, None)
            self.overrides[key] = val
        self.show()

    def undo(self) -> None:
        """
        Remove a mixin or override from the composition, the last one added
        unless another is chosen by name or dotted key.
        """
        if not self.history:
            return self.console.print("[yellow]No mixins or overrides to remove[/]")

        kind, name = self.history[-1]
        if len(self.history) > 1:
            choice = self.ask(
                f"[bold]Remove which mixin or override[/] [dim](blank for the last, "
                f"[{'green' if kind == 'mixin' else 'magenta'}]{name}[/])[/]"
            )
            if choice:
                choice = choice.removeprefix("--")
                entry = find(self.items, choice)
                if entry and entry.name in self.mixins:
                    kind, name = "mixin", entry.name
                elif choice in self.overrides:
                    kind, name = "override", choice
                else:
                    names = [*self.mixins, *self.overrides]
                    return self.console.print(
                        unknown(choice, names, "layered mixin or override")
                    )

        self.history.remove((kind, name))
        if kind == "mixin":
            self.mixins.remove(name)
            self.console.print(f"Removed [bold green]{name}[/]")
        else:
            self.overrides.pop(name)
            self.console.print(f"Removed [bold magenta]--{name}[/]")
        self.show()

    def reset(self) -> None:
        """Discard the composition."""
        self.base, self.mixins, self.overrides, self.history = None, [], {}, []
        self.lines = {}
        self.saved, self.raw = None, False
        self.console.print("Composition cleared, press [bold]c[/] to start another")

    def save(self) -> None:
        """
        Prompt for a file and write the composition to it with :func:`save`.

        A ``raw`` suffix (e.g. ``run.yml raw``) keeps the ``${...}`` references
        instead of resolving them. Left blank, the composition is saved as it
        last was, or else to ``<base>.yml``. An existing file is only
        overwritten when confirmed, unless it is the one last saved to.
        """
        default = self.saved or Path(f"{self.base}.yml")
        last = f"{default}{' raw' if self.saved and self.raw else ''}"
        choice = self.ask(
            f"[bold]Save to[/] [dim](.yml, .yaml, or .json; add [cyan]raw[/] to "
            f"keep ${{...}} references; blank for [cyan]{last}[/])[/]"
        )

        if choice:
            name, _, flag = (
                choice.rpartition(" ") if choice.endswith(" raw") else (choice, "", "")
            )
            file, raw = Path(name.strip()).expanduser(), bool(flag)
        else:
            file, raw = default, self.raw and self.saved is not None

        if file.exists() and file != self.saved:
            confirm = self.ask(f"[yellow]{file} exists, overwrite?[/] [dim](y/N)[/]")
            if confirm.lower() not in ("y", "yes"):
                return self.console.print("Not saved")

        try:
            written = save(file, self.base, self.mixins, self.overrides, interp=not raw)
        except (KeyError, ValueError, OSError) as e:
            return self.console.print(f"[red]Not saved: {e}[/]")

        self.saved, self.raw = written, raw
        kind = "raw" if raw else "interpolated"
        self.console.print(f"[green]✔ Saved the {kind} config to [bold]{written}[/][/]")

    def view(self) -> None:
        """Prompt for a base or mixin and show it."""
        choice = self.ask(
            "[bold]View[/] [dim](number or name of a base or mixin, blank to cancel)[/]"
        )
        if not choice:
            return
        if entry := find(self.items, choice):
            return self.console.print(details(entry))
        self.console.print(unknown(choice, [entry.name for entry in self.items]))

    def list(self) -> None:
        """Show every base and mixin."""
        self.console.print(overview(self.items))


def browse(console: Optional[Console] = None) -> None:
    """
    Interactively browse the bases and mixins, and compose configs from them.

    Lists them with :func:`overview`, then reads single-key :data:`Buttons`
    until ``q``, Ctrl-C, or end of input:

    - ``c`` compose: prompt for a base to start composing, or, while composing,
      show the active composition with :func:`compose`
    - ``d`` diff: show what each mixin and override changes with
      :func:`changes_table`
    - ``i`` interpolate: preview what each ``${...}`` reference resolves to
      with :func:`references_table`, optionally within a section
    - ``m`` mixin: layer mixins on top of the composition
    - ``o`` override: set overrides, applied after the mixins, by line
      numbers of the shown composition or as dotted keys
    - ``u`` undo: remove a mixin or override, the last added by default
    - ``r`` reset: discard the composition
    - ``s`` save: write the composition to a file with :func:`save`
    - ``v`` view: show a base or mixin with :func:`details`
    - ``l`` list: show every base and mixin again

    ``d``, ``i``, ``m``, ``o``, ``u``, ``r``, and ``s`` only apply while composing.

    Parameters
    ----------
    console : rich.console.Console, optional
        Console to render to and read from. Defaults to a new console.
    """
    session = Session(console or Console(), entries())
    session.console.print(overview(session.items))

    while True:
        session.console.print()
        session.console.print(session.buttons())
        try:
            key = readkey()
            if key not in Buttons:
                if key:
                    session.console.print(f"[red]No button [bold]{key!r}[/][/]")
                continue

            session.console.print(f"[dim]›[/] {Buttons[key]}")
            if key == "q":
                break
            if key in Composing and not session.composing:
                session.console.print(
                    "[yellow]No config is being composed, press [bold]c[/] to start one[/]"
                )
                continue

            getattr(session, Buttons[key])()
        except (EOFError, KeyboardInterrupt):
            session.console.print()
            break
