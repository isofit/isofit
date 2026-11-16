"""
Rich formatting for config validation errors.

Errors are grouped into a tree by their path in the config, with tree guides
drawn down the Field column:

    Field                    Error                            Got
   ───────────────────────────────────────────────────────────────────────
    forward_model
    └─ atmosphere
       ├─ engine (sixs)
       │  └─ unknown_option  Extra inputs are not permitted   {'a': 1}
       └─ lut_grid
          └─ AOT550          Input should be a valid list     'abc'
"""

from collections.abc import Iterator, Mapping, Sequence
from types import UnionType
from typing import Annotated, Any, Optional, Union, get_args, get_origin

from pydantic import BaseModel, Discriminator, Tag, ValidationError
from pydantic_core import ErrorDetails
from rich import box
from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, RenderResult
from rich.measure import Measurement
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text


class Tagged(str):
    """
    A path label for the selected member of a discriminated union, e.g.
    ``engine (sixs)``. Section collapsing stops at these so the keys under a
    union stay nested beneath it.
    """


def segment(part: Union[str, int]) -> str:
    """
    Format a single segment of an error location.

    Parameters
    ----------
    part : str or int
        A key, or a list index.

    Returns
    -------
    str
        The key, or the index as ``[i]``.
    """
    return f"[{part}]" if isinstance(part, int) else str(part)


def got(err: ErrorDetails) -> Optional[Any]:
    """
    Retrieve the offending input of an error.

    Parameters
    ----------
    err : pydantic_core.ErrorDetails
        A single error from :meth:`ValidationError.errors`.

    Returns
    -------
    any or None
        The input, or None for missing fields since their input is the entire
        parent section.
    """
    return None if err["type"] == "missing" else err["input"]


def unwrap(tp: Any, meta: list) -> tuple[Any, list]:
    """
    Strip ``Annotated`` and ``Optional`` from a type annotation.

    Parameters
    ----------
    tp : type
        The type annotation.
    meta : list
        Metadata collected so far, such as a ``Field``'s metadata.

    Returns
    -------
    tuple of (type, list)
        The unwrapped type and the metadata with any ``Annotated`` extras added.
    """
    while True:
        origin = get_origin(tp)
        if origin is Annotated:
            tp, *extra = get_args(tp)
            meta = [*meta, *extra]
        elif origin in (Union, UnionType) and type(None) in get_args(tp):
            args = [arg for arg in get_args(tp) if arg is not type(None)]
            if len(args) > 1:
                return Union[tuple(args)], meta
            tp = args[0]
        else:
            return tp, meta


def tagged_options(tp: Any, meta: list) -> Optional[dict[str, Any]]:
    """
    Retrieve the members of a discriminated union by their tag.

    Parameters
    ----------
    tp : type
        An unwrapped type annotation, see :func:`unwrap`.
    meta : list
        The annotation's metadata.

    Returns
    -------
    dict or None
        ``{tag: type}`` if ``tp`` is a discriminated union, otherwise None.
    """
    if get_origin(tp) not in (Union, UnionType):
        return None
    if not any(isinstance(m, Discriminator) for m in meta):
        return None

    options = {}
    for arg in get_args(tp):
        for m in get_args(arg)[1:] if get_origin(arg) is Annotated else ():
            if isinstance(m, Tag):
                options[m.tag] = arg
    return options


def tag_indices(model: type[BaseModel], loc: tuple) -> set[int]:
    """
    Find the segments of an error location that are discriminated union tags.

    Pydantic inserts the tag of the selected union member into the location,
    e.g. ``engine.sixs.key`` even though the config only has ``engine.key``.
    These are found by walking the model's field types along the path rather
    than the input data, since base configs merge in keys during validation.

    Parameters
    ----------
    model : type of pydantic.BaseModel
        The model that was validated.
    loc : tuple
        The location of a single error.

    Returns
    -------
    set of int
        Indices into ``loc`` of the union tags.
    """
    tags = set()
    tp, meta = model, []
    for i, part in enumerate(loc):
        tp, meta = unwrap(tp, meta)

        options = tagged_options(tp, meta)
        if options is not None:
            if part not in options:
                break
            tags.add(i)
            tp, meta = options[part], []
        elif isinstance(tp, type) and issubclass(tp, BaseModel):
            fields = {f.alias or name: f for name, f in tp.model_fields.items()}
            field = tp.model_fields.get(part) or fields.get(part)
            if field is None:
                break
            tp, meta = field.annotation, list(field.metadata)
        elif get_origin(tp) in (dict, Mapping):
            tp, meta = get_args(tp)[-1], []
        elif get_origin(tp) in (list, tuple, set, Sequence):
            tp, meta = get_args(tp)[0], []
        else:
            break
    return tags


def display_loc(model: type[BaseModel], loc: tuple) -> list[str]:
    """
    Format an error location for display.

    Parameters
    ----------
    model : type of pydantic.BaseModel
        The model that was validated.
    loc : tuple
        The location of a single error.

    Returns
    -------
    list of str
        The path segments, with union tags folded into their key as
        :class:`Tagged` labels, e.g. ``["engine (sixs)", "key"]``.
    """
    tags = tag_indices(model, loc)
    parts = []
    for i, part in enumerate(loc):
        if i in tags and parts:
            parts[-1] = Tagged(f"{parts[-1]} ({part})")
        else:
            parts.append(segment(part))
    return parts


def build_tree(model: type[BaseModel], error: ValidationError) -> dict:
    """
    Nest the errors by their display path.

    Parameters
    ----------
    model : type of pydantic.BaseModel
        The model that was validated.
    error : pydantic.ValidationError
        The validation error.

    Returns
    -------
    dict
        Nested dict of path segments, in order of first appearance. Errors at a
        node are stored as a list under the ``None`` key.
    """
    root = {}
    for err in error.errors():
        node = root
        # Model validator errors have an empty location
        for part in display_loc(model, err["loc"]) or ["<root>"]:
            node = node.setdefault(part, {})
        node.setdefault(None, []).append(err)
    return root


def collapse(name: str, child: dict) -> tuple[str, dict]:
    """
    Merge a chain of sections that each contain a single subsection into one
    label, e.g. ``forward_model.atmosphere``.

    A :class:`Tagged` union ends the chain so its keys stay nested beneath it
    rather than reading as ``engine (sixs).key``.

    Parameters
    ----------
    name : str
        Label of the section.
    child : dict
        The section's node, see :func:`build_tree`.

    Returns
    -------
    tuple of (str, dict)
        The merged label and the node at the end of the chain.
    """
    last = name
    while None not in child and len(child) == 1 and not isinstance(last, Tagged):
        ((last, child),) = child.items()
        name = f"{name}{last}" if last.startswith("[") else f"{name}.{last}"
    return name, child


def tree_rows(model: type[BaseModel], error: ValidationError) -> Iterator[tuple]:
    """
    Flatten the error tree into table rows with their tree guides.

    Parameters
    ----------
    model : type of pydantic.BaseModel
        The model that was validated.
    error : pydantic.ValidationError
        The validation error.

    Yields
    ------
    head : str
        Tree guides prefixing the first line of the row.
    tail : str
        Tree guides prefixing the continuation lines of the row.
    name : str
        Label of the key or section.
    err : pydantic_core.ErrorDetails or None
        The error, or None for section header rows.
    has_children : bool
        Whether there are rows nested beneath this one.
    """

    def walk(node, prefix, depth):
        children = [(k, v) for k, v in node.items() if k is not None]
        for i, (name, child) in enumerate(children):
            name, child = collapse(name, child)

            last = i == len(children) - 1
            if depth == 0:
                head = tail = ""
            else:
                head = prefix + ("└─ " if last else "├─ ")
                tail = prefix + ("   " if last else "│  ")

            has_children = any(k is not None for k in child)
            errs = child.get(None, [])
            if not errs:
                yield head, tail, name, None, has_children
            for j, err in enumerate(errs):
                yield (head if j == 0 else tail), tail, name, err, has_children

            yield from walk(child, tail, depth + 1)

    yield from walk(build_tree(model, error), "", 0)


class ErrorsTable:
    """
    Rich renderable of validation errors as a table grouped by config section,
    with tree guides drawn down the Field column.

    Rich renders each cell independently, so a cell cannot know how tall its row
    is. To draw the guides through the full height of a row, the column widths
    are fixed here and the Error/Got cells are pre-rendered to find each row's
    height.

    Parameters
    ----------
    model : type of pydantic.BaseModel
        The model that was validated.
    error : pydantic.ValidationError
        The validation error.
    """

    # Cell padding (1 each side of 3 columns) + 2 column dividers
    overhead = 8

    def __init__(self, model: type[BaseModel], error: ValidationError):
        self.rows = list(tree_rows(model, error))

    @staticmethod
    def cells(err: ErrorDetails) -> tuple[Text, Union[Text, Pretty]]:
        """
        Create the Error and Got cells of an error row.

        Parameters
        ----------
        err : pydantic_core.ErrorDetails
            A single error.

        Returns
        -------
        tuple of (rich.text.Text, rich.text.Text or rich.pretty.Pretty)
            The error message and the offending input. Text() so user-provided
            values are not parsed as rich markup.
        """
        value = got(err)
        if value is None:
            value = Text("")
        else:
            value = Pretty(
                value, max_length=6, max_string=120, overflow="fold", indent_size=2
            )
        # Overflow is set on the Text rather than the column so that rows are
        # pre-rendered the same way the table renders them
        return Text(err["msg"], overflow="fold"), value

    def widths(self, max_width: int) -> tuple[int, int, int]:
        """
        Determine the widths of the Field, Error, and Got columns.

        The Field column is never wrapped. If the Error and Got columns do not
        both fit in the remaining space, it is split between them, giving either
        more than half only when the other needs less than half.

        Since this runs both when rich measures the table and when it renders it
        (at the measured width), the widths must be the same for both.

        Parameters
        ----------
        max_width : int
            Maximum width available to the table.

        Returns
        -------
        tuple of (int, int, int)
            Widths of the Field, Error, and Got columns.
        """
        errs = [err for *_, err, _ in self.rows if err]
        values = [got(err) for err in errs]

        # Columns are at least as wide as their headers
        field = max(cell_len("Field"), *(cell_len(h + n) for h, _, n, *_ in self.rows))
        error = max(cell_len("Error"), *(cell_len(err["msg"]) for err in errs))
        value = max(
            cell_len("Got"), *(cell_len(repr(v)) for v in values if v is not None)
        )

        space = max(max_width - self.overhead - field, 20)
        if error + value > space:
            error = min(error, max(space // 2, space - value))
            value = space - error
        return field, error, value

    def __rich_measure__(
        self, console: Console, options: ConsoleOptions
    ) -> Measurement:
        width = self.overhead + sum(self.widths(options.max_width))
        return Measurement(width, min(width, options.max_width))

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        field_w, error_w, value_w = self.widths(options.max_width)

        table = Table(box=box.SIMPLE_HEAD, show_edge=False)
        table.add_column("Field", style="cyan", width=field_w, no_wrap=True)
        table.add_column("Error", style="red", width=error_w)
        table.add_column("Got", width=value_w)

        for head, tail, name, err, has_children in self.rows:
            if err is None:
                field = Text(head, style="dim")
                field.append(name, style="dim")
                table.add_row(field, "", "")
                continue

            msg, value = self.cells(err)
            height = max(
                len(console.render_lines(msg, options.update(width=error_w))),
                len(console.render_lines(value, options.update(width=value_w))),
            )

            # Continue the sibling guide, plus a child guide if this key has children
            cont = tail + ("│" if has_children else "")

            field = Text(head, style="dim")
            field.append(name, style="bold")
            for _ in range(height - 1):
                field.append("\n" + cont, style="dim")

            table.add_row(field, msg, value)

        yield table


def errors_table(
    error: ValidationError, model: Optional[type[BaseModel]] = None
) -> Panel:
    """
    Format a pydantic validation error as a rich table of its individual errors.

    Parameters
    ----------
    error : pydantic.ValidationError
        The validation error to format.
    model : type of pydantic.BaseModel, optional
        The model that was validated, used to detect discriminated union tags in
        the error paths. Defaults to :class:`~isofit.configs.Config`.

    Returns
    -------
    rich.panel.Panel
        A panel containing one row per error, grouped by config section: the
        field, the error message, and the offending input.

    Examples
    --------
    >>> from rich.console import Console
    >>> from isofit.configs import Config
    >>> try:
    ...     Config.model_validate(data)
    ... except ValidationError as e:
    ...     Console().print(errors_table(e))
    """
    if model is None:
        # Deferred: Config imports every section
        from isofit.configs import Config as model

    count = error.error_count()
    return Panel(
        ErrorsTable(model, error),
        title=f"{count} config validation error{'s' if count > 1 else ''}",
        border_style="red",
        expand=False,
    )
