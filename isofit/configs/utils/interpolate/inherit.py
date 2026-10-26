"""
Stage 1 of the pipeline: inheritance via the ``^^`` key.

A section declaring ``^^: <ref>`` has the referenced section(s) merged in
underneath it. References may be same-file (``^^: default``) or cross-file
(``^^: bases/default.yml:srtmnet``).
"""

import copy
from pathlib import Path
from typing import Optional

from .common import Loaders, merge, read


def patch(
    box: dict,
    full: dict,
    _seen: Optional[set] = None,
    base_dir: Optional[Path] = None,
    cache: Optional[dict] = None,
) -> dict:
    """
    Merge one or more referenced sections into ``box`` in place.

    Consumes the ``^^`` key from ``box``, whose value names either a single section
    (str) or several sections (list). Each name is treated as an inherited
    default; they are applied lowest-priority first, in the order listed, and
    ``box``'s own keys always win on conflict. Referenced sections are resolved
    recursively, so a section that itself declares ``^^`` has its own ancestors
    folded in first.

    For example ``A: {^^: [B, C]}`` with ``B: {^^: D}`` yields the precedence
    ``D < B < C < A`` (``A``'s own keys highest, ``D`` lowest).

    References may point either at a section in the same file (``^^: default``) or
    at a section in another file using ``file:section`` syntax (``^^:
    bases/default.yml:srtmnet``). The referenced file may be YAML or JSON. The
    file portion is resolved relative to ``base_dir`` (the directory of the file
    being loaded); a bare ``file`` with no ``:section`` inherits that file's
    top-level mapping.

    Parameters
    ----------
    box : dict
        The config subsection to patch; mutated in place. Its ``^^`` key is removed.
    full : dict
        The full config used to resolve same-file section names.
    _seen : set, optional
        Internal set of (file, section) keys currently being resolved on this
        recursion chain, used to detect circular ``^^`` references. Should not be
        passed by callers.
    base_dir : pathlib.Path, optional
        Directory used to resolve the file portion of cross-file references.
        Defaults to the current working directory.
    cache : dict, optional
        Internal mapping of resolved file path to its loaded config, so each
        referenced file is read from disk only once per ``patch`` tree.

    Returns
    -------
    dict
        The same ``box``, updated with the merged sections.
    """
    if base_dir is None:
        base_dir = Path.cwd()
    if cache is None:
        cache = {}
    _seen = _seen or set()

    sects = box.pop("^^", [])
    if isinstance(sects, str):
        sects = [sects]

    # Accumulate referenced sections lowest-priority first (list order), then let
    # box's own keys win by merging them in last.
    merged = {}
    for sect in sects:
        # Resolve the reference to (its full config, the subsection copy, the
        # base_dir for *its* own ^^ refs, and a hashable identity for cycle
        # detection). Cross-file refs carry a supported file extension, either
        # bare (`path.yml`) or with a `:section` suffix (`path.yml:section`).
        if any(ext in sect for ext in Loaders):
            ref_full, ref, ref_dir, ident = resolve_file(sect, base_dir, cache)
        else:
            if sect not in full:
                raise KeyError(f"^^ references an unknown section: {sect!r}")
            ref_full = full
            ref = copy.deepcopy(full[sect])
            ref_dir = base_dir
            ident = (None, sect)

        if ident in _seen:
            raise ValueError(f"Circular ^^ reference detected for section: {sect!r}")

        if "^^" in ref:
            ref = patch(ref, ref_full, _seen | {ident}, base_dir=ref_dir, cache=cache)

        merge(merged, ref)

    merge(merged, box)

    box.clear()
    box.update(merged)

    return box


def resolve_file(sect: str, base_dir: Path, cache: dict) -> tuple:
    """
    Resolve a cross-file ``^^`` reference of the form ``path[:section]``.

    The path is taken relative to ``base_dir`` (the directory of the file that
    declared the reference) and loaded at most once per ``patch`` tree via
    ``cache``. With no ``:section`` suffix the referenced file's top-level
    mapping is inherited; otherwise the named section within it is used.

    The split is made on the colon that immediately follows a supported file
    extension (``.yml`` / ``.yaml`` / ``.json``), so colons elsewhere in the
    path are left untouched.

    Parameters
    ----------
    sect : str
        The raw reference string, e.g. ``bases/default.yml:srtmnet`` or
        ``../common.json``.
    base_dir : pathlib.Path
        Directory used to resolve ``path`` relative references.
    cache : dict
        Mapping of resolved file path to loaded config, mutated in place.

    Returns
    -------
    tuple
        ``(ref_full, ref, ref_dir, ident)`` where ``ref_full`` is the referenced
        file's full config (for resolving that file's own same-file ``^^`` refs),
        ``ref`` is a copy of the selected subsection, ``ref_dir`` is the directory of
        the referenced file (base for its cross-file refs), and ``ident`` is a
        hashable ``(path, section)`` identity for cycle detection.
    """
    # Split on the colon directly after the file extension; a bare path (no
    # trailing `:section`) inherits the referenced file's top-level mapping.
    for ext in Loaders:
        marker = sect.find(ext + ":")
        if marker != -1:
            split = marker + len(ext)
            path, name = sect[:split], sect[split + 1 :]
            break
    else:
        path, name = sect, None
    name = name or None

    abspath = (base_dir / path).resolve()
    if abspath not in cache:
        if not abspath.exists():
            raise FileNotFoundError(
                f"^^ references a file that does not exist: {str(abspath)!r} (from {sect!r})"
            )
        cache[abspath] = read(abspath)

    ref_full = cache[abspath]

    if name is None:
        ref = copy.deepcopy(ref_full)
    else:
        if name not in ref_full:
            raise KeyError(
                f"^^ references an unknown section {name!r} in file {str(abspath)!r}"
            )
        ref = copy.deepcopy(ref_full[name])

    ref_dir = abspath.parent
    return ref_full, ref, ref_dir, (abspath, name)
