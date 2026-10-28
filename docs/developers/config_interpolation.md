# Config Loading, Inheritance & Interpolation

Package: `isofit/configs/utils/interpolate/`
CLI: `isofit config ...`

## What it is

The `interpolate` package is the **pre-validation layer** of the ISOFIT config system. It
takes raw YAML/JSON files and turns them into a single, fully resolved plain
`dict` *before* anything is handed to the Pydantic models in
`isofit/configs/sections`. Every stage takes and returns ordinary nested
`dict` / `list` trees; there is no wrapper type.

It adds three capabilities that plain YAML/JSON do not have:

| Feature | Syntax | Purpose |
|---|---|---|
| **Inheritance** | `^^: section` / `^^: file.yml:section` / `^^: [a, b]` | Build a config on top of one or more other configs, same-file or cross-file |
| **Overrides** | `--dotted.key value` (CLI) or `{"dotted.key": value}` (Python) | Change any nested value without editing the file |
| **Interpolation** | `${key.path}` (absolute) / `${.key}` (relative) | Reference other values in the config, e.g. `${output.dir}/lut` |

It also exposes a small `click` CLI (`isofit config preview|copy|validate|list`)
for inspecting and materialising configs.

The module does **no schema validation**. It only produces a dict. Validation
is entirely the job of `Config.model_validate` (via
`isofit.configs.loader.create_new_config`).

## Why it matters to ISOFIT

A complete ISOFIT config is large: `input`, `output`, `forward_model`
(`surface`, `atmosphere` + engine, `instrument`), and `implementation`, with
dozens of fields each. Most runs differ from each other in only a handful of
values. Without this layer every user config would have to repeat every default,
and every path would have to be written out in full.

The module makes that manageable:

1. **One source of truth for defaults.** `isofit/configs/templates/default.yml`
   holds a `default` section that mirrors every Pydantic field. Each RT engine
   (`modtran`, `sixs`, `srtmnet`, `libradtran`, `kernelflows`, `prebuilt`) is a
   small section that does `^^: default` and overrides only the engine block.
2. **Use-case templates.** `vswir.yml`, `thermal.yml`, `simulation.yml`,
   `aquatic_glint.yml`, and `mcmc.yml` inherit cross-file
   (`^^: default.yml:srtmnet`, `^^: default.yml:modtran`, ...) and set only the
   retrieval methodology.
3. **Defaults that follow user values.** Defaults like
   `lut_path: ${output.dir}/lut/lut.zarr` are interpolated *after* the user's
   config is overlaid, so changing `output.dir` moves every derived path with it.
4. **Install paths from the ISOFIT ini.** `loader.load` injects the
   `isofit.ini` environment as an `env` section, so engine paths are written as
   `${env.sixs}`, `${env.srtmnet}/${env.srtmnet-file}`, and so on, and resolve
   correctly on every machine.
5. **Legacy configs get defaults automatically.** Every config loaded through
   `create_new_config` / `Config(...)` (including `apply_oe` output and old
   JSON configs) is overlaid onto the matching engine's base config, so missing
   fields get the shipped defaults.

## Pipeline overview

```
            ┌──────────────┐
 file ────▶ │   read()     │  YAML/JSON → dict
            └──────┬───────┘
                   │  optional: select `section`
            ┌──────▼───────┐
            │   patch()    │  fold in `^^` ancestors (recursive, cross-file)
            └──────┬───────┘     └─ resolve_file() for `file.yml:section`
            ┌──────▼───────┐
            │  override()  │  apply dotted-key overrides (CLI / list / dict)
            └──────┬───────┘     └─ parse_overrides() → _parse_cli_overrides()
            ┌──────▼───────┐
            │ interpolate()│  walk tree, resolve every `${...}` via interp()
            └──────┬───────┘
                   ▼
             resolved dict  ──▶  Config.model_validate (Pydantic)
```

`load()` runs these stages in this order, so:

- **Inheritance happens first.** Overrides and `${}` see the merged tree.
- **Overrides happen before interpolation.** `--output.dir /scratch` is
  reflected in every `${output.dir}` reference.
- **Interpolation happens last** and sees the final values.

## Package layout

Each pipeline stage lives in its own module. The package `__init__` re-exports
the public functions, so `from isofit.configs.utils import interpolate` followed
by `interpolate.load(...)`, `interpolate.read(...)`, etc. works unchanged.

| Module | Contents | Stage |
|---|---|---|
| `common.py` | `Loaders`, `read()`, `get_dotted()`, `locate()`, `set_dotted()`, `merge()` | Shared by every stage; imports nothing else from the package |
| `inherit.py` | `patch()`, `resolve_file()` | 1. `^^` inheritance |
| `overrides.py` | `parse_overrides()`, `_parse_cli_overrides()`, `override()` | 2. dotted-key overrides |
| `resolve.py` | `Interp`, `interp()`, `interpolate()` | 3. `${...}` interpolation |
| `load.py` | `load()` | Entry point that chains the stages above |
| `cli.py` | `load_cli()` and the `isofit config` click group | CLI wrapper |

The dependency graph is a strict DAG: `common` ← `inherit` / `overrides` /
`resolve` ← `load` ← `cli`.

## Components

### Module-level constants

| Name | Description |
|---|---|
| `Interp` | Regex `\${([^}]+)}` that matches `${...}` references. |
| `Loaders` | Extension → parser map (`.yml`/`.yaml` → `yaml.safe_load`, `.json` → `json.load`). It drives both `read()` and cross-file detection in `patch()`. |

### `read(config)`

Loads a single file into a dict, dispatching on its extension. An empty file
loads as `{}`. Raises `ValueError` for unsupported extensions or when the
file's top level is not a mapping. It is the only function that touches disk,
apart from `resolve_file`, which calls `read`.

YAML is parsed with PyYAML's `safe_load`, which follows YAML 1.1. Watch out
for these values in configs:

| Value | Parses as |
|---|---|
| `yes` / `no` / `on` / `off` | bool |
| `1e-5` (no decimal point) | string; write `1.0e-5` to get a float |
| `012` | octal int (10) |

### Dict helpers: `get_dotted`, `locate`, `set_dotted`, `merge`

These replace what a wrapper type would otherwise provide.

- **`get_dotted(data, key)`** looks up `"a.b.c"` in nested dicts, with `[i]`
  indexing into lists: `"a.b[1].c"`, `"a.m[0][1]"`, and negative indices such
  as `"a.b[-1]"`. A literal key always wins over descending, and each separator
  is tried as a split point from left to right. That is what lets
  `${env.srtmnet.file}` reach the ini key `srtmnet.file` even though `env` also
  has a plain `srtmnet` key. It raises `KeyError` if nothing matches, including
  an out-of-range index or a key that descends into a scalar.
- **`locate(data, key)`** resolves the key the same way but returns the path
  taken, a list of `(container, slot)` steps, instead of the value.
  `get_dotted` is a thin wrapper around it. `interp()` uses it to find a
  referenced value's own section and to write the resolved value back.
- **`set_dotted(data, key, value)`** assigns using the same key resolution. It
  creates missing (or `null`) intermediate sections as dicts, and raises
  `TypeError` when an intermediate exists but is the wrong container (a scalar
  or list before `.`, anything but a list before `[i]`). Lists are never
  created or extended, so `[i]` must be in range or `KeyError` is raised. A
  mapping assigned onto an existing mapping is deep-merged with `merge()`;
  anything else replaces the value. The value is deep-copied.
- **`merge(dst, src)`** deep-merges `src` into `dst` in place, with `src`
  winning. Nested mappings are merged key by key. Lists, scalars, and an
  explicit `null` replace the value outright. Values from `src` are
  deep-copied, so `dst` never aliases `src`.

### `load(config, section=None, ctx=None, interp=True)`

The main entry point. It runs the full pipeline:

1. `read()` the file (kept as `full`, used to resolve same-file `^^` refs).
2. If `section` is given, narrow to a copy of `full[section]`. An unknown
   section raises `KeyError` listing the available sections.
3. If the result has a `^^` key, call `patch()` with `base_dir` set to the
   file's directory and a cache seeded with the already-loaded file.
4. If `ctx` is given, call `override()`.
5. If `interp` is true, call `interpolate()`.

`interp=False` is used to get the inherited and overridden tree *without*
resolving references, so more values can be overlaid first. See
[Integration with the loader](#integration-with-the-loader).

### `patch(box, full, ...)`: inheritance via `^^`

`patch` consumes the `^^` key and merges the referenced sections underneath the
current one.

**Precedence.** References are applied lowest-priority first in list order, and
the section's own keys always win:

```yaml
A:
  ^^: [B, C]
B:
  ^^: D
```

The resulting precedence is **`D < B < C < A`**.

**Reference forms.**

| Form | Meaning |
|---|---|
| `^^: default` | Section `default` in the same file |
| `^^: [base, extras]` | Several sections, merged in order |
| `^^: bases/default.yml:srtmnet` | Section `srtmnet` of another file |
| `^^: ../common.json` | The **whole top-level mapping** of another file |

A reference is treated as cross-file when it contains a supported extension
(`.yml`, `.yaml`, `.json`). Relative file paths resolve against the
**directory of the file that declares the `^^`**, not the current working
directory. YAML and JSON can be mixed freely.

**Merging** uses `merge()`, which is a deep merge. Nested mappings are
merged key-by-key, and lists and scalars are replaced wholesale. An explicit
`null` in the child overrides a parent value. Each referenced section is
deep-copied before merging, so two sections inheriting the same parent never
share nested objects.

**Safety.**

- Unknown same-file sections raise `KeyError`.
- Missing files raise `FileNotFoundError`, and unknown sections in another file
  raise `KeyError`.
- Each file is read from disk only once per `patch` tree (`cache`).
- Circular chains (`A → B → A`) raise `ValueError`. The chain is tracked in
  `_seen` as `(file, section)` identities, where same-file refs use the path of
  the file that owns them (passed down as `patch(..., file=...)`). Sections
  with the same name in different files are therefore distinct, and
  `^^: default` and `^^: this.yml:default` count as the same section.

### `resolve_file(sect, base_dir, cache)`

This helper handles cross-file references. It splits `path.yml:section` on the
colon *directly after the extension*, so colons elsewhere in a path (for
example a Windows drive letter) are preserved. It then resolves the path,
loads the file through the cache, and returns:

- `ref_full`: the referenced file's whole config, used for *its own* same-file
  `^^` refs.
- `ref`: the selected section, or the whole file if no `:section` was given.
- `ref_dir`: the referenced file's directory, used as the base for *its*
  cross-file refs.
- `ident`: `(abspath, section)`, used for cycle detection.

Because `ref_full` and `ref_dir` are passed down, inheritance chains spanning
several files resolve correctly: each file's references are interpreted
relative to that file.

### `parse_overrides(source)` and `_parse_cli_overrides(args)`

`parse_overrides` normalises every accepted override shape into one
`{dotted.key: value}` dict:

| Input | Result |
|---|---|
| `None` | `{}` |
| `dict` | returned as-is, with values **not** parsed (`{"forward_model.instrument.SNR": 500}`) |
| `click.Context` | its extra args are parsed with `_parse_cli_overrides` |
| `list[str]` | parsed with `_parse_cli_overrides` |

`_parse_cli_overrides` parses CLI tokens:

- `--key value` parses `value` with `ast.literal_eval`, so `4` becomes an int,
  `[1, 2]` a list, and `True` a bool. If that fails, the value is kept as a
  plain string. A warning is logged only when the value *looked like* an
  intended literal (it starts with a bracket, quote, digit, or `-`) but did not
  parse.
- A bare `--key` at the end of the args, or followed by another `--flag`,
  becomes `True`.
- Tokens that don't start with `--` and aren't consumed as a value are ignored.

### `override(box, ctx)`

Applies the normalised overrides one at a time with `set_dotted()`, so
`"forward_model.instrument.SNR"` reaches into nested sections and missing
sections are created. Overriding a section with a mapping deep-merges into it,
so the section's other keys are kept. Lists and scalars replace the existing
value outright, and `--key None` sets `None`. To replace a whole section,
clear it first: `--key None --key "{...}"`.

### `interp(val, rel, full)`: resolving one string

`interp` resolves every `${...}` in a single string:

- `${a.b.c}` is **absolute** and is looked up in `full` (the whole config) with
  `locate()`, which follows the same key rules as `get_dotted()` but also
  returns where the value lives.
- `${.c}` is **relative** and is looked up in `rel`, the subsection that
  contains the value.
- The referenced value is resolved **depth-first, in its own context**, before
  it is substituted:
  - a string containing `${` is resolved with its *own* section as `rel`, so
    `${.c}` reached through `${s.v}` means `s.c`;
  - a list or mapping is fully interpolated first, so `${l}` never comes back
    as `['${d}']`.

  The resolved value is written back into the config, so each value is resolved
  once and the result does not depend on key order.
- Circular references raise `ValueError`, including cycles through containers
  (`{m: "${s}", s: {a: "${m}"}}`). Values on the current chain are tracked by
  location (`(container, key)`), not by reference text, so the same `${.c}` in
  two sections is not mistaken for a cycle.
- A missing key raises `KeyError`.
- After substitution, the whole string is passed through `ast.literal_eval`,
  so a fully substituted string can become a typed value:

```yaml
n: 4
a: ${n}        # → 4   (int)
b: "${n}0"     # → 40  (int, not "40")
c: ${dir}/x    # → "/data/x" (stays a string)
```

### `interpolate(box, full=None, rel=None)`: walking the tree

`interpolate` walks a dict or list recursively and calls `interp()` on every
string.

- On the first call, `full` defaults to the tree itself. Because `interp()`
  resolves referenced values depth-first and writes them back, the walk order
  does not affect the result.
- On entering each mapping, `rel` is set to that mapping, which is what makes
  `${.key}` relative to the value's own section.
- Lists are walked too. Items inherit the enclosing `rel`, and mappings inside
  lists get their own `rel`.
- The box is mutated in place and also returned.

### `load_cli(ctx, config, section=None)`

This is the shared entry for the CLI commands. `config` may be either:

- a **path** to a `.yml`, `.yaml`, or `.json` file, which is loaded with
  `load()`, or
- the **name of a shipped base config** (`vswir`, `sixs`, ...), which is
  loaded with `loader.load(..., interp=True)` and gets the `env` injection.

An existing file always wins over a base-config name. CLI overrides are
applied in both cases.

## Integration with the loader

`isofit/configs/loader.py` is the main consumer.

### `list_bases()`

Indexes every top-level section of every `isofit/configs/templates/*.yml`
file (using `interpolate.read`), mapping the base config name to its file. The
first file in sorted order wins on duplicates. The index is cached for the
life of the process.

### `load(key, section=None, overrides=None, interp=False)`

1. Resolves `key` to a file: an existing path wins, otherwise it is looked up
   in `list_bases()`.
2. Calls `interpolate.load(file, section=<base name>, interp=False)` to resolve
   `^^` inheritance, then `legacy.convert` to upgrade older schemas. The result
   is cached in `loader.Cache` under `(file, base name)`.
3. Deep-copies the cached dict and applies `overrides` to `section`. Overrides
   are relative to `section`, not to the full config.
4. If `interp` is set, adds `env` (the ISOFIT ini, `isofit.data.env`) to the
   **full** config, interpolates it, then removes `env`. This is what makes
   `${env.sixs}` work. Interpolating the full config (rather than only
   `section`) lets a section reference values outside itself, e.g.
   `forward_model.atmosphere.lut_path: ${output.dir}/lut/lut.zarr`.
5. Returns only `section`.

```python
from isofit.configs.loader import load

cfg = load("vswir", interp=True, overrides={
    "input.measured_radiance_file": "rdn.hdr",
    "forward_model.surface.surface_file": "surface.mat",
    "implementation.n_cores": 4,
})
```

### `create_new_config(file, **kwargs)` and `SectionModel(base=...)`

`create_new_config` calls `load(file, **kwargs)` and passes the result to
`Config.model_validate`. Every `SectionModel` (including the root `Config`) has
a `before` validator that seeds itself from a base config:
`load(base, section=cls.section, overrides=data, interp=True)`. `base`
defaults to `default`, or to the engine's own profile (`sixs`, `modtran`, ...)
for engine sections. The user's values are applied as overrides **before**
interpolation, which is why `lut_path: ${output.dir}/lut/lut.zarr` follows the
*user's* `output.dir`.

```python
from isofit.configs import Config

config = Config(
    base="vswir",
    forward_model={"surface": {"surface_file": "surface.mat"}},
    implementation={"n_cores": 4},
)
```

## The `isofit config` CLI

It is registered in `isofit/__main__.py` as `"config":
"isofit.configs.utils.interpolate.cli"`. Every command accepts a base-config name
or a file path, plus arbitrary `--dotted.key value` overrides.

```bash
# List shipped base configs
isofit config list

# Show the fully resolved config (inheritance + overrides + interpolation)
isofit config preview sixs --implementation.n_cores 4
isofit config preview my_run.yml -s vswir --noflow
isofit config preview vswir --validate        # also run Pydantic validation

# Materialise a resolved config to a new file (.yml/.yaml/.json)
isofit config copy vswir -o my_vswir.yml --output.dir /scratch/run1

# Validate only (prints nothing on success besides a confirmation)
isofit config validate my_run.yml -s vswir
```

| Option | Applies to | Meaning |
|---|---|---|
| `-s/--section` | preview, copy, validate | Load only this top-level section (file paths only) |
| `--noflow` | preview, copy | Use block-style YAML lists instead of `[a, b]` |
| `-v/--validate` | preview | Also validate with Pydantic after previewing |
| `-o/--output` | copy | Destination file; the extension picks YAML or JSON |

`preview` is the fastest way to debug a config. It shows exactly what Pydantic
will receive.

## Worked example

`bases/common.yml`:

```yaml
shared:
  output:
    dir: /data/out
    reset: false
  implementation:
    n_cores: 2
```

`run.yml`:

```yaml
base:
  ^^: bases/common.yml:shared      # cross-file, relative to run.yml
  implementation:
    n_cores: 4                     # beats shared's 2
child:
  ^^: base                         # same-file; transitively pulls in shared
  output:
    rfl: ${output.dir}/rfl         # absolute reference
    nested:
      a: 3
      b: ${.a}                     # relative to `nested`
      lst: ["${output.dir}", "${implementation.n_cores}"]
```

```python
load("run.yml", section="child",
     ctx=["--output.dir", "/scratch", "--implementation.debug"])
```

```yaml
output:
  dir: /scratch            # override, applied before interpolation
  reset: false             # inherited from bases/common.yml
  rfl: /scratch/rfl
  nested: {a: 3, b: 3, lst: [/scratch, 4]}   # "4" literal-parsed to int
implementation:
  n_cores: 4
  debug: true              # bare flag → True
```

## Gotchas

- **Quote `${...}` inside YAML flow sequences.** `lst: [${a}, b]` is a YAML
  *parse error* because `{` starts a flow mapping. Write
  `lst: ["${a}", b]`. A scalar value like `key: ${a}/x` is fine unquoted.
- **Interpolated strings are literal-parsed.** `"${major}.${minor}"` with
  `1`/`0` becomes the float `1.0`, and `"${n}0"` becomes an int. If you need
  a string, make sure the result can't parse as a Python literal.
- **Trailing slashes are not normalised.** `default.yml` sets
  `output.dir: ./isofit-output/`, so `${output.dir}/lut/lut.zarr` gives
  `./isofit-output//lut/lut.zarr`. This is harmless for filesystem paths but
  visible in `preview`.
- **Dict overrides are not parsed.** `{"forward_model.instrument.SNR": "500"}`
  sets the *string* `"500"`. Pass real Python objects.
- **`^^` replaces lists.** Inheriting does not concatenate lists. The child's
  list wins entirely.
- **`env` exists only for base configs.** `${env.*}` works when the config is
  loaded through `loader.load(..., interp=True)` (or `isofit config` with a
  base-config name, or via `create_new_config` / `Config(...)`). A raw `interpolate.load("my.yml")` on a
  file using `${env.sixs}` raises `KeyError` unless you pass `env` yourself as
  an override.

## Testing

Unit tests for the package live in `isofit/test/test_interpolate.py`. They
cover the dict helpers (including the `env.srtmnet-file` lookup),
`read`, `^^` precedence, cycles and errors (including same-named sections
across files), overrides, interpolation (including key-order independence and
cycles through containers), and the `copy` CLI command.

End-to-end coverage lives in `isofit/test/test_configs.py`:
`test_defaults_interpolate_after_overlay`, `test_resolve_is_idempotent`,
`test_resolve_does_not_mutate_input`, the `test_section_from_base*` tests, and
`test_example_configs` (every shipped example config resolves).
