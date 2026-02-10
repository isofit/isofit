Documentation
=============

Documentation is an ongoing effort for the isofit codebase.  Your contributions are greatly appreciated. In general, we prefer the use of Google Doc Strings, and the use of Python 3.6+ typing specification, where possible. Good models for how documentation should be updated are the [isofit/utils/apply_oe.py](https://github.com/isofit/isofit/blob/dev/isofit/utils/apply_oe.py) and [isofit/core/common.py](https://github.com/isofit/isofit/blob/dev/isofit/core/common.py) files.

We use mkdocs to build the documentation automatically. The `mkdocs.yml` file on the repo root contains the configuration for this automation. You can build the docs locally via:

```
$ uv sync --optional docs # Make sure mkdocs and its dependencies are installed
$ uv run mkdocs serve # Temporarly build the docs and host them locally
```

Alternatively, you can build a static set of docs via:

```
$ uv run mkdocs build --clean # May want the clean flag to cleanup any previous build
$ uv run python -m http.server -d site 8001 # Locally host the newly create site/ directory
```
