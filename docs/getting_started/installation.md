# Installation

## Conda-Forge (Recommended)

Recommended approach!

New environment:

```
$ mamba create -n isofit_env -c conda-forge isofit
$ mamba activate isofit_env
```

or

```
$ conda create -n isofit_env -c conda-forge isofit
$ conda activate isofit_env
```

Within an existing environment:

```
$ mamba install -c conda-forge isofit
```

or

```
$ conda install -c conda-forge isofit
```

## PyPI (`pip`)

???+ warning

    The commands below use ``$ pip``, however `$ python -m pip` is often a safer choice. It is possible for the `pip` executable to point to a different version of Python than the `python` executable. Using `python -m pip` at least ensures that the package is installed against the Python interpreter in use. The issue is further compounded on systems that also have `python3` and `pip3` executables, or executables for specific versions of Python like `python3.11` and `pip3.11`.

ISOFIT can be installed from the [Python Package Index](https://pypi.org/project/isofit/) with:

```
$ pip install isofit
```

In order to support a wide variety of environments, ISOFIT does not overly constrain its dependencies, however this means that in some cases `pip` can take a very long time to resolve ISOFIT's dependency tree. Some users may need to provide constraints for specific packages, or install ISOFIT last.
`pip` also supports installing from a remote git repository – this installs against the `main` branch:

```
$ pip install "git+https://github.com/isofit/isofit.git@main"
```

## Manual (GitHub)

### Mamba

We recommend using [Mamba](https://mamba.readthedocs.io/en/latest/) to create a virtual environment:

```
$ git clone https://github.com/isofit/isofit
$ mamba env create -f isofit/recipe/isofit.yml
$ mamba activate isofit_env
$ pip install -e ./isofit
```

Developers may need to install additional packages provided by alternative YAML files in the `recipe` directory:

```
$ mamba install --name isofit_env --file ISOFIT/recipe/docker.yml
```

### uv

Alternatively, you may also use [uv](https://github.com/astral-sh/uv) for installing ISOFIT from source:

```
$ uv sync
```

The above command will install the default python version and pinned packages that are confirmed to be working with ISOFIT. Once installed, the CLI can be accessed via `uv run`:

```
$ uv run isofit --help
```

Alternatively, activate the virtual environment to skip the `uv run` command:

```
$ source .venv/bin/activate
$ isofit --help
```

For additional packages, such as those required for development:

```
$ uv sync --extra dev
```

Advanced users may switch python versions via:

```
$ uv python pin 3.13
$ uv sync
```

## GPU (optional)

ISOFIT ships an opt-in `torch` backend for the analytical line retrieval, requested with `--backend torch`. It is off by default, and no GPU is needed to install or run ISOFIT: the default `numpy` backend is unaffected by everything below. CUDA is the only supported accelerator.

### Configuration

The backend is configured through the `implementation` section of the ISOFIT
config; the first three are also flags on `apply_oe` and `analytical_line`.
Every option is inert unless `backend` is `"torch"`, and setting one anyway
produces a config warning rather than being silently ignored.

Option | Default | Description
-|-|-
`backend` | `"numpy"` | `"numpy"` or `"torch"`.
`torch_device` | `"auto"` | `"auto"`, `"cpu"`, `"mps"`, `"cuda"`, or `"cuda:N"`. `"auto"` prefers cuda, then mps, then cpu.
`torch_batch_size` | `"auto"` | Spectra per batched call. `"auto"` sizes the batch against free device memory.
`torch_dtype` | `"auto"` | `"auto"`, `"float32"`, or `"float64"`. `"auto"` is float64 everywhere except mps, which has no float64. float32 is faster and measurably less accurate.
`torch_num_gpu_workers` | `null` | GPU worker actors, defaulting to the visible CUDA device count. **`n_cores` does not control this** -- one CUDA context per core would thrash the device.

An explicit device is never silently downgraded: requesting `cuda` on a
CPU-only torch build raises rather than quietly running twenty times slower.

`torch >= 2.3` is recommended. ISOFIT itself does not bound the version, but older releases predate parts of the batched linear algebra the backend leans on.

### uv

A bare `$ uv sync` installs the CPU-only torch wheel. That is the correct install for every machine without a CUDA device, and it is what the CPU parity tests run against. For a CUDA build, enable the `cuda` extra and turn off the default `cpu` group:

```
$ uv sync --extra cuda --no-group cpu
```

`cpu` and `cuda` are declared as conflicting, so uv refuses to enable both rather than silently resolving to one of them. The `cuda` extra pulls torch from the [cu128 wheel index](https://download.pytorch.org/whl/cu128), so the host driver must be recent enough for CUDA 12.8.

A container recipe is provided in `recipe/Dockerfile.cuda`. It is built by hand and must be run with `--gpus all`.

### pip

Both extras exist under `pip` as well, but they are no-ops there – `pip` has no notion of the wheel index they select, so each simply requests `torch`. Install the CUDA build directly instead:

```
$ pip install isofit
$ pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall torch
```

### Conda

`recipe/isofit.yml` lists plain `pytorch`, which resolves to a CPU-only build on most platforms. GPU users should additionally request `pytorch-cuda`, which is what makes conda select a CUDA-enabled `pytorch`:

```
$ mamba install --name isofit_env -c pytorch -c nvidia pytorch-cuda
```

???+ warning

    Apple Silicon (`mps`) is **not** a supported target for this backend. Metal Performance Shaders do not implement float64, which the retrieval math requires, so `mps` is useful for plumbing work only – no parity or performance result should be taken from it.

# Downloading Extra Files

Once ISOFIT is installed, the CLI provides an easy way to download additional files that may be useful. These can be acquired via the `isofit download` command, and the current list of downloads we support is available via `isofit download --help`. See [data](../extra_downloads/data.md) for more information.

???+ note

    The default location for downloading extra files is ``~/.isofit/``. First time invoking the ISOFIT CLI will instantiate this directory and an ``isofit.ini`` file for storing the paths to downloaded products.

# Setting Environment Variables

Depending on the selected RTM, specific environment variables pointing to the RTM's base directory have to be set prior to running ISOFIT. In the following, general instructions on how to set these variables on MacOS, Linux and Windows are provided.

## MacOS

Most MacOS systems load environment variables from the user's .bash_profile configuration file. Open this file with your preferred text editor, such as vim:

```
$ vim ~/.bash_profile
```

Add this line to your .bash_profile:

```
$ export VARIABLE_NAME=DIRECTORY (use your actual path)
```

Save your changes and run:

```
$ source ~/.bash_profile
```

## Linux

Most Linux profiles use either bash or csh/tcsh shells.  These shells load environment variables from the user's .bashrc or .cshrc configuration files.

### BASH

Add this parameter to the .bashrc (see MacOS description):

```
$ export VARIABLE_NAME=DIRECTORY (use your actual path)
```

### T/CSH

Add this parameter to the .cshrc (see MacOS description):

```
$ setenv VARIABLE_NAME=DIRECTORY (use your actual path)
```

## Windows

Using a command prompt, type one of the following:

```
$ setx /M VARIABLE_NAME "DIRECTORY" (use your actual path)
$ setx VARIABLE_NAME "DIRECTORY" (use your actual path)
```

# ISOFIT variables

The following environment variables are actively used within ISOFIT:

Variable | Purpose
-|-
`MKL_NUM_THREADS` & `OMP_NUM_THREADS` | These control the threading of various packages within ISOFIT. It is important to set these to `1` to ensure ISOFIT performs to its fullest capabilities. By default, ISOFIT will insert these into the environment if they are not set and/or not set correctly.
`ISOFIT_NO_SET_THREADS` | This will disable automatically setting the MKL and OMP environment variables. Only recommended for advanced users that know what they are doing and can mitigate the consequences.
`ISOFIT_DEBUG` | Disables the `ray` package across ISOFIT to force single-core execution. Primarily used as a debugging tool by developers and is not recommended for normal use.
