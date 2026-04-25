# ISOFIT — Imaging Spectrometer Optimal FITting

[![Documentation](https://img.shields.io/static/v1?label=Docs&message=isofit.github.io&color=blue)](https://isofit.github.io/isofit/latest/)
[![PyPI version](https://img.shields.io/pypi/v/isofit.svg)](https://pypi.python.org/pypi/isofit)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/isofit.svg)](https://anaconda.org/conda-forge/isofit)
[![License](https://img.shields.io/pypi/l/isofit.svg)](https://github.com/isofit/isofit/blob/master/LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/isofit.svg)](https://img.shields.io/pypi/pyversions/isofit.svg)
[![Downloads](https://static.pepy.tech/badge/isofit)](https://pepy.tech/project/isofit)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6908949.svg)](https://doi.org/10.5281/zenodo.6908949)

ISOFIT is a Python toolkit for retrieving surface reflectance, atmospheric state, and instrument calibration from imaging spectrometer radiance data. It is built around the [optimal estimation](https://isofit.github.io/isofit/latest/information/bibliography/) framework and is designed for flexibility — users can combine different radiative transfer models, surface priors, and uncertainty models to suit their sensors and science questions. The effort began with a straightforward optimal estimation [implementation](https://doi.org/10.1016/j.rse.2018.07.003), and has grown considerably - see the [bibilography](https://isofit.github.io/isofit/latest/information/bibliography/) for the latest science.

> ISOFIT is on version ![PyPI version](https://img.shields.io/pypi/v/isofit.svg).  Major revisions are not backwards compatible with one another due to updates to the configuration files.   See [Release History](https://github.com/isofit/isofit/releases) for older versions.


ISOFIT combines surface models, atmospheric radiative transfer, and instrument characteristics into a forward model.  This forward model is then solved by inverting relative to a measurement, via optimization or MCMC, to generate the Maximum a Posterior estimate of the joint surface and atmospheric state vector.


```
+-------------------------------------------------------------------------+
|                                 isofit                                  |
|                                                                         |
|  +--------+      +---------------+      +----------------------------+  |
|  |        |      |               |      |          forward           |  |
|  |   io   | ===> |   inversion   | <=== |                            |  |
|  |        |      |               |      | +----------+  +----------+ |  |
|  +--------+      +---------------+      | |atmosphere|  | surface  | |  |
|                          |              | +----------+  +----------+ |  |
|                          |              |                            |  |
|                          |              |       +------------+       |  |
|                          |              |       | instrument |       |  |
|                          |              |       +------------+       |  |
|                          |              +----------------------------+  |
|                          |                                              |
+--------------------------|----------------------------------------------+
                           |
                           | produces
                           v
   +------------------------------------------------+
   |    Joint Surface & Atmospheric State Vector    |
   +------------------------------------------------+
```

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Radiative Transfer Models](#radiative-transfer-models)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Components

| Capability | Details |
|---|---|
| **Radiative transfer** | MODTRAN, 6S, LibRadTran, sRTMnet neural-network emulator |
| **Surface models** | Multicomponent Gaussian, Glint, LUT, Emissive (thermal) |
| **Instrument models** | SNR, parametric noise, nonlinear response, and more |

---

## Installation

### Conda (recommended)

```bash
mamba create -n isofit_env -c conda-forge isofit
mamba activate isofit_env
```

[Full Installation Instructions](https://isofit.github.io/isofit/latest/getting_started/installation/)


For all options including developer installs, see the **[Installation guide](https://isofit.github.io/isofit/latest/getting_started/installation/)**.

---

## Quick Start

### 1. Download supporting data and build examples

```bash
isofit download all
isofit build
```

This fetches the sRTMnet emulator, 6S, surface libraries, and pre-configured example datasets into `~/.isofit/`.  
See [Additional Data](https://isofit.github.io/isofit/latest/extra_downloads/data/) for customising download paths.

### 2. Run an example

```bash
cd $(isofit path examples)/image_cube/small
bash default.sh
```

### 3. Run on your own data

Point ISOFIT at a radiance cube, location file, and observation file using a JSON config.  
The [sRTMnet quickstart](https://isofit.github.io/isofit/latest/getting_started/quickstarts/srtmnet/) walks through a complete configuration for new users.

---

## Examples

ISOFIT ships with a range of worked examples covering different sensors, scenes, and retrieval configurations:

| Example | Description |
|---|---|
| `image_cube/small` | Fast analytical retrieval — good first run |
| `20151026_SantaMonica` | AVIRIS-NG scene with 6S radiative transfer |
| `20171108_Pasadena` | Urban hyperspectral scene |
| `20190806_ThermalIR` | Thermal infrared joint VSWIR+TIR retrieval |
| `20231110_Prism_Multisurface` | Multi-component surface model |
| `NEON` | National Ecological Observatory Network data |
| `SeaBASS_prism_001` | Coastal / water target single-pixel case |

Full instructions: **[Running Examples](https://isofit.github.io/isofit/latest/examples/)**

---

## Radiative Transfer Models

ISOFIT supports several RTM backends.  The recommended starting point for new users is **sRTMnet**, a neural-network emulator that requires no licensed software:

- **[sRTMnet quickstart](https://isofit.github.io/isofit/latest/getting_started/quickstarts/srtmnet/)** — automatic install via `isofit download`
- **[MODTRAN quickstart](https://isofit.github.io/isofit/latest/getting_started/quickstarts/modtran/)** — requires a MODTRAN 6 licence


## Contributing

Contributions are welcome — bug reports, tests, documentation, and new features.  
Please read the **[Contributing guide](https://isofit.github.io/isofit/latest/developers/contributing/)** before opening a pull request.

Active development happens on the [`dev`](https://github.com/isofit/isofit/tree/dev) branch.  
Issues and feature requests are tracked in the [issue tracker](https://github.com/isofit/isofit/issues).

---

## Citation

If you use ISOFIT in your work, please cite the relevant papers listed in the **[Bibliography](https://isofit.github.io/isofit/latest/information/bibliography/)**.  
Package citation information is in the [CITATION.cff](https://github.com/isofit/isofit/blob/dev/CITATION.cff) file.

---

## License

Apache License v2 — see [LICENSE](https://github.com/isofit/isofit/blob/master/LICENSE).

Images in this repository are licensed under [CC0](https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt).
