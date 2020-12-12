Imaging Spectrometer Optimal FITting (ISOFIT) Overview
======================================================

You can find documentation for isofit at our `readthedocs site <https://isofit.readthedocs.io/en/latest/index.html>`_.

This repository has two main branches:

`current-release <https://github.com/isofit/isofit/tree/current-release/>`__: Latest stable, versioned release and

`master <https://github.com/isofit/isofit/>`__: The latest non-breaking changes, sometimes pre-version

This codebase contains a set of routines and utilities for fitting surface,
atmosphere and instrument models to imaging spectrometer data.  It is
written primarily in Python, with JSON format configuration files and some
dependencies on widely-available numerical and scientific libraries such as
scipy, numpy, and scikit-learn.  It is designed for maximum flexibility, so
that users can swap in and evaluate model components based on different
radiative transfer models (RTMs) and various statistical descriptions of
surface, instrument, and atmosphere.  It can run on individual radiance
spectra in text format, or imaging spectrometer data cubes.

The subdirectories contain:

* bin/       - command line scripts for calling isofit and sunposition
* data/      - shared data files
* docs/      - documentation
* examples/  - example runs packaged with input data and configuration files
* isofit/    - the isofit Python module including utilities and tests
* logs/      - Pytest logs
* recipe/    - conda release recipe

If you use ISOFIT in your research or production, we ask that you cite the 
precursor publication:

  Thompson, David R., Vijay Natraj, Robert O. Green, Mark C. Helmlinger, Bo-Cai Gao, and Michael L. Eastwood. "Optimal estimation for imaging spectrometer atmospheric correction." Remote Sensing of Environment 216 (2018): 355-373. 


Installation Instructions
-------------------------

From Github
***********

The code repository, development branches, and user community are found on
`GitHub <https://github.com/davidraythompson/isofit>`_. To install:

1. Download or clone the git repo located at https://github.com/isofit/isofit, using either the `current-release <https://github.com/isofit/isofit/tree/current-release>`_ or `master (current-release + reviewed development) <https://github.com/isofit/isofit>`_ branch.

2. Install the ISOFIT using pip - be sure to use a full path reference.

.. code::

    pip install --editable /path/to/isofit --use-feature=2020-resolver

From PyPI
*********

Also, the latest release is always hosted on `PyPI <https://pypi.python.org/pypi/isofit>`_,
so if you have `pip` installed, you can install ISOFIT from the command line with

.. code::

    pip install isofit

This will install the "isofit" package into your environment as well as its dependencies.

Using Utils
***********

Several utilities are provided to facilitate using ISOFIT in different workflows.  Some
of the utilities (such as `apply_oe.py <https://github.com/isofit/isofit/blob/master/isofit/utils/apply_oe.py>`_)
require GDAL, which is not required in setup.py currently to facilitate diverse compatibility.
An example installation is available in the `utils workflow <https://github.com/isofit/isofit/blob/master/.github/workflows/utils-workflow.yml>`_

Quick Start using MODTRAN 6.0
-----------------------------

This quick start presumes that you have an installation of the MODTRAN 6.0
radiative transfer model.  This is the preferred radiative transfer option if available, though we have also included an interface to the open source LibRadTran RT code.  Other open source options and neural network emulators will be integrated in the future. 

1. Configure your environment with the variable MODTRAN_DIR pointing to the base MODTRAN 6.0 directory.

2. Run the following code

.. code::

    cd examples/20171108_Pasadena
    ./run_examples_modtran.sh

3. This will build a surface model and run the retrieval. The default example uses a lookup table approximation, and the code should recognize that the tables do not currently exist.  It will call MODTRAN to rebuild them, which will take a few minutes.

4. Look for output data in examples/20171108_Pasadena/output/.  Each retrieval writes diagnostic images to examples/20171108_Pasadena/images/ as it runs.

Quick Start with LibRadTran 2.0.x
---------------------------------

This quick start presumes that you have an installation of the open source libRadTran radiative transfer model (`LibRadTran <http://www.libradtran.org/doku.php>`)_ .  We have tested with the 2.0.2 release.  You will need the "REPTRAN" absorption parameterization - follow the instructions on the libradtran installation page to get that data.

1. Configure your environment with the variable LIBRADTRAN_DIR pointing to the base libRadTran directory.

2. Run the following code

.. code::

    cd examples/20171108_Pasadena
    ./run_example_libradtran.sh

3. This will build a surface model and run the retrieval. The default example uses a lookup table approximation, and the code should recognize that the tables do not currently exist.  It will call libRadTran to rebuild them, which will take a few minutes.

4. Look for output data in examples/20171108_Pasadena/output/.  Diagnostic images are written to examples/20171108_Pasadena/images/.

Additional Installation Info for Mac OSX
------------------------------------------

1. Install the command-line compiler

.. code::

  xcode-select --install

2. Download the python3 installer from https://www.python.org/downloads/mac-osx/


Known Incompatibilities
-----------------------
Ray may have compatability issues with older machines with glibc < 2.14.
