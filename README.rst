=============================================
ISOFIT - Imaging Spectrometer Optimal FITting
=============================================

Welcome to ISOFIT 3x.  This is a major update to the ISOFIT codebase, and is not backwards compatible with ISOFIT 2x.
To view the previous version of ISOFIT, please see `dev_2x <https://github.com/isofit/isofit/tree/dev_2x>`__. Updates
and performance enhancements are still underway, but testing and feedback are encouraged! A list of new 3x features is
compiled below.


ISOFIT contains a set of routines and utilities for fitting surface, atmosphere and instrument models to imaging
spectrometer data. It is written primarily in Python, with JSON format configuration files and some dependencies on
widely-available numerical and scientific libraries such as scipy, numpy, and scikit-learn. It is designed for maximum
flexibility, so that users can swap in and evaluate model components based on different radiative transfer models (RTMs)
and various statistical descriptions of surface, instrument, and atmosphere. It can run on individual radiance spectra
in text format, or imaging spectrometer data cubes.

* Please check the documentation_ for installation and usage instructions and in depth information.

* There are three main branches:

 * `main <https://github.com/isofit/isofit/tree/main>`__ (in-line with the current release)
 * `dev <https://github.com/isofit/isofit/tree/dev>`__ (for activate development of ISOFIT 3x)
 * `dev_2x <https://github.com/isofit/isofit/tree/dev_2x>`__ (archived version of ISOFIT 2x)

* Information on how to **cite the ISOFIT Python package** can be found in the
  `CITATION <https://github.com/isofit/isofit/blob/dev/CITATION.cff>`__ file.


License
-------
Free software: Apache License v2

All images contained in any (sub-)directory of this repository are licensed under the CC0 license which can be found
`here <https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt>`__.

Major ISOFIT 3x features
------------------------

* new handling of look-up-tables (LUTs), including the option to provide custom prebuilt LUTs
* centralized radiative transfer physics for more flexible development and experimentation
* test coverage for major functionality
* click command line utilities, including download of external data and example files
* a more flexible isofit.ini file used to discover various paths such as tests, data, and examples
* instructions for dev environment setup and a collection of setup scripts
* numpy implementation of the sRTMnet emulator (removes tensorflow dependency)

Basic features
--------------

* utilities for fitting surface, atmosphere and instrument models to imaging spectrometer data
* a selection of radiative transfer models (RTMs) incl. MODTRAN and 6S
* sRTMnet emulator for MODTRAN 6 by coupling a neural network with a surrogate RTM (6S v2.1)
* various statistical descriptions of surface, instrument, and atmosphere
* application to both individual radiance spectra and imaging spectrometer data cubes
* custom instrument models to handle new sensors
* observation uncertainties to account for model discrepancy errors
* prior distribution based on background knowledge of the state vector

Status
------

|badge1| |badge2| |badge3| |badge4| |badge5| |badge6| |badge7| |badge8|

.. |badge1| image:: https://img.shields.io/static/v1?label=Documentation&message=readthedocs&color=blue
    :target: https://isofit.readthedocs.io/en/latest/index.html

.. |badge2| image:: https://readthedocs.org/projects/pip/badge/?version=stable
    :target: https://pip.pypa.io/en/stable/?badge=stable

.. |badge3| image:: https://img.shields.io/pypi/v/isofit.svg
    :target: https://pypi.python.org/pypi/isofit

.. |badge4| image:: https://img.shields.io/conda/vn/conda-forge/isofit.svg
    :target: https://anaconda.org/conda-forge/isofit

.. |badge5| image:: https://img.shields.io/pypi/l/isofit.svg
    :target: https://github.com/isofit/isofit/blob/master/LICENSE

.. |badge6| image:: https://img.shields.io/pypi/pyversions/isofit.svg
    :target: https://img.shields.io/pypi/pyversions/isofit.svg

.. |badge7| image:: https://img.shields.io/pypi/dm/isofit.svg
    :target: https://pypi.python.org/pypi/isofit

.. |badge8| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6908949.svg
   :target: https://doi.org/10.5281/zenodo.6908949

.. _documentation: https://isofit.readthedocs.io/en/latest/index.html
