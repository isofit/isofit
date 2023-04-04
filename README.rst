=============================================
ISOFIT - Imaging Spectrometer Optimal FITting
=============================================

ISOFIT contains a set of routines and utilities for fitting surface, atmosphere and instrument models to imaging
spectrometer data. It is written primarily in Python, with JSON format configuration files and some dependencies on
widely-available numerical and scientific libraries such as scipy, numpy, and scikit-learn. It is designed for maximum
flexibility, so that users can swap in and evaluate model components based on different radiative transfer models (RTMs)
and various statistical descriptions of surface, instrument, and atmosphere. It can run on individual radiance spectra
in text format, or imaging spectrometer data cubes.

* Please check the documentation_ for installation and usage instructions and in depth information.

* There are two main branches:

 * `dev <https://github.com/isofit/isofit/tree/dev>`__ (for activate development)
 * `main <https://github.com/isofit/isofit/tree/main>`__ (in-line with the current release)

* Information on how to **cite the ISOFIT Python package** can be found in the
  `CITATION <https://github.com/unbohn/isofit_build_workflow/blob/master/CITATION>`__ file.


License
-------
Free software: Apache License v2

All images contained in any (sub-)directory of this repository are licensed under the CC0 license which can be found
`here <https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt>`__.

Feature overview
----------------

* utilities for fitting surface, atmosphere and instrument models to imaging spectrometer data
* a selection of radiative transfer models (RTMs) incl. MODTRAN, LibRadTran, and 6S
* sRTMnet emulator for MODTRAN 6 by coupling a neural network with a surrogate RTM (6S v2.1)
* various statistical descriptions of surface, instrument, and atmosphere
* application to both individual radiance spectra and imaging spectrometer data cubes
* custom instrument models to handle new sensors
* observation uncertanities to account for model discrepancy errors
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
