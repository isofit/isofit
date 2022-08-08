=====
About
=====

ISOFIT contains a set of routines and utilities for fitting surface, atmosphere and instrument models to imaging
spectrometer data. It is written primarily in Python, with JSON format configuration files and some dependencies on
widely-available numerical and scientific libraries such as scipy, numpy, and scikit-learn. It is designed for maximum
flexibility, so that users can swap in and evaluate model components based on different radiative transfer models (RTMs)
and various statistical descriptions of surface, instrument, and atmosphere. It can run on individual radiance spectra
in text format, or imaging spectrometer data cubes.

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
