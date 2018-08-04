Release Notes (version 0.3.0)
=============================

*August 4, 2018*

We are very excited to share this release of ISOFIT with the community!

The routines and utilities in this codebase fit surface, atmosphere and instrument models to imaging spectrometer data using an Optimal Estimation (OE) approach.  This method can be used to perform atmospheric correction given a measured calibrated radiance file, either as a single spectra text file or as an imaging spectrometer data cube.

The project is undergoing rapid development, and we are providing an early release with the understanding that there will be many changes and improvements in the near future. 

Release 0.3.0 includes a Markov Chain Monte Carlo (MCMC) inversion method to sample directly from the posterior.  This functionality should be considered highly experimental! It also contains minor improvements:

* Better aerosol parameterization 
* PEP8 style compliance unit tests
* Better PEP8 compliance overall

Examples
---------

We have included examples of retrievals running on individual radiance spectra in text format. These spectra were acquired by JPL's Airborne Visible/Infrared Imaging Spectrometer Next Generation (AVIRIS-NG) instrument over Pasadena on 8 November 2017.  Measured radiance files can found in the "examples/20171108_Pasadena/remote" folder and the full configuration files for running the retrieval can be found in the "examples/20171108_Pasadena/configs" folder.

Supported Radiative Transfer Codes
----------------------------------

We designed he ISOFIT code for flexibility and modularity.  It allows for different implementations of radiative transfer codes including open-source options.  Currently two interfaces are supported: MODTRAN 6.0 (preferred) and the open source package `LibRadTran <http://www.libradtran.org/doku.php>`_.

Roadmap
-------

We are developing the following upgrades for the near future:

* Add additional radiative transfer model interfaces
* Optimize code execution speed for improved performance
* Remove dependency on "spectral" package
* Improve unit test coverage 
* Augment documentation

Known Issues
------------

Issues will be added as they are discovered.  See the `issue tracker <https://github.com/davidraythompson/isofit/issues>`_ for more details.
