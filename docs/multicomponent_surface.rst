The Multicomponent Surface Model 
================================

This section describes the surface routine *surfmodel.py* used to generate multicomponent surface model files.  These are used by ISOFIT for both the multicomponent_surface and glint_surface models, among others.  The casual user may not need it; they can simply use one of the provided land or water surface model options which should be fairly universal.  However, specific applications may desire other kinds of surface prior distributions, such as restricted priors designed for optimum retrievals over well-understood terrain.  

The multicomponent model represents the surface prior distribution using a collection of multivariate Gaussian means and covariance matrices, with one dimension per instrument channel.  For each iteration, the Gaussian closest to the current reflectance estimate is used as the prior. 

The surfmodel.py Utility  
------------------------

The routine utils/surfmodel.py generates multicomponent surface reflectance models by fitting multivariate Gaussians to a library of reflectance spectra.  It accepts a single argument corresponding to a JSON configuration file:   

.. code-block:: bash

  python3 surfmodel.py <configuration_file.json>

It need be run only once at any time before the retrievals toprepare the model. 

Configuration Files 
-------------------

 The configuration file determines the fitting procedure, spectral sampling and normalization strategies.  It is a JSON file, a single dictionary that contains the following: 

* **reference_windows**: A list of lists, where each sublist has two elements corresponding to the beginning and end wavelength of a reference interval used for normalization.  We recommend specifying intervals that exclude very deep, opaque atmospheric absorption features. 

* **output_model_file**: The path to the output location where surfmodel.py writes its fitting result.  The format is given in the next chapter; by convention it is a MATLAB-format file suffixed with ".mat"

* **normalize** : Normalization strategy, one of "Euclidean", "RMS", or "None".  Unless "None", reflectance model components will be normalized.  At runtime, ISOFIT will normalize its current reflectance estimate prior to selecting a component, and then rescale the component to the appropriate magnitude to form the prior.  Normalized models do not constrain the magnitudes of surface reflectances, but just the shape.  Only the wavelength intervals specified by "reference_windows" (below) contribute to the normalization.  In theory, this normalization decision would not have to made in the model creation step, but could  be handled entriely within ISOFIT. 

* **wavelength_file**: File specifying output wavelengths to which reflectances are resampled. A three column space-delimited ASCII file, with columns containing channel number, channel center wavelength in microns, and Full Width at Half Maximum (FWHM) of the Gaussian spectral response function in microns, respectively.  Currently, surfmodel.py ignores the FWHM column and performs linear interpolation. to resample the library data to the output center wavelengths. 

* **sources**: A list of dictionaries, one per source library used in the fitting.  surfmodel.py fits each library is separately, and appends those components to the general collection. The dictionary for each component contains:

    * **input_spectrum_files**: The path to a dataset of reflectance spectra, given as a binary matrix of four-byte floating point numbers in Band Interleaved by Pixel format.  The dimensions and wavelengths of the input data are specified by a detached human-readable ASCII header in ENVI format.  The header file name is the same as the dataset, but with the suffix ".hdr" appended.  The header must also contain wavelength information for each reflectance spectrum; the spectra will be interpolated linearly to the output wavelength grid given by the "wavelength_file" option.
    * **n_components**:  The number of components used to fit the library spectra.
    * **windows**:A list of dictionaries, where each dictionary specifies fitting parameters for a different wavelength interval. Breaking the spectral range into intervals allows surfmodel.py to control and modify the fitting strategy independently for each range.  This is useful for adjusting the degree of regularization, restricting the prior flexibility in the surface in areas of critical atmospehric information. The dictionary always contains three key-value pairs:

         * **interval**: A two-element list of numbers, the beginning and end wavelengths of the range. 
         * **regularizer**: A regularization value added to the diagonal of the covariance matrix for this range.  Values are in "variance of surface reflectance" units.  For example, a value of 1e-6 corresponds to an extra regularization having a standard deviation of 0.1% in Lambertian surface reflectance values.  Regularization promotes numerical stability and flexibility in the refelctance model. 
         * **correlation**: Either "EM" or "decorrelated."  If the former, off-diagonal elements are estimated directly from the data.  If the latter, off-diagonal elements are set to zero. 

Model Definition Files 
----------------------
The surface model is stored in MATLAB ('.mat') format, and contains the following fields: 

* **means**: a 2D array of component means sized [(number of components) x (number of wavelengths)]
* **covs**: a 3D array of covariance matrices sized [(number of components) x (number of channels) x (number of channels)] 
* **wl**: a vector of center wavelengths for each channel that should ideally match the instrument description; 
* **normalize**: modifies the magnidude of the components to match the state vector, and can be either "None", "Euclidean", or "RMS" 
* **refwl**: A vector of reference wavelengths used for the normalization and distance comparison.  Careful selection of "refwl" ensures that spectra are not compared using (for example) the deep water absorption features or other low signal areas.



