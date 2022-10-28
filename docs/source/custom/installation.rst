Installation
============

From conda-forge (recommended)
******************************

ISOFIT can be installed from the conda-forge channel by using different types of package managers. It is highly
recommended to use the Mambaforge version of the Miniforge_ minimal installer. It allows to install the conda package
manager with some useful features pre-configured:

- conda-forge is set as the default (and only) channel
- support for Mamba_ in place of Conda, which is a reimplementation of the conda package manager in C++ that allows parallel downloading and much faster dependency solving
- emphasis on supporting various CPU architectures (x86_64, ppc64le, and aarch64 including Apple M1)

Using Mamba_ (latest version recommended), ISOFIT can be installed by first creating a virtual environment
(optional but recommended), followed by the installation itself:

.. code-block:: bash

    $ mamba create -n isofit_env isofit
    $ mamba activate isofit_env

Alternatively, you can of course install ISOFIT in an already existing environment by simply running:

.. code-block:: bash

    $ mamba install isofit

Of course, you can also use the Conda_ installer from the Anaconda_ or Miniconda_ package managers. The installation
procedure is equal to using Mamba_, but you should make sure to set the conda-forge channel as default prior to
installing any packages:

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda config --set channel_priority strict

Mamba_ or Conda_ are the preferred methods to install ISOFIT, as they will always install the most recent stable
release and automatically resolve all the dependencies.

The Ray_ package, which is a unified framework for scaling AI and Python applications, is currently not available for
MacOS on conda-forge. Furthermore, the Ray conda package is maintained by the community, not the Ray team. While using
a mamba or conda environment, please install Ray from PyPi using pip. Additionally, both the ndsplines library and the
latest version of xxhash are not available on conda-forge, so that they need to be added to the pip command:

.. code-block:: bash

    $ pip install ray ndsplines xxhash --upgrade

From PyPI (not recommended)
***************************

There is also a pip_ installer for ISOFIT. However, please note that ISOFIT depends on some open source packages that
may cause problems when installed with pip. Therefore, we strongly recommend to resolve the following dependencies
before the pip installer is run:

    * gdal>=2.0.0
    * matplotlib-base>=2.2.2
    * ndsplines>=0.1.2
    * numpy>=1.20
    * pandas>=0.24.0
    * pep8>=1.7.1
    * pytest>=3.5.1
    * python-xxhash>=1.2.0
    * pyyaml>=5.3.2
    * ray>=1.2.0
    * scikit-learn>=0.19.1
    * scikit-image >=0.17.0
    * scipy>=1.3.0
    * spectral>=0.19
    * tensorflow>=2.0.1

Then, the pip installer can be run by:

.. code-block:: bash

    $ pip install isofit

If you don't have pip_ installed, this `Python installation guide`_ can guide you through the process.

From Github
***********

Alternatively, you can install ISOFIT from source by cloning the respective repository hosted on Github:

.. code-block:: bash

    $ git clone https://github.com/isofit/isofit

The repository contains an environment file that includes all needed dependencies. It is recommended to create this
specific environment prior to installing ISOFIT from source:

.. code-block:: bash

    $ cd isofit/recipe
    $ mamba env create -f environment_isofit_basic.yml
    $ mamba activate isofit_env

Finally, install ISOFIT in editable mode:

.. code-block:: bash

    $ cd ..
    $ pip install -e .


Setting environment variables
=============================

Depending on the selected RTM, specific environment variables pointing to the RTM's base directory have to be set prior to running ISOFIT.
In the following, general instructions on how to set these variables on MacOS, Linux and Windows are provided.

MacOS
*****

- Most MacOS systems load environment variables from the user's .bash_profile configuration file. Open this file with your preferred text editor, such as vim:

.. code::

    vim ~/.bash_profile

- Add this line to your .bash_profile:

.. code::

    export VARIABLE_NAME=DIRECTORY (use your actual path)

- Save your changes and run:

.. code::

    source ~/.bash_profile

Linux
*****

- Most Linux profiles use either bash or csh/tcsh shells.  These shells load environment variables from the user's .bashrc or .cshrc configuration files.

- (BASH) Add this parameter to the .bashrc (see MacOS description):

.. code::

    export VARIABLE_NAME=DIRECTORY (use your actual path)

- (T/CSH) Add this parameter to the .cshrc (see MacOS description):

.. code::

    setenv VARIABLE_NAME=DIRECTORY (use your actual path)

Windows
*******

- Using a command prompt, type one of the following:

.. code::

    setx /M VARIABLE_NAME "DIRECTORY" (use your actual path)

    setx VARIABLE_NAME "DIRECTORY" (use your actual path)


Quick Start using MODTRAN 6.0
=============================

This quick start presumes that you have an installation of the MODTRAN 6.0 radiative transfer model. This is the
preferred radiative transfer option if available, though we have also included interfaces to the open source
LibRadTran RT code as well as to neural network emulators.

1. Create an environment variable MODTRAN_DIR pointing to the base MODTRAN 6.0 directory.

2. Run the following code

.. code::

    cd examples/20171108_Pasadena
    ./run_examples_modtran.sh

3. This will build a surface model and run the retrieval. The default example uses a lookup table approximation, and the code should recognize that the tables do not currently exist.  It will call MODTRAN to rebuild them, which will take a few minutes.

4. Look for output data in examples/20171108_Pasadena/output/.


Quick Start with LibRadTran 2.0.x
=================================

This quick start requires an installation of the open source LibRadTran radiative transfer model (`LibRadTran <http://www.libradtran.org/doku.php>`_).
A few important steps have to be considered when installing the software, which are outlined below. We have tested with the latest 2.0.4 release.

1. Download and unpack the latest version of LibRadTran:

.. code::

    wget -nv http://www.libradtran.org/download/libRadtran-2.0.4.tar.gz
    tar -xf libRadtran-2.0.4.tar.gz

2. Download and unpack the "REPTRAN" absorption parameterization:

.. code::

    wget -nv http://www.meteo.physik.uni-muenchen.de/~libradtran/lib/exe/fetch.php?media=download:reptran_2017_all.tar.gz -O reptran-2017-all.tar.gz
    tar -xf reptran-2017-all.tar.gz

3. Unpacking REPTRAN will create a folder called 'data' with a subfolder 'correlated_k'. Copy this subfolder to the LibRadTran data directory:

.. code::

    cp -r data/correlated_k libRadtran-2.0.4/data

4. Go to the LibRadTran base directory, configure and compile the software. It's important to set python2 as interpreter and 'ignore-errors' when running the 'make' command:

.. code::

    cd libRadtran-2.0.4
    PYTHON=$(which python2) ./configure --prefix=$(pwd)
    make --ignore-errors

5. Create an environment variable LIBRADTRAN_DIR pointing to the base libRadTran directory.

6. Run the following code

.. code::

    cd examples/20171108_Pasadena
    ./run_example_libradtran.sh

7. This will build a surface model and run the retrieval. The default example uses a lookup table approximation, and the code should recognize that the tables do not currently exist.  It will call LibRadTran to rebuild them, which will take a few minutes.

8. Look for output data in examples/20171108_Pasadena/output/.


Quick Start with sRTMnet
========================

sRTMnet is an emulator for MODTRAN 6, that works by coupling a neural network with a surrogate RTM (6S v2.1).
Installation requires two steps:

1. Download `6S v2.1 <https://salsa.umd.edu/files/6S/6sV2.1.tar>`_, and compile.  If you use a modern system,
it is likely you will need to specify a legacy compiling configuration by changing line 3 of the Makefile to:

.. code::

    EXTRA   = -O -ffixed-line-length-132 -std=legacy

2. Configure your environment by pointing the SIXS_DIR variable to point to your installation directory.

3. Download the `pre-trained sRTMnet neural network <https://zenodo.org/record/4096627>`_, and (for the example below)
point the environment variable EMULATOR_PATH to the base unzipped path.

4. Run the following code

.. code::

    cd examples/image_cube/
    sh ./run_example_cube.sh


Additional Installation Info for Mac OSX
========================================

1. Install the command-line compiler

.. code::

  xcode-select --install

2. Download the python3 installer from https://www.python.org/downloads/mac-osx/


Known Incompatibilities
=======================

Ray may have compatability issues with older machines with glibc < 2.14.


.. _Conda: https://conda.io/docs/
.. _Miniforge: https://github.com/conda-forge/miniforge
.. _Mamba: https://github.com/mamba-org/mamba
.. _Anaconda: https://www.anaconda.com/products/distribution
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _Ray: https://docs.ray.io/en/latest/index.html
