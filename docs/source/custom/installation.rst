.. _installation:

Installation
============

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


Setting environment variables
-----------------------------

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
-----------------------------

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
---------------------------------

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
------------------------

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
------------------------------------------

1. Install the command-line compiler

.. code::

  xcode-select --install

2. Download the python3 installer from https://www.python.org/downloads/mac-osx/


Known Incompatibilities
-----------------------
Ray may have compatability issues with older machines with glibc < 2.14.