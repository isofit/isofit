============
Installation
============

.. contents:: Table of Contents
    :depth: 2

Conda-Forge (Recommended)
-------------------------

Recommended approach!

New environment:

.. code-block:: bash

    mamba create -n isofit_env -c conda-forge isofit
    mamba activate isofit_env

or

.. code-block:: bash

    conda create -n isofit_env -c conda-forge isofit
    conda activate isofit_env

Within an existing environment:

.. code-block:: bash

    mamba install -c conda-forge isofit

or

.. code-block:: bash

    conda install -c conda-forge isofit


PyPI (``pip``)
--------------

.. note::

    The commands below use ``$ pip``, however ``$ python -m pip`` is often a
    safer choice. It is possible for the ``$ pip`` executable to point to a
    different version of Python than the ``$ python`` executable. Using
    ``$ python -m pip`` at least ensures that the package is installed against
    the Python interpreter in use. The issue is further compounded on systems
    that also have ``$ python3`` and ``$ pip3`` executables, or executables for
    specific versions of Python like ``$ python3.11`` and ``$ pip3.11``.

ISOFIT can be installed from the `Python Package Index <https://pypi.org/project/isofit/>`_ with:

.. code-block:: bash

    $ pip install isofit

In order to support a wide variety of environments, ISOFIT does not overly
constrain its dependencies, however this means that in some cases ``pip`` can
take a very long time to resolve ISOFIT's dependency tree. Some users may need
to provide constraints for specific packages, or install ISOFIT last.
``pip`` also supports installing from a remote git repository – this installs
against the ``main`` branch:

.. code-block:: bash

    $ pip install "git+https://github.com/isofit/isofit.git@main"


Manual (GitHub)
---------------

We recommend using `Mamba <https://mamba.readthedocs.io/en/latest/>`_ to create a virtual environment:

.. code-block:: bash

    $ git clone https://github.com/isofit/isofit
    $ mamba env create -f isofit/recipe/isofit.yml
    $ mamba activate isofit_env
    $ pip install -e ./isofit

Developers may need to install additional packages provided by alternative YAML files in the `recipe` directory:

.. code-block:: bash

    $ micromamba install --name isofit_env --file ISOFIT/recipe/docker.yml


Downloading Extra Files
=======================

Once ISOFIT is installed, the CLI provides an easy way to download additional files that may be useful.
These can be acquired via the ``isofit download`` command, and the current list of downloads we support is available via ``isofit download --help``.
See :ref:`data` for more information.

> **_NOTE:_**  The default location for downloading extra files is ``~/.isofit/``. First time invoking the ISOFIT CLI will instantiate this directory and an ``isofit.ini`` file for storing the paths to downloaded products.


Setting Environment Variables
=============================

Depending on the selected RTM, specific environment variables pointing to the RTM's base directory have to be set prior to running ISOFIT.
In the following, general instructions on how to set these variables on MacOS, Linux and Windows are provided.

MacOS
-----

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
-----

- Most Linux profiles use either bash or csh/tcsh shells.  These shells load environment variables from the user's .bashrc or .cshrc configuration files.

- (BASH) Add this parameter to the .bashrc (see MacOS description):

.. code::

    export VARIABLE_NAME=DIRECTORY (use your actual path)

- (T/CSH) Add this parameter to the .cshrc (see MacOS description):

.. code::

    setenv VARIABLE_NAME=DIRECTORY (use your actual path)

Windows
-------

- Using a command prompt, type one of the following:

.. code::

    setx /M VARIABLE_NAME "DIRECTORY" (use your actual path)

    setx VARIABLE_NAME "DIRECTORY" (use your actual path)

ISOFIT variables
----------------

The following environment variables are actively used within ISOFIT:

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - Variable
      - Purpose
    * - ``MKL_NUM_THREADS``, ``OMP_NUM_THREADS``
      - These control the threading of various packages within ISOFIT. It is important to set these to ``1`` to ensure ISOFIT performs to its fullest capabilities. By default, ISOFIT will insert these into the environment if they are not set and/or not set correctly.
    * - ``ISOFIT_NO_SET_THREADS``
      - This will disable automatically setting the MKL and OMP environment variables. Only recommended for advanced users that know what they are doing and can mitigate the consequences.
    * - ``ISOFIT_DEBUG``
      - Disables the ``ray`` package across ISOFIT to force single-core execution. Primarily used as a debugging tool by developers and is not recommended for normal use.


====================================================
Quick Start with sRTMnet (Recommended for new users)
====================================================

sRTMnet is an emulator for MODTRAN 6, that works by coupling a neural network with a surrogate RTM (6S v2.1).


Automatic (Recommended)
-----------------------

ISOFIT can automatically install 6S and sRTMnet with the latest versions:

.. code::

    $ isofit download sixs
    $ isofit download srtmnet

The above commands will ensure these models are built and available for ISOFIT.

.. note::

  A commonly useful option ``-b [path]``, ``--base [path]`` will set the download location for all products:

  .. code-block::

      $ isofit -b extra-downloads/ download all

  This will change the download directory from the default ``~`` to ``./extra-downloads/``

  See :ref:`data` for more information.

Manual (Advanced)
-----------------

The following procedure walks through the steps required to install sRTMnet manually:

#. Download `6S v2.1 <https://salsa.umd.edu/files/6S/6sV2.1.tar>`_, and compile.
   If you use a modern system, it is likely you will need to specify a legacy compiling configuration by changing line 3 of the Makefile to::

      EXTRA = -O -ffixed-line-length-132 -std=legacy

#. Configure your environment by pointing the SIXS_DIR variable to point to your installation directory.

#. Download the `pre-trained sRTMnet neural network <https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/sRTMnet_v120.h5>`_, as well as some `auxiliary data <https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/sRTMnet_v120_aux.npz>`_.
   This will give you an hdf5 and an aux file. It is important that you store both in the same directory.

   You will likely need to set the path to 6S and sRTMnet for the ISOFIT ini file as well as rebuild the examples.
   To do this, execute::

      $ isofit --path sixs /path/to/sixs/ --path srtmnet /path/to/sRTMnet/ build

#. Run one of the following examples:

.. code::

    # Small example pixel-by-pixel
    $ cd $(isofit path examples)/image_cube/small/
    $ ./default.sh

.. code::

    # Medium example with empirical line solution
    $ cd $(isofit path examples)/image_cube/medium/
    $ ./empirical.sh

.. code::

    # Medium example with analytical line solution
    $ cd $(isofit path examples)/image_cube/medium/
    $ ./analytical.sh


Quick Start using MODTRAN 6.0
=============================

This quick start presumes that you have an installation of the MODTRAN 6.0 radiative transfer model. This is the
preferred radiative transfer option if available, though we have also included interfaces to the open source
LibRadTran RT code as well as to neural network emulators.

#. Create an environment variable MODTRAN_DIR pointing to the base MODTRAN 6.0 directory.

#. Run the following code::

    $ cd $(isofit path examples)/20171108_Pasadena
    $ ./modtran.sh

#. This will build a surface model and run the retrieval. The default example uses a lookup table approximation, and the code should recognize that the tables do not currently exist. It will call MODTRAN to rebuild them, which will take a few minutes.

#. Look for output data in ``$(isofit path examples)/20171108_Pasadena/output/``.


Known Incompatibilities
=======================

Ray may have compatibility issues with older machines with glibc < 2.14.

Additional Installation Info for Developers
===========================================

Be sure to read the :ref:`contributing` page as additional installation steps must be performed.
