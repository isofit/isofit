===============
Extra Downloads
===============

To get started with ISOFIT examples, simply execute the two following commands:

.. code-block::

    $ isofit download all
    $ isofit build

The first will download all additional ISOFIT dependencies and configure them for the current system.
The second will build the ISOFIT examples using the configured dependencies.
From there, examples will be available under ``~/.isofit/examples/``.
Each subdirectory will have one or more scripts that are prepared for execution.

If there are any issues, please report them to the `ISOFIT repository<https://github.com/isofit/isofit/issues>`_.

The contents below go into further details about additional commands.

=======================
Extra Downloads Options
=======================

.. contents:: Table of Contents
    :depth: 2

ISOFIT uses INI files to configure the location of extra dependencies that are not included in the default ISOFIT installation.
These include things like larger data files and the ISOFIT examples.

.. note::

    The below commands assume a user is in their ``/Users/[user]/`` directory, aka ``~``

When the ``isofit`` command is first executed, it will create a directory under the user's home directory named ``.isofit`` as well as initialize a default ``isofit.ini`` file:

.. code-block::

    $ isofit
    Wrote to file: /Users/[user]/.isofit/isofit.ini

    $ cat /Users/[user]/.isofit/isofit.ini
    [DEFAULT]
    data = /Users/[user]/.isofit/data
    examples = /Users/[user]/.isofit/examples
    imagecube = /Users/[user]/.isofit/imagecube
    srtmnet = /Users/[user]/.isofit/srtmnet
    sixs = /Users/[user]/.isofit/sixs
    modtran = /Users/[user]/.isofit/modtran

Notice the default location for all paths is ``~/.isofit/``. These can be modified by either directly editing the INI file or by using the ISOFIT CLI:

.. code-block::

    $ isofit --help
    Usage: isofit [OPTIONS] COMMAND [ARGS]...

      This houses the subcommands of ISOFIT

    Options:
      -v, --version           Print the current version
      -i, --ini TEXT          Override path to an isofit.ini file
      -b, --base TEXT         Override the base directory for all products
      -s, --section TEXT      Switches which section of the ini to use
      -d, --data TEXT         Override path to data directory
      -e, --examples TEXT     Override path to examples directory
      -c, --imagecube TEXT    Override path to imagecube data directory
      -em, --srtmnet TEXT     Override path to sRTMnet installation
      -6s, --sixs TEXT        Override path to SixS installation
      --save / -S, --no-save  Save the ini file
      --help                  Show this message and exit.

Using a data override flag (``-d``, ``-e``, ``-c``, ``-em``, ``-6s``) will update the the INI with the provided path:

.. code-block::

    $ isofit -e tutorials
    Wrote to file: /Users/[user]/.isofit/isofit.ini

    $ isofit download paths
    Download paths will default to:
    - data = /Users/[user]/.isofit/data
    - examples = /Users/[user]/tutorials
    - imagecube = /Users/[user]/.isofit/imagecube
    - srtmnet = /Users/[user]/.isofit/srtmnet
    - sixs = /Users/[user]/.isofit/sixs
    - modtran = /Users/[user]/.isofit/modtran

For advanced users, the INI file itself as well as the base directory and the section of the INI may be modified:

.. code-block::

    $ isofit -i test.ini -b test -s test -d test
    Wrote to file: test.ini

    $ cat test.ini
    [DEFAULT]
    data = /Users/[user]/.isofit/data
    examples = /Users/[user]/tutorials
    imagecube = /Users/[user]/.isofit/imagecube
    srtmnet = /Users/[user]/.isofit/srtmnet
    sixs = /Users/[user]/.isofit/sixs
    modtran = /Users/[user]/.isofit/modtran

    [test]
    data = /Users/[user]/dev/test
    examples = /Users/[user]/dev/test/examples
    imagecube = /Users/[user]/dev/test/imagecube
    srtmnet = /Users/[user]/dev/test/srtmnet
    sixs = /Users/[user]/dev/test/sixs
    modtran = /Users/[user]/dev/test/modtran

The ``DEFAULT`` section is still instantiated, but now there's a ``test`` section with a different ``data`` path than the default.
Also note the default ``examples`` is different -- this is because the above examples changed it in the default INI, which is still read if available.

Additionally, these paths may be used in command-line arguments via the ``isofit path`` command. For example:

.. code-block::

    $ cd $(isofit path examples)
    $ ls $(isofit path data)/reflectance
    $ cd $(isofit -i test.ini -s test path srtmnet)

Downloads
=========

ISOFIT comes with a ``download`` command that provides users the ability to download and install extra files such as larger data files and examples.
To get started, execute the ``isofit download --help`` in a terminal. At this time, there are 7 subcommands:

.. list-table::
    :widths: 25 75
    :header-rows: 1

    * - Command
      - Description
    * - ``paths``
      - Displays the currently configured path for a download
    * - ``all``
      - Executes all of the download commands below
    * - ``data``
      - Downloads ISOFIT data files from https://github.com/isofit/isofit-data
    * - ``examples``
      - Downloads the ISOFIT examples from https://github.com/isofit/isofit-tutorials
    * - ``imagecube``
      - Downloads required data for the image_cube example
    * - ``sRTMnet``
      - Downloads the sRTMnet model
    * - ``sixs``
      - Downloads and builds 6sv-2.1


The paths for each download are defined in the currently active INI.
Download paths can be modified by either directly modifying the ``~/.isofit/isofit.ini`` or by using ``isofit --help`` flags (shown above).
Additionally, download paths may be temporarily overridden and not saved to the active INI by providing a ``--output [path]``. For example:

.. code-block::

    $ isofit download data --help
    Usage: isofit download data [OPTIONS]

    Downloads the extra ISOFIT data files from the repository
    https://github.com/isofit/isofit-data.

    Run ``isofit download paths`` to see default path locations.
    There are two ways to specify output directory:
      - ``isofit --data /path/data download data``: Override the ini file. This will save the provided path for future reference.
      - ``isofit download data --output /path/data``: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.

    Options:
    -o, --output TEXT  Root directory to download data files to, ie. [path]/data
    -t, --tag TEXT     Release tag to pull  [default: latest]
    --help             Show this message and exit.

Some subcommands have additional flags to further tweak the download, such as ``data`` and ``examples`` having a ``--tag`` to download specific tag releases, or ``sRTMnet`` having ``--version`` for different model versions, but it is recommended to use the default to pull the most up-to-date download for each.


Building
========

ISOFIT examples rely on the ``isofit build`` command to generate configuration files and scripts dependent on a user's active INI file.
Each example contains a set of template files generate the required files for the example.
By default, a user will not need to modify these templates.
If an advanced user desires to change the configuration of an example, it is strongly recommended to run the build command first and edit the generated outputs.
However, every example should work out-of-the-box with the default downloads and build.

==========
Developers
==========

This section is specifically for developers seeking to expand either the examples.


Creating Examples
=================

ISOFIT leverages specially-designed templates to build the example configurations depending on the installation environment defined by an INI.
Creating a new example must define one or more templates for the given example type.


Templates
---------

Templates are used to generate configuration and script files relative to a user's installation environment.
Changes to the ISOFIT INI may rebuild the examples quickly for a new environent.
Instead of hardcoding relative paths, the ``isofit build`` command will replace values within the templates with the values defined by a given INI.
For example, a template may define ``{examples}``, this will be replaced with the INI's ``examples`` string.

There are two types of examples supported at this time:

1. Direct ``Isofit`` calls. These examples build configuration files to pass directly into the ``Isofit`` class to call ``.run()``

For existing examples of this type include `SantaMonica<https://github.com/isofit/isofit-tutorials/tree/main/20151026_SantaMonica>`_, `Pasadena<https://github.com/isofit/isofit-tutorials/tree/main/20171108_Pasadena>`_, and `ThermalIR<https://github.com/isofit/isofit-tutorials/tree/main/20190806_ThermalIR>`_.
Depending on the example, extra directories may be included such as prebuilt simulation files in the ``lut`` directory.

A bash and python script will be generated for each directory under the templates directory. For example, given a template directory:

.. code-block::

    [example]/
    └─ templates/
      ├─ reduced/
      | ├─ config1.json
      | └─ config2.json
      ├─ advanced/
      | └─ config3.yml
      └─ surface.json

will generate the following configs and scripts:

.. code-block::

    [example]/
    └─ configs/
    | ├─ reduced/
    | | ├─ config1.json
    | | └─ config2.json
    | ├─ advanced/
    | | └─ config3.json
    | └─ surface.json
    ├─ reduced.sh
    ├─ reduced.py
    ├─ advanced.sh
    └─ advanced.py

Each script will have the configs for it. For example, ``reduced.sh`` would contain:

.. code-block::

    # Build a surface model first
    echo 'Building surface model: surface.json'
    isofit surface_model ~/.isofit/examples/[example]/configs/surface.json

    # Now run retrievals
    echo 'Running 1/2: config1.json'
    isofit run --level DEBUG ~/.isofit/examples/[example]/configs/reduced/config1.json

    echo 'Running 2/2: config2.json'
    isofit run --level DEBUG ~/.isofit/examples/[example]/configs/reduced/config2.json


2. ``apply_oe`` scripts. These examples use templates to define the arguments for a call to the ``isofit apply_oe`` utility.

Existing examples of this type include the `small<https://github.com/isofit/isofit-tutorials/tree/main/image_cube/small/templates>`_ and ``medium image cube<https://github.com/isofit/isofit-tutorials/tree/main/image_cube/medium/templates>`_ examples.
These templates are a list of arguments in a ``[name].args.json`` file. For each ``[name]`` file, separate scripts will be generated.
For example, given the following templates:

.. code-block::

    [example]/
    └─ templates/
      ├─ simple.args.json
      └─ advanced.args.json

will generate the following scripts:

.. code-block::

    [example]/
    ├─ simple.sh
    └─ advanced.sh

The small image cube example's ``default.args.json`` is currently defined as:

.. code-block:: json

    [
    "{imagecube}/medium/ang20170323t202244_rdn_7k-8k",
    "{imagecube}/medium/ang20170323t202244_obs_7k-8k",
    "{imagecube}/medium/ang20170323t202244_loc_7k-8k",
    "{examples}/image_cube/medium",
    "ang",
    "--surface_path {examples}/image_cube/medium/configs/surface.json",
    "--emulator_base {srtmnet}/sRTMnet_v120.h5",
    "--n_cores {cores}",
    "--presolve",
    "--segmentation_size 400",
    "--pressure_elevation"
    ]

This will generate ``default.sh``:

.. code-block::

    isofit apply_oe \
      ~/.isofit/examples/imagecube/small/ang20170323t202244_rdn_7000-7010 \
      ~/.isofit/examples/imagecube/small/ang20170323t202244_obs_7000-7010 \
      ~/.isofit/examples/imagecube/small/ang20170323t202244_loc_7000-7010 \
      ~/.isofit/examples/examples/image_cube/small \
      ang \
      --surface_path ~/.isofit/examples/examples/image_cube/small/configs/surface.json \
      --n_cores 10 \
      --presolve \
      --segmentation_size 400 \
      --pressure_elevation


Once the the example with its templates are finalized, it must be integrated into the `ISOFIT Tutorials<https://github.com/isofit/isofit-tutorials>`_ repository.
Create a new pull request with a description of the example being created and maintainers will review it then merge and release a new version.
