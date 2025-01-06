.. _contributing:

Contributing
============

Thank you for your interest in contributing to ISOFIT! If you are just getting
started, please review the guidelines below to understand how and where you can
best support and contribute to this project.  Typical contributions may include:

* Unit tests
* Code patches
* Feature enhancements
* Documentation improvements
* Bug reports

If you have ideas for new additions, that's great - please contact the maintainers
at the addresses given below, and we can coordinate efforts.  Our general policy
is to for the maintainers to delegate technical authority to individuals to make
changes and additions to specific topics.

.. contents:: Table of Contents
    :depth: 2


Getting Started
---------------

First of all, to get a sense of the project's current status and roadmap, please
be sure to spend some time reviewing issues in the `issue tracker <https://github.com/isofit/isofit/issues>`_.

If you have have discovered a new issue or task, then go ahead and create a `new
issue <https://github.com/isofit/isofit/issues/new>`_.


Fork and Create a Branch
------------------------

ISOFIT follows the `Standard Fork and Pull Request <https://gist.github.com/Chaser324/ce0505fbed06b947d962>`_ workflow.

When you have chosen an issue to work on, start by `forking <https://help.github.com/articles/fork-a-repo/>`_ the repo.

Then, create a branch with a descriptive name.  A good branch name starts with
the issue number you are working on followed by some descriptive text.  For
example, if you are working on issue #314 you would run:

.. code::

  git checkout -b 314-update-docs-libradtran


Developer Environment
---------------------

Generally, the process for creating a Python virtual environment for development
is:

.. code-block:: console

    $ python3 -m venv venv
    $ ./scripts/setup-devenv.sh venv

however, developers on some platforms may need to install additional non-Python
dependencies using an appropriate package manager for their system.

``setup-devenv.sh`` should be run periodically to refresh a development
environment to pick up new dependencies, updates to the ``isofit`` build
process, etc.

Additionally, be sure to download the extra ISOFIT data files and examples via the `isofit download all` command. See :ref:`data` for more.


Testing
-------

Tests live in `isofit/tests/ <isofit/tests/>`_, and are executed using
`pytest <https://pytest.org>_`.

Our development strategy employs continuous integration and unit testing to validate all changes.  We appreciate you writing additional tests for new modifications or features.  In the interest of validating your code, please also be sure to run realistic examples like this:

.. code::

  cd $(isofit path examples)/20171108_Pasadena
  ./modtran.sh


Debug
-----

ISOFIT uses (ray)[https://www.ray.io/] as the multiprocessing backend; however, this package can be unstable for some systems and difficult to develop with. As such, ISOFIT has a debug mode that can be activated via the `ISOFIT_DEBUG` environment variable.
This is the only environment variable of ISOFIT and is strictly for enabling the DEBUG features of the code. The flag simply disables the ray package and instead uses an in-house wrapper module to emulate the functions of ray.
This enables complete circumvention of ray while supporting ray-like syntax in the codebase.

To enable, set the environment variable `ISOFIT_DEBUG` to any value before runtime. For example:

.. code::

  export ISOFIT_DEBUG=1  # Enable debug
  python isofit.py ...
  export ISOFIT_DEBUG="" # Disable debug

Additionally, you may pass it as a temporary environment variable via:

.. code::

  ISOFIT_DEBUG=1 python isofit.py ...


A Note about Style
------------------

We use (Black)[https://github.com/psf/black] and (isort)[https://github.com/PyCQA/isort] to maintain style consistency.
These are included via (pre-commit)[https://pre-commit.com] and should be installed once ISOFIT is installed. To do so, simply run:

.. code::

  pre-commit install

Every commit from here on will auto-apply the above packages. Additionally, upon a PR to the `dev` branch, `Black` consistency will be checked.
Any PRs failing this check will be rejected by the maintainers until it is passing.

If you must apply Black manually, you must first `pip install black` and then run `black isofit` from the root of the repository.

Implement Your Changes and Create a Pull Request
------------------------------------------------

At this point, you are ready to implement your changes!

As you develop, you should make sure that your branch doesn't veer too far from
ISOFIT's dev branch.  To do this, switch back to your dev branch and make
sure it's up to date with ISOFIT's dev branch:

.. code::

  git remote add upstream https://github.com/isofit/isofit.git
  git checkout dev
  git pull upstream dev


Then update your feature branch from your local copy of dev, and push it!

.. code::

  git checkout 314-update-docs-libradtran
  git rebase dev
  git push --set-upstream origin 314-update-docs-libradtran


When you are ready to submit your changes back to the ISOFIT repo, go to GitHub
and make a `Pull Request <https://help.github.com/articles/creating-a-pull-request/>`_

Keeping your Pull Request Updated
---------------------------------

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code
has changed, and that you need to update your branch so it's easier to merge.

Here's the suggested workflow:

.. code::

  git checkout 314-update-docs-libradtran
  git pull --rebase upstream dev
  git push --force-with-lease 314-update-docs-libradtran

Project Decision Making
-----------------------

Minor changes follow an expedited acceptance process.  These are things like:

* Bug fixes
* Unit tests
* Documentation
* Consolidation that does not change algorithm results or provide significant new functionality
* New functionality initiated by maintainers, or over which authority has been delegated in advance by maintainers (e.g. through issue assignment)

Minor change pull requests are accepted once they:

* Pass unit tests and adhere to project coding conventions
* Get signoff from at least one maintainer, with no objections from any other maintainer

Accepted minor changes will be released in the next major or minor release version. Hotfixes will be expedited as needed.

Major changes include:

* New functionality, including examples, data, and algorithm changes, over which authority was not delegated in advance.
* Official releases
* Project policy updates

These are accepted through consensus of a quorum of maintainers.  **If you would like to include any new algorithms or examples, we highly recommend that they are supported by peer reviewed scientific research.**

Release Steps (for Maintainers)
-------------------------------

Releases should trigger a new PyPi upload, and subsequently a fresh upload to conda-forge.  Therefore,
the revised steps for versioning are:

* Submit version number change to setup.cfg in dev
* Trigger a PR from dev to main
* Accept the PR
* Go to https://github.com/isofit/isofit/releases
* Click "Draft a new release"
* Enter tag version as "v3.8.0" (depending on latest version), and input release title and description
* Click "Publish release"

Contributors
------------

The github maintainers, responsible for handling pull requests, are:

* David R. Thompson: david.r.thompson@jpl.nasa.gov
* Philip G. Brodrick philip.brodrick@jpl.nasa.gov
* Niklas Bohn urs.n.bohn@jpl.nasa.gov

Thanks to the following regular contributors:

* Alexey Shiklomanov (NASA Goddard)
* James Montgomery (NASA JPL)
* Jay Fahlen (NASA JPL)
* Kevin Wurster (Planet)
* Nimrod Carmon (NASA JPL)
* Regina Eckert (NASA JPL)


The ISOFIT codebase was made possible with support from various sources.
The initial algorithm and code was developed by the NASA Earth Science
Division data analysis program “Utilization of Airborne Visible/Infrared
Imaging Spectrometer Next Generation Data from an Airborne Campaign in
India," program NNH16ZDA001N-AVRSNG, managed by Woody Turner.  Later
research and maturation was provided by the Jet Propulsion Laboratory and
California Institute of Technology President and Director’s Fund, and the
Jet Propulsion Laboratory Research and Technology Development Program.
The project is currently supported by the Open Source Tools, Frameworks,
and Libraries Program (NNH20ZDA001N), managed by Dr. Steven Crawford.
Neural network radiative transfer is supported by the NASA Center
Innovation Fund managed in conjunction with the Jet Propulsion Laboratory
Office of the Chief Scientist and Technologist. The initial research took
place at the Jet Propulsion Laboratory, California Institute of Technology,
4800 Oak Grove Dr., Pasadena, CA 91109 USA.
