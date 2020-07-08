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

Unit Tests
----------

Unit tests are implemented in the tests/ subdirectory, and use the pytest library.  You can run the tests from the base directory simply by running:

.. code::

  pytest

Our development strategy employs continuous integration and unit testing to validate all changes.  We appreciate your writing additional tests for new modifications or features.  In the interest of validating your code, please also be sure to run realistic examples like this:

.. code::

  cd examples/20171108_Pasadena
  ./run_example_modtran.sh

A Note about Style
------------------

The coding style should adhere to `PEP 8 (Style Guide for Python Code) <https://www.python.org/dev/peps/pep-0008/>`_ and
`PEP 257 (Docstring Conventions): <https://www.python.org/dev/peps/pep-0257/>`_.

We recommend using autopep8, which you can install and run like this:

.. code::

  pip install autopep8
  autopep8 mycode.py

Implement Your Changes and Create a Pull Request
------------------------------------------------

At this point, you are ready to implement your changes!

As you develop, you should make sure that your branch doesn't veer too far from
ISOFIT's master branch.  To do this, switch back to your master branch and make
sure it's up to date with ISOFIT's master branch:

.. code::

  git remote add upstream https://github.com/isofit/isofit.git
  git checkout master
  git pull upstream master


Then update your feature branch from your local copy of master, and push it!

.. code::

  git checkout 314-update-docs-libradtran
  git rebase master
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
  git pull --rebase upstream master
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

Note, not all accepted PRs need to have a release.  Good reasons to not issue a release are that while the PR is stable, additional updates are expected, or multiple separate PRs are coming together.  However, if you accept a PR and choose not to make a release, update the `Roadmap <https://github.com/isofit/isofit/projects/4>`_ to indicate the current progress towards the next release.

Get latest public repo:

.. code::

  git clone https://github.com/isofit/isofit.git
  cd isofit

Create release branch and pull in changes from pull request:

.. code::

  git checkout -b release-1.2.0 master
  git pull https://github.com/davidraythompson/isofit.git feature-branch  (NOTE: this is the pull request branch)

Update version number:

.. code::

  vi isofit/__init__.py
  vi recipe/meta.yaml

Commit changes to release branch:

.. code::

  git add -A
  git commit -m “Prepares version 1.2.0 for release."
  
Merge release branch into master:

.. code::

  git checkout master
  git merge --no-ff release-1.2.0
  git push origin master
  
Update the 'current-release' branch:

.. code::

  git checkout current-release
  git merge master
  git push origin current-release
  git checkout master

Create release tag and release archive:

* Go to https://github.com/isofit/isofit/releases
* Click "Draft a new release"
* Enter tag version as "v1.2.0" (depending on latest version), and input release title and description
* Click "Publish release"

Update sha256 hash value for conda recipe:

* Download latest tar.gz from https://github.com/isofit/isofit/releases/tag/v1.2.0
* Run "openssl dgst -sha256 isofit-1.2.0.tar.gz"
* Add sha256 hash value to reciples/meta.yaml and update master with the following:

.. code::

  git add -A
  git commit -m "Adds sha256 hash value to conda-forge recipe"
  git push origin master

Create and upload Pypi distribution:

.. code::

  python3 setup.py sdist bdist_wheel
  twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
  
Lastly, update the `Roadmap <https://github.com/isofit/isofit/projects/4>`_ to note the current release changes.

Contributors
------------

The github maintainers, responsible for handling pull requests, are:

* David R. Thompson: david.r.thompson@jpl.nasa.gov
* Winston Olson Duvall winston.olson-duvall@jpl.nasa.gov
* Philip Brodrick philip.brodrick@jpl.nasa.gov

Thanks to the following regular contributors:

* Jay Fahlen (NASA JPL)
* Adam Erickson (NASA GSFC)
* Shanti Rao (NASA JPL)
* Terry Mullen (UMass)

The ISOFIT codebase was made possible with support from various sources.
The initial algorithm and code was developed by the NASA Earth Science
Division data analysis program “Utilization of Airborne Visible/Infrared
Imaging Spectrometer Next Generation Data from an Airborne Campaign in
India," program NNH16ZDA001N-AVRSNG, managed by Woody Turner.  Later
research and maturation was provided by the Jet Propulsion Laboratory and
California Institue of Technology President and Director’s Fund, and the
Jet Propulsion Laboratory Research and Technology Development Program.
Neural network radiative transfer is supported by the NASA Center
Innovation Fund managed in conjunction with the Jet Propulsion Laboratory
Office of the Chief Scientist and Technologist. The initial research took
place at the Jet Propulsion Laboratory, California Institute of Technology,
4800 Oak Grove Dr., Pasadena, CA 91109 USA.



