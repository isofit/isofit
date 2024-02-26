#!/bin/bash


# This script serves as a reference for how to set up a developer environment.
# Some packages, like GDAL, must be installed prior to running this script.
# On Ubuntu these packages are likely installed via '$ apt' or '$ apt-get',
# and on MacOS Homebrew is a common choice. Developers should read through
# this script, and the other scripts it invokes, and determine if it is
# appropriate to execute for their environment. At a high-level the steps
# should be matched, but the specific commands may need to be modified.


set -x
set -o errexit
set -o pipefail
set -o nounset


VENV_PATH="venv"


# Path to directory containing this script.
SCRIPT_DIR=$(dirname "${0}")


# Do not attempt to modify an existing virtual environment. We do not know what
# it contains.
if [ -d "${VENV_PATH}" ]; then
  echo "Virtual environment already exists: ${VENV_PATH}"
  exit 0
fi

# Create virtual environment.
python3 -m venv "${VENV_PATH}"

# Activate virtual environment. SUPER CRITICAL that this activation happens.
# Some shell scripts assume they are in an isolated environment.
source "${VENV_PATH}/bin/activate"
if [ "${VIRTUAL_ENV}" == "" ]; then
  echo "ERROR: Virtual environment did not activate: ${VENV_PATH}"
  exit 1
fi

# Useful for debugging
which python3
python3 --version

# Install and upgrade packaging dependencies. The presence of 'wheel' triggers
# a lot of the emerging Python packaging ecosystem changes (PEP-517).
python3 -m pip install setuptools wheel --upgrade

# Install ISOFIT.
python3 -m pip install -e ".[dev,test]"

# Install commit hooks.
python3 -m pre_commit install

# Download and unpack additional dependencies.
"./${SCRIPT_DIR}/download-and-unpack-sRTMnet.sh"
"./${SCRIPT_DIR}/download-and-build-6s.sh"
