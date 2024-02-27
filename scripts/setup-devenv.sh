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

# The version of Python inside of the virtual environment is isolated, and can
# be used to install packages inside of the environment without activating it.
PYTHON="${VENV_PATH}/bin/python3"

# Useful for debugging
"${PYTHON}" --version

# Install and upgrade packaging dependencies. The presence of 'wheel' triggers
# a lot of the emerging Python packaging ecosystem changes (PEP-517).
"${PYTHON}" -m pip install pip setuptools wheel --upgrade

# Install ISOFIT and dependencies in editable mode.
"${PYTHON}" -m pip install -e ".[dev,test]"

# Install commit hooks.
"${PYTHON}" -m pre_commit install

# Download and unpack additional dependencies.
"./${SCRIPT_DIR}/download-and-unpack-sRTMnet.sh"
"./${SCRIPT_DIR}/download-and-build-6s.sh"
