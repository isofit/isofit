#!/bin/bash


# This script sets up, and refreshes, a development environment. Users must
# create a virtual environment, and then point this script to it like:
#
#     $ python3.11 -m venv venv
#     $ ./scripts/setup-devenv.sh venv
#
# Note that different developers may use different versions of Python. Some
# environments may require installing additional non-Python dependencies using
# an appropriate package manage.


set -x
set -o errexit
set -o pipefail
set -o nounset


# Path to directory containing this script. Needed to determine the location of
# other scripts required for setup.
SCRIPT_DIR=$(dirname "${0}")


# Parse arguments.
if [ $# -eq 1 ]; then
  VENV_PATH="$1"
else
  echo "Usage: ${SCRIPT_DIR}/$(basename "$0") path/to/venv"
  exit 1
fi


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
