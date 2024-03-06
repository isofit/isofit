#!/bin/bash


# Download, modify, and build the 6S library.


set -x
set -o errexit
set -o pipefail
set -o nounset


SIXS_DIR="6sv-2.1"
SIXS_MAKEFILE_PATH="${SIXS_DIR}/Makefile"


# If the 'Makefile' doesn't exist then 6s probably has not been downloaded.
# Given that the project is built with '$ make' it seems prudent to always run
# '$ make', but only download the library if necessary.
if [ ! -f "${SIXS_MAKEFILE_PATH}" ]; then

  mkdir -p "${SIXS_DIR}"
  wget \
    --no-verbose \
    --directory-prefix "${SIXS_DIR}" \
    https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar

fi


# Build 6s.
#
# Run as a subprocess to allow 'cd'-ing into a subdirectory while automatically
# returning to the current working directory once the subprocess exits.
(

  cd "${SIXS_DIR}"
  tar -xf 6sv-2.1.tar

  # Modify build flags
  cp Makefile Makefile.bak
  sed -i -e 's/FFLAGS.*/& -std=legacy/' Makefile

  # Historically '$ nproc' is not the most portable command, but that may have
  # changed. This could be replaced with:
  #   python -c 'import os; print(os.cpu_count())'
  make -j "$(nproc)"
)
