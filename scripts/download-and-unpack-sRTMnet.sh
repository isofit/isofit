#!/bin/bash


# Download and unpack sRTM net.


set -x
set -o errexit
set -o pipefail
set -o nounset


SRTMNET_DIR="sRTMnet_v120"
SRTMNET_PATH="${SRTMNET_DIR}/sRTMnet_v120.h5"
SRTMNET_MODEL_FILENAME="sRTMnet_v120.h5"
SRTMNET_AUX_FILENAME="sRTMnet_v120_aux.npz"

# If the desired sRTM net file already exists just assume it is correct and do
# nothing.
if [ -f "${SRTMNET_PATH}" ]; then
  echo "sRTM net already exists - nothing to do: ${SRTMNET_PATH}"
  exit 0
fi

mkdir -p "${SRTMNET_DIR}"

# Download sRTMnet model file
wget \
  --no-verbose \
  --directory-prefix "${SRTMNET_DIR}" \
  "https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/${SRTMNET_MODEL_FILENAME}"

# Download sRTMnet aux file
wget \
   --no-verbose \
   --directory-prefix "${SRTMNET_DIR}" \
   "https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/${SRTMNET_AUX_FILENAME}"
