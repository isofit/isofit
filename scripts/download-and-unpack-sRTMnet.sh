#!/bin/bash


# Download and unpack sRTM net.


set -x
set -o errexit
set -o pipefail
set -o nounset


SRTMNET_DIR="sRTMnet_v100"
SRTMNET_PATH="${SRTMNET_DIR}/sRTMnet_v100.h5"
SRTMNET_MODEL_FILENAME="sRTMnet_v100.h5"
SRTMNET_ZIPFILENAME="sRTMnet_v100.zip"

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
   "https://zenodo.org/record/4096627/files/${SRTMNET_ZIPFILENAME}"

 # Unpack and delete some extraneous files.
 #
 # Run as a subprocess to allow 'cd'-ing into a subdirectory while automatically
 # returning to the current working directory once the subprocess exits.
 (

   cd "${SRTMNET_DIR}"
   unzip -o "${SRTMNET_DIR}"

   # Remove MacOS specific files that the original author erroneously included
   # in the archive.
   rm -rf "__MACOSX"
   rm -f .DS_Store sRTMnet_v100/.DS_Store

   # The archive is quite large and no longer needed.
   rm "${SRTMNET_ZIPFILENAME}"

 )