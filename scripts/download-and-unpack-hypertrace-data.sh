#!/bin/bash


# Download and unpack Hypertrace data


set -x
set -o errexit
set -o pipefail
set -o nounset


HYPERTRACE_PATH="hypertrace-data/"
HYPERTRACE_ARCHIVE_NAME="hypertrace-data.tar.gz"


# If the Hypertrace data directory exists locally, assume it has already been
# downloaded.
if [ -d "${HYPERTRACE_PATH}" ]; then
  echo "Hypertrace data already exists - nothing to do: ${HYPERTRACE_PATH}"
  exit 0
fi


# Download
wget \
  --no-verbose \
  "https://github.com/ashiklom/isofit/releases/download/hypertrace-data/${HYPERTRACE_ARCHIVE_NAME}"

# Unpack
tar -xf "${HYPERTRACE_ARCHIVE_NAME}"


# Cleanup
rm "${HYPERTRACE_ARCHIVE_NAME}"
