#!/bin/bash

# Download relevant datasets - unlike other examples, these datasets are too large to place in
if test -f "test_data_rev.zip"; then
  echo "Test zip already present, skipping download"
else
  curl -O https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/test_data_rev.zip
  unzip test_data_rev.zip
fi

file_base=ang20170323t202244

# Small test (10x10 pixels, no empirical line) - this should take 4-5 minutes with n_cores = 4.
isofit apply_oe small_chunk/${file_base}_rdn_7000-7010 small_chunk/${file_base}_loc_7000-7010 small_chunk/${file_base}_obs_7000-7010 small_chunk/L2A_reflectance ang --surface_path surface.mat --n_cores 4 --presolve --emulator_base ${EMULATOR_PATH} --segmentation_size 400 --pressure_elevation
