#!/bin/bash

# Download relevant datasets - unlike other examples, these datasets are too large to place in
if test -f "test_data_rev.zip"; then
  echo "Test zip already present, skipping download"
else
  curl -O https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/test_data_rev.zip
  unzip test_data_rev.zip
fi

n_cores=4

isofit_base_path=$(python -c "import isofit; import os; print(os.path.dirname(isofit.__file__))")
file_base=ang20170323t202244

python -c "from isofit.utils import surface_model; surface_model('configs/basic_surface.json')"

# Small test (10x10 pixels, no empirical line) - this should take   2-3 minutes with n_cores = 4.
python  ${isofit_base_path}/utils/apply_oe.py    \
        small_chunk/${file_base}_rdn_7000-7010   \
        small_chunk/${file_base}_loc_7000-7010   \
        small_chunk/${file_base}_obs_7000-7010   \
        small_chunk_test                         \
        ang                                      \
        --presolve=1                             \
        --empirical_line=0                       \
        --emulator_base=${EMULATOR_PATH}         \
        --n_cores ${n_cores}                     \
        --surface_path configs/basic_surface.mat \
        --copy_input_files 0
