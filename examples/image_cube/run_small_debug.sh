#!/bin/bash

# Download relevant datasets - unlike other examples, these datasets are too large to place in
if test -f "test_data_rev.zip"; then
  echo "Test zip already present, skipping download"
else
  curl -O https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/test_data_rev.zip
  unzip test_data_rev.zip
fi

isofit_base_path=$(python -c "import isofit; import os; print(os.path.dirname(isofit.__file__))")
file_base=ang20170323t202244

# Adjust a few config settings
jq '.general_options.empirical_line=$newVal' --argjson newVal false configs/basic_config.json > tmp.$$.json && mv tmp.$$.json configs/basic_config.json
jq '.general_options.debug_mode=$newVal' --argjson newVal true configs/basic_config.json > tmp.$$.json && mv tmp.$$.json configs/basic_config.json
jq '.processors.general_inversion_parameters.filepaths.emulator_base=env.EMULATOR_PATH' configs/basic_config.json >> tmp.$$.json && mv tmp.$$.json configs/basic_config.json

# Small test with debugging (10x10 pixels, no empirical line) - this should take ~15 minutes with n_cores = 4.
python -m cProfile -o medium_profile.dat "${isofit_base_path}"/utils/multisurface_oe.py small_chunk/${file_base}_rdn_7000-7010 small_chunk/${file_base}_loc_7000-7010 small_chunk/${file_base}_obs_7000-7010 small_chunk_test configs/basic_config.json
