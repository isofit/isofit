
# Download relevant datasets - unlike other examples, these datasets are too large to place in
if test -f "test_data.zip"; then
  echo "Test zip already present, skipping download"
else
  curl -O https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/test_data.zip
  unzip test_data.zip
fi

isofit_base_path=$(python -c "import isofit; import os; print(os.path.dirname(isofit.__file__))")
file_base=ang20170323t202244

jq '.general_options.debug_mode=$newVal' --argjson newVal false configs/basic_config.json > tmp.$$.json && mv tmp.$$.json configs/basic_config.json
jq '.processors.general_inversion_parameters.filepaths.emulator_base=env.EMULATOR_PATH' configs/basic_config.json >> tmp.$$.json && mv tmp.$$.json configs/basic_config.json

# Small test (10x10 pixels, no empirical line) - this should take 4-5 minutes with n_cores = 4.
jq '.general_options.empirical_line=$newVal' --argjson newVal false configs/basic_config.json > tmp.$$.json && mv tmp.$$.json configs/basic_config.json
python "${isofit_base_path}"/utils/multisurface_oe.py small_chunk/${file_base}_rdn_7000-7010 small_chunk/${file_base}_loc_7000-7010 small_chunk/${file_base}_obs_7000-7010 small_chunk_test configs/basic_config.json

# Medium test (1000x598 pixels, empirical line) - this should take ~45 minutes with n_cores = 4
jq '.general_options.empirical_line=$newVal' --argjson newVal true configs/basic_config.json > tmp.$$.json && mv tmp.$$.json configs/basic_config.json
python "${isofit_base_path}"/utils/multisurface_oe.py medium_chunk/${file_base}_rdn_7k-8k medium_chunk/${file_base}_loc_7k-8k medium_chunk/${file_base}_obs_7k-8k medium_chunk_test configs/basic_config.json
