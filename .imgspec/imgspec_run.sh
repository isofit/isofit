#!/bin/bash

# Description:
#
# The top-level run script to execute ISOFIT on ImgSPEC. This script accepts the inputs described in the
# algorithm_config.yaml file and pre-processes them as needed to pass into isofit/util/apply_oe.py.  This script
# is currently compatible with AVIRIS Classic, AVIRIS-NG, and PRISMA data.
#
# Inputs:
#
# $1: EcoSIS URL of surface_reflectance_spectra
# $2: EcoSIS URL of vegetation_reflectance_spectra
# $3: EcoSIS URL of water_reflectance_spectra
# $4: EcoSIS URL of snow_and_liquids_reflectance_spectra
# $5: ISOFIT apply_oe.py segmentation_size argument
# $6: ISOFIT apply_oe.py n_cores argument
# $7: URL of radiance_factors_file
#
# In addition to the positional arguments, this script expects a downloaded radiance granule to be present in a folder
# called "input".

# Get directories and paths for scripts
imgspec_dir=$( cd "$(dirname "$0")" ; pwd -P )
isofit_dir=$(dirname $imgspec_dir)

echo "imgspec_dir is $imgspec_dir"
echo "isofit_dir is $isofit_dir"

# input/output dirs
input="input"
mkdir -p output

# .imgspec paths
apply_glt_exe="$imgspec_dir/apply_glt.py"
wavelength_file_exe="$imgspec_dir/wavelength_file.py"
upgrade_exe="$imgspec_dir/upgrade.py"
covnert_csv_to_envi_exe="$imgspec_dir/convert_csv_to_envi.py"
surface_json_path="$imgspec_dir/surface.json"

# utils paths
surface_model_exe="$isofit_dir/isofit/utils/surface_model.py"
apply_oe_exe="$isofit_dir/isofit/utils/apply_oe.py"

# ecosis input spectra paths
filtered_other_csv_path="$input/surface-reflectance-spectra.csv"
filtered_veg_csv_path="$input/vegetation_reflectance_spectra.csv"
filtered_ocean_csv_path="$input/water_reflectance_spectra.csv"
surface_liquids_csv_path="$input/snow_and_liquids_reflectance_spectra.csv"

# Process positional args to get EcoSIS CSV files
curl --retry 10 --output $filtered_other_csv_path $1
curl --retry 10 --output $filtered_veg_csv_path $2
curl --retry 10 --output $filtered_ocean_csv_path $3
curl --retry 10 --output $surface_liquids_csv_path $4

# Converted spectra ENVI paths
filtered_other_img_path=${filtered_other_csv_path/.csv/}
filtered_veg_img_path=${filtered_veg_csv_path/.csv/}
filtered_ocean_img_path=${filtered_ocean_csv_path/.csv/}
surface_liquids_img_path=${surface_liquids_csv_path/.csv/}

# Get input paths
rdn_path=$(python ${imgspec_dir}/get_paths_from_granules.py -p rdn)
echo "Found input radiance file: $rdn_path"
loc_path=$(python ${imgspec_dir}/get_paths_from_granules.py -p loc)
echo "Found input loc file: $loc_path"
igm_path=$(python ${imgspec_dir}/get_paths_from_granules.py -p igm)
echo "Found input igm file: $igm_path"
glt_path=$(python ${imgspec_dir}/get_paths_from_granules.py -p glt)
echo "Found input glt file: $glt_path"
obs_ort_path=$(python ${imgspec_dir}/get_paths_from_granules.py -p obs_ort)
echo "Found input obs_ort file: $obs_ort_path"

# Get instrument type
rdn_name=$(basename $rdn_path)
instrument=""
if [[ $rdn_name == f* ]]; then
    instrument="avcl"
elif [[ $rdn_name == ang* ]]; then
    instrument="ang"
elif [[ $rdn_name == PRS* ]]; then
    instrument="prisma"
fi
echo "Instrument is $instrument"

# Get the orthocorrected loc/igm file depending on instrument
ort_suffix="_ort"
loc_ort_path=""
if [[ $instrument == "avcl" ]]; then
    # AVIRIS Classic typically includes an IGM file with lon, lat, alt bands
    loc_ort_path=$igm_path$ort_suffix
    apply_glt_cmd="python $apply_glt_exe $igm_path $glt_path $loc_ort_path --one_based_glt=1"
    echo "Executing command: $apply_glt_cmd"
    $apply_glt_cmd
elif [[ $instrument == "ang" ]]; then
    # For AVIRIS-NG we must orthocorrect the given loc file
    loc_ort_path=$loc_path$ort_suffix
    apply_glt_cmd="python $apply_glt_exe $loc_path $glt_path $loc_ort_path --one_based_glt=1"
    echo "Executing command: $apply_glt_cmd"
    $apply_glt_cmd
elif [[ $instrument == "prisma" ]]; then
    # PRISMA already has a projected loc file
    loc_ort_path=$loc_path
fi
echo "Based on instrument, using loc_ort_path: $loc_ort_path"

# Convert AVIRIS Classic radiance to 32-bit float and scale
if [[ $instrument == "avcl" ]]; then
    upgrade_suffix="_up"
    rdn_up_path=$rdn_path$upgrade_suffix
    upgrade_cmd="python $upgrade_exe $rdn_path $rdn_up_path"
    echo "Executing command: $upgrade_cmd"
    $upgrade_cmd
    # Use new upgraded file in place of original radiance file
    rdn_path=$rdn_up_path
fi

# Create wavelength file
wavelength_file_cmd="python $wavelength_file_exe $rdn_path.hdr $input/wavelengths.txt"
echo "Executing command: $wavelength_file_cmd"
$wavelength_file_cmd

# Build surface model based on surface.json template and input spectra CSV
# First convert CSV to ENVI for 3 spectra files
convert_csv_to_envi_cmd="python $covnert_csv_to_envi_exe $filtered_other_csv_path"
echo "Executing command: convert_csv_to_envi_cmd on $filtered_other_csv_path"
$convert_csv_to_envi_cmd

convert_csv_to_envi_cmd="python $covnert_csv_to_envi_exe $filtered_veg_csv_path"
echo "Executing command: convert_csv_to_envi_cmd on $filtered_veg_csv_path"
$convert_csv_to_envi_cmd

convert_csv_to_envi_cmd="python $covnert_csv_to_envi_exe $filtered_ocean_csv_path"
echo "Executing command: convert_csv_to_envi_cmd on $filtered_ocean_csv_path"
$convert_csv_to_envi_cmd

convert_csv_to_envi_cmd="python $covnert_csv_to_envi_exe $surface_liquids_csv_path"
echo "Executing command: convert_csv_to_envi_cmd on $surface_liquids_csv_path"
$convert_csv_to_envi_cmd

sed -e "s|\${output_model_file}|\./surface.mat|g" \
    -e "s|\${wavelength_file}|\./wavelengths.txt|g" \
    -e "s|\${input_spectrum_filtered_other}|${filtered_other_img_path/input/\.}|g" \
    -e "s|\${input_spectrum_filtered_veg}|${filtered_veg_img_path/input/\.}|g" \
    -e "s|\${input_spectrum_filtered_ocean}|${filtered_ocean_img_path/input/\.}|g" \
    -e "s|\${input_spectrum_surface_liquids}|${surface_liquids_img_path/input/\.}|g" \
    $surface_json_path > $input/surface.json
echo "Building surface model using config file $input/surface.json"
python -c "from isofit.utils import surface_model; surface_model('$input/surface.json')"

# Run isofit
working_dir=$(pwd)
isofit_cmd=""
if [[ $instrument == "avcl" ]] || [[ $instrument == "ang" ]]; then
    isofit_cmd="""python $apply_oe_exe $rdn_path $loc_ort_path $obs_ort_path $working_dir $instrument --presolve=1 \
    --empirical_line=1 --emulator_base=$EMULATOR_DIR --n_cores $6 --wavelength_path $input/wavelengths.txt \
    --segmentation_size $5 --surface_path $input/surface.mat --log_file isofit.log"""
elif [[ $instrument == "prisma" ]]; then
    # Use NA-YYYYMMDD for instrument
    prisma_prefix="NA-"
    instrument=$prisma_prefix$(echo $rdn_name | cut -c5-12)
    echo "For PRISMA, using $instrument as instrument argument in apply_oe command"
    # Get rdn_factors_file
    rdn_factors_path="$input/rdn_factors.txt"
    echo "Getting radiance_factors file from $7"
    curl --retry 10 --output $rdn_factors_path $7
    isofit_cmd="""python $apply_oe_exe $rdn_path $loc_ort_path $obs_ort_path $working_dir $instrument --presolve=1 \
    --empirical_line=1 --emulator_base=$EMULATOR_DIR --n_cores $6 --wavelength_path $input/wavelengths.txt \
    --segmentation_size $5 --rdn_factors_path $rdn_factors_path --surface_path $input/surface.mat \
    --log_file isofit.log"""
fi

echo "Executing command: $isofit_cmd"
$isofit_cmd

# Clean up output directory
rm -f output/*lbl*
rm -f output/*subs*
