#!/bin/bash

# Get directories and paths for scripts
basedir=$( cd "$(dirname "$0")" ; pwd -P )
isofit_dir=$(dirname $basedir)

echo "basedir is $basedir"
echo "isofit_dir is $isofit_dir"

# .imgspec paths
apply_glt_exe="$isofit_dir/.imgspec/apply_glt.py"
wavelength_file_exe="$isofit_dir/.imgspec/wavelength_file.py"
basic_surface_json_path="$isofit_dir/.imgspec/basic_surface.json"

# utils paths
surface_model_exe="$isofit_dir/isofit/utils/surface_model.py"
apply_oe_exe="$isofit_dir/isofit/utils/apply_oe.py"

# data path
input_spectrum_files_path="$isofit_dir/data/reflectance/surface_model_ucsb"

# input/output dirs
input="input"
mkdir output

# Get instrument type
gzip_path=$(ls $input/*gz)
echo "Found input gzip file: $gzip_path"
gzip_file=$(basename $gzip_path)
instrument=""
if [[ $gzip_file == f* ]]; then
    instrument="AVCL"
elif [[ $gzip_file == ang* ]]; then
    instrument="AVNG"
fi
echo "Instrument is $instrument"

# Unzip file and remove gzip
echo "Unzipping files..."
for file in $input/*tar.gz; do
    tar xzvf $file --directory $input/
#    rm -f $file
done

# Get rdn, loc, and obs based on instrument type
if [[ $instrument == "AVNG" ]]; then
    rdn_path=$(ls $input/ang*rdn*/ang*rdn*img)
    if [[ -f $rdn_path ]]; then
        echo "Found rdn file $rdn_path"
    else
        echo "Couldn't find rdn file, exiting..."
        exit 1
    fi

    loc_path=$(ls $input/ang*rdn*/ang*rdn*loc)
    if [[ -f $loc_path ]]; then
        echo "Found loc file $loc_path"
    else
        echo "Couldn't find loc file, exiting..."
        exit 1
    fi

    glt_path=$(ls $input/ang*rdn*/ang*rdn*glt)
    if [[ -f $glt_path ]]; then
        echo "Found glt file $glt_path"
    else
        echo "Couldn't find glt file, exiting..."
        exit 1
    fi

    obs_ort_path=$(ls $input/ang*rdn*/ang*rdn*obs_ort)
    if [[ -f $obs_ort_path ]]; then
        echo "Found obs_ort file $obs_ort_path"
    else
        echo "Couldn't find obs_ort file, exiting..."
        exit 1
    fi
fi

# TODO: Add block for AVCL files

# Create wavelength file
wavelength_file_cmd="python $wavelength_file_exe $rdn_path.hdr $input/wavelengths.txt"
echo "Executing command: $wavelength_file_cmd"
$wavelength_file_cmd

# Orthocorrect the loc file
ort_suffix="_ort"
loc_ort_path=$loc_path$ort_suffix
apply_glt_cmd="python $apply_glt_exe $loc_path $glt_path $loc_ort_path"
echo "Executing command: $apply_glt_cmd"
$apply_glt_cmd

# Build surface model based on basic_surface.json template
sed -e "s|\${output_model_file}|\./basic_surface.mat|g" \
    -e "s|\${wavelength_file}|\./wavelengths.txt|g" \
    -e "s|\${input_spectrum_files}|$input_spectrum_files_path|g" \
    $basic_surface_json_path > $input/basic_surface.json
echo "Building surface model using config file $input/basic_surface.json"
python -c "from isofit.utils import surface_model; surface_model('$input/basic_surface.json')"

# Run isofit
working_dir=$(pwd)
isofit_cmd="""python $apply_oe_exe $rdn_path $loc_ort_path $obs_ort_path $working_dir ang --presolve=1 \
--empirical_line=1 --emulator_base=$EMULATOR_DIR --n_cores 24 --wavelength_path $input/wavelengths.txt \
--surface_path $input/basic_surface.mat --log_file isofit.log"""
echo "Executing command: $isofit_cmd"
$isofit_cmd

# Clean up output directory
rm -f output/*lbl*
rm -f output/*subs*
