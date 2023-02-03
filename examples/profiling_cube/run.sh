#!/bin/bash

# Download relevant datasets - unlike other examples, these datasets are too large to place in
if test -f "test_data_rev.zip"; then
  echo "Test zip already present, skipping download"
else
  curl -O https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/test_data_rev.zip
  unzip test_data_rev.zip
  rm -r medium_chunk
fi

isofit_base_path=$(python -c "import isofit; import os; print(os.path.dirname(isofit.__file__))")
file_base=ang20170323t202244

for i in {0..4}; do
  echo "Running $i/5 and saving to $1.$i.dat"

  if [ -n "$(ls -A small_chunk/output 2>/dev/null)" ]; then
    echo "Removing previous output files"
    rm small_chunk/output/*
  fi

  python \
    -m cProfile -o "results/$1.$i.dat" \
    run.py -c $2
done

echo "Done"
