#!/bin/bash

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
