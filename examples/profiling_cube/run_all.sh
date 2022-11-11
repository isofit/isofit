#!/bin/bash

echo "Running RegularGrid interpolation"
echo "Be sure to update small_chunk/config/rg.json with the paths for the emulators for your system"
./run.sh rg small_chunk/config/rg.json

echo "Running NDSplines interpolation"
echo "Be sure to update small_chunk/config/nds.json with the paths for the emulators for your system"
./run.sh nds small_chunk/config/nds.json

echo "Running Multilinear Grid interpolation"
echo "Be sure to update small_chunk/config/mlg.json with the paths for the emulators for your system"
./run.sh mlg small_chunk/config/mlg.json

echo "Done"
