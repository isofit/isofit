# David R Thompson

# Build the surface model
pythonw ../../utils/surfmodel.py configs/prm20151026t173213_surface_coastal.json

# Run retrievals
isofit --level DEBUG configs/prm20151026t173213_D8W_6s.json
isofit --level DEBUG configs/prm20151026t173213_D8p5W_6s.json
isofit --level DEBUG configs/prm20151026t173213_D9W_6s.json
isofit --level DEBUG configs/prm20151026t173213_D9p5W_6s.json
