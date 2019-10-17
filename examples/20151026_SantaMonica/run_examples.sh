# David R Thompson
export isofit="../../bin/isofit"
export surfmodel="../../utils/surfmodel.py"

# Build the surface model
pythonw ${surfmodel} configs/prm20151026t173213_surface_coastal.json

# Run retrievals
pythonw ${isofit} --level DEBUG configs/prm20151026t173213_D8W_6s.json
pythonw ${isofit} --level DEBUG configs/prm20151026t173213_D8p5W_6s.json
pythonw ${isofit} --level DEBUG configs/prm20151026t173213_D9W_6s.json
pythonw ${isofit} --level DEBUG configs/prm20151026t173213_D9p5W_6s.json
