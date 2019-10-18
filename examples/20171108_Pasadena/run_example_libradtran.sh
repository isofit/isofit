# David R Thompson

# Create surface model
pythonw ../../utils/surfmodel.py configs/ang20171108t184227_surface.json

# Run experiment
isofit --level DEBUG configs/ang20171108t184227_beckmanlawn-libradtran.json
