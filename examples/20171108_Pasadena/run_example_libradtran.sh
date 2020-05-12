# David R Thompson, Adam Erickson

# Create surface model
python3 -c "from isofit.utils import surface_model; surface_model('configs/ang20171108t184227_surface.json')"

# Run experiment
isofit --level DEBUG configs/ang20171108t184227_beckmanlawn-libradtran.json
