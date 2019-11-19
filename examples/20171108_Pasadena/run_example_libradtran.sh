# David R Thompson, Adam Erickson

# Create surface model
python3 -c "from isofit.utils.utils import surfmodel; surfmodel('configs/ang20171108t184227_surface.json')"

# Run experiment
isofit --level DEBUG configs/ang20171108t184227_beckmanlawn-libradtran.json
