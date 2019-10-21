# David R Thompson, Adam Erickson

# Create surface model
pythonw -c "from isofit.utils import surfmodel; surfmodel(configs/ang20171108t184227_surface.json)"

# Run experiments
isofit --level DEBUG configs/ang20171108t173546_darklot.json
isofit --level DEBUG configs/ang20171108t173546_horse.json
isofit --level DEBUG configs/ang20171108t184227_astrored.json
isofit --level DEBUG configs/ang20171108t184227_astrogreen.json
isofit --level DEBUG configs/ang20171108t184227_beckmanlawn.json
isofit --level DEBUG configs/ang20171108t184227_beckmanlawn-oversmoothed.json
isofit --level DEBUG configs/ang20171108t184227_beckmanlawn-undersmoothed.json
