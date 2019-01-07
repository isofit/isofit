# David R Thompson

# Create surface model
python3 ../../utils/surfmodel.py configs/ang20171108t184227_surface.json

# Run experiments 
python3 ../../isofit/isofit.py --level DEBUG configs/ang20171108t173546_darklot.json 
python3 ../../isofit/isofit.py --level DEBUG configs/ang20171108t173546_horse.json 
python3 ../../isofit/isofit.py --level DEBUG configs/ang20171108t184227_astrored.json 
python3 ../../isofit/isofit.py --level DEBUG configs/ang20171108t184227_astrogreen.json 
python3 ../../isofit/isofit.py --level DEBUG configs/ang20171108t184227_beckmanlawn.json 
python3 ../../isofit/isofit.py --level DEBUG configs/ang20171108t184227_beckmanlawn-oversmoothed.json 
python3 ../../isofit/isofit.py --level DEBUG configs/ang20171108t184227_beckmanlawn-undersmoothed.json 

